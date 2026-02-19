#!/usr/bin/env python3
"""Create embeddings for all podcast transcripts.

This script indexes transcript documents by:
- Parsing transcript format (full text + timestamped sections)
- Chunking with semantic similarity
- Creating embeddings with sentence-transformers
- Storing in ChromaDB with episode metadata

Run this whenever you add or modify transcript files.

Usage:
    python create_embeddings.py [--force]

Use --force to reindex all files, even if already indexed.
"""

import argparse
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import chromadb
import logfire
from chromadb.config import Settings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

from config import config, get_device

# Suppress logfire warning
os.environ.setdefault("LOGFIRE_IGNORE_NO_CONFIG", "1")


class DocumentIndexer:
    """Handles indexing of transcript documents."""

    def __init__(self):
        """Initialize the document indexer with embedding models and ChromaDB."""
        print("\n🚀 Initializing DocumentIndexer...")
        logfire.info("Initializing DocumentIndexer")

        # Detect best device
        self.device = get_device()
        device_name = {
            "mps": "Apple Metal (GPU)",
            "cuda": "NVIDIA CUDA (GPU)",
            "cpu": "CPU",
        }.get(self.device, "CPU")
        print(f"🔧 Using device: {device_name}")
        if self.device == "mps":
            print("   ⚡ M3 Max detected - expect 20-50x speedup!")
        logfire.info(f"Using device: {self.device}")

        # Ensure directories exist
        config.ensure_directories()
        print(f"📁 Using ChromaDB directory: {config.chroma_db_dir}")

        # Initialize embedding model for creating embeddings
        print(f"\n⏳ Loading embedding model: {config.embedding_model}")
        print("   (Loading to GPU - may take 10-20 seconds...)")
        self.embedding_model = SentenceTransformer(
            config.embedding_model,
            device=self.device,
        )
        print(f"✅ Loaded embedding model to {device_name}")
        logfire.info(
            f"Loaded embedding model: {config.embedding_model} on {self.device}"
        )

        # Initialize ChromaDB
        print("\n⏳ Connecting to ChromaDB...")
        self.chroma_client = chromadb.PersistentClient(
            path=str(config.chroma_db_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name=config.documents_collection,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"✅ ChromaDB collection: {config.documents_collection}")
        logfire.info(f"ChromaDB collection ready: {config.documents_collection}")

        # Initialize semantic chunker for intelligent text splitting
        print("\n⏳ Initializing semantic chunker...")
        print("   (Loading model for chunking to GPU...)")
        langchain_embeddings = HuggingFaceEmbeddings(
            model_name=config.embedding_model, model_kwargs={"device": self.device}
        )
        self.semantic_chunker = SemanticChunker(
            langchain_embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=int(config.semantic_chunk_threshold * 100),
        )
        print(f"✅ Semantic chunker ready on {device_name}")
        logfire.info("SemanticChunker initialized")

        # Track indexed files
        print("\n⏳ Loading index metadata...")
        self._indexed_files: set[str] = self._load_indexed_files()
        print(f"✅ Found {len(self._indexed_files)} previously indexed files")

    def _load_indexed_files(self) -> set[str]:
        """Load the set of already indexed files."""
        try:
            result = self.collection.get(ids=["__index_metadata__"])
            if result and result["metadatas"]:
                files_str = result["metadatas"][0].get("indexed_files", "")
                if files_str:
                    return set(files_str.split("|"))
        except Exception:
            logfire.debug("No existing index metadata")
        return set()

    def _save_indexed_files(self) -> None:
        """Save the set of indexed files."""
        try:
            files_str = "|".join(self._indexed_files)
            self.collection.upsert(
                ids=["__index_metadata__"],
                metadatas=[{"indexed_files": files_str}],
                documents=["Index metadata"],
            )
        except Exception as e:
            logfire.error(f"Failed to save index metadata: {e}")

    def parse_transcript(self, content: str) -> dict[str, Any]:
        """Parse transcript into full text and timestamped sections.

        Args:
            content: Raw transcript content (or preprocessed)

        Returns:
            Dictionary with "full_text" and "timestamped_chunks" keys
        """
        sections = {"full_text": "", "timestamped_chunks": []}

        # Check if this is a preprocessed file (no section headers, just timestamped entries)
        if "=" * 80 not in content and content.strip().startswith("["):
            # Preprocessed format - entire file is timestamped sections
            chunks = self._parse_timestamped_section(content)
            sections["timestamped_chunks"] = chunks
            return sections

        # Original format with section markers
        parts = content.split("=" * 80)

        for i, part in enumerate(parts):
            part = part.strip()
            if not part:
                continue

            if "FULL TRANSCRIPT" in part:
                # Next part is the full text
                if i + 1 < len(parts):
                    sections["full_text"] = parts[i + 1].strip()

            elif "TIMESTAMPED TRANSCRIPT WITH SPEAKERS" in part:
                # Parse timestamped sections
                if i + 1 < len(parts):
                    timestamped_text = parts[i + 1]
                    chunks = self._parse_timestamped_section(timestamped_text)
                    sections["timestamped_chunks"] = chunks

        return sections

    def _parse_timestamped_section(self, text: str) -> list[dict[str, str]]:
        """Parse timestamped section into individual chunks.

        Args:
            text: Text containing timestamped sections

        Returns:
            List of dicts with timestamp, speaker, and text
        """
        chunks = []
        lines = text.split("\n")

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Look for timestamp pattern: [HH:MM:SS.mmm - HH:MM:SS.mmm] Speaker:
            # Handles both formats:
            # 1. Text on same line: [timestamp] Speaker: text...
            # 2. Text on next lines: [timestamp] Speaker:\n text...
            # Speaker can be any name (Greek names or "Speaker N" format)
            timestamp_match = re.match(
                r"\[(\d{2}:\d{2}:\d{2}\.\d+) - \d{2}:\d{2}:\d{2}\.\d+\]\s+([^:]+):\s*(.*)",
                line,
            )

            if timestamp_match:
                timestamp = timestamp_match.group(1)
                speaker = timestamp_match.group(2)
                first_line_text = timestamp_match.group(3).strip()

                # Collect text starting with same-line text
                text_lines = [first_line_text] if first_line_text else []

                # Collect text from following lines until next timestamp or empty line
                i += 1
                while (
                    i < len(lines) and lines[i].strip() and not lines[i].startswith("[")
                ):
                    text_lines.append(lines[i].strip())
                    i += 1

                if text_lines and any(
                    text_lines
                ):  # Check if we have any non-empty text
                    chunks.append(
                        {
                            "timestamp": timestamp,
                            "speaker": speaker,
                            "text": " ".join(t for t in text_lines if t).strip(),
                        }
                    )
            else:
                i += 1

        return chunks

    def combine_timestamped_chunks(
        self, timestamped_sections: list[dict[str, str]]
    ) -> list[dict[str, Any]]:
        """Combine small timestamped sections into larger chunks meeting min_size.

        Args:
            timestamped_sections: List of dicts with timestamp, speaker, text

        Returns:
            List of combined chunks with metadata
        """
        if not timestamped_sections:
            return []

        combined_chunks = []
        current_chunk = {
            "text": "",
            "start_timestamp": None,
            "end_timestamp": None,
            "speakers": [],
        }

        for section in timestamped_sections:
            text = section["text"]
            timestamp = section.get("timestamp", "")
            speaker = section.get("speaker", "")

            # Start new chunk if empty
            if not current_chunk["text"]:
                current_chunk["text"] = text
                current_chunk["start_timestamp"] = timestamp
                current_chunk["end_timestamp"] = timestamp
                if speaker and speaker not in current_chunk["speakers"]:
                    current_chunk["speakers"].append(speaker)
            else:
                # Add to current chunk
                current_chunk["text"] += " " + text
                current_chunk["end_timestamp"] = timestamp
                if speaker and speaker not in current_chunk["speakers"]:
                    current_chunk["speakers"].append(speaker)

            # If we've reached min_size, save this chunk and start new one
            if len(current_chunk["text"]) >= config.chunk_min_size:
                # Don't exceed max_size
                if len(current_chunk["text"]) <= config.chunk_max_size:
                    combined_chunks.append(dict(current_chunk))
                    current_chunk = {
                        "text": "",
                        "start_timestamp": None,
                        "end_timestamp": None,
                        "speakers": [],
                    }
                # If over max_size, split it
                elif len(current_chunk["text"]) > config.chunk_max_size:
                    # Save what we have and continue with overflow
                    combined_chunks.append(dict(current_chunk))
                    current_chunk = {
                        "text": "",
                        "start_timestamp": None,
                        "end_timestamp": None,
                        "speakers": [],
                    }

        # Add final chunk if not empty
        if current_chunk["text"]:
            combined_chunks.append(current_chunk)

        return combined_chunks

    def create_semantic_chunks(self, text: str) -> list[str]:
        """Split text into semantic chunks respecting size constraints.

        Args:
            text: Text to chunk

        Returns:
            List of text chunks
        """
        if not text:
            return []

        # If text is smaller than min_size, return as is
        if len(text) < config.chunk_min_size:
            return [text]

        try:
            # Apply semantic chunking to find topically coherent boundaries
            raw_chunks = self.semantic_chunker.split_text(text)

            # Combine small chunks and split large ones to meet size constraints
            final_chunks = []
            current_chunk = ""

            for chunk in raw_chunks:
                chunk = chunk.strip()
                if not chunk:
                    continue

                # If current chunk would become too large, save it and start new one
                if (
                    current_chunk
                    and len(current_chunk) + len(chunk) + 1 > config.chunk_max_size
                ):
                    if len(current_chunk) >= config.chunk_min_size:
                        final_chunks.append(current_chunk)
                        current_chunk = chunk
                    else:
                        # Current chunk is still too small, keep adding
                        current_chunk += " " + chunk
                # If no current chunk, start one
                elif not current_chunk:
                    current_chunk = chunk
                # If combined would be under max_size, combine
                else:
                    current_chunk += " " + chunk

                # If current chunk reached min_size and is at a semantic boundary, consider saving
                if len(current_chunk) >= config.chunk_min_size:
                    # Check if next iteration would exceed max_size
                    # For now, continue accumulating until we hit max or run out of chunks
                    pass

            # Add final chunk if not empty
            if current_chunk:
                final_chunks.append(current_chunk)

            return final_chunks if final_chunks else [text]

        except Exception as e:
            logfire.warn(f"Semantic chunking failed: {e}, using whole text")
            return [text]

    def create_chunks_with_metadata(
        self,
        text: str,
        filename: str,
        timestamp: str | None = None,
        speaker: str | None = None,
    ) -> list[dict[str, Any]]:
        """Create chunks with metadata.

        Args:
            text: Text to chunk
            filename: Source file name (episode name)
            timestamp: Optional timestamp
            speaker: Optional speaker name

        Returns:
            List of chunks with metadata
        """
        semantic_chunks = self.create_semantic_chunks(text)

        chunks_with_metadata = []
        for chunk in semantic_chunks:
            metadata = {
                "episode": filename,
                "timestamp": timestamp or "",
                "speaker": speaker or "",
            }
            chunks_with_metadata.append({"text": chunk, "metadata": metadata})

        return chunks_with_metadata

    def create_embedding(self, text: str) -> list[float]:
        """Create embedding for text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        embedding = self.embedding_model.encode(text, convert_to_tensor=False)
        # Handle both ndarray and already-list cases
        if hasattr(embedding, "tolist"):
            return embedding.tolist()
        return embedding

    def store_chunks(self, chunks: list[dict[str, Any]]) -> None:
        """Store chunks in ChromaDB with batch embedding creation.

        Args:
            chunks: List of chunks with text and metadata
        """
        if not chunks:
            return

        ids = []
        documents = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            # Create unique ID
            episode = chunk["metadata"]["episode"]
            chunk_id = f"{episode}_{i}"

            ids.append(chunk_id)
            documents.append(chunk["text"])
            metadatas.append(chunk["metadata"])

        # Create embeddings in batch (much faster!)
        print(f"   ⏳ Creating {len(documents)} embeddings in batch...")
        embeddings_array = self.embedding_model.encode(
            documents, batch_size=32, show_progress_bar=False, convert_to_tensor=False
        )

        # Convert to list of lists
        embeddings = []
        for embedding in embeddings_array:
            if hasattr(embedding, "tolist"):
                embeddings.append(embedding.tolist())
            else:
                embeddings.append(embedding)

        # Store in ChromaDB
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

        logfire.info(f"Stored {len(chunks)} chunks in ChromaDB")

    def index_file(self, file_path: Path) -> dict[str, Any]:
        """Index a single transcript file.

        Args:
            file_path: Path to transcript file

        Returns:
            Dict with success status and chunks_created count
        """
        try:
            print(f"\n📄 Indexing: {file_path.name}")
            logfire.info(f"Indexing {file_path.name}")

            # Read transcript
            print("   ⏳ Reading file...")
            content = file_path.read_text(encoding="utf-8")
            print(f"   ✅ Read {len(content):,} characters")

            # Parse sections
            print("   ⏳ Parsing transcript sections...")
            sections = self.parse_transcript(content)
            print(
                f"   ✅ Found {len(sections['timestamped_chunks'])} timestamped sections"
            )

            all_chunks = []

            # Create chunks from timestamped sections (preferred)
            if sections["timestamped_chunks"]:
                print(f"   ⏳ Creating semantic chunks from transcript...")
                # Concatenate all timestamped text for semantic chunking
                full_text = " ".join(
                    [s["text"] for s in sections["timestamped_chunks"]]
                )

                # Apply semantic chunking to find topically coherent chunks
                chunks = self.create_chunks_with_metadata(
                    text=full_text,
                    filename=file_path.name,
                )
                all_chunks.extend(chunks)
            # Fallback to full text if no timestamped sections
            elif sections["full_text"]:
                chunks = self.create_chunks_with_metadata(
                    text=sections["full_text"],
                    filename=file_path.name,
                )
                all_chunks.extend(chunks)

            # Store chunks
            if all_chunks:
                print(f"   ⏳ Storing {len(all_chunks)} chunks in ChromaDB...")
                self.store_chunks(all_chunks)
                print(f"   ✅ Stored successfully")

            print(f"✅ Successfully indexed {file_path.name}: {len(all_chunks)} chunks")
            logfire.info(
                f"✅ Successfully indexed {file_path.name}: {len(all_chunks)} chunks"
            )

            return {"success": True, "chunks_created": len(all_chunks)}

        except Exception as e:
            logfire.error(f"❌ Failed to index {file_path.name}: {e}")
            return {"success": False, "chunks_created": 0, "error": str(e)}

    def mark_as_indexed(self, file_path: Path) -> None:
        """Mark file as indexed."""
        self._indexed_files.add(file_path.name)
        self._save_indexed_files()

    def should_skip_file(self, file_path: Path, force: bool = False) -> bool:
        """Check if file should be skipped.

        Args:
            file_path: Path to file
            force: If True, don't skip any files

        Returns:
            True if file should be skipped
        """
        if force:
            return False
        return file_path.name in self._indexed_files


def index_all_transcripts(force: bool = False, workers: int = 3) -> dict[str, Any]:
    """Index all transcript files in the configured transcripts directory.

    Args:
        force: If True, reindex all files even if already indexed
        workers: Number of parallel workers

    Returns:
        Dictionary with indexing statistics
    """
    transcripts_dir = Path(config.transcripts_dir)

    if not transcripts_dir.exists():
        print(f"❌ Transcripts directory not found: {transcripts_dir}")
        logfire.error(f"Transcripts directory not found: {transcripts_dir}")
        return {"total_files": 0, "successful": 0, "failed": 0}

    # Get all transcript files
    print(f"\n📂 Scanning transcripts directory: {transcripts_dir}")
    transcript_files = list(transcripts_dir.glob("*.txt"))

    if not transcript_files:
        print("⚠️  No transcript files found")
        logfire.warn("No transcript files found")
        return {"total_files": 0, "successful": 0, "failed": 0}

    print(f"📊 Found {len(transcript_files)} transcript files")
    logfire.info(f"Found {len(transcript_files)} transcript files")

    print("\n" + "=" * 60)
    print("INITIALIZING MODELS (This is the slow part!)")
    print("=" * 60)
    indexer = DocumentIndexer()
    print("\n" + "=" * 60)
    print("STARTING INDEXING")
    print("=" * 60)

    # Filter files to index
    files_to_index = [
        f for f in transcript_files if not indexer.should_skip_file(f, force=force)
    ]

    if not files_to_index:
        print("\n✅ All files already indexed. Use --force to reindex.")
        logfire.info("All files already indexed. Use --force to reindex.")
        return {
            "total_files": len(transcript_files),
            "successful": len(transcript_files),
            "failed": 0,
            "skipped": len(transcript_files),
        }

    print(
        f"\n📋 Files to index: {len(files_to_index)} (skipping {len(transcript_files) - len(files_to_index)} already indexed)"
    )
    if force:
        print("⚠️  Force mode: reindexing all files")
    print(f"⚙️  Using {workers} parallel workers\n")
    logfire.info(f"Indexing {len(files_to_index)} files with {workers} workers")

    successful = 0
    failed = 0
    failed_files = []
    completed = 0
    total = len(files_to_index)

    # Index files in parallel
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_file = {
            executor.submit(indexer.index_file, file_path): file_path
            for file_path in files_to_index
        }

        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            completed += 1
            try:
                result = future.result()
                if result["success"]:
                    successful += 1
                    indexer.mark_as_indexed(file_path)
                    print(f"\n[{completed}/{total}] ✅ Success: {file_path.name}")
                else:
                    failed += 1
                    failed_files.append(file_path.name)
                    print(f"\n[{completed}/{total}] ❌ Failed: {file_path.name}")
            except Exception as e:
                print(f"\n[{completed}/{total}] ❌ Exception: {file_path.name} - {e}")
                logfire.error(f"Exception processing {file_path.name}: {e}")
                failed += 1
                failed_files.append(file_path.name)

    print("\n" + "=" * 60)
    print("INDEXING COMPLETE")
    print("=" * 60)

    results = {
        "total_files": len(transcript_files),
        "successful": successful,
        "failed": failed,
        "skipped": len(transcript_files) - len(files_to_index),
    }

    if failed_files:
        results["failed_files"] = failed_files

    logfire.info(f"Indexing complete: {successful} successful, {failed} failed")

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Index podcast transcripts")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reindex of all files",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Number of parallel workers (default: 3)",
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("📚 HISTORICON TRANSCRIPT INDEXER")
    print("=" * 60)
    print(f"🤖 Embedding model: {config.embedding_model}")
    print(f"💾 Database: {config.chroma_db_dir}")
    print(f"🔧 Workers: {args.workers}")
    if args.force:
        print("⚠️  Force mode: ON")
    print("=" * 60)

    logfire.info("Starting transcript indexing")
    results = index_all_transcripts(force=args.force, workers=args.workers)

    print(f"\n{'=' * 60}")
    print("📊 FINAL RESULTS")
    print(f"{'=' * 60}")
    print(f"📁 Total files: {results['total_files']}")
    print(f"✅ Successful: {results['successful']}")
    print(f"❌ Failed: {results['failed']}")
    print(f"⏭️  Skipped: {results.get('skipped', 0)}")
    if "failed_files" in results:
        print(f"\n❌ Failed files:")
        for fname in results["failed_files"]:
            print(f"   - {fname}")
    print(f"{'=' * 60}")

    if results["successful"] > 0:
        print("\n💡 TIP: Model loading is slow on first run.")
        print("   Subsequent runs will be faster - only new files are indexed.")
        print("   Use --force to reindex everything.\n")
    else:
        print()


if __name__ == "__main__":
    main()

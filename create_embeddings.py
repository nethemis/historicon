#!/usr/bin/env python3
"""Create embeddings for all podcast transcripts.

This script indexes transcript documents by:
- Parsing transcript format (full text + timestamped sections)
- Chunking with speaker-aware semantic similarity
- Creating embeddings with sentence-transformers
- Storing in ChromaDB with episode metadata

Semantic Chunking Approach:
- Never splits a single speaker's continuous speech
- Groups consecutive speaker segments if semantically similar
- Compares each segment to chunk's average embedding (cosine similarity)
- Enforces min_size before applying similarity cutoff

Configuration (config.json):
- similarity_threshold (0.0-1.0): Cosine similarity for grouping segments
  Lower values (0.5-0.7): Strict grouping, only very similar topics → smaller chunks
  Higher values (0.8-0.95): Loose grouping, related topics together → larger chunks
- chunk_min_size: Minimum characters per chunk (enforced first)
- chunk_max_size: Maximum characters (soft limit, speakers never split)

Run this whenever you add or modify transcript files.

Usage:
    python create_embeddings.py [--force]

Use --force to reindex all files, even if already indexed.
"""

import argparse
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import chromadb
import logfire
import numpy as np
from chromadb.config import Settings
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

    def create_semantic_chunks_from_speakers(
        self, timestamped_sections: list[dict[str, str]], filename: str
    ) -> list[dict[str, Any]]:
        """Group speaker segments into semantic chunks without splitting speakers.

        This method:
        - Never splits what a single speaker says continuously
        - Groups consecutive speaker segments if semantically similar
        - Compares each segment to the average embedding of the current chunk
        - Enforces min_size before considering similarity cutoff

        Args:
            timestamped_sections: List of dicts with timestamp, speaker, text
            filename: Source file name (episode name)

        Returns:
            List of chunks with text and metadata
        """
        if not timestamped_sections:
            return []

        print(
            f"   ⏳ Creating embeddings for {len(timestamped_sections)} speaker segments..."
        )
        # Create embeddings for all speaker segments at once (faster)
        segment_texts = [seg["text"] for seg in timestamped_sections]
        segment_embeddings = self.embedding_model.encode(
            segment_texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_tensor=False,
        )

        print(
            f"   ⏳ Grouping segments by semantic similarity (threshold: {config.similarity_threshold})..."
        )
        chunks = []
        current_chunk = {
            "text": "",
            "start_timestamp": None,
            "end_timestamp": None,
            "speakers": [],
            "embeddings": [],  # Track all embeddings for averaging
        }

        for i, section in enumerate(timestamped_sections):
            text = section["text"]
            timestamp = section.get("timestamp", "")
            speaker = section.get("speaker", "")
            segment_embedding = segment_embeddings[i]

            # Format text with speaker name for clarity
            speaker_text = f"{speaker}: {text}" if speaker else text

            # Start first chunk
            if not current_chunk["text"]:
                current_chunk["text"] = speaker_text
                current_chunk["start_timestamp"] = timestamp
                current_chunk["end_timestamp"] = timestamp
                if speaker and speaker not in current_chunk["speakers"]:
                    current_chunk["speakers"].append(speaker)
                current_chunk["embeddings"].append(segment_embedding)
                continue

            # Calculate similarity to current chunk (average of all segments in chunk)
            chunk_avg_embedding = np.mean(current_chunk["embeddings"], axis=0)
            similarity = np.dot(chunk_avg_embedding, segment_embedding) / (
                np.linalg.norm(chunk_avg_embedding) * np.linalg.norm(segment_embedding)
            )

            # Decide whether to add to current chunk or start new one
            should_add = False

            # If under min_size, keep adding regardless of similarity
            if len(current_chunk["text"]) < config.chunk_min_size:
                should_add = True
            # If semantically similar, add to chunk
            elif similarity >= config.similarity_threshold:
                should_add = True

            if should_add:
                # Add to current chunk with space separator
                current_chunk["text"] += " " + speaker_text
                current_chunk["end_timestamp"] = timestamp
                if speaker and speaker not in current_chunk["speakers"]:
                    current_chunk["speakers"].append(speaker)
                current_chunk["embeddings"].append(segment_embedding)
            else:
                # Save current chunk and start new one
                chunks.append(
                    {
                        "text": current_chunk["text"],
                        "metadata": {
                            "episode": filename,
                            "timestamp": current_chunk["start_timestamp"],
                            "speaker": ", ".join(current_chunk["speakers"]),
                        },
                    }
                )

                # Start new chunk
                current_chunk = {
                    "text": speaker_text,
                    "start_timestamp": timestamp,
                    "end_timestamp": timestamp,
                    "speakers": [speaker] if speaker else [],
                    "embeddings": [segment_embedding],
                }

        # Add final chunk
        if current_chunk["text"]:
            chunks.append(
                {
                    "text": current_chunk["text"],
                    "metadata": {
                        "episode": filename,
                        "timestamp": current_chunk["start_timestamp"],
                        "speaker": ", ".join(current_chunk["speakers"]),
                    },
                }
            )

        avg_chunk_size = (
            sum(len(c["text"]) for c in chunks) / len(chunks) if chunks else 0
        )
        print(
            f"   ✅ Created {len(chunks)} chunks (avg size: {avg_chunk_size:.0f} chars)"
        )

        return chunks

    def create_simple_chunks(self, text: str, filename: str) -> list[dict[str, Any]]:
        """Fallback chunking for transcripts without speaker info.

        Simply splits text into chunks of approximately chunk_min_size,
        breaking at sentence boundaries when possible.

        Args:
            text: Text to chunk
            filename: Source file name (episode name)

        Returns:
            List of chunks with metadata
        """
        if not text:
            return []

        # If text is smaller than min_size, return as single chunk
        if len(text) < config.chunk_min_size:
            return [
                {
                    "text": text,
                    "metadata": {
                        "episode": filename,
                        "timestamp": "",
                        "speaker": "",
                    },
                }
            ]

        chunks = []
        current_pos = 0

        while current_pos < len(text):
            # Try to get a chunk of target size
            end_pos = min(current_pos + config.chunk_max_size, len(text))

            # If not at the end, try to break at sentence boundary
            if end_pos < len(text):
                # Look for sentence endings (., !, ?) within last 20% of chunk
                search_start = current_pos + int(config.chunk_min_size * 0.8)
                chunk_text = text[search_start:end_pos]

                # Find last sentence boundary
                for delimiter in [". ", "! ", "? ", ".\n", "!\n", "?\n"]:
                    last_delim = chunk_text.rfind(delimiter)
                    if last_delim != -1:
                        end_pos = search_start + last_delim + 1
                        break

            chunk = text[current_pos:end_pos].strip()
            if chunk:
                chunks.append(
                    {
                        "text": chunk,
                        "metadata": {
                            "episode": filename,
                            "timestamp": "",
                            "speaker": "",
                        },
                    }
                )

            current_pos = end_pos

        return chunks

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

            # Create chunks from timestamped sections (preferred method)
            # This approach respects speaker boundaries and groups semantically
            if sections["timestamped_chunks"]:
                chunks = self.create_semantic_chunks_from_speakers(
                    timestamped_sections=sections["timestamped_chunks"],
                    filename=file_path.name,
                )
                all_chunks.extend(chunks)
            # Fallback to simple chunking if no timestamped sections
            elif sections["full_text"]:
                print("   ⚠️  No speaker info found, using simple chunking...")
                chunks = self.create_simple_chunks(
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

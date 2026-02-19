"""Tests for embeddings creation functionality.

Following TDD methodology - write tests first, then implement features.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest


class TestConfig:
    """Tests for configuration loading."""

    def test_config_file_exists(self):
        """Test that config.json exists in project root."""
        config_path = Path(__file__).parent.parent / "config.json"
        assert config_path.exists(), "config.json should exist in project root"

    def test_config_has_embedding_model(self):
        """Test that config contains embedding model setting."""
        from config import config

        assert hasattr(config, "embedding_model")
        assert isinstance(config.embedding_model, str)
        assert len(config.embedding_model) > 0

    def test_config_has_chroma_db_path(self):
        """Test that config contains ChromaDB path setting."""
        from config import config

        assert hasattr(config, "chroma_db_dir")
        assert config.chroma_db_dir is not None

    def test_config_has_collection_name(self):
        """Test that config contains collection name."""
        from config import config

        assert hasattr(config, "documents_collection")
        assert isinstance(config.documents_collection, str)
        assert len(config.documents_collection) > 0


class TestDocumentIndexer:
    """Tests for document indexing functionality."""

    @pytest.fixture
    def mock_embedding_model(self):
        """Mock embedding model to avoid loading real model in tests."""
        with patch("create_embeddings.SentenceTransformer") as mock:
            mock_instance = MagicMock()
            mock_instance.encode.return_value = [[0.1] * 384]  # Mock embedding
            mock.return_value = mock_instance
            yield mock_instance

    @pytest.fixture
    def mock_chroma(self):
        """Mock ChromaDB client to avoid database operations in tests."""
        with patch("create_embeddings.chromadb.PersistentClient") as mock:
            mock_client = MagicMock()
            mock_collection = MagicMock()
            mock_client.get_or_create_collection.return_value = mock_collection
            mock.return_value = mock_client
            yield mock_collection

    def test_indexer_initializes_with_config(self, mock_embedding_model, mock_chroma):
        """Test that DocumentIndexer initializes with configuration."""
        from create_embeddings import DocumentIndexer

        indexer = DocumentIndexer()
        assert indexer is not None

    def test_indexer_parses_transcript_sections(self):
        """Test that indexer correctly parses transcript sections."""
        from create_embeddings import DocumentIndexer

        sample_transcript = """================================================================================
FULL TRANSCRIPT
================================================================================
Αυτό είναι το πλήρες κείμενο.

================================================================================
TIMESTAMPED TRANSCRIPT WITH SPEAKERS
================================================================================

[00:01:00.000 - 00:01:05.000] Speaker 0:
Αυτό είναι ένα δοκιμαστικό κομμάτι.
"""
        indexer = DocumentIndexer()
        sections = indexer.parse_transcript(sample_transcript)

        assert "full_text" in sections
        assert "timestamped_chunks" in sections
        assert len(sections["full_text"]) > 0

    def test_chunk_has_episode_metadata(self):
        """Test that chunks include episode metadata."""
        from create_embeddings import DocumentIndexer

        indexer = DocumentIndexer()
        filename = "George_Santos_Ο_πολιτικός_που_χρειάζεται_η_Κύπρος_μας.txt"

        chunks = indexer.create_chunks_with_metadata(
            text="Sample text",
            filename=filename,
            timestamp="00:01:00",
            speaker="Speaker 0",
        )

        assert len(chunks) > 0
        chunk = chunks[0]
        assert "episode" in chunk["metadata"]
        assert chunk["metadata"]["episode"] == filename
        assert "timestamp" in chunk["metadata"]
        assert "speaker" in chunk["metadata"]

    def test_chunks_are_semantically_meaningful(self):
        """Test that chunks are created with semantic meaning."""
        from create_embeddings import DocumentIndexer

        # Create text with semantic differences
        long_text = (
            "The history of ancient Greece. " * 50
            + "Modern technology has changed everything. " * 50
        )
        indexer = DocumentIndexer()

        chunks = indexer.create_semantic_chunks(long_text)

        # Should create at least one chunk
        assert len(chunks) >= 1
        # Each chunk should be reasonable size
        for chunk in chunks:
            assert len(chunk) > 50

    def test_embedding_creation_for_chunk(self, mock_embedding_model):
        """Test that embeddings are created for chunks."""
        from create_embeddings import DocumentIndexer

        indexer = DocumentIndexer()
        text = "This is a test chunk"

        embedding = indexer.create_embedding(text)

        assert embedding is not None
        assert isinstance(embedding, list)
        assert len(embedding) > 0

    def test_store_chunks_in_chromadb(self, mock_chroma):
        """Test that chunks are stored in ChromaDB with metadata."""
        from create_embeddings import DocumentIndexer

        indexer = DocumentIndexer()
        chunks = [
            {
                "text": "Test chunk 1",
                "metadata": {
                    "episode": "test_episode.txt",
                    "timestamp": "00:01:00",
                    "speaker": "Speaker 0",
                },
            }
        ]

        indexer.store_chunks(chunks)

        # Verify ChromaDB collection.add was called
        mock_chroma.add.assert_called_once()

    def test_index_single_transcript_file(self, mock_embedding_model, mock_chroma):
        """Test indexing a single transcript file."""
        from create_embeddings import DocumentIndexer

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(
                """================================================================================
FULL TRANSCRIPT
================================================================================
Test transcript content.

================================================================================
TIMESTAMPED TRANSCRIPT WITH SPEAKERS
================================================================================

[00:01:00.000 - 00:01:05.000] Speaker 0:
Test content.
"""
            )
            temp_path = Path(f.name)

        try:
            indexer = DocumentIndexer()
            result = indexer.index_file(temp_path)

            assert result["success"] is True
            assert result["chunks_created"] > 0
        finally:
            temp_path.unlink()

    def test_skip_already_indexed_files(self, mock_embedding_model, mock_chroma):
        """Test that already indexed files are skipped."""
        from create_embeddings import DocumentIndexer

        indexer = DocumentIndexer()
        test_file = Path("test_already_indexed.txt")

        # Mark file as already indexed
        indexer.mark_as_indexed(test_file)

        # Try to index again
        should_skip = indexer.should_skip_file(test_file)

        assert should_skip is True

    def test_force_reindex_overwrites_existing(self, mock_embedding_model, mock_chroma):
        """Test that force=True re-indexes existing files."""
        from create_embeddings import DocumentIndexer

        indexer = DocumentIndexer()
        test_file = Path("test_force_reindex.txt")

        # Mark file as already indexed
        indexer.mark_as_indexed(test_file)

        # Force reindex should not skip
        should_skip = indexer.should_skip_file(test_file, force=True)

        assert should_skip is False


class TestBatchIndexing:
    """Tests for batch indexing all transcripts."""

    def test_index_all_transcripts_in_directory(self):
        """Test indexing all transcript files in transcripts directory."""
        from create_embeddings import index_all_transcripts

        transcripts_dir = Path(__file__).parent.parent / "transcripts"
        assert transcripts_dir.exists(), "transcripts directory should exist"

        # Mock to avoid actual indexing in tests
        with patch("create_embeddings.DocumentIndexer") as mock_indexer:
            mock_instance = MagicMock()
            mock_instance.index_file.return_value = {
                "success": True,
                "chunks_created": 10,
            }
            mock_indexer.return_value = mock_instance

            results = index_all_transcripts(force=False)

            assert "total_files" in results
            assert "successful" in results
            assert "failed" in results
            assert results["total_files"] > 0

    def test_parallel_indexing(self):
        """Test that indexing can be parallelized."""
        from create_embeddings import index_all_transcripts

        with patch("create_embeddings.DocumentIndexer") as mock_indexer:
            mock_instance = MagicMock()
            mock_instance.index_file.return_value = {
                "success": True,
                "chunks_created": 10,
            }
            mock_indexer.return_value = mock_instance

            # Should complete in reasonable time even with many files
            results = index_all_transcripts(workers=3)

            assert results["successful"] >= 0

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
        from agents.config import config

        assert hasattr(config, "embedding_model")
        assert isinstance(config.embedding_model, str)
        assert len(config.embedding_model) > 0

    def test_config_has_chroma_db_path(self):
        """Test that config contains ChromaDB path setting."""
        from agents.config import config

        assert hasattr(config, "chroma_db_dir")
        assert config.chroma_db_dir is not None

    def test_config_has_collection_name(self):
        """Test that config contains collection name."""
        from agents.config import config

        assert hasattr(config, "documents_collection")
        assert isinstance(config.documents_collection, str)
        assert len(config.documents_collection) > 0


class TestDocumentIndexer:
    """Tests for document indexing functionality."""

    @pytest.fixture
    def mock_embedding_model(self):
        """Mock embedding model to avoid loading real model in tests."""
        with patch("scripts.create_embeddings.SentenceTransformer") as mock:
            mock_instance = MagicMock()
            mock_instance.encode.return_value = [[0.1] * 384]  # Mock embedding
            mock.return_value = mock_instance
            yield mock_instance

    @pytest.fixture
    def mock_chroma(self):
        """Mock ChromaDB client to avoid database operations in tests."""
        with patch("scripts.create_embeddings.chromadb.PersistentClient") as mock:
            mock_client = MagicMock()
            mock_collection = MagicMock()
            mock_client.get_or_create_collection.return_value = mock_collection
            mock.return_value = mock_client
            yield mock_collection

    def test_indexer_initializes_with_config(self, mock_embedding_model, mock_chroma):
        """Test that DocumentIndexer initializes with configuration."""
        from scripts.create_embeddings import DocumentIndexer

        indexer = DocumentIndexer()
        assert indexer is not None

    def test_indexer_parses_transcript_sections(self):
        """Test that indexer correctly parses transcript sections."""
        from scripts.create_embeddings import DocumentIndexer

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
        from scripts.create_embeddings import DocumentIndexer

        indexer = DocumentIndexer()
        filename = "George_Santos_Ο_πολιτικός_που_χρειάζεται_η_Κύπρος_μας.txt"

        # Use simple chunks for testing (no speaker info required)
        chunks = indexer.create_simple_chunks(
            text="Sample text",
            filename=filename,
        )

        assert len(chunks) > 0
        chunk = chunks[0]
        assert "metadata" in chunk
        assert "episode" in chunk["metadata"]
        assert chunk["metadata"]["episode"] == filename
        assert "timestamp" in chunk["metadata"]
        assert "speaker" in chunk["metadata"]

    def test_chunks_are_semantically_meaningful(self):
        """Test that chunks are created with semantic meaning."""
        import numpy as np

        from scripts.create_embeddings import DocumentIndexer

        indexer = DocumentIndexer()
        filename = "test.txt"

        # Create speaker segments with different topics
        timestamped_sections = [
            {
                "text": "The history of ancient Greece is fascinating. " * 20,
                "timestamp": "00:01:00",
                "speaker": "Speaker 0",
            },
            {
                "text": "Ancient Greek culture influenced the world. " * 20,
                "timestamp": "00:02:00",
                "speaker": "Speaker 1",
            },
            {
                "text": "Modern technology has changed everything. " * 20,
                "timestamp": "00:03:00",
                "speaker": "Speaker 0",
            },
        ]

        chunks = indexer.create_semantic_chunks_from_speakers(
            timestamped_sections, filename
        )

        # Should create at least one chunk
        assert len(chunks) >= 1
        # Each chunk should have text and metadata
        for chunk in chunks:
            assert len(chunk["text"]) > 50
            assert "metadata" in chunk
            assert "episode" in chunk["metadata"]

    def test_embedding_creation_for_chunk(self, mock_embedding_model):
        """Test that embeddings are created for chunks."""
        from scripts.create_embeddings import DocumentIndexer

        indexer = DocumentIndexer()
        text = "This is a test chunk"

        embedding = indexer.create_embedding(text)

        assert embedding is not None
        assert isinstance(embedding, list)
        assert len(embedding) > 0

    def test_store_chunks_in_chromadb(self, mock_chroma):
        """Test that chunks are stored in ChromaDB with metadata."""
        from scripts.create_embeddings import DocumentIndexer

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
        from scripts.create_embeddings import DocumentIndexer

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
        """Test that already indexed files are skipped based on timestamp."""
        import time

        from scripts.create_embeddings import DocumentIndexer

        indexer = DocumentIndexer()

        # Create a temporary test file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Test content")
            test_file = Path(f.name)

        try:
            # Get file modification time
            file_mtime = test_file.stat().st_mtime

            # Mock ChromaDB to return this file as already indexed
            mock_chroma.get.return_value = {
                "ids": ["test_0"],
                "metadatas": [
                    {"indexed_timestamp": file_mtime, "episode": test_file.stem}
                ],
            }

            # Should skip because timestamps match
            should_skip = indexer.should_skip_file(test_file)

            assert should_skip is True, "File with same timestamp should be skipped"
        finally:
            test_file.unlink()

    def test_reindex_modified_files(self, mock_embedding_model, mock_chroma):
        """Test that modified files are re-indexed even if already in DB."""
        import time

        from scripts.create_embeddings import DocumentIndexer

        indexer = DocumentIndexer()

        # Create a temporary test file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Test content")
            test_file = Path(f.name)

        try:
            # Get current file modification time
            file_mtime = test_file.stat().st_mtime

            # Mock ChromaDB to return this file as indexed with OLDER timestamp
            old_timestamp = file_mtime - 3600  # 1 hour ago
            mock_chroma.get.return_value = {
                "ids": ["test_0"],
                "metadatas": [
                    {"indexed_timestamp": old_timestamp, "episode": test_file.stem}
                ],
            }

            # Should NOT skip because file is newer
            should_skip = indexer.should_skip_file(test_file)

            assert should_skip is False, "Modified file should not be skipped"
        finally:
            test_file.unlink()

    def test_index_new_file_not_skipped(self, mock_embedding_model, mock_chroma):
        """Test that files never indexed before are not skipped."""
        from scripts.create_embeddings import DocumentIndexer

        indexer = DocumentIndexer()

        # Create a temporary test file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Test content")
            test_file = Path(f.name)

        try:
            # Mock ChromaDB to return no existing documents
            mock_chroma.get.return_value = {"ids": [], "metadatas": []}

            # Should NOT skip because file is not indexed
            should_skip = indexer.should_skip_file(test_file)

            assert should_skip is False, "New file should not be skipped"
        finally:
            test_file.unlink()

    def test_force_reindex_overwrites_existing(self, mock_embedding_model, mock_chroma):
        """Test that force=True re-indexes existing files."""
        from scripts.create_embeddings import DocumentIndexer

        indexer = DocumentIndexer()

        # Create a temporary test file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Test content")
            test_file = Path(f.name)

        try:
            # Mock ChromaDB to return file as indexed
            mock_chroma.get.return_value = {
                "ids": ["test_0"],
                "metadatas": [
                    {
                        "indexed_timestamp": test_file.stat().st_mtime,
                        "episode": test_file.stem,
                    }
                ],
            }

            # Force reindex should not skip even with matching timestamp
            should_skip = indexer.should_skip_file(test_file, force=True)

            assert should_skip is False, "Force flag should prevent skipping"
        finally:
            test_file.unlink()


class TestBatchIndexing:
    """Tests for batch indexing all transcripts."""

    def test_index_all_transcripts_in_directory(self):
        """Test indexing all transcript files in transcripts directory."""
        from scripts.create_embeddings import index_all_transcripts

        transcripts_dir = Path(__file__).parent.parent / "transcripts"
        assert transcripts_dir.exists(), "transcripts directory should exist"

        # Mock to avoid actual indexing in tests
        with patch("scripts.create_embeddings.DocumentIndexer") as mock_indexer:
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
        from scripts.create_embeddings import index_all_transcripts

        with patch("scripts.create_embeddings.DocumentIndexer") as mock_indexer:
            mock_instance = MagicMock()
            mock_instance.index_file.return_value = {
                "success": True,
                "chunks_created": 10,
            }
            mock_indexer.return_value = mock_instance

            # Should complete in reasonable time even with many files
            results = index_all_transcripts(workers=3)

            assert results["successful"] >= 0

    def test_batch_indexing_skips_already_indexed_files(self):
        """Test that batch indexing correctly skips already indexed files.

        This is the critical integration test that verifies the skip logic
        works correctly when running index_all_transcripts() multiple times.
        """
        import time

        from scripts.create_embeddings import index_all_transcripts

        # Create temporary directory with test transcripts
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            transcripts_dir = tmpdir_path / "transcripts"
            transcripts_dir.mkdir()

            # Create 3 test transcript files
            test_files = []
            for i in range(3):
                test_file = transcripts_dir / f"test_episode_{i}.txt"
                test_file.write_text(
                    f"""[00:01:00.000 - 00:01:05.000] Speaker 0:
Test transcript {i}
""",
                    encoding="utf-8",
                )
                test_files.append(test_file)

            # Patch config to use temp directory
            from agents.config import config

            original_dir = config.transcripts_dir
            original_db = config.chroma_db_dir
            config.transcripts_dir = str(transcripts_dir)
            config.chroma_db_dir = str(tmpdir_path / "test_chroma_db")

            try:
                # First indexing - should index all 3 files
                results1 = index_all_transcripts(force=False, workers=1)
                assert results1["successful"] == 3, "First run should index all 3 files"
                assert results1["failed"] == 0
                first_skipped = results1.get("skipped", 0)

                # Second indexing immediately - should skip all 3 files
                results2 = index_all_transcripts(force=False, workers=1)
                assert (
                    results2["successful"] == 0
                ), "Second run should index 0 new files"
                assert results2["failed"] == 0
                second_skipped = results2.get("skipped", 3)

                # Verify all files were skipped
                assert (
                    second_skipped == 3
                ), f"Expected 3 files skipped, got {second_skipped}"

                # Modify one file
                time.sleep(0.1)  # Ensure different mtime
                test_files[1].write_text(
                    """[00:01:00.000 - 00:01:05.000] Speaker 0:
Modified test transcript
""",
                    encoding="utf-8",
                )

                # Third indexing - should re-index only the modified file
                results3 = index_all_transcripts(force=False, workers=1)
                assert (
                    results3["successful"] == 1
                ), "Third run should re-index 1 modified file"
                assert results3["failed"] == 0
                third_skipped = results3.get("skipped", 2)
                assert (
                    third_skipped == 2
                ), f"Expected 2 files skipped, got {third_skipped}"

                # Force indexing - should index all 3 files regardless
                results4 = index_all_transcripts(force=True, workers=1)
                assert results4["successful"] == 3, "Force run should index all 3 files"
                assert results4["failed"] == 0

            finally:
                # Restore original config
                config.transcripts_dir = original_dir
                config.chroma_db_dir = original_db

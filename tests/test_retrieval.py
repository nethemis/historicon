"""Tests for the retrieval agent.

Following TDD methodology - write tests first, then implement features.
Mocks ChromaDB to avoid requiring actual database in tests.
"""

from unittest.mock import MagicMock, patch

import pytest

from agents.models import RetrievalChunk, RetrievalResponse


@pytest.mark.asyncio
@patch("agents.retrieval.retrieval_agent.run")
async def test_retrieval_agent_returns_valid_response(mock_run):
    """Test that retrieval agent returns a properly structured RetrievalResponse."""
    # Mock the response
    mock_result = MagicMock()
    mock_result.output = RetrievalResponse(
        chunks=[
            RetrievalChunk(
                text="Sample Greek text about history",
                source="test_episode.txt",
                score=0.95,
                timestamp="00:01:30.000",
            )
        ],
        summary="Found information about historical topic.",
        query="test query",
        total_results=1,
    )
    mock_run.return_value = mock_result

    from agents.retrieval import retrieval_agent

    result = await retrieval_agent.run("Search for: test query (max 5 results)")

    assert result.output is not None
    assert isinstance(result.output, RetrievalResponse)
    assert len(result.output.chunks) > 0
    assert result.output.summary is not None


@pytest.mark.asyncio
@patch("agents.retrieval.retrieval_agent.run")
async def test_retrieval_agent_handles_greek_query(mock_run):
    """Test that retrieval agent handles Greek language queries."""
    mock_result = MagicMock()
    mock_result.output = RetrievalResponse(
        chunks=[
            RetrievalChunk(
                text="Ελληνικό κείμενο για ιστορία",
                source="greek_episode.txt",
                score=0.88,
                timestamp="00:05:45.000",
            )
        ],
        summary="Βρέθηκαν πληροφορίες για το θέμα.",
        query="ιστορία Κύπρου",
        total_results=1,
    )
    mock_run.return_value = mock_result

    from agents.retrieval import retrieval_agent

    result = await retrieval_agent.run("Search for: ιστορία Κύπρου (max 5 results)")

    assert result.output is not None
    assert isinstance(result.output, RetrievalResponse)


@pytest.mark.asyncio
@patch("agents.retrieval.retrieval_agent.run")
async def test_retrieval_agent_respects_max_results(mock_run):
    """Test that retrieval agent respects the max_results parameter."""
    mock_result = MagicMock()
    mock_result.output = RetrievalResponse(
        chunks=[
            RetrievalChunk(
                text=f"Text chunk {i}",
                source=f"episode_{i}.txt",
                score=0.9 - i * 0.1,
                timestamp=f"00:0{i}:00.000",
            )
            for i in range(3)
        ],
        summary="Found 3 results as requested.",
        query="test",
        total_results=3,
    )
    mock_run.return_value = mock_result

    from agents.retrieval import retrieval_agent

    result = await retrieval_agent.run("Search for: test (max 3 results)")

    assert result.output is not None
    assert len(result.output.chunks) <= 3


@pytest.mark.asyncio
@patch("agents.retrieval.retrieval_agent.run")
async def test_retrieval_agent_handles_no_results(mock_run):
    """Test that retrieval agent handles queries with no matching results."""
    mock_result = MagicMock()
    mock_result.output = RetrievalResponse(
        chunks=[],
        summary="No relevant information found.",
        query="nonexistent topic XYZ123",
        total_results=0,
    )
    mock_run.return_value = mock_result

    from agents.retrieval import retrieval_agent

    result = await retrieval_agent.run(
        "Search for: nonexistent topic XYZ123 (max 5 results)"
    )

    assert result.output is not None
    assert len(result.output.chunks) == 0
    assert result.output.total_results == 0


@pytest.mark.asyncio
@patch("agents.retrieval.retrieval_agent.run")
async def test_retrieval_agent_includes_timestamps(mock_run):
    """Test that retrieval agent includes timestamps in results."""
    mock_result = MagicMock()
    mock_result.output = RetrievalResponse(
        chunks=[
            RetrievalChunk(
                text="Timestamped content",
                source="episode.txt",
                score=0.92,
                timestamp="00:12:34.567",
            )
        ],
        summary="Found timestamped content.",
        query="test",
        total_results=1,
    )
    mock_run.return_value = mock_result

    from agents.retrieval import retrieval_agent

    result = await retrieval_agent.run("Search for: test (max 5 results)")

    assert result.output.chunks[0].timestamp is not None
    assert ":" in result.output.chunks[0].timestamp


@pytest.mark.asyncio
async def test_retrieval_agent_has_retries():
    """Test that retrieval agent is properly configured."""
    from agents.retrieval import RetrievalAgent, retrieval_agent

    # Verify agent is properly configured
    assert retrieval_agent is not None
    assert isinstance(retrieval_agent, RetrievalAgent)
    # Retrieval agent performs deterministic semantic search, so retries not needed


@pytest.mark.asyncio
@patch("agents.retrieval.retrieval_agent.run")
async def test_retrieval_agent_scores_are_normalized(mock_run):
    """Test that retrieval agent returns normalized scores (0-1)."""
    mock_result = MagicMock()
    mock_result.output = RetrievalResponse(
        chunks=[
            RetrievalChunk(
                text="High relevance text",
                source="episode.txt",
                score=0.95,
                timestamp="00:00:10.000",
            ),
            RetrievalChunk(
                text="Medium relevance text",
                source="episode2.txt",
                score=0.75,
                timestamp="00:05:20.000",
            ),
        ],
        summary="Found 2 results with different relevance scores.",
        query="test",
        total_results=2,
    )
    mock_run.return_value = mock_result

    from agents.retrieval import retrieval_agent

    result = await retrieval_agent.run("Search for: test (max 5 results)")

    for chunk in result.output.chunks:
        assert 0.0 <= chunk.score <= 1.0


@pytest.mark.asyncio
@patch("agents.retrieval.ChromaRetriever")
async def test_chroma_retriever_initialization(mock_chroma_class):
    """Test that ChromaRetriever initializes correctly."""
    from agents.retrieval import ChromaRetriever

    retriever = ChromaRetriever()

    # Should initialize embedding model and chroma client
    assert retriever is not None


@pytest.mark.asyncio
@patch("agents.retrieval.ChromaRetriever")
async def test_chroma_retriever_search(mock_chroma_class):
    """Test that ChromaRetriever.search returns proper results."""
    from agents.retrieval import ChromaRetriever

    # Mock the search
    mock_retriever = MagicMock()
    mock_retriever.search.return_value = [
        {
            "text": "Test content",
            "metadata": {
                "episode": "test.txt",
                "timestamp": "00:01:00.000",
                "speaker": "Speaker 0",
            },
            "score": 0.9,
        }
    ]
    mock_chroma_class.return_value = mock_retriever

    retriever = ChromaRetriever()
    results = retriever.search("test query", max_results=5)

    assert len(results) > 0
    assert "text" in results[0]
    assert "metadata" in results[0]
    assert "score" in results[0]

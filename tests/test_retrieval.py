"""Tests for the retrieval module.

The retrieval layer is now a plain function (`search_transcripts`) backed by
`ChromaRetriever`, which does Chroma top-k + CrossEncoder reranking. Tests
mock both the Chroma collection and the CrossEncoder to avoid touching real
models or the database.
"""

from unittest.mock import MagicMock, patch

import pytest

from agents.models import RetrievalResponse


def _make_retriever(docs, metadatas, rerank_logits):
    """Build a MagicMock ChromaRetriever-shape with the canned Chroma + rerank output."""
    retriever = MagicMock()
    retriever.collection.query.return_value = {
        "documents": [docs],
        "metadatas": [metadatas],
        "distances": [[0.1] * len(docs)],
    }
    retriever.embedding_model.encode.return_value = MagicMock(
        tolist=lambda: [0.0] * 8
    )
    retriever.reranker.predict.return_value = rerank_logits
    return retriever


def test_search_transcripts_happy_path():
    """search_transcripts returns chunks with sources and a summary."""
    from agents import retrieval

    fake = _make_retriever(
        docs=["Greek text about history"],
        metadatas=[{"episode": "ep1.txt", "timestamp": "00:01:30.000"}],
        rerank_logits=[2.0],
    )
    # Drive the real ChromaRetriever.search through the mocked attributes.
    with patch.object(retrieval, "get_retriever") as get_r:
        real_search = retrieval.ChromaRetriever.search.__get__(fake)
        fake.search = real_search
        get_r.return_value = fake

        result = retrieval.search_transcripts("ιστορία Κύπρου", max_results=5)

    assert isinstance(result, RetrievalResponse)
    assert result.total_results == 1
    assert result.chunks[0].source == "ep1.txt"
    assert result.chunks[0].timestamp == "00:01:30.000"
    assert 0.0 <= result.chunks[0].score <= 1.0
    assert "ep1.txt" in result.summary


def test_search_transcripts_no_results():
    """Empty Chroma results produce an empty RetrievalResponse with a clear summary."""
    from agents import retrieval

    fake = _make_retriever(docs=[], metadatas=[], rerank_logits=[])
    with patch.object(retrieval, "get_retriever") as get_r:
        fake.search = retrieval.ChromaRetriever.search.__get__(fake)
        get_r.return_value = fake

        result = retrieval.search_transcripts("nonexistent topic", max_results=5)

    assert result.total_results == 0
    assert result.chunks == []
    assert "No relevant information" in result.summary


def test_search_transcripts_swallows_errors():
    """Retrieval failures return an empty response with the error in the summary."""
    from agents import retrieval

    with patch.object(retrieval, "get_retriever", side_effect=RuntimeError("boom")):
        result = retrieval.search_transcripts("query", max_results=5)

    assert result.total_results == 0
    assert "Search error" in result.summary
    assert "boom" in result.summary


def test_reranker_reorders_results():
    """CrossEncoder logits drive the final order, not Chroma distances.

    Chroma returns A, B, C; the reranker scores them 0.1, 0.9, 0.5.
    Expect chunks in order B, C, A.
    """
    from agents import retrieval

    docs = ["doc A", "doc B", "doc C"]
    metas = [{"episode": f"ep{c}.txt"} for c in "ABC"]
    fake = _make_retriever(docs=docs, metadatas=metas, rerank_logits=[0.1, 0.9, 0.5])

    with patch.object(retrieval, "get_retriever") as get_r:
        fake.search = retrieval.ChromaRetriever.search.__get__(fake)
        get_r.return_value = fake

        result = retrieval.search_transcripts("query", max_results=3)

    assert [c.source for c in result.chunks] == ["epB.txt", "epC.txt", "epA.txt"]
    # Scores monotonically decreasing after sort.
    scores = [c.score for c in result.chunks]
    assert scores == sorted(scores, reverse=True)
    # Sigmoid keeps everything in [0, 1].
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_reranker_respects_max_results():
    """max_results truncates the reranked list."""
    from agents import retrieval

    docs = [f"doc {i}" for i in range(5)]
    metas = [{"episode": f"ep{i}.txt"} for i in range(5)]
    fake = _make_retriever(docs=docs, metadatas=metas, rerank_logits=[5, 4, 3, 2, 1])

    with patch.object(retrieval, "get_retriever") as get_r:
        fake.search = retrieval.ChromaRetriever.search.__get__(fake)
        get_r.return_value = fake

        result = retrieval.search_transcripts("query", max_results=2)

    assert result.total_results == 2
    assert [c.source for c in result.chunks] == ["ep0.txt", "ep1.txt"]

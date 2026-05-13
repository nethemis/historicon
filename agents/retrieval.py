"""Retrieval module — RAG search over HistoriCon transcripts.

Two-stage retrieval: ChromaDB semantic search retrieves top-k candidates,
then a CrossEncoder reranks them. Exposed as a plain function (no LLM agent).
"""

import math
import os
from pathlib import Path

import chromadb
import logfire
from chromadb.config import Settings
from sentence_transformers import CrossEncoder, SentenceTransformer

from agents import logfire_setup  # noqa: F401 — import-time Logfire bootstrap
from agents.config import config, get_device
from agents.models import RetrievalChunk, RetrievalResponse


def _sigmoid(x: float) -> float:
    """Map an unbounded CrossEncoder logit into [0, 1]."""
    return 1.0 / (1.0 + math.exp(-x))


class ChromaRetriever:
    """Retrieves and reranks chunks from a ChromaDB collection."""

    def __init__(self):
        logfire.info("Initializing ChromaRetriever")

        self.device = get_device()
        logfire.info(f"Using device: {self.device}")

        logfire.info(f"Loading embedding model: {config.embedding_model}")
        self.embedding_model = SentenceTransformer(
            config.embedding_model,
            device=self.device,
        )

        logfire.info(f"Loading reranking model: {config.reranking_model}")
        self.reranker = CrossEncoder(config.reranking_model, device=self.device)

        logfire.info(f"Connecting to ChromaDB at {config.chroma_db_dir}")
        self.chroma_client = chromadb.PersistentClient(
            path=str(config.chroma_db_dir),
            settings=Settings(anonymized_telemetry=False),
        )

        try:
            self.collection = self.chroma_client.get_collection(
                name=config.documents_collection
            )
            logfire.info(
                f"Connected to collection: {config.documents_collection} "
                f"with {self.collection.count()} documents"
            )
        except Exception as e:
            logfire.error(f"Failed to get collection: {e}")
            raise

    def search(self, query: str, max_results: int = 5) -> list[dict]:
        """Retrieve top-k from Chroma then rerank with a CrossEncoder.

        Args:
            query: Search query (Greek or English).
            max_results: Number of reranked chunks to return.

        Returns:
            List of dicts with keys: text, metadata, score. Score is the
            CrossEncoder relevance squashed to [0, 1] via sigmoid.
        """
        logfire.info(
            f"Searching: {query[:100]} (candidates={config.retrieval_top_k}, "
            f"return={max_results})"
        )

        query_embedding = self.embedding_model.encode(
            query, convert_to_tensor=False
        ).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=config.retrieval_top_k,
            include=["documents", "metadatas", "distances"],
        )

        docs = results["documents"][0] if results["documents"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else []
        if not docs:
            return []

        pairs = [(query, doc) for doc in docs]
        rerank_logits = self.reranker.predict(pairs)

        ranked = sorted(
            zip(docs, metadatas, rerank_logits),
            key=lambda x: x[2],
            reverse=True,
        )[:max_results]

        logfire.info(
            f"Reranked {len(docs)} candidates → returning {len(ranked)}"
        )

        return [
            {"text": doc, "metadata": meta, "score": _sigmoid(float(logit))}
            for doc, meta, logit in ranked
        ]


# Global retriever instance (lazy initialization)
_retriever: ChromaRetriever | None = None


def get_retriever() -> ChromaRetriever:
    """Get or create the global retriever instance."""
    global _retriever
    if _retriever is None:
        _retriever = ChromaRetriever()
    return _retriever


def search_transcripts(
    query: str,
    max_results: int = 5,
) -> RetrievalResponse:
    """Search the podcast transcript database for relevant chunks.

    Args:
        query: Search query in Greek or English.
        max_results: Maximum number of chunks to return (default: 5).

    Returns:
        RetrievalResponse with reranked chunks and a concatenated summary.
    """
    logfire.info("Executing transcript search", query=query[:100], max_results=max_results)

    try:
        results = get_retriever().search(query, max_results=max_results)

        chunks = [
            RetrievalChunk(
                text=r["text"],
                source=r["metadata"].get("episode", "unknown"),
                score=r["score"],
                timestamp=r["metadata"].get("timestamp") or None,
            )
            for r in results
        ]

        if chunks:
            summary = "\n\n".join(f"[{c.source}] {c.text}" for c in chunks)
        else:
            summary = "No relevant information found in the transcripts."

        logfire.info("Search complete", num_chunks=len(chunks), summary_length=len(summary))

        return RetrievalResponse(
            chunks=chunks,
            summary=summary,
            query=query,
            total_results=len(chunks),
        )

    except Exception as e:
        logfire.error(f"Search failed: {e}")
        return RetrievalResponse(
            chunks=[],
            summary=f"Search error: {e}",
            query=query,
            total_results=0,
        )

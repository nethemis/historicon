"""Retrieval module — RAG search over HistoriCon transcripts.

Two-stage retrieval: ChromaDB semantic search retrieves top-k candidates,
then a CrossEncoder reranks them. Exposed as a plain function (no LLM agent).
"""

import math
import os
from pathlib import Path

import chromadb
from chromadb.config import Settings
from opentelemetry import trace
from sentence_transformers import CrossEncoder, SentenceTransformer

from agents import otel_setup  # noqa: F401 — import-time OpenTelemetry bootstrap
from agents.config import config, get_device
from agents.models import RetrievalChunk, RetrievalResponse

# Get tracer for this module
_tracer = otel_setup.get_tracer("historicon.retrieval")


def _sigmoid(x: float) -> float:
    """Map an unbounded CrossEncoder logit into [0, 1]."""
    return 1.0 / (1.0 + math.exp(-x))


class ChromaRetriever:
    """Retrieves and reranks chunks from a ChromaDB collection."""

    def __init__(self):
        with _tracer.start_as_current_span("ChromaRetriever_init") as span:
            print("Initializing ChromaRetriever")

            self.device = get_device()
            print(f"Using device: {self.device}")
            span.set_attribute("device", str(self.device))

            print(f"Loading embedding model: {config.embedding_model}")
            self.embedding_model = SentenceTransformer(
                config.embedding_model,
                device=self.device,
            )
            span.set_attribute("embedding_model", config.embedding_model)

            print(f"Loading reranking model: {config.reranking_model}")
            self.reranker = CrossEncoder(config.reranking_model, device=self.device)
            span.set_attribute("reranking_model", config.reranking_model)

            print(f"Connecting to ChromaDB at {config.chroma_db_dir}")
            self.chroma_client = chromadb.PersistentClient(
                path=str(config.chroma_db_dir),
                settings=Settings(anonymized_telemetry=False),
            )
            span.set_attribute("chroma_db_path", str(config.chroma_db_dir))

            try:
                self.collection = self.chroma_client.get_collection(
                    name=config.documents_collection
                )
                count = self.collection.count()
                print(
                    f"Connected to collection: {config.documents_collection} "
                    f"with {count} documents (path: {config.chroma_db_dir})"
                )
                span.set_attribute("collection_name", config.documents_collection)
                span.set_attribute("document_count", count)
                if count == 0:
                    print(
                        f"Collection '{config.documents_collection}' at "
                        f"{config.chroma_db_dir} is EMPTY — every search will return "
                        "no results. Run scripts/create_embeddings.py."
                    )
                    span.set_attribute("warning", "collection_empty")
            except Exception as e:
                span.record_exception(e)
                print(f"Failed to get collection: {e}")
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
        with _tracer.start_as_current_span("ChromaRetriever_search") as span:
            span.set_attribute("query", query[:100])
            span.set_attribute("max_results", max_results)
            span.set_attribute("candidates", config.retrieval_top_k)

            query_embedding = self.embedding_model.encode(
                query, convert_to_tensor=False
            ).tolist()
            span.set_attribute("embedding_dimension", len(query_embedding))

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=config.retrieval_top_k,
                include=["documents", "metadatas", "distances"],
            )
            span.set_attribute(
                "chroma_results_count", len(results.get("documents", [[]])[0])
            )

            docs = results["documents"][0] if results["documents"] else []
            metadatas = results["metadatas"][0] if results["metadatas"] else []
            if not docs:
                span.set_attribute("result", "no_results")
                return []

            pairs = [(query, doc) for doc in docs]
            rerank_logits = self.reranker.predict(pairs)

            ranked = sorted(
                zip(docs, metadatas, rerank_logits),
                key=lambda x: x[2],
                reverse=True,
            )[:max_results]

            span.set_attribute("reranked_count", len(ranked))
            if ranked:
                span.set_attribute("top_result.source", str(ranked[0][1].get("episode", "unknown")))
                span.set_attribute("top_result.score", round(_sigmoid(float(ranked[0][2])), 4))
                span.set_attribute("top_result.preview", ranked[0][0][:200])

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
    with _tracer.start_as_current_span("search_transcripts") as span:
        span.set_attribute("query", query[:100])
        span.set_attribute("max_results", max_results)

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

            span.set_attribute("num_chunks", len(chunks))
            span.set_attribute("summary_length", len(summary))
            if chunks:
                span.set_attribute("top_result.source", chunks[0].source)
                span.set_attribute("top_result.score", round(chunks[0].score, 4))
                span.set_attribute("top_result.timestamp", chunks[0].timestamp or "")

            return RetrievalResponse(
                chunks=chunks,
                summary=summary,
                query=query,
                total_results=len(chunks),
            )

        except Exception as e:
            span.record_exception(e)
            span.set_attribute("error", str(e))
            return RetrievalResponse(
                chunks=[],
                summary=f"Search error: {e}",
                query=query,
                total_results=0,
                error=str(e),
            )

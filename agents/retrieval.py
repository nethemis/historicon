"""Retrieval Agent - Handles RAG document search for HistoriCon transcripts.

This agent searches through indexed podcast transcripts stored in ChromaDB,
retrieves relevant chunks, and synthesizes responses.
"""

import os
from pathlib import Path

import chromadb
import logfire
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Handle imports for both script execution and module import
try:
    from .logfire_setup import configure_logfire
    from .models import RetrievalChunk, RetrievalResponse
except ImportError:
    from logfire_setup import configure_logfire
    from models import RetrievalChunk, RetrievalResponse

from config import config, get_device

# Configure Logfire for observability (skip in tests)
if os.getenv("LOGFIRE_AUTO_CONFIGURE", "true").lower() == "true":
    configure_logfire()


def load_instructions(agent_name: str) -> str:
    """Load agent instructions from file."""
    instructions_dir = Path(__file__).parent.parent / "instructions"
    instruction_file = instructions_dir / f"{agent_name}.txt"

    if instruction_file.exists():
        return instruction_file.read_text(encoding="utf-8")
    else:
        logfire.warn(f"Instruction file not found: {instruction_file}")
        return "You are a retrieval agent for searching podcast transcripts."


class ChromaRetriever:
    """Handles retrieval from ChromaDB vector store."""

    def __init__(self):
        """Initialize the retriever with embedding model and ChromaDB connection."""
        logfire.info("Initializing ChromaRetriever")

        # Detect best device
        self.device = get_device()
        logfire.info(f"Using device: {self.device}")

        # Initialize embedding model
        logfire.info(f"Loading embedding model: {config.embedding_model}")
        self.embedding_model = SentenceTransformer(
            config.embedding_model,
            device=self.device,
        )

        # Initialize ChromaDB
        logfire.info(f"Connecting to ChromaDB at {config.chroma_db_dir}")
        self.chroma_client = chromadb.PersistentClient(
            path=str(config.chroma_db_dir),
            settings=Settings(anonymized_telemetry=False),
        )

        # Get the collection
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
        """Search ChromaDB for relevant chunks.

        Args:
            query: Search query in Greek or English
            max_results: Maximum number of results to return

        Returns:
            List of dicts with text, metadata, and score
        """
        logfire.info(f"Searching for: {query[:100]}... (max {max_results} results)")

        # Create embedding for the query
        query_embedding = self.embedding_model.encode(
            query, convert_to_tensor=False
        ).tolist()

        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=max_results,
            include=["documents", "metadatas", "distances"],
        )

        logfire.info(f"Found {len(results['documents'][0])} results")

        # Convert to standardized format
        formatted_results = []
        if results["documents"] and len(results["documents"][0]) > 0:
            for i in range(len(results["documents"][0])):
                # Convert distance to similarity score (cosine distance: 0=identical, 2=opposite)
                distance = results["distances"][0][i]
                score = max(0.0, min(1.0, 1.0 - (distance / 2.0)))

                formatted_results.append(
                    {
                        "text": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "score": score,
                    }
                )

        return formatted_results


# Global retriever instance (lazy initialization)
_retriever = None


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
    """
    Search through podcast transcript database for relevant content.

    Args:
        query: The search query (Greek or English)
        max_results: Maximum number of chunks to return (default: 5)

    Returns:
        RetrievalResponse with chunks, summary, and metadata
    """
    logfire.info(
        "Executing transcript search",
        query=query[:100],
        max_results=max_results,
    )

    try:
        # Get retriever and perform search
        retriever = get_retriever()
        results = retriever.search(query, max_results=max_results)

        # Convert to RetrievalChunk objects
        chunks = []
        for result in results:
            chunk = RetrievalChunk(
                text=result["text"],
                source=result["metadata"].get("episode", "unknown"),
                score=result["score"],
                timestamp=result["metadata"].get("timestamp") or None,
            )
            chunks.append(chunk)

        # Create simple summary by concatenating chunk texts
        if chunks:
            summary = "\n\n".join(
                [f"[{chunk.source}] {chunk.text}" for chunk in chunks]
            )
        else:
            summary = "No relevant information found in the transcripts."

        logfire.info(
            "Search complete",
            num_chunks=len(chunks),
            summary_length=len(summary),
        )

        return RetrievalResponse(
            chunks=chunks,
            summary=summary,
            query=query,
            total_results=len(chunks),
        )

    except Exception as e:
        logfire.error(f"Search failed: {e}")
        # Return empty response on error
        return RetrievalResponse(
            chunks=[],
            summary=f"Search error: {str(e)}",
            query=query,
            total_results=0,
        )


class RetrievalAgentResult:
    """Mock result object to match pydantic-ai Agent interface."""

    def __init__(self, output: RetrievalResponse, usage_val=None):
        self.output = output
        self._usage = usage_val

    def usage(self):
        """Return usage information."""
        return self._usage if self._usage else type("Usage", (), {"token_count": 0})()


class RetrievalAgent:
    """Retrieval agent that performs RAG search over podcast transcripts.

    This provides an Agent-like interface but directly performs retrieval
    without LLM calls, since pure semantic search doesn't require reasoning.
    """

    async def run(self, query: str, usage=None) -> RetrievalAgentResult:
        """
        Execute a search query against the transcript database.

        Args:
            query: Natural language query (formatted as "Search for: X (max N results)")
            usage: Optional usage tracking object

        Returns:
            Result object with `.output` field containing RetrievalResponse
        """
        logfire.info(f"Retrieval agent processing query: {query}")

        # Parse the query format: "Search for: {query} (max {N} results)"
        import re

        match = re.search(r"Search for: (.+?) \(max (\d+) results\)", query)
        if match:
            search_query = match.group(1)
            max_results = int(match.group(2))
        else:
            # Fallback: use the whole query
            search_query = query
            max_results = 5

        # Perform the search
        response = search_transcripts(search_query, max_results)

        return RetrievalAgentResult(output=response, usage_val=usage)


# Create the retrieval agent instance
retrieval_agent = RetrievalAgent()

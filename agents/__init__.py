"""HistoriCon MCP server package.

Exports:
- search_transcripts: Direct ChromaDB + CrossEncoder search function.
- Shared Pydantic models for cross-agent IO.
"""

from .models import (
    MemoryResponse,
    MemoryType,
    OrchestratorRequest,
    OrchestratorResponse,
    RetrievalChunk,
    RetrievalResponse,
)
from .retrieval import search_transcripts

__all__ = [
    "search_transcripts",
    "RetrievalChunk",
    "RetrievalResponse",
    "MemoryResponse",
    "MemoryType",
    "OrchestratorRequest",
    "OrchestratorResponse",
]

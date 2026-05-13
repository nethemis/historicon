"""HistoriCon multi-agent RAG package.

Exports:
- web_orchestrator: Pydantic-AI agent serving the web UI / API.
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
from .web_orchestrator import web_orchestrator

__all__ = [
    "search_transcripts",
    "web_orchestrator",
    "RetrievalChunk",
    "RetrievalResponse",
    "MemoryResponse",
    "MemoryType",
    "OrchestratorRequest",
    "OrchestratorResponse",
]

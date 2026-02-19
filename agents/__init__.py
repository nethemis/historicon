"""HistoriCon Multi-Agent RAG System.

This package provides a multi-agent system for querying Greek podcast transcripts
using retrieval-augmented generation (RAG).

Agents:
- web_orchestrator: Main entry point, orchestrates requests
- retrieval: Searches and retrieves relevant document chunks (dummy implementation)

All agents follow project conventions:
- Pydantic BaseModels for all inputs/outputs
- retries=5 configured
- Logfire observability integration
"""

from .models import (
    MemoryResponse,
    MemoryType,
    OrchestratorRequest,
    OrchestratorResponse,
    RetrievalChunk,
    RetrievalResponse,
)
from .retrieval import retrieval_agent
from .web_orchestrator import web_orchestrator

__all__ = [
    "retrieval_agent",
    "web_orchestrator",
    "RetrievalChunk",
    "RetrievalResponse",
    "MemoryResponse",
    "MemoryType",
    "OrchestratorRequest",
    "OrchestratorResponse",
]

"""Pydantic models for the multi-agent system.

All inputs and outputs follow project convention of using Pydantic BaseModels.
"""

from datetime import datetime

from pydantic import BaseModel, Field


class RetrievalChunk(BaseModel):
    """A single retrieved chunk from the RAG system."""

    text: str = Field(description="The content of the retrieved chunk")
    source: str = Field(description="Source filename or identifier")
    score: float = Field(description="Relevance score (0-1)", ge=0.0, le=1.0)
    timestamp: str | None = Field(
        default=None, description="Timestamp from transcript if available"
    )


class RetrievalResponse(BaseModel):
    """Response from the retrieval agent."""

    chunks: list[RetrievalChunk] = Field(description="Retrieved document chunks")
    summary: str = Field(description="Synthesized summary of the retrieved information")
    query: str = Field(description="Original query")
    total_results: int = Field(description="Total number of results found")


class MemoryType(BaseModel):
    """Types of memory operations."""

    STORE: str = "store"
    RETRIEVE: str = "retrieve"


class MemoryResponse(BaseModel):
    """Response from the memory agent."""

    operation: str = Field(description="Type of memory operation performed")
    content: str | None = Field(default=None, description="Retrieved memory content")
    success: bool = Field(description="Whether the operation succeeded")
    message: str = Field(description="Status message")


class OrchestratorRequest(BaseModel):
    """Input request to the web orchestrator."""

    query: str = Field(description="User's question or request")
    max_results: int = Field(
        default=5, description="Maximum number of results to retrieve", ge=1, le=50
    )
    include_timestamps: bool = Field(
        default=True, description="Whether to include timestamps in results"
    )


class OrchestratorResponse(BaseModel):
    """Response from the web orchestrator."""

    answer: str = Field(description="Final answer to user's query")
    sources: list[str] = Field(description="List of source files used")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Response timestamp"
    )

"""Configuration management for HistoriCon RAG system.

Loads settings from config.json and provides type-safe access to configuration.
"""

import json
from pathlib import Path

import torch
from pydantic import BaseModel, Field


def get_device() -> str:
    """Detect and return the best available device for embeddings.

    Returns:
        Device string: 'mps' for Apple Silicon, 'cuda' for NVIDIA, 'cpu' otherwise
    """
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Config(BaseModel):
    """Configuration settings for the RAG system."""

    embedding_model: str = Field(
        description="HuggingFace model for generating embeddings"
    )
    reranking_model: str = Field(description="CrossEncoder model for reranking results")
    chroma_db_dir: str = Field(description="Directory for ChromaDB storage")
    transcripts_dir: str = Field(
        description="Directory containing processed transcript files"
    )
    raw_transcripts_dir: str = Field(
        description="Directory containing raw transcript files from Deepgram"
    )
    audio_input_dir: str = Field(
        description="Directory containing audio files for transcription"
    )
    documents_collection: str = Field(description="Name of ChromaDB collection")
    retrieval_top_k: int = Field(
        default=20, description="Number of candidates to retrieve before reranking"
    )
    similarity_threshold: float = Field(
        default=0.85,
        description="Cosine similarity threshold for grouping speaker segments (0-1). Higher = looser grouping, larger chunks.",
        ge=0.0,
        le=1.0,
    )
    chunk_min_size: int = Field(
        default=1000, description="Minimum chunk size in characters"
    )
    chunk_max_size: int = Field(
        default=10000,
        description="Maximum chunk size in characters (soft limit, speakers never split)",
    )
    max_transcript_tokens: int = Field(
        default=10000,
        description="Maximum tokens to return per get_full_transcript call (for pagination)",
    )
    max_context_tokens: int = Field(
        default=190000,
        description="Maximum total context tokens before issuing warnings (safety margin for 200k limit)",
    )
    embedding_batch_size: int = Field(
        default=32, description="Batch size for creating embeddings"
    )
    indexing_workers: int = Field(
        default=3, description="Number of parallel workers for indexing transcripts"
    )
    transcription_workers: int = Field(
        default=5, description="Number of parallel workers for Deepgram transcription"
    )
    deepgram_model: str = Field(
        default="nova-3", description="Deepgram model for transcription"
    )
    deepgram_language: str = Field(
        default="el", description="Language code for transcription (el=Greek)"
    )
    speaker_map: dict[str, str] = Field(
        default_factory=dict,
        description="Diarized-speaker label → display-name mapping (e.g. 'Speaker 0' → 'Κωνσταντίνος Ψυλλίδης')",
    )

    def ensure_directories(self) -> None:
        """Ensure required directories exist."""
        chroma_path = Path(self.chroma_db_dir)
        chroma_path.mkdir(parents=True, exist_ok=True)


def load_config() -> Config:
    """Load configuration from config.json at the project root."""
    config_path = Path(__file__).parent.parent / "config.json"

    if not config_path.exists():
        raise FileNotFoundError(
            f"config.json not found at {config_path}. "
            "Please create config.json with required settings."
        )

    with open(config_path, "r", encoding="utf-8") as f:
        config_data = json.load(f)

    return Config(**config_data)


# Global config instance
config = load_config()

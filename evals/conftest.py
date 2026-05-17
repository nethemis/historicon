"""Shared fixtures and helpers for evals.

Unlike the tests/ conftest, this does NOT mock the Anthropic API key — evals
intentionally make real LLM calls. Set ANTHROPIC_API_KEY in your environment
before running slow/integration evals.
"""

import os
import sys
from pathlib import Path

import pytest

# Ensure the project root is importable when running eval scripts directly.
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Disable Logfire auto-bootstrap so evals don't need a LOGFIRE_TOKEN.
os.environ.setdefault("LOGFIRE_AUTO_CONFIGURE", "false")


def require_chroma_db():
    """Skip a test if chroma_db/ is not populated."""
    chroma_dir = _PROJECT_ROOT / "chroma_db" / "chroma.sqlite3"
    if not chroma_dir.exists():
        pytest.skip("chroma_db/ not populated — run scripts/create_embeddings.py first")


def require_anthropic_key():
    """Skip a test if ANTHROPIC_API_KEY is not set."""
    if not os.environ.get("ANTHROPIC_API_KEY") or os.environ.get(
        "ANTHROPIC_API_KEY", ""
    ).startswith("sk-ant-test"):
        pytest.skip("ANTHROPIC_API_KEY not set — required for orchestrator evals")


# ── Shared real-episode fixture ────────────────────────────────────────────────

REAL_EPISODE_NAME = "217._Ιστορία_της_Μαφίας_Lucky_Luciano.txt"
REAL_EPISODE_PATH = _PROJECT_ROOT / "transcripts_processed" / REAL_EPISODE_NAME

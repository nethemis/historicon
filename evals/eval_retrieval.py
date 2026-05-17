"""Retrieval quality evals — search_transcripts() returns relevant chunks.

Each case provides a query (Greek or English) and asserts that the returned
chunks contain a fragment matching the expected episode filename.

Requires a populated chroma_db/ directory. Run:
    uv run python evals/eval_retrieval.py

Or via pytest (integration marker):
    uv run pytest evals/eval_retrieval.py -m integration -v
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

from agents.models import RetrievalResponse
from agents.retrieval import search_transcripts
from evals.conftest import require_chroma_db

# ── Task ───────────────────────────────────────────────────────────────────────


async def retrieval_task(query: str) -> RetrievalResponse:
    """Call search_transcripts() — no LLM involved, pure vector + rerank."""
    return search_transcripts(query, max_results=5)


# ── Custom evaluator ───────────────────────────────────────────────────────────


@dataclass
class RetrievalHasSource(Evaluator[str, RetrievalResponse]):
    """Pass if any returned chunk's .source contains the expected filename fragment."""

    fragment: str

    def evaluate(self, ctx: EvaluatorContext[str, RetrievalResponse]) -> bool:
        if ctx.output is None or not ctx.output.chunks:
            return False
        return any(self.fragment in chunk.source for chunk in ctx.output.chunks)


@dataclass
class RetrievalNonEmpty(Evaluator[str, RetrievalResponse]):
    """Pass if at least one chunk was returned."""

    def evaluate(self, ctx: EvaluatorContext[str, RetrievalResponse]) -> bool:
        if ctx.output is None:
            return False
        return ctx.output.total_results > 0


# ── Dataset ────────────────────────────────────────────────────────────────────

CASES: list[Case[str, RetrievalResponse]] = [
    Case(
        name="koskotas_greek",
        inputs="Ποιος ήταν ο Κοσκωτάς και ποιο σκάνδαλο έκανε;",
        evaluators=[RetrievalHasSource("Κοσκωτάς"), RetrievalNonEmpty()],
        metadata={"language": "el", "topic": "political scandal"},
    ),
    Case(
        name="hulk_hogan_greek",
        inputs="Hulk Hogan ιστορία πάλης",
        evaluators=[RetrievalHasSource("Hulk_Hogan"), RetrievalNonEmpty()],
        metadata={"language": "el", "topic": "wrestling"},
    ),
    Case(
        name="lucky_luciano_greek",
        inputs="Λούκι Λουτσιάνο μαφία Νέα Υόρκη",
        evaluators=[RetrievalHasSource("Lucky_Luciano"), RetrievalNonEmpty()],
        metadata={"language": "el", "topic": "mafia"},
    ),
    Case(
        name="george_santos_greek",
        inputs="George Santos ψέματα πολιτικός",
        evaluators=[RetrievalHasSource("George_Santos"), RetrievalNonEmpty()],
        metadata={"language": "el", "topic": "politician"},
    ),
    Case(
        name="eurovision_cyprus",
        inputs="Πώς η Κύπρος επηρέασε την Eurovision;",
        evaluators=[RetrievalHasSource("Eurovision"), RetrievalNonEmpty()],
        metadata={"language": "el", "topic": "music"},
    ),
    Case(
        name="oppenheimer_greek",
        inputs="Oppenheimer ατομική βόμβα Μανχάταν",
        evaluators=[RetrievalHasSource("Oppenheimer"), RetrievalNonEmpty()],
        metadata={"language": "el", "topic": "physics history"},
    ),
    Case(
        name="mafia_series_greek",
        inputs="ιστορία της μαφίας σειρά επεισόδια",
        evaluators=[RetrievalHasSource("Μαφίας"), RetrievalNonEmpty()],
        metadata={"language": "el", "topic": "mafia series"},
    ),
    Case(
        name="jonestown_greek",
        inputs="σφαγή Jonestown Jim Jones λατρεία",
        evaluators=[RetrievalHasSource("Τζόουνσταουν"), RetrievalNonEmpty()],
        metadata={"language": "el", "topic": "cult massacre"},
    ),
]

retrieval_dataset: Dataset[str, RetrievalResponse] = Dataset(
    name="historicon_retrieval_quality",
    cases=CASES,
)


# ── pytest integration ─────────────────────────────────────────────────────────


@pytest.mark.integration
def test_retrieval_quality() -> None:
    """All retrieval quality cases: each Greek/English query returns the right episode."""
    require_chroma_db()
    report = retrieval_dataset.evaluate_sync(retrieval_task)
    report.print(include_input=True, include_output=False, include_durations=True)

    # Assert no total failures (individual case evaluators already record pass/fail)
    failed = [
        c.name
        for c in report.cases
        if any(r.value is False for r in c.assertions.values())
    ]
    assert not failed, f"{len(failed)} retrieval case(s) failed: " + ", ".join(failed)


# ── standalone script ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    chroma_db_path = _PROJECT_ROOT / "chroma_db" / "chroma.sqlite3"
    if not chroma_db_path.exists():
        print(
            "❌ chroma_db/ not found — run: uv run python scripts/create_embeddings.py"
        )
        sys.exit(1)

    print("🔍 Running retrieval quality evals…\n")
    report = retrieval_dataset.evaluate_sync(retrieval_task)
    report.print(include_input=True, include_output=False, include_durations=True)

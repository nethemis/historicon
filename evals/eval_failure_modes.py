"""Failure mode evals — graceful handling of edge cases.

No LLM or heavy models needed. Tests that the system:
  - Does not crash on empty/garbage queries to search_transcripts
  - Returns informative error strings (not exceptions) for bad tool inputs
  - Handles out-of-range timestamps and reversed time windows

All tests are fast. Run standalone:
    uv run python evals/eval_failure_modes.py

Or via pytest:
    uv run pytest evals/eval_failure_modes.py -v
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from pydantic import BaseModel
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

from agents.models import RetrievalResponse
from agents.retrieval import search_transcripts
from agents.web_orchestrator import get_transcript_section

# ── Helpers ────────────────────────────────────────────────────────────────────

# Use a real episode that should be present in transcripts_processed/
_REAL_EPISODE = "217._Ιστορία_της_Μαφίας_Lucky_Luciano.txt"
_REAL_EPISODE_PATH = _PROJECT_ROOT / "transcripts_processed" / _REAL_EPISODE


def _mock_ctx() -> MagicMock:
    """Minimal RunContext mock — only needs .usage for context size check."""
    ctx = MagicMock()
    ctx.usage = MagicMock()
    ctx.usage.total_tokens = MagicMock(return_value=0)
    return ctx


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset 1 — search_transcripts failure modes
# ═══════════════════════════════════════════════════════════════════════════════


class SearchResult(BaseModel):
    """Wraps the outcome of a search_transcripts call."""

    response: RetrievalResponse | None
    error: str | None = None


async def search_task(query: str) -> SearchResult:
    """Call search_transcripts, catching any unexpected exception."""
    try:
        resp = search_transcripts(query, max_results=5)
        return SearchResult(response=resp)
    except Exception as exc:
        return SearchResult(response=None, error=str(exc))


@dataclass
class NoException(Evaluator[str, SearchResult]):
    """Pass if search_transcripts did not raise an unhandled exception."""

    def evaluate(self, ctx: EvaluatorContext[str, SearchResult]) -> bool:
        return ctx.output.error is None


@dataclass
class ReturnsValidResponse(Evaluator[str, SearchResult]):
    """Pass if a RetrievalResponse object was returned (even if empty)."""

    def evaluate(self, ctx: EvaluatorContext[str, SearchResult]) -> bool:
        return ctx.output.response is not None


SEARCH_FAILURE_CASES: list[Case[str, SearchResult]] = [
    Case(
        name="empty_query",
        inputs="",
        evaluators=[NoException(), ReturnsValidResponse()],
        metadata={"scenario": "empty string query"},
    ),
    Case(
        name="garbage_query",
        inputs="xyzzy_$$_total_nonsense_αβγδ_xyz_!!!",
        evaluators=[NoException(), ReturnsValidResponse()],
        metadata={"scenario": "garbage / no matching content"},
    ),
    Case(
        name="very_long_query",
        inputs="ιστορία " * 200,  # 200 repetitions — abnormally long query
        evaluators=[NoException(), ReturnsValidResponse()],
        metadata={"scenario": "abnormally long query string"},
    ),
    Case(
        name="unicode_only",
        inputs="🔥🎙️📻🇨🇾",
        evaluators=[NoException(), ReturnsValidResponse()],
        metadata={"scenario": "emoji-only query"},
    ),
]

search_failure_dataset: Dataset[str, SearchResult] = Dataset(
    name="historicon_search_failure_modes",
    cases=SEARCH_FAILURE_CASES,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset 2 — get_transcript_section failure modes
# ═══════════════════════════════════════════════════════════════════════════════


class TranscriptSectionInput(BaseModel):
    episode_name: str
    start_time: str
    end_time: str


class TranscriptSectionResult(BaseModel):
    result: str
    error: str | None = None


async def transcript_section_task(
    inputs: TranscriptSectionInput,
) -> TranscriptSectionResult:
    """Call get_transcript_section, catching any unexpected exception."""
    ctx = _mock_ctx()
    try:
        result = await get_transcript_section(
            ctx, inputs.episode_name, inputs.start_time, inputs.end_time
        )
        return TranscriptSectionResult(result=result)
    except Exception as exc:
        return TranscriptSectionResult(result="", error=str(exc))


@dataclass
class TranscriptNoException(Evaluator[TranscriptSectionInput, TranscriptSectionResult]):
    """Pass if get_transcript_section did not raise an unhandled exception."""

    def evaluate(
        self, ctx: EvaluatorContext[TranscriptSectionInput, TranscriptSectionResult]
    ) -> bool:
        return ctx.output.error is None


@dataclass
class TranscriptReturnsString(
    Evaluator[TranscriptSectionInput, TranscriptSectionResult]
):
    """Pass if a string was returned (even if empty or an error message)."""

    def evaluate(
        self, ctx: EvaluatorContext[TranscriptSectionInput, TranscriptSectionResult]
    ) -> bool:
        return isinstance(ctx.output.result, str)


@dataclass
class TranscriptResultContains(
    Evaluator[TranscriptSectionInput, TranscriptSectionResult]
):
    """Pass if the result string contains at least one of the given substrings."""

    fragments: list[str]

    def evaluate(
        self, ctx: EvaluatorContext[TranscriptSectionInput, TranscriptSectionResult]
    ) -> bool:
        return any(f in ctx.output.result for f in self.fragments)


TRANSCRIPT_FAILURE_CASES: list[
    Case[TranscriptSectionInput, TranscriptSectionResult]
] = [
    Case(
        name="nonexistent_episode",
        inputs=TranscriptSectionInput(
            episode_name="this_episode_does_not_exist.txt",
            start_time="00:00:00",
            end_time="00:01:00",
        ),
        evaluators=[
            TranscriptNoException(),
            TranscriptReturnsString(),
            TranscriptResultContains(fragments=["Error", "not found", "error"]),
        ],
        metadata={"scenario": "episode file does not exist"},
    ),
    Case(
        name="out_of_range_time",
        inputs=TranscriptSectionInput(
            episode_name=_REAL_EPISODE,
            start_time="99:00:00",
            end_time="99:59:00",
        ),
        evaluators=[
            TranscriptNoException(),
            TranscriptReturnsString(),
        ],
        metadata={
            "scenario": "timestamps beyond end of episode — empty result expected"
        },
    ),
    Case(
        name="reversed_time_window",
        inputs=TranscriptSectionInput(
            episode_name=_REAL_EPISODE,
            start_time="00:10:00",
            end_time="00:05:00",  # end < start
        ),
        evaluators=[
            TranscriptNoException(),
            TranscriptReturnsString(),
        ],
        metadata={
            "scenario": "end timestamp earlier than start — graceful empty result"
        },
    ),
    Case(
        name="empty_episode_name",
        inputs=TranscriptSectionInput(
            episode_name="",
            start_time="00:00:00",
            end_time="00:01:00",
        ),
        evaluators=[
            TranscriptNoException(),
            TranscriptReturnsString(),
        ],
        metadata={"scenario": "empty episode name string"},
    ),
    Case(
        name="malformed_timestamps",
        inputs=TranscriptSectionInput(
            episode_name=_REAL_EPISODE,
            start_time="not_a_timestamp",
            end_time="also_not_a_timestamp",
        ),
        evaluators=[
            TranscriptNoException(),
            TranscriptReturnsString(),
        ],
        metadata={
            "scenario": "malformed timestamp strings — parse_timestamp falls back to 0.0"
        },
    ),
]

transcript_failure_dataset: Dataset[TranscriptSectionInput, TranscriptSectionResult] = (
    Dataset(
        name="historicon_transcript_section_failure_modes",
        cases=TRANSCRIPT_FAILURE_CASES,
    )
)


# ═══════════════════════════════════════════════════════════════════════════════
# pytest integration
# ═══════════════════════════════════════════════════════════════════════════════


def test_search_failure_modes() -> None:
    """search_transcripts() handles edge-case inputs without crashing."""
    report = search_failure_dataset.evaluate_sync(search_task)
    report.print(include_input=True, include_output=True, include_durations=False)

    failed = [
        c.name
        for c in report.cases
        if any(r.value is False for r in c.assertions.values())
    ]
    assert not failed, f"Search failure mode eval failures: {failed}"


def test_transcript_section_failure_modes() -> None:
    """get_transcript_section() handles bad inputs without raising exceptions."""
    report = transcript_failure_dataset.evaluate_sync(transcript_section_task)
    report.print(include_input=False, include_output=True, include_durations=False)

    failed = [
        c.name
        for c in report.cases
        if any(r.value is False for r in c.assertions.values())
    ]
    assert not failed, f"Transcript section failure mode eval failures: {failed}"


# ═══════════════════════════════════════════════════════════════════════════════
# standalone script
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("━" * 60)
    print("🔍  Search failure mode evals…\n")
    r1 = search_failure_dataset.evaluate_sync(search_task)
    r1.print(include_input=True, include_output=True, include_durations=False)

    print("\n" + "━" * 60)
    print("📄  Transcript section failure mode evals…\n")
    r2 = transcript_failure_dataset.evaluate_sync(transcript_section_task)
    r2.print(include_input=False, include_output=True, include_durations=False)

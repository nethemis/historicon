"""Orchestrator evals — tool selection, answer grounding, and end-to-end behaviour.

Uses real Claude (ANTHROPIC_API_KEY required) and real chroma_db.
All tests are marked @pytest.mark.slow + @pytest.mark.integration.

Three datasets:
  1. tool_selection_dataset  — agent calls search_documents for history questions.
     Uses custom span-based evaluator via pydantic-ai's UsageLimits + tool tracking.
  2. answer_grounding_dataset — agent response is non-empty, in the right language,
     and references source episode(s).
  3. e2e_dataset             — end-to-end queries checking answer content.

Run standalone:
    uv run python evals/eval_orchestrator.py

Or via pytest:
    uv run pytest evals/eval_orchestrator.py -v -m "slow and integration"
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from pydantic import BaseModel
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

from agents.config import config
from agents.web_orchestrator import web_orchestrator
from evals.conftest import require_anthropic_key, require_chroma_db

# ── Shared result model ────────────────────────────────────────────────────────


class OrchestratorEvalResult(BaseModel):
    """Captures the agent response and which tools were called."""

    answer: str
    tools_called: list[str]
    error: str | None = None


# ── Task ───────────────────────────────────────────────────────────────────────


async def orchestrator_task(query: str) -> OrchestratorEvalResult:
    """Run the web orchestrator agent with Claude and track tool calls."""
    import anthropic as _anthropic  # noqa: F401 — ensure key is loaded

    tools_called: list[str] = []

    try:
        result = await web_orchestrator.run(
            query,
            model=config.eval_model,
        )

        # Inspect message history for tool calls
        if hasattr(result, "all_messages"):
            for msg in result.all_messages():
                if hasattr(msg, "parts"):
                    for part in msg.parts:
                        tool_name = getattr(part, "tool_name", None)
                        if tool_name:
                            tools_called.append(tool_name)

        return OrchestratorEvalResult(
            answer=result.output,
            tools_called=tools_called,
        )
    except Exception as exc:
        return OrchestratorEvalResult(
            answer="",
            tools_called=tools_called,
            error=str(exc),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset 1 — Tool selection
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class CalledSearchDocuments(Evaluator[str, OrchestratorEvalResult]):
    """Pass if the agent called search_documents at least once."""

    def evaluate(self, ctx: EvaluatorContext[str, OrchestratorEvalResult]) -> bool:
        return "search_documents" in ctx.output.tools_called


@dataclass
class NoError(Evaluator[str, OrchestratorEvalResult]):
    """Pass if the agent run completed without an unhandled exception."""

    def evaluate(self, ctx: EvaluatorContext[str, OrchestratorEvalResult]) -> bool:
        return ctx.output.error is None


TOOL_SELECTION_CASES: list[Case[str, OrchestratorEvalResult]] = [
    Case(
        name="search_for_luciano",
        inputs="Ποιος ήταν ο Lucky Luciano;",
        evaluators=[CalledSearchDocuments(), NoError()],
        metadata={"language": "el", "expected_tool": "search_documents"},
    ),
    Case(
        name="search_for_jonestown",
        inputs="Τι συνέβη στο Jonestown;",
        evaluators=[CalledSearchDocuments(), NoError()],
        metadata={"language": "el", "expected_tool": "search_documents"},
    ),
    Case(
        name="search_mafia_greek",
        inputs="Πες μου για τη μαφία",
        evaluators=[CalledSearchDocuments(), NoError()],
        metadata={"language": "el", "expected_tool": "search_documents"},
    ),
    Case(
        name="search_koskotas_greek",
        inputs="Τι γνωρίζεις για το σκάνδαλο Κοσκωτά;",
        evaluators=[CalledSearchDocuments(), NoError()],
        metadata={"language": "el", "expected_tool": "search_documents"},
    ),
    Case(
        name="search_english_query",
        inputs="Tell me about the Hulk Hogan episode",
        evaluators=[CalledSearchDocuments(), NoError()],
        metadata={"language": "en", "expected_tool": "search_documents"},
    ),
]

tool_selection_dataset: Dataset[str, OrchestratorEvalResult] = Dataset(
    name="historicon_tool_selection",
    cases=TOOL_SELECTION_CASES,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset 2 — Answer grounding
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class AnswerIsNonEmpty(Evaluator[str, OrchestratorEvalResult]):
    """Pass if the answer is a non-empty string."""

    def evaluate(self, ctx: EvaluatorContext[str, OrchestratorEvalResult]) -> bool:
        return bool(ctx.output.answer and ctx.output.answer.strip())


@dataclass
class GreekResponseToGreekQuery(Evaluator[str, OrchestratorEvalResult]):
    """Pass if a Greek-language query gets a response that contains Greek characters."""

    def evaluate(self, ctx: EvaluatorContext[str, OrchestratorEvalResult]) -> bool:
        # Check response contains at least some Greek Unicode characters
        return any(
            "\u0370" <= ch <= "\u03ff" or "\u1f00" <= ch <= "\u1fff"
            for ch in ctx.output.answer
        )


@dataclass
class AnswerMentionsAnyOf(Evaluator[str, OrchestratorEvalResult]):
    """Pass if the answer mentions at least one of the given keywords (case-insensitive)."""

    keywords: list[str]

    def evaluate(self, ctx: EvaluatorContext[str, OrchestratorEvalResult]) -> bool:
        answer_lower = ctx.output.answer.lower()
        return any(kw.lower() in answer_lower for kw in self.keywords)


GROUNDING_CASES: list[Case[str, OrchestratorEvalResult]] = [
    Case(
        name="greek_query_greek_response",
        inputs="Πες μου για τον Oppenheimer",
        evaluators=[AnswerIsNonEmpty(), GreekResponseToGreekQuery(), NoError()],
        metadata={"language": "el", "check": "Greek response to Greek query"},
    ),
    Case(
        name="mafia_query_mentions_episode",
        inputs="Ποιος ήταν ο Frank Costello;",
        evaluators=[
            AnswerIsNonEmpty(),
            AnswerMentionsAnyOf(keywords=["Costello", "μαφία", "mafia", "Κοστέλο"]),
            NoError(),
        ],
        metadata={"language": "en", "check": "answer mentions Frank Costello context"},
    ),
    Case(
        name="greek_political_scandal",
        inputs="Τι ξέρεις για τον Κοσκωτά;",
        evaluators=[
            AnswerIsNonEmpty(),
            GreekResponseToGreekQuery(),
            AnswerMentionsAnyOf(
                keywords=["Κοσκωτά", "Koskotas", "σκάνδαλο", "scandal", "τράπεζα"]
            ),
            NoError(),
        ],
        metadata={
            "language": "el",
            "check": "answer mentions Koskotas scandal context",
        },
    ),
]

grounding_dataset: Dataset[str, OrchestratorEvalResult] = Dataset(
    name="historicon_answer_grounding",
    cases=GROUNDING_CASES,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset 3 — End-to-end
# ═══════════════════════════════════════════════════════════════════════════════

E2E_CASES: list[Case[str, OrchestratorEvalResult]] = [
    Case(
        name="e2e_mafia_series_greek",
        inputs="Ποια επεισόδια για τη μαφία έχει το podcast;",
        evaluators=[
            AnswerIsNonEmpty(),
            CalledSearchDocuments(),
            AnswerMentionsAnyOf(
                keywords=[
                    "Lucky Luciano",
                    "John Gotti",
                    "Frank Costello",
                    "Μαφία",
                    "mafia",
                ]
            ),
            NoError(),
        ],
        metadata={"language": "el", "check": "lists multiple mafia episodes"},
    ),
    Case(
        name="e2e_cypriot_history_greek",
        inputs="Πες μου για ένα κυπριακό ιστορικό επεισόδιο",
        evaluators=[
            AnswerIsNonEmpty(),
            GreekResponseToGreekQuery(),
            CalledSearchDocuments(),
            NoError(),
        ],
        metadata={"language": "el", "check": "references a Cypriot history episode"},
    ),
    Case(
        name="e2e_koskotas_greek",
        inputs="Τι γνωρίζεις για τον Γιώργο Κοσκωτά;",
        evaluators=[
            AnswerIsNonEmpty(),
            CalledSearchDocuments(),
            AnswerMentionsAnyOf(
                keywords=[
                    "Κοσκωτά",
                    "Koskotas",
                    "Bank of Crete",
                    "Τράπεζα Κρήτης",
                    "σκάνδαλο",
                ]
            ),
            NoError(),
        ],
        metadata={"language": "el", "check": "provides substantive Koskotas content"},
    ),
    Case(
        name="e2e_george_santos_english",
        inputs="Tell me about the George Santos episode",
        evaluators=[
            AnswerIsNonEmpty(),
            CalledSearchDocuments(),
            AnswerMentionsAnyOf(
                keywords=[
                    "Santos",
                    "politician",
                    "Congress",
                    "Κύπρος",
                    "Cyprus",
                    "πολιτικός",
                ]
            ),
            NoError(),
        ],
        metadata={"language": "en", "check": "describes George Santos episode"},
    ),
]

e2e_dataset: Dataset[str, OrchestratorEvalResult] = Dataset(
    name="historicon_e2e",
    cases=E2E_CASES,
)


# ═══════════════════════════════════════════════════════════════════════════════
# pytest integration
# ═══════════════════════════════════════════════════════════════════════════════


def _assert_no_failures(report, label: str) -> None:
    failed = [
        c.name
        for c in report.cases
        if any(r.value is False for r in c.assertions.values())
    ]
    assert not failed, f"{label} failures: {failed}"


@pytest.mark.slow
@pytest.mark.integration
def test_tool_selection() -> None:
    """Agent calls search_documents for all history/podcast queries."""
    require_anthropic_key()
    require_chroma_db()
    report = tool_selection_dataset.evaluate_sync(orchestrator_task)
    report.print(include_input=True, include_output=False, include_durations=True)
    _assert_no_failures(report, "Tool selection")


@pytest.mark.slow
@pytest.mark.integration
def test_answer_grounding() -> None:
    """Agent returns non-empty, language-appropriate, on-topic answers."""
    require_anthropic_key()
    require_chroma_db()
    report = grounding_dataset.evaluate_sync(orchestrator_task)
    report.print(include_input=True, include_output=False, include_durations=True)
    _assert_no_failures(report, "Answer grounding")


@pytest.mark.slow
@pytest.mark.integration
def test_e2e() -> None:
    """End-to-end: full pipeline returns substantive answers for Greek/English queries."""
    require_anthropic_key()
    require_chroma_db()
    report = e2e_dataset.evaluate_sync(orchestrator_task)
    report.print(include_input=True, include_output=False, include_durations=True)
    _assert_no_failures(report, "E2E")


# ═══════════════════════════════════════════════════════════════════════════════
# standalone script
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import os

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("❌ ANTHROPIC_API_KEY not set")
        sys.exit(1)

    chroma_db_path = _PROJECT_ROOT / "chroma_db" / "chroma.sqlite3"
    if not chroma_db_path.exists():
        print(
            "❌ chroma_db/ not found — run: uv run python scripts/create_embeddings.py"
        )
        sys.exit(1)

    print("🔧  Running tool selection evals…\n")
    r1 = tool_selection_dataset.evaluate_sync(orchestrator_task)
    r1.print(include_input=True, include_output=False, include_durations=True)

    print("\n📝  Running answer grounding evals…\n")
    r2 = grounding_dataset.evaluate_sync(orchestrator_task)
    r2.print(include_input=True, include_output=False, include_durations=True)

    print("\n🚀  Running end-to-end evals…\n")
    r3 = e2e_dataset.evaluate_sync(orchestrator_task)
    r3.print(include_input=True, include_output=False, include_durations=True)

"""Guardrails evals — on-topic classifier and anti-fabrication validator.

Two datasets:
  1. on_topic_dataset   — check_on_topic() correctly allows/blocks queries.
     Uses the real facebook/bart-large-mnli model (~1.6 GB, loaded once).
     Marked @pytest.mark.slow.

  2. anti_fab_dataset   — validate_no_fabrication() raises ModelRetry for
     invented quotes and passes for grounded text.
     No LLM or heavy model needed — fast.

Run standalone:
    uv run python evals/eval_guardrails.py

Or via pytest:
    uv run pytest evals/eval_guardrails.py -v                  # all
    uv run pytest evals/eval_guardrails.py -v -m "not slow"    # anti-fab only
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from pydantic import BaseModel
from pydantic_ai import ModelRetry
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

from agents.guardrails import (
    _CANDIDATE_LABELS,
    _OFF_TOPIC_LABEL,
    _OFF_TOPIC_THRESHOLD,
    _retrieved_chunks_var,
    get_classifier,
    validate_no_fabrication,
)
from agents.models import RetrievalChunk

# ═══════════════════════════════════════════════════════════════════════════════
# Part 1 — On-topic classifier
# ═══════════════════════════════════════════════════════════════════════════════


class OnTopicResult(BaseModel):
    """Rich result from the on-topic classifier — visible in the eval output table."""

    is_on_topic: bool
    top_label: str
    top_score: float
    all_labels: list[str]
    all_scores: list[float]
    error: str | None = None


async def ontopic_task(query: str) -> OnTopicResult:
    """Call the classifier directly so the full label/score is visible in eval output."""
    try:
        classifier = get_classifier()
        result = classifier(query, _CANDIDATE_LABELS)
        top_label: str = result["labels"][0]
        top_score: float = result["scores"][0]
        is_on_topic = not (
            top_label == _OFF_TOPIC_LABEL and top_score >= _OFF_TOPIC_THRESHOLD
        )
        return OnTopicResult(
            is_on_topic=is_on_topic,
            top_label=top_label,
            top_score=round(top_score, 4),
            all_labels=result["labels"],
            all_scores=[round(s, 4) for s in result["scores"]],
        )
    except Exception as exc:
        return OnTopicResult(
            is_on_topic=True,  # fail open
            top_label="error",
            top_score=0.0,
            all_labels=[],
            all_scores=[],
            error=str(exc),
        )


@dataclass
class IsOnTopic(Evaluator[str, OnTopicResult]):
    """Pass if the classifier's on-topic decision matches expected."""

    expected: bool

    def evaluate(self, ctx: EvaluatorContext[str, OnTopicResult]) -> bool:
        return ctx.output.is_on_topic == self.expected


ONTOPIC_CASES: list[Case[str, OnTopicResult]] = [
    # ── On-topic ──────────────────────────────────────────────────────────────
    Case(
        name="history_question_greek",
        inputs="Ποιος ήταν ο Κοσκωτάς και ποιο σκάνδαλο έκανε;",
        evaluators=[IsOnTopic(expected=True)],
        metadata={"language": "el", "category": "on-topic"},
    ),
    Case(
        name="podcast_meta_greek",
        inputs="Πόσα επεισόδια έχει το HistoriCon podcast;",
        evaluators=[IsOnTopic(expected=True)],
        metadata={"language": "el", "category": "on-topic"},
    ),
    Case(
        name="historical_figure_english",
        inputs="Tell me about Lucky Luciano's rise to power in the American Mafia",
        evaluators=[IsOnTopic(expected=True)],
        metadata={"language": "en", "category": "on-topic"},
    ),
    Case(
        name="cypriot_history_greek",
        inputs="Πες μου για την ιστορία της Κύπρου στον μεσαίωνα",
        evaluators=[IsOnTopic(expected=True)],
        metadata={"language": "el", "category": "on-topic"},
    ),
    # ── Off-topic ─────────────────────────────────────────────────────────────
    Case(
        name="off_topic_weather",
        inputs="What's the weather forecast in Nicosia for this weekend?",
        evaluators=[IsOnTopic(expected=False)],
        metadata={"language": "en", "category": "off-topic"},
    ),
    Case(
        name="off_topic_coding",
        inputs="Write me a Python script to sort a list of numbers",
        evaluators=[IsOnTopic(expected=False)],
        metadata={"language": "en", "category": "off-topic"},
    ),
    Case(
        name="off_topic_recipe_greek",
        inputs="Ποια είναι η καλύτερη συνταγή για παραδοσιακό μουσακά;",
        evaluators=[IsOnTopic(expected=False)],
        metadata={"language": "el", "category": "off-topic"},
    ),
    Case(
        name="off_topic_sports_scores",
        inputs="Who won the Champions League final last night?",
        evaluators=[IsOnTopic(expected=False)],
        metadata={"language": "en", "category": "off-topic"},
    ),
]

ontopic_dataset: Dataset[str, OnTopicResult] = Dataset(
    name="historicon_on_topic_classifier",
    cases=ONTOPIC_CASES,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Part 2 — Anti-fabrication validator
# ═══════════════════════════════════════════════════════════════════════════════


class AntiFabInput(BaseModel):
    """Input for the anti-fabrication eval task."""

    output: str
    chunks: list[RetrievalChunk] = field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}


class AntiFabResult(BaseModel):
    """Result of running the anti-fabrication validator."""

    passed: bool
    raised_model_retry: bool = False


def _make_chunk(text: str) -> RetrievalChunk:
    return RetrievalChunk(text=text, source="test_episode.txt", score=0.9)


def _mock_ctx() -> MagicMock:
    ctx = MagicMock()
    ctx.usage = MagicMock(return_value=MagicMock(total_tokens=100))
    return ctx


async def antifab_task(inputs: AntiFabInput) -> AntiFabResult:
    """Run validate_no_fabrication with the given output and chunk context."""
    _retrieved_chunks_var.set(inputs.chunks)
    ctx = _mock_ctx()
    try:
        await validate_no_fabrication(ctx, inputs.output)
        return AntiFabResult(passed=True, raised_model_retry=False)
    except ModelRetry:
        return AntiFabResult(passed=False, raised_model_retry=True)


@dataclass
class AntiFabPasses(Evaluator[AntiFabInput, AntiFabResult]):
    """Pass if validate_no_fabrication did NOT raise ModelRetry."""

    def evaluate(self, ctx: EvaluatorContext[AntiFabInput, AntiFabResult]) -> bool:
        return ctx.output.passed


@dataclass
class AntiFabBlocks(Evaluator[AntiFabInput, AntiFabResult]):
    """Pass if validate_no_fabrication DID raise ModelRetry (fabrication detected)."""

    def evaluate(self, ctx: EvaluatorContext[AntiFabInput, AntiFabResult]) -> bool:
        return ctx.output.raised_model_retry


ANTIFAB_CASES: list[Case[AntiFabInput, AntiFabResult]] = [
    Case(
        name="no_quotes_passes",
        inputs=AntiFabInput(
            output="Ο Lucky Luciano ήταν ένας από τους πιο σημαντικούς μαφιόζους.",
            chunks=[],
        ),
        evaluators=[AntiFabPasses()],
        metadata={"scenario": "plain prose, no quotes — always passes"},
    ),
    Case(
        name="quote_found_in_chunks_passes",
        inputs=AntiFabInput(
            output='Στο επεισόδιο λέγεται ότι "ο Luciano ήταν ο αρχιτέκτονας της μοντέρνας μαφίας".',
            chunks=[
                _make_chunk(
                    "ο Luciano ήταν ο αρχιτέκτονας της μοντέρνας μαφίας και το πιστεύουμε."
                )
            ],
        ),
        evaluators=[AntiFabPasses()],
        metadata={"scenario": "quote exists verbatim in retrieved chunk"},
    ),
    Case(
        name="greek_guillemet_found_passes",
        inputs=AntiFabInput(
            output="Όπως αναφέρεται: «Ο ρόλος της Κύπρου ήταν καθοριστικός στην Eurovision».",
            chunks=[
                _make_chunk(
                    "Ο ρόλος της Κύπρου ήταν καθοριστικός στην Eurovision και αυτό είναι αλήθεια."
                )
            ],
        ),
        evaluators=[AntiFabPasses()],
        metadata={"scenario": "Greek guillemet quote found in chunks — passes"},
    ),
    Case(
        name="fabricated_english_quote_blocked",
        inputs=AntiFabInput(
            output='The host clearly stated that "this completely invented phrase was said on air".',
            chunks=[
                _make_chunk(
                    "Ο Κοσκωτάς ήταν τραπεζίτης που κατηγορήθηκε για υπεξαίρεση."
                )
            ],
        ),
        evaluators=[AntiFabBlocks()],
        metadata={"scenario": "invented English quote → ModelRetry raised"},
    ),
    Case(
        name="fabricated_greek_quote_blocked",
        inputs=AntiFabInput(
            output='Στο podcast είπαν ότι "αυτή είναι μια εντελώς φανταστική πρόταση που δεν είπε κανείς".',
            chunks=[
                _make_chunk(
                    "Ο Κοσκωτάς ήταν τραπεζίτης που κατηγορήθηκε για υπεξαίρεση."
                )
            ],
        ),
        evaluators=[AntiFabBlocks()],
        metadata={"scenario": "invented Greek quote → ModelRetry raised"},
    ),
    Case(
        name="short_quote_ignored_passes",
        inputs=AntiFabInput(
            output='Ο host είπε "ναι" και συνέχισε.',
            chunks=[],
        ),
        evaluators=[AntiFabPasses()],
        metadata={"scenario": "quote < 10 chars is ignored by extract_quotes"},
    ),
    Case(
        name="no_chunks_no_validation_passes",
        inputs=AntiFabInput(
            output='Ο Luciano είπε "I never lie to my friends, only to my enemies".',
            chunks=[],
        ),
        evaluators=[AntiFabPasses()],
        metadata={
            "scenario": "no chunks retrieved → validator passes through (no search was done)"
        },
    ),
]

antifab_dataset: Dataset[AntiFabInput, AntiFabResult] = Dataset(
    name="historicon_anti_fabrication",
    cases=ANTIFAB_CASES,
)


# ═══════════════════════════════════════════════════════════════════════════════
# pytest integration
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.slow
def test_on_topic_classifier() -> None:
    """On-topic classifier correctly allows history queries and blocks off-topic ones."""
    report = ontopic_dataset.evaluate_sync(ontopic_task)
    report.print(include_input=True, include_output=True, include_durations=True)

    failed = [
        c.name
        for c in report.cases
        if any(r.value is False for r in c.assertions.values())
    ]
    assert not failed, f"On-topic eval failures: {failed}"


def test_anti_fabrication() -> None:
    """Anti-fabrication validator passes grounded quotes and blocks invented ones."""
    report = antifab_dataset.evaluate_sync(antifab_task)
    report.print(include_input=False, include_output=True, include_durations=False)

    failed = [
        c.name
        for c in report.cases
        if any(r.value is False for r in c.assertions.values())
    ]
    assert not failed, f"Anti-fabrication eval failures: {failed}"


# ═══════════════════════════════════════════════════════════════════════════════
# standalone script
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("━" * 60)
    print("🛡️  Anti-fabrication evals (fast, no model needed)…\n")
    antifab_report = antifab_dataset.evaluate_sync(antifab_task)
    antifab_report.print(
        include_input=False, include_output=True, include_durations=False
    )

    print("\n" + "━" * 60)
    print("🧠  On-topic classifier evals (loads ~1.6 GB model)…\n")
    ontopic_report = ontopic_dataset.evaluate_sync(ontopic_task)
    ontopic_report.print(
        include_input=True, include_output=True, include_durations=True
    )

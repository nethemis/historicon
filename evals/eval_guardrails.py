"""Guardrails evals — on-topic classifier and grounding validator.

Two datasets:
  1. on_topic_dataset   — check_on_topic() correctly allows/blocks queries.
     Uses the real MoritzLaurer/mDeBERTa-v3-base-mnli-xnli model (~280 MB).
     Marked @pytest.mark.slow.

  2. grounding_dataset  — check_grounding() detects hallucinated responses and
     passes grounded ones. Uses the same NLI model.
     Marked @pytest.mark.slow.

Run standalone:
    uv run python evals/eval_guardrails.py

Or via pytest:
    uv run pytest evals/eval_guardrails.py -v                  # all
    uv run pytest evals/eval_guardrails.py -v -m "not slow"    # (nothing — both are slow)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from pydantic import BaseModel
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

from agents.guardrails import (
    _CANDIDATE_LABELS,
    _OFF_TOPIC_LABEL,
    _OFF_TOPIC_THRESHOLD,
    _ON_TOPIC_HYPOTHESIS_TEMPLATE,
    check_grounding,
    get_classifier,
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
        result = classifier(
            query, _CANDIDATE_LABELS, hypothesis_template=_ON_TOPIC_HYPOTHESIS_TEMPLATE
        )
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
# Part 2 — Grounding validator
# ═══════════════════════════════════════════════════════════════════════════════


class GroundingInput(BaseModel):
    """Input for the grounding eval task."""

    response: str
    chunks: list[RetrievalChunk] = field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}


class GroundingResult(BaseModel):
    """Result of running the grounding check."""

    is_grounded: bool


def _make_chunk(text: str) -> RetrievalChunk:
    return RetrievalChunk(
        text=text, source="test_episode.txt", score=0.9, timestamp=None
    )


def grounding_task(inputs: GroundingInput) -> GroundingResult:
    """Run check_grounding with the given response and chunk context."""
    context_chunks = [chunk.text for chunk in inputs.chunks]
    is_grounded, _ = check_grounding(inputs.response, context_chunks)
    return GroundingResult(is_grounded=is_grounded)


@dataclass
class GroundingPasses(Evaluator[GroundingInput, GroundingResult]):
    """Pass if check_grounding returned grounded=True."""

    def evaluate(self, ctx: EvaluatorContext[GroundingInput, GroundingResult]) -> bool:
        return ctx.output.is_grounded


@dataclass
class GroundingBlocks(Evaluator[GroundingInput, GroundingResult]):
    """Pass if check_grounding returned grounded=False (hallucination detected)."""

    def evaluate(self, ctx: EvaluatorContext[GroundingInput, GroundingResult]) -> bool:
        return not ctx.output.is_grounded


GROUNDING_CASES: list[Case[GroundingInput, GroundingResult]] = [
    Case(
        name="empty_context_fails_open",
        inputs=GroundingInput(
            response="Ο Lucky Luciano ήταν ένας από τους πιο σημαντικούς μαφιόζους.",
            chunks=[],
        ),
        evaluators=[GroundingPasses()],
        metadata={"scenario": "no chunks retrieved → fail-open, always grounded"},
    ),
    Case(
        name="grounded_response_passes",
        inputs=GroundingInput(
            response="Ο Luciano ήταν ο αρχιτέκτονας της μοντέρνας μαφίας.",
            chunks=[
                _make_chunk(
                    "Ο Luciano ήταν ο αρχιτέκτονας της μοντέρνας μαφίας και "
                    "άλλαξε τη δομή του οργανωμένου εγκλήματος στην Αμερική."
                )
            ],
        ),
        evaluators=[GroundingPasses()],
        metadata={"scenario": "response directly supported by retrieved chunk"},
    ),
    Case(
        name="grounded_greek_response_passes",
        inputs=GroundingInput(
            response="Ο ρόλος της Κύπρου ήταν καθοριστικός στην Eurovision.",
            chunks=[
                _make_chunk(
                    "Ο ρόλος της Κύπρου ήταν καθοριστικός στην Eurovision και "
                    "αυτό είναι αλήθεια σύμφωνα με τα ιστορικά στοιχεία."
                )
            ],
        ),
        evaluators=[GroundingPasses()],
        metadata={"scenario": "Greek response grounded in Greek chunk"},
    ),
    Case(
        name="fabricated_english_response_blocked",
        inputs=GroundingInput(
            response=(
                "The host clearly stated that this completely invented phrase "
                "was said on air and that Greece will become a superpower by 2050."
            ),
            chunks=[
                _make_chunk(
                    "Ο Κοσκωτάς ήταν τραπεζίτης που κατηγορήθηκε για υπεξαίρεση."
                )
            ],
        ),
        evaluators=[GroundingBlocks()],
        metadata={"scenario": "invented English claim not in context → blocked"},
    ),
    Case(
        name="fabricated_greek_response_blocked",
        inputs=GroundingInput(
            response=(
                "Στο podcast είπαν ότι αυτή είναι μια εντελώς φανταστική πρόταση "
                "που δεν είπε κανείς ποτέ στο ιστορικό αρχείο."
            ),
            chunks=[
                _make_chunk(
                    "Ο Κοσκωτάς ήταν τραπεζίτης που κατηγορήθηκε για υπεξαίρεση."
                )
            ],
        ),
        evaluators=[GroundingBlocks()],
        metadata={"scenario": "invented Greek claim not in context → blocked"},
    ),
]

grounding_dataset: Dataset[GroundingInput, GroundingResult] = Dataset(
    name="historicon_grounding",
    cases=GROUNDING_CASES,
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


@pytest.mark.slow
def test_grounding_validator() -> None:
    """Grounding validator passes responses supported by context and blocks fabrications."""
    report = grounding_dataset.evaluate_sync(grounding_task)
    report.print(include_input=False, include_output=True, include_durations=True)

    failed = [
        c.name
        for c in report.cases
        if any(r.value is False for r in c.assertions.values())
    ]
    assert not failed, f"Grounding eval failures: {failed}"


# ═══════════════════════════════════════════════════════════════════════════════
# standalone script
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("━" * 60)
    print("🛡️  Grounding evals (loads NLI model)…\n")
    grounding_report = grounding_dataset.evaluate_sync(grounding_task)
    grounding_report.print(
        include_input=False, include_output=True, include_durations=True
    )

    print("\n" + "━" * 60)
    print("🧠  On-topic classifier evals (loads NLI model)…\n")
    ontopic_report = ontopic_dataset.evaluate_sync(ontopic_task)
    ontopic_report.print(
        include_input=True, include_output=True, include_durations=True
    )

"""Guardrails for the HistoriCon MCP server.

On-topic filter — zero-shot classifier (facebook/bart-large-mnli via
transformers) that blocks queries unrelated to Greek/Cypriot history before
any retrieval is performed.

Call check_on_topic() from tool handlers to guard free-text query parameters.
The classifier is loaded lazily on the first real query (~1.6 GB, cached).

Usage:
    is_ok, error_msg = check_on_topic(query)
    if not is_ok:
        return {"error": error_msg, ...}

In tests, pass a mock classifier to avoid triggering the real model load:
    check_on_topic(query, classifier=my_mock)
"""

from functools import lru_cache

import logfire

# ─── Candidate labels for zero-shot classification ────────────────────────────

_CANDIDATE_LABELS = [
    "a question about historical events or historical figures (not food or sports), or the HistoriCon podcast",
    "a cooking recipe, sports result, weather forecast, or programming task",
]

_OFF_TOPIC_LABEL = (
    "a cooking recipe, sports result, weather forecast, or programming task"
)
_OFF_TOPIC_THRESHOLD = 0.5

_OFF_TOPIC_MSG = (
    "Η ερώτησή σου δεν σχετίζεται με το podcast HistoriCon. "
    "Παρακαλώ ρώτα για την ελληνική ή κυπριακή ιστορία, τα επεισόδια ή τους ομιλητές. "
    "/ This question is outside the scope of the HistoriCon podcast. "
    "Please ask about Greek or Cypriot history, the podcast episodes, or its speakers."
)


def _load_classifier():
    """Load the zero-shot classification pipeline (~1.6 GB, called once)."""
    from transformers import pipeline  # type: ignore[import]

    logfire.info("Loading zero-shot classification model (facebook/bart-large-mnli)")
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")


@lru_cache(maxsize=1)
def get_classifier():
    """Lazy-loaded classifier singleton — heavy load deferred to first query."""
    return _load_classifier()


def check_on_topic(query: str, classifier=None) -> tuple[bool, str]:
    """Return (is_on_topic, error_message). Fails open on classifier errors.

    Pass a mock classifier= in tests to avoid loading the real model.

    Uses multi_label=True so each candidate label gets an independent
    entailment score rather than a normalised softmax. This prevents
    semantically related but off-topic queries (e.g. "Who won the Champions
    League?") from being diluted into the on-topic bucket when competing
    against a historical-events label.
    """
    if classifier is None:
        classifier = get_classifier()

    try:
        result = classifier(query, _CANDIDATE_LABELS, multi_label=True)
        top_label: str = result["labels"][0]
        top_score: float = result["scores"][0]

        logfire.info(
            "On-topic classification",
            query_preview=query[:60],
            top_label=top_label,
            top_score=top_score,
        )

        if top_label == _OFF_TOPIC_LABEL and top_score >= _OFF_TOPIC_THRESHOLD:
            return False, _OFF_TOPIC_MSG

        return True, ""

    except Exception as exc:
        logfire.error("On-topic classifier error", error=str(exc))
        return True, ""  # fail open — don't block legitimate queries on errors

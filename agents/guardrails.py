"""Guardrails for the HistoriCon podcast.

On-topic filter — zero-shot classifier (MoritzLaurer/mDeBERTa-v3-base-mnli-xnli
via transformers) that blocks queries unrelated to Greek/Cypriot history.
The model is multilingual and handles Greek and Cypriot dialect queries natively.

The classifier is loaded lazily on the first real query (~280 MB, cached).

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

# Bilingual labels (English + Greek) improve accuracy for Greek and Cypriot
# queries when using a cross-lingual NLI model.
_ON_TOPIC_LABEL = (
    "a question about Greek history, Cypriot history, the Byzantine Empire, "
    "historical events or figures covered by the HistoriCon Greek-Cypriot podcast "
    "/ ερώτηση σχετική με ελληνική ιστορία, κυπριακή ιστορία, Βυζαντινή Αυτοκρατορία "
    "ή ιστορικά γεγονότα του podcast HistoriCon"
)
_OFF_TOPIC_LABEL = (
    "a question unrelated to Greek or Cypriot history, such as general trivia, "
    "current events, other countries, cooking, sports, programming, or everyday tasks "
    "/ ερώτηση άσχετη με ελληνική ή κυπριακή ιστορία, όπως γενικές γνώσεις, "
    "επικαιρότητα, μαγειρική, αθλητισμός, προγραμματισμός ή καθημερινές εργασίες"
)

_CANDIDATE_LABELS = [_ON_TOPIC_LABEL, _OFF_TOPIC_LABEL]

# Minimum confidence required before blocking a query as off-topic.
# Only fires when off-topic wins AND exceeds this threshold — fail-open otherwise.
_OFF_TOPIC_MIN_SCORE = 0.6

_OFF_TOPIC_MSG = (
    "Η ερώτησή σου δεν σχετίζεται με το podcast HistoriCon. "
    "Παρακαλώ ρώτα για την ελληνική ή κυπριακή ιστορία, τα επεισόδια ή τους ομιλητές. "
    "/ This question is outside the scope of the HistoriCon podcast. "
    "Please ask about Greek or Cypriot history, the podcast episodes, or its speakers."
)


def _load_classifier():
    """Load the zero-shot classification pipeline (~280 MB, called once)."""
    from transformers import pipeline  # type: ignore[import]

    logfire.info(
        "Loading zero-shot classification model "
        "(MoritzLaurer/mDeBERTa-v3-base-mnli-xnli)"
    )
    return pipeline(
        "zero-shot-classification",
        model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
    )


@lru_cache(maxsize=1)
def get_classifier():
    """Lazy-loaded classifier singleton — heavy load deferred to first query."""
    return _load_classifier()


def check_on_topic(query: str, classifier=None) -> tuple[bool, str]:
    """Return (is_on_topic, error_message). Fails open on classifier errors.

    Pass a mock classifier= in tests to avoid loading the real model.

    Uses multi_label=False (softmax over both labels) so the winning label is
    always clear. A query is blocked only when the off-topic label wins AND
    its score meets _OFF_TOPIC_MIN_SCORE. Anything ambiguous passes through
    (fail-open) to avoid blocking legitimate on-topic questions.
    """
    if classifier is None:
        classifier = get_classifier()

    try:
        result = classifier(query, _CANDIDATE_LABELS, multi_label=False)

        # Build a score lookup by label name
        scores_by_label: dict[str, float] = dict(
            zip(result["labels"], result["scores"])
        )
        on_topic_score = scores_by_label.get(_ON_TOPIC_LABEL, 0.0)
        off_topic_score = scores_by_label.get(_OFF_TOPIC_LABEL, 0.0)

        logfire.info(
            "On-topic classification",
            query_preview=query[:60],
            on_topic_score=on_topic_score,
            off_topic_score=off_topic_score,
        )

        # Block only when off-topic wins with sufficient confidence
        if off_topic_score > on_topic_score and off_topic_score >= _OFF_TOPIC_MIN_SCORE:
            return False, _OFF_TOPIC_MSG

        return True, ""

    except Exception as exc:
        logfire.error("On-topic classifier error", error=str(exc))
        return True, ""  # fail open — don't block legitimate queries on errors

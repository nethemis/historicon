"""Tests for guardrail functions.

After removing anti-fabrication code, this file covers only the on-topic
filter (check_on_topic + get_classifier).

Run with: uv run pytest tests/test_guardrails.py -v
"""

from unittest.mock import MagicMock, patch

import pytest

# ─── On-topic classifier ──────────────────────────────────────────────────────


def test_get_classifier_returns_callable():
    from agents.guardrails import get_classifier

    with patch("agents.guardrails._load_classifier", return_value=MagicMock()):
        get_classifier.cache_clear()
        result = get_classifier()
        assert result is not None
        get_classifier.cache_clear()


def test_check_on_topic_history_passes():
    from agents.guardrails import _CANDIDATE_LABELS, check_on_topic

    mock_clf = MagicMock(
        return_value={
            "labels": [_CANDIDATE_LABELS[0], _CANDIDATE_LABELS[1]],
            "scores": [0.85, 0.15],
        }
    )
    is_ok, msg = check_on_topic(
        "Tell me about the 1974 Cyprus coup", classifier=mock_clf
    )
    assert is_ok is True
    assert msg == ""


def test_check_on_topic_coding_blocked():
    from agents.guardrails import _CANDIDATE_LABELS, _OFF_TOPIC_LABEL, check_on_topic

    mock_clf = MagicMock(
        return_value={
            "labels": [_OFF_TOPIC_LABEL, _CANDIDATE_LABELS[0]],
            "scores": [0.9, 0.1],
        }
    )
    is_ok, msg = check_on_topic(
        "Write me a Python function to sort a list", classifier=mock_clf
    )
    assert is_ok is False
    assert len(msg) > 0


def test_check_on_topic_weather_blocked():
    from agents.guardrails import _CANDIDATE_LABELS, _OFF_TOPIC_LABEL, check_on_topic

    mock_clf = MagicMock(
        return_value={
            "labels": [_OFF_TOPIC_LABEL, _CANDIDATE_LABELS[0]],
            "scores": [0.88, 0.12],
        }
    )
    is_ok, _ = check_on_topic("What is the weather today?", classifier=mock_clf)
    assert is_ok is False


def test_check_on_topic_classifier_error_fails_open():
    """When the classifier throws, check_on_topic returns True (fail-open)."""
    from agents.guardrails import check_on_topic

    def _bad_clf(query, labels, multi_label=False):
        raise RuntimeError("Model crashed")

    is_ok, msg = check_on_topic("some query", classifier=_bad_clf)
    assert is_ok is True
    assert msg == ""


def test_check_on_topic_below_threshold_passes():
    """Off-topic label doesn't win (on-topic scores higher) → query passes."""
    from agents.guardrails import _CANDIDATE_LABELS, _OFF_TOPIC_LABEL, check_on_topic

    mock_clf = MagicMock(
        return_value={
            "labels": [_OFF_TOPIC_LABEL, _CANDIDATE_LABELS[0]],
            "scores": [0.3, 0.7],  # off_topic loses → passes through
        }
    )
    is_ok, _ = check_on_topic("Ambiguous query", classifier=mock_clf)
    assert is_ok is True


# ─── Positive-matching: general knowledge that bypassed the old narrow labels ─


def test_check_on_topic_general_trivia_blocked():
    """'What is the capital of France?' should be blocked (not Greek/Cypriot history)."""
    from agents.guardrails import _CANDIDATE_LABELS, _OFF_TOPIC_LABEL, check_on_topic

    # With positive matching the on-topic label loses → blocked
    mock_clf = MagicMock(
        return_value={
            "labels": [_OFF_TOPIC_LABEL, _CANDIDATE_LABELS[0]],
            "scores": [0.65, 0.35],
        }
    )
    is_ok, msg = check_on_topic("What is the capital of France?", classifier=mock_clf)
    assert is_ok is False
    assert len(msg) > 0


def test_check_on_topic_celebrity_question_blocked():
    """'Who is Elon Musk?' should be blocked (not Greek/Cypriot history)."""
    from agents.guardrails import _CANDIDATE_LABELS, _OFF_TOPIC_LABEL, check_on_topic

    mock_clf = MagicMock(
        return_value={
            "labels": [_OFF_TOPIC_LABEL, _CANDIDATE_LABELS[0]],
            "scores": [0.72, 0.28],
        }
    )
    is_ok, msg = check_on_topic("Who is Elon Musk?", classifier=mock_clf)
    assert is_ok is False
    assert len(msg) > 0


def test_check_on_topic_low_confidence_both_passes():
    """When both labels score low the query passes (fail-open — don't block ambiguous queries)."""
    from agents.guardrails import _CANDIDATE_LABELS, _OFF_TOPIC_LABEL, check_on_topic

    # on_topic wins but both scores are very low → uncertain → fail open
    mock_clf = MagicMock(
        return_value={
            "labels": [_CANDIDATE_LABELS[0], _OFF_TOPIC_LABEL],
            "scores": [0.25, 0.18],  # on_topic wins but neither is confident
        }
    )
    is_ok, msg = check_on_topic("Random ambiguous query xyz", classifier=mock_clf)
    assert is_ok is True
    assert msg == ""


def test_check_on_topic_greek_history_high_confidence_passes():
    """Clear Greek history query with strong on-topic score should always pass."""
    from agents.guardrails import _CANDIDATE_LABELS, _OFF_TOPIC_LABEL, check_on_topic

    mock_clf = MagicMock(
        return_value={
            "labels": [_CANDIDATE_LABELS[0], _OFF_TOPIC_LABEL],
            "scores": [0.90, 0.10],
        }
    )
    is_ok, msg = check_on_topic(
        "Πες μου για τον Κοσκωτά και το σκάνδαλο", classifier=mock_clf
    )
    assert is_ok is True
    assert msg == ""


def test_check_on_topic_podcast_question_english_passes():
    """'How did Petros the First die, according to the podcast?' should pass."""
    from agents.guardrails import _CANDIDATE_LABELS, _OFF_TOPIC_LABEL, check_on_topic

    mock_clf = MagicMock(
        return_value={
            "labels": [_CANDIDATE_LABELS[0], _OFF_TOPIC_LABEL],
            "scores": [0.75, 0.25],
        }
    )
    is_ok, msg = check_on_topic(
        "How did Petros the First die, according to the podcast?",
        classifier=mock_clf,
    )
    assert is_ok is True
    assert msg == ""


def test_check_on_topic_podcast_question_greek_passes():
    """Greek podcast query 'Σύμφωνα με το podcast πώς πέθανε ο Πέτρος ο Πρώτος;' should pass."""
    from agents.guardrails import _CANDIDATE_LABELS, _OFF_TOPIC_LABEL, check_on_topic

    mock_clf = MagicMock(
        return_value={
            "labels": [_CANDIDATE_LABELS[0], _OFF_TOPIC_LABEL],
            "scores": [0.78, 0.22],
        }
    )
    is_ok, msg = check_on_topic(
        "Σύμφωνα με το podcast πώς πέθανε ο Πέτρος ο Πρώτος;",
        classifier=mock_clf,
    )
    assert is_ok is True
    assert msg == ""

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
    """Off-topic label below threshold should not block the query."""
    from agents.guardrails import _CANDIDATE_LABELS, _OFF_TOPIC_LABEL, check_on_topic

    mock_clf = MagicMock(
        return_value={
            "labels": [_OFF_TOPIC_LABEL, _CANDIDATE_LABELS[0]],
            "scores": [0.3, 0.7],  # below 0.5 threshold
        }
    )
    is_ok, _ = check_on_topic("Ambiguous query", classifier=mock_clf)
    assert is_ok is True

"""Tests for the HistoriCon guardrails HTTP server.

All classifier calls are mocked to avoid loading the NLI model.
"""

from unittest.mock import MagicMock, patch

import pytest
from starlette.testclient import TestClient

from agents.guardrails import _OFF_TOPIC_LABEL, _ON_TOPIC_LABEL
from agents.guardrails_server import app

# ─── helpers ──────────────────────────────────────────────────────────────────


def _make_classifier(is_on_topic: bool):
    mock = MagicMock()
    if is_on_topic:
        mock.return_value = {
            "labels": [_ON_TOPIC_LABEL, _OFF_TOPIC_LABEL],
            "scores": [0.9, 0.1],
        }
    else:
        mock.return_value = {
            "labels": [_OFF_TOPIC_LABEL, _ON_TOPIC_LABEL],
            "scores": [0.8, 0.2],
        }
    return mock


# ─── /check-topic ─────────────────────────────────────────────────────────────


def test_check_topic_on_topic_returns_true():
    """/check-topic returns on_topic=true for a history query."""
    with patch(
        "agents.guardrails.get_classifier",
        return_value=_make_classifier(is_on_topic=True),
    ):
        client = TestClient(app)
        resp = client.post(
            "/check-topic", json={"query": "Τι έγινε στην Κύπρο το 1974;"}
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["on_topic"] is True
    assert data["message"] == ""


def test_check_topic_off_topic_returns_false():
    """/check-topic returns on_topic=false with an error message for off-topic queries."""
    with patch(
        "agents.guardrails.get_classifier",
        return_value=_make_classifier(is_on_topic=False),
    ):
        client = TestClient(app)
        resp = client.post(
            "/check-topic", json={"query": "What is the best pasta recipe?"}
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["on_topic"] is False
    assert len(data["message"]) > 0


def test_check_topic_empty_query_fails_open():
    """/check-topic with an empty query returns on_topic=true without calling classifier."""
    with patch("agents.guardrails.get_classifier") as mock_get:
        client = TestClient(app)
        resp = client.post("/check-topic", json={"query": ""})

    mock_get.assert_not_called()
    assert resp.status_code == 200
    assert resp.json()["on_topic"] is True


def test_check_topic_bad_body_fails_open():
    """/check-topic with a malformed body returns on_topic=true (fail open)."""
    with patch("agents.guardrails.get_classifier"):
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            "/check-topic",
            content=b"not json",
            headers={"content-type": "application/json"},
        )

    assert resp.status_code == 200
    assert resp.json()["on_topic"] is True


def test_check_topic_classifier_error_fails_open():
    """/check-topic fails open when the classifier raises an exception."""

    def _exploding(query, labels, multi_label=False):
        raise RuntimeError("Model unavailable")

    with patch("agents.guardrails.get_classifier", return_value=_exploding):
        client = TestClient(app)
        resp = client.post("/check-topic", json={"query": "Τι έγινε το 1821;"})

    assert resp.status_code == 200
    assert resp.json()["on_topic"] is True

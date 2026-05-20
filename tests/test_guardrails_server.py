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


def _make_grounding_classifier(is_grounded: bool):
    mock = MagicMock()
    # Direct NLI: single-label, entailment score via multi_label=True
    mock.return_value = {
        "labels": ["hypothesis"],
        "scores": [0.85 if is_grounded else 0.2],
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


# ─── /check-grounding ─────────────────────────────────────────────────────────


def test_check_grounding_grounded_returns_true():
    """/check-grounding returns grounded=true when response matches context."""
    with patch(
        "agents.guardrails.get_classifier",
        return_value=_make_grounding_classifier(is_grounded=True),
    ):
        client = TestClient(app)
        resp = client.post(
            "/check-grounding",
            json={
                "response": "Ο Κοσκωτάς ήταν τραπεζίτης.",
                "context_chunks": [
                    "Ο Κοσκωτάς ήταν τραπεζίτης που κατηγορήθηκε για υπεξαίρεση."
                ],
            },
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["grounded"] is True
    assert data["message"] == ""


def test_check_grounding_not_grounded_returns_false():
    """/check-grounding returns grounded=false with a message when not grounded."""
    with patch(
        "agents.guardrails.get_classifier",
        return_value=_make_grounding_classifier(is_grounded=False),
    ):
        client = TestClient(app)
        resp = client.post(
            "/check-grounding",
            json={
                "response": "Ο Κοσκωτάς έγινε πρόεδρος της Ελλάδας.",
                "context_chunks": ["Ο Κοσκωτάς ήταν τραπεζίτης."],
            },
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["grounded"] is False
    assert len(data["message"]) > 0


def test_check_grounding_bad_body_fails_open():
    """/check-grounding with malformed body returns grounded=true (fail open)."""
    with patch("agents.guardrails.get_classifier"):
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            "/check-grounding",
            content=b"not json",
            headers={"content-type": "application/json"},
        )

    assert resp.status_code == 200
    assert resp.json()["grounded"] is True


def test_check_grounding_empty_response_fails_open():
    """/check-grounding with empty response returns grounded=true without calling classifier."""
    with patch("agents.guardrails.get_classifier") as mock_get:
        client = TestClient(app)
        resp = client.post(
            "/check-grounding",
            json={"response": "", "context_chunks": ["Some context."]},
        )

    mock_get.assert_not_called()
    assert resp.status_code == 200
    assert resp.json()["grounded"] is True


def test_check_grounding_empty_chunks_fails_open():
    """/check-grounding with no context chunks returns grounded=true without calling classifier."""
    with patch("agents.guardrails.get_classifier") as mock_get:
        client = TestClient(app)
        resp = client.post(
            "/check-grounding",
            json={"response": "Some response.", "context_chunks": []},
        )

    mock_get.assert_not_called()
    assert resp.status_code == 200
    assert resp.json()["grounded"] is True


def test_check_grounding_response_includes_unverified_sentences_field():
    """/check-grounding always includes unverified_sentences in the response."""
    with patch(
        "agents.guardrails.get_classifier",
        return_value=_make_grounding_classifier(is_grounded=True),
    ):
        client = TestClient(app)
        resp = client.post(
            "/check-grounding",
            json={
                "response": "Ο Κοσκωτάς ήταν τραπεζίτης.",
                "context_chunks": ["Ο Κοσκωτάς ήταν τραπεζίτης."],
            },
        )

    assert resp.status_code == 200
    data = resp.json()
    assert "unverified_sentences" in data
    assert isinstance(data["unverified_sentences"], list)


def test_check_grounding_not_grounded_returns_unverified_sentences():
    """/check-grounding returns specific unverified sentences when not grounded."""
    from agents.guardrails import _NOT_GROUNDED_MSG

    def _mock_detailed(response, context_chunks):
        return False, _NOT_GROUNDED_MSG, ["The fabricated claim here."]

    with patch(
        "agents.guardrails_server.check_grounding_detailed", side_effect=_mock_detailed
    ):
        client = TestClient(app)
        resp = client.post(
            "/check-grounding",
            json={
                "response": "Real sentence. The fabricated claim here.",
                "context_chunks": ["Some context."],
            },
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["grounded"] is False
    assert data["unverified_sentences"] == ["The fabricated claim here."]

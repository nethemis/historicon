"""Integration tests for the guardrails pipeline and OpenWebUI filter.

Organized into four categories:

1. Real-model grounding (``slow``) — loads mDeBERTa and runs actual NLI.
   Catches regressions such as the _MAX_CONTEXT_CHARS token-overflow bug that
   caused every sentence to score below the entailment threshold and the server
   to return an empty ``unverified_sentences`` list.

2. Guardrails server HTTP (``slow``) — uses Starlette TestClient with the real
   NLI model.  Verifies that /check-grounding returns a non-empty
   ``unverified_sentences`` list for fabricated content so the filter can show
   specific claims rather than a generic disclaimer.

3. Filter outlet logic (fast, mocked server) — verifies that the OpenWebUI
   filter correctly formats the server response:
   - non-empty ``unverified_sentences`` → bullet list of specific claims
   - empty ``unverified_sentences``     → generic disclaimer (the fallback)

4. Filter context extraction (fast) — verifies that the filter correctly reads
   ``role="tool"`` messages as context chunks from the OpenWebUI message body.

Run all:           uv run pytest tests/test_integration.py -v
Skip slow:         uv run pytest tests/test_integration.py -v -m "not slow"
Run only slow:     uv run pytest tests/test_integration.py -v -m slow
"""

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from starlette.testclient import TestClient

# ─── Helpers ─────────────────────────────────────────────────────────────────


def _load_filter():
    """Dynamically import the Filter class from openwebui/historicon_filter.py.

    The filter lives outside the ``agents`` package and is designed to be pasted
    into OpenWebUI, so it is not importable via the normal package system.
    """
    filter_path = Path(__file__).parent.parent / "openwebui" / "historicon_filter.py"
    spec = importlib.util.spec_from_file_location("historicon_filter", filter_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Filter


def _body_with_tool_messages(assistant_content: str, *tool_contents: str) -> dict:
    """Build a realistic OpenWebUI messages body that includes MCP tool results.

    ``tool_contents`` are passed as plain strings.  For JSON-formatted MCP
    results use ``_body_with_json_tool_messages`` instead.
    """
    messages = [{"role": "user", "content": "Who is the oldest person in the podcast?"}]
    # Simulate the assistant making a tool call then receiving results
    messages.append(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": "search_documents", "arguments": "{}"}}
            ],
        }
    )
    for i, content in enumerate(tool_contents):
        messages.append(
            {"role": "tool", "content": content, "tool_call_id": f"call_{i}"}
        )
    messages.append({"role": "assistant", "content": assistant_content})
    return {"messages": messages}


def _body_without_tool_messages(assistant_content: str) -> dict:
    """Build a body where the model answered directly without calling any MCP tool."""
    return {
        "messages": [
            {"role": "user", "content": "Who is the oldest person?"},
            {"role": "assistant", "content": assistant_content},
        ]
    }


# ─── Category 1: Real-model grounding (slow) ─────────────────────────────────


@pytest.mark.slow
def test_real_model_grounded_sentence_passes():
    """A sentence directly supported by the context passes the grounding check."""
    from agents.guardrails import check_grounding, get_classifier

    clf = get_classifier()
    context = ["Ο Κοσκωτάς κατηγορήθηκε για υπεξαίρεση από την Τράπεζα Κρήτης."]
    sentence = "Ο Κοσκωτάς κατηγορήθηκε για υπεξαίρεση."

    is_ok, _ = check_grounding(sentence, context, classifier=clf)

    assert is_ok is True, "A grounded sentence should not be flagged."


@pytest.mark.slow
def test_real_model_fabricated_sentence_flagged():
    """A sentence not supported by the context is flagged and returns a warning."""
    from agents.guardrails import check_grounding, get_classifier

    clf = get_classifier()
    context = ["Ο Κοσκωτάς κατηγορήθηκε για υπεξαίρεση από την Τράπεζα Κρήτης."]
    fabricated = "Ο Κοσκωτάς έγινε πρόεδρος της Ελλάδας το 1990."

    is_ok, msg = check_grounding(fabricated, context, classifier=clf)

    assert is_ok is False, "A fabricated sentence should be flagged."
    assert len(msg) > 0


@pytest.mark.slow
def test_real_model_grounding_detailed_returns_specific_unverified_sentences():
    """check_grounding_detailed returns a non-empty unverified_sentences list.

    Regression test for the _MAX_CONTEXT_CHARS=1500 token-overflow bug.
    When the context exceeded ~500 tokens the hypothesis was truncated to
    nothing, every sentence scored below the entailment threshold, and
    check_grounding_detailed fell back to the whole-response check which
    returns ``unverified_sentences=[]``.  The filter then showed the generic
    disclaimer instead of the specific fabricated claim.
    """
    from agents.guardrails import check_grounding_detailed, get_classifier

    clf = get_classifier()
    context = [
        "Ο Κοσκωτάς ήταν τραπεζίτης που κατηγορήθηκε για υπεξαίρεση από την "
        "Τράπεζα Κρήτης. Συνελήφθη στις ΗΠΑ και εκδόθηκε στην Ελλάδα."
    ]
    # Mix a grounded sentence with a clearly fabricated one
    response = (
        "Ο Κοσκωτάς κατηγορήθηκε για υπεξαίρεση. "
        "Μετά τη δίκη έγινε πρόεδρος της Ελλάδας το 1990."
    )

    is_grounded, _, unverified = check_grounding_detailed(
        response, context, classifier=clf
    )

    assert is_grounded is False
    assert len(unverified) > 0, (
        "Expected at least one unverified sentence. Got an empty list — this "
        "may indicate a token-overflow regression: check _MAX_CONTEXT_CHARS."
    )


@pytest.mark.slow
def test_real_model_grounding_detailed_all_grounded_returns_empty_list():
    """When all sentences are supported by the context, unverified_sentences is []."""
    from agents.guardrails import check_grounding_detailed, get_classifier

    clf = get_classifier()
    context = [
        "Ο Κοσκωτάς κατηγορήθηκε για υπεξαίρεση από την Τράπεζα Κρήτης. "
        "Συνελήφθη στις ΗΠΑ και εκδόθηκε στην Ελλάδα."
    ]
    response = "Ο Κοσκωτάς κατηγορήθηκε για υπεξαίρεση από την Τράπεζα Κρήτης."

    is_grounded, _, unverified = check_grounding_detailed(
        response, context, classifier=clf
    )

    assert is_grounded is True
    assert unverified == []


# ─── Category 2: Guardrails server HTTP (slow) ───────────────────────────────


@pytest.mark.slow
def test_server_grounding_returns_nonempty_unverified_for_fabricated():
    """POST /check-grounding returns a non-empty unverified_sentences list.

    If this list is empty the filter falls back to the generic disclaimer
    instead of showing which specific claims could not be verified.
    """
    from agents.guardrails_server import app

    with TestClient(app) as client:
        resp = client.post(
            "/check-grounding",
            json={
                "response": "Ο Κοσκωτάς έγινε πρόεδρος της Ελλάδας το 1990.",
                "context_chunks": [
                    "Ο Κοσκωτάς κατηγορήθηκε για υπεξαίρεση από την Τράπεζα Κρήτης."
                ],
            },
        )

    data = resp.json()
    assert resp.status_code == 200
    assert data["grounded"] is False
    assert isinstance(data["unverified_sentences"], list)
    assert len(data["unverified_sentences"]) > 0, (
        "Server returned empty unverified_sentences for clearly fabricated content. "
        "The filter will show a generic disclaimer instead of the specific claim. "
        "Check _MAX_CONTEXT_CHARS — token overflow causes this."
    )


@pytest.mark.slow
def test_server_grounding_returns_empty_unverified_for_grounded():
    """POST /check-grounding returns grounded=True and [] for well-grounded content."""
    from agents.guardrails_server import app

    with TestClient(app) as client:
        resp = client.post(
            "/check-grounding",
            json={
                "response": "Ο Κοσκωτάς κατηγορήθηκε για υπεξαίρεση από την Τράπεζα Κρήτης.",
                "context_chunks": [
                    "Ο Κοσκωτάς ήταν τραπεζίτης που κατηγορήθηκε για υπεξαίρεση "
                    "από την Τράπεζα Κρήτης. Συνελήφθη στις ΗΠΑ."
                ],
            },
        )

    data = resp.json()
    assert resp.status_code == 200
    assert data["grounded"] is True
    assert data["unverified_sentences"] == []


# ─── Category 3: Filter outlet formatting (fast, mocked server) ──────────────


async def test_filter_outlet_shows_bullet_list_when_server_returns_unverified():
    """When server returns non-empty unverified_sentences, outlet shows a bullet list.

    This is the correct production behaviour.  If it fails, the filter is
    swallowing the specific sentences and showing the generic disclaimer.
    """
    Filter = _load_filter()
    f = Filter()
    body = _body_with_tool_messages(
        "Ο Κοσκωτάς έγινε πρόεδρος της Ελλάδας το 1990.",
        "Ο Κοσκωτάς κατηγορήθηκε για υπεξαίρεση από την Τράπεζα Κρήτης.",
    )

    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "grounded": False,
        "message": "generic warning",
        "unverified_sentences": ["Ο Κοσκωτάς έγινε πρόεδρος της Ελλάδας το 1990."],
    }
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.post", return_value=mock_resp):
        result = await f.outlet(body)

    last = result["messages"][-1]["content"]
    assert "- Ο Κοσκωτάς έγινε πρόεδρος" in last, (
        "Expected a bullet-list item for the unverified sentence, got:\n" + last
    )
    assert "generic warning" not in last


async def test_filter_outlet_shows_generic_disclaimer_when_unverified_list_empty():
    """grounded=False but unverified_sentences=[] → generic disclaimer, no bullet list.

    Documents the fallback behaviour when the NLI layer returns an empty list
    (e.g., due to token overflow).  The filter itself is correct here — the
    root cause is in the server/model layer.
    """
    Filter = _load_filter()
    f = Filter()
    body = _body_with_tool_messages("Some response.", "Some context.")

    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "grounded": False,
        "message": "server message",
        "unverified_sentences": [],  # simulates the token-overflow fallback
    }
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.post", return_value=mock_resp):
        result = await f.outlet(body)

    last = result["messages"][-1]["content"]
    assert "server message" in last or "⚠️" in last
    assert "- " not in last, "Should not contain a bullet list when unverified is empty"


async def test_filter_outlet_no_tool_messages_uses_generic_disclaimer_without_calling_server():
    """No tool results → generic disclaimer appended immediately, server not called.

    If the model answered without using any MCP tool there is no retrieved
    context to ground against, so the filter flags the response unconditionally.
    """
    Filter = _load_filter()
    f = Filter()
    body = _body_without_tool_messages("Some response about history.")

    with patch("requests.post") as mock_post:
        result = await f.outlet(body)

    mock_post.assert_not_called()
    last = result["messages"][-1]["content"]
    assert "⚠️" in last


async def test_filter_outlet_grounded_response_unchanged():
    """When server says grounded=True, the assistant message is not modified."""
    Filter = _load_filter()
    f = Filter()
    original_content = "Ο Κοσκωτάς κατηγορήθηκε για υπεξαίρεση."
    body = _body_with_tool_messages(
        original_content,
        "Ο Κοσκωτάς κατηγορήθηκε για υπεξαίρεση από την Τράπεζα Κρήτης.",
    )

    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "grounded": True,
        "message": "",
        "unverified_sentences": [],
    }
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.post", return_value=mock_resp):
        result = await f.outlet(body)

    last = result["messages"][-1]["content"]
    assert (
        last == original_content
    ), "Grounded response should not have disclaimer appended."


# ─── Category 4: Filter context extraction (fast) ────────────────────────────


async def test_filter_outlet_calls_server_with_tool_messages_as_context_chunks():
    """Tool messages are passed as context_chunks in the /check-grounding POST body."""
    Filter = _load_filter()
    f = Filter()
    body = _body_with_tool_messages("answer", "first tool result", "second tool result")

    captured: dict = {}
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "grounded": True,
        "message": "",
        "unverified_sentences": [],
    }
    mock_resp.raise_for_status = MagicMock()

    def _capture(_url, json=None, **_kwargs):
        captured.update(json or {})
        return mock_resp

    with patch("requests.post", side_effect=_capture):
        await f.outlet(body)

    assert captured.get("context_chunks") == ["first tool result", "second tool result"]
    assert captured.get("response") == "answer"


async def test_filter_outlet_calls_server_with_correct_assistant_content():
    """The response field sent to the server is the last assistant message."""
    Filter = _load_filter()
    f = Filter()
    body = _body_with_tool_messages("The actual LLM answer.", "some context")

    captured: dict = {}
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "grounded": True,
        "message": "",
        "unverified_sentences": [],
    }
    mock_resp.raise_for_status = MagicMock()

    def _capture(_url, json=None, **_kwargs):
        captured.update(json or {})
        return mock_resp

    with patch("requests.post", side_effect=_capture):
        await f.outlet(body)

    assert captured.get("response") == "The actual LLM answer."


async def test_filter_outlet_no_assistant_message_returns_body_unchanged():
    """If there is no assistant message the outlet returns the body unmodified."""
    Filter = _load_filter()
    f = Filter()
    body = {"messages": [{"role": "user", "content": "question"}]}

    with patch("requests.post") as mock_post:
        result = await f.outlet(body)

    mock_post.assert_not_called()
    assert result == body


# ─── Category 5: JSON tool message extraction (fast) ─────────────────────────
# These tests catch the production bug where search_documents returns a JSON
# dict ({"chunks": [...], ...}) that the filter was naively passing as-is to
# the NLI model, filling the token budget with JSON scaffolding instead of
# readable Greek text and causing every sentence to appear ungrounded.


def _make_search_documents_result(*texts: str) -> str:
    """Return a JSON string matching the search_documents model_dump() format."""
    import json as _json

    chunks = [
        {"text": t, "source": "episode.txt", "score": 0.9, "timestamp": "00:10:00"}
        for t in texts
    ]
    return _json.dumps(
        {"chunks": chunks, "summary": "", "query": "test", "total_results": len(chunks)}
    )


def _body_with_json_tool_messages(assistant_content: str, *texts: str) -> dict:
    """Build a body where the tool message content is a JSON search_documents result."""
    messages = [{"role": "user", "content": "question"}]
    messages.append(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": "search_documents", "arguments": "{}"}}
            ],
        }
    )
    messages.append(
        {
            "role": "tool",
            "content": _make_search_documents_result(*texts),
            "tool_call_id": "call_0",
        }
    )
    messages.append({"role": "assistant", "content": assistant_content})
    return {"messages": messages}


async def test_filter_extracts_text_from_json_tool_result():
    """context_chunks sent to the server contain plain text, not JSON scaffolding.

    Regression test for the production bug where search_documents returns
    {'chunks': [{'text': '...'}, ...], ...} as a JSON string.  The filter was
    passing this blob directly to the NLI model which saw JSON keys and brackets
    instead of readable Greek prose, causing every sentence to appear ungrounded.
    """
    Filter = _load_filter()
    f = Filter()
    body = _body_with_json_tool_messages(
        "Ο Κοσκωτάς κατηγορήθηκε για υπεξαίρεση.",
        "Ο Κοσκωτάς κατηγορήθηκε για υπεξαίρεση από την Τράπεζα Κρήτης.",
        "Συνελήφθη στις ΗΠΑ και εκδόθηκε στην Ελλάδα.",
    )

    captured: dict = {}
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "grounded": True,
        "message": "",
        "unverified_sentences": [],
    }
    mock_resp.raise_for_status = MagicMock()

    def _capture(_url, json=None, **_kwargs):
        captured.update(json or {})
        return mock_resp

    with patch("requests.post", side_effect=_capture):
        await f.outlet(body)

    chunks = captured.get("context_chunks", [])
    assert len(chunks) == 1, f"Expected 1 context chunk, got {len(chunks)}"
    chunk_text = chunks[0]
    # Must contain the actual Greek text
    assert (
        "Κοσκωτάς" in chunk_text
    ), f"Greek text missing from context chunk: {chunk_text!r}"
    # Must NOT look like raw JSON (keys, braces, score numbers)
    assert "{" not in chunk_text, f"Raw JSON leaked into context chunk: {chunk_text!r}"
    assert (
        '"score"' not in chunk_text
    ), f"JSON key leaked into context chunk: {chunk_text!r}"


async def test_filter_plain_string_tool_result_passes_through_unchanged():
    """get_transcript_section returns a plain string — it should not be mangled."""
    Filter = _load_filter()
    f = Filter()
    plain_text = (
        "[00:10:00 - 00:11:00] Speaker 0:\nΟ Κοσκωτάς κατηγορήθηκε για υπεξαίρεση."
    )
    body = _body_with_tool_messages("answer", plain_text)

    captured: dict = {}
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "grounded": True,
        "message": "",
        "unverified_sentences": [],
    }
    mock_resp.raise_for_status = MagicMock()

    def _capture(_url, json=None, **_kwargs):
        captured.update(json or {})
        return mock_resp

    with patch("requests.post", side_effect=_capture):
        await f.outlet(body)

    chunks = captured.get("context_chunks", [])
    assert chunks == [plain_text], f"Plain text tool result was modified: {chunks!r}"


def test_extract_text_from_tool_result_search_documents():
    """_extract_text_from_tool_result joins chunk texts from search_documents output."""
    import json as _json

    mod_path = Path(__file__).parent.parent / "openwebui" / "historicon_filter.py"
    spec = importlib.util.spec_from_file_location("historicon_filter", mod_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = mod._extract_text_from_tool_result

    payload = _json.dumps(
        {
            "chunks": [
                {
                    "text": "First chunk.",
                    "source": "ep.txt",
                    "score": 0.9,
                    "timestamp": None,
                },
                {
                    "text": "Second chunk.",
                    "source": "ep.txt",
                    "score": 0.8,
                    "timestamp": None,
                },
            ],
            "summary": "",
            "query": "q",
            "total_results": 2,
        }
    )

    result = fn(payload)
    assert result == "First chunk. Second chunk."


def test_extract_text_from_tool_result_plain_string():
    """_extract_text_from_tool_result returns plain strings unchanged."""
    mod_path = Path(__file__).parent.parent / "openwebui" / "historicon_filter.py"
    spec = importlib.util.spec_from_file_location("historicon_filter", mod_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = mod._extract_text_from_tool_result

    plain = "[00:10:00] Speaker 0:\nΟ Κοσκωτάς κατηγορήθηκε."
    assert fn(plain) == plain

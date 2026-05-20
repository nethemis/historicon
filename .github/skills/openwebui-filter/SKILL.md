---
name: openwebui-filter
description: 'Create, update, or debug an OpenWebUI filter (inlet/outlet) for the HistoriCon project. Use when modifying guardrail logic, context extraction, disclaimer display, or exporting the filter JSON for upload. Covers the sources-field architecture, per-chunk context extraction, event emitter for real-time output, versioning, and JSON export workflow.'
argument-hint: 'Describe what you want to change or add to the filter'
---

# OpenWebUI Filter Workflow

## When to Use
- Adding or changing inlet (on-topic) or outlet (grounding) logic
- Debugging why context chunks are 0 or guardrail warnings are wrong
- Exporting the filter for upload to OpenWebUI Admin → Functions
- Writing tests for filter inlet/outlet methods

---

## Architecture: How Tool Results Reach the Filter

**Critical:** OpenWebUI with MCP Tool Servers does NOT pass `role=tool` messages to the filter outlet. The outlet receives only `[user_message, final_assistant_message]` — the entire tool call/result cycle is internal.

Tool results from the MCP server are stored in the `sources` field on the **assistant message**:

```python
body["messages"][-1]["sources"] = [
    {
        "source": {"name": "historicon_search_documents"},
        "document": ['{"chunks": [{"text": "...", "source": "...", "score": 0.9}, ...]}']
    }
]
```

### Context Extraction Order (try all three)

```python
# Format 1 — PRIMARY (MCP Tool Servers via sources field)
for m in body.get("messages", []):
    for src in m.get("sources") or []:
        for field in ("document", "content", "text"):
            val = src.get(field)
            if isinstance(val, list):
                for item in val:
                    chunks = _extract_texts_from_tool_result(str(item))
                    context_chunks.extend(chunks)
                if context_chunks:
                    break  # stop at first populated field

# Format 2 — FALLBACK (OpenAI-style role=tool messages)
for m in body.get("messages", []):
    if m.get("role") == "tool":
        ...

# Format 3 — FALLBACK (Anthropic content-block type=tool_result)
```

### Parsing MCP JSON: Return One String Per Chunk

`_extract_texts_from_tool_result` must return **individual chunk texts** (not joined), so grounding NLI checks each chunk independently:

```python
def _extract_texts_from_tool_result(raw: str) -> list[str]:
    data = json.loads(raw)
    if "chunks" in data:
        return [c["text"] for c in data["chunks"] if c.get("text")]
    # fallback: try data["content"] or [raw]
```

---

## Outlet: Real-Time Disclaimer Display

Modifying `body["messages"][-1]["content"]` alone may not show after streaming completes. Fire the event emitter as well:

```python
asyncio.ensure_future(__event_emitter__({
    "type": "message",
    "data": {"content": "\n\n---\n" + disclaimer}
}))
```

---

## Filter Files

| File | Purpose |
|------|---------|
| `openwebui/historicon_filter.py` | Source of truth — edit this |
| `openwebui/historicon_filter.json` | OpenWebUI-importable export — regenerate after edits |

### Version Bump Convention
Increment the version in the `class Filter` valve/meta block (e.g. `"version": "1.7.0"`) on every change that gets exported.

---

## Exporting the Filter JSON

After editing `historicon_filter.py`:

1. Open `openwebui/export_filter.py` and run it, OR copy the Python source manually.
2. Paste the new source into `openwebui/historicon_filter.json` under the `"content"` key (escape quotes as needed).
3. Upload in OpenWebUI: **Admin → Functions → Import** (or edit existing function inline and save).

If editing inline in OpenWebUI, copy the updated source back to `historicon_filter.py` to keep the repo in sync.

---

## Testing the Filter

```python
# conftest.py pattern — instantiate Filter directly
from openwebui.historicon_filter import Filter

@pytest.fixture
def filter_instance():
    return Filter()

async def test_outlet_extracts_context(filter_instance):
    body = {
        "messages": [
            {"role": "user", "content": "..."},
            {
                "role": "assistant",
                "content": "...",
                "sources": [{"source": {"name": "historicon_search_documents"},
                              "document": ['{"chunks": [{"text": "some fact", "source": "ep.txt", "score": 0.9}]}']}]
            }
        ]
    }
    # mock __event_emitter__ and guardrails HTTP calls
    ...
```

Mock the guardrails server calls with `respx` or `unittest.mock.patch`:
```python
with patch("openwebui.historicon_filter.httpx.AsyncClient.post") as mock_post:
    mock_post.return_value = httpx.Response(200, json={"on_topic": True, ...})
    result = await filter_instance.outlet(body, user={}, __event_emitter__=emitter)
```

---

## Common Failure Modes

| Symptom | Cause | Fix |
|---------|-------|-----|
| `context_chunks: 0` | Only checking `role=tool`, not `sources` | Add Format 1 extraction |
| Generic warning always shown | `_split_sentences` returned empty or 1 sentence | Check pysbd / sentence length filter |
| Disclaimer not visible | Only modifying `content`, no event emitter | Add `asyncio.ensure_future(__event_emitter__(...))` |
| All sentences flagged as unverified | All chunks joined → truncated to 900 chars | Return individual chunk strings, not joined |
| Markdown headers flagged | `# Title` passing NLI as hypothesis | Filter with `_MARKDOWN_HEADING_RE` before NLI |

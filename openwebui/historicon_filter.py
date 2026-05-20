"""
title: HistoriCon On-Topic Filter
author: HistoriCon
version: 1.7.0
description: >
  Blocks off-topic questions before any LLM call (inlet hook) and validates
  that the model's response is grounded in retrieved MCP tool results (outlet
  hook). Calls the HistoriCon guardrails server's /check-topic and
  /check-grounding endpoints respectively.

Upload via: OpenWebUI Admin → Functions → + (new function) → paste this file.
Set the Valve URLs if your guardrails server runs on a different host/port.
"""

import json
import sys
import requests
from pydantic import BaseModel

_DEFAULT_REFUSAL = (
    "Αυτή η ερώτηση δεν σχετίζεται με το HistoriCon. "
    "Μπορώ να βοηθήσω μόνο με ερωτήσεις για την ελληνική ή κυπριακή ιστορία "
    "και τα επεισόδια του podcast. / "
    "This question is outside the scope of HistoriCon. "
    "I can only help with questions about Greek or Cypriot history and the podcast episodes."
)

_GROUNDING_DISCLAIMER = (
    "⚠️ Μέρος αυτής της απάντησης μπορεί να μην βασίζεται αποκλειστικά στο περιεχόμενο "
    "του podcast. Παρακαλώ επαλήθευσε τις πληροφορίες με το πρωτότυπο επεισόδιο. "
    "/ ⚠️ Part of this response may not be fully grounded in the retrieved podcast content. "
    "Please verify the information against the original episode."
)


def _extract_text_from_tool_result(raw: str) -> str:
    """Extract readable text from an MCP tool result.

    search_documents returns a JSON-serialised dict::

        {"chunks": [{"text": "...", "source": "...", "score": 0.9, ...}], ...}

    Passing that JSON blob to the NLI model as a premise fills the token budget
    with scaffolding (keys, brackets, floats) instead of meaningful Greek prose,
    making every sentence appear ungrounded.

    This helper pulls out just the ``text`` fields so the NLI model sees clean
    transcript text.  Plain-string results (e.g. get_transcript_section) are
    returned unchanged.
    """
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw  # already plain text

    if isinstance(data, dict):
        # search_documents: {"chunks": [{"text": "..."}], ...}
        if "chunks" in data and isinstance(data["chunks"], list):
            texts = [
                chunk["text"]
                for chunk in data["chunks"]
                if isinstance(chunk, dict) and chunk.get("text")
            ]
            return " ".join(texts) if texts else raw
        # Any other dict: fall back to raw JSON string
        return raw

    if isinstance(data, list):
        texts = [
            item["text"] if isinstance(item, dict) and item.get("text") else str(item)
            for item in data
        ]
        return " ".join(texts)

    return str(data)


def _extract_texts_from_tool_result(raw: str) -> list[str]:
    """Like _extract_text_from_tool_result but returns one string per chunk.

    search_documents returns multiple chunks; joining them all and truncating
    means evidence from later chunks is invisible to the NLI model.  Returning
    each chunk as a separate string lets check_grounding check them
    independently and take the best (max-score) result.
    """
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return [raw] if raw.strip() else []

    if isinstance(data, dict) and "chunks" in data and isinstance(data["chunks"], list):
        return [
            chunk["text"]
            for chunk in data["chunks"]
            if isinstance(chunk, dict) and chunk.get("text")
        ]

    # Fallback: delegate to the single-string extractor
    text = _extract_text_from_tool_result(raw)
    return [text] if text.strip() else []


class Filter:
    class Valves(BaseModel):
        # Guardrails server is on the host machine; Docker resolves host.docker.internal
        check_topic_url: str = "http://host.docker.internal:8002/check-topic"
        check_grounding_url: str = "http://host.docker.internal:8002/check-grounding"
        # Seconds to wait for the guardrail endpoint before failing open
        timeout_seconds: int = 30

    def __init__(self):
        self.valves = self.Valves()

    async def inlet(
        self,
        body: dict,
        __event_emitter__=None,
        __user__: dict = {},
    ) -> dict:
        """Run before the LLM. Emits a polite refusal for off-topic queries."""
        messages = body.get("messages", [])
        # Find the most recent user message
        last_user_msg = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"),
            "",
        )

        if not last_user_msg:
            return body

        try:
            resp = requests.post(
                self.valves.check_topic_url,
                json={"query": last_user_msg},
                timeout=self.valves.timeout_seconds,
            )
            resp.raise_for_status()
            data = resp.json()

            if not data.get("on_topic", True):
                refusal = data.get("message", _DEFAULT_REFUSAL)

                if __event_emitter__:
                    # Emit as a proper assistant message — chat continues normally
                    await __event_emitter__(
                        {"type": "message", "data": {"content": refusal}}
                    )
                    await __event_emitter__(
                        {"type": "status", "data": {"done": True, "description": ""}}
                    )
                    # Return an empty messages list so the LLM is never called
                    body["messages"] = []
                    return body
                else:
                    # Fallback: event emitter unavailable
                    raise Exception(refusal)

        except requests.RequestException:
            # MCP server unreachable — fail open so legitimate users are not blocked
            pass

        return body

    async def outlet(
        self,
        body: dict,
        __event_emitter__=None,
        __user__: dict = {},
    ) -> dict:
        """Run after the LLM generates a response.

        Checks whether the response is grounded in the MCP tool results that
        were retrieved during the conversation. If no tool results exist the
        response cannot be grounded by definition. If the grounding check fails,
        a bilingual disclaimer is appended to the assistant message.

        Always fails open — any error leaves the response unmodified.
        """
        messages = body.get("messages", [])

        # Diagnostic: visible in `docker compose logs open-webui`
        roles = [m.get("role", "?") for m in messages]
        print(
            f"[HistoriCon] outlet body keys: {sorted(body.keys())}",
            file=sys.stderr,
            flush=True,
        )
        print(
            f"[HistoriCon] outlet: {len(messages)} messages, roles={roles}",
            file=sys.stderr,
            flush=True,
        )
        for i, m in enumerate(messages):
            content = m.get("content", "")
            preview = str(content)[:120].replace("\n", " ")
            extra_keys = [k for k in m.keys() if k not in ("role", "content")]
            print(
                f"[HistoriCon]   [{i}] role={m.get('role')} type={type(content).__name__} extra_keys={extra_keys} | {preview!r}",
                file=sys.stderr,
                flush=True,
            )
            if m.get("sources"):
                print(
                    f"[HistoriCon]   [{i}] sources ({len(m['sources'])} items): {str(m['sources'])[:500]}",
                    file=sys.stderr,
                    flush=True,
                )

        # Find the last assistant message with actual text content
        last_assistant_idx = None
        last_assistant_content = None
        for i in range(len(messages) - 1, -1, -1):
            m = messages[i]
            if m.get("role") == "assistant" and m.get("content"):
                last_assistant_idx = i
                last_assistant_content = m["content"]
                break

        if last_assistant_content is None or last_assistant_idx is None:
            return body

        # Collect MCP tool results as grounding context.
        # OpenWebUI stores tool/search results in the assistant message's 'sources'
        # field rather than as separate role=tool messages. We try three formats in
        # order of likelihood for this OpenWebUI + MCP setup.

        context_chunks: list[str] = []

        # ── Format 1: OpenWebUI 'sources' on the assistant message ───────────
        # sources is a list of objects, each representing one tool call result.
        # The document text may be under "document", "content", or nested inside
        # an "output" JSON string (raw MCP response).
        for m in messages:
            sources = m.get("sources")
            if not sources or not isinstance(sources, list):
                continue
            for src in sources:
                if not isinstance(src, dict):
                    continue
                for field in ("document", "content", "text"):
                    val = src.get(field)
                    if val:
                        if isinstance(val, list):
                            for item in val:
                                context_chunks.extend(
                                    _extract_texts_from_tool_result(str(item))
                                )
                        else:
                            context_chunks.extend(
                                _extract_texts_from_tool_result(str(val))
                            )
                        break  # only use the first populated field per source
                # Raw MCP JSON output field (fallback)
                if not context_chunks:
                    raw_output = src.get("output") or src.get("result") or ""
                    if raw_output:
                        context_chunks.extend(
                            _extract_texts_from_tool_result(str(raw_output))
                        )

        # ── Format 2: OpenAI role=tool messages ───────────────────────────────
        if not context_chunks:
            context_chunks = [
                _extract_text_from_tool_result(str(m["content"]))
                for m in messages
                if m.get("role") == "tool" and m.get("content")
            ]

        # ── Format 3: Anthropic content-block tool results ────────────────────
        if not context_chunks:
            for m in messages:
                content = m.get("content")
                if not isinstance(content, list):
                    continue
                for item in content:
                    if not isinstance(item, dict) or item.get("type") != "tool_result":
                        continue
                    raw = item.get("content", "")
                    if isinstance(raw, list):
                        raw = " ".join(
                            r.get("text", "") for r in raw if isinstance(r, dict)
                        )
                    if raw:
                        context_chunks.append(_extract_text_from_tool_result(str(raw)))

        print(
            f"[HistoriCon] context_chunks found: {len(context_chunks)}",
            file=sys.stderr,
            flush=True,
        )

        # No tool results → response cannot be grounded in podcast data → always flag
        if not context_chunks:
            body["messages"][last_assistant_idx]["content"] = (
                last_assistant_content + "\n\n---\n" + _GROUNDING_DISCLAIMER
            )
            return body

        # Ask the guardrails server whether the response is grounded
        try:
            resp = requests.post(
                self.valves.check_grounding_url,
                json={
                    "response": last_assistant_content,
                    "context_chunks": context_chunks,
                },
                timeout=self.valves.timeout_seconds,
            )
            resp.raise_for_status()
            data = resp.json()

            if not data.get("grounded", True):
                unverified = data.get("unverified_sentences", [])
                if unverified:
                    bullets = "\n".join(f"- {s}" for s in unverified)
                    disclaimer = (
                        "⚠️ Τα παρακάτω σημεία δεν επαληθεύτηκαν στο περιεχόμενο του podcast:\n"
                        + bullets
                        + "\n/ ⚠️ The following claims could not be verified in the retrieved podcast content:\n"
                        + bullets
                    )
                else:
                    disclaimer = data.get("message", _GROUNDING_DISCLAIMER)

                print(
                    f"[HistoriCon] appending disclaimer ({len(unverified)} unverified sentences)",
                    file=sys.stderr,
                    flush=True,
                )
                # Persist in stored message
                body["messages"][last_assistant_idx]["content"] = (
                    last_assistant_content + "\n\n---\n" + disclaimer
                )
                # Also emit for real-time display in case streaming already finished
                if __event_emitter__:
                    import asyncio

                    asyncio.ensure_future(
                        __event_emitter__(
                            {
                                "type": "message",
                                "data": {"content": "\n\n---\n" + disclaimer},
                            }
                        )
                    )

        except requests.RequestException:
            # Guardrails server unreachable — fail open, leave response unchanged
            pass

        return body

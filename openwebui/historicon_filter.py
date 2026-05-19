"""
title: HistoriCon On-Topic Filter
author: HistoriCon
version: 1.2.0
description: >
  Blocks off-topic questions before any LLM call is made.
  Calls the HistoriCon guardrails server's /check-topic endpoint to classify
  the user's last message. If off-topic, emits a polite assistant-style
  reply via the event emitter — the user can keep chatting normally.

Upload via: OpenWebUI Admin → Functions → + (new function) → paste this file.
Set the check_topic_url Valve if your guardrails server runs on a different host/port.
"""

import requests
from pydantic import BaseModel

_DEFAULT_REFUSAL = (
    "Αυτή η ερώτηση δεν σχετίζεται με το HistoriCon. "
    "Μπορώ να βοηθήσω μόνο με ερωτήσεις για την ελληνική ή κυπριακή ιστορία "
    "και τα επεισόδια του podcast. / "
    "This question is outside the scope of HistoriCon. "
    "I can only help with questions about Greek or Cypriot history and the podcast episodes."
)


class Filter:
    class Valves(BaseModel):
        # Guardrails server is on the host machine; Docker resolves host.docker.internal
        check_topic_url: str = "http://host.docker.internal:8002/check-topic"
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

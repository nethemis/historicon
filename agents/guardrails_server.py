"""Standalone guardrails HTTP server for HistoriCon.

Exposes POST /check-topic used by the OpenWebUI filter to classify user
messages before any LLM call is made. Runs independently from the MCP server
so the retrieval layer stays free of guardrail logic.

Run:
    uv run python agents/guardrails_server.py
    # Listens on http://0.0.0.0:8002/check-topic
"""

import asyncio

import logfire
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from agents import logfire_setup  # noqa: F401 — import-time Logfire bootstrap
from agents.guardrails import check_on_topic


async def _check_topic_async(query: str) -> tuple[bool, str]:
    """Run the blocking BART classifier in a thread pool."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, check_on_topic, query)


async def check_topic(request: Request) -> JSONResponse:
    """POST /check-topic

    Request body: {"query": "<user message>"}
    Response:     {"on_topic": bool, "message": str}

    Returns on_topic=true on any error so the filter fails open.
    """
    try:
        body = await request.json()
        query: str = body.get("query", "")
    except Exception:
        return JSONResponse({"on_topic": True, "message": ""})

    if not query:
        return JSONResponse({"on_topic": True, "message": ""})

    is_ok, msg = await _check_topic_async(query)
    logfire.info("check_topic", query_preview=query[:60], on_topic=is_ok)
    return JSONResponse({"on_topic": is_ok, "message": msg})


app = Starlette(routes=[Route("/check-topic", check_topic, methods=["POST"])])

if __name__ == "__main__":
    logfire.info("Starting HistoriCon guardrails server on http://0.0.0.0:8002")
    uvicorn.run(app, host="0.0.0.0", port=8002)

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
from agents.guardrails import check_grounding_detailed, check_on_topic


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


async def _check_grounding_async(
    response: str, context_chunks: list[str]
) -> tuple[bool, str, list[str]]:
    """Run the blocking sentence-level NLI grounding check in a thread pool."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, check_grounding_detailed, response, context_chunks
    )


async def check_grounding_endpoint(request: Request) -> JSONResponse:
    """POST /check-grounding

    Request body: {"response": "<model output>", "context_chunks": ["<chunk1>", ...]}
    Response:     {"grounded": bool, "message": str}

    Returns grounded=true on any error so the filter fails open.
    """
    try:
        body = await request.json()
        response: str = body.get("response", "")
        context_chunks: list[str] = body.get("context_chunks", [])
    except Exception:
        return JSONResponse(
            {"grounded": True, "message": "", "unverified_sentences": []}
        )

    if not response or not context_chunks:
        return JSONResponse(
            {"grounded": True, "message": "", "unverified_sentences": []}
        )

    is_grounded, msg, unverified = await _check_grounding_async(
        response, context_chunks
    )
    logfire.info(
        "check_grounding",
        response_preview=response[:60],
        num_chunks=len(context_chunks),
        grounded=is_grounded,
        unverified_count=len(unverified),
    )
    return JSONResponse(
        {"grounded": is_grounded, "message": msg, "unverified_sentences": unverified}
    )


app = Starlette(
    routes=[
        Route("/check-topic", check_topic, methods=["POST"]),
        Route("/check-grounding", check_grounding_endpoint, methods=["POST"]),
    ]
)

if __name__ == "__main__":
    logfire.info("Starting HistoriCon guardrails server on http://0.0.0.0:8002")
    uvicorn.run(app, host="0.0.0.0", port=8002)

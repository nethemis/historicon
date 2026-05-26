"""Standalone guardrails HTTP server for HistoriCon.

Exposes POST /check-topic used by the OpenWebUI filter to classify user
messages before any LLM call is made. Runs independently from the MCP server
so the retrieval layer stays free of guardrail logic.

Run:
    uv run python agents/guardrails_server.py
    # Listens on http://0.0.0.0:8002/check-topic

The NLI classifier (~280 MB) is preloaded on startup to avoid cold-start delays.
"""

import asyncio

import uvicorn
from opentelemetry import trace
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from agents import otel_setup  # noqa: F401 — import-time OpenTelemetry bootstrap
from agents.guardrails import check_grounding_detailed, check_on_topic, get_classifier

# Get tracer for this module
_tracer = otel_setup.get_tracer("historicon.guardrails_server")


def preload_classifier() -> None:
    """Preload the NLI classifier on server startup.

    This avoids cold-start delays on the first user request. The classifier
    (~280 MB) is loaded once and cached via lru_cache.
    """
    with _tracer.start_as_current_span("preload_classifier") as span:
        try:
            print(
                "Preloading NLI classifier (MoritzLaurer/mDeBERTa-v3-base-mnli-xnli)..."
            )
            classifier = get_classifier()
            span.set_attribute("status", "success")
            print("NLI classifier loaded successfully")
        except Exception as exc:
            span.record_exception(exc)
            span.set_attribute("status", "failed")
            print(f"Failed to preload NLI classifier: {exc}")
            print("Guardrails will load the classifier on first request instead")


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
    with _tracer.start_as_current_span("check_topic") as span:
        try:
            body = await request.json()
            query: str = body.get("query", "")
        except Exception:
            span.set_attribute("error", "parse_failed")
            return JSONResponse({"on_topic": True, "message": ""})

        if not query:
            span.set_attribute("error", "empty_query")
            return JSONResponse({"on_topic": True, "message": ""})

        span.set_attribute("query_preview", query[:60])
        is_ok, msg = await _check_topic_async(query)
        span.set_attribute("on_topic", is_ok)
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
    with _tracer.start_as_current_span("check_grounding") as span:
        try:
            body = await request.json()
            response: str = body.get("response", "")
            context_chunks: list[str] = body.get("context_chunks", [])
        except Exception:
            span.set_attribute("error", "parse_failed")
            return JSONResponse(
                {"grounded": True, "message": "", "unverified_sentences": []}
            )

        if not response or not context_chunks:
            span.set_attribute("error", "empty_input")
            return JSONResponse(
                {"grounded": True, "message": "", "unverified_sentences": []}
            )

        span.set_attribute("response_preview", response[:60])
        span.set_attribute("num_chunks", len(context_chunks))
        is_grounded, msg, unverified = await _check_grounding_async(
            response, context_chunks
        )
        span.set_attribute("grounded", is_grounded)
        span.set_attribute("unverified_count", len(unverified))
        return JSONResponse(
            {
                "grounded": is_grounded,
                "message": msg,
                "unverified_sentences": unverified,
            }
        )


app = Starlette(
    routes=[
        Route("/check-topic", check_topic, methods=["POST"]),
        Route("/check-grounding", check_grounding_endpoint, methods=["POST"]),
    ]
)

if __name__ == "__main__":
    print("Starting HistoriCon guardrails server on http://0.0.0.0:8002")
    preload_classifier()
    uvicorn.run(app, host="0.0.0.0", port=8002)

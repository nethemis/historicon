"""Pydantic Evals for the HistoriCon RAG agent system.

Run individual eval files directly:
    uv run python evals/eval_guardrails.py     # fast, no LLM
    uv run python evals/eval_failure_modes.py  # fast, no LLM
    uv run python evals/eval_retrieval.py      # requires populated chroma_db
    uv run python evals/eval_orchestrator.py   # requires ANTHROPIC_API_KEY

Or via pytest:
    uv run pytest evals/ -m "not slow and not integration"  # fast subset
    uv run pytest evals/ -m "slow"                          # LLM-backed evals
    uv run pytest evals/ -m "integration"                   # requires chroma_db
"""

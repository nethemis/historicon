# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

HistoriCon is a RAG system for a Greek-Cypriot history podcast. It downloads episodes from Patreon, transcribes Greek audio with Deepgram (speaker diarization), indexes transcripts in ChromaDB, and serves queries through a Pydantic-AI multi-agent web app.

## Commands

Dependencies are managed with `uv`. Python 3.12 (pinned `>=3.12,<3.13` in pyproject.toml — note this contradicts the README's "3.13+" claim; trust pyproject.toml).

```bash
uv pip install -e ".[dev]"          # install incl. dev deps

uv run python agents/web_orchestrator.py    # start web app on :8001
uv run ./run_pipeline.py                     # full setup pipeline

# Individual pipeline stages (in order):
uv run python scripts/download_patreon.py
uv run python scripts/transcribe_deepgram.py
uv run python scripts/preprocess_transcripts.py
uv run python scripts/create_embeddings.py   # must run before agent works

uv run pytest                                # all tests
uv run pytest tests/test_retrieval.py        # single file
uv run pytest -m "not slow"                  # skip slow tests
uv run pytest --cov=agents --cov-report=html
```

Required env vars: `ANTHROPIC_API_KEY` and/or `OLLAMA_API_KEY` (the agent picks whichever are set; a local `qwen3:8b` Ollama model is always offered). `DEEPGRAM_API_KEY` is needed only for the transcription pipeline. `LOGFIRE_TOKEN` is optional.

## Architecture

**Data flow:** Patreon RSS → `inputs/` (audio, gitignored) → `transcripts/` (raw Deepgram output with timestamps) → `transcripts_processed/` (cleaned, speaker-merged, committed) → `chroma_db/` (vector index, gitignored) → agents.

**Agents** (`agents/`):
- `web_orchestrator.py` — Pydantic-AI `Agent` exposed via `to_web()` (FastAPI/uvicorn on :8001). Registers tools: `search_documents` (calls `search_transcripts` directly), `get_transcript_section` (time-range slice of a raw transcript), `list_podcast_info_sections` + `get_podcast_info_section` (reads `podcast_info.json` metadata). Available models are built dynamically from env vars.
- `retrieval.py` — Plain `search_transcripts(query, max_results)` function backed by `ChromaRetriever`: ChromaDB returns `retrieval_top_k` candidates (sentence-transformers multilingual embeddings), then a CrossEncoder (`reranking_model`) reranks them; final scores are sigmoid-squashed into `[0, 1]` for `RetrievalChunk.score`. No LLM, no tool-calling.
- `models.py` — Pydantic types shared between agents (`RetrievalChunk`, `RetrievalResponse`).
- `_utils.py` — `load_instructions()` shared loader for `instructions/{agent_name}.txt` prompts.
- `logfire_setup.py` — `configure_logfire()` is idempotent and auto-runs on import; toggle off in tests via `LOGFIRE_AUTO_CONFIGURE=false`.

**Config:** All knobs live in `config.json` (project root). `agents/config.py:Config` is the Pydantic loader; import it as `from agents.config import config, get_device`. Device selection auto-picks `mps`/`cuda`/`cpu`. Diarized-speaker renames live under `speaker_map` in `config.json`.

**Context-size guard:** `web_orchestrator.check_context_size()` reads `ctx.usage` and refuses to add more transcript content past `config.max_context_tokens` (default 190k, safety margin for Claude's 200k). `get_full_transcript` is intentionally commented out — use iterative `get_transcript_section` calls instead.

**Chunking** (in `scripts/create_embeddings.py`): speaker-aware semantic chunking — never splits a single speaker's continuous speech, groups consecutive segments by cosine similarity to running chunk-average embedding. Tuned by `similarity_threshold`, `chunk_min_size`, `chunk_max_size` in `config.json`.

## Conventions to preserve

- **Greek UTF-8 everywhere.** Always `encoding="utf-8"` on file I/O. Transcript filenames contain Greek characters — don't sanitize them away.
- **`retries=5`** on every Pydantic-AI `Agent` (project convention).
- Pipeline scripts use parallel workers (downloads: 10, Deepgram: 5 — API limit). They skip already-existing files to save API credits; preserve this.
- Tests use `asyncio_mode = auto` (pytest.ini). Markers: `slow`, `integration`. Run with `LOGFIRE_AUTO_CONFIGURE=false` to skip the Logfire bootstrap.
- Agent modules use absolute imports (`from agents.config import ...`). The project must be installed editable (`uv pip install -e ".[dev]"`) for scripts and tests to resolve `agents.*` from any cwd.

## Pipeline orchestration

`run_pipeline.py` runs stages serially: **download → transcribe → preprocess → embeddings**. Each stage is a `Stage` dataclass with optional `required_env`. Flags: `--from-stage <name>` to resume, `--only <name>` to run a single stage. Preflight fails fast with a clear message if any required env var is missing.

## Gotchas

- `search_transcripts` (the retrieval path) catches all exceptions and returns an empty `RetrievalResponse` with the error in `summary`. Callers should not rely on exceptions to signal retrieval failure — check `total_results` instead.
- CrossEncoder logits are unbounded; `RetrievalChunk.score` is sigmoid-normalized into `[0, 1]`. Don't compare scores across different rerankers or models — the sigmoid mapping isn't calibrated.
- `chroma_db/` and `inputs/` are gitignored; `transcripts_processed/` is committed (it's the source of truth for embeddings).
- Several scripts (`scripts/vad_transcribe.py`, `scripts/llm_correct.py`, `scripts/model_comparison.py`) and dirs (`greek_whisper/`, `model_comparisons/`) are experimental Whisper/LoRA/correction work, untracked and not wired into `run_pipeline.py`.

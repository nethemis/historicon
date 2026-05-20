# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

HistoriCon is a RAG system for a Greek-Cypriot history podcast. It downloads episodes from Patreon, transcribes Greek audio with Deepgram (speaker diarization), indexes transcripts in ChromaDB, and exposes retrieval tools via a **FastMCP server**. Users interact through **OpenWebUI** (Docker) using any Ollama or OpenAI-compatible model for synthesis. There is no LLM inside the MCP server itself.

## Commands

Dependencies are managed with `uv`. Python 3.12 (pinned `>=3.12,<3.13` in pyproject.toml — note this contradicts the README's "3.13+" claim; trust pyproject.toml).

```bash
uv pip install -e ".[dev]"          # install incl. dev deps

# Three terminals required to run the system:
uv run python agents/mcp_server.py          # Terminal 1: MCP server on :8001
uv run python agents/guardrails_server.py   # Terminal 2: guardrails server on :8002
docker compose up -d                         # Terminal 3: OpenWebUI on :3000
uv run ./run_pipeline.py                     # full setup pipeline

# Individual pipeline stages (in order):
uv run python scripts/download_patreon.py
uv run python scripts/transcribe_deepgram.py
uv run python scripts/preprocess_transcripts.py
uv run python scripts/create_embeddings.py   # must run before MCP server works

uv run pytest                                # all tests
uv run pytest tests/test_mcp_server.py       # single file
uv run pytest -m "not slow"                  # skip slow tests
uv run pytest --cov=agents --cov-report=html
```

Required env vars: `DEEPGRAM_API_KEY` (transcription pipeline only). The MCP server requires no API keys. `LOGFIRE_TOKEN` is optional.

## Architecture

**Data flow:** Patreon RSS → `inputs/` (audio, gitignored) → `transcripts/` (raw Deepgram output with timestamps) → `transcripts_processed/` (cleaned, speaker-merged, committed) → `chroma_db/` (vector index, gitignored) → agents.

**Agents** (`agents/`):
- `mcp_server.py` — FastMCP server on `:8001` (streamable-http). Exposes 4 tools: `search_documents` (ChromaDB semantic search), `get_transcript_section` (time-range slice), `list_podcast_info_sections`, `get_podcast_info_section`. No LLM inside, no guardrails — returns raw retrieval data for OpenWebUI's chosen model to synthesise.
- `guardrails_server.py` — Standalone Starlette/uvicorn HTTP server on `:8002`. Exposes `POST /check-topic` (input guard: blocks off-topic queries before the LLM call) and `POST /check-grounding` (output guard: detects hallucinations in LLM responses). Both endpoints wrap functions from `guardrails.py`, run in a thread pool executor, and fail open on any error.
- `guardrails.py` — NLI-based input and output guardrails. `get_classifier()` (lru_cache singleton, loads `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` ~280 MB lazily). `check_on_topic(query) -> (bool, str)`: zero-shot classification with short labels (`"Greek or Cypriot history"` / `"an unrelated topic"`) and `hypothesis_template="This question is about {}."`. `check_grounding(response, context_chunks) -> (bool, str)`: direct NLI — each context chunk is checked independently as a premise against the sentence hypothesis; the **maximum entailment score** across all chunks is used (short-circuits on first grounded chunk). Flags when max score < `_GROUNDING_ENTAILMENT_MIN` (0.45). `check_grounding_detailed` wraps per-sentence with `_split_sentences` (pysbd-based, auto-detects Greek vs English via `_detect_language`, filters markdown headers/HRs). Fails open on errors.
- `retrieval.py` — Plain `search_transcripts(query, max_results)` function backed by `ChromaRetriever`: ChromaDB returns `retrieval_top_k` candidates (sentence-transformers multilingual embeddings), then a CrossEncoder (`reranking_model`) reranks them; final scores are sigmoid-squashed into `[0, 1]` for `RetrievalChunk.score`. No LLM, no tool-calling.
- `models.py` — Pydantic types shared between agents (`RetrievalChunk`, `RetrievalResponse`).
- `_utils.py` — `load_instructions()` shared loader for `instructions/{agent_name}.txt` prompts.
- `logfire_setup.py` — `configure_logfire()` is idempotent and auto-runs on import; toggle off in tests via `LOGFIRE_AUTO_CONFIGURE=false`.

**Deleted (no longer in repo):** `agents/web_orchestrator.py`, `agents/server.py`, `frontend/` (entire Next.js AG-UI directory), `tests/test_web_orchestrator.py`, `tests/test_server.py`.

**Config:** All knobs live in `config.json` (project root). `agents/config.py:Config` is the Pydantic loader; import it as `from agents.config import config, get_device`. Device selection auto-picks `mps`/`cuda`/`cpu`. Diarized-speaker renames live under `speaker_map` in `config.json`.

**Chunking** (in `scripts/create_embeddings.py`): speaker-aware semantic chunking — never splits a single speaker's continuous speech, groups consecutive segments by cosine similarity to running chunk-average embedding. Tuned by `similarity_threshold`, `chunk_min_size`, `chunk_max_size` in `config.json`.

## Deployment

`docker-compose.yml` runs OpenWebUI on port 3000. Key env vars:
- `OLLAMA_BASE_URL=http://host.docker.internal:11434` — points to Ollama on the host machine. **Do not set this to empty string** — that bypasses the `/ollama` Dockerfile default which auto-resolves to `host.docker.internal`.
- `OPENAI_API_BASE_URLS=https://api.anthropic.com/v1` + `OPENAI_API_KEYS=${ANTHROPIC_API_KEY}` — exposes Claude models in OpenWebUI via Anthropic's OpenAI-compatible endpoint. `ANTHROPIC_API_KEY` must be present in the shell environment before running `docker compose up`.
- `TOOL_SERVER_CONNECTIONS` — JSON array pre-seeding the HistoriCon MCP server. Seeds the OpenWebUI database **on first launch only**. If the DB already exists without it, add manually in Admin → Settings → Tool Servers.
- `WEBUI_SECRET_KEY` — change to a random value (`openssl rand -hex 32`) before sharing access.
- `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET` — **not hardcoded**. Must be present in the shell environment before running `docker compose up`. Load them via the `secrets` function in `~/.zshrc` (reads from 1Password). Both are stored in 1Password under `op://Employee/GOOGLE_CLIENT_ID/credential` and `op://Employee/GOOGLE_CLIENT_SECRET/credential`. If these vars are missing when Docker starts, Google OAuth will silently fail.

System prompt template lives at `openwebui/system_prompt.md`.

## Conventions to preserve

- **Greek UTF-8 everywhere.** Always `encoding="utf-8"` on file I/O. Transcript filenames contain Greek characters — don't sanitize them away.
- **No LLM in MCP server.** The server returns raw retrieval data only. LLM synthesis is OpenWebUI's responsibility.
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
- `check_on_topic` and `check_grounding` both load `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` (~280 MB) lazily via `get_classifier()` (`@lru_cache(maxsize=1)`) on the first call to the guardrails server. Tests must mock `agents.guardrails.get_classifier` to avoid triggering this download. Call `get_classifier.cache_clear()` in tests that patch the underlying loader.
- `_PODCAST_INFO_PATH` in `mcp_server.py` is a module-level `Path` — patch it directly in tests: `patch("agents.mcp_server._PODCAST_INFO_PATH", tmp_path / "info.json")`.
- FastMCP tool listing API: use `asyncio.run(mcp.list_tools())` returning `FunctionTool` objects with `.name`. `mcp._tool_manager.tools` does not exist.
- `TOOL_SERVER_CONNECTIONS` in OpenWebUI uses `PersistentConfig` — the env var value is saved to the database on first launch and subsequent env changes are ignored unless `RESET_CONFIG_ON_START=true` or the Docker volume is deleted.
- `guardrails-ai>=0.6.0` is in `pyproject.toml` but not yet used — it was quarantined on PyPI (May 2026). The on-topic classifier uses `transformers` directly for now.
- **OpenWebUI filter + tool results:** OpenWebUI does NOT pass `role=tool` messages to filter outlets. Tool call results are stored in the `sources` field of the assistant message (`body["messages"][-1]["sources"]`), as a list of objects with `{"source": {"name": "tool_name"}, "document": ["<raw JSON string>"]}`. The filter (`openwebui/historicon_filter.py`) uses `_extract_texts_from_tool_result` to parse each chunk from the raw MCP JSON and passes them individually to the guardrails server.
- **Sentence splitting:** `_split_sentences` in `guardrails.py` uses `pysbd` (not regex). Language is auto-detected via `_detect_language` (Greek Unicode char ratio). Markdown ATX headers (`# Title`) and horizontal rules (`---`) are filtered out before NLI checking — they are formatting, not verifiable claims.

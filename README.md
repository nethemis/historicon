# HistoriCon - Greek Podcast RAG

RAG (Retrieval Augmented Generation) system for the **HistoriCon** Greek-Cypriot history/storytelling podcast. Downloads audio from Patreon, transcribes to Greek with speaker diarization, and enables AI-powered question answering via a **FastMCP server** with **OpenWebUI** as the chat interface.

## 🎙️ Project Overview

HistoriCon is a RAG system that:
1. **Downloads** podcast episodes from Patreon RSS feeds
2. **Transcribes** Greek audio with speaker diarization using Deepgram
3. **Indexes** transcripts for semantic search (ChromaDB + sentence-transformers)
4. **Exposes** retrieval tools over the MCP protocol (FastMCP, streamable-http)
5. **Serves** a multi-user chat UI via OpenWebUI (Docker), using any Ollama or OpenAI-compatible model for synthesis

## 🏗️ Architecture

### Data Pipeline
```
Patreon RSS → Audio Files → Deepgram → Greek Transcripts → Vector DB → MCP Tools → OpenWebUI
```

### System Components
- **MCP Server** (`agents/mcp_server.py`) — FastMCP server on `:8001`, exposes 4 retrieval tools
- **OpenWebUI** — Docker-based chat UI on `:3000`, connects to MCP server and any Ollama model
- **Retrieval** (`agents/retrieval.py`) — ChromaDB semantic search + CrossEncoder reranking
- **Guardrails** (`agents/guardrails.py`) — On-topic BART classifier (blocks non-history queries)

### MCP Tools
| Tool | Description |
|------|-------------|
| `search_documents` | Semantic search across all podcast transcripts (with on-topic filtering) |
| `get_transcript_section` | Retrieve a time-range slice from a specific episode |
| `list_podcast_info_sections` | List available metadata sections from `podcast_info.json` |
| `get_podcast_info_section` | Get a specific metadata section |

## 🚀 Quick Start

### Prerequisites
- **Docker** — for OpenWebUI
- **Ollama** — for local LLM inference ([ollama.com/download](https://ollama.com/download))
- Pull at least one model: `ollama pull qwen3:8b`

### Step 1: Install Dependencies

**1. Install `uv` (fast Python package manager):**
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with pip
pip install uv
```

**2. Create virtual environment and install packages:**
```bash
# Create virtual environment
uv venv

# Activate the virtual environment
source .venv/bin/activate  # macOS/Linux
# Or on Windows: .venv\Scripts\activate

# Install all dependencies
uv pip install -e ".[dev]"
```

---

### Step 3: Create Embeddings Database

To create the vector embeddings from the processed transcripts:

```bash
# Create embeddings from processed transcripts
uv run python scripts/create_embeddings.py
```

This will:
- Read all transcripts from `transcripts_processed/`
- Generate multilingual embeddings (supports Greek)
- Create semantic chunks using speaker-aware chunking
- Store everything in `chroma_db/` for fast retrieval

**Note:** This step takes a few minutes depending on the hardware, but you only need to run it once (unless you add new episodes).

---

### Step 4: Start the MCP Server

```bash
# Start the FastMCP server (runs on http://localhost:8001/mcp)
uv run python agents/mcp_server.py
```

The server exposes 4 retrieval tools over MCP streamable-http at `http://localhost:8001/mcp`.

---

### Step 5: Start OpenWebUI

```bash
# Start OpenWebUI in Docker (runs on http://localhost:3000)
docker compose up -d
```

OpenWebUI is pre-configured via `docker-compose.yml` to:
- Connect to Ollama on the host machine (`http://host.docker.internal:11434`)
- Connect to the HistoriCon MCP server (`http://host.docker.internal:8001/mcp`)

**First launch only:** The first account created becomes the admin.

Set the system prompt in OpenWebUI (Admin → Settings → System Prompt) using the template in `openwebui/system_prompt.md`.

> **Note:** `TOOL_SERVER_CONNECTIONS` only seeds the database on first launch. If OpenWebUI was previously started without it, go to Admin → Settings → Tool Servers and add the MCP server manually with URL `http://host.docker.internal:8001/mcp`.

> **Security:** Change `WEBUI_SECRET_KEY` in `docker-compose.yml` to a random value before sharing access: `openssl rand -hex 32`

---

### Step 6: (Optional) Set Up Podcast Pipeline

If you want to download and transcribe new podcast episodes:

**1. Set up Deepgram API key (required for transcription):**
```bash
export DEEPGRAM_API_KEY="your-deepgram-api-key"
```
Get your key at [https://console.deepgram.com](https://console.deepgram.com)

**2. Run the full pipeline:**
```bash
# Downloads, transcribes, preprocesses, and creates embeddings
uv run ./run_pipeline.py

# Or run individual scripts:
uv run python scripts/download_patreon.py
uv run python scripts/transcribe_deepgram.py
uv run python scripts/preprocess_transcripts.py
uv run python scripts/create_embeddings.py
```

## 📋 Requirements

- **Python 3.12** (pinned `>=3.12,<3.13` in `pyproject.toml`)
- **Package manager:** `uv` (fast Python installer)

### Environment Variables Reference

**For Podcast Pipeline:**
- `DEEPGRAM_API_KEY` - For Greek transcription (see Quick Start Step 6)

**Optional (Observability):**
- `LOGFIRE_TOKEN` - For tracking agent performance
- `LOGFIRE_SERVICE_NAME` - Service name in logs (default: "historicon-rag-agent")
- `ENVIRONMENT` - Environment name (default: "development")

**No LLM API key required** — the MCP server does not call any LLM. The model used for synthesis lives in OpenWebUI (Ollama or any OpenAI-compatible endpoint).

## ⚙️ Configuration

The system is configured via `config.json` in the project root. All settings are validated using Pydantic.

### config.json Settings

```json
{
  "embedding_model": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
  "reranking_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
  "chroma_db_dir": "./chroma_db",
  "transcripts_dir": "./transcripts_processed",
  "documents_collection": "podcast_transcripts",
  "retrieval_top_k": 20,
  "similarity_threshold": 0.85,
  "chunk_min_size": 1000,
  "chunk_max_size": 10000,
  "max_transcript_tokens": 10000,
  "max_context_tokens": 190000
}
```

#### Model Settings

- **`embedding_model`** (string)
  - HuggingFace model for generating text embeddings
  - Current: Multilingual model that supports Greek
  - Used for semantic search and chunking

- **`reranking_model`** (string)
  - CrossEncoder model for reranking search results
  - Improves relevance by re-scoring retrieved chunks
  - Applied after initial semantic search

#### Storage Settings

- **`chroma_db_dir`** (string, default: `"./chroma_db"`)
  - Directory where ChromaDB vector database is stored
  - Contains indexed embeddings and metadata

- **`transcripts_dir`** (string, default: `"./transcripts_processed"`)
  - Directory containing preprocessed transcript files
  - Used by both embedding creation and web agent tools

- **`documents_collection`** (string, default: `"podcast_transcripts"`)
  - Name of the ChromaDB collection
  - All transcript chunks stored under this collection name

#### Retrieval Settings

- **`retrieval_top_k`** (integer, default: `20`)
  - Number of candidate chunks to retrieve from vector DB
  - Higher = more context but slower
  - Reranking narrows these down to most relevant

#### Chunking Settings

The system uses **speaker-aware semantic chunking** that:
- Never splits a single speaker's continuous speech
- Groups consecutive speaker segments if semantically similar
- Compares each segment to the chunk's average embedding

- **`similarity_threshold`** (float, 0.0-1.0, default: `0.85`)
  - Cosine similarity threshold for grouping speaker segments
  - **Lower values (0.5-0.7)**: Strict grouping, only very similar topics → smaller chunks
  - **Higher values (0.8-0.95)**: Loose grouping, related topics together → larger chunks
  - Current: 0.85 for natural conversational flow

- **`chunk_min_size`** (integer, default: `1000`)
  - Minimum chunk size in characters
  - Enforced first before applying similarity threshold

- **`chunk_max_size`** (integer, default: `10000`)
  - Maximum chunk size in characters (soft limit)
  - Speakers are **never split** even if they exceed this size

#### Context Management Settings

- **`max_transcript_tokens`** (integer, default: `10000`)
  - Maximum tokens per page when using `get_full_transcript` tool
  - Controls pagination size to avoid exceeding API limits
  - ~40,000 characters for Greek/English mixed text

- **`max_context_tokens`** (integer, default: `190000`)
  - Safety threshold for total conversation context
  - Triggers warnings when approaching Claude's 200k limit
  - Prevents "prompt is too long" errors
  - Agent refuses to add more content when exceeded

### Adjusting Configuration

Edit `config.json` and restart the agent:

```bash
# Example: Stricter grouping for more granular chunks
{
  "similarity_threshold": 0.70,  # Only group very similar topics
  "chunk_min_size": 500,          # Allow smaller chunks
  "chunk_max_size": 5000
}

# Example: Looser grouping for larger conversational chunks
{
  "similarity_threshold": 0.90,  # Group loosely related topics
  "chunk_min_size": 2000,
  "chunk_max_size": 15000
}

# Example: Increase context safety margin
{
  "max_context_tokens": 180000  # More conservative limit
}
```

Changes take effect immediately on next agent run (no reindexing needed for context settings).

## 📁 Project Structure

```
historicon/
├── agents/                          # RAG system
│   ├── __init__.py                  # Package exports
│   ├── config.py                    # Pydantic config loader (config.json)
│   ├── models.py                    # Shared Pydantic types
│   ├── guardrails.py                # On-topic BART classifier
│   ├── logfire_setup.py             # Observability config
│   ├── mcp_server.py                # FastMCP server (4 tools, port 8001)
│   ├── retrieval.py                 # Semantic search (ChromaDB + CrossEncoder)
│   └── _utils.py                    # Shared helpers
├── openwebui/                       # OpenWebUI setup docs
│   ├── system_prompt.md             # HistoriCon assistant system prompt
│   └── README.md                    # OpenWebUI setup guide
├── docker-compose.yml               # OpenWebUI container config
├── instructions/                    # (Legacy) agent system prompts
│   └── retrieval.txt
├── tests/                           # Pytest suite (51 tests)
│   ├── test_guardrails.py
│   ├── test_retrieval.py
│   ├── test_mcp_server.py
│   ├── test_create_embeddings.py
│   └── test_preprocess_transcripts.py
├── scripts/                         # Setup pipeline scripts
│   ├── download_patreon.py          # Patreon RSS downloader
│   ├── transcribe_deepgram.py       # Deepgram transcription
│   ├── preprocess_transcripts.py    # Clean transcripts
│   └── create_embeddings.py         # Index for RAG
├── inputs/                          # Downloaded audio (gitignored)
├── transcripts/                     # Raw transcripts with timestamps
├── transcripts_processed/           # Cleaned transcripts (committed)
├── chroma_db/                       # Vector database (gitignored)
├── run_pipeline.py                  # Run all setup scripts
├── trim_audio.py                    # Audio trimming utility
├── pyproject.toml                   # Dependencies
└── README.md                        # This file
```

## 🔧 Development

### Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=agents --cov-report=html

# Run specific test file
uv run pytest tests/test_retrieval.py
```

### Project Conventions

✅ **Always:**
- Use Pydantic `BaseModel` for all inputs/outputs
- Configure agents with `retries=5`
- Use parallel processing (ThreadPoolExecutor) for I/O
- Support Greek UTF-8 encoding throughout
- Follow TDD methodology (write tests first)
- Use eval-driven development for AI agents

❌ **Never:**
- Remove filename sanitization (Greek filenames need special handling)
- Skip existing-file checks (wastes API credits)
- Use ASCII encoding for Greek text
- Commit files in `inputs/` directory
- Hardcode API keys in code

### Adding New Agents

1. Create `agents/{agent_name}.py` with Pydantic types
2. Add instructions to `instructions/{agent_name}.txt`
3. Write pytests first (TDD)
4. Implement agent with `retries=5`
5. Update `agents/__init__.py`
6. Import in orchestrator if needed

## 📊 Pipeline Details

### Download Pipeline
- **Concurrency:** 10 workers
- **Source:** Patreon RSS feed
- **Output:** `inputs/` directory
- **Formats:** `.mp3`, `.wav`, `.m4a`, etc.
- **Smart Skip:** Checks for existing files

### Transcription Pipeline
- **Concurrency:** 5 workers (API limit)
- **Model:** Deepgram Nova-3
- **Language:** Greek (`el`)
- **Features:** Speaker diarization, timestamps, smart formatting, punctuation
- **Output:** `transcripts/*.txt` with full text + timestamped sections
- **Processing:** Transcripts are then preprocessed (speakers combined, headers removed) and stored in `transcripts_processed/`

## 🤖 MCP Server

The MCP server (`agents/mcp_server.py`) exposes podcast retrieval as tools via the FastMCP streamable-http protocol. There is **no LLM in the server** — it returns raw data (search results, transcript text, metadata). The model selected in OpenWebUI does the synthesis.

### Features

✅ **Implemented:**
- Multilingual embeddings (supports Greek)
- ChromaDB vector database with semantic search
- Speaker-aware semantic chunking
- CrossEncoder reranking for improved relevance
- On-topic filtering via BART NLI classifier (blocks non-history queries)
- Podcast metadata access via `podcast_info.json`

### MCP Endpoint

```
http://localhost:8001/mcp   (streamable-http transport)
```

List available tools:
```bash
uv run python -c "import asyncio; from agents.mcp_server import mcp; tools = asyncio.run(mcp.list_tools()); [print(t.name) for t in tools]"
```

## 📚 Tech Stack

- **Python 3.12** with `uv` for fast package management
- **FastMCP 3.0+** - MCP server (streamable-http transport)
- **OpenWebUI** - Multi-user chat UI (Docker, port 3000)
- **Ollama** - Local LLM inference (any model, runs on host)
- **Pydantic** - Type-safe data models
- **Deepgram Nova-3** - Greek speech-to-text transcription
- **ChromaDB** - Vector database for semantic search
- **sentence-transformers** - Multilingual embeddings + CrossEncoder reranking
- **transformers (BART)** - On-topic classification
- **Logfire** - Observability and monitoring
- **feedparser, requests, pydub** - Data processing utilities

## 🐛 Common Issues

### "Import could not be resolved" Errors
These are expected before installing dependencies:
```bash
uv pip install -e ".[dev]"
```

### Transcription Failures
- Check `DEEPGRAM_API_KEY` is set
- Verify audio file format is supported
- Check API rate limits (stay at 5 workers)

### Greek Text Encoding Issues
All files must use UTF-8:
```python
open(file, 'w', encoding='utf-8')
```

## 📝 License

[Add license information]

## 🤝 Contributing

1. Write pytests first (TDD)
2. Follow project conventions
3. Ensure Greek UTF-8 support
4. Don't commit audio files (`inputs/`)
5. Test with small audio samples first

## 📞 Contact

[Add contact information]

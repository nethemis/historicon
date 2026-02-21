# HistoriCon - Greek Podcast RAG Agent

RAG (Retrieval Augmented Generation) system for the **HistoriCon** Greek-Cypriot history/storytelling podcast. Downloads audio from Patreon, transcribes to Greek with speaker diarization, and enables AI-powered question answering.

## 🎙️ Project Overview

HistoriCon is a multi-agent AI system that:
1. **Downloads** podcast episodes from Patreon RSS feeds
2. **Transcribes** Greek audio with speaker diarization using Deepgram
3. **Indexes** transcripts for semantic search
4. **Answers** questions about podcast content using RAG

## 🏗️ Architecture

### Data Pipeline
```
Patreon RSS → Audio Files → Deepgram → Greek Transcripts → Vector DB → AI Agents
```

### Multi-Agent System
- **Web Orchestrator** - Main entry point, handles user queries
- **Retrieval Agent** - Searches transcripts and returns relevant chunks
- *(Future)* Memory Agent - Manages conversation context

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Setup secrets and environment (project-specific command)
secrets && se

# Install dependencies with uv
uv pip install -e ".[dev]"
```

### 2. Download & Transcribe Podcasts

```bash
# Download new episodes from Patreon
python download_patreon.py

# Transcribe to Greek with speaker diarization
python transcribe_deepgram.py
```

### 3. Run the RAG Agent

```bash
# Start the web server
python -m agents.web_orchestrator

# Or run the example script
python agents/example_usage.py
```

## 📋 Requirements

### Environment Variables

**Required:**
- `ANTHROPIC_API_KEY` - Claude API key for AI agents
- `DEEPGRAM_API_KEY` - For Greek transcription

**Optional:**
- `LOGFIRE_TOKEN` - For observability (tracking agent performance)
- `LOGFIRE_SERVICE_NAME` - Service name in logs (default: "historicon-rag-agent")
- `ENVIRONMENT` - Environment name (default: "development")

### Python Version
- Python 3.13+
- Package manager: `uv` (fast Python installer)

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
├── agents/                          # Multi-agent RAG system
│   ├── __init__.py                  # Package exports
│   ├── models.py                    # Pydantic models
│   ├── logfire_setup.py             # Observability config
│   ├── retrieval.py                 # Retrieval agent (dummy)
│   ├── web_orchestrator.py          # Main orchestrator
│   ├── example_usage.py             # Usage examples
│   └── README.md                    # Agent system docs
├── instructions/                    # Agent system prompts
│   ├── web_orchestrator.txt
│   └── retrieval.txt
├── tests/                           # Pytests
│   ├── test_retrieval.py
│   └── test_web_orchestrator.py
├── inputs/                          # Downloaded audio (gitignored)
├── transcripts/                     # Greek transcripts with timestamps
├── download_patreon.py              # Patreon RSS downloader
├── transcribe_deepgram.py           # Deepgram transcription
├── trim_audio.py                    # Audio trimming utility
├── pyproject.toml                   # Dependencies
└── README.md                        # This file
```

## 🔧 Development

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=agents --cov-report=html

# Run specific test file
pytest tests/test_retrieval.py
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
- **Features:** 
  - Speaker diarization
  - Smart formatting
  - Punctuation
  - Timestamps
- **Output:** `transcripts/*.txt` with full text + timestamped sections

### Transcript Format
```
================================================================================
FULL TRANSCRIPT
================================================================================
[Complete Greek text without timestamps]

================================================================================
TIMESTAMPED TRANSCRIPT WITH SPEAKERS
================================================================================
[00:01:28.990 - 00:01:30.350] Speaker 0:
[Greek text for this segment]
```

## 🤖 Agent System

### Current Status

⚠️ **Dummy Implementation**: The retrieval agent currently returns mock data.

### To Implement Full RAG:
1. Add embedding model (e.g., `sentence-transformers` with Greek model)
2. Add vector database (e.g., ChromaDB, FAISS, Qdrant)
3. Index transcripts from `transcripts/` directory
4. Replace dummy `search_transcripts` with real vector search
5. Implement semantic chunking strategy
6. Add reranking for better results

### API Usage

```python
from agents import web_orchestrator, retrieval_agent

# Query the orchestrator
result = await web_orchestrator.run("Πες μου για τον Γ. Κοσκωτά")
print(result.output)

# Direct retrieval query
retrieval_result = await retrieval_agent.run(
    "Search for: Κοσκωτάς (max 5 results)"
)
print(retrieval_result.output.summary)
```

### Web API

```bash
# Start server on port 8001
python -m agents.web_orchestrator

# Query via HTTP
curl -X POST http://localhost:8001/run \
  -H "Content-Type: application/json" \
  -d '{"query": "Tell me about George Santos"}'
```

## 📚 Tech Stack

### Core
- **Python 3.13+** with `uv` for fast package management
- **Pydantic** - Data validation and models
- **Pydantic AI** - AI agent framework with type safety

### AI/ML
- **Anthropic Claude** - Sonnet 4.5 for agent reasoning
- **Deepgram** - Greek speech-to-text transcription
- **Logfire** - AI observability and monitoring

### Data Processing
- **feedparser** - Patreon RSS parsing
- **requests** - HTTP downloads with streaming
- **pydub** - Audio file manipulation

### Future RAG Stack
- **sentence-transformers** - Text embeddings
- **ChromaDB/FAISS** - Vector database
- **langchain** (optional) - RAG utilities

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

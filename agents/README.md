# HistoriCon Multi-Agent RAG System

Multi-agent system for querying Greek podcast transcripts using retrieval-augmented generation (RAG).

## Architecture

### Agents
- **Web Orchestrator** (`web_orchestrator.py`) - Main entry point that handles user requests
- **Retrieval Agent** (`retrieval.py`) - Searches transcripts and returns relevant chunks (currently dummy implementation)

### Key Features
- Pydantic-based strict typing for all inputs/outputs
- Logfire observability integration
- Web API via `uvicorn`
- Designed for Greek-language podcast transcripts with timestamps
- Speaker diarization support

## Setup

### Install Dependencies

```bash
# Setup environment (always run first)
secrets && se

# Install dependencies with uv
uv pip install -e .
```

### Environment Variables

Required:
- `ANTHROPIC_API_KEY` - Claude API key for AI agents
- `DEEPGRAM_API_KEY` - For transcription (existing pipeline)

Optional:
- `LOGFIRE_TOKEN` - For observability (optional)
- `LOGFIRE_SERVICE_NAME` - Service name in logs (default: "historicon-rag-agent")
- `ENVIRONMENT` - Environment name (default: "development")

## Usage

### Running the Web Server

```bash
# Start the web server on port 8001
python -m agents.web_orchestrator
```

The server exposes a web API for the orchestrator agent.

### Using the Agents Programmatically

```python
from agents import web_orchestrator, retrieval_agent

# Query the orchestrator
result = await web_orchestrator.run("Πες μου για τον Γ. Κοσκωτά")
print(result.output)

# Directly query retrieval agent (for testing)
retrieval_result = await retrieval_agent.run(
    "Search for: Κοσκωτάς (max 5 results)"
)
print(retrieval_result.output.summary)
print(retrieval_result.output.chunks)
```

## Current Status

⚠️ **Dummy Implementation**: The retrieval agent currently returns mock data. 

### To Implement Full RAG:
1. Add embedding model (e.g., `sentence-transformers` with Greek model)
2. Add vector database (e.g., ChromaDB, FAISS)
3. Index transcripts from `transcripts/` directory
4. Replace `search_transcripts` dummy implementation with real vector search

## File Structure

```
agents/
├── __init__.py              # Package exports
├── models.py                # Pydantic models for all data
├── logfire_setup.py         # Observability configuration
├── retrieval.py             # Retrieval agent (dummy)
├── web_orchestrator.py      # Main orchestrator agent
instructions/
├── web_orchestrator.txt     # Orchestrator system prompt
├── retrieval.txt            # Retrieval agent system prompt
```

## Development

### Following Project Conventions

All code follows HistoriCon conventions:
- ✅ Pydantic BaseModels for all inputs/outputs
- ✅ `retries=5` on all agents
- ✅ Parallel processing where applicable
- ✅ Greek UTF-8 support throughout
- ✅ Eval-driven development for agents

### Testing

Create pytests for each agent following TDD methodology:

```python
# tests/test_retrieval.py
async def test_retrieval_returns_chunks():
    result = await retrieval_agent.run("test query")
    assert isinstance(result.output, RetrievalResponse)
    assert len(result.output.chunks) > 0
```

### Adding New Agents

1. Create agent file in `agents/`
2. Add instructions to `instructions/{agent_name}.txt`
3. Import in `agents/__init__.py`
4. Add tools to orchestrator if needed
5. Write evals/pytests

## API Reference

### Models

- `RetrievalChunk` - Single document chunk with score and timestamp
- `RetrievalResponse` - Search results with summary
- `OrchestratorRequest` - Input to orchestrator
- `OrchestratorResponse` - Final answer with sources

### Tools

**Web Orchestrator:**
- `search_documents(query, max_results)` - Search transcripts
- `get_available_transcripts()` - List all episodes

**Retrieval Agent:**
- `search_transcripts(query, max_results)` - Find relevant chunks
- `get_transcript_list()` - List transcript files

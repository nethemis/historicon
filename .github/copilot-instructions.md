# HistoriCon - Greek Podcast RAG Agent

## Project Overview
RAG (Retrieval Augmented Generation) agent for the **HistoriCon** Greek-Cypriot history/storytelling podcast. The system downloads audio from Patreon RSS feeds, transcribes to Greek text with speaker diarization, and enables AI-powered question answering and content retrieval from podcast episodes.

## 🚨 CRITICAL: Test-Driven Development is MANDATORY
**ALWAYS write tests FIRST before any implementation code. Run `uv run pytest` to verify.**

## Tech Stack

- **Language:** Python
- **Package Management:** `uv` (fast Python package installer and resolver)
- **Key Libraries:**
  - `pydantic` - Data validation and settings management
  - `guardrails` - AI output validation
  - `anthropic` - Claude API integration
  - `sentence-transformers` - Embeddings for RAG
  - `feedparser` - Patreon RSS parsing
  - `requests` - HTTP downloads
  - `deepgram` - Greek transcription API
  - `pydub` - Audio manipulation

## Architecture & Data Flow

**Sequential Pipeline:**
1. `download_patreon.py` → downloads Patreon audio to `inputs/`
2. `transcribe_deepgram.py` → transcribes audio from `inputs/` to `transcripts/`
3. `preprocess_transcripts.py` → cleans and combines transcripts into `transcripts_processed/`
4. `create_embeddings.py` → creates embeddings and stores in `chroma_db/`
5. `agents/web_orchestrator.py` → RAG agent for querying podcast content

**Directory Structure:**
- `inputs/` - Downloaded audio files (`.mp3`, `.wav`, `.m4a`, etc.) - **gitignored**
- `transcripts/` - Raw transcripts from Deepgram with timestamps and speakers
- `transcripts_processed/` - Cleaned transcripts (combined speakers, removed headers) - **committed to git**
- `chroma_db/` - ChromaDB vector database with embeddings - **gitignored**
- `agents/` - Multi-agent RAG system (web_orchestrator, retrieval)
- `instructions/` - Agent system prompts (web_orchestrator.txt, retrieval.txt)
- `tests/` - Pytest test suite (42 tests total)
- `config.json` - Centralized configuration for all components
- `venv/` or `.venv/` - Python virtual environment - **gitignored**

## Development Methodology

### Test-Driven Development (TDD) - MANDATORY
**ALWAYS FOLLOW TDD. NO EXCEPTIONS.**

**Everything must have corresponding pytests.** When starting ANY feature:
1. **WRITE THE TEST FIRST** - Do not write any implementation code before tests exist
2. Run the test (it should fail - red)
3. Write minimal code to make the test pass (green)
4. Refactor if needed while keeping tests passing
5. Repeat for next feature

**Critical TDD Rules:**
- Tests must be written BEFORE implementation code
- Tests should be runnable with `uv run pytest tests/`
- Tests must not require manual setup beyond `secrets && se`
- Mock external API calls to avoid authentication and cost issues
- Use pytest fixtures for common setup
- Run tests frequently during development

### Eval-Driven Development (for AI Agents)
**Before implementing each agent:**
1. Write simple Pydantic evaluations that define expected behavior
2. Implement the agent to perform well on the evals
3. Iterate and reevaluate evals themselves as needed
4. Mock LLM calls in unit tests to avoid API costs

### Pydantic Conventions
**Strict typing with Pydantic BaseModels:**
- Every function's input and output must be a Pydantic `BaseModel`
- Every agent's input and output must be a Pydantic AI model
- All agents should have `retries=5` configured
- Use Pydantic for data validation throughout the codebase

## Critical Patterns

### Parallel Processing
- **Downloads:** 10 concurrent workers (`ThreadPoolExecutor(max_workers=10)`)
- **Transcriptions:** 5 concurrent workers (`ThreadPoolExecutor(max_workers=5)`)
- All scripts use `concurrent.futures` for parallel I/O operations

### Filename Sanitization (Greek-Specific)
See `sanitize_filename()` in [download_patreon.py](../download_patreon.py):
- Spaces → underscores
- Keeps only: `[a-zA-Z0-9_\-\.\(\)]` (Greek UTF-8 preserved)
- Removes consecutive underscores
- Example: `Ο Τζο Λό(ος) τ αδρώπου.mp3` → `Ο_Τζο_Λόος_τ_αδρώπου.mp3`

### Smart Skip Logic
**Both scripts check for existing files before processing:**
- Downloads: Checks if filename prefix exists in `inputs/` (handles duplicates)
- Transcriptions: Checks if `.txt` exists in `transcripts/` before transcribing
- This is critical for rerunning scripts without wasting API calls

### Transcript Structure
**Raw transcripts** (`transcripts/`) have two sections:
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

**Processed transcripts** (`transcripts_processed/`) are cleaned:
- FULL TRANSCRIPT section removed
- Consecutive same-speaker entries combined into single blocks
- Empty lines removed
- File size reduced by ~97% (e.g., 3558 lines → 112 lines)
- Format: `[HH:MM:SS.mmm - HH:MM:SS.mmm] Speaker X:\n[text]`

## Dependencies & Configuration

### Configuration File: `config.json`
**Centralized settings for entire system** (loaded via `config.py` Pydantic model):

```json
{
  "transcripts_dir": "transcripts_processed",
  "chroma_db_path": "chroma_db",
  "collection_name": "historicon_transcripts",
  "embedding_model": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
  "similarity_threshold": 0.85,
  "chunk_min_size": 1000,
  "chunk_max_size": 10000,
  "max_transcript_tokens": 10000,
  "max_context_tokens": 190000
}
```

**Key Settings:**
- `similarity_threshold`: Cosine similarity for grouping speaker segments (0.85 = loose grouping, larger chunks; 0.5 = strict, smaller chunks)
- `chunk_min_size`/`chunk_max_size`: Character limits for chunks (speakers never split even if exceeding max)
- `max_transcript_tokens`: Page size for paginated transcript access (10k tokens)
- `max_context_tokens`: Safety limit for conversation context (190k = buffer for Claude 200k limit)
- `transcripts_dir`: Source directory for embeddings (use `transcripts_processed`, not raw `transcripts`)

### Required Environment Variables
- `DEEPGRAM_API_KEY` - **Must be set** for transcription (Deepgram client reads from `os.environ`)
- `ANTHROPIC_API_KEY` - For Claude API access in RAG agents

### Key Dependencies
- `feedparser` - Patreon RSS parsing
- `requests` - HTTP downloads with streaming
- `deepgram` - Greek transcription API (Nova-3 model)
- `pydub` - Audio file manipulation

### Deepgram Configuration
Hardcoded in [transcribe_deepgram.py](../transcribe_deepgram.py#L31-L39):
```python
model="nova-3",
language="el",  # Greek
smart_format=True,
punctuate=True,
paragraphs=True,
utterances=True,
diarize=True,  # Speaker diarization
```

### RSS Configuration
**Hardcoded in** [download_patreon.py](../download_patreon.py#L9):
```python
RSS_URL = "https://www.patreon.com/rss/istorikon?auth=<TOKEN>&show=866770"
```
⚠️ **Contains authentication token** - keep this file private

## Development Workflows

### Environment Setup
**At every new terminal session, run:**
```bash
secrets && se
```
This sets up the environment with required secrets and configuration.

### Running the Complete Pipeline
```bash
# 1. Setup environment (always run first)
secrets && se

# 2. Download new episodes from Patreon RSS
python download_patreon.py

# 3. Transcribe new audio files to Greek text
python transcribe_deepgram.py

# 4. Preprocess transcripts (combine speakers, remove headers)
python preprocess_transcripts.py
# Optional: dry-run mode to preview changes
python preprocess_transcripts.py --dry-run

# 5. Create embeddings from processed transcripts
python create_embeddings.py
# Optional: force reindex all files
python create_embeddings.py --force-reindex

# 6. Run RAG agent web server
uv run python agents/web_orchestrator.py
# Or use example usage script
uv run python agents/example_usage.py
```

### Testing with Small Samples
Use [trim_audio.py](../trim_audio.py) to extract short clips for API testing:
```python
audio = AudioSegment.from_mp3("sample_1m.mp3")
first_28_seconds = audio[:28000]  # milliseconds
first_28_seconds.export("sample_28s.mp3", format="mp3")
```

### Handling Greek UTF-8
- **All files use UTF-8 encoding** for Greek text
- File operations: `open(file, 'w', encoding='utf-8')`
- Don't ASCII-normalize Greek characters in filenames

## RAG Agent System

### Available Tools in Web Orchestrator

**1. `search_documents` (PRIMARY - Use First!)**
- Semantic search across all podcast transcripts
- Returns top N chunks with episode names, timestamps, and relevance scores
- Best for: answering questions, finding topics, discovering episodes
- Example: `search_documents("George Santos scandal", max_results=5)`

**2. `get_transcript_section` (TARGETED ACCESS)**
- Retrieve specific time range from an episode
- Takes episode filename + start/end timestamps (HH:MM:SS format)
- Best for: getting context around a specific timestamp from search results
- Example: `get_transcript_section("George_Santos.txt", "00:15:30", "00:20:00")`

**3. `get_full_transcript` (PAGINATED - Use Sparingly!)**
- Access full transcript page by page (10k tokens per page)
- Returns dict with: `{"content": str, "end_of_file": bool, "page": int, "total_pages": int}`
- Best for: sequential reading of entire episode (rare use case)
- **WARNING:** Full transcripts are very long, prefer `search_documents`
- Example: `get_full_transcript("Episode.txt", page=1)`

### Tool Usage Pattern
```python
# 1. Start with search to find relevant content
results = search_documents("Κοσκωτάς σκάνδαλο", max_results=5)

# 2. If need more context around a timestamp, use section retrieval
section = get_transcript_section(
    episode_name="Γ._Κοσκωτάς.txt",
    start_time="00:15:00",
    end_time="00:20:00"
)

# 3. Only use full transcript if explicitly requested
page1 = get_full_transcript("Episode.txt", page=1)
if not page1["end_of_file"]:
    page2 = get_full_transcript("Episode.txt", page=2)
```

### Context Size Limits
- Claude Sonnet 4.5 has 200k token limit
- `max_context_tokens` set to 190k for safety buffer
- `check_context_size()` function monitors usage via `ctx.usage`
- Agent warns/blocks when approaching limit (though tracking is imperfect)
- **Best practice:** Use `search_documents` over pagination to avoid context overflow

## Common Tasks

### Adding Support for New Audio Formats
Update `audio_extensions` set in [transcribe_deepgram.py](../transcribe_deepgram.py#L147):
```python
audio_extensions = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".webm"}
```

### Adjusting Concurrency
- **Downloads faster, fewer API limits** → increase workers in `download_single_file` executor
- **Transcriptions API-limited** → keep at 5 workers to avoid rate limits

### Running Tests
```bash
# Run all tests with verbose output
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_preprocess_transcripts.py -v

# Run with coverage report
uv run pytest tests/ --cov=. --cov-report=html
```

**Current Test Suite (42 tests):**
- `test_create_embeddings.py` - Config, indexing, chunking, embeddings (15 tests)
- `test_preprocess_transcripts.py` - Preprocessing logic (8 tests)
- `test_retrieval.py` - Retrieval agent functionality (9 tests)
- `test_web_orchestrator.py` - Web orchestrator tools and delegation (10 tests)

### Preprocessing Transcripts
**What `preprocess_transcripts.py` does:**
1. Reads raw transcripts from `transcripts/`
2. Combines consecutive same-speaker entries into single blocks
3. Removes the "FULL TRANSCRIPT" header section
4. Removes empty lines
5. Writes cleaned output to `transcripts_processed/`
6. Reduces file size by ~97% (e.g., 3558 → 112 lines)

**Usage:**
```bash
# Normal mode - writes to transcripts_processed/
python preprocess_transcripts.py

# Dry-run mode - shows changes without writing
python preprocess_transcripts.py --dry-run
```

**Why preprocessing matters:**
- Creates cleaner chunks for semantic embedding
- Reduces redundancy in vector database
- Removes non-content sections that waste token space
- Maintains all timestamps and speaker information

### Debugging Failed Transcriptions
Check return tuple from `transcribe_audio()`:
```python
(status, filename, message, transcript_length)
# status: "success" | "error"
# message: Error details when status="error"
```

## Anti-Patterns to Avoid

❌ **Don't** remove filename sanitization - Greek filenames need special handling  
❌ **Don't** skip the existing-file checks - wastes API credits  
❌ **Don't** use ASCII encoding for Greek text  
❌ **Don't** commit files in `inputs/` directory (raw audio files)  
❌ **Don't** commit files in `chroma_db/` directory (can regenerate from processed transcripts)  
❌ **Don't** hardcode API keys in code - use environment variables  
❌ **Don't** create embeddings from raw `transcripts/` - use `transcripts_processed/`  
❌ **Don't** use `get_full_transcript` for information retrieval - use `search_documents` first  
❌ **Don't** implement code before writing tests - TDD is mandatory

## Project-Specific Conventions

- Error handling uses try/except with tuple returns, not raised exceptions
- Progress indicators use emoji: ✅ (success), ⏭️ (skipped), ❌ (error)
- All user-facing messages in Greek where appropriate
- Keep parallel workers conservative to respect API rate limits
- All Pydantic AI agents have `retries=5` configured
- Agent instructions live in `instructions/` directory as `.txt` files
- Config settings are centralized in `config.json` and loaded via `config.py`
- Use `transcripts_processed/` (not raw `transcripts/`) for downstream processing
- RAG workflow: `search_documents` → `get_transcript_section` → `get_full_transcript` (in order of preference)

"""Web Orchestrator Agent - Main entry point for the multi-agent system.

This agent acts as the orchestrator, handling user requests and delegating
to specialized agents (retrieval agent) for information access.
"""

import os
from datetime import datetime
from pathlib import Path

import logfire
import uvicorn
from pydantic_ai import Agent, RunContext

# Handle imports for both script execution and module import
try:
    from .logfire_setup import configure_logfire
    from .models import RetrievalResponse
    from .retrieval import retrieval_agent
except ImportError:
    from logfire_setup import configure_logfire
    from models import RetrievalResponse
    from retrieval import retrieval_agent

# Import config from parent directory
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import config

# Configure Logfire for observability (skip in tests)
if os.getenv("LOGFIRE_AUTO_CONFIGURE", "true").lower() == "true":
    configure_logfire()


def load_instructions(agent_name: str) -> str:
    """Load agent instructions from file."""
    instructions_dir = Path(__file__).parent.parent / "instructions"
    instruction_file = instructions_dir / f"{agent_name}.txt"

    if instruction_file.exists():
        return instruction_file.read_text(encoding="utf-8")
    else:
        logfire.warn(f"Instruction file not found: {instruction_file}")
        return "You are a helpful AI assistant for the HistoriCon Greek podcast."


# Web orchestrator agent - delegates to specialized agents
web_orchestrator = Agent(
    "anthropic:claude-sonnet-4-5",
    system_prompt=load_instructions("web_orchestrator"),
    retries=5,  # Project convention: all agents have retries=5
)


def check_context_size(ctx: RunContext[None]) -> tuple[bool, int]:
    """
    Check if context is approaching token limits.

    Args:
        ctx: The run context with usage tracking

    Returns:
        Tuple of (is_over_limit, current_tokens)
    """
    try:
        usage = ctx.usage
        current_tokens = 0

        # Try multiple ways to get token count from pydantic-ai
        if hasattr(usage, "total_tokens"):
            current_tokens = usage.total_tokens()
        elif hasattr(usage, "token_count"):
            current_tokens = usage.token_count()
        elif callable(usage):
            usage_obj = usage()
            if hasattr(usage_obj, "total_tokens"):
                current_tokens = usage_obj.total_tokens
            elif hasattr(usage_obj, "request_tokens"):
                current_tokens = usage_obj.request_tokens

        # Log current usage for debugging
        logfire.info(
            f"Context size check: {current_tokens} tokens",
            current_tokens=current_tokens,
            limit=config.max_context_tokens,
        )

        is_over_limit = current_tokens > config.max_context_tokens

        if is_over_limit:
            logfire.warn(
                f"⚠️ Context size exceeds safe limit: {current_tokens} > {config.max_context_tokens}",
                current_tokens=current_tokens,
                limit=config.max_context_tokens,
            )

        return is_over_limit, current_tokens
    except Exception as e:
        logfire.error(f"Failed to check context size: {e}", error=str(e))
        # If we can't check, assume we're fine but log the issue
        return False, 0


@web_orchestrator.tool
async def search_documents(
    ctx: RunContext[None],
    query: str,
    max_results: int = 5,
) -> RetrievalResponse:
    """
    Search through indexed transcript documents for relevant information.

    Args:
        ctx: The run context
        query: The search query or question (Greek or English)
        max_results: Maximum number of results to return

    Returns:
        RetrievalResponse with chunks and synthesized summary
    """
    logfire.info(
        "Delegating document search to retrieval agent",
        query=query,
        max_results=max_results,
    )

    # Delegate to retrieval agent
    result = await retrieval_agent.run(
        f"Search for: {query} (max {max_results} results)",
        usage=ctx.usage,
    )

    logfire.info(
        "Document search complete",
        num_chunks=len(result.output.chunks),
    )

    return result.output


@web_orchestrator.tool
async def get_transcript_section(
    ctx: RunContext[None],
    episode_name: str,
    start_time: str,
    end_time: str,
) -> str:
    """
    Get a specific time range from a podcast episode transcript.

    Use this when:
    - User asks about a specific time in an episode
    - Need continuous context around a timestamp from search results
    - Want to see exact dialogue in a time window

    IMPORTANT: Use search_documents first to find relevant timestamps!

    Args:
        ctx: The run context
        episode_name: Episode filename (e.g., "George_Santos.txt")
        start_time: Start timestamp in HH:MM:SS format (e.g., "00:15:30")
        end_time: End timestamp in HH:MM:SS format (e.g., "00:20:00")

    Returns:
        Transcript text for the specified time range

    Example:
        get_transcript_section("Episode_Name.txt", "00:15:00", "00:20:00")
    """
    # Check context size before adding more content
    is_over_limit, current_tokens = check_context_size(ctx)
    if is_over_limit:
        return (
            f"⚠️ WARNING: Context size limit exceeded ({current_tokens} / {config.max_context_tokens} tokens).\n"
            f"Cannot add more transcript content. The conversation is too long.\n"
            f"Please start a new conversation or use search_documents instead of requesting full sections."
        )

    logfire.info(
        "Fetching transcript section",
        episode_name=episode_name,
        start_time=start_time,
        end_time=end_time,
    )

    transcripts_dir = Path(config.transcripts_dir)
    transcript_path = transcripts_dir / episode_name

    def parse_timestamp(ts: str) -> float:
        """Convert HH:MM:SS.mmm to seconds"""
        try:
            parts = ts.split(":")
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
            return hours * 3600 + minutes * 60 + seconds
        except:
            return 0.0

    try:
        if not transcript_path.exists():
            return f"Error: Transcript '{episode_name}' not found"

        full_text = transcript_path.read_text(encoding="utf-8")
        lines = full_text.split("\n")

        start_seconds = parse_timestamp(start_time)
        end_seconds = parse_timestamp(end_time)

        # Extract lines within time range
        selected_lines = []
        import re

        for line in lines:
            # Match timestamp pattern: [HH:MM:SS.mmm - HH:MM:SS.mmm]
            match = re.match(r"\[(\d{2}:\d{2}:\d{2}\.\d{3})", line)
            if match:
                line_timestamp = match.group(1)
                line_seconds = parse_timestamp(line_timestamp)

                if start_seconds <= line_seconds <= end_seconds:
                    selected_lines.append(line)

        if not selected_lines:
            return f"No content found in time range {start_time} - {end_time}"

        result = "\n".join(selected_lines)
        logfire.info(
            f"Retrieved transcript section: {episode_name}",
            lines_count=len(selected_lines),
        )
        return result

    except Exception as e:
        logfire.error(f"Failed to load transcript section {episode_name}: {e}")
        return f"Error loading transcript section: {str(e)}"


@web_orchestrator.tool
async def get_full_transcript(
    ctx: RunContext[None],
    episode_name: str,
    page: int = 1,
) -> dict[str, any]:
    """
    Get full transcript with pagination (returns up to 10,000 tokens per call).

    ⚠️  WARNING: Use search_documents first! Only use this for sequential reading.

    PAGINATION GUIDE:
    - Returns dict with: {"content": str, "end_of_file": bool, "page": int}
    - If end_of_file=False, more content is available
    - Call with page=2, page=3, etc. to get subsequent pages
    - Each page contains up to 10,000 tokens (configurable)

    Use this when:
    - User explicitly wants to read entire episode sequentially
    - Need to browse through full transcript page by page

    DO NOT use this when:
    - Looking for specific information (use search_documents instead!)
    - Need a specific time range (use get_transcript_section instead!)

    Args:
        ctx: The run context
        episode_name: Episode filename (e.g., "George_Santos.txt")
        page: Page number (starts at 1)

    Returns:
        Dict with:
        - content: The transcript text for this page
        - end_of_file: True if this is the last page, False if more pages available
        - page: Current page number
        - total_pages: Total number of pages in the transcript

    Example:
        result = get_full_transcript("Episode.txt", page=1)
        if not result["end_of_file"]:
            next_page = get_full_transcript("Episode.txt", page=2)
    """
    # Check context size before adding more content
    is_over_limit, current_tokens = check_context_size(ctx)
    if is_over_limit:
        return {
            "content": (
                f"⚠️ WARNING: Context size limit exceeded ({current_tokens} / {config.max_context_tokens} tokens).\n"
                f"Cannot add more transcript content. The conversation is too long.\n\n"
                f"RECOMMENDATION: Start a new conversation or use search_documents to find specific information "
                f"instead of reading full transcripts page by page."
            ),
            "end_of_file": True,
            "page": page,
            "total_pages": 0,
            "error": "context_limit_exceeded",
        }

    logfire.info(
        "Fetching full transcript (paginated)",
        episode_name=episode_name,
        page=page,
    )

    transcripts_dir = Path(config.transcripts_dir)
    transcript_path = transcripts_dir / episode_name

    try:
        if not transcript_path.exists():
            return {
                "content": f"Error: Transcript '{episode_name}' not found",
                "end_of_file": True,
                "page": page,
                "total_pages": 0,
            }

        full_text = transcript_path.read_text(encoding="utf-8")

        # Approximate token count (Greek/English mixed: ~3.5 chars per token)
        max_chars = config.max_transcript_tokens * 4  # Conservative estimate

        # Calculate total pages
        total_chars = len(full_text)
        total_pages = (total_chars + max_chars - 1) // max_chars  # Ceiling division

        # Validate page number
        if page < 1:
            page = 1
        if page > total_pages:
            return {
                "content": f"Error: Page {page} exceeds total pages ({total_pages})",
                "end_of_file": True,
                "page": page,
                "total_pages": total_pages,
            }

        # Extract page content
        start_idx = (page - 1) * max_chars
        end_idx = min(page * max_chars, total_chars)
        content = full_text[start_idx:end_idx]

        end_of_file = page >= total_pages

        logfire.info(
            f"Retrieved transcript page {page}/{total_pages}: {episode_name}",
            content_length=len(content),
            end_of_file=end_of_file,
        )

        return {
            "content": content,
            "end_of_file": end_of_file,
            "page": page,
            "total_pages": total_pages,
        }

    except Exception as e:
        logfire.error(f"Failed to load transcript {episode_name}: {e}")
        return {
            "content": f"Error loading transcript: {str(e)}",
            "end_of_file": True,
            "page": page,
            "total_pages": 0,
        }


# Create the web app
app = web_orchestrator.to_web()

if __name__ == "__main__":
    uvicorn.run(app=app, host="0.0.0.0", port=8001)

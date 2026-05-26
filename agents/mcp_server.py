"""MCP server for HistoriCon — exposes podcast retrieval tools over streamable-http.

Tools:
  search_documents           — semantic search across all transcript chunks
  get_transcript_section     — time-range slice of a processed transcript
  list_podcast_info_sections — list section keys from podcast_info.json
  get_podcast_info_section   — get content of a named section

Run:
    uv run python agents/mcp_server.py
    # Listens on http://0.0.0.0:8001/mcp (streamable-http transport)
"""

import json
import re
from pathlib import Path

from opentelemetry import trace

from agents import otel_setup  # noqa: F401 — import-time OpenTelemetry bootstrap
from agents.config import config
from agents.retrieval import search_transcripts
from fastmcp import FastMCP

# Get tracer for this module
_tracer = otel_setup.get_tracer("historicon.mcp_server")

# Module-level path — can be patched in tests
_PODCAST_INFO_PATH: Path = Path(__file__).parent.parent / "podcast_info.json"

mcp = FastMCP("HistoriCon - Greek History Podcast")


# ─── helpers ──────────────────────────────────────────────────────────────────


def _parse_timestamp(ts: str) -> float:
    """Convert HH:MM:SS or HH:MM:SS.mmm to seconds. Returns 0.0 on parse errors."""
    try:
        parts = ts.split(":")
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    except (ValueError, IndexError):
        return 0.0


# ─── tools ────────────────────────────────────────────────────────────────────


@mcp.tool(
    description=(
        "Search through indexed HistoriCon podcast transcripts for relevant information. "
        "Use this for any question about Greek or Cypriot history, podcast episodes, "
        "or speakers. Returns ranked chunks with episode name, timestamp, and score. "
        "Accepts queries in Greek or English."
    )
)
async def search_documents(query: str, max_results: int = 5) -> dict:
    """Search podcast transcripts and return matching chunks.

    Args:
        query: The search query or question (Greek or English).
        max_results: Maximum number of results to return (default 5).

    Returns:
        Dict with keys: chunks, summary, query, total_results.
    """
    with _tracer.start_as_current_span("search_documents") as span:
        span.set_attribute("query", query)
        span.set_attribute("max_results", max_results)
        response = search_transcripts(query, max_results=max_results)
        span.set_attribute("total_results", len(response.chunks))
        if response.chunks:
            top = response.chunks[0]
            span.set_attribute("top_result.source", top.source)
            span.set_attribute("top_result.score", top.score)
            span.set_attribute("top_result.preview", top.text[:200])
        return response.model_dump()


@mcp.tool(
    description=(
        "Retrieve a specific time range from a HistoriCon episode transcript. "
        "Call this proactively whenever you need broader context: on follow-up questions, "
        "when search chunks are short or incomplete, when building a narrative answer, "
        "or when the user asks for more detail. "
        "Use the EXACT episode filename from search_documents results (full Greek title, .txt extension) "
        "and the timestamp returned by search_documents. "
        "Expand the window ±5 minutes around the timestamp for richer context."
    )
)
def get_transcript_section(episode_name: str, start_time: str, end_time: str) -> str:
    """Get transcript lines in a time range from a processed transcript file.

    Args:
        episode_name: EXACT episode filename from search_documents (e.g.
                      'Γ._Κοσκωτάς_Το_σκάνδαλο.txt'). Do NOT abbreviate.
        start_time: Start timestamp in HH:MM:SS format (e.g. '00:15:30').
        end_time:   End timestamp in HH:MM:SS format (e.g. '00:20:00').

    Returns:
        Matching transcript lines as a string, or an error/empty-range message.
    """
    with _tracer.start_as_current_span("get_transcript_section") as span:
        span.set_attribute("episode_name", episode_name)
        span.set_attribute("start_time", start_time)
        span.set_attribute("end_time", end_time)

        transcript_path = Path(config.transcripts_dir) / episode_name

        if not transcript_path.exists():
            span.set_attribute("error", "transcript_not_found")
            return (
                f"Error: Transcript '{episode_name}' not found in {config.transcripts_dir}/"
            )

        try:
            lines = transcript_path.read_text(encoding="utf-8").split("\n")
            start_sec = _parse_timestamp(start_time)
            end_sec = _parse_timestamp(end_time)

            selected = []
            for line in lines:
                m = re.match(r"\[(\d{2}:\d{2}:\d{2}\.\d{3})", line)
                if m and start_sec <= _parse_timestamp(m.group(1)) <= end_sec:
                    selected.append(line)

            if not selected:
                span.set_attribute("error", "empty_time_range")
                return f"No content found in time range {start_time} – {end_time} for '{episode_name}'"

            result_text = "\n".join(selected)
            span.set_attribute("lines_returned", len(selected))
            span.set_attribute("result_preview", result_text[:300])
            return result_text

        except Exception as exc:
            span.record_exception(exc)
            span.set_attribute("error", str(exc))
            return f"Error reading transcript: {exc}"


@mcp.tool(
    description=(
        "List all available information sections about the HistoriCon podcast. "
        "Returns 'KEY - Title' strings usable with get_podcast_info_section. "
        "Call this first to discover what podcast metadata is available."
    )
)
def list_podcast_info_sections() -> list[str]:
    """List section keys and titles from podcast_info.json."""
    with _tracer.start_as_current_span("list_podcast_info_sections") as span:
        try:
            if not _PODCAST_INFO_PATH.exists():
                span.set_attribute("error", "file_not_found")
                return ["Error: podcast_info.json not found"]

            data = json.loads(_PODCAST_INFO_PATH.read_text(encoding="utf-8"))
            sections = [f"{key} - {val['title']}" for key, val in data.items()]
            span.set_attribute("count", len(sections))
            return sections

        except Exception as exc:
            span.record_exception(exc)
            span.set_attribute("error", str(exc))
            return [f"Error: {exc}"]


@mcp.tool(
    description=(
        "Get detailed information about a specific aspect of the HistoriCon podcast. "
        "Call list_podcast_info_sections first to find the section key. "
        "For episode content use search_documents instead."
    )
)
def get_podcast_info_section(section_key: str) -> str:
    """Get content of a named section from podcast_info.json.

    Args:
        section_key: Section identifier from list_podcast_info_sections()
                     (e.g. 'ΒΑΣΙΚΑ_ΣΤΟΙΧΕΙΑ', 'ΠΑΡΟΥΣΙΑΣΤΕΣ__ΟΜΑΔΑ').

    Returns:
        Formatted section content, or an error message if not found.
    """
    with _tracer.start_as_current_span("get_podcast_info_section") as span:
        span.set_attribute("section_key", section_key)
        try:
            if not _PODCAST_INFO_PATH.exists():
                span.set_attribute("error", "file_not_found")
                return "Error: podcast_info.json not found"

            data = json.loads(_PODCAST_INFO_PATH.read_text(encoding="utf-8"))

            # Exact match
            if section_key in data:
                span.set_attribute("match_type", "exact")
                return f"# {data[section_key]['title']}\n\n{data[section_key]['content']}"

            # Case-insensitive partial match
            key_upper = section_key.upper()
            for k, v in data.items():
                if key_upper in k.upper():
                    span.set_attribute("match_type", "partial")
                    span.set_attribute("matched_key", k)
                    return f"# {v['title']}\n\n{v['content']}"

            available = ", ".join(list(data.keys())[:5])
            span.set_attribute("error", "section_not_found")
            return (
                f"Error: Section '{section_key}' not found. "
                f"Available sections include: {available}... "
                f"(call list_podcast_info_sections for the full list)"
            )

        except Exception as exc:
            span.record_exception(exc)
            span.set_attribute("error", str(exc))
            return f"Error loading section: {exc}"


# ─── entrypoint ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Starting HistoriCon MCP server on http://0.0.0.0:8001/mcp")
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8001)

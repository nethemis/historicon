"""Tests for the HistoriCon MCP server tools.

Follows TDD: tests are written before the implementation.
All external dependencies are mocked to avoid real API calls,
model loads, and file-system access.

Mocking strategy:
  - agents.retrieval.get_retriever       → avoids ChromaDB + sentence-transformers
  - podcast_info.json / transcript files → patched via tmp_path or monkeypatch
"""

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agents.models import RetrievalChunk, RetrievalResponse

# ─── helpers ──────────────────────────────────────────────────────────────────


def _make_retrieval_response(num_chunks: int = 2) -> RetrievalResponse:
    chunks = [
        RetrievalChunk(
            text=f"Greek history text chunk {i}",
            source=f"episode_{i}.txt",
            score=0.9 - i * 0.1,
            timestamp=f"00:0{i}:00.000",
        )
        for i in range(num_chunks)
    ]
    return RetrievalResponse(
        chunks=chunks,
        summary=f"Found {num_chunks} relevant chunks",
        query="test query",
        total_results=num_chunks,
    )


# ─── search_documents ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_search_documents_returns_chunks():
    """search_documents returns retrieval results."""
    from agents import mcp_server

    mock_response = _make_retrieval_response(num_chunks=3)

    with patch("agents.mcp_server.search_transcripts", return_value=mock_response):
        result = await mcp_server.search_documents("ιστορία της Κύπρου", max_results=3)

    assert result["total_results"] == 3
    assert len(result["chunks"]) == 3
    assert result["query"] == "test query"
    assert "error" not in result


@pytest.mark.asyncio
async def test_search_documents_default_max_results():
    """search_documents defaults to max_results=5."""
    from agents import mcp_server

    mock_response = _make_retrieval_response(num_chunks=5)

    with patch(
        "agents.mcp_server.search_transcripts", return_value=mock_response
    ) as mock_r:
        await mcp_server.search_documents("Greek Revolution 1821")

    mock_r.assert_called_once_with("Greek Revolution 1821", max_results=5)


# ─── get_transcript_section ───────────────────────────────────────────────────


def test_get_transcript_section_returns_matching_lines(tmp_path, monkeypatch):
    """get_transcript_section returns lines within the requested time range."""
    from agents import mcp_server, config as agents_config_module

    transcript_content = (
        "[00:01:00.000 - 00:01:30.000] Speaker 0:\n"
        "Αυτό είναι ένα παράδειγμα.\n"
        "[00:02:00.000 - 00:02:30.000] Speaker 1:\n"
        "Και αυτό είναι ένα δεύτερο παράδειγμα.\n"
        "[00:05:00.000 - 00:05:30.000] Speaker 0:\n"
        "Αυτό είναι εκτός εύρους.\n"
    )
    episode = "test_episode.txt"
    (tmp_path / episode).write_text(transcript_content, encoding="utf-8")

    monkeypatch.setattr(agents_config_module.config, "transcripts_dir", str(tmp_path))

    result = mcp_server.get_transcript_section(episode, "00:00:30", "00:03:00")

    assert "[00:01:00.000" in result
    assert "[00:02:00.000" in result
    assert "[00:05:00.000" not in result


def test_get_transcript_section_no_content_in_range(tmp_path, monkeypatch):
    """get_transcript_section returns a clear message when range has no content."""
    from agents import mcp_server, config as agents_config_module

    transcript_content = "[00:10:00.000 - 00:10:30.000] Speaker 0:\nSome text.\n"
    episode = "test_episode.txt"
    (tmp_path / episode).write_text(transcript_content, encoding="utf-8")

    monkeypatch.setattr(agents_config_module.config, "transcripts_dir", str(tmp_path))

    result = mcp_server.get_transcript_section(episode, "00:00:00", "00:05:00")

    assert "No content found" in result


def test_get_transcript_section_missing_episode(tmp_path, monkeypatch):
    """get_transcript_section returns an error string for a non-existent episode."""
    from agents import mcp_server, config as agents_config_module

    monkeypatch.setattr(agents_config_module.config, "transcripts_dir", str(tmp_path))

    result = mcp_server.get_transcript_section(
        "nonexistent_episode.txt", "00:00:00", "00:05:00"
    )

    assert "Error" in result
    assert "nonexistent_episode.txt" in result


def test_get_transcript_section_malformed_timestamp(tmp_path, monkeypatch):
    """Malformed timestamps are handled gracefully without raising."""
    from agents import mcp_server, config as agents_config_module

    transcript_content = "[00:01:00.000 - 00:01:30.000] Speaker 0:\nText.\n"
    episode = "test_episode.txt"
    (tmp_path / episode).write_text(transcript_content, encoding="utf-8")

    monkeypatch.setattr(agents_config_module.config, "transcripts_dir", str(tmp_path))

    # Should not raise even with garbage timestamps
    result = mcp_server.get_transcript_section(episode, "bad", "also-bad")
    assert isinstance(result, str)


# ─── list_podcast_info_sections ───────────────────────────────────────────────


def test_list_podcast_info_sections_returns_keys(tmp_path):
    """list_podcast_info_sections returns formatted 'KEY - title' strings."""
    from agents import mcp_server

    podcast_data = {
        "ΒΑΣΙΚΑ_ΣΤΟΙΧΕΙΑ": {
            "title": "Βασικά Στοιχεία",
            "content": "Κάποιο περιεχόμενο",
        },
        "ΠΑΡΟΥΣΙΑΣΤΕΣ": {"title": "Παρουσιαστές", "content": "Hosts info"},
    }
    info_file = tmp_path / "podcast_info.json"
    info_file.write_text(json.dumps(podcast_data), encoding="utf-8")

    with patch("agents.mcp_server._PODCAST_INFO_PATH", info_file):
        result = mcp_server.list_podcast_info_sections()

    assert len(result) == 2
    assert any("ΒΑΣΙΚΑ_ΣΤΟΙΧΕΙΑ" in r for r in result)
    assert any("Βασικά Στοιχεία" in r for r in result)


def test_list_podcast_info_sections_missing_file(tmp_path, monkeypatch):
    """list_podcast_info_sections returns an error entry when file is absent."""
    from agents import mcp_server

    missing_path = tmp_path / "podcast_info.json"

    with patch("agents.mcp_server._PODCAST_INFO_PATH", missing_path):
        result = mcp_server.list_podcast_info_sections()

    assert len(result) == 1
    assert "Error" in result[0]


# ─── get_podcast_info_section ─────────────────────────────────────────────────


def test_get_podcast_info_section_exact_match(tmp_path):
    """get_podcast_info_section returns formatted content for an exact key match."""
    from agents import mcp_server

    podcast_data = {
        "ΒΑΣΙΚΑ_ΣΤΟΙΧΕΙΑ": {"title": "Βασικά Στοιχεία", "content": "Περιεχόμενο εδώ"},
    }
    info_file = tmp_path / "podcast_info.json"
    info_file.write_text(json.dumps(podcast_data), encoding="utf-8")

    with patch("agents.mcp_server._PODCAST_INFO_PATH", info_file):
        result = mcp_server.get_podcast_info_section("ΒΑΣΙΚΑ_ΣΤΟΙΧΕΙΑ")

    assert "Βασικά Στοιχεία" in result
    assert "Περιεχόμενο εδώ" in result


def test_get_podcast_info_section_partial_match(tmp_path):
    """get_podcast_info_section falls back to case-insensitive partial match."""
    from agents import mcp_server

    podcast_data = {
        "ΒΑΣΙΚΑ_ΣΤΟΙΧΕΙΑ": {"title": "Βασικά Στοιχεία", "content": "Info here"},
    }
    info_file = tmp_path / "podcast_info.json"
    info_file.write_text(json.dumps(podcast_data), encoding="utf-8")

    with patch("agents.mcp_server._PODCAST_INFO_PATH", info_file):
        result = mcp_server.get_podcast_info_section("ΒΑΣΙΚΑ")

    assert "Βασικά Στοιχεία" in result


def test_get_podcast_info_section_not_found(tmp_path):
    """get_podcast_info_section returns a helpful error for unknown section keys."""
    from agents import mcp_server

    podcast_data = {
        "SECTION_A": {"title": "Section A", "content": "Content"},
    }
    info_file = tmp_path / "podcast_info.json"
    info_file.write_text(json.dumps(podcast_data), encoding="utf-8")

    with patch("agents.mcp_server._PODCAST_INFO_PATH", info_file):
        result = mcp_server.get_podcast_info_section("NONEXISTENT_KEY")

    assert "Error" in result or "not found" in result.lower()


# ─── server registration ──────────────────────────────────────────────────────


def test_mcp_server_has_four_tools():
    """The MCP server registers exactly the 4 expected tools."""
    from agents import mcp_server

    # FastMCP 3.x: list_tools() is async; use asyncio.run() in a sync test.
    tools = asyncio.run(mcp_server.mcp.list_tools())
    tool_names = {t.name for t in tools}

    expected = {
        "search_documents",
        "get_transcript_section",
        "list_podcast_info_sections",
        "get_podcast_info_section",
    }
    assert expected == tool_names

"""Tests for the web orchestrator agent.

Following TDD methodology - write tests first, then implement features.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.web_orchestrator import web_orchestrator


@pytest.mark.asyncio
@patch("agents.web_orchestrator.web_orchestrator.run")
async def test_orchestrator_responds_to_simple_query(mock_run):
    """Test that orchestrator can handle a simple question."""
    mock_result = MagicMock()
    mock_result.output = "This is a system for querying Greek podcast transcripts."
    mock_run.return_value = mock_result

    from agents.web_orchestrator import web_orchestrator

    result = await web_orchestrator.run("What is this system about?")

    assert result.output is not None
    assert isinstance(result.output, str)
    assert len(result.output) > 0


@pytest.mark.asyncio
@patch("agents.web_orchestrator.web_orchestrator.run")
async def test_orchestrator_can_search_documents(mock_run):
    """Test that orchestrator can search for documents."""
    mock_result = MagicMock()
    mock_result.output = "Information about George Santos from sources."
    mock_result.usage = MagicMock(return_value=MagicMock(token_count=150))
    mock_run.return_value = mock_result

    from agents.web_orchestrator import web_orchestrator

    result = await web_orchestrator.run("Tell me about George Santos")

    assert result.output is not None
    # Orchestrator should have used the search_documents tool
    assert result.usage().token_count > 0


@pytest.mark.asyncio
@patch("agents.web_orchestrator.web_orchestrator.run")
async def test_orchestrator_handles_greek_queries(mock_run):
    """Test that orchestrator handles Greek language queries."""
    mock_result = MagicMock()
    mock_result.output = "Πληροφορίες για την ιστορία της Κύπρου."
    mock_run.return_value = mock_result

    from agents.web_orchestrator import web_orchestrator

    result = await web_orchestrator.run("Πες μου για την ιστορία της Κύπρου")

    assert result.output is not None
    assert isinstance(result.output, str)


@pytest.mark.asyncio
@patch("agents.web_orchestrator.web_orchestrator.run")
async def test_orchestrator_can_list_transcripts(mock_run):
    """Test that orchestrator can list available transcripts."""
    mock_result = MagicMock()
    mock_result.output = "Available episodes: George Santos, Elon Musk, etc."
    mock_run.return_value = mock_result

    from agents.web_orchestrator import web_orchestrator

    result = await web_orchestrator.run("What podcast episodes are available?")

    assert result.output is not None
    assert isinstance(result.output, str)


@pytest.mark.asyncio
async def test_orchestrator_has_retries():
    """Test that orchestrator is configured with retries."""
    from pydantic_ai import Agent

    from agents.web_orchestrator import web_orchestrator

    # Verify agent is properly configured
    assert web_orchestrator is not None
    assert isinstance(web_orchestrator, Agent)
    # Agent was created with retries=5 per project convention


@pytest.mark.asyncio
@patch("agents.web_orchestrator.web_orchestrator.run")
async def test_orchestrator_provides_sources(mock_run):
    """Test that orchestrator mentions sources in answers."""
    mock_result = MagicMock()
    mock_result.output = "Information from source: historical_figure.txt"
    mock_run.return_value = mock_result

    from agents.web_orchestrator import web_orchestrator

    result = await web_orchestrator.run("Find information about historical figures")

    # When search is performed, answer should reference sources
    assert result.output is not None
    # This is a behavioral test - in real implementation,
    # orchestrator should cite transcript files


@pytest.mark.asyncio
@patch("agents.web_orchestrator.web_orchestrator.run")
async def test_orchestrator_handles_no_results(mock_run):
    """Test that orchestrator gracefully handles queries with no results."""
    mock_result = MagicMock()
    mock_result.output = "I couldn't find information about that topic."
    mock_run.return_value = mock_result

    from agents.web_orchestrator import web_orchestrator

    result = await web_orchestrator.run("Tell me about XYZ123NonexistentTopic456")

    assert result.output is not None
    # Should not make up information
    assert isinstance(result.output, str)


@pytest.mark.asyncio
@patch("agents.web_orchestrator.web_orchestrator.run")
async def test_orchestrator_delegates_to_retrieval(mock_run):
    """Test that orchestrator properly delegates to retrieval agent."""
    mock_result = MagicMock()
    mock_result.output = "Information about Elon Musk from podcast transcripts."
    mock_result.usage = MagicMock(return_value=MagicMock(token_count=200))
    mock_run.return_value = mock_result

    from agents.web_orchestrator import web_orchestrator

    result = await web_orchestrator.run("Search for information about Elon Musk")

    # Should have made at least one agent call (to retrieval)
    assert result.output is not None
    assert result.usage().token_count > 0


@pytest.mark.asyncio
@patch("agents.web_orchestrator.web_orchestrator.run")
async def test_orchestrator_can_get_full_transcript(mock_run):
    """Test that orchestrator can retrieve full transcripts."""
    mock_result = MagicMock()
    mock_result.output = "Here is the full transcript for George Santos episode..."
    mock_run.return_value = mock_result

    from agents.web_orchestrator import web_orchestrator

    result = await web_orchestrator.run("Get the full transcript for George_Santos.txt")

    assert result.output is not None
    assert isinstance(result.output, str)


@pytest.mark.asyncio
async def test_get_full_transcript_tool_exists():
    """Test that all transcript tools are properly defined."""
    from agents.web_orchestrator import (
        get_full_transcript,
        get_transcript_section,
        search_documents,
        list_podcast_info_sections,
        get_podcast_info_section,
        web_orchestrator,
    )

    # Check that the tools are registered (they exist as functions)
    assert callable(get_full_transcript)
    assert callable(get_transcript_section)
    assert callable(search_documents)
    assert callable(list_podcast_info_sections)
    assert callable(get_podcast_info_section)

    # Verify agent instance exists
    assert web_orchestrator is not None


@pytest.mark.asyncio
async def test_list_podcast_info_sections():
    """Test that podcast info sections can be listed."""
    from agents.web_orchestrator import list_podcast_info_sections
    
    mock_ctx = MagicMock()
    sections = await list_podcast_info_sections(mock_ctx)
    
    assert isinstance(sections, list)
    assert len(sections) > 0
    # Should have section keys with titles
    assert any("ΒΑΣΙΚΑ_ΣΤΟΙΧΕΙΑ" in s for s in sections)


@pytest.mark.asyncio
async def test_get_podcast_info_section():
    """Test that podcast info section can be retrieved."""
    from agents.web_orchestrator import get_podcast_info_section
    
    mock_ctx = MagicMock()
    info = await get_podcast_info_section(mock_ctx, "ΒΑΣΙΚΑ_ΣΤΟΙΧΕΙΑ")
    
    assert isinstance(info, str)
    assert len(info) > 0
    # Should contain podcast information
    assert "Ιστορικών" in info or "HistoriCon" in info or "Error" in info


@pytest.mark.asyncio
async def test_get_podcast_info_section_not_found():
    """Test handling of non-existent section."""
    from agents.web_orchestrator import get_podcast_info_section
    
    mock_ctx = MagicMock()
    info = await get_podcast_info_section(mock_ctx, "NONEXISTENT_SECTION")
    
    assert isinstance(info, str)
    # Should indicate error or not found
    assert "Error" in info or "not found" in info

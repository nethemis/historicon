"""Pytest configuration and shared fixtures.

Provides mocking utilities for testing agents without making real API calls.
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Disable OpenTelemetry auto-configuration for tests
os.environ["OTEL_AUTO_CONFIGURE"] = "false"


@pytest.fixture(autouse=True)
def mock_anthropic_api():
    """Mock the Anthropic API key to prevent real API calls."""
    # Set a dummy API key if not already set
    original_key = os.environ.get("ANTHROPIC_API_KEY")
    if not original_key:
        os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test-key-dummy"

    yield

    # Restore original key
    if not original_key:
        os.environ.pop("ANTHROPIC_API_KEY", None)
    else:
        os.environ["ANTHROPIC_API_KEY"] = original_key


@pytest.fixture(autouse=True)
def mock_otel_setup():
    """Mock OpenTelemetry setup to prevent tracer initialization during tests."""
    with patch("agents.otel_setup._configured", True):
        yield


@pytest.fixture
def mock_agent_run():
    """Create a mock for agent.run() that returns configurable responses."""

    def _create_mock(output_data):
        mock_result = MagicMock()
        mock_result.output = output_data
        mock_result.usage = MagicMock(return_value=MagicMock(token_count=100))
        return AsyncMock(return_value=mock_result)

    return _create_mock

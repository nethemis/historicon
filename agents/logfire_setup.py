"""Logfire configuration for observability across all agents.

Configures Logfire for tracking agent execution, tool calls, and performance.
"""

import os
import sys
from pathlib import Path

import logfire
from logfire import ConsoleOptions

# Use tomllib (Python 3.11+) or fall back to tomli
if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


_configured = False


def configure_logfire() -> None:
    """Configure Logfire with project-specific settings.

    Idempotent and respects ``LOGFIRE_AUTO_CONFIGURE`` (default true). Safe to
    call from any module's import-time path — repeat calls are no-ops, and
    tests can opt out by setting ``LOGFIRE_AUTO_CONFIGURE=false``.
    """
    global _configured
    if _configured:
        return
    if os.getenv("LOGFIRE_AUTO_CONFIGURE", "true").lower() != "true":
        _configured = True
        return

    service_name = os.getenv("LOGFIRE_SERVICE_NAME", "historicon-rag-agent")
    environment = os.getenv("ENVIRONMENT", "development")

    # Check if authenticated by looking for logfire config file
    logfire_config = Path.home() / ".logfire" / "default.toml"
    has_cloud = logfire_config.exists()

    # Configure logfire
    # Logfire will automatically detect credentials from ~/.logfire/default.toml
    logfire.configure(
        service_name=service_name,
        environment=environment,
        send_to_logfire="if-token-present",  # Send to cloud if authenticated
        console=ConsoleOptions(
            colors="auto", verbose=True
        ),  # Always show console output
    )

    # Instrument pydantic_ai for automatic agent tracking
    logfire.instrument_pydantic()

    mode = "cloud + console" if has_cloud else "console only"
    logfire.info(
        "Logfire configured successfully",
        service_name=service_name,
        environment=environment,
        mode=mode,
    )
    print(f"✅ Logfire configured: {mode}")

    # Print dashboard link
    if has_cloud:
        # Read logfire config to get the actual project URL
        try:
            config_path = Path.home() / ".logfire" / "default.toml"
            with open(config_path, "rb") as f:
                config_data = tomllib.load(f)
                base_url = config_data.get("base_url", "https://logfire.pydantic.dev")
                # Try to get organization from config
                org = config_data.get("organization")
                if org and service_name:
                    dashboard_url = f"{base_url}/{org}/{service_name}"
                else:
                    dashboard_url = base_url
        except Exception:
            # Fallback to main dashboard
            dashboard_url = "https://logfire.pydantic.dev/"

        print(f"📊 Dashboard: {dashboard_url}")
        if dashboard_url.endswith("/"):
            print(f"   → Project '{service_name}' will appear after first agent run")
    else:
        print("📊 Dashboard: Run 'uv run logfire auth' to enable cloud dashboard")
        print("   Or view logs in console above ⬆️")

    _configured = True


# Auto-configure on import; the env-var guard lives inside configure_logfire().
configure_logfire()

"""OpenTelemetry configuration for observability across all HistoriCon components.

Configures OpenTelemetry SDK with OTLP exporter for Jaeger, plus console
exporter for development debugging. Idempotent and respects
``OTEL_AUTO_CONFIGURE`` (default true).

Usage:
    from agents import otel_setup  # Auto-configures on import

Or in tests:
    OTEL_AUTO_CONFIGURE=false uv run pytest tests/
"""

import os
from contextlib import contextmanager
from typing import Optional

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)

# Module-level state
_configured = False
_tracer_provider: Optional[TracerProvider] = None


def configure_otel(
    service_name: Optional[str] = None,
    environment: Optional[str] = None,
    otlp_endpoint: Optional[str] = None,
) -> None:
    """Configure OpenTelemetry with OTLP exporter for Jaeger.

    Idempotent and safe to call from any module's import-time path.
    Tests can opt out by setting ``OTEL_AUTO_CONFIGURE=false``.

    Args:
        service_name: Override for service.name resource attribute.
        environment: Override for deployment.environment (default: development).
        otlp_endpoint: Override for OTLP HTTP exporter endpoint.
    """
    global _configured, _tracer_provider
    if _configured:
        return

    if os.getenv("OTEL_AUTO_CONFIGURE", "true").lower() != "true":
        _configured = True
        return

    # Defaults
    service_name = service_name or os.getenv(
        "OTEL_SERVICE_NAME", "historicon-rag-agent"
    )
    environment = environment or os.getenv("ENVIRONMENT", "development")
    otlp_endpoint = otlp_endpoint or os.getenv(
        "OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318"
    )

    # Create resource with service metadata
    resource = Resource.create(
        {
            "service.name": service_name,
            "deployment.environment": environment,
        }
    )

    # Create tracer provider
    _tracer_provider = TracerProvider(resource=resource)

    # Add OTLP exporter for Jaeger
    otlp_exporter = OTLPSpanExporter(endpoint=f"{otlp_endpoint}/v1/traces")
    _tracer_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

    # Add console exporter for development (verbose output)
    if environment == "development":
        console_exporter = ConsoleSpanExporter()
        _tracer_provider.add_span_processor(SimpleSpanProcessor(console_exporter))

    # Set as global tracer provider
    trace.set_tracer_provider(_tracer_provider)

    # Log configuration (using print since logger isn't configured yet)
    mode = "OTLP + console" if environment == "development" else "OTLP only"
    print(f"✅ OpenTelemetry configured: {mode}")
    print(f"   Service: {service_name}")
    print(f"   Environment: {environment}")
    print(f"   OTLP Endpoint: {otlp_endpoint}")
    print(f"   Jaeger UI: http://localhost:16686")

    _configured = True


def get_tracer(instrumentation_name: str) -> trace.Tracer:
    """Get a tracer for the specified instrumentation.

    Args:
        instrumentation_name: Name of the instrumented component (e.g., 'mcp_server').

    Returns:
        OpenTelemetry Tracer instance.
    """
    if not _configured:
        configure_otel()
    return trace.get_tracer(instrumentation_name)


@contextmanager
def trace_span(name: str, **attributes):
    """Context manager for creating spans with attributes.

    Usage:
        with trace_span("operation_name", key="value"):
            # do work

    Args:
        name: Span name.
        **attributes: Key-value pairs to add as span attributes.
    """
    tracer = get_tracer("historicon.utils")
    with tracer.start_as_current_span(name) as span:
        for key, value in attributes.items():
            span.set_attribute(key, str(value))
        yield span


# Auto-configure on import
configure_otel()

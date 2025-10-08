"""OpenTelemetry tracing configuration for agent monitoring.

This module provides comprehensive tracing for:
- Agent execution flow
- LLM calls (reasoning, argument generation, etc.)
- Tool executions with arguments and results
- Performance metrics
"""

from typing import Optional, Dict, Any, Callable
from functools import wraps
import os
from loguru import logger

# OpenTelemetry imports (optional dependency)
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.trace import Status, StatusCode, SpanKind
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    logger.warning("OpenTelemetry not installed. Install with: pip install opentelemetry-api opentelemetry-sdk")


class TelemetryConfig:
    """Configuration for OpenTelemetry tracing."""

    def __init__(
        self,
        service_name: str = "reasoning-agent",
        exporter_type: str = "console",  # console, otlp, jaeger
        otlp_endpoint: Optional[str] = None,
        jaeger_endpoint: Optional[str] = None,
        enabled: bool = True
    ):
        """Initialize telemetry configuration.

        Args:
            service_name: Name of the service for tracing
            exporter_type: Type of exporter (console, otlp, jaeger)
            otlp_endpoint: OTLP endpoint URL (e.g., "http://localhost:4317")
            jaeger_endpoint: Jaeger endpoint (e.g., "localhost", port 6831)
            enabled: Whether tracing is enabled
        """
        self.service_name = service_name
        self.exporter_type = exporter_type
        self.otlp_endpoint = otlp_endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
        self.jaeger_endpoint = jaeger_endpoint or os.getenv("JAEGER_AGENT_HOST", "localhost")
        self.enabled = enabled and OTEL_AVAILABLE

        if not OTEL_AVAILABLE and enabled:
            logger.warning("Telemetry requested but OpenTelemetry not available")


def setup_telemetry(config: TelemetryConfig) -> Optional[trace.Tracer]:
    """Setup OpenTelemetry tracing.

    Args:
        config: Telemetry configuration

    Returns:
        Tracer instance if enabled, None otherwise
    """
    if not config.enabled or not OTEL_AVAILABLE:
        return None

    try:
        # Create resource with service name
        resource = Resource(attributes={
            SERVICE_NAME: config.service_name
        })

        # Create tracer provider
        provider = TracerProvider(resource=resource)

        # Configure exporter
        if config.exporter_type == "console":
            exporter = ConsoleSpanExporter()
            logger.info("[TELEMETRY] Using console exporter")
        elif config.exporter_type == "otlp":
            exporter = OTLPSpanExporter(endpoint=config.otlp_endpoint)
            logger.info(f"[TELEMETRY] Using OTLP exporter at {config.otlp_endpoint}")
        elif config.exporter_type == "jaeger":
            exporter = JaegerExporter(
                agent_host_name=config.jaeger_endpoint,
                agent_port=6831
            )
            logger.info(f"[TELEMETRY] Using Jaeger exporter at {config.jaeger_endpoint}")
        else:
            logger.error(f"[TELEMETRY] Unknown exporter type: {config.exporter_type}")
            return None

        # Add span processor
        provider.add_span_processor(BatchSpanProcessor(exporter))

        # Set as global provider
        trace.set_tracer_provider(provider)

        # Get tracer
        tracer = trace.get_tracer(__name__)
        logger.info(f"[TELEMETRY] Tracing initialized for service '{config.service_name}'")

        return tracer

    except Exception as e:
        logger.error(f"[TELEMETRY] Failed to setup telemetry: {e}")
        return None


class AgentTracer:
    """Tracer wrapper for agent operations."""

    def __init__(self, tracer: Optional[trace.Tracer] = None):
        """Initialize agent tracer.

        Args:
            tracer: OpenTelemetry tracer instance
        """
        self.tracer = tracer
        self.enabled = tracer is not None and OTEL_AVAILABLE

    def trace_agent_run(
        self,
        user_input: str,
        agent_type: str = "ReasoningAgent"
    ) -> Any:
        """Create a span for agent execution.

        Args:
            user_input: User's input query
            agent_type: Type of agent

        Returns:
            Span context manager
        """
        if not self.enabled:
            from contextlib import nullcontext
            return nullcontext()

        return self.tracer.start_as_current_span(
            "agent.run",
            kind=SpanKind.SERVER,
            attributes={
                "agent.type": agent_type,
                "agent.input": user_input[:500],  # Limit input size
            }
        )

    def trace_reasoning_phase(
        self,
        input_text: str,
        iteration: int
    ) -> Any:
        """Create a span for reasoning phase.

        Args:
            input_text: Input to reasoning phase
            iteration: Current iteration number

        Returns:
            Span context manager
        """
        if not self.enabled:
            from contextlib import nullcontext
            return nullcontext()

        return self.tracer.start_as_current_span(
            "agent.reasoning",
            attributes={
                "agent.reasoning.input": input_text[:500],
                "agent.iteration": iteration,
            }
        )

    def trace_llm_call(
        self,
        prompt: str,
        model: str,
        call_type: str = "completion"
    ) -> Any:
        """Create a span for LLM call.

        Args:
            prompt: Prompt sent to LLM
            model: Model name
            call_type: Type of call (reasoning, tool_args, generate)

        Returns:
            Span context manager
        """
        if not self.enabled:
            from contextlib import nullcontext
            return nullcontext()

        return self.tracer.start_as_current_span(
            f"llm.{call_type}",
            attributes={
                "llm.model": model,
                "llm.prompt": prompt[:1000],  # Limit prompt size
                "llm.call_type": call_type,
            }
        )

    def trace_tool_execution(
        self,
        tool_name: str,
        tool_args: Dict[str, Any]
    ) -> Any:
        """Create a span for tool execution.

        Args:
            tool_name: Name of the tool
            tool_args: Arguments passed to tool

        Returns:
            Span context manager
        """
        if not self.enabled:
            from contextlib import nullcontext
            return nullcontext()

        return self.tracer.start_as_current_span(
            f"tool.{tool_name}",
            attributes={
                "tool.name": tool_name,
                "tool.args": str(tool_args)[:500],  # Limit args size
            }
        )

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Add an event to the current span.

        Args:
            name: Event name
            attributes: Event attributes
        """
        if not self.enabled:
            return

        span = trace.get_current_span()
        if span:
            span.add_event(name, attributes=attributes or {})

    def set_attribute(self, key: str, value: Any):
        """Set an attribute on the current span.

        Args:
            key: Attribute key
            value: Attribute value
        """
        if not self.enabled:
            return

        span = trace.get_current_span()
        if span:
            span.set_attribute(key, value)

    def record_exception(self, exception: Exception):
        """Record an exception in the current span.

        Args:
            exception: Exception to record
        """
        if not self.enabled:
            return

        span = trace.get_current_span()
        if span:
            span.record_exception(exception)
            span.set_status(Status(StatusCode.ERROR, str(exception)))

    def set_status(self, status_code: str, description: str = ""):
        """Set the status of the current span.

        Args:
            status_code: Status code (OK, ERROR)
            description: Status description
        """
        if not self.enabled:
            return

        span = trace.get_current_span()
        if span:
            if status_code == "OK":
                span.set_status(Status(StatusCode.OK, description))
            elif status_code == "ERROR":
                span.set_status(Status(StatusCode.ERROR, description))


# Global tracer instance
_global_tracer: Optional[AgentTracer] = None


def get_tracer() -> AgentTracer:
    """Get the global agent tracer.

    Returns:
        AgentTracer instance
    """
    global _global_tracer
    if _global_tracer is None:
        _global_tracer = AgentTracer()
    return _global_tracer


def initialize_telemetry(
    service_name: str = "reasoning-agent",
    exporter_type: str = "console",
    otlp_endpoint: Optional[str] = None,
    jaeger_endpoint: Optional[str] = None,
    enabled: bool = True
) -> AgentTracer:
    """Initialize global telemetry.

    Args:
        service_name: Name of the service
        exporter_type: Type of exporter (console, otlp, jaeger)
        otlp_endpoint: OTLP endpoint URL
        jaeger_endpoint: Jaeger endpoint
        enabled: Whether to enable tracing

    Returns:
        AgentTracer instance
    """
    global _global_tracer

    config = TelemetryConfig(
        service_name=service_name,
        exporter_type=exporter_type,
        otlp_endpoint=otlp_endpoint,
        jaeger_endpoint=jaeger_endpoint,
        enabled=enabled
    )

    tracer = setup_telemetry(config)
    _global_tracer = AgentTracer(tracer)

    return _global_tracer


def is_telemetry_available() -> bool:
    """Check if OpenTelemetry is available.

    Returns:
        True if available, False otherwise
    """
    return OTEL_AVAILABLE


def trace_method(span_name: str, **span_attributes):
    """Decorator to trace a method.

    Args:
        span_name: Name of the span
        **span_attributes: Static attributes to add to span

    Usage:
        @trace_method("agent.reasoning", agent_type="ReasoningAgent")
        def _reasoning_call(self, input_text: str):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            tracer = getattr(self, 'tracer', get_tracer())
            if not tracer.enabled:
                return func(self, *args, **kwargs)

            # Create span
            if OTEL_AVAILABLE:
                with tracer.tracer.start_as_current_span(span_name) as span:
                    # Add static attributes
                    for key, value in span_attributes.items():
                        span.set_attribute(key, value)

                    # Execute function
                    try:
                        result = func(self, *args, **kwargs)
                        span.set_status(Status(StatusCode.OK))
                        return result
                    except Exception as e:
                        span.record_exception(e)
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        raise
            else:
                return func(self, *args, **kwargs)

        return wrapper
    return decorator

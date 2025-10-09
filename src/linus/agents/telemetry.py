"""OpenTelemetry tracing configuration for agent monitoring.

This module provides comprehensive tracing for:
- Agent execution flow
- LLM calls (reasoning, argument generation, etc.)
- Tool executions with arguments and results
- Performance metrics
"""

from typing import Optional, Dict, Any, Callable, TYPE_CHECKING
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
    trace = None  # Set to None when not available
    logger.warning("OpenTelemetry not installed. Install with: pip install opentelemetry-api opentelemetry-sdk")

# Langfuse imports (optional dependency)
try:
    from langfuse import Langfuse
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    logger.warning("Langfuse not installed. Install with: pip install langfuse")


class TelemetryConfig:
    """Configuration for telemetry (OpenTelemetry and Langfuse)."""

    def __init__(
        self,
        service_name: str = "reasoning-agent",
        exporter_type: str = "console",  # console, otlp, jaeger, langfuse
        otlp_endpoint: Optional[str] = None,
        jaeger_endpoint: Optional[str] = None,
        langfuse_public_key: Optional[str] = None,
        langfuse_secret_key: Optional[str] = None,
        langfuse_host: Optional[str] = None,
        enabled: bool = True
    ):
        """Initialize telemetry configuration.

        Args:
            service_name: Name of the service for tracing
            exporter_type: Type of exporter (console, otlp, jaeger, langfuse)
            otlp_endpoint: OTLP endpoint URL (e.g., "http://localhost:4317")
            jaeger_endpoint: Jaeger endpoint (e.g., "localhost", port 6831)
            langfuse_public_key: Langfuse public API key
            langfuse_secret_key: Langfuse secret API key
            langfuse_host: Langfuse host URL (default: "https://cloud.langfuse.com")
            enabled: Whether tracing is enabled
        """
        self.service_name = service_name
        self.exporter_type = exporter_type
        self.otlp_endpoint = otlp_endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
        self.jaeger_endpoint = jaeger_endpoint or os.getenv("JAEGER_AGENT_HOST", "localhost")

        # Langfuse configuration
        self.langfuse_public_key = langfuse_public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
        self.langfuse_secret_key = langfuse_secret_key or os.getenv("LANGFUSE_SECRET_KEY")
        self.langfuse_host = langfuse_host or os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

        # Check if the selected exporter is available
        if exporter_type == "langfuse":
            self.enabled = enabled and LANGFUSE_AVAILABLE
            if not LANGFUSE_AVAILABLE and enabled:
                logger.warning("Langfuse requested but not available. Install with: pip install langfuse")
        else:
            self.enabled = enabled and OTEL_AVAILABLE
            if not OTEL_AVAILABLE and enabled:
                logger.warning("Telemetry requested but OpenTelemetry not available")


def setup_telemetry(config: TelemetryConfig) -> Optional[Any]:
    """Setup tracing (OpenTelemetry or Langfuse).

    Args:
        config: Telemetry configuration

    Returns:
        Tracer instance if enabled, None otherwise
    """
    if not config.enabled:
        return None

    try:
        # Setup Langfuse
        if config.exporter_type == "langfuse":
            if not LANGFUSE_AVAILABLE:
                logger.error("[TELEMETRY] Langfuse requested but not available")
                return None

            if not config.langfuse_public_key or not config.langfuse_secret_key:
                logger.error("[TELEMETRY] Langfuse credentials not provided")
                return None

            # Set environment variables for get_client() to pick up
            import os
            os.environ['LANGFUSE_PUBLIC_KEY'] = config.langfuse_public_key
            os.environ['LANGFUSE_SECRET_KEY'] = config.langfuse_secret_key
            os.environ['LANGFUSE_HOST'] = config.langfuse_host

            # Initialize Langfuse client directly for better compatibility
            langfuse_client = Langfuse(
                public_key=config.langfuse_public_key,
                secret_key=config.langfuse_secret_key,
                host=config.langfuse_host
            )
            logger.info(f"[TELEMETRY] Langfuse client initialized: {type(langfuse_client)}")
            logger.info(f"[TELEMETRY] Langfuse initialized at {config.langfuse_host}")
            logger.info(f"[TELEMETRY] Langfuse public key: {config.langfuse_public_key[:8]}...")

            # Check available methods for debugging
            available_methods = [m for m in dir(langfuse_client) if not m.startswith('_') and callable(getattr(langfuse_client, m))]
            logger.info(f"[TELEMETRY] Available Langfuse methods: {available_methods}")

            return langfuse_client

        # Setup OpenTelemetry
        if not OTEL_AVAILABLE:
            logger.error("[TELEMETRY] OpenTelemetry requested but not available")
            return None

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
        logger.exception(f"[TELEMETRY] Failed to setup telemetry: {e}")
        return None


class LangfuseTracer:
    """Tracer wrapper for Langfuse observability."""

    def __init__(self, langfuse_client: Optional[Any] = None, session_id: Optional[str] = None):
        """Initialize Langfuse tracer.

        Args:
            langfuse_client: Langfuse client instance
            session_id: Session ID for grouping related traces
        """
        self.client = langfuse_client
        self.enabled = langfuse_client is not None and LANGFUSE_AVAILABLE
        self.session_id = session_id

    def trace_agent_run(
        self,
        user_input: str,
        agent_type: str = "ReasoningAgent"
    ) -> Any:
        """Create a trace for agent execution.

        Args:
            user_input: User's input query
            agent_type: Type of agent

        Returns:
            Trace context (AsyncContextManager for async compatibility)
        """
        if not self.enabled:
            from contextlib import nullcontext
            return nullcontext()

        # Create metadata including session_id if available
        metadata = {"agent_type": agent_type}
        if self.session_id:
            metadata["session_id"] = self.session_id

        logger.info(f"[LANGFUSE] Creating trace span: agent_run with input: {user_input[:50]}...")
        logger.info(f"[LANGFUSE] Client type: {type(self.client)}, enabled: {self.enabled}")
        logger.info(f"[LANGFUSE] Metadata: {metadata}")

        try:
            # Use Langfuse's context manager API properly
            from contextlib import asynccontextmanager

            @asynccontextmanager
            async def async_trace_wrapper():
                """Wrap Langfuse context manager for async usage."""
                # Use start_as_current_span to create a trace-level span
                with self.client.start_as_current_span(
                    name="agent_run",
                    input={"query": user_input},
                    metadata=metadata
                ) as span:
                    logger.info(f"[LANGFUSE] Trace span started: {type(span)}")
                    # Yield the span for the context
                    try:
                        yield span
                    finally:
                        logger.info("[LANGFUSE] Trace span ended")

            return async_trace_wrapper()
        except Exception as e:
            logger.exception(f"[LANGFUSE] Error creating trace span: {e}")
            from contextlib import nullcontext
            return nullcontext()

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
            Span context (AsyncContextManager for async compatibility)
        """
        if not self.enabled:
            from contextlib import nullcontext
            return nullcontext()

        # Use Langfuse's nested span API
        logger.debug(f"[LANGFUSE] Creating reasoning span for iteration {iteration}")

        # Create nested span using Langfuse context manager
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def async_span_wrapper():
            """Create reasoning span within current trace."""
            try:
                with self.client.start_as_current_span(
                    name="reasoning_phase",
                    input={"text": input_text, "iteration": iteration},
                    metadata={"iteration": iteration}
                ) as span:
                    logger.debug(f"[LANGFUSE] Reasoning span created: {type(span)}")
                    yield span
            except Exception as e:
                logger.warning(f"[LANGFUSE] Error creating reasoning span: {e}")
                yield None

        return async_span_wrapper()

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
            Generation context (AsyncContextManager for async compatibility)
        """
        if not self.enabled:
            from contextlib import nullcontext
            return nullcontext()

        # Use Langfuse's generation API
        logger.debug(f"[LANGFUSE] Creating generation span: llm_{call_type}")

        # Create generation using Langfuse context manager
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def async_observation_wrapper():
            """Create generation within current trace."""
            try:
                with self.client.start_as_current_observation(
                    as_type='generation',
                    name=f"llm_{call_type}",
                    model=model,
                    input=prompt,
                    metadata={"call_type": call_type}
                ) as generation:
                    logger.debug(f"[LANGFUSE] Generation created: {type(generation)}")
                    yield generation
            except Exception as e:
                logger.warning(f"[LANGFUSE] Error creating generation: {e}")
                yield None

        return async_observation_wrapper()

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
            Span context (AsyncContextManager for async compatibility)
        """
        if not self.enabled:
            from contextlib import nullcontext
            return nullcontext()

        # Use Langfuse's span API
        logger.debug(f"[LANGFUSE] Creating tool execution span: tool_{tool_name}")

        # Create span using Langfuse context manager
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def async_span_wrapper():
            """Create tool execution span within current trace."""
            try:
                with self.client.start_as_current_span(
                    name=f"tool_{tool_name}",
                    input=tool_args,
                    metadata={"tool": tool_name}
                ) as span:
                    logger.debug(f"[LANGFUSE] Tool span created: {type(span)}")
                    yield span
            except Exception as e:
                logger.warning(f"[LANGFUSE] Error creating tool span: {e}")
                yield None

        return async_span_wrapper()

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Add an event to the current trace.

        Args:
            name: Event name
            attributes: Event attributes
        """
        if not self.enabled:
            return

        # Events are logged for debugging - Langfuse doesn't have a direct event API
        logger.debug(f"[LANGFUSE] Event: {name} - {attributes}")

    def set_attribute(self, key: str, value: Any):
        """Set metadata on the current trace/span.

        Args:
            key: Attribute key
            value: Attribute value
        """
        if not self.enabled or not self._current_trace:
            return

        # Update the current trace's metadata
        try:
            # Langfuse doesn't support directly setting attributes on active spans
            # Store them for later update or log them
            logger.debug(f"[LANGFUSE] Attribute set: {key}={value}")
        except Exception as e:
            logger.warning(f"[LANGFUSE] Failed to set attribute: {e}")

    def record_exception(self, exception: Exception):
        """Record an exception in the current trace.

        Args:
            exception: Exception to record
        """
        if not self.enabled:
            return

        # Log exception - will be captured in span/generation metadata if needed
        logger.error(f"[LANGFUSE] Exception: {type(exception).__name__}: {str(exception)}")

    def set_status(self, status_code: str, description: str = ""):
        """Set the status of the current trace.

        Args:
            status_code: Status code (OK, ERROR)
            description: Status description
        """
        if not self.enabled or not self._current_trace:
            return

        # Store status for later - will be used when trace is ended
        logger.debug(f"[LANGFUSE] Status set: {status_code} - {description}")

    def update_generation(self, output: Any, usage: Optional[Dict[str, int]] = None):
        """Update the current LLM generation with output and usage stats.

        Args:
            output: Generated output (can be string or dict)
            usage: Token usage stats (prompt_tokens, completion_tokens, total_tokens)
        """
        if not self.enabled:
            return

        # Update the current generation using Langfuse API
        output_len = len(str(output)) if output else 0
        logger.debug(f"[LANGFUSE] Generation update called with {output_len} chars output")

        try:
            # Use the client's update methods with correct parameter names
            update_kwargs = {"output": output}
            if usage:
                # Langfuse expects usage_details, not usage
                update_kwargs["usage_details"] = usage
                logger.debug(f"[LANGFUSE] Updating with usage: {usage}")

            self.client.update_current_generation(**update_kwargs)
            logger.info(f"[LANGFUSE] Successfully updated generation with output and usage")
        except Exception as e:
            logger.warning(f"[LANGFUSE] Failed to update generation: {e}")

    def record_metrics(self, metrics: Dict[str, Any]):
        """Record metrics on the current trace.

        Args:
            metrics: Dictionary of metrics to record
        """
        if not self.enabled:
            return

        try:
            # Update the current trace with metrics metadata
            self.client.update_current_trace(metadata={"metrics": metrics})
            logger.debug(f"[LANGFUSE] Recorded metrics: {list(metrics.keys())}")
        except Exception as e:
            logger.warning(f"[LANGFUSE] Failed to record metrics: {e}")

    def flush(self):
        """Flush pending traces to Langfuse."""
        if self.enabled and self.client:
            try:
                self.client.flush()
                logger.info("[LANGFUSE] Flushed traces to Langfuse server")
            except Exception as e:
                logger.error(f"[LANGFUSE] Failed to flush traces: {e}")

    def set_attribute(self, key: str, value: Any):
        """Set metadata on the current trace.

        Args:
            key: Attribute key
            value: Attribute value
        """
        if not self.enabled:
            return

        try:
            # Update current trace/span metadata
            metadata = {key: value}
            self.client.update_current_trace(metadata=metadata)
            logger.debug(f"[LANGFUSE] Attribute set: {key}={value}")
        except Exception as e:
            logger.warning(f"[LANGFUSE] Failed to set attribute: {e}")

    def record_exception(self, exception: Exception):
        """Record an exception in the current trace.

        Args:
            exception: Exception to record
        """
        if not self.enabled:
            return

        try:
            # Record error in trace/span
            error_info = {
                "error_type": type(exception).__name__,
                "error_message": str(exception),
                "status": "ERROR"
            }
            self.client.update_current_trace(metadata={"error": error_info})
            logger.error(f"[LANGFUSE] Exception recorded: {type(exception).__name__}: {str(exception)}")
        except Exception as e:
            logger.warning(f"[LANGFUSE] Failed to record exception: {e}")

    def set_status(self, status_code: str, description: str = ""):
        """Set the status of the current trace.

        Args:
            status_code: Status code (OK, ERROR)
            description: Status description
        """
        if not self.enabled:
            return

        try:
            # Update trace with status
            status_info = {"status": status_code, "description": description}
            self.client.update_current_trace(metadata={"status": status_info})
            logger.debug(f"[LANGFUSE] Status set: {status_code} - {description}")
        except Exception as e:
            logger.warning(f"[LANGFUSE] Failed to set status: {e}")


class AgentTracer:
    """Tracer wrapper for agent operations."""

    def __init__(self, tracer: Optional[Any] = None):
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
            Span context manager (AsyncContextManager for async compatibility)
        """
        if not self.enabled:
            from contextlib import nullcontext
            return nullcontext()

        # Wrap OpenTelemetry's sync context manager for async usage
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def async_span_wrapper():
            """Wrap sync OpenTelemetry context manager for async usage."""
            with self.tracer.start_as_current_span(
                "agent.run",
                kind=SpanKind.SERVER,
                attributes={
                    "agent.type": agent_type,
                    "agent.input": user_input,  # No truncation
                }
            ) as span:
                yield span

        return async_span_wrapper()

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
            Span context manager (AsyncContextManager for async compatibility)
        """
        if not self.enabled:
            from contextlib import nullcontext
            return nullcontext()

        # Wrap OpenTelemetry's sync context manager for async usage
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def async_span_wrapper():
            """Wrap sync OpenTelemetry context manager for async usage."""
            with self.tracer.start_as_current_span(
                "agent.reasoning",
                attributes={
                    "agent.reasoning.input": input_text,  # No truncation
                    "agent.iteration": iteration,
                }
            ) as span:
                yield span

        return async_span_wrapper()

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
            Span context manager (AsyncContextManager for async compatibility)
        """
        if not self.enabled:
            from contextlib import nullcontext
            return nullcontext()

        # Wrap OpenTelemetry's sync context manager for async usage
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def async_span_wrapper():
            """Wrap sync OpenTelemetry context manager for async usage."""
            with self.tracer.start_as_current_span(
                f"llm.{call_type}",
                attributes={
                    "llm.model": model,
                    "llm.prompt": prompt,  # No truncation
                    "llm.call_type": call_type,
                }
            ) as span:
                yield span

        return async_span_wrapper()

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
            Span context manager (AsyncContextManager for async compatibility)
        """
        if not self.enabled:
            from contextlib import nullcontext
            return nullcontext()

        # Wrap OpenTelemetry's sync context manager for async usage
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def async_span_wrapper():
            """Wrap sync OpenTelemetry context manager for async usage."""
            with self.tracer.start_as_current_span(
                f"tool.{tool_name}",
                attributes={
                    "tool.name": tool_name,
                    "tool.args": str(tool_args),  # No truncation
                }
            ) as span:
                yield span

        return async_span_wrapper()

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

    def record_metrics(self, metrics: Dict[str, Any]):
        """Record metrics as span attributes.

        Args:
            metrics: Dictionary of metrics to record
        """
        if not self.enabled:
            return

        span = trace.get_current_span()
        if span:
            for key, value in metrics.items():
                # Convert value to a type that OpenTelemetry can handle
                if isinstance(value, (str, bool, int, float)):
                    span.set_attribute(f"agent.metrics.{key}", value)
                else:
                    span.set_attribute(f"agent.metrics.{key}", str(value))


# Global tracer instance (can be either AgentTracer or LangfuseTracer)
_global_tracer: Optional[Any] = None


def get_tracer():
    """Get the global tracer.

    Returns:
        AgentTracer or LangfuseTracer instance
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
    langfuse_public_key: Optional[str] = None,
    langfuse_secret_key: Optional[str] = None,
    langfuse_host: Optional[str] = None,
    session_id: Optional[str] = None,
    enabled: bool = True
):
    """Initialize global telemetry.

    Args:
        service_name: Name of the service
        exporter_type: Type of exporter (console, otlp, jaeger, langfuse)
        otlp_endpoint: OTLP endpoint URL
        jaeger_endpoint: Jaeger endpoint
        langfuse_public_key: Langfuse public API key
        langfuse_secret_key: Langfuse secret API key
        langfuse_host: Langfuse host URL
        session_id: Session ID for grouping related traces (Langfuse only)
        enabled: Whether to enable tracing

    Returns:
        AgentTracer or LangfuseTracer instance
    """
    global _global_tracer

    config = TelemetryConfig(
        service_name=service_name,
        exporter_type=exporter_type,
        otlp_endpoint=otlp_endpoint,
        jaeger_endpoint=jaeger_endpoint,
        langfuse_public_key=langfuse_public_key,
        langfuse_secret_key=langfuse_secret_key,
        langfuse_host=langfuse_host,
        enabled=enabled
    )

    tracer_backend = setup_telemetry(config)

    # Return appropriate tracer based on exporter type
    if config.exporter_type == "langfuse":
        _global_tracer = LangfuseTracer(tracer_backend, session_id=session_id)
    else:
        _global_tracer = AgentTracer(tracer_backend)

    return _global_tracer


def is_telemetry_available() -> bool:
    """Check if OpenTelemetry is available.

    Returns:
        True if available, False otherwise
    """
    return OTEL_AVAILABLE


def is_langfuse_available() -> bool:
    """Check if Langfuse is available.

    Returns:
        True if available, False otherwise
    """
    return LANGFUSE_AVAILABLE


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

            # Create span - handle both AgentTracer (OpenTelemetry) and LangfuseTracer
            if isinstance(tracer, AgentTracer) and OTEL_AVAILABLE and tracer.tracer:
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
                # For LangfuseTracer or when OpenTelemetry is not available,
                # just execute the function without tracing
                return func(self, *args, **kwargs)

        return wrapper
    return decorator

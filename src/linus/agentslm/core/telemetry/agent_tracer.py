from typing import Optional, Dict, Any
from loguru import logger

# OpenTelemetry imports (optional dependency)
try:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode, SpanKind
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    trace = None  # Set to None when not available
    logger.warning("OpenTelemetry not installed. Install with: pip install opentelemetry-api opentelemetry-sdk")
    
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
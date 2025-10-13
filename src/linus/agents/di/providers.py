"""Concrete implementations of logger and telemetry providers."""

from typing import Any, Dict, Optional, AsyncContextManager
from loguru import logger as loguru_logger

from .interfaces import ILogger, ITelemetry


class LoguruLogger(ILogger):
    """Loguru-based logger implementation."""

    def __init__(self, logger_instance=None):
        """Initialize with optional custom logger instance.

        Args:
            logger_instance: Optional loguru logger instance (uses global if None)
        """
        self._logger = logger_instance or loguru_logger

    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self._logger.debug(message, **kwargs)

    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self._logger.info(message, **kwargs)

    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self._logger.warning(message, **kwargs)

    def error(self, message: str, **kwargs) -> None:
        """Log error message."""
        self._logger.error(message, **kwargs)

    def exception(self, message: str, **kwargs) -> None:
        """Log exception with traceback."""
        self._logger.exception(message, **kwargs)


class TelemetryAdapter(ITelemetry):
    """Adapter for existing telemetry tracers (AgentTracer or LangfuseTracer)."""

    def __init__(self, tracer: Any):
        """Initialize with a tracer instance.

        Args:
            tracer: AgentTracer or LangfuseTracer instance
        """
        self._tracer = tracer

    @property
    def enabled(self) -> bool:
        """Check if telemetry is enabled."""
        return self._tracer.enabled if self._tracer else False

    @property
    def client(self):
        """Access underlying tracer client (for Langfuse compatibility)."""
        return getattr(self._tracer, 'client', None)

    def trace_agent_run(
        self,
        user_input: str,
        agent_type: str = "ReasoningAgent"
    ) -> AsyncContextManager:
        """Create a trace for agent execution."""
        if not self._tracer:
            from contextlib import nullcontext
            return nullcontext()
        return self._tracer.trace_agent_run(user_input, agent_type)

    def trace_reasoning_phase(
        self,
        input_text: str,
        iteration: int
    ) -> AsyncContextManager:
        """Create a span for reasoning phase."""
        if not self._tracer:
            from contextlib import nullcontext
            return nullcontext()
        return self._tracer.trace_reasoning_phase(input_text, iteration)

    def trace_llm_call(
        self,
        prompt: str,
        model: str,
        call_type: str = "completion"
    ) -> AsyncContextManager:
        """Create a span for LLM call."""
        if not self._tracer:
            from contextlib import nullcontext
            return nullcontext()
        return self._tracer.trace_llm_call(prompt, model, call_type)

    def trace_tool_execution(
        self,
        tool_name: str,
        tool_args: Dict[str, Any]
    ) -> AsyncContextManager:
        """Create a span for tool execution."""
        if not self._tracer:
            from contextlib import nullcontext
            return nullcontext()
        return self._tracer.trace_tool_execution(tool_name, tool_args)

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """Add an event to the current trace."""
        if self._tracer:
            self._tracer.add_event(name, attributes)

    def set_attribute(self, key: str, value: Any) -> None:
        """Set an attribute on the current trace/span."""
        if self._tracer:
            self._tracer.set_attribute(key, value)

    def record_exception(self, exception: Exception) -> None:
        """Record an exception in the current trace."""
        if self._tracer:
            self._tracer.record_exception(exception)

    def set_status(self, status_code: str, description: str = "") -> None:
        """Set the status of the current trace."""
        if self._tracer:
            self._tracer.set_status(status_code, description)

    def record_metrics(self, metrics: Dict[str, Any]) -> None:
        """Record metrics on the current trace."""
        if self._tracer:
            self._tracer.record_metrics(metrics)

    def flush(self) -> None:
        """Flush pending traces."""
        if self._tracer and hasattr(self._tracer, 'flush'):
            self._tracer.flush()

    def update_generation(self, output: Any, usage: Optional[Dict[str, int]] = None) -> None:
        """Update the current LLM generation (Langfuse-specific)."""
        if self._tracer and hasattr(self._tracer, 'update_generation'):
            self._tracer.update_generation(output, usage)


# Factory functions for creating providers
class LoggerProvider:
    """Factory for creating logger instances."""

    @staticmethod
    def create_loguru_logger(logger_instance=None) -> ILogger:
        """Create a Loguru logger.

        Args:
            logger_instance: Optional custom loguru logger instance

        Returns:
            ILogger implementation using Loguru
        """
        return LoguruLogger(logger_instance)

    @staticmethod
    def create_noop_logger() -> ILogger:
        """Create a no-op logger.

        Returns:
            ILogger implementation that does nothing
        """
        from .interfaces import NoOpLogger
        return NoOpLogger()


class TelemetryProvider:
    """Factory for creating telemetry instances."""

    @staticmethod
    def create_from_tracer(tracer: Any) -> ITelemetry:
        """Create telemetry adapter from existing tracer.

        Args:
            tracer: AgentTracer or LangfuseTracer instance

        Returns:
            ITelemetry implementation wrapping the tracer
        """
        if tracer is None:
            from .interfaces import NoOpTelemetry
            return NoOpTelemetry()
        return TelemetryAdapter(tracer)

    @staticmethod
    def create_noop_telemetry() -> ITelemetry:
        """Create a no-op telemetry.

        Returns:
            ITelemetry implementation that does nothing
        """
        from .interfaces import NoOpTelemetry
        return NoOpTelemetry()

    @staticmethod
    def create_from_config(
        service_name: str = "reasoning-agent",
        exporter_type: str = "console",
        session_id: Optional[str] = None,
        enabled: bool = True,
        **kwargs
    ) -> ITelemetry:
        """Create telemetry from configuration.

        Args:
            service_name: Name of the service
            exporter_type: Type of exporter (console, otlp, jaeger, langfuse)
            session_id: Session ID for grouping traces (Langfuse only)
            enabled: Whether to enable telemetry
            **kwargs: Additional configuration options

        Returns:
            ITelemetry implementation
        """
        if not enabled:
            from .interfaces import NoOpTelemetry
            return NoOpTelemetry()

        # Import telemetry module and initialize
        from ..telemetry import initialize_telemetry

        tracer = initialize_telemetry(
            service_name=service_name,
            exporter_type=exporter_type,
            session_id=session_id,
            enabled=enabled,
            **kwargs
        )

        return TelemetryAdapter(tracer)

"""Interfaces for dependency injection.

These interfaces define the contracts for logging and telemetry services
that can be injected into agents, tools, memory, and MCP clients.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, AsyncContextManager
from contextlib import nullcontext


class ILogger(ABC):
    """Interface for logging service."""

    @abstractmethod
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        pass

    @abstractmethod
    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        pass

    @abstractmethod
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        pass

    @abstractmethod
    def error(self, message: str, **kwargs) -> None:
        """Log error message."""
        pass

    @abstractmethod
    def exception(self, message: str, **kwargs) -> None:
        """Log exception with traceback."""
        pass


class ITelemetry(ABC):
    """Interface for telemetry/tracing service."""

    @property
    @abstractmethod
    def enabled(self) -> bool:
        """Check if telemetry is enabled."""
        pass

    @abstractmethod
    def trace_agent_run(
        self,
        user_input: str,
        agent_type: str = "ReasoningAgent",
        name: Optional[str] = None
    ) -> AsyncContextManager:
        """Create a trace for agent execution.

        Args:
            user_input: User's input query
            agent_type: Type of agent
            name: Optional custom name for the trace (e.g., "agent.ReasoningAgent")

        Returns:
            Async context manager for the trace span
        """
        pass

    @abstractmethod
    def trace_reasoning_phase(
        self,
        input_text: str,
        iteration: int,
        name: Optional[str] = None
    ) -> AsyncContextManager:
        """Create a span for reasoning phase.

        Args:
            input_text: Input to reasoning phase
            iteration: Current iteration number
            name: Optional custom name for the span (e.g., "agent.reasoning")

        Returns:
            Async context manager for the span
        """
        pass

    @abstractmethod
    def trace_llm_call(
        self,
        prompt: str,
        model: str,
        call_type: str = "completion",
        name: Optional[str] = None
    ) -> AsyncContextManager:
        """Create a span for LLM call.

        Args:
            prompt: Prompt sent to LLM
            model: Model name
            call_type: Type of call (reasoning, tool_args, generate)
            name: Optional custom name for the span (e.g., "llm.reasoning")

        Returns:
            Async context manager for the span
        """
        pass

    @abstractmethod
    def trace_tool_execution(
        self,
        tool_name: str,
        tool_args: Dict[str, Any]
    ) -> AsyncContextManager:
        """Create a span for tool execution.

        Args:
            tool_name: Name of the tool
            tool_args: Arguments passed to tool

        Returns:
            Async context manager for the span
        """
        pass

    @abstractmethod
    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """Add an event to the current trace.

        Args:
            name: Event name
            attributes: Event attributes
        """
        pass

    @abstractmethod
    def set_attribute(self, key: str, value: Any) -> None:
        """Set an attribute on the current trace/span.

        Args:
            key: Attribute key
            value: Attribute value
        """
        pass

    @abstractmethod
    def record_exception(self, exception: Exception) -> None:
        """Record an exception in the current trace.

        Args:
            exception: Exception to record
        """
        pass

    @abstractmethod
    def set_status(self, status_code: str, description: str = "") -> None:
        """Set the status of the current trace.

        Args:
            status_code: Status code (OK, ERROR)
            description: Status description
        """
        pass

    @abstractmethod
    def record_metrics(self, metrics: Dict[str, Any]) -> None:
        """Record metrics on the current trace.

        Args:
            metrics: Dictionary of metrics to record
        """
        pass

    @abstractmethod
    def flush(self) -> None:
        """Flush pending traces."""
        pass

    def update_generation(self, output: Any, usage: Optional[Dict[str, int]] = None) -> None:
        """Update the current LLM generation with output and usage stats.

        This is specific to Langfuse and may not be supported by all telemetry providers.

        Args:
            output: Generated output (can be string or dict)
            usage: Token usage stats (prompt_tokens, completion_tokens, total_tokens)
        """
        # Optional method - default implementation does nothing
        pass


class NoOpLogger(ILogger):
    """No-op logger implementation for when logging is disabled."""

    def debug(self, message: str, **kwargs) -> None:
        pass

    def info(self, message: str, **kwargs) -> None:
        pass

    def warning(self, message: str, **kwargs) -> None:
        pass

    def error(self, message: str, **kwargs) -> None:
        pass

    def exception(self, message: str, **kwargs) -> None:
        pass


class NoOpTelemetry(ITelemetry):
    """No-op telemetry implementation for when telemetry is disabled."""

    @property
    def enabled(self) -> bool:
        return False

    def trace_agent_run(
        self,
        user_input: str,
        agent_type: str = "ReasoningAgent",
        agent_name: Optional[str] = None
    ) -> AsyncContextManager:
        return nullcontext()

    def trace_reasoning_phase(
        self,
        input_text: str,
        iteration: int
    ) -> AsyncContextManager:
        return nullcontext()

    def trace_llm_call(
        self,
        prompt: str,
        model: str,
        call_type: str = "completion",
        llm_name: Optional[str] = None
    ) -> AsyncContextManager:
        return nullcontext()

    def trace_tool_execution(
        self,
        tool_name: str,
        tool_args: Dict[str, Any]
    ) -> AsyncContextManager:
        return nullcontext()

    def trace_subagent_execution(
        self,
        subagent_name: str,
        input_data: str
    ) -> AsyncContextManager:
        return nullcontext()

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        pass

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def record_exception(self, exception: Exception) -> None:
        pass

    def set_status(self, status_code: str, description: str = "") -> None:
        pass

    def record_metrics(self, metrics: Dict[str, Any]) -> None:
        pass

    def flush(self) -> None:
        pass

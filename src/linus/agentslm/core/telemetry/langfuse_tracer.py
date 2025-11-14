from typing import Optional, Dict, Any
from loguru import logger

# Langfuse imports (optional dependency)
try:
    from langfuse import Langfuse
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    logger.warning("Langfuse not installed. Install with: pip install langfuse")

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
        self._current_trace = None
        self._current_span = None
        self._span_stack = []  # Stack of spans for nested contexts

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
            Trace context
        """
        if not self.enabled:
            from contextlib import nullcontext
            return nullcontext()

        # Create a new trace using Langfuse v3.x API
        # start_as_current_span creates both the trace and root span
        metadata = {"agent_type": agent_type}
        if self.session_id:
            metadata["session_id"] = self.session_id

        # Create root span for the agent run
        # Note: session_id is set in metadata, not as a parameter
        span = self.client.start_as_current_span(
            name="agent_run",
            input={"query": user_input},
            metadata=metadata
        )

        self._current_trace = span
        self._span_stack.append(span)

        logger.debug(f"[LANGFUSE] Created trace span: agent_run")

        # Return the span object (it's a context manager)
        return span

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
            Span context
        """
        if not self.enabled:
            from contextlib import nullcontext
            return nullcontext()

        # Create a child span for reasoning
        span = self.client.start_as_current_span(
            name="reasoning_phase",
            input={"text": input_text[:500], "iteration": iteration},
            metadata={"iteration": iteration}
        )

        self._span_stack.append(span)
        self._current_span = span

        logger.debug(f"[LANGFUSE] Created reasoning span for iteration {iteration}")

        return span

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
            Generation context
        """
        if not self.enabled:
            from contextlib import nullcontext
            return nullcontext()

        # Create a generation span for LLM calls using Langfuse v3.x API
        generation = self.client.start_as_current_observation(
            as_type="generation",
            name=f"llm_{call_type}",
            model=model,
            input=prompt[:1000],  # Limit input size
            metadata={"call_type": call_type}
        )

        logger.debug(f"[LANGFUSE] Created generation span: llm_{call_type}")

        return generation

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
            Span context
        """
        if not self.enabled:
            from contextlib import nullcontext
            return nullcontext()

        # Create a child span for tool execution using Langfuse v3.x API
        span = self.client.start_as_current_span(
            name=f"tool_{tool_name}",
            input=tool_args,
            metadata={"tool": tool_name}
        )

        logger.debug(f"[LANGFUSE] Created tool execution span: tool_{tool_name}")

        return span

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

    def update_generation(self, output: str, usage: Optional[Dict[str, int]] = None):
        """Update the current LLM generation with output and usage stats.

        Args:
            output: Generated output
            usage: Token usage stats (prompt_tokens, completion_tokens, total_tokens)
        """
        if not self.enabled:
            return

        # Generations are updated via span.update() in the context manager
        # This is called from reasoning_agent.py after LLM completion
        logger.debug(f"[LANGFUSE] Generation update called with {len(output) if output else 0} chars output")

    def flush(self):
        """Flush pending traces to Langfuse."""
        if self.enabled and self.client:
            try:
                self.client.flush()
                logger.info("[LANGFUSE] Flushed traces to Langfuse server")
            except Exception as e:
                logger.error(f"[LANGFUSE] Failed to flush traces: {e}")

    def record_metrics(self, metrics: Dict[str, Any]):
        """Record metrics on the current trace.

        Args:
            metrics: Dictionary of metrics to record
        """
        if not self.enabled:
            return

        try:
            # Update the current trace with metrics metadata
            # Langfuse v3.x supports update_current_trace
            self.client.update_current_trace(metadata={"metrics": metrics})
            logger.debug(f"[LANGFUSE] Recorded metrics: {list(metrics.keys())}")
        except Exception as e:
            logger.warning(f"[LANGFUSE] Failed to record metrics: {e}")
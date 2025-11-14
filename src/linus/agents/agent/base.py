"""Base Agent class with OpenAI client integration and auto tool calling mode detection."""

from typing import List, Dict, Any, Optional, Type, Union
import json
import re
from enum import Enum
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel

# Import SharedState and backends from graph module
from ..graph.state import SharedState, ConversationMemoryBackend, KeyValueBackend
from .tool_base import BaseTool
from ..di import ILogger, ITelemetry, get_container


class ToolCallingMode(Enum):
    """Tool calling mode for agents."""
    NATIVE = "native"      # Use OpenAI native function calling
    MANUAL = "manual"      # Parse tool calls from text (for models without native support)
    AUTO = "auto"          # Auto-detect based on model name


class Agent:
    """Base Agent class with OpenAI client integration."""

    def __init__(
        self,
        llm: Union[AsyncOpenAI, OpenAI],
        model: str,
        tools: List[BaseTool],
        verbose: bool = False,
        input_schema: Optional[Type[BaseModel]] = None,
        output_schema: Optional[Type[BaseModel]] = None,
        output_key: Optional[str] = None,
        state: Optional[SharedState] = None,
        memory: Optional[SharedState] = None,
        tool_calling_mode: Union[ToolCallingMode, str] = ToolCallingMode.AUTO,
        logger: Optional[ILogger] = None,
        telemetry: Optional[ITelemetry] = None,
        agent_name: Optional[str] = None
    ):
        """Initialize the base agent.

        Args:
            llm: OpenAI client instance (AsyncOpenAI or OpenAI)
            model: Model name to use (e.g., "gemma3:27b")
            tools: List of available tools
            verbose: Whether to print debug information
            input_schema: Optional Pydantic BaseModel for structured input validation
            output_schema: Optional Pydantic BaseModel for structured output
            output_key: Optional key to save output in shared state
            state: Optional SharedState instance for workflow/task data (uses KeyValueBackend)
            memory: Optional SharedState instance for conversation history (uses ConversationMemoryBackend)
            tool_calling_mode: Tool calling mode - AUTO (default), NATIVE, or MANUAL
                - AUTO: Auto-detect based on model name
                - NATIVE: Force native OpenAI function calling
                - MANUAL: Force manual tool calling (parse from text)
            logger: Optional logger instance (uses DI container if None)
            telemetry: Optional telemetry instance (uses DI container if None)
            agent_name: Optional name for the agent (used in hierarchical tracing)
        """
        self.llm = llm
        self.model = model
        self.tools = tools
        self.tool_map = {tool.name: tool for tool in tools}
        self.verbose = verbose
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.output_key = output_key
        self.agent_name = agent_name or "default"

        # Workflow state (key-value pairs shared between agents)
        self.state = state or SharedState(backend=KeyValueBackend())

        # Conversation memory (sequential interaction history)
        self.memory = memory or SharedState(backend=ConversationMemoryBackend())

        # Dependency injection for logger and telemetry
        container = get_container()
        self.logger = logger or container.get_logger()
        self.telemetry = telemetry or container.get_telemetry()

        # Update telemetry tracer with agent_name if it has the attribute
        if hasattr(self.telemetry, 'agent_name'):
            self.telemetry.agent_name = self.agent_name

        # Tool calling mode configuration
        # Convert string to enum if needed
        if isinstance(tool_calling_mode, str):
            tool_calling_mode = ToolCallingMode(tool_calling_mode.lower())

        self.tool_calling_mode = tool_calling_mode

        # Determine active mode (resolve AUTO to NATIVE or MANUAL)
        if tool_calling_mode == ToolCallingMode.AUTO:
            self.active_tool_mode = self._detect_tool_calling_mode()
            self.logger.info(f"[BASE-AGENT] Auto-detected tool calling mode: {self.active_tool_mode.value}")
        else:
            self.active_tool_mode = tool_calling_mode
            self.logger.info(f"[BASE-AGENT] Using configured tool calling mode: {self.active_tool_mode.value}")

    def run(self, input_data: Union[str, BaseModel, Dict[str, Any]]) -> Union[str, BaseModel]:
        """Run the agent on the given input.

        Args:
            input_data: The user's request (string, Pydantic model, or dict)

        Returns:
            The final response or result (string or Pydantic model)
        """
        raise NotImplementedError("Subclasses must implement the run method")

    def _validate_and_convert_input(self, input_data: Union[str, BaseModel, Dict[str, Any]]) -> str:
        """Validate and convert input to string format.

        Args:
            input_data: Input in various formats

        Returns:
            String representation of the input
        """
        if self.input_schema:
            if isinstance(input_data, dict):
                validated_input = self.input_schema(**input_data)
            elif isinstance(input_data, BaseModel):
                validated_input = input_data
            else:
                # Try to parse string as JSON for the schema
                try:
                    data = json.loads(input_data) if isinstance(input_data, str) else input_data
                    validated_input = self.input_schema(**data)
                except (json.JSONDecodeError, Exception):
                    # Fall back to using the raw input
                    return str(input_data)
            return validated_input.model_dump_json()

        if isinstance(input_data, BaseModel):
            return input_data.model_dump_json()
        elif isinstance(input_data, dict):
            return json.dumps(input_data)
        return str(input_data)

    def _format_output(self, result: str) -> Union[str, BaseModel]:
        """Format output according to output_schema if provided.

        Args:
            result: Raw result string

        Returns:
            Formatted output (string or Pydantic model)
        """
        if self.output_schema:
            try:
                # Try to parse result as JSON
                json_match = re.search(r'\{.*\}', result, re.DOTALL)
                if json_match:
                    result_data = json.loads(json_match.group())
                else:
                    result_data = json.loads(result)

                output_obj = self.output_schema(**result_data)

                # Save to state if output_key is provided
                if self.output_key:
                    self.state.set(self.output_key, output_obj, source="agent")
                    self._log(f"Saved output to state['{self.output_key}']")

                return output_obj
            except (json.JSONDecodeError, Exception) as e:
                self._log(f"Error parsing output with schema: {e}")
                # Fall back to string result
                if self.output_key:
                    self.state.set(self.output_key, result, source="agent")
                return result

        # No output schema, save raw result if output_key is provided
        if self.output_key:
            self.state.set(self.output_key, result, source="agent")
            self._log(f"Saved output to state['{self.output_key}']")

        return result

    def _log(self, message: str):
        """Log a message if verbose mode is enabled."""
        if self.verbose:
            self.logger.info(message)

    def _update_token_usage(self, response: Any):
        """Extract and update token usage from OpenAI response.

        Args:
            response: The response from the OpenAI API call
        """
        if self.current_metrics is None:
            return

        self.current_metrics.llm_calls += 1

        # Extract token usage from OpenAI response
        try:
            if hasattr(response, 'usage'):
                usage = response.usage
                self.current_metrics.total_input_tokens += getattr(usage, 'prompt_tokens', 0)
                self.current_metrics.total_output_tokens += getattr(usage, 'completion_tokens', 0)
                self.current_metrics.total_tokens += getattr(usage, 'total_tokens', 0)
            # Fallback: estimate tokens if usage not available
            elif hasattr(response, 'choices') and len(response.choices) > 0:
                content = response.choices[0].message.content
                # Rough estimation: ~4 characters per token
                estimated_tokens = len(content) // 4 if content else 0
                self.current_metrics.total_output_tokens += estimated_tokens
                self.current_metrics.total_tokens += estimated_tokens
        except Exception as e:
            self.logger.debug(f"Could not extract token usage: {e}")

    def _detect_tool_calling_mode(self) -> ToolCallingMode:
        """Auto-detect if model supports native function calling.

        Returns:
            ToolCallingMode.NATIVE if model supports native tools, MANUAL otherwise
        """
        # Models known to support native function calling
        native_supported_models = [
            # OpenAI models
            "gpt-4", "gpt-3.5-turbo", "gpt-4-turbo", "gpt-4o",
            # Anthropic Claude
            "claude-3", "claude-2",
            # Google models
            "gemini-pro", "gemini-1.5", "gemini-2",
            # Mistral
            "mistral-large", "mistral-medium", "mistral-small",
            # Other providers with native support
            "command-r", "dbrx"
        ]

        model_lower = self.model.lower()

        # Check if model name contains any known native-supported model
        for supported in native_supported_models:
            if supported in model_lower:
                return ToolCallingMode.NATIVE

        # Models known to NOT support native function calling (require manual mode)
        manual_only_models = [
            "gemma", "llama", "phi", "qwen", "vicuna", "alpaca",
            "orca", "wizardlm", "nous-hermes"
        ]

        for manual in manual_only_models:
            if manual in model_lower:
                return ToolCallingMode.MANUAL

        # Default to MANUAL for unknown models to be safe
        # (models can always fall back to manual mode)
        self.logger.warning(
            f"[BASE-AGENT] Unknown model '{self.model}', defaulting to MANUAL mode. "
            f"Use tool_calling_mode='native' to force native function calling."
        )
        return ToolCallingMode.MANUAL

    def _convert_tools_to_openai_format(self) -> List[Dict[str, Any]]:
        """Convert BaseTool instances to OpenAI function calling format.

        Returns:
            List of tool dictionaries in OpenAI format
        """
        openai_tools = []

        for tool in self.tools:
            # Get tool schema
            tool_schema = {}
            if hasattr(tool, 'args_schema') and tool.args_schema:
                if hasattr(tool.args_schema, 'model_json_schema'):
                    schema = tool.args_schema.model_json_schema()
                    tool_schema = {
                        "type": "object",
                        "properties": schema.get("properties", {}),
                        "required": schema.get("required", [])
                    }

            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool_schema
                }
            }

            openai_tools.append(openai_tool)

        return openai_tools

    def _extract_tool_call_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract tool call from model response text (for MANUAL mode).

        Looks for JSON in format: {"tool": "tool_name", "arguments": {...}}

        Args:
            text: Model response text

        Returns:
            Dictionary with 'tool' and 'arguments' keys, or None if no tool call found
        """
        if not text:
            return None

        # Strategy 1: Look for JSON in markdown code blocks
        code_block_patterns = [
            r'```json\s*(\{[^`]*"tool"[^`]*\})\s*```',
            r'```\s*(\{[^`]*"tool"[^`]*\})\s*```'
        ]

        for pattern in code_block_patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                try:
                    tool_call = json.loads(match.group(1))
                    if "tool" in tool_call and "arguments" in tool_call:
                        self.logger.debug(f"[BASE-AGENT] Extracted tool call from code block: {tool_call['tool']}")
                        return tool_call
                except json.JSONDecodeError:
                    pass

        # Strategy 2: Look for JSON object with "tool" and "arguments" keys
        json_pattern = r'\{[^{}]*"tool"[^{}]*"arguments"[^{}]*\}'
        matches = re.finditer(json_pattern, text, re.DOTALL)

        for match in matches:
            try:
                tool_call = json.loads(match.group())
                if "tool" in tool_call and "arguments" in tool_call:
                    self.logger.debug(f"[BASE-AGENT] Extracted tool call from JSON: {tool_call['tool']}")
                    return tool_call
            except json.JSONDecodeError:
                continue

        # Strategy 3: More lenient - find any JSON object, check if it looks like a tool call
        json_objects = re.finditer(r'\{[^{}]*\}', text, re.DOTALL)
        for match in json_objects:
            try:
                obj = json.loads(match.group())
                # Check if it has tool-like structure
                if isinstance(obj, dict) and "tool" in obj:
                    # If it has "tool" key, assume arguments are the rest
                    if "arguments" not in obj:
                        # Gather all other keys as arguments
                        arguments = {k: v for k, v in obj.items() if k != "tool"}
                        obj["arguments"] = arguments

                    self.logger.debug(f"[BASE-AGENT] Extracted lenient tool call: {obj['tool']}")
                    return obj
            except json.JSONDecodeError:
                continue

        return None

    # Memory convenience methods (backward compatibility with MemoryManager API)
    def add_memory(
        self,
        content: str,
        source: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a conversation memory entry.

        Convenience method that wraps self.memory.set() for backward compatibility.

        Args:
            content: Memory content (conversation text)
            source: Source of the memory (e.g., "user", "assistant")
            metadata: Optional metadata
        """
        # Use auto-generated key for conversation entries
        key = f"auto_{self.memory.backend.count()}"
        self.memory.set(key=key, value=content, source=source, metadata=metadata)

    def get_memory_context(
        self,
        max_tokens: Optional[int] = None,
        query: Optional[str] = None
    ) -> str:
        """Get memory context for prompt inclusion.

        Convenience method that wraps self.memory.get_context().

        Args:
            max_tokens: Maximum tokens for context
            query: Optional search query

        Returns:
            Formatted memory context string
        """
        # Use CLIP strategy for conversation memory by default
        from ..graph.state import StateContextStrategy

        if query:
            # Search-based context
            entries = self.memory.search(query, limit=10)
            if not entries:
                return ""

            context_text = "=== Relevant Conversation History ===\n"
            for entry in reversed(entries):  # Chronological order
                context_text += f"{entry.value}\n"
            return context_text
        else:
            # Recent context with token limit
            return self.memory.get_context(
                strategy=StateContextStrategy.CLIP,
                max_tokens=max_tokens
            )

    def clear_memory(self) -> None:
        """Clear all conversation memory.

        Convenience method that wraps self.memory.clear().
        """
        self.memory.clear()

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics.

        Convenience method that wraps self.memory.get_state_stats().

        Returns:
            Dictionary with memory statistics
        """
        return self.memory.get_state_stats()

"""Base Agent class with OpenAI client integration."""

from typing import List, Dict, Any, Optional, Type, Union
import json
import re
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel
from loguru import logger

# Import SharedState from graph module
from ..graph.state import SharedState
from .tool_base import BaseTool


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
        memory_manager: Optional['MemoryManager'] = None
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
            state: Optional SharedState instance for state management
            memory_manager: Optional memory manager for context persistence
        """
        self.llm = llm
        self.model = model
        self.tools = tools
        self.tool_map = {tool.name: tool for tool in tools}
        self.verbose = verbose
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.output_key = output_key

        # Use SharedState directly
        self.state = state or SharedState()

        self.memory_manager = memory_manager

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
            logger.info(message)

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
            logger.debug(f"Could not extract token usage: {e}")

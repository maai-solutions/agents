"""LightAgent - A lightweight agent with native tool calling support.

This agent provides a simpler, more direct interface compared to ReasoningAgent,
using OpenAI's native function calling instead of custom reasoning phases.

The agent now supports BOTH native function calling (for GPT-4, Claude, etc.)
AND manual tool calling (for Gemma, Llama, and other models without native support).
The mode is auto-detected based on the model name, but can be manually overridden.
"""

from typing import List, Dict, Any, Optional, Type, Union, Generator, AsyncGenerator
import json
import re
import time
from datetime import datetime
from enum import Enum
from pydantic import BaseModel
from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletionChunk

from .base import Agent
from .models import AgentMetrics, AgentResponse, Citation
from .tool_base import BaseTool
from ..graph.state import SharedState
from ..di import ILogger, ITelemetry
from linus.agents.agent.memory import MemoryManager


class ToolCallingMode(Enum):
    """Tool calling mode for LightAgent."""
    NATIVE = "native"      # Use OpenAI native function calling
    MANUAL = "manual"      # Parse tool calls from text (for models without native support)
    AUTO = "auto"          # Auto-detect based on model name


class LightAgent(Agent):
    """A lightweight agent that supports both native and manual tool calling.

    LightAgent is designed for:
    - Direct interaction with OpenAI-compatible APIs
    - Native function calling (tools parameter in chat completions) for supported models
    - Manual tool calling for models without native support (Gemma, Llama, etc.)
    - Auto-detection of tool calling capabilities
    - Simpler reasoning without multi-phase planning
    - Optional streaming support
    - Task transfer within agent swarms

    Key features:
    - AUTO mode: Automatically detects if model supports native function calling
    - NATIVE mode: Uses OpenAI tool calling (for GPT-4, Claude, etc.)
    - MANUAL mode: Parses tool calls from text (for Gemma, Llama, etc.)
    - Works with any OpenAI-compatible endpoint (OpenAI, Ollama, etc.)
    """

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
        memory_manager: Optional[MemoryManager] = None,
        # LightAgent-specific parameters
        instructions: str = "You are a helpful AI assistant.",
        role: Optional[str] = None,
        max_tool_iterations: int = 10,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stream: bool = False,
        # Tool calling mode
        tool_calling_mode: Union[ToolCallingMode, str] = ToolCallingMode.AUTO,
        # Integration parameters
        logger: Optional[ILogger] = None,
        telemetry: Optional[ITelemetry] = None,
        agent_name: Optional[str] = None,
    ):
        """Initialize the LightAgent.

        Args:
            llm: OpenAI client instance (AsyncOpenAI or OpenAI)
            model: Model name to use (e.g., "gpt-4", "gemma3:27b")
            tools: List of available tools
            verbose: Whether to print debug information
            input_schema: Optional Pydantic BaseModel for structured input validation
            output_schema: Optional Pydantic BaseModel for structured output
            output_key: Optional key to save output in shared state
            state: Optional SharedState instance for state management
            memory_manager: Optional memory manager for context persistence
            instructions: System instructions for the agent
            role: Optional role description for the agent
            max_tool_iterations: Maximum tool calling iterations (default: 10)
            temperature: Sampling temperature for LLM calls
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter (Ollama-specific)
            stream: Enable streaming responses (default: False)
            tool_calling_mode: Tool calling mode - AUTO (default), NATIVE, or MANUAL
                - AUTO: Auto-detect based on model name
                - NATIVE: Force native OpenAI function calling
                - MANUAL: Force manual tool calling (parse from text)
            logger: Optional logger instance
            telemetry: Optional telemetry instance
            agent_name: Optional name for the agent
        """
        super().__init__(
            llm, model, tools, verbose, input_schema, output_schema,
            output_key, state, memory_manager, logger, telemetry, agent_name
        )

        # LightAgent-specific configuration
        self.instructions = instructions
        self.role = role
        self.max_tool_iterations = max_tool_iterations
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.stream = stream

        # Tool calling mode configuration
        # Convert string to enum if needed
        if isinstance(tool_calling_mode, str):
            tool_calling_mode = ToolCallingMode(tool_calling_mode.lower())

        self.tool_calling_mode = tool_calling_mode

        # Determine active mode (resolve AUTO to NATIVE or MANUAL)
        if tool_calling_mode == ToolCallingMode.AUTO:
            self.active_mode = self._detect_tool_calling_mode()
            self.logger.info(f"[LIGHT-AGENT] Auto-detected tool calling mode: {self.active_mode.value}")
        else:
            self.active_mode = tool_calling_mode
            self.logger.info(f"[LIGHT-AGENT] Using configured tool calling mode: {self.active_mode.value}")

        # Message history for conversation
        self.messages: List[Dict[str, Any]] = []

        # Metrics
        self.current_metrics: Optional[AgentMetrics] = None

        # Swarm context (set by Swarm when agent is part of one)
        self.swarm: Optional[Any] = None

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
            f"[LIGHT-AGENT] Unknown model '{self.model}', defaulting to MANUAL mode. "
            f"Use tool_calling_mode='native' to force native function calling."
        )
        return ToolCallingMode.MANUAL

    def _get_generation_kwargs(self) -> Dict[str, Any]:
        """Build kwargs for LLM generation."""
        kwargs = {
            "model": self.model,
            "temperature": self.temperature,
            "stream": self.stream
        }

        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens

        if self.top_p is not None:
            kwargs["top_p"] = self.top_p

        if self.top_k is not None:
            kwargs["extra_body"] = {"top_k": self.top_k}

        return kwargs

    def _build_system_message(self) -> str:
        """Build the system message with instructions and role.

        For MANUAL mode, includes tool descriptions and calling format.
        """
        now = datetime.now()
        current_date = now.strftime("%Y-%m-%d")
        current_time = now.strftime("%H:%M:%S")

        message = f"Agent: {self.agent_name}\n"
        message += f"Instructions: {self.instructions}\n"

        if self.role:
            message += f"Role: {self.role}\n"

        message += f"\nCurrent date: {current_date}\n"
        message += f"Current time: {current_time}\n"

        # Add tool descriptions for MANUAL mode
        if self.active_mode == ToolCallingMode.MANUAL and self.tools:
            message += self._build_manual_tool_instructions()

        return message

    def _build_manual_tool_instructions(self) -> str:
        """Build tool instructions for MANUAL mode.

        Returns:
            Formatted tool instructions to include in system prompt
        """
        instructions = "\n\n## Available Tools\n"
        instructions += "You have access to the following tools. To use a tool, output JSON in this EXACT format:\n"
        instructions += '```json\n{"tool": "tool_name", "arguments": {...}}\n```\n\n'
        instructions += "After you call a tool, I will provide the result and you can continue your response.\n\n"

        for tool in self.tools:
            instructions += f"### {tool.name}\n"
            instructions += f"**Description:** {tool.description}\n"

            # Add parameter schema if available
            if hasattr(tool, 'args_schema') and tool.args_schema:
                if hasattr(tool.args_schema, 'model_json_schema'):
                    schema = tool.args_schema.model_json_schema()
                    properties = schema.get('properties', {})
                    required = schema.get('required', [])

                    if properties:
                        instructions += "**Parameters:**\n"
                        for param_name, param_info in properties.items():
                            param_type = param_info.get('type', 'any')
                            param_desc = param_info.get('description', 'No description')
                            is_required = ' (required)' if param_name in required else ' (optional)'
                            instructions += f"  - `{param_name}` ({param_type}){is_required}: {param_desc}\n"

            instructions += "\n"

        instructions += "**Important:** Only call ONE tool at a time. Wait for the result before calling another tool.\n"

        return instructions

    def _convert_tools_to_openai_format(self) -> List[Dict[str, Any]]:
        """Convert BaseTool instances to OpenAI function calling format."""
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
                        self.logger.debug(f"[LIGHT-AGENT] Extracted tool call from code block: {tool_call['tool']}")
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
                    self.logger.debug(f"[LIGHT-AGENT] Extracted tool call from JSON: {tool_call['tool']}")
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

                    self.logger.debug(f"[LIGHT-AGENT] Extracted lenient tool call: {obj['tool']}")
                    return obj
            except json.JSONDecodeError:
                continue

        return None

    async def run(
        self,
        input_data: Union[str, BaseModel, Dict[str, Any]],
        return_metrics: bool = True,
        history: Optional[List[Dict[str, Any]]] = None
    ) -> Union[str, BaseModel, AgentResponse]:
        """Run the agent on the given input.

        Args:
            input_data: The user's request
            return_metrics: If True, return AgentResponse with metrics
            history: Optional conversation history

        Returns:
            AgentResponse (if return_metrics=True) or just the result
        """
        # Validate and convert input
        input_text = self._validate_and_convert_input(input_data)

        # Start tracing
        async with self.telemetry.trace_agent_run(
            user_input=input_text,
            agent_type="LightAgent",
            agent_name=self.agent_name
        ) as trace:
            return await self._run_with_trace(input_text, return_metrics, history, trace)

    async def _run_with_trace(
        self,
        input_text: str,
        return_metrics: bool,
        history: Optional[List[Dict[str, Any]]],
        trace: Any = None
    ) -> Union[str, BaseModel, AgentResponse]:
        """Internal run method with tracing support.

        Routes to appropriate implementation based on active_mode.
        """
        self.logger.info(f"[LIGHT-AGENT] Starting execution in {self.active_mode.value} mode: {input_text}")

        # Route to appropriate implementation
        if self.active_mode == ToolCallingMode.NATIVE:
            return await self._run_native_mode(input_text, return_metrics, history, trace)
        else:  # MANUAL mode
            return await self._run_manual_mode(input_text, return_metrics, history, trace)

    async def _run_native_mode(
        self,
        input_text: str,
        return_metrics: bool,
        history: Optional[List[Dict[str, Any]]],
        trace: Any = None
    ) -> Union[str, BaseModel, AgentResponse]:
        """Run with native OpenAI function calling."""
        # Initialize metrics
        metrics = AgentMetrics()
        self.current_metrics = metrics
        start_time = time.time()

        self.logger.info(f"[LIGHT-AGENT-NATIVE] Starting native mode execution")

        # Build messages
        messages = []

        # Add system message
        system_message = self._build_system_message()
        messages.append({"role": "system", "content": system_message})

        # Add history if provided
        if history:
            messages.extend(history)

        # Add memory context if available
        if self.memory_manager:
            memory_context = self.memory_manager.get_context(
                max_tokens=1000,
                include_summary=True,
                query=input_text
            )
            if memory_context:
                messages.append({"role": "system", "content": f"Context from memory:\n{memory_context}"})

        # Add user message
        messages.append({"role": "user", "content": input_text})

        # Store messages for potential history access
        self.messages = messages.copy()

        # Convert tools to OpenAI format
        openai_tools = self._convert_tools_to_openai_format()

        # Track execution
        citations = []
        iteration = 0
        final_response = None

        try:
            while iteration < self.max_tool_iterations:
                iteration += 1
                self.logger.info(f"[LIGHT-AGENT] Iteration {iteration}/{self.max_tool_iterations}")

                # Prepare API call parameters
                api_params = {
                    "messages": messages,
                    **self._get_generation_kwargs()
                }

                if openai_tools:
                    api_params["tools"] = openai_tools
                    api_params["tool_choice"] = "auto"

                # Make LLM call
                async with self.telemetry.trace_llm_call(
                    prompt=messages,
                    model=self.model,
                    call_type="light_agent",
                    llm_name=f"light_agent_iter_{iteration}"
                ):
                    response = await self.llm.chat.completions.create(**api_params)

                    # Track metrics
                    self._update_token_usage(response)

                    # Update telemetry
                    if hasattr(self.telemetry, 'update_generation'):
                        usage = None
                        if hasattr(response, 'usage') and response.usage:
                            usage = {
                                "prompt_tokens": response.usage.prompt_tokens,
                                "completion_tokens": response.usage.completion_tokens,
                                "total_tokens": response.usage.total_tokens
                            }
                        self.telemetry.update_generation(
                            output={"message": response.choices[0].message.content},
                            usage=usage
                        )

                choice = response.choices[0]
                message = choice.message

                # Check if there are tool calls
                if message.tool_calls:
                    self.logger.info(f"[LIGHT-AGENT] Processing {len(message.tool_calls)} tool calls")

                    # Add assistant message to history
                    messages.append({
                        "role": "assistant",
                        "content": message.content,
                        "tool_calls": [tc.model_dump() for tc in message.tool_calls]
                    })

                    # Execute each tool call
                    for tool_call in message.tool_calls:
                        tool_name = tool_call.function.name
                        tool_args_str = tool_call.function.arguments

                        self.logger.info(f"[LIGHT-AGENT] Calling tool: {tool_name}")

                        try:
                            # Parse arguments
                            tool_args = json.loads(tool_args_str)

                            # Get tool
                            if tool_name not in self.tool_map:
                                raise ValueError(f"Tool '{tool_name}' not found")

                            tool = self.tool_map[tool_name]

                            # Execute tool
                            async with self.telemetry.trace_tool_execution(tool_name, tool_args):
                                result = await tool.arun(tool_args)

                                # Track metrics
                                if self.current_metrics:
                                    self.current_metrics.tool_executions += 1
                                    self.current_metrics.successful_tool_calls += 1

                                # Extract citations if available
                                try:
                                    result_data = json.loads(str(result))
                                    if "citations" in result_data:
                                        for citation_data in result_data["citations"]:
                                            citation = Citation(
                                                document_id=citation_data.get("document_id", "unknown"),
                                                chunk_number=citation_data.get("chunk_number", 0),
                                                score=citation_data.get("score"),
                                                content_preview=citation_data.get("content_preview")
                                            )
                                            citations.append(citation)
                                except (json.JSONDecodeError, KeyError):
                                    pass

                                # Add tool response to messages
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "name": tool_name,
                                    "content": str(result)
                                })

                                self.logger.debug(f"[LIGHT-AGENT] Tool result: {str(result)[:200]}")

                        except Exception as e:
                            self.logger.exception(f"[LIGHT-AGENT] Tool execution error: {e}")

                            # Track failed tool call
                            if self.current_metrics:
                                self.current_metrics.tool_executions += 1
                                self.current_metrics.failed_tool_calls += 1

                            # Add error to messages
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": tool_name,
                                "content": f"Error: {str(e)}"
                            })

                    # Continue to next iteration to get response after tool execution
                    continue

                else:
                    # No tool calls - we have the final response
                    final_response = message.content
                    self.logger.info("[LIGHT-AGENT] Received final response")
                    break

            # Calculate metrics
            metrics.total_iterations = iteration
            metrics.execution_time_seconds = time.time() - start_time
            metrics.task_completed = final_response is not None

            # Store in memory if available
            if self.memory_manager and final_response:
                self.memory_manager.add_memory(
                    content=f"User: {input_text}\nAssistant: {final_response[:500]}",
                    metadata={"role": "interaction", "type": "completion"},
                    importance=1.0,
                    entry_type="interaction"
                )

            # Format output
            if final_response is None:
                final_response = "Maximum iterations reached without final response"

            formatted_result = self._format_output(final_response)

            # Update trace
            if trace and hasattr(trace, 'update'):
                trace.update(
                    output=str(formatted_result),
                    level="DEFAULT",
                    metadata={
                        "iterations": iteration,
                        "execution_time": metrics.execution_time_seconds
                    }
                )

            # Record metrics
            self.telemetry.record_metrics(metrics.to_dict())

            # Return based on return_metrics flag
            if return_metrics:
                return AgentResponse(
                    result=formatted_result,
                    metrics=metrics,
                    execution_history=[],
                    completion_status={
                        "is_complete": True,
                        "reasoning": "Task completed"
                    },
                    citations=citations
                )
            else:
                return formatted_result

        except Exception as e:
            self.logger.exception(f"[LIGHT-AGENT-NATIVE] Error during execution: {e}")
            metrics.execution_time_seconds = time.time() - start_time
            metrics.task_completed = False

            error_message = f"Error during execution: {str(e)}"

            if return_metrics:
                return AgentResponse(
                    result=error_message,
                    metrics=metrics,
                    execution_history=[],
                    completion_status={
                        "is_complete": False,
                        "reasoning": str(e)
                    },
                    citations=[]
                )
            else:
                return error_message

    async def _run_manual_mode(
        self,
        input_text: str,
        return_metrics: bool,
        history: Optional[List[Dict[str, Any]]],
        trace: Any = None
    ) -> Union[str, BaseModel, AgentResponse]:
        """Run with manual tool calling (parse tool calls from text)."""
        # Initialize metrics
        metrics = AgentMetrics()
        self.current_metrics = metrics
        start_time = time.time()

        self.logger.info(f"[LIGHT-AGENT-MANUAL] Starting manual mode execution")

        # Build messages
        messages = []

        # Add system message (includes tool descriptions)
        system_message = self._build_system_message()
        messages.append({"role": "system", "content": system_message})

        # Add history if provided
        if history:
            messages.extend(history)

        # Add memory context if available
        if self.memory_manager:
            memory_context = self.memory_manager.get_context(
                max_tokens=1000,
                include_summary=True,
                query=input_text
            )
            if memory_context:
                messages.append({"role": "system", "content": f"Context from memory:\n{memory_context}"})

        # Add user message
        messages.append({"role": "user", "content": input_text})

        # Store messages for potential history access
        self.messages = messages.copy()

        # Track execution
        citations = []
        iteration = 0
        final_response = None

        try:
            while iteration < self.max_tool_iterations:
                iteration += 1
                self.logger.info(f"[LIGHT-AGENT-MANUAL] Iteration {iteration}/{self.max_tool_iterations}")

                # Prepare API call parameters (no tools parameter in manual mode)
                api_params = {
                    "messages": messages,
                    **self._get_generation_kwargs()
                }

                # Make LLM call
                async with self.telemetry.trace_llm_call(
                    prompt=messages,
                    model=self.model,
                    call_type="light_agent_manual",
                    llm_name=f"light_agent_manual_iter_{iteration}"
                ):
                    response = await self.llm.chat.completions.create(**api_params)

                    # Track metrics
                    self._update_token_usage(response)

                    # Update telemetry
                    if hasattr(self.telemetry, 'update_generation'):
                        usage = None
                        if hasattr(response, 'usage') and response.usage:
                            usage = {
                                "prompt_tokens": response.usage.prompt_tokens,
                                "completion_tokens": response.usage.completion_tokens,
                                "total_tokens": response.usage.total_tokens
                            }
                        self.telemetry.update_generation(
                            output={"message": response.choices[0].message.content},
                            usage=usage
                        )

                choice = response.choices[0]
                message = choice.message
                response_text = message.content

                self.logger.debug(f"[LIGHT-AGENT-MANUAL] Response: {response_text[:200]}...")

                # Try to extract tool call from response
                tool_call = self._extract_tool_call_from_text(response_text)

                if tool_call:
                    tool_name = tool_call.get("tool")
                    tool_args = tool_call.get("arguments", {})

                    self.logger.info(f"[LIGHT-AGENT-MANUAL] Extracted tool call: {tool_name}")

                    # Add assistant message to history
                    messages.append({
                        "role": "assistant",
                        "content": response_text
                    })

                    try:
                        # Get tool
                        if tool_name not in self.tool_map:
                            raise ValueError(f"Tool '{tool_name}' not found")

                        tool = self.tool_map[tool_name]

                        # Execute tool
                        async with self.telemetry.trace_tool_execution(tool_name, tool_args):
                            result = await tool.arun(tool_args)

                            # Track metrics
                            if self.current_metrics:
                                self.current_metrics.tool_executions += 1
                                self.current_metrics.successful_tool_calls += 1

                            # Extract citations if available
                            try:
                                result_data = json.loads(str(result))
                                if "citations" in result_data:
                                    for citation_data in result_data["citations"]:
                                        citation = Citation(
                                            document_id=citation_data.get("document_id", "unknown"),
                                            chunk_number=citation_data.get("chunk_number", 0),
                                            score=citation_data.get("score"),
                                            content_preview=citation_data.get("content_preview")
                                        )
                                        citations.append(citation)
                            except (json.JSONDecodeError, KeyError):
                                pass

                            # Add tool result to messages
                            messages.append({
                                "role": "user",
                                "content": f"Tool '{tool_name}' result:\n{str(result)}"
                            })

                            self.logger.debug(f"[LIGHT-AGENT-MANUAL] Tool result: {str(result)[:200]}")

                    except Exception as e:
                        self.logger.exception(f"[LIGHT-AGENT-MANUAL] Tool execution error: {e}")

                        # Track failed tool call
                        if self.current_metrics:
                            self.current_metrics.tool_executions += 1
                            self.current_metrics.failed_tool_calls += 1

                        # Add error to messages
                        messages.append({
                            "role": "user",
                            "content": f"Tool '{tool_name}' error: {str(e)}"
                        })

                    # Continue to next iteration to get response after tool execution
                    continue

                else:
                    # No tool call found - this is the final response
                    final_response = response_text
                    self.logger.info("[LIGHT-AGENT-MANUAL] Received final response (no tool call detected)")
                    break

            # Calculate metrics
            metrics.total_iterations = iteration
            metrics.execution_time_seconds = time.time() - start_time
            metrics.task_completed = final_response is not None

            # Store in memory if available
            if self.memory_manager and final_response:
                self.memory_manager.add_memory(
                    content=f"User: {input_text}\nAssistant: {final_response[:500]}",
                    metadata={"role": "interaction", "type": "completion"},
                    importance=1.0,
                    entry_type="interaction"
                )

            # Format output
            if final_response is None:
                final_response = "Maximum iterations reached without final response"

            formatted_result = self._format_output(final_response)

            # Update trace
            if trace and hasattr(trace, 'update'):
                trace.update(
                    output=str(formatted_result),
                    level="DEFAULT",
                    metadata={
                        "iterations": iteration,
                        "execution_time": metrics.execution_time_seconds
                    }
                )

            # Record metrics
            self.telemetry.record_metrics(metrics.to_dict())

            # Return based on return_metrics flag
            if return_metrics:
                return AgentResponse(
                    result=formatted_result,
                    metrics=metrics,
                    execution_history=[],
                    completion_status={
                        "is_complete": True,
                        "reasoning": "Task completed"
                    },
                    citations=citations
                )
            else:
                return formatted_result

        except Exception as e:
            self.logger.exception(f"[LIGHT-AGENT-MANUAL] Error during execution: {e}")
            metrics.execution_time_seconds = time.time() - start_time
            metrics.task_completed = False

            error_message = f"Error during execution: {str(e)}"

            if return_metrics:
                return AgentResponse(
                    result=error_message,
                    metrics=metrics,
                    execution_history=[],
                    completion_status={
                        "is_complete": False,
                        "reasoning": str(e)
                    },
                    citations=[]
                )
            else:
                return error_message

    def get_history(self) -> List[Dict[str, Any]]:
        """Get the conversation history.

        Returns:
            List of message dictionaries in OpenAI format
        """
        return self.messages.copy()

    def clear_history(self):
        """Clear the conversation history."""
        self.messages = []

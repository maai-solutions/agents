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
from pydantic import BaseModel
from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletionChunk

from .base import Agent, ToolCallingMode
from .models import AgentMetrics, AgentResponse, Citation
from .tool_base import BaseTool
from ..graph.state import SharedState
from ..di import ILogger, ITelemetry


class LightAgent(Agent):
    """A lightweight agent that supports both native and manual tool calling.

    LightAgent is designed for:
    - Direct interaction with OpenAI-compatible APIs
    - Native function calling (tools parameter in chat completions) for supported models
    - Manual tool calling for models without native support (Gemma, Llama, etc.)
    - Auto-detection of tool calling capabilities
    - Tree of Thought reasoning for complex tasks
    - Adaptive tool filtering based on task analysis
    - Self-learning from memory context
    - Optional streaming support
    - Task transfer within agent swarms
    - Dynamic tool creation

    Key features:
    - **Tool Calling Modes:**
      - AUTO mode: Automatically detects if model supports native function calling
      - NATIVE mode: Uses OpenAI tool calling (for GPT-4, Claude, etc.)
      - MANUAL mode: Parses tool calls from text (for Gemma, Llama, etc.)

    - **Tree of Thought (ToT):**
      - Enables multi-phase reasoning: initial thought → reflection → refinement
      - Adaptive tool filtering based on task requirements
      - Separate reasoning model support for better planning

    - **Memory & Learning:**
      - Context-aware memory integration
      - Self-learning mode: agents learn from previous interactions
      - User preference tracking

    - **Agent Swarms:**
      - Automatic task transfer between agents based on capabilities
      - Intent detection for routing requests
      - Collaborative agent workflows

    - **MCP Integration:**
      - Compatible with Model Context Protocol (MCP) servers
      - Use mcp_client.py to add MCP tools to the agent

    - **Extensibility:**
      - Dynamic tool creation from natural language descriptions
      - WebSocket support for streaming
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
        memory: Optional[SharedState] = None,
        # LightAgent-specific parameters
        instructions: str = "You are a helpful AI assistant.",
        role: Optional[str] = None,
        max_tool_iterations: int = 10,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stream: bool = False,
        # Tree of Thought parameters
        tree_of_thought: bool = False,
        tot_model: Optional[str] = None,
        tot_llm: Optional[Union[AsyncOpenAI, OpenAI]] = None,
        # Tool filtering
        filter_tools: bool = True,
        # Self-learning
        self_learning: bool = False,
        # WebSocket support
        websocket_base_url: Optional[str] = None,
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
            memory: Optional SharedState instance for conversation history (uses ConversationMemoryBackend)
            instructions: System instructions for the agent
            role: Optional role description for the agent
            max_tool_iterations: Maximum tool calling iterations (default: 10)
            temperature: Sampling temperature for LLM calls
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter (Ollama-specific)
            stream: Enable streaming responses (default: False)
            tree_of_thought: Enable Tree of Thought reasoning (default: False)
            tot_model: Model for ToT reasoning (defaults to main model)
            tot_llm: Separate LLM client for ToT (defaults to main llm)
            filter_tools: Enable adaptive tool filtering based on task (default: True)
            self_learning: Enable self-learning from memory (default: False)
            websocket_base_url: WebSocket base URL for streaming
            tool_calling_mode: Tool calling mode - AUTO (default), NATIVE, or MANUAL
                - AUTO: Auto-detect based on model name
                - NATIVE: Force native OpenAI function calling
                - MANUAL: Force manual tool calling (parse from text)
            logger: Optional logger instance
            telemetry: Optional telemetry instance
            agent_name: Optional name for the agent
        """
        # Pass tool_calling_mode to base class
        super().__init__(
            llm, model, tools, verbose, input_schema, output_schema,
            output_key, state, memory, tool_calling_mode,
            logger, telemetry, agent_name
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
        self.websocket_base_url = websocket_base_url

        # Tree of Thought configuration
        self.tree_of_thought = tree_of_thought
        self.tot_model = tot_model or model
        self.tot_llm = tot_llm or llm

        # Tool filtering and self-learning
        self.filter_tools = filter_tools
        self.self_learning = self_learning

        # Use active_mode from base class (already set by base __init__)
        self.active_mode = self.active_tool_mode

        # Message history for conversation
        self.messages: List[Dict[str, Any]] = []

        # Metrics
        self.current_metrics: Optional[AgentMetrics] = None

        # Swarm context (set by Swarm when agent is part of one)
        self.swarm: Optional[Any] = None

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

    def _build_system_message(self, tot_context: str = "") -> str:
        """Build the system message with instructions and role.

        For MANUAL mode, includes tool descriptions and calling format.

        Args:
            tot_context: Optional Tree of Thought reasoning context to include
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

        # Add Tree of Thought context if provided
        if tot_context:
            message += f"\n## Supplementary Analysis\n{tot_context}\n"

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

    async def run(
        self,
        input_data: Union[str, BaseModel, Dict[str, Any]],
        return_metrics: bool = True,
        history: Optional[List[Dict[str, Any]]] = None,
        light_swarm: Optional[Any] = None
    ) -> Union[str, BaseModel, AgentResponse]:
        """Run the agent on the given input.

        Args:
            input_data: The user's request
            return_metrics: If True, return AgentResponse with metrics
            history: Optional conversation history
            light_swarm: Optional LightSwarm instance for agent transfer

        Returns:
            AgentResponse (if return_metrics=True) or just the result
        """
        # Validate and convert input
        input_text = self._validate_and_convert_input(input_data)

        # Check for agent transfer if swarm is available
        if light_swarm:
            transfer_result = await self._handle_task_transfer(input_text, light_swarm, return_metrics, history)
            if transfer_result is not None:
                return transfer_result

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

        # Run Tree of Thought if enabled
        tot_context = ""
        active_tools = []
        if self.tree_of_thought:
            self.logger.info("[LIGHT-AGENT-NATIVE] Running Tree of Thought...")
            tot_response, filtered_tools = await self.run_thought(input_text)
            tot_context = tot_response
            if filtered_tools:
                active_tools = filtered_tools
            self.logger.debug(f"[LIGHT-AGENT-NATIVE] ToT filtered {len(active_tools)} tools")

        # Build messages
        messages = []

        # Add system message (with ToT context if available)
        system_message = self._build_system_message(tot_context)
        messages.append({"role": "system", "content": system_message})

        # Add history if provided
        if history:
            messages.extend(history)

        # Add memory context if available
        if self.memory:
            # User preferences/history
            memory_context = self.get_memory_context(
                max_tokens=1000,
                query=input_text
            )
            if memory_context:
                messages.append({"role": "system", "content": f"## User Preferences\nThe user previously mentioned:\n{memory_context}"})

            # Self-learning: Agent's own learned knowledge
            if self.self_learning:
                agent_memory = self.get_memory_context(
                    max_tokens=500,
                    query=input_text
                )
                if agent_memory:
                    messages.append({"role": "system", "content": f"## Relevant Supplementary Information\n{agent_memory}"})

        # Add user message
        messages.append({"role": "user", "content": input_text})

        # Store messages for potential history access
        self.messages = messages.copy()

        # Convert tools to OpenAI format (use filtered tools if available)
        if active_tools:
            openai_tools = active_tools
        else:
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
            if self.memory and final_response:
                self.add_memory(
                    content=f"User: {input_text}\nAssistant: {final_response[:500]}",
                    source="interaction",
                    metadata={"role": "interaction", "type": "completion"}
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

        # Run Tree of Thought if enabled
        tot_context = ""
        if self.tree_of_thought:
            self.logger.info("[LIGHT-AGENT-MANUAL] Running Tree of Thought...")
            tot_response, _ = await self.run_thought(input_text)
            tot_context = tot_response
            self.logger.debug(f"[LIGHT-AGENT-MANUAL] ToT context added")

        # Build messages
        messages = []

        # Add system message (includes tool descriptions and ToT context)
        system_message = self._build_system_message(tot_context)
        messages.append({"role": "system", "content": system_message})

        # Add history if provided
        if history:
            messages.extend(history)

        # Add memory context if available
        if self.memory:
            # User preferences/history
            memory_context = self.get_memory_context(
                max_tokens=1000,
                query=input_text
            )
            if memory_context:
                messages.append({"role": "system", "content": f"## User Preferences\nThe user previously mentioned:\n{memory_context}"})

            # Self-learning: Agent's own learned knowledge
            if self.self_learning:
                agent_memory = self.get_memory_context(
                    max_tokens=500,
                    query=input_text
                )
                if agent_memory:
                    messages.append({"role": "system", "content": f"## Relevant Supplementary Information\n{agent_memory}"})

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
            if self.memory and final_response:
                self.add_memory(
                    content=f"User: {input_text}\nAssistant: {final_response[:500]}",
                    source="interaction",
                    metadata={"role": "interaction", "type": "completion"}
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

    def get_tool(self, tool_name: str) -> BaseTool:
        """Get a loaded tool by name.

        Args:
            tool_name: Name of the tool to retrieve

        Returns:
            The tool instance

        Raises:
            ValueError: If tool is not found
        """
        if tool_name in self.tool_map:
            return self.tool_map[tool_name]
        raise ValueError(f"Tool `{tool_name}` is not loaded.")

    async def run_thought(self, query: str) -> tuple:
        """Use Tree of Thought reasoning to plan tool usage.

        This method:
        1. Generates an initial plan with tool usage
        2. Reflects on the plan to refine it
        3. Extracts and filters the most relevant tools

        Args:
            query: The user's query to analyze

        Returns:
            Tuple of (refined_reasoning, filtered_tools)
        """
        if not self.tree_of_thought:
            self.logger.warning("[LIGHT-AGENT] run_thought called but tree_of_thought is disabled")
            return "", []

        now = datetime.now()
        current_date = now.strftime("%Y-%m-%d")
        current_time = now.strftime("%H:%M:%S")

        # Build tool descriptions
        tools_str = self._get_tools_description_str()

        system_prompt = f"""You are an intelligent assistant. Based on the user's question, analyze the task and plan which tools to use.

Today's date: {current_date}
Current time: {current_time}

Available tools:
{tools_str}

Please analyze the task step by step and identify which tools are needed."""

        self.logger.debug(f"[LIGHT-AGENT-TOT] Starting Tree of Thought for: {query}")

        try:
            # Phase 1: Initial thought generation
            tot_client = self.tot_llm if isinstance(self.tot_llm, AsyncOpenAI) else self.llm

            params = {
                "model": self.tot_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                "temperature": 0.8,  # Higher temperature for creative reasoning
            }

            response = await tot_client.chat.completions.create(**params)
            thought_response = response.choices[0].message.content
            self.logger.debug(f"[LIGHT-AGENT-TOT] Initial thought: {thought_response[:200]}...")

            # Phase 2: Reflection
            reflection_prompt = """Please reflect on your answer. Ensure you only use tools from the <Available tools> list.
Do not create or mention tools that don't exist. Output a refined plan focusing ONLY on available tools."""

            reflection_params = {
                "model": self.tot_model,
                "messages": [
                    {"role": "user", "content": f"{system_prompt}\n\nQuestion: {query}"},
                    {"role": "assistant", "content": thought_response},
                    {"role": "user", "content": reflection_prompt}
                ],
                "temperature": 0.7,
            }

            reflection_response = await tot_client.chat.completions.create(**reflection_params)
            refined_content = reflection_response.choices[0].message.content
            self.logger.debug(f"[LIGHT-AGENT-TOT] Reflection: {refined_content[:200]}...")

            # Phase 3: Extract tool names
            if self.filter_tools:
                tool_reflection_prompt = """Based on the analysis, output ONLY a JSON object listing the tools needed.
Use this exact format:
{"tools": [{"name": "tool_name1"}, {"name": "tool_name2"}]}

Only include tools from the available tools list. No explanations, just JSON."""

                tool_params = {
                    "model": self.tot_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Question: {query}"},
                        {"role": "assistant", "content": refined_content},
                        {"role": "user", "content": tool_reflection_prompt}
                    ],
                    "temperature": 0.3,  # Lower temperature for structured output
                }

                tool_response = await tot_client.chat.completions.create(**tool_params)
                tool_reflection_result = tool_response.choices[0].message.content
                self.logger.debug(f"[LIGHT-AGENT-TOT] Tool extraction: {tool_reflection_result}")

                # Filter tools
                filtered_tools = self._filter_tools_from_response(tool_reflection_result)
                self.logger.info(f"[LIGHT-AGENT-TOT] Filtered {len(filtered_tools)} tools")

                return refined_content, filtered_tools
            else:
                return refined_content, []

        except Exception as e:
            self.logger.exception(f"[LIGHT-AGENT-TOT] Error during Tree of Thought: {e}")
            return "", []

    def _get_tools_description_str(self) -> str:
        """Get formatted string of all available tools."""
        tool_descriptions = []
        for tool in self.tools:
            tool_descriptions.append(f"- {tool.name}: {tool.description}")
        return "\n".join(tool_descriptions)

    def _filter_tools_from_response(self, tool_reflection_result: str) -> List[Dict[str, Any]]:
        """Filter tools based on LLM response.

        Args:
            tool_reflection_result: JSON string with tool names

        Returns:
            List of tool schemas in OpenAI format
        """
        try:
            # Clean up JSON (remove markdown code blocks if present)
            refined_content = tool_reflection_result.strip()
            if refined_content.startswith('```json') and refined_content.endswith('```'):
                refined_content = refined_content[7:-3].strip()
            elif refined_content.startswith('```') and refined_content.endswith('```'):
                refined_content = refined_content[3:-3].strip()

            # Parse JSON
            parsed_data = json.loads(refined_content)
            valid_tool_names = {tool["name"].strip().lower() for tool in parsed_data.get("tools", [])}

            # Convert tools to OpenAI format
            openai_tools = self._convert_tools_to_openai_format()

            # Filter based on valid tool names
            filtered = [
                schema for schema in openai_tools
                if isinstance(schema, dict) and
                   schema.get("function", {}).get("name", "").strip().lower() in valid_tool_names
            ]

            return filtered

        except (json.JSONDecodeError, KeyError, AttributeError) as e:
            self.logger.exception(f"[LIGHT-AGENT] Tool filtering failed: {e}")
            return []

    async def _handle_task_transfer(
        self,
        query: str,
        light_swarm: Any,
        return_metrics: bool,
        history: Optional[List[Dict[str, Any]]]
    ) -> Optional[Union[str, BaseModel, AgentResponse]]:
        """Handle task transfer to another agent if needed.

        Args:
            query: User's query
            light_swarm: LightSwarm instance containing registered agents
            return_metrics: Whether to return metrics
            history: Conversation history

        Returns:
            Result from transferred agent, or None if no transfer needed
        """
        try:
            intent = await self._detect_intent(query, light_swarm)
            if intent and intent.get("transfer_to"):
                target_agent_name = intent["transfer_to"]
                self.logger.info(f"[LIGHT-AGENT-TRANSFER] Detected transfer to: {target_agent_name}")

                # Don't transfer to self
                if target_agent_name == self.agent_name:
                    self.logger.info("[LIGHT-AGENT-TRANSFER] Transfer to self detected, ignoring")
                    return None

                # Get target agent
                if not hasattr(light_swarm, 'agents') or target_agent_name not in light_swarm.agents:
                    self.logger.warning(f"[LIGHT-AGENT-TRANSFER] Target agent '{target_agent_name}' not found")
                    return None

                target_agent = light_swarm.agents[target_agent_name]

                # Transfer to target agent
                self.logger.info(f"[LIGHT-AGENT-TRANSFER] Transferring from {self.agent_name} to {target_agent_name}")
                return await target_agent.run(
                    input_data=query,
                    return_metrics=return_metrics,
                    history=history,
                    light_swarm=light_swarm
                )

            return None

        except Exception as e:
            self.logger.exception(f"[LIGHT-AGENT-TRANSFER] Error during task transfer: {e}")
            return None

    async def _detect_intent(self, query: str, light_swarm: Any) -> Optional[Dict[str, str]]:
        """Detect if the query should be transferred to another agent.

        Args:
            query: User's query
            light_swarm: LightSwarm instance

        Returns:
            Dictionary with transfer_to key if transfer is needed, None otherwise
        """
        if not light_swarm or not hasattr(light_swarm, 'agents'):
            return None

        # Build agent information
        agents_info = []
        for agent_name, agent in light_swarm.agents.items():
            if hasattr(agent, 'instructions'):
                agents_info.append(f"Agent: {agent_name}, Instructions: {agent.instructions}")
            else:
                agents_info.append(f"Agent: {agent_name}")

        agents_info_str = "\n".join(agents_info)

        # Prompt for intent detection
        prompt = f"""Analyze the user's request and determine if it should be handled by a different agent.

Available agents:
{agents_info_str}

User request: {query}

If the request should be transferred to another agent, respond with ONLY:
transfer to <agent_name>

Otherwise, respond with:
no transfer

Your response:"""

        try:
            # Use main LLM for intent detection
            response = await self.llm.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": prompt}],
                temperature=0.3,
                max_tokens=50
            )

            intent_text = response.choices[0].message.content.strip().lower()
            self.logger.debug(f"[LIGHT-AGENT-INTENT] Detected: {intent_text}")

            # Parse intent
            for agent_name in light_swarm.agents.keys():
                if f"transfer to {agent_name.lower()}" in intent_text:
                    return {"transfer_to": agent_name}

            return None

        except Exception as e:
            self.logger.exception(f"[LIGHT-AGENT-INTENT] Error detecting intent: {e}")
            return None

    async def create_tool(self, user_input: str) -> Optional[str]:
        """Create a new tool dynamically based on user description.

        This method uses the LLM to generate Python code for a new tool
        based on the user's description.

        Args:
            user_input: Description of the tool to create

        Returns:
            The name of the created tool, or None if creation failed
        """
        system_prompt = """You are a Python code generator. Generate a tool function based on the user's description.

Output ONLY valid JSON in this format:
{
    "tool_name": "function_name",
    "tool_code": "import statements\\ndef function_name(param: type) -> str:\\n    ...\\n    return result"
}

The tool_code should:
1. Include necessary imports
2. Have a descriptive function name
3. Include type hints
4. Have a docstring
5. Return a string result

Example:
{
    "tool_name": "get_weather",
    "tool_code": "import requests\\ndef get_weather(city: str) -> str:\\n    \\\"\\\"\\\"Get weather for a city\\\"\\\"\\\"\\n    # Implementation\\n    return 'Weather data'"
}"""

        try:
            response = await self.llm.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Create a tool for: {user_input}"}
                ],
                temperature=0.7
            )

            response_text = response.choices[0].message.content
            self.logger.debug(f"[LIGHT-AGENT-CREATE-TOOL] Generated: {response_text[:200]}...")

            # Parse JSON response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                tool_data = json.loads(json_match.group())
            else:
                tool_data = json.loads(response_text)

            tool_name = tool_data.get("tool_name")
            tool_code = tool_data.get("tool_code")

            if not tool_name or not tool_code:
                self.logger.error("[LIGHT-AGENT-CREATE-TOOL] Missing tool_name or tool_code")
                return None

            self.logger.info(f"[LIGHT-AGENT-CREATE-TOOL] Created tool: {tool_name}")
            self.logger.debug(f"[LIGHT-AGENT-CREATE-TOOL] Code:\n{tool_code}")

            # Note: Actually registering the tool would require dynamic code execution
            # which is a security risk. This method primarily demonstrates the capability.
            # In production, generated code should be reviewed before execution.

            return tool_name

        except Exception as e:
            self.logger.exception(f"[LIGHT-AGENT-CREATE-TOOL] Error creating tool: {e}")
            return None

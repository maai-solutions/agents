"""ReasoningAgent class implementation."""

from typing import List, Dict, Any, Optional, Type, Union
import json
import re
import time
import asyncio
from pydantic import BaseModel
from loguru import logger
from openai import OpenAI, AsyncOpenAI

from linus.agents.agent.memory import MemoryManager

from .base import Agent
from .models import ReasoningResult, TaskExecution, AgentMetrics, AgentResponse
from .tool_base import BaseTool
from ..graph.state import SharedState
from ..telemetry import get_tracer, AgentTracer, trace_method

# Try to import rich for enhanced logging
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import track
    RICH_AVAILABLE = True
    _console = Console()
except ImportError:
    RICH_AVAILABLE = False
    _console = None


class ReasoningAgent(Agent):
    """Agent that uses two-call approach for Gemma3:27b without tool support.

    First call: Reasoning phase to analyze the task and plan actions
    Second call: Execution phase to generate tool arguments and execute
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
        max_iterations: int = 10,
        memory_manager: Optional[MemoryManager] = None,
        memory_context_ratio: float = 0.3,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        api_base: Optional[str] = None,
        use_json_format: bool = False
    ):
        """Initialize the reasoning agent.

        Args:
            llm: OpenAI client instance (AsyncOpenAI or OpenAI)
            model: Model name to use (e.g., "gemma3:27b")
            tools: List of available tools
            verbose: Whether to print debug information
            input_schema: Optional Pydantic BaseModel for structured input validation
            output_schema: Optional Pydantic BaseModel for structured output
            output_key: Optional key to save output in shared state
            state: Optional SharedState instance for state management
            max_iterations: Maximum number of reasoning-execution loops before stopping
            memory_manager: Optional memory manager for context persistence
            memory_context_ratio: Ratio of context window to use for memory (0.0 to 1.0)
            temperature: Sampling temperature for LLM calls
            max_tokens: Maximum tokens to generate in completion
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            api_base: Optional API base URL for reference
            use_json_format: Whether to use response_format={"type": "json_object"} (not all models support this)
        """
        super().__init__(llm, model, tools, verbose, input_schema, output_schema, output_key, state, memory_manager)
        self.reasoning_prompt = self._create_reasoning_prompt()
        self.execution_prompt = self._create_execution_prompt()
        self.completion_check_prompt = self._create_completion_check_prompt()
        self.max_iterations = max_iterations
        self.current_metrics: Optional[AgentMetrics] = None
        self.memory_context_ratio = max(0.0, min(1.0, memory_context_ratio))  # Clamp to 0-1

        # LLM generation parameters
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.api_base = api_base
        self.use_json_format = use_json_format

        # Telemetry tracer
        self.tracer = get_tracer()

    def _get_generation_kwargs(self) -> Dict[str, Any]:
        """Build kwargs for LLM generation with configured parameters.

        Returns:
            Dictionary of generation parameters for OpenAI API calls
        """
        kwargs = {
            "model": self.model,
            "temperature": self.temperature
        }

        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens

        if self.top_p is not None:
            kwargs["top_p"] = self.top_p

        # Note: top_k is not part of OpenAI API spec, but Ollama supports it as extra_body
        if self.top_k is not None:
            kwargs["extra_body"] = {"top_k": self.top_k}

        return kwargs

    def _create_reasoning_prompt(self) -> str:
        """Create the prompt template for the reasoning phase."""
        tool_descriptions = "\n".join([
            f"- {tool.name}: {tool.description}"
            for tool in self.tools
        ])

        return f"""You are an AI assistant that helps plan and execute tasks.

Available tools:
{tool_descriptions}

Your task is to analyze the user's request and determine:
1. Whether you have enough information to attempt the task
2. What specific steps/tasks need to be performed
3. Which tools (if any) are needed for each step

IMPORTANT GUIDELINES:
- Set "has_sufficient_info" to true if you can ATTEMPT the task, even if success is uncertain
- Only set "has_sufficient_info" to false if the request is completely unclear or missing critical parameters that cannot be inferred
- If a tool might not return results, plan alternative tools or approaches in subsequent tasks
- When previous tools returned no results, plan different tools or different queries
- Be persistent and creative in planning alternative approaches

IMPORTANT: You must respond with ONLY valid JSON. Do not include any text before or after the JSON.

Respond in the following JSON format:
{{
    "has_sufficient_info": true,
    "reasoning": "Your analysis of the request",
    "tasks": [
        {{
            "description": "Description of the task",
            "tool_name": "Name of tool to use (or null if no tool needed)",
            "requires_user_input": false
        }}
    ]
}}

User request: """

    def _create_execution_prompt(self) -> str:
        """Create the prompt template for the execution phase."""
        return """You are an AI assistant that generates tool arguments in JSON format.

Given a task and a tool, generate the appropriate arguments for the tool call.

Tool: {tool_name}
Tool Description: {tool_description}
Tool Parameters Schema: {tool_parameters}

Task: {task_description}
Context: {context}

CRITICAL INSTRUCTIONS:
1. Respond with ONLY a valid JSON object containing the tool arguments
2. Do NOT include any explanatory text before or after the JSON
3. Do NOT use markdown code blocks (no ```json or ```)
4. Match the exact parameter names from the schema
5. Use appropriate data types (strings, numbers, booleans, objects, arrays)

Example format: {{"query": "search term", "limit": 10}}

Generate the JSON tool arguments now:"""

    def _create_completion_check_prompt(self) -> str:
        """Create the prompt template for checking task completion."""
        return """You are an AI assistant that validates task completion.

Original Request: {original_request}

Execution History:
{execution_history}

Based on the original request and the execution history, determine if the task has been completed successfully.

IMPORTANT GUIDELINES:
- If a tool returned no results or empty data, the task is NOT complete unless all reasonable alternatives have been tried
- Suggest trying alternative tools or different query approaches when tools return no results
- Only mark as complete when the user's request has been successfully answered or when all reasonable attempts have been exhausted
- Be persistent: if one approach didn't work, plan to try another approach rather than giving up

Respond in the following JSON format:
{{
    "is_complete": true/false,
    "reasoning": "Explanation of why the task is or isn't complete",
    "missing_steps": ["List of steps still needed (empty if complete)"],
    "next_action": "What should be done next (or 'none' if complete)"
}}

Response:"""

    async def run(self, input_data: Union[str, BaseModel, Dict[str, Any]], return_metrics: bool = True) -> Union[str, BaseModel, AgentResponse]:
        """Run the agent using an iterative reasoning-execution loop with validation.

        Args:
            input_data: The user's request (string, Pydantic model, or dict)
            return_metrics: If True, return AgentResponse with metrics; if False, return only the result

        Returns:
            AgentResponse (if return_metrics=True) or just the result (if return_metrics=False)
        """
        # Validate and convert input
        input_text = self._validate_and_convert_input(input_data)

        # Start tracing for agent run (use async with for async compatibility)
        async with self.tracer.trace_agent_run(user_input=input_text, agent_type="ReasoningAgent") as trace:
            return await self._run_with_trace(input_text, return_metrics, trace)

    async def _run_with_trace(self, input_text: str, return_metrics: bool, trace: Any = None) -> Union[str, BaseModel, AgentResponse]:
        """Internal run method with tracing support."""
        # Initialize metrics
        metrics = AgentMetrics()
        self.current_metrics = metrics
        start_time = time.time()
        logger.info(f"[RUN] Starting task: {input_text}")

        # Store user input in memory
        if self.memory_manager:
            self.memory_manager.add_memory(
                content=f"User: {input_text}",
                metadata={"role": "user", "type": "input"},
                importance=1.0,
                entry_type="interaction"
            )

        # Track execution history for all iterations
        execution_history = []
        iteration = 0
        is_complete = False
        completion_status = None

        # Agent loop: keep reasoning and executing until task is complete or max iterations reached
        while not is_complete and iteration < self.max_iterations:
            iteration += 1
            # Update metrics
            if self.current_metrics:
                self.current_metrics.total_iterations = iteration
            logger.info(f"[RUN] === Iteration {iteration}/{self.max_iterations} ===")

            # Phase 1: Reasoning
            # Build context from previous iterations and memory
            context = input_text

            # Add memory context if available
            if self.memory_manager:
                memory_tokens = int(self.memory_manager.max_context_tokens * self.memory_context_ratio)
                
                # Create a contextual query that evolves with the conversation
                # On first iteration, use the original question
                # On subsequent iterations, include context from recent execution results
                if iteration == 1:
                    memory_query = input_text
                else:
                    # Use the most recent execution results to create a more specific query
                    recent_context = input_text
                    if execution_history:
                        recent_tasks = [item['task'] for item in execution_history[-2:]]  # Last 2 tasks
                        recent_context = f"{input_text} | Recent work: {' | '.join(recent_tasks)}"
                    memory_query = recent_context
                
                memory_context = self.memory_manager.get_context(
                    max_tokens=memory_tokens,
                    include_summary=True,
                    query=memory_query
                )
                if memory_context:
                    context = f"{memory_context}\n\n=== Current Task ===\n{context}"
                    logger.debug(f"[MEMORY] Added {memory_tokens} token memory context with query: {memory_query[:100]}...")

            # Add state context if available
            state_data = self.state.get_all()
            if state_data:
                state_context = f"\n\nShared state: {json.dumps({k: str(v) for k, v in state_data.items()})}"
                context = context + state_context

            if execution_history:
                history_context = "\n\nPrevious execution results:\n" + "\n".join([
                    f"- {item['task']}: {item['result'][:200]}" for item in execution_history[-5:]  # Last 5 items
                ])
                context = context + history_context

                # Trace context being passed to reasoning
                self.tracer.add_event("reasoning_input_with_history", {
                    "iteration": iteration,
                    "has_execution_history": True,
                    "history_items_count": len(execution_history),
                    "context_preview": context[-500:] if len(context) > 500 else context,
                    "context_length": len(context)
                })

            # Trace the reasoning phase with iteration context
            with self.tracer.trace_reasoning_phase(input_text=context, iteration=iteration) as reasoning_span:
                reasoning_result = self._reasoning_call(context)

                # Update reasoning span with results for Langfuse
                if hasattr(reasoning_span, 'update'):
                    reasoning_span.update(
                        output={
                            "has_sufficient_info": reasoning_result.has_sufficient_info,
                            "tasks_planned": len(reasoning_result.tasks),
                            "reasoning": reasoning_result.reasoning[:200]
                        }
                    )
            logger.debug(f"[RUN] Reasoning result: {reasoning_result}")

            # If no sufficient info and this is the first iteration, exit early
            # But allow retries in subsequent iterations with updated context
            if not reasoning_result.has_sufficient_info and iteration == 1:
                result = f"I need more information to complete this task. {reasoning_result.reasoning}"
                logger.warning(f"[RUN] Insufficient information: {reasoning_result.reasoning}")

                # Return immediately only if there are no tasks planned
                if not reasoning_result.tasks:
                    return self._format_output(result)

                # If there are tasks planned, continue to execute them
                logger.info(f"[RUN] Proceeding with {len(reasoning_result.tasks)} planned tasks despite insufficient info flag")

            # Phase 2: Execute planned tasks
            iteration_results = []
            for task_data in reasoning_result.tasks:
                task = TaskExecution(
                    description=task_data["description"],
                    tool_name=task_data.get("tool_name")
                )

                logger.info(f"[RUN] Executing task: {task.description}")

                if task.tool_name:
                    # Generate tool arguments and execute
                    task_result = await self._execute_task_with_tool(task, context)
                    iteration_results.append(task_result)

                    # Record in execution history
                    execution_history.append({
                        "iteration": iteration,
                        "task": task.description,
                        "tool": task.tool_name,
                        "result": task_result,
                        "status": "completed" if task.completed else "failed"
                    })

                    # Update context with results for subsequent tasks
                    context = f"{context}\n\nLatest result: {task_result}"

                    # Trace context update for visibility
                    self.tracer.add_event("context_updated_with_tool_result", {
                        "tool_name": task.tool_name,
                        "result_preview": str(task_result)[:200],
                        "context_length": len(context)
                    })
                    logger.debug(f"[RUN] Context updated with tool result: {str(task_result)[:200]}")
                else:
                    # Direct LLM response without tool
                    response = await self._generate_response(task.description, context)
                    iteration_results.append(response)

                    execution_history.append({
                        "iteration": iteration,
                        "task": task.description,
                        "tool": None,
                        "result": response,
                        "status": "completed"
                    })

                    context = f"{context}\n\nLatest result: {response}"

                    # Trace context update
                    self.tracer.add_event("context_updated_with_llm_response", {
                        "response_preview": response[:200],
                        "context_length": len(context)
                    })
                    logger.debug(f"[RUN] Context updated with LLM response: {response[:200]}")

            # Phase 3: Check if task is complete
            completion_status = await self._check_completion(input_text, execution_history)
            is_complete = completion_status["is_complete"]

            logger.info(f"[RUN] Completion check - Complete: {is_complete}, Reason: {completion_status['reasoning']}")

            # Trace completion check result
            self.tracer.add_event("task_completion_checked", {
                "is_complete": is_complete,
                "iteration": iteration,
                "reasoning": completion_status['reasoning'][:200]
            })

            if not is_complete and iteration < self.max_iterations:
                logger.info(f"[RUN] Task not complete. Next action: {completion_status['next_action']}")
                logger.info(f"[RUN] Missing steps: {completion_status['missing_steps']}")

                # Trace iteration continuation
                self.tracer.add_event("continuing_to_next_iteration", {
                    "next_iteration": iteration + 1,
                    "next_action": completion_status['next_action'],
                    "missing_steps": completion_status['missing_steps'],
                    "accumulated_context_length": len(context)
                })
                # Loop continues with updated context
            elif not is_complete and iteration >= self.max_iterations:
                logger.warning(f"[RUN] Max iterations ({self.max_iterations}) reached without completion")
                break

        # Calculate final metrics
        metrics.total_iterations = iteration
        metrics.execution_time_seconds = time.time() - start_time
        metrics.task_completed = is_complete
        metrics.iterations_to_completion = iteration if is_complete else None

        # Task completed successfully
        if is_complete:
            logger.info(f"[RUN] Task completed successfully in {iteration} iteration(s)")
        else:
            logger.warning(f"[RUN] Task incomplete after {iteration} iteration(s)")

        final_result = await self._format_final_response_with_history(input_text, execution_history, completion_status)

        # Format output according to schema and save to state
        formatted_result = self._format_output(final_result)

        # Store agent response in memory
        if self.memory_manager:
            self.memory_manager.add_memory(
                content=f"Assistant: {str(formatted_result)[:500]}",  # Limit length
                metadata={
                    "role": "assistant",
                    "type": "output",
                    "iterations": iteration,
                    "completed": is_complete
                },
                importance=1.0,
                entry_type="interaction"
            )

            # Log memory stats
            mem_stats = self.memory_manager.get_memory_stats()
            logger.debug(f"[MEMORY] Stats: {mem_stats}")

        # Log final metrics with rich formatting if available
        if RICH_AVAILABLE and _console:
            self._display_metrics_rich(metrics)
        else:
            logger.info(f"[METRICS] {metrics.to_dict()}")

        # Return based on return_metrics flag
        if return_metrics:
            return AgentResponse(
                result=formatted_result,
                metrics=metrics,
                execution_history=execution_history,
                completion_status=completion_status
            )
        else:
            return formatted_result

    async def arun(self, input_data: Union[str, BaseModel, Dict[str, Any]], return_metrics: bool = True) -> Union[str, BaseModel, AgentResponse]:
        """Async version: Run the agent using an iterative reasoning-execution loop with validation.

        Args:
            input_data: The user's request (string, Pydantic model, or dict)
            return_metrics: If True, return AgentResponse with metrics; if False, return only the result

        Returns:
            AgentResponse (if return_metrics=True) or just the result (if return_metrics=False)
        """
        # Initialize metrics
        metrics = AgentMetrics()
        self.current_metrics = metrics
        start_time = time.time()

        # Validate and convert input
        input_text = self._validate_and_convert_input(input_data)
        logger.info(f"[ARUN] Starting task: {input_text}")

        # Store user input in memory
        if self.memory_manager:
            self.memory_manager.add_memory(
                content=f"User: {input_text}",
                metadata={"role": "user", "type": "input"},
                importance=1.0,
                entry_type="interaction"
            )

        # Track execution history for all iterations
        execution_history = []
        iteration = 0
        is_complete = False
        completion_status = None

        # Agent loop: keep reasoning and executing until task is complete or max iterations reached
        while not is_complete and iteration < self.max_iterations:
            iteration += 1
            logger.info(f"[ARUN] === Iteration {iteration}/{self.max_iterations} ===")

            # Phase 1: Reasoning
            # Build context from previous iterations and memory
            context = input_text

            # Add memory context if available
            if self.memory_manager:
                memory_tokens = int(self.memory_manager.max_context_tokens * self.memory_context_ratio)
                
                # Create a contextual query that evolves with the conversation
                # On first iteration, use the original question
                # On subsequent iterations, include context from recent execution results
                if iteration == 1:
                    memory_query = input_text
                else:
                    # Use the most recent execution results to create a more specific query
                    recent_context = input_text
                    if execution_history:
                        recent_tasks = [item['task'] for item in execution_history[-2:]]  # Last 2 tasks
                        recent_context = f"{input_text} | Recent work: {' | '.join(recent_tasks)}"
                    memory_query = recent_context
                
                memory_context = self.memory_manager.get_context(
                    max_tokens=memory_tokens,
                    include_summary=True,
                    query=memory_query
                )
                if memory_context:
                    context = f"{memory_context}\n\n=== Current Task ===\n{context}"
                    logger.debug(f"[MEMORY] Added {memory_tokens} token memory context with query: {memory_query[:100]}...")

            # Add state context if available
            state_data = self.state.get_all()
            if state_data:
                state_context = f"\n\nShared state: {json.dumps({k: str(v) for k, v in state_data.items()})}"
                context = context + state_context

            if execution_history:
                history_context = "\n\nPrevious execution results:\n" + "\n".join([
                    f"- {item['task']}: {item['result'][:200]}" for item in execution_history[-5:]  # Last 5 items
                ])
                context = context + history_context

                # Trace context being passed to reasoning
                self.tracer.add_event("reasoning_input_with_history", {
                    "iteration": iteration,
                    "has_execution_history": True,
                    "history_items_count": len(execution_history),
                    "context_preview": context[-500:] if len(context) > 500 else context,
                    "context_length": len(context)
                })

            # Trace the reasoning phase with iteration context
            with self.tracer.trace_reasoning_phase(input_text=context, iteration=iteration) as reasoning_span:
                reasoning_result = await self._areasoning_call(context)

                # Update reasoning span with results for Langfuse
                if hasattr(reasoning_span, 'update'):
                    reasoning_span.update(
                        output={
                            "has_sufficient_info": reasoning_result.has_sufficient_info,
                            "tasks_planned": len(reasoning_result.tasks),
                            "reasoning": reasoning_result.reasoning[:200]
                        }
                    )
            logger.debug(f"[ARUN] Reasoning result: {reasoning_result}")

            # If no sufficient info and this is the first iteration, exit early
            # But allow retries in subsequent iterations with updated context
            if not reasoning_result.has_sufficient_info and iteration == 1:
                result = f"I need more information to complete this task. {reasoning_result.reasoning}"
                logger.warning(f"[ARUN] Insufficient information: {reasoning_result.reasoning}")

                # Return immediately only if there are no tasks planned
                if not reasoning_result.tasks:
                    return self._format_output(result)

                # If there are tasks planned, continue to execute them
                logger.info(f"[ARUN] Proceeding with {len(reasoning_result.tasks)} planned tasks despite insufficient info flag")

            # Phase 2: Execute planned tasks
            iteration_results = []
            for task_data in reasoning_result.tasks:
                task = TaskExecution(
                    description=task_data["description"],
                    tool_name=task_data.get("tool_name")
                )

                logger.info(f"[ARUN] Executing task: {task.description}")

                if task.tool_name:
                    # Generate tool arguments and execute
                    task_result = await self._aexecute_task_with_tool(task, context)
                    iteration_results.append(task_result)

                    # Record in execution history
                    execution_history.append({
                        "iteration": iteration,
                        "task": task.description,
                        "tool": task.tool_name,
                        "result": task_result,
                        "status": "completed" if task.completed else "failed"
                    })

                    # Update context with results for subsequent tasks
                    context = f"{context}\n\nLatest result: {task_result}"

                    # Trace context update for visibility
                    self.tracer.add_event("context_updated_with_tool_result", {
                        "tool_name": task.tool_name,
                        "result_preview": str(task_result)[:200],
                        "context_length": len(context)
                    })
                    logger.debug(f"[ARUN] Context updated with tool result: {str(task_result)[:200]}")
                else:
                    # Direct LLM response without tool
                    response = await self._agenerate_response(task.description, context)
                    iteration_results.append(response)

                    execution_history.append({
                        "iteration": iteration,
                        "task": task.description,
                        "tool": None,
                        "result": response,
                        "status": "completed"
                    })

                    context = f"{context}\n\nLatest result: {response}"

                    # Trace context update
                    self.tracer.add_event("context_updated_with_llm_response", {
                        "response_preview": response[:200],
                        "context_length": len(context)
                    })
                    logger.debug(f"[ARUN] Context updated with LLM response: {response[:200]}")

            # Phase 3: Check if task is complete
            completion_status = await self._acheck_completion(input_text, execution_history)
            is_complete = completion_status["is_complete"]

            logger.info(f"[ARUN] Completion check - Complete: {is_complete}, Reason: {completion_status['reasoning']}")

            # Trace completion check result
            self.tracer.add_event("task_completion_checked", {
                "is_complete": is_complete,
                "iteration": iteration,
                "reasoning": completion_status['reasoning'][:200]
            })

            if not is_complete and iteration < self.max_iterations:
                logger.info(f"[ARUN] Task not complete. Next action: {completion_status['next_action']}")
                logger.info(f"[ARUN] Missing steps: {completion_status['missing_steps']}")

                # Trace iteration continuation
                self.tracer.add_event("continuing_to_next_iteration", {
                    "next_iteration": iteration + 1,
                    "next_action": completion_status['next_action'],
                    "missing_steps": completion_status['missing_steps'],
                    "accumulated_context_length": len(context)
                })
                # Loop continues with updated context
            elif not is_complete and iteration >= self.max_iterations:
                logger.warning(f"[ARUN] Max iterations ({self.max_iterations}) reached without completion")
                break

        # Calculate final metrics
        metrics.total_iterations = iteration
        metrics.execution_time_seconds = time.time() - start_time
        metrics.task_completed = is_complete
        metrics.iterations_to_completion = iteration if is_complete else None

        # Task completed successfully
        if is_complete:
            logger.info(f"[ARUN] Task completed successfully in {iteration} iteration(s)")
        else:
            logger.warning(f"[ARUN] Task incomplete after {iteration} iteration(s)")

        final_result = self._format_final_response_with_history(input_text, execution_history, completion_status)

        # Format output according to schema and save to state
        formatted_result = self._format_output(final_result)

        # Store agent response in memory
        if self.memory_manager:
            self.memory_manager.add_memory(
                content=f"Assistant: {str(formatted_result)[:500]}",  # Limit length
                metadata={
                    "completed": is_complete,
                    "iterations": iteration,
                    "execution_time": metrics.execution_time_seconds
                }
            )

        # Record metrics in telemetry
        self.tracer.record_metrics(metrics.to_dict())

        # Flush traces to ensure they're sent to Langfuse
        if hasattr(self.tracer, 'flush'):
            self.tracer.flush()

        # Log final metrics with rich formatting if available
        if RICH_AVAILABLE and _console:
            self._display_metrics_rich(metrics)
        else:
            logger.info(f"[METRICS] {metrics.to_dict()}")

        # Return based on return_metrics flag
        if return_metrics:
            return AgentResponse(
                result=formatted_result,
                metrics=metrics,
                execution_history=execution_history,
                completion_status=completion_status
            )
        else:
            return formatted_result

    @trace_method("agent.reasoning")
    async def _format_final_response(self, original_request: str, results: List[str]) -> str:
        """Format the final response from all task results (legacy method).

        Args:
            original_request: The original user request
            results: List of results from executed tasks

        Returns:
            ReasoningResult containing the analysis and planned tasks
        """
        messages = [
            {"role": "system", "content": self.reasoning_prompt},
            {"role": "user", "content": input_text}
        ]

        logger.debug(f"[REASONING] Input text: {input_text}")
        logger.debug(f"[REASONING] System prompt: {self.reasoning_prompt}")
        logger.debug(f"[REASONING] Messages: {messages}")

        # Add telemetry for LLM call
        self.tracer.set_attribute("llm.input", input_text[:500])
        self.tracer.set_attribute("llm.model", self.model)

        # Try to force JSON response format if supported
        gen_kwargs = self._get_generation_kwargs()

        logger.debug(f"[REASONING] params: {json.dumps(gen_kwargs, indent=2)}")

        # Add response_format for models that support it (OpenAI, some others)
        if self.use_json_format:
            try:
                gen_kwargs["response_format"] = {"type": "json_object"}
                logger.debug("[REASONING] Using response_format=json_object")
            except Exception:
                pass  # Some models don't support this parameter

        # Trace LLM call
        with self.tracer.trace_llm_call(
            prompt=f"{self.reasoning_prompt}\n\nUser: {input_text}",
            model=self.model,
            call_type="reasoning"
        ) as llm_span:
            response = self.llm.chat.completions.create(
                **gen_kwargs,
                messages=messages,
            )
            response_text = response.choices[0].message.content

            # Update Langfuse generation with output and usage
            if hasattr(self.tracer, 'update_generation') and hasattr(llm_span, 'update'):
                usage_dict = None
                if hasattr(response, 'usage') and response.usage:
                    usage_dict = {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    }
                    llm_span.update(
                        output=response_text,
                        usage=usage_dict
                    )

        logger.debug(f"[REASONING] Raw response: {response_text}")

        # Record LLM output
        self.tracer.set_attribute("llm.output", response_text[:500])

        # Track metrics
        if self.current_metrics:
            self.current_metrics.reasoning_calls += 1
            self._update_token_usage(response)

        try:
            # Parse JSON response
            # Extract JSON from the response (in case there's extra text)
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                response_data = json.loads(json_match.group())
            else:
                response_data = json.loads(response_text)

            result = ReasoningResult(
                has_sufficient_info=response_data.get("has_sufficient_info", False),
                tasks=response_data.get("tasks", []),
                reasoning=response_data.get("reasoning", "")
            )
            logger.debug(f"[REASONING] Parsed result: {result}")
            return result
        except (json.JSONDecodeError, KeyError) as e:
            logger.exception(f"[REASONING] Error parsing response: {e}")
            logger.error(f"[REASONING] Raw response was: {response_text[:500]}")
            # Fallback: treat as insufficient info
            return ReasoningResult(
                has_sufficient_info=False,
                tasks=[],
                reasoning=f"Failed to parse reasoning response. Model returned: {response_text[:200]}"
            )
    
    @trace_method("agent.execute_task")
    def _execute_task_with_tool(self, task: TaskExecution, context: str) -> str:
        """Execute a task that requires a tool.

        Args:
            task: The task to execute
            context: Current context including previous results

        Returns:
            The result of the tool execution
        """
        logger.debug(f"[EXECUTION] Task: {task.description}")
        logger.debug(f"[EXECUTION] Tool: {task.tool_name}")
        logger.debug(f"[EXECUTION] Context: {context}")

        # Add telemetry
        self.tracer.set_attribute("task.description", task.description)
        self.tracer.set_attribute("task.tool_name", task.tool_name or "none")

        if task.tool_name not in self.tool_map:
            logger.error(f"[EXECUTION] Tool '{task.tool_name}' not found in tool_map")
            self.tracer.add_event("tool_not_found", {"tool_name": task.tool_name})
            return f"Error: Tool '{task.tool_name}' not found"

        tool = self.tool_map[task.tool_name]

        # Generate tool arguments
        tool_args = self._generate_tool_arguments(task, tool, context)

        if tool_args is None:
            logger.error(f"[EXECUTION] Failed to generate arguments for tool '{task.tool_name}'")
            self.tracer.add_event("tool_args_generation_failed", {"tool_name": task.tool_name})
            return f"Error: Failed to generate arguments for tool '{task.tool_name}'"

        # Execute the tool
        try:
            logger.debug(f"[EXECUTION] Executing tool {task.tool_name} with args: {tool_args}")

            # Add telemetry for tool execution
            self.tracer.set_attribute("tool.args", str(tool_args)[:500])

            # Track metrics
            if self.current_metrics:
                self.current_metrics.tool_executions += 1

            # Trace tool execution
            with self.tracer.trace_tool_execution(
                tool_name=task.tool_name,
                tool_args=tool_args
            ) as tool_span:
                result = tool.run(tool_args)

                # Update tool span with result
                if hasattr(tool_span, 'update'):
                    tool_span.update(output={"result": str(result)[:500]})

            logger.debug(f"[EXECUTION] Tool result: {result}")

            # Add telemetry for tool result
            self.tracer.set_attribute("tool.result", str(result)[:500])
            self.tracer.add_event("tool_execution_completed", {
                "tool_name": task.tool_name,
                "result_length": len(str(result)),
                "status": "success"
            })

            task.completed = True
            task.result = result

            # Track successful tool call
            if self.current_metrics:
                self.current_metrics.successful_tool_calls += 1

            return str(result)
        except Exception as e:
            logger.exception(f"[EXECUTION] Error executing tool {task.tool_name}: {e}")

            # Track failed tool call
            if self.current_metrics:
                self.current_metrics.failed_tool_calls += 1

            return f"Error executing tool '{task.tool_name}': {str(e)}"
    
    def _generate_tool_arguments(self, task: TaskExecution, tool: BaseTool, context: str) -> Optional[Dict[str, Any]]:
        """Generate arguments for a tool call using the LLM.
        
        Args:
            task: The task requiring the tool
            tool: The tool to be called
            context: Current context
            
        Returns:
            Dictionary of tool arguments or None if generation failed
        """
        # Get tool parameter schema
        tool_params = {}
        if hasattr(tool, 'args_schema') and tool.args_schema:
            # Extract parameters from Pydantic model
            tool_params = tool.args_schema.model_json_schema()
        
        prompt = self.execution_prompt.format(
            tool_name=tool.name,
            tool_description=tool.description,
            tool_parameters=json.dumps(tool_params, indent=2),
            task_description=task.description,
            context=context
        )

        logger.debug(f"[TOOL_ARGS] Tool name: {tool.name}")
        logger.debug(f"[TOOL_ARGS] Tool params schema: {tool_params}")
        logger.debug(f"[TOOL_ARGS] Prompt: {prompt}")

        messages = [
            {"role": "system", "content": "Summarize the following task results into a coherent response."},
            {"role": "user", "content": f"Original request: {original_request}\n\nResults:\n{combined}"}
        ]

        response = await self.llm.chat.completions.create(
            messages=messages,
            **self._get_generation_kwargs()
        )
        response_text = response.choices[0].message.content
        logger.debug(f"[FINAL] Final response: {response_text}")

        # Track metrics
        self._update_token_usage(response)

        return response_text

    @trace_method("agent.final_formatting")
    async def _format_final_response_with_history(self, input_text: str, execution_history: List[Dict[str, Any]], completion_status: Dict[str, Any]) -> str:
        """Format the final response using execution history and completion status.

        Args:
            input_text: The original user request
            execution_history: List of executed tasks with their results
            completion_status: Dictionary containing completion status and reasoning

        Returns:
            Formatted final response
        """
        logger.debug(f"[FINAL-HISTORY] Formatting response for: {input_text}")
        logger.debug(f"[FINAL-HISTORY] Execution history: {len(execution_history)} items")
        logger.debug(f"[FINAL-HISTORY] Completion status: {completion_status}")

        # Extract results from execution history
        task_results = []
        for item in execution_history:
            if item.get('status') == 'completed' and item.get('result'):
                task_results.append(item['result'])

        # If we have no successful results, use the completion status reasoning
        if not task_results:
            if completion_status.get('reasoning'):
                return completion_status['reasoning']
            else:
                return "I apologize, but I was unable to find relevant information to answer your question."

        # If we have only one result, return it directly
        if len(task_results) == 1:
            return task_results[0]

        # Combine multiple results with context
        combined_results = "\n\n".join([
            f"Finding {i+1}: {result}"
            for i, result in enumerate(task_results)
        ])

        # Create a comprehensive final response using LLM
        messages = [
            {"role": "system", "content": """You are an assistant that synthesizes information from multiple sources to provide comprehensive answers. 
            Given the original question and findings from various tools/searches, create a coherent, well-structured response that:
            1. Directly answers the original question
            2. Integrates information from all findings
            3. Provides clear, factual information
            4. Is well-organized and easy to read"""},
            {"role": "user", "content": f"""Original question: {input_text}

Research findings:
{combined_results}

Please provide a comprehensive answer to the original question based on these findings."""}
        ]

        try:
            response = await self.llm.chat.completions.create(
                messages=messages,
                **self._get_generation_kwargs()
            )
            response_text = response.choices[0].message.content
            
            # Track metrics
            self._update_token_usage(response)
            
            logger.debug(f"[FINAL-HISTORY] Generated response: {response_text[:200]}...")
            return response_text
            
        except Exception as e:
            logger.exception(f"[FINAL-HISTORY] Error generating final response: {e}")
            # Fallback to simple concatenation
            return f"Based on my research:\n\n{combined_results}"

    
    async def _reasoning_call(self, input_text: str, iteration: int = 1) -> ReasoningResult:
        """Async version: Perform the reasoning phase.

        Args:
            input_text: The user's request
            iteration: Current iteration number

        Returns:
            ReasoningResult containing the analysis and planned tasks
        """
        # Trace the reasoning phase
        async with self.tracer.trace_reasoning_phase(input_text, iteration):
            messages = [
                {"role": "system", "content": self.reasoning_prompt},
                {"role": "user", "content": input_text}
            ]

            logger.debug(f"[REASONING] Input text: {input_text}")

            # Try to force JSON response format if supported
            gen_kwargs = self._get_generation_kwargs()

            # Add response_format for models that support it
            if self.use_json_format:
                try:
                    gen_kwargs["response_format"] = {"type": "json_object"}
                    logger.debug("[REASONING] Using response_format=json_object")
                except Exception:
                    pass

            # Trace the LLM call within the reasoning phase
            async with self.tracer.trace_llm_call(messages[1]["content"], self.model, "reasoning"):
                response = await self.llm.chat.completions.create(
                    messages=messages,
                    **gen_kwargs
                )
                response_text = response.choices[0].message.content
                logger.debug(f"[REASONING] Raw response: {response_text}")

                # Extract usage if available
                usage = None
                if hasattr(response, 'usage') and response.usage:
                    usage = {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    }

                # Parse JSON response
                try:
                    # Extract JSON from the response (in case there's extra text)
                    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                    if json_match:
                        response_data = json.loads(json_match.group())
                    else:
                        response_data = json.loads(response_text)

                    result = ReasoningResult(
                        has_sufficient_info=response_data.get("has_sufficient_info", False),
                        tasks=response_data.get("tasks", []),
                        reasoning=response_data.get("reasoning", "")
                    )
                    logger.debug(f"[REASONING] Parsed result: {result}")

                    # Update generation with parsed output WHILE still inside the context
                    if hasattr(self.tracer, 'update_generation'):
                        output_data = {
                            "has_sufficient_info": result.has_sufficient_info,
                            "reasoning": result.reasoning,
                            "tasks_count": len(result.tasks),
                            "tasks": result.tasks
                        }
                        self.tracer.update_generation(output=output_data, usage=usage)
                        logger.debug(f"[REASONING] Updated generation with parsed output")

                except (json.JSONDecodeError, KeyError) as e:
                    logger.exception(f"[REASONING] Error parsing response: {e}")
                    logger.error(f"[REASONING] Raw response was: {response_text[:500]}")

                    # Update generation with error
                    if hasattr(self.tracer, 'update_generation'):
                        error_output = {
                            "error": str(e),
                            "failed_to_parse": response_text[:500]
                        }
                        self.tracer.update_generation(output=error_output, usage=usage)
                        logger.debug(f"[REASONING] Updated generation with error")

                    result = ReasoningResult(
                        has_sufficient_info=False,
                        tasks=[],
                        reasoning=f"Failed to parse reasoning response. Model returned: {response_text[:200]}"
                    )

            # After LLM call completes, update the reasoning_phase span with the result
            # This must happen INSIDE the reasoning_phase context but AFTER the LLM context closes
            if hasattr(self.tracer, 'client') and self.tracer.enabled:
                try:
                    span_output = {
                        "has_sufficient_info": result.has_sufficient_info,
                        "reasoning": result.reasoning,
                        "tasks_count": len(result.tasks),
                        "tasks": result.tasks
                    }
                    self.tracer.client.update_current_span(output=span_output)
                    logger.debug(f"[REASONING] Updated reasoning_phase span with output")
                except Exception as e:
                    logger.warning(f"[REASONING] Failed to update reasoning_phase span: {e}")

            # Track metrics (after LLM context closes)
            if self.current_metrics:
                self.current_metrics.reasoning_calls += 1
                self._update_token_usage(response)

            return result

    async def _execute_task_with_tool(self, task: TaskExecution, context: str) -> str:
        """Async version: Execute a task that requires a tool.

        Args:
            task: The task to execute
            context: Current context including previous results

        Returns:
            The result of the tool execution
        """
        logger.debug(f"[EXECUTION] Task: {task.description}")
        logger.debug(f"[EXECUTION] Tool: {task.tool_name}")

        if task.tool_name not in self.tool_map:
            logger.error(f"[EXECUTION] Tool not found: {task.tool_name}")
            return f"Error: Tool '{task.tool_name}' not available"

        tool = self.tool_map[task.tool_name]

        # Generate arguments for the tool
        tool_args = await self._generate_tool_arguments(task, tool, context)
        if tool_args is None:
            task.completed = False
            if self.current_metrics:
                self.current_metrics.failed_tool_calls += 1
            return f"Error: Could not generate valid arguments for tool '{task.tool_name}'"

        # Execute the tool
        try:
            logger.info(f"[EXECUTION] Executing {task.tool_name} with args: {tool_args}")

            # Trace tool execution
            async with self.tracer.trace_tool_execution(task.tool_name, tool_args) as tool_span:
                # Execute tool asynchronously
                result = await tool.arun(tool_args)

            logger.info(f"[ASYNC-EXECUTION] Tool result: {result}")

            # Add telemetry for tool execution completion
            self.tracer.add_event("tool_execution_completed", {
                "tool_name": task.tool_name,
                "result_length": len(str(result)),
                "status": "success"
            })

            task.completed = True
            task.result = result

                # Update tool span with result - for spans we can use update methods
                if tool_span and hasattr(self.tracer, 'client'):
                    try:
                        # Update current span with output
                        self.tracer.client.update_current_span(output=str(result))
                    except Exception as e:
                        logger.warning(f"Failed to update tool span: {e}")

            # Track metrics
            if self.current_metrics:
                self.current_metrics.tool_executions += 1
                self.current_metrics.successful_tool_calls += 1

            return str(result)
        except Exception as e:
            logger.exception(f"[EXECUTION] Tool execution failed: {e}")
            task.completed = False
            if self.current_metrics:
                self.current_metrics.tool_executions += 1
                self.current_metrics.failed_tool_calls += 1
            return f"Error executing tool: {str(e)}"

    async def _generate_tool_arguments(self, task: TaskExecution, tool: BaseTool, context: str) -> Optional[Dict[str, Any]]:
        """Async version: Generate arguments for a tool call.

        Args:
            task: The task requiring tool execution
            tool: The tool to use
            context: Current context

        Returns:
            Dictionary of tool arguments or None if generation failed
        """
        # Get tool schema using Pydantic v2 model_json_schema()
        tool_schema = {}
        if hasattr(tool, 'args_schema') and tool.args_schema:
            if hasattr(tool.args_schema, 'model_json_schema'):
                tool_schema = tool.args_schema.model_json_schema()
            else:
                logger.warning(f"[TOOL-ARGS] Tool {tool.name} has args_schema without model_json_schema()")
                # Fallback to input_schema_dict if available
                if hasattr(tool, 'input_schema_dict'):
                    tool_schema = tool.input_schema_dict

        prompt = self.execution_prompt.format(
            tool_name=tool.name,
            tool_description=tool.description,
            tool_parameters=json.dumps(tool_schema, indent=2),
            task_description=task.description,
            context=context
        )

        messages = [
            {"role": "system", "content": "You are a helpful assistant that generates tool arguments."},
            {"role": "user", "content": prompt}
        ]

        logger.debug(f"[TOOL-ARGS] Generating args for {tool.name}")

        # Use lower temperature for more consistent JSON generation
        kwargs = self._get_generation_kwargs()
        kwargs["temperature"] = min(0.3, kwargs.get("temperature", 0.7))

        # Trace the tool argument generation LLM call
        async with self.tracer.trace_llm_call(prompt, self.model, "tool_args"):
            response = await self.llm.chat.completions.create(
                messages=messages,
                **kwargs
            )

            # Get response text
            response_text = response.choices[0].message.content

            # Extract usage if available
            usage = None
            if hasattr(response, 'usage') and response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }

            # Parse JSON arguments and update generation with parsed output
            try:
                # Parse JSON arguments with multiple fallback strategies
                args = None

                # Strategy 1: Try to extract JSON from markdown code blocks
                code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
                if code_block_match:
                    logger.debug(f"[TOOL-ARGS] Found JSON in markdown code block")
                    args = json.loads(code_block_match.group(1))
                else:
                    # Strategy 2: Extract first JSON object from response
                    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                    if json_match:
                        args = json.loads(json_match.group())
                    else:
                        # Strategy 3: Try parsing the entire response as JSON
                        args = json.loads(response_text.strip())

                logger.debug(f"[TOOL-ARGS] Generated args: {args}")

                # Update generation with parsed arguments as structured output
                # MUST be called inside the trace_llm_call context
                if hasattr(self.tracer, 'update_generation'):
                    self.tracer.update_generation(output=args, usage=usage)
                    logger.debug(f"[TOOL-ARGS] Updated generation with parsed args: {args}")

            except (json.JSONDecodeError, KeyError) as e:
                logger.exception(f"[TOOL-ARGS] Error parsing arguments: {e}")
                logger.error(f"[TOOL-ARGS] Failed to parse response: {response_text}")

                # Update generation with error information
                # MUST be called inside the trace_llm_call context
                if hasattr(self.tracer, 'update_generation'):
                    error_output = {
                        "error": str(e),
                        "failed_to_parse": response_text[:500]
                    }
                    self.tracer.update_generation(output=error_output, usage=usage)
                    logger.debug(f"[TOOL-ARGS] Updated generation with error: {str(e)}")

                args = None

        # Track metrics (after LLM context closes)
        self._update_token_usage(response)

        return args

    async def _generate_response(self, task_description: str, context: str) -> str:
        """Async version: Generate a direct response without using tools.

        Args:
            task_description: Description of the task
            context: Current context

        Returns:
            Generated response
        """
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant. Provide a clear and concise response to the task."},
            {"role": "user", "content": f"Context: {context}\n\nTask: {task_description}\n\nYour response:"}
        ]

        logger.debug(f"[ASYNC-RESPONSE] Generating response for: {task_description}")

        response = await self.llm.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7
        )

        # Track metrics
        self._update_token_usage(response)

        response_text = response.choices[0].message.content
        logger.debug(f"[ASYNC-RESPONSE] Generated: {response_text}")
        return response_text

    async def _check_completion(self, original_request: str, execution_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Async version: Check if the task has been completed.

        Args:
            original_request: The original user request
            execution_history: History of task executions

        Returns:
            Dictionary with completion status and reasoning
        """
        history_summary = "\n".join([
            f"- {item['task']}: {item['result'][:200]}"
            for item in execution_history
        ])

        prompt = self.completion_check_prompt.format(
            original_request=original_request,
            execution_history=history_summary
        )

        messages = [
            {"role": "system", "content": self.completion_check_prompt},
            {"role": "user", "content": f"Original request: {original_request}\n\nExecution history:\n{history_summary}\n\nIs the task complete?"}
        ]

        logger.debug(f"[ASYNC-COMPLETION] Checking completion for: {original_request}")

        response = await self.llm.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7
        )

        # Track metrics
        if self.current_metrics:
            self.current_metrics.completion_checks += 1
        self._update_token_usage(response)

        try:
            response_text = response.choices[0].message.content
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                completion_status = json.loads(json_match.group())
            else:
                completion_status = json.loads(response_text)

            logger.debug(f"[ASYNC-COMPLETION] Status: {completion_status}")
            return completion_status
        except (json.JSONDecodeError, KeyError) as e:
            logger.exception(f"[ASYNC-COMPLETION] Error parsing completion status: {e}")
            # Default to continuing if we can't parse
            return {
                "is_complete": False,
                "reasoning": "Failed to parse completion check",
                "next_action": "Continue with next iteration",
                "missing_steps": []
            }

    def _display_metrics_rich(self, metrics: AgentMetrics):
        """Display metrics using rich formatting.

        Args:
            metrics: AgentMetrics instance to display
        """
        if not RICH_AVAILABLE or not _console:
            return

        # Create metrics table
        table = Table(title="🤖 Agent Execution Metrics", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")

        metrics_data = metrics.to_dict()

        # Format metrics for display
        formatted_metrics = {
            "Total Iterations": metrics_data.get("total_iterations", 0),
            "Total Tokens": f"{metrics_data.get('total_tokens', 0):,}",
            "Execution Time": f"{metrics_data.get('execution_time_seconds', 0):.2f}s",
            "LLM Calls": metrics_data.get("llm_calls", 0),
            "Tool Executions": metrics_data.get("tool_executions", 0),
            "Successful Tools": metrics_data.get("successful_tool_calls", 0),
            "Failed Tools": metrics_data.get("failed_tool_calls", 0),
            "Task Completed": "✅ Yes" if metrics_data.get("task_completed") else "❌ No",
            "Avg Tokens/Call": f"{metrics_data.get('avg_tokens_per_llm_call', 0):.1f}",
            "Success Rate": f"{metrics_data.get('success_rate', 0):.1%}"
        }

        for key, value in formatted_metrics.items():
            table.add_row(key, str(value))

        _console.print(table)

    def _display_reasoning_rich(self, reasoning: ReasoningResult):
        """Display reasoning results using rich formatting.

        Args:
            reasoning: ReasoningResult to display
        """
        if not RICH_AVAILABLE or not _console:
            return

        # Create panel for reasoning
        panel_content = f"[yellow]{reasoning.reasoning}[/yellow]\n\n"
        panel_content += "[bold cyan]Planned Tasks:[/bold cyan]\n"

        for i, task in enumerate(reasoning.tasks, 1):
            tool_name = task.get("tool_name", "None")
            panel_content += f"  {i}. {task['description']} [dim](Tool: {tool_name})[/dim]\n"

        _console.print(Panel(panel_content, title="🧠 Reasoning Phase", border_style="blue"))



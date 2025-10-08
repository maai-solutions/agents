"""ReasoningAgent class implementation."""

from typing import List, Dict, Any, Optional, Type, Union
import json
import re
import time
import asyncio
from pydantic import BaseModel
from loguru import logger
from openai import OpenAI, AsyncOpenAI

from .base import Agent
from .models import ReasoningResult, TaskExecution, AgentMetrics, AgentResponse
from .tool_base import BaseTool
from ..graph.state import SharedState


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
        memory_manager: Optional['MemoryManager'] = None,
        memory_context_ratio: float = 0.3,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None
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
1. Whether you have enough information to complete the task
2. What specific steps/tasks need to be performed
3. Which tools (if any) are needed for each step

Respond in the following JSON format:
{{
    "has_sufficient_info": true/false,
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
        return """You are an AI assistant that generates tool arguments.

Given a task and a tool, generate the appropriate arguments for the tool call.

Tool: {tool_name}
Tool Description: {tool_description}
Tool Parameters: {tool_parameters}

Task: {task_description}
Context: {context}

Respond with ONLY a valid JSON object containing the tool arguments.
Example: {{"query": "search term", "limit": 10}}

Tool arguments:"""

    def _create_completion_check_prompt(self) -> str:
        """Create the prompt template for checking task completion."""
        return """You are an AI assistant that validates task completion.

Original Request: {original_request}

Execution History:
{execution_history}

Based on the original request and the execution history, determine if the task has been completed successfully.

Respond in the following JSON format:
{{
    "is_complete": true/false,
    "reasoning": "Explanation of why the task is or isn't complete",
    "missing_steps": ["List of steps still needed (empty if complete)"],
    "next_action": "What should be done next (or 'none' if complete)"
}}

Response:"""

    def run(self, input_data: Union[str, BaseModel, Dict[str, Any]], return_metrics: bool = True) -> Union[str, BaseModel, AgentResponse]:
        """Run the agent using an iterative reasoning-execution loop with validation.

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
            logger.info(f"[RUN] === Iteration {iteration}/{self.max_iterations} ===")

            # Phase 1: Reasoning
            # Build context from previous iterations and memory
            context = input_text

            # Add memory context if available
            if self.memory_manager:
                memory_tokens = int(self.memory_manager.max_context_tokens * self.memory_context_ratio)
                memory_context = self.memory_manager.get_context(
                    max_tokens=memory_tokens,
                    include_summary=True,
                    query=input_text
                )
                if memory_context:
                    context = f"{memory_context}\n\n=== Current Task ===\n{context}"
                    logger.debug(f"[MEMORY] Added {memory_tokens} token memory context")

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

            reasoning_result = self._reasoning_call(context)
            logger.debug(f"[RUN] Reasoning result: {reasoning_result}")

            if not reasoning_result.has_sufficient_info:
                result = f"I need more information to complete this task. {reasoning_result.reasoning}"
                logger.warning(f"[RUN] Insufficient information: {reasoning_result.reasoning}")
                return self._format_output(result)

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
                    task_result = self._execute_task_with_tool(task, context)
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
                else:
                    # Direct LLM response without tool
                    response = self._generate_response(task.description, context)
                    iteration_results.append(response)

                    execution_history.append({
                        "iteration": iteration,
                        "task": task.description,
                        "tool": None,
                        "result": response,
                        "status": "completed"
                    })

                    context = f"{context}\n\nLatest result: {response}"

            # Phase 3: Check if task is complete
            completion_status = self._check_completion(input_text, execution_history)
            is_complete = completion_status["is_complete"]

            logger.info(f"[RUN] Completion check - Complete: {is_complete}, Reason: {completion_status['reasoning']}")

            if not is_complete and iteration < self.max_iterations:
                logger.info(f"[RUN] Task not complete. Next action: {completion_status['next_action']}")
                logger.info(f"[RUN] Missing steps: {completion_status['missing_steps']}")
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

        final_result = self._format_final_response_with_history(input_text, execution_history, completion_status)

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

        # Log final metrics
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
                memory_context = self.memory_manager.get_context(
                    max_tokens=memory_tokens,
                    include_summary=True,
                    query=input_text
                )
                if memory_context:
                    context = f"{memory_context}\n\n=== Current Task ===\n{context}"
                    logger.debug(f"[MEMORY] Added {memory_tokens} token memory context")

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

            reasoning_result = await self._areasoning_call(context)
            logger.debug(f"[ARUN] Reasoning result: {reasoning_result}")

            if not reasoning_result.has_sufficient_info:
                result = f"I need more information to complete this task. {reasoning_result.reasoning}"
                logger.warning(f"[ARUN] Insufficient information: {reasoning_result.reasoning}")
                return self._format_output(result)

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

            # Phase 3: Check if task is complete
            completion_status = await self._acheck_completion(input_text, execution_history)
            is_complete = completion_status["is_complete"]

            logger.info(f"[ARUN] Completion check - Complete: {is_complete}, Reason: {completion_status['reasoning']}")

            if not is_complete and iteration < self.max_iterations:
                logger.info(f"[ARUN] Task not complete. Next action: {completion_status['next_action']}")
                logger.info(f"[ARUN] Missing steps: {completion_status['missing_steps']}")
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

        # Log final metrics
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

    def _reasoning_call(self, input_text: str) -> ReasoningResult:
        """Perform the reasoning phase.

        Args:
            input_text: The user's request

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

        response = self.llm.chat.completions.create(
            messages=messages,
            **self._get_generation_kwargs()
        )
        response_text = response.choices[0].message.content
        logger.debug(f"[REASONING] Raw response: {response_text}")

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
            logger.error(f"[REASONING] Error parsing response: {e}")
            logger.debug(f"[REASONING] Failed to parse: {response_text}")
            # Fallback: treat as insufficient info
            return ReasoningResult(
                has_sufficient_info=False,
                tasks=[],
                reasoning="Failed to parse reasoning response"
            )
    
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

        if task.tool_name not in self.tool_map:
            logger.error(f"[EXECUTION] Tool '{task.tool_name}' not found in tool_map")
            return f"Error: Tool '{task.tool_name}' not found"

        tool = self.tool_map[task.tool_name]

        # Generate tool arguments
        tool_args = self._generate_tool_arguments(task, tool, context)
        
        if tool_args is None:
            logger.error(f"[EXECUTION] Failed to generate arguments for tool '{task.tool_name}'")
            return f"Error: Failed to generate arguments for tool '{task.tool_name}'"

        # Execute the tool
        try:
            logger.debug(f"[EXECUTION] Executing tool {task.tool_name} with args: {tool_args}")

            # Track metrics
            if self.current_metrics:
                self.current_metrics.tool_executions += 1

            result = tool.run(tool_args)
            logger.debug(f"[EXECUTION] Tool result: {result}")
            task.completed = True
            task.result = result

            # Track successful tool call
            if self.current_metrics:
                self.current_metrics.successful_tool_calls += 1

            return str(result)
        except Exception as e:
            logger.error(f"[EXECUTION] Error executing tool {task.tool_name}: {e}")

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
            tool_params = tool.args_schema.schema()
        
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
            {"role": "system", "content": "You are a helpful assistant that generates tool arguments."},
            {"role": "user", "content": prompt}
        ]

        response = self.llm.chat.completions.create(
            messages=messages,
            **self._get_generation_kwargs()
        )
        response_text = response.choices[0].message.content
        logger.debug(f"[TOOL_ARGS] Raw response: {response_text}")

        # Track metrics
        self._update_token_usage(response)

        try:
            # Parse JSON arguments
            # Extract JSON from the response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                args = json.loads(json_match.group())
            else:
                args = json.loads(response_text)

            logger.debug(f"[TOOL_ARGS] Parsed arguments: {args}")
            return args
        except json.JSONDecodeError as e:
            logger.error(f"[TOOL_ARGS] Error parsing tool arguments: {e}")
            logger.debug(f"[TOOL_ARGS] Failed to parse: {response_text}")
            return None
    
    def _generate_response(self, task_description: str, context: str) -> str:
        """Generate a direct LLM response without tools.

        Args:
            task_description: Description of what to generate
            context: Current context

        Returns:
            The LLM's response
        """
        logger.debug(f"[GENERATE] Task description: {task_description}")
        logger.debug(f"[GENERATE] Context: {context}")

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Context: {context}\n\nTask: {task_description}"}
        ]

        response = self.llm.chat.completions.create(
            messages=messages,
            **self._get_generation_kwargs()
        )
        response_text = response.choices[0].message.content
        logger.debug(f"[GENERATE] Response: {response_text}")

        # Track metrics
        self._update_token_usage(response)

        return response_text
    
    def _check_completion(self, original_request: str, execution_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check if the task has been completed successfully.

        Args:
            original_request: The original user request
            execution_history: History of all task executions

        Returns:
            Dictionary with completion status and details
        """
        if not execution_history:
            return {
                "is_complete": False,
                "reasoning": "No tasks have been executed yet",
                "missing_steps": ["Start executing the planned tasks"],
                "next_action": "Execute the first task"
            }

        # Format execution history for the prompt
        history_text = "\n".join([
            f"Iteration {item['iteration']}: {item['task']} "
            f"[Tool: {item['tool'] or 'None'}] -> {item['result'][:300]}... (Status: {item['status']})"
            for item in execution_history
        ])

        prompt = self.completion_check_prompt.format(
            original_request=original_request,
            execution_history=history_text
        )

        messages = [
            {"role": "system", "content": "You are a task completion validator."},
            {"role": "user", "content": prompt}
        ]

        logger.debug(f"[COMPLETION_CHECK] Checking completion for: {original_request}")
        logger.debug(f"[COMPLETION_CHECK] History: {history_text[:500]}")

        response = self.llm.chat.completions.create(
            messages=messages,
            **self._get_generation_kwargs()
        )
        response_text = response.choices[0].message.content
        logger.debug(f"[COMPLETION_CHECK] Raw response: {response_text}")

        # Track metrics
        if self.current_metrics:
            self.current_metrics.completion_checks += 1
            self._update_token_usage(response)

        try:
            # Parse JSON response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                completion_data = json.loads(json_match.group())
            else:
                completion_data = json.loads(response_text)

            result = {
                "is_complete": completion_data.get("is_complete", False),
                "reasoning": completion_data.get("reasoning", ""),
                "missing_steps": completion_data.get("missing_steps", []),
                "next_action": completion_data.get("next_action", "unknown")
            }

            logger.debug(f"[COMPLETION_CHECK] Parsed result: {result}")
            return result

        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"[COMPLETION_CHECK] Error parsing completion check: {e}")
            # Conservative fallback: assume incomplete
            return {
                "is_complete": False,
                "reasoning": "Failed to parse completion check response",
                "missing_steps": ["Verify task completion manually"],
                "next_action": "retry"
            }

    def _format_final_response_with_history(
        self,
        original_request: str,
        execution_history: List[Dict[str, Any]],
        completion_status: Dict[str, Any]
    ) -> str:
        """Format the final response including execution history.

        Args:
            original_request: The original user request
            execution_history: Complete execution history
            completion_status: Completion check results

        Returns:
            Formatted final response
        """
        logger.debug(f"[FINAL] Formatting response for: {original_request}")

        # Extract all results
        results = [item["result"] for item in execution_history]

        if len(results) == 1:
            final_answer = results[0]
        else:
            # Combine multiple results with context
            combined = "\n\n".join([
                f"Step {i+1} (Iteration {item['iteration']}): {item['task']}\n"
                f"Tool: {item['tool'] or 'Direct response'}\n"
                f"Result: {item['result']}"
                for i, item in enumerate(execution_history)
            ])

            # Use LLM to create a coherent final response
            messages = [
                {"role": "system", "content": "Summarize the following task execution history into a clear, coherent final answer to the user's original request."},
                {"role": "user", "content": f"Original request: {original_request}\n\nExecution history:\n{combined}\n\nProvide a concise final answer:"}
            ]

            response = self.llm.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7
            )

            # Track metrics
            self._update_token_usage(response)

            final_answer = response.choices[0].message.content

        # Add completion status if not complete
        if not completion_status["is_complete"]:
            final_answer += f"\n\n⚠️ Note: Task may be incomplete. {completion_status['reasoning']}"
            if completion_status["missing_steps"]:
                final_answer += f"\nMissing steps: {', '.join(completion_status['missing_steps'])}"

        logger.debug(f"[FINAL] Final answer: {final_answer[:500]}")
        return final_answer

    def _format_final_response(self, original_request: str, results: List[str]) -> str:
        """Format the final response from all task results (legacy method).

        Args:
            original_request: The original user request
            results: List of results from executed tasks

        Returns:
            Formatted final response
        """
        logger.debug(f"[FINAL] Original request: {original_request}")
        logger.debug(f"[FINAL] Results: {results}")

        if len(results) == 1:
            return results[0]

        # Combine multiple results
        combined = "\n\n".join([
            f"Step {i+1}: {result}"
            for i, result in enumerate(results)
        ])

        logger.debug(f"[FINAL] Combined results: {combined}")

        # Use LLM to create a coherent final response
        messages = [
            {"role": "system", "content": "Summarize the following task results into a coherent response."},
            {"role": "user", "content": f"Original request: {original_request}\n\nResults:\n{combined}"}
        ]

        response = self.llm.chat.completions.create(
            messages=messages,
            **self._get_generation_kwargs()
        )
        response_text = response.choices[0].message.content
        logger.debug(f"[FINAL] Final response: {response_text}")

        # Track metrics
        self._update_token_usage(response)

        return response_text

    # ===== ASYNC VERSIONS OF HELPER METHODS =====

    async def _areasoning_call(self, input_text: str) -> ReasoningResult:
        """Async version: Perform the reasoning phase.

        Args:
            input_text: The user's request

        Returns:
            ReasoningResult containing the analysis and planned tasks
        """
        messages = [
            {"role": "system", "content": self.reasoning_prompt},
            {"role": "user", "content": input_text}
        ]

        logger.debug(f"[ASYNC-REASONING] Input text: {input_text}")

        response = await self.llm.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7
        )
        response_text = response.choices[0].message.content
        logger.debug(f"[ASYNC-REASONING] Raw response: {response_text}")

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
            logger.debug(f"[ASYNC-REASONING] Parsed result: {result}")
            return result
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"[ASYNC-REASONING] Error parsing response: {e}")
            return ReasoningResult(
                has_sufficient_info=False,
                tasks=[],
                reasoning="Failed to parse reasoning response"
            )

    async def _aexecute_task_with_tool(self, task: TaskExecution, context: str) -> str:
        """Async version: Execute a task that requires a tool.

        Args:
            task: The task to execute
            context: Current context including previous results

        Returns:
            The result of the tool execution
        """
        logger.debug(f"[ASYNC-EXECUTION] Task: {task.description}")
        logger.debug(f"[ASYNC-EXECUTION] Tool: {task.tool_name}")

        if task.tool_name not in self.tool_map:
            logger.error(f"[ASYNC-EXECUTION] Tool not found: {task.tool_name}")
            return f"Error: Tool '{task.tool_name}' not available"

        tool = self.tool_map[task.tool_name]

        # Generate arguments for the tool
        tool_args = await self._agenerate_tool_arguments(task, tool, context)
        if tool_args is None:
            task.completed = False
            if self.current_metrics:
                self.current_metrics.failed_tool_calls += 1
            return f"Error: Could not generate valid arguments for tool '{task.tool_name}'"

        # Execute the tool
        try:
            logger.info(f"[ASYNC-EXECUTION] Executing {task.tool_name} with args: {tool_args}")

            # Run tool in thread pool since most tools are sync
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, tool.run, tool_args)

            logger.info(f"[ASYNC-EXECUTION] Tool result: {result}")
            task.completed = True
            task.result = result

            # Track metrics
            if self.current_metrics:
                self.current_metrics.tool_executions += 1
                self.current_metrics.successful_tool_calls += 1

            return str(result)
        except Exception as e:
            logger.error(f"[ASYNC-EXECUTION] Tool execution failed: {e}")
            task.completed = False
            if self.current_metrics:
                self.current_metrics.tool_executions += 1
                self.current_metrics.failed_tool_calls += 1
            return f"Error executing tool: {str(e)}"

    async def _agenerate_tool_arguments(self, task: TaskExecution, tool: BaseTool, context: str) -> Optional[Dict[str, Any]]:
        """Async version: Generate arguments for a tool call.

        Args:
            task: The task requiring tool execution
            tool: The tool to use
            context: Current context

        Returns:
            Dictionary of tool arguments or None if generation failed
        """
        # Get tool schema
        tool_schema = tool.args_schema.schema() if hasattr(tool, 'args_schema') and tool.args_schema else {}

        prompt = self.execution_prompt.format(
            tool_name=tool.name,
            tool_description=tool.description,
            tool_parameters=json.dumps(tool_schema, indent=2),
            task_description=task.description,
            context=context
        )

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Generate the arguments for this tool call."}
        ]

        logger.debug(f"[ASYNC-TOOL-ARGS] Generating args for {tool.name}")

        response = await self.llm.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7
        )

        # Track metrics
        self._update_token_usage(response)

        try:
            # Extract JSON from response
            response_text = response.choices[0].message.content
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                args = json.loads(json_match.group())
            else:
                args = json.loads(response_text)

            logger.debug(f"[ASYNC-TOOL-ARGS] Generated args: {args}")
            return args
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"[ASYNC-TOOL-ARGS] Error parsing arguments: {e}")
            logger.debug(f"[ASYNC-TOOL-ARGS] Failed to parse: {response.content}")
            return None

    async def _agenerate_response(self, task_description: str, context: str) -> str:
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

    async def _acheck_completion(self, original_request: str, execution_history: List[Dict[str, Any]]) -> Dict[str, Any]:
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
            logger.error(f"[ASYNC-COMPLETION] Error parsing completion status: {e}")
            # Default to continuing if we can't parse
            return {
                "is_complete": False,
                "reasoning": "Failed to parse completion check",
                "next_action": "Continue with next iteration",
                "missing_steps": []
            }



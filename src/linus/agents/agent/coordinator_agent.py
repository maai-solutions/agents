"""CoordinatorAgent class implementation for orchestrating multiple subagents."""

from typing import List, Dict, Any, Optional, Type, Union
import json
import re
import time
import asyncio
from pydantic import BaseModel
from openai import OpenAI, AsyncOpenAI

from linus.agents.agent.memory import MemoryManager

from .base import Agent
from .models import AgentMetrics, AgentResponse, Citation
from .tool_base import BaseTool
from .config import AgentParams, MemoryConfig, LLMConfig
from ..graph.state import SharedState
from ..di import ILogger, ITelemetry
from ..telemetry import trace_method

# Try to import rich for enhanced logging
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.tree import Tree
    RICH_AVAILABLE = True
    _console = Console()
except ImportError:
    RICH_AVAILABLE = False
    _console = None


class SubAgent:
    """Wrapper for subagents with metadata."""

    def __init__(
        self,
        agent: Agent,
        name: str,
        description: str,
        capabilities: List[str]
    ):
        """Initialize a subagent wrapper.

        Args:
            agent: The actual agent instance
            name: Name of the subagent
            description: Description of what the subagent does
            capabilities: List of capabilities/tools the subagent provides
        """
        self.agent = agent
        self.name = name
        self.description = description
        self.capabilities = capabilities


class CoordinatorAgent(Agent):
    """Agent that coordinates multiple subagents to accomplish complex tasks.

    The coordinator:
    1. Creates a high-level plan based on the request and available subagents
    2. Executes each step using the appropriate subagent
    3. Evaluates progress after each step
    4. Recalculates the plan if needed based on results
    """

    def __init__(
        self,
        llm: Union[AsyncOpenAI, OpenAI],
        model: str,
        subagents: List[SubAgent],
        tools: Optional[List[BaseTool]] = None,
        verbose: bool = False,
        input_schema: Optional[Type[BaseModel]] = None,
        output_schema: Optional[Type[BaseModel]] = None,
        output_key: Optional[str] = None,
        state: Optional[SharedState] = None,
        max_iterations: int = 15,
        memory_manager: Optional[MemoryManager] = None,
        memory_context_ratio: float = 0.3,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        api_base: Optional[str] = None,
        use_json_format: bool = False,
        logger: Optional[ILogger] = None,
        telemetry: Optional[ITelemetry] = None,
        agent_name: Optional[str] = None
    ):
        """Initialize the coordinator agent.

        Args:
            llm: OpenAI client instance (AsyncOpenAI or OpenAI)
            model: Model name to use (e.g., "gemma3:27b")
            subagents: List of SubAgent instances to coordinate
            tools: Optional list of tools available to the coordinator (not subagents)
            verbose: Whether to print debug information
            input_schema: Optional Pydantic BaseModel for structured input validation
            output_schema: Optional Pydantic BaseModel for structured output
            output_key: Optional key to save output in shared state
            state: Optional SharedState instance for state management
            max_iterations: Maximum number of plan-execute-evaluate loops
            memory_manager: Optional memory manager for context persistence
            memory_context_ratio: Ratio of context window to use for memory (0.0 to 1.0)
            temperature: Sampling temperature for LLM calls (deprecated, use agent_params)
            max_tokens: Maximum tokens to generate in completion (deprecated, use agent_params)
            top_p: Nucleus sampling parameter (deprecated, use agent_params)
            top_k: Top-k sampling parameter (deprecated, use agent_params)
            api_base: Optional API base URL for reference (deprecated, use agent_params.llm_config)
            use_json_format: Whether to use response_format={"type": "json_object"}
            logger: Optional logger instance (uses DI container if None)
            telemetry: Optional telemetry instance (uses DI container if None)
            agent_name: Optional name for the agent (used in hierarchical tracing)
        """
        super().__init__(
            llm, model, tools or [], verbose, input_schema, output_schema,
            output_key, state, memory_manager, logger, telemetry, agent_name
        )

        self.subagents = subagents
        self.subagent_map = {sa.name: sa for sa in subagents}
        self.max_iterations = max_iterations
        self.current_metrics: Optional[AgentMetrics] = None
        self.memory_context_ratio = max(0.0, min(1.0, memory_context_ratio))

        
        # LLM generation parameters - use config if provided, otherwise use individual params
        
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.top_k = top_k
        # Create default configs if not provided
        self.memory_config = MemoryConfig()
        self.llm_config = LLMConfig(api_base=api_base or "http://localhost:11434/v1", model=model, api_key="not-needed")

        self.api_base = api_base
        self.use_json_format = use_json_format

        # Create prompts
        self.planning_prompt = self._create_planning_prompt()
        self.evaluation_prompt = self._create_evaluation_prompt()

    def _get_generation_kwargs(self) -> Dict[str, Any]:
        """Build kwargs for LLM generation with configured parameters."""
        kwargs = {
            "model": self.model,
            "temperature": self.temperature
        }

        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens

        if self.top_p is not None:
            kwargs["top_p"] = self.top_p

        if self.top_k is not None:
            kwargs["extra_body"] = {"top_k": self.top_k}

        return kwargs

    def _create_planning_prompt(self) -> str:
        """Create the prompt template for planning."""
        subagent_descriptions = "\n".join([
            f"- {sa.name}: {sa.description}\n  Capabilities: {', '.join(sa.capabilities)}"
            for sa in self.subagents
        ])

        return f"""You are a coordinator agent that orchestrates multiple specialized subagents to accomplish complex tasks.

Available subagents:
{subagent_descriptions}

Your task is to create a high-level execution plan based on:
1. The user's request
2. Available subagents and their capabilities
3. Previous execution results (if any)

IMPORTANT GUIDELINES:
- Break down complex tasks into clear, sequential steps
- Assign each step to the most appropriate subagent
- Consider dependencies between steps
- Be specific about what each step should accomplish
- Plan for potential failures or missing information

Respond in the following JSON format:
{{
    "reasoning": "Your analysis of the task and planning rationale",
    "plan": [
        {{
            "step_number": 1,
            "description": "Clear description of what this step accomplishes",
            "assigned_subagent": "subagent_name",
            "input": "What information/context to provide to the subagent",
            "expected_output": "What you expect this step to produce",
            "dependencies": ["List of previous step numbers this depends on (or empty)"]
        }}
    ],
    "overall_goal": "Summary of what the complete plan should achieve"
}}

User request: """

    def _create_evaluation_prompt(self) -> str:
        """Create the prompt template for evaluating progress."""
        return """You are evaluating the progress of a multi-step plan execution.

Original Request: {original_request}

Original Plan:
{original_plan}

Execution History:
{execution_history}

Based on the execution history, determine:
1. Whether the current plan is still valid
2. Whether the task has been completed
3. What adjustments (if any) are needed

IMPORTANT GUIDELINES:
- If a step failed or produced unexpected results, consider replanning
- If new information reveals the original plan was insufficient, replan
- Only mark as complete when the original request has been fully satisfied
- Be adaptive: unexpected results may require a different approach

Respond in the following JSON format:
{{
    "task_completed": true/false,
    "plan_still_valid": true/false,
    "evaluation_reasoning": "Explanation of current progress and any issues",
    "next_action": "continue/replan/complete",
    "suggested_changes": "If replanning needed, what should change (or null if not needed)",
    "completion_summary": "If complete, summarize what was accomplished (or null if not complete)"
}}

Response:"""

    async def run(
        self,
        input_data: Union[str, BaseModel, Dict[str, Any]],
        return_metrics: bool = True
    ) -> Union[str, BaseModel, AgentResponse]:
        """Run the coordinator agent using plan-execute-evaluate loop.

        Args:
            input_data: The user's request (string, Pydantic model, or dict)
            return_metrics: If True, return AgentResponse with metrics

        Returns:
            AgentResponse (if return_metrics=True) or just the result
        """
        # Validate and convert input
        input_text = self._validate_and_convert_input(input_data)

        # Start tracing for agent run
        async with self.telemetry.trace_agent_run(
            user_input=input_text,
            agent_type="CoordinatorAgent",
            agent_name=self.agent_name
        ) as trace:
            return await self._run_with_trace(input_text, return_metrics, trace)

    async def _run_with_trace(
        self,
        input_text: str,
        return_metrics: bool,
        trace: Any = None
    ) -> Union[str, BaseModel, AgentResponse]:
        """Internal run method with tracing support."""
        # Initialize metrics
        metrics = AgentMetrics()
        self.current_metrics = metrics
        start_time = time.time()
        self.logger.info(f"[COORDINATOR] Starting task: {input_text}")

        # Store user input in memory
        if self.memory_manager:
            self.memory_manager.add_memory(
                content=f"User: {input_text}",
                metadata={"role": "user", "type": "input"},
                importance=1.0,
                entry_type="interaction"
            )

        # Track execution history
        execution_history = []
        citations = []  # Collect citations from subagents
        iteration = 0
        current_plan = None
        is_complete = False
        evaluation_result = None
        previous_history_len = 0  # Track progress between iterations
        consecutive_continues = 0  # Track consecutive "continue" actions
        consecutive_replans = 0  # Track consecutive "replan" actions

        # Coordinator loop: plan -> execute -> evaluate -> replan if needed
        while not is_complete and iteration < self.max_iterations:
            iteration += 1
            metrics.total_iterations = iteration
            self.logger.info(f"[COORDINATOR] === Iteration {iteration}/{self.max_iterations} ===")

            # Phase 1: Planning (or replanning)
            context = self._build_context(input_text, execution_history, current_plan)
            current_plan = await self._create_plan(context, iteration)

            if not current_plan or not current_plan.get("plan"):
                self.logger.error("[COORDINATOR] Failed to create plan")
                break

            self.logger.info(f"[COORDINATOR] Plan created with {len(current_plan['plan'])} steps")
            if RICH_AVAILABLE and _console:
                self._display_plan_rich(current_plan)

            # Phase 2: Execute plan steps
            step_results = await self._execute_plan(current_plan, execution_history, input_text)

            # Add step results to history and collect citations
            for step_result in step_results:
                execution_history.append(step_result)
                # Collect citations from this step
                if "citations" in step_result and step_result["citations"]:
                    citations.extend(step_result["citations"])
                    self.logger.debug(f"[COORDINATOR] Collected {len(step_result['citations'])} citations from step {step_result['step_number']}")

            # Check for progress - if no new steps executed, we may be stalled
            if len(execution_history) == previous_history_len:
                self.logger.warning(f"[COORDINATOR] No progress made in iteration {iteration}")
            else:
                previous_history_len = len(execution_history)

            # Phase 3: Evaluate progress
            evaluation_result = await self._evaluate_progress(
                input_text,
                current_plan,
                execution_history
            )

            self.logger.info(
                f"[COORDINATOR] Evaluation - Complete: {evaluation_result['task_completed']}, "
                f"Plan valid: {evaluation_result['plan_still_valid']}, "
                f"Next: {evaluation_result['next_action']}"
            )

            # Determine next action
            if evaluation_result["next_action"] == "complete":
                is_complete = True
                consecutive_continues = 0
                consecutive_replans = 0
            elif evaluation_result["next_action"] == "replan":
                consecutive_replans += 1
                consecutive_continues = 0  # Reset continue counter

                # Force completion after 3 consecutive replans
                if consecutive_replans >= 3:
                    self.logger.warning("[COORDINATOR] Too many consecutive replans, forcing completion")
                    is_complete = True
                    evaluation_result["completion_summary"] = await self._format_final_response(input_text, execution_history)
                else:
                    self.logger.info(f"[COORDINATOR] Replanning based on evaluation (replan #{consecutive_replans})")
            elif evaluation_result["next_action"] == "continue":
                consecutive_continues += 1
                consecutive_replans = 0  # Reset replan counter

                # Force completion after 3 consecutive continues with no progress
                if consecutive_continues >= 3:
                    self.logger.warning("[COORDINATOR] Too many consecutive continues with no progress, forcing completion")
                    is_complete = True
                    evaluation_result["completion_summary"] = await self._format_final_response(input_text, execution_history)
                else:
                    # Plan was executed, check if actually complete
                    is_complete = evaluation_result["task_completed"]

            if not is_complete and iteration >= self.max_iterations:
                self.logger.warning(f"[COORDINATOR] Max iterations reached")
                break

        # Calculate final metrics
        metrics.execution_time_seconds = time.time() - start_time
        metrics.task_completed = is_complete

        # Generate final response
        if is_complete and evaluation_result and evaluation_result.get("completion_summary"):
            final_result = evaluation_result["completion_summary"]
        else:
            final_result = await self._format_final_response(input_text, execution_history)

        # Format output according to schema
        formatted_result = self._format_output(final_result)

        # Store in memory
        if self.memory_manager:
            self.memory_manager.add_memory(
                content=f"Assistant: {str(formatted_result)[:500]}",
                metadata={
                    "role": "assistant",
                    "type": "output",
                    "iterations": iteration,
                    "completed": is_complete
                },
                importance=1.0,
                entry_type="interaction"
            )

        # Update trace
        if trace and hasattr(trace, 'update'):
            trace.update(
                output=str(formatted_result),
                level="DEFAULT" if is_complete else "WARNING",
                metadata={
                    "completed": is_complete,
                    "iterations": iteration,
                    "execution_time": metrics.execution_time_seconds
                }
            )

        # Record metrics
        self.telemetry.record_metrics(metrics.to_dict())

        if hasattr(self.telemetry, 'flush'):
            self.telemetry.flush()

        # Display metrics
        if RICH_AVAILABLE and _console:
            self._display_metrics_rich(metrics)
        else:
            self.logger.info(f"[METRICS] {metrics.to_dict()}")

        # Return based on return_metrics flag
        if return_metrics:
            # Log total citations collected
            if citations:
                self.logger.info(f"[COORDINATOR] Collected {len(citations)} total citations from subagents")

            return AgentResponse(
                result=formatted_result,
                metrics=metrics,
                execution_history=execution_history,
                completion_status=evaluation_result,
                citations=citations
            )
        else:
            return formatted_result

    def _build_context(
        self,
        input_text: str,
        execution_history: List[Dict[str, Any]],
        current_plan: Optional[Dict[str, Any]]
    ) -> str:
        """Build context for planning including history and memory."""
        # If replanning (history exists), focus on what's left to do
        if execution_history:
            context = f"Task progress update for: {input_text}"
        else:
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

        # Add state context (with token management)
        from linus.agents.graph.state import StateContextStrategy

        state_context = self.state.get_context(
            strategy=getattr(self.state, 'context_strategy', StateContextStrategy.FULL),
            max_tokens=getattr(self.state, 'max_context_tokens', None),
            include_summary=True
        )
        if state_context:
            context = context + state_context

        # Add execution history if replanning
        if execution_history:
            history_context = "\n\n=== Previous Execution Results ===\n" + "\n".join([
                f"- Step {item['step_number']}: {item['subagent']} - {item['result'][:200]}"
                for item in execution_history[-10:]  # Last 10 items
            ])
            context = context + history_context

        return context

    @trace_method("coordinator.planning")
    async def _create_plan(self, context: str, iteration: int) -> Dict[str, Any]:
        """Create or update the execution plan.

        Args:
            context: Current context including request and history
            iteration: Current iteration number

        Returns:
            Dictionary containing the plan
        """
        self.logger.debug(f"[COORDINATOR-PLAN] Creating plan for iteration {iteration}")

        messages = [
            {"role": "system", "content": self.planning_prompt},
            {"role": "user", "content": context}
        ]

        gen_kwargs = self._get_generation_kwargs()
        if self.use_json_format:
            try:
                gen_kwargs["response_format"] = {"type": "json_object"}
            except Exception:
                pass

        # Combine system prompt with user context to capture the full prompt in tracing
        full_planning_prompt = f"{self.planning_prompt}{context}"
        async with self.telemetry.trace_llm_call(
            full_planning_prompt, self.model, "planning", llm_name=self.model
        ):
            response = await self.llm.chat.completions.create(
                messages=messages,
                **gen_kwargs
            )
            response_text = response.choices[0].message.content

            try:
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    plan = json.loads(json_match.group())
                else:
                    plan = json.loads(response_text)

                self.logger.debug(f"[COORDINATOR-PLAN] Created plan: {plan}")

                # Update metrics
                self._update_token_usage(response)

                # Record the generated plan in Langfuse (if Langfuse tracer is active)
                if hasattr(self.telemetry, "update_generation"):
                    # Extract usage information if available
                    usage = None
                    if hasattr(response, "usage") and response.usage:
                        usage = {
                            "prompt_tokens": response.usage.prompt_tokens,
                            "completion_tokens": response.usage.completion_tokens,
                            "total_tokens": response.usage.total_tokens,
                        }
                    self.telemetry.update_generation(
                        output=plan,
                        usage=usage,
                    )
                    self.logger.debug("[COORDINATOR-PLAN] Updated generation with plan output")

                return plan

            except (json.JSONDecodeError, KeyError) as e:
                self.logger.exception(f"[COORDINATOR-PLAN] Error parsing plan: {e}")
                return {"plan": [], "reasoning": f"Failed to parse plan: {str(e)}"}

    async def _execute_plan(
        self,
        plan: Dict[str, Any],
        execution_history: List[Dict[str, Any]],
        original_request: str
    ) -> List[Dict[str, Any]]:
        """Execute all steps in the plan.

        Args:
            plan: The execution plan
            execution_history: Previous execution history
            original_request: Original user request

        Returns:
            List of step execution results
        """
        step_results = []
        # Only count successfully completed steps for dependency checking
        # Include both previous history AND current iteration results
        completed_steps = {
            item["step_number"] for item in execution_history
            if item.get("status") == "completed"
        }

        self.logger.debug(f"[COORDINATOR-EXEC] Completed steps from history: {completed_steps}")
        self.logger.debug(f"[COORDINATOR-EXEC] Total history items: {len(execution_history)}")

        for step in plan["plan"]:
            step_number = step["step_number"]

            # Skip already completed steps
            if step_number in completed_steps:
                self.logger.info(f"[COORDINATOR-EXEC] Skipping completed step {step_number}")
                continue

            # Check dependencies
            dependencies = step.get("dependencies", [])
            if dependencies and not all(dep in completed_steps for dep in dependencies):
                self.logger.warning(
                    f"[COORDINATOR-EXEC] Step {step_number} dependencies {dependencies} not met. "
                    f"Completed: {completed_steps}"
                )
                step_results.append({
                    "step_number": step_number,
                    "subagent": step["assigned_subagent"],
                    "status": "skipped",
                    "result": f"Dependencies not met: requires {dependencies}, have {completed_steps}"
                })
                continue

            # Execute step
            result = await self._execute_step(step, execution_history, original_request)
            step_results.append(result)

            # Add to completed_steps immediately if successful so subsequent steps can depend on it
            if result.get("status") == "completed":
                completed_steps.add(step_number)
                self.logger.debug(f"[COORDINATOR-EXEC] Step {step_number} completed, updated completed_steps: {completed_steps}")

                # Store full result in shared state for other agents to access
                state_key = f"step_{step_number}_result"
                self.state.set(
                    key=state_key,
                    value=result.get("result"),
                    source=result.get("subagent"),
                    metadata={
                        "step_number": step_number,
                        "description": result.get("description"),
                        "status": result.get("status")
                    }
                )
                self.logger.debug(f"[COORDINATOR-STATE] Stored {state_key} in shared state")

        return step_results

    @trace_method("coordinator.step_execution")
    async def _execute_step(
        self,
        step: Dict[str, Any],
        execution_history: List[Dict[str, Any]],
        original_request: str
    ) -> Dict[str, Any]:
        """Execute a single plan step using the assigned subagent.

        Args:
            step: Step configuration
            execution_history: Previous execution history
            original_request: Original user request

        Returns:
            Step execution result
        """
        step_number = step["step_number"]
        subagent_name = step["assigned_subagent"]
        step_input = step["input"]

        self.logger.info(
            f"[COORDINATOR-EXEC] Step {step_number}: {step['description']} "
            f"(using {subagent_name})"
        )

        # Get the subagent
        if subagent_name not in self.subagent_map:
            self.logger.error(f"[COORDINATOR-EXEC] Subagent '{subagent_name}' not found")
            return {
                "step_number": step_number,
                "subagent": subagent_name,
                "status": "failed",
                "result": f"Subagent '{subagent_name}' not available"
            }

        subagent = self.subagent_map[subagent_name]

        # Prepare input with context
        # Only include original request for first step to avoid scope creep
        if not execution_history:
            enriched_input = f"""Original request: {original_request}

Current step: {step['description']}

{step_input}"""
        else:
            enriched_input = f"""Step objective: {step['description']}

Task details: {step_input}"""

        # Add relevant previous results from shared state
        if execution_history:
            prev_results_list = []
            for item in execution_history[-3:]:  # Last 3 steps
                step_num = item['step_number']
                state_key = f"step_{step_num}_result"
                # Get full result from shared state (not truncated)
                full_result = self.state.get(state_key)
                if full_result:
                    prev_results_list.append(f"- Step {step_num}: {full_result}")
                else:
                    # Fallback to truncated result if not in state
                    prev_results_list.append(f"- Step {step_num}: {item['result'][:150]}")

            if prev_results_list:
                enriched_input += f"\n\nPrevious results:\n" + "\n".join(prev_results_list)

        try:
            # Execute subagent (use hierarchical tracing)
            async with self.telemetry.trace_subagent_execution(
                subagent_name, enriched_input
            ):
                # Get full AgentResponse to collect citations
                result = await subagent.agent.run(enriched_input, return_metrics=True)

                self.logger.info(f"[COORDINATOR-EXEC] Step {step_number} completed")

                # Extract result text and citations
                if isinstance(result, AgentResponse):
                    result_text = str(result.result)
                    step_citations = result.citations if result.citations else []
                else:
                    result_text = str(result)
                    step_citations = []

                return {
                    "step_number": step_number,
                    "subagent": subagent_name,
                    "description": step["description"],
                    "status": "completed",
                    "result": result_text,
                    "citations": step_citations
                }

        except Exception as e:
            self.logger.exception(f"[COORDINATOR-EXEC] Step {step_number} failed: {e}")
            return {
                "step_number": step_number,
                "subagent": subagent_name,
                "description": step["description"],
                "status": "failed",
                "result": f"Error: {str(e)}",
                "citations": []
            }

    @trace_method("coordinator.evaluation")
    async def _evaluate_progress(
        self,
        original_request: str,
        current_plan: Dict[str, Any],
        execution_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Evaluate progress and determine if replanning is needed.

        Args:
            original_request: Original user request
            current_plan: Current execution plan
            execution_history: Complete execution history

        Returns:
            Evaluation result dictionary
        """
        self.logger.debug("[COORDINATOR-EVAL] Evaluating progress")

        plan_summary = "\n".join([
            f"Step {s['step_number']}: {s['description']} (using {s['assigned_subagent']})"
            for s in current_plan.get("plan", [])
        ])

        # Build history summary with full results from shared state
        history_parts = []
        for item in execution_history:
            step_num = item['step_number']
            state_key = f"step_{step_num}_result"
            # Get full result from shared state (not truncated)
            full_result = self.state.get(state_key)
            if full_result:
                history_parts.append(
                    f"Step {step_num}: {item['subagent']} - "
                    f"Status: {item['status']}, Result: {full_result}"
                )
            else:
                # Fallback to truncated result if not in state
                history_parts.append(
                    f"Step {step_num}: {item['subagent']} - "
                    f"Status: {item['status']}, Result: {item['result'][:200]}"
                )
        history_summary = "\n".join(history_parts)

        prompt = self.evaluation_prompt.format(
            original_request=original_request,
            original_plan=plan_summary,
            execution_history=history_summary
        )

        messages = [
            {"role": "system", "content": "You are an evaluator assessing task progress."},
            {"role": "user", "content": prompt}
        ]

        async with self.telemetry.trace_llm_call(
            prompt, self.model, "evaluation", llm_name=self.model
        ):
            response = await self.llm.chat.completions.create(
                messages=messages,
                **self._get_generation_kwargs()
            )
            # Record token usage
            self._update_token_usage(response)

            try:
                response_text = response.choices[0].message.content
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    evaluation = json.loads(json_match.group())
                else:
                    evaluation = json.loads(response_text)

                # Record the evaluation result in Langfuse (if Langfuse tracer is active)
                if hasattr(self.telemetry, "update_generation"):
                    usage = None
                    if hasattr(response, "usage") and response.usage:
                        usage = {
                            "prompt_tokens": response.usage.prompt_tokens,
                            "completion_tokens": response.usage.completion_tokens,
                            "total_tokens": response.usage.total_tokens,
                        }
                    self.telemetry.update_generation(
                        output=evaluation,
                        usage=usage,
                    )
                    self.logger.debug("[COORDINATOR-EVAL] Updated generation with evaluation output")

                return evaluation

            except (json.JSONDecodeError, KeyError) as e:
                self.logger.exception(f"[COORDINATOR-EVAL] Error parsing evaluation: {e}")
                return {
                    "task_completed": False,
                    "plan_still_valid": True,
                    "evaluation_reasoning": "Failed to parse evaluation",
                    "next_action": "continue",
                    "suggested_changes": None,
                    "completion_summary": None
                }

    async def _format_final_response(
        self,
        original_request: str,
        execution_history: List[Dict[str, Any]]
    ) -> str:
        """Format the final response from execution history.

        Args:
            original_request: Original user request
            execution_history: Complete execution history

        Returns:
            Formatted final response
        """
        if not execution_history:
            return "I was unable to process your request."

        successful_results = [
            item for item in execution_history
            if item.get("status") == "completed"
        ]

        if not successful_results:
            return "I attempted to process your request but encountered errors."

        if len(successful_results) == 1:
            # Get full result from shared state
            step_num = successful_results[0]["step_number"]
            state_key = f"step_{step_num}_result"
            full_result = self.state.get(state_key)
            return full_result if full_result else successful_results[0]["result"]

        # Combine multiple results with full data from shared state
        combined_parts = []
        for item in successful_results:
            step_num = item['step_number']
            state_key = f"step_{step_num}_result"
            full_result = self.state.get(state_key)
            result_text = full_result if full_result else item['result']
            combined_parts.append(f"Step {step_num} ({item['subagent']}): {result_text}")
        combined = "\n\n".join(combined_parts)

        # Use LLM to create coherent response
        messages = [
            {"role": "system", "content": "Synthesize the following step results into a comprehensive response."},
            {"role": "user", "content": f"Original request: {original_request}\n\nResults:\n{combined}"}
        ]

        try:
            response = await self.llm.chat.completions.create(
                messages=messages,
                **self._get_generation_kwargs()
            )
            self._update_token_usage(response)

            # Record the final LLM generation in Langfuse (if Langfuse tracer is active)
            if hasattr(self.telemetry, "update_generation"):
                usage = None
                if hasattr(response, "usage") and response.usage:
                    usage = {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    }
                # The response text is the final answer
                final_output = response.choices[0].message.content
                self.telemetry.update_generation(
                    output={"response": final_output},
                    usage=usage,
                )
                self.logger.debug("[COORDINATOR-FINAL] Updated generation with final response")

            return response.choices[0].message.content
        except Exception as e:
            self.logger.exception(f"[COORDINATOR] Error formatting final response: {e}")
            return combined

    def _display_plan_rich(self, plan: Dict[str, Any]):
        """Display plan using rich formatting."""
        if not RICH_AVAILABLE or not _console:
            return

        tree = Tree("📋 Execution Plan")
        tree.add(f"[yellow]Reasoning:[/yellow] {plan.get('reasoning', 'N/A')}")
        tree.add(f"[yellow]Goal:[/yellow] {plan.get('overall_goal', 'N/A')}")

        steps_node = tree.add("[bold cyan]Steps:[/bold cyan]")
        for step in plan.get("plan", []):
            step_text = (
                f"[green]Step {step['step_number']}:[/green] {step['description']}\n"
                f"  [dim]Subagent:[/dim] {step['assigned_subagent']}\n"
                f"  [dim]Expected:[/dim] {step['expected_output']}"
            )
            if step.get("dependencies"):
                step_text += f"\n  [dim]Dependencies:[/dim] {step['dependencies']}"
            steps_node.add(step_text)

        _console.print(tree)

    def _display_metrics_rich(self, metrics: AgentMetrics):
        """Display metrics using rich formatting."""
        if not RICH_AVAILABLE or not _console:
            return

        table = Table(title="🎯 Coordinator Metrics", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")

        metrics_data = metrics.to_dict()
        formatted_metrics = {
            "Total Iterations": metrics_data.get("total_iterations", 0),
            "Total Tokens": f"{metrics_data.get('total_tokens', 0):,}",
            "Execution Time": f"{metrics_data.get('execution_time_seconds', 0):.2f}s",
            "LLM Calls": metrics_data.get("llm_calls", 0),
            "Task Completed": "✅ Yes" if metrics_data.get("task_completed") else "❌ No",
        }

        for key, value in formatted_metrics.items():
            table.add_row(key, str(value))

        _console.print(table)

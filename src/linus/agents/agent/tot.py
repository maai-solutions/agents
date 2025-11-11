"""Tree of Thought Agent implementation.

This agent implements the Tree of Thought (ToT) reasoning approach where:
1. The agent uses a reasoning model to analyze the task and plan tool usage
2. It performs reflection on the initial plan to refine it
3. It filters and selects the most relevant tools for the task
4. Finally, it executes the plan using the main execution model

This is particularly useful for complex tasks that benefit from deliberate planning.
"""

from typing import List, Dict, Any, Optional, Type, Union
import json
import re
import time
from datetime import datetime
from pydantic import BaseModel
from openai import OpenAI, AsyncOpenAI

from .base import Agent
from .models import AgentMetrics, AgentResponse, Citation
from .tool_base import BaseTool
from ..graph.state import SharedState
from ..di import ILogger, ITelemetry
from linus.agents.agent.memory import MemoryManager


class ThoughtNode(BaseModel):
    """Represents a single thought in the tree of thought."""
    reasoning: str
    planned_tools: List[str]
    planned_steps: List[Dict[str, Any]]
    confidence: float = 0.0


class ReflectionResult(BaseModel):
    """Result from the reflection phase."""
    refined_reasoning: str
    refined_steps: List[Dict[str, Any]]
    tool_names: List[str]
    alternative_approaches: List[str] = []


class TreeOfThoughtAgent(Agent):
    """Agent that uses Tree of Thought reasoning for complex task planning.

    The ToT agent follows a multi-phase approach:
    1. Initial Thought Generation: Analyzes the task and generates initial plan
    2. Reflection Phase: Critically evaluates the initial plan and refines it
    3. Tool Selection: Identifies and filters the most relevant tools
    4. Execution Phase: Executes the refined plan with selected tools

    This approach is particularly effective for:
    - Complex multi-step tasks requiring careful planning
    - Tasks where tool selection is critical to success
    - Situations requiring creative problem-solving
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
        # ToT-specific parameters
        reasoning_model: Optional[str] = None,
        reasoning_llm: Optional[Union[AsyncOpenAI, OpenAI]] = None,
        enable_tool_filtering: bool = True,
        enable_reflection: bool = True,
        max_reflection_depth: int = 2,
        # LLM generation parameters
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        reasoning_temperature: float = 0.8,  # Higher temperature for creative reasoning
        logger: Optional[ILogger] = None,
        telemetry: Optional[ITelemetry] = None,
        agent_name: Optional[str] = None,
    ):
        """Initialize the Tree of Thought agent.

        Args:
            llm: OpenAI client instance for execution (AsyncOpenAI or OpenAI)
            model: Model name for execution (e.g., "gemma3:27b")
            tools: List of available tools
            verbose: Whether to print debug information
            input_schema: Optional Pydantic BaseModel for structured input validation
            output_schema: Optional Pydantic BaseModel for structured output
            output_key: Optional key to save output in shared state
            state: Optional SharedState instance for state management
            max_iterations: Maximum number of execution iterations
            memory_manager: Optional memory manager for context persistence
            reasoning_model: Model name for reasoning phase (defaults to main model)
            reasoning_llm: Separate LLM client for reasoning (defaults to main llm)
            enable_tool_filtering: Whether to filter tools based on task analysis
            enable_reflection: Whether to enable reflection phase
            max_reflection_depth: Maximum number of reflection iterations
            temperature: Sampling temperature for execution LLM calls
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            reasoning_temperature: Temperature for reasoning phase (higher for creativity)
            logger: Optional logger instance
            telemetry: Optional telemetry instance
            agent_name: Optional name for the agent
        """
        super().__init__(
            llm, model, tools, verbose, input_schema, output_schema,
            output_key, state, memory_manager, logger, telemetry, agent_name
        )

        # ToT-specific configuration
        self.reasoning_model = reasoning_model or model
        self.reasoning_llm = reasoning_llm or llm
        self.enable_tool_filtering = enable_tool_filtering
        self.enable_reflection = enable_reflection
        self.max_reflection_depth = max_reflection_depth
        self.max_iterations = max_iterations

        # LLM parameters
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.reasoning_temperature = reasoning_temperature

        # Metrics
        self.current_metrics: Optional[AgentMetrics] = None

    def _get_generation_kwargs(self, for_reasoning: bool = False) -> Dict[str, Any]:
        """Build kwargs for LLM generation.

        Args:
            for_reasoning: If True, use reasoning-specific temperature

        Returns:
            Dictionary of generation parameters
        """
        kwargs = {
            "model": self.reasoning_model if for_reasoning else self.model,
            "temperature": self.reasoning_temperature if for_reasoning else self.temperature
        }

        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens

        if self.top_p is not None:
            kwargs["top_p"] = self.top_p

        if self.top_k is not None:
            kwargs["extra_body"] = {"top_k": self.top_k}

        return kwargs

    def _create_tool_list_str(self) -> str:
        """Create a formatted string of all available tools."""
        tool_descriptions = []
        for tool in self.tools:
            tool_info = f"- **{tool.name}**: {tool.description}"
            tool_descriptions.append(tool_info)
        return "\n".join(tool_descriptions)

    async def run(
        self,
        input_data: Union[str, BaseModel, Dict[str, Any]],
        return_metrics: bool = True
    ) -> Union[str, BaseModel, AgentResponse]:
        """Run the Tree of Thought agent.

        Args:
            input_data: The user's request
            return_metrics: If True, return AgentResponse with metrics

        Returns:
            AgentResponse (if return_metrics=True) or just the result
        """
        # Validate and convert input
        input_text = self._validate_and_convert_input(input_data)

        # Start tracing for agent run
        async with self.telemetry.trace_agent_run(
            user_input=input_text,
            agent_type="TreeOfThoughtAgent",
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

        self.logger.info(f"[TOT-RUN] Starting Tree of Thought execution: {input_text}")

        # Track execution history
        execution_history = []
        citations = []

        try:
            # Phase 1: Initial Thought Generation
            self.logger.info("[TOT] === Phase 1: Initial Thought Generation ===")
            initial_thought = await self._generate_initial_thought(input_text)

            # Phase 2: Reflection (if enabled)
            reflection_result = None
            if self.enable_reflection:
                self.logger.info("[TOT] === Phase 2: Reflection and Refinement ===")
                reflection_result = await self._reflect_on_thought(
                    input_text,
                    initial_thought
                )
            else:
                # Use initial thought directly
                reflection_result = ReflectionResult(
                    refined_reasoning=initial_thought.reasoning,
                    refined_steps=initial_thought.planned_steps,
                    tool_names=initial_thought.planned_tools,
                    alternative_approaches=[]
                )

            # Phase 3: Tool Selection and Filtering
            selected_tools = self.tools
            if self.enable_tool_filtering:
                self.logger.info("[TOT] === Phase 3: Tool Selection and Filtering ===")
                selected_tools = await self._filter_tools(reflection_result.tool_names)
                self.logger.info(f"[TOT] Selected {len(selected_tools)} tools: {[t.name for t in selected_tools]}")

            # Phase 4: Execution with selected tools
            self.logger.info("[TOT] === Phase 4: Task Execution ===")
            execution_result = await self._execute_plan(
                input_text=input_text,
                plan=reflection_result,
                tools=selected_tools,
                execution_history=execution_history,
                citations=citations
            )

            # Calculate final metrics
            metrics.execution_time_seconds = time.time() - start_time
            metrics.task_completed = True

            self.logger.info(f"[TOT-RUN] Task completed in {metrics.execution_time_seconds:.2f}s")

            # Format output
            formatted_result = self._format_output(execution_result)

            # Return based on return_metrics flag
            if return_metrics:
                return AgentResponse(
                    result=formatted_result,
                    metrics=metrics,
                    execution_history=execution_history,
                    completion_status={
                        "is_complete": True,
                        "reasoning": "Task completed using Tree of Thought approach"
                    },
                    citations=citations
                )
            else:
                return formatted_result

        except Exception as e:
            self.logger.exception(f"[TOT-RUN] Error during execution: {e}")
            metrics.execution_time_seconds = time.time() - start_time
            metrics.task_completed = False

            error_message = f"Error during Tree of Thought execution: {str(e)}"

            if return_metrics:
                return AgentResponse(
                    result=error_message,
                    metrics=metrics,
                    execution_history=execution_history,
                    completion_status={
                        "is_complete": False,
                        "reasoning": str(e)
                    },
                    citations=citations
                )
            else:
                return error_message

    async def _generate_initial_thought(self, input_text: str) -> ThoughtNode:
        """Generate initial thought and plan for the task.

        Args:
            input_text: The user's request

        Returns:
            ThoughtNode with initial reasoning and plan
        """
        now = datetime.now()
        current_date = now.strftime("%Y-%m-%d")
        current_time = now.strftime("%H:%M:%S")

        tools_description = self._create_tool_list_str()

        system_prompt = f"""You are an intelligent planning assistant. Analyze the user's request and create a detailed plan.

Current date: {current_date}
Current time: {current_time}

Available tools:
{tools_description}

Your task:
1. Carefully analyze the user's request
2. Identify what information or actions are needed
3. Plan which tools should be used and in what order
4. Break down the task into clear, executable steps

Respond in the following JSON format:
{{
    "reasoning": "Your detailed analysis of the task",
    "planned_tools": ["tool_name1", "tool_name2"],
    "planned_steps": [
        {{
            "step_number": 1,
            "description": "What to do in this step",
            "tool": "tool_name",
            "rationale": "Why this step is needed"
        }}
    ],
    "confidence": 0.8
}}

IMPORTANT: Respond with ONLY valid JSON, no additional text."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Please analyze and plan for this task:\n{input_text}"}
        ]

        self.logger.debug(f"[TOT-THOUGHT] Generating initial thought for: {input_text}")

        # Use reasoning LLM for thought generation
        client = self.reasoning_llm if isinstance(self.reasoning_llm, AsyncOpenAI) else self.llm

        async with self.telemetry.trace_llm_call(
            prompt=messages,
            model=self.reasoning_model,
            call_type="initial_thought",
            llm_name="tot_initial_thought"
        ):
            response = await client.chat.completions.create(
                messages=messages,
                **self._get_generation_kwargs(for_reasoning=True)
            )

            response_text = response.choices[0].message.content
            self.logger.debug(f"[TOT-THOUGHT] Raw response: {response_text}")

            # Parse JSON response
            try:
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    thought_data = json.loads(json_match.group())
                else:
                    thought_data = json.loads(response_text)

                thought = ThoughtNode(
                    reasoning=thought_data.get("reasoning", ""),
                    planned_tools=thought_data.get("planned_tools", []),
                    planned_steps=thought_data.get("planned_steps", []),
                    confidence=thought_data.get("confidence", 0.5)
                )

                self.logger.info(f"[TOT-THOUGHT] Generated thought with {len(thought.planned_steps)} steps")
                self.logger.debug(f"[TOT-THOUGHT] Planned tools: {thought.planned_tools}")

                # Update telemetry
                if hasattr(self.telemetry, 'update_generation'):
                    usage = None
                    if hasattr(response, 'usage') and response.usage:
                        usage = {
                            "prompt_tokens": response.usage.prompt_tokens,
                            "completion_tokens": response.usage.completion_tokens,
                            "total_tokens": response.usage.total_tokens
                        }
                    self.telemetry.update_generation(output=thought_data, usage=usage)

                # Track metrics
                self._update_token_usage(response)

                return thought

            except (json.JSONDecodeError, KeyError) as e:
                self.logger.exception(f"[TOT-THOUGHT] Error parsing thought: {e}")
                # Return a default thought
                return ThoughtNode(
                    reasoning=f"Failed to parse initial thought: {str(e)}",
                    planned_tools=[],
                    planned_steps=[],
                    confidence=0.0
                )

    async def _reflect_on_thought(
        self,
        input_text: str,
        initial_thought: ThoughtNode
    ) -> ReflectionResult:
        """Reflect on the initial thought and refine the plan.

        Args:
            input_text: The original user request
            initial_thought: The initial thought to reflect on

        Returns:
            ReflectionResult with refined plan
        """
        tools_description = self._create_tool_list_str()

        reflection_prompt = f"""You are a critical thinking assistant. Review the initial plan and improve it.

Available tools:
{tools_description}

Initial analysis:
{initial_thought.reasoning}

Initial planned tools:
{', '.join(initial_thought.planned_tools)}

Initial steps:
{json.dumps(initial_thought.planned_steps, indent=2)}

Your task:
1. Critically evaluate the initial plan
2. Identify any gaps, inefficiencies, or errors
3. Suggest improvements or alternative approaches
4. Refine the tool selection (only use tools from the available list)
5. Create an improved, more efficient plan

IMPORTANT: Only use tools that are in the available tools list. Do not invent new tools.

Respond in the following JSON format:
{{
    "refined_reasoning": "Your improved analysis with identified issues and improvements",
    "refined_steps": [
        {{
            "step_number": 1,
            "description": "What to do in this step",
            "tool": "tool_name",
            "rationale": "Why this step is needed"
        }}
    ],
    "tool_names": ["tool_name1", "tool_name2"],
    "alternative_approaches": ["Alternative approach 1", "Alternative approach 2"]
}}

IMPORTANT: Respond with ONLY valid JSON, no additional text."""

        messages = [
            {"role": "system", "content": "You are a critical thinking assistant that improves plans."},
            {"role": "user", "content": f"Original request: {input_text}\n\n{reflection_prompt}"}
        ]

        self.logger.debug(f"[TOT-REFLECT] Reflecting on initial thought")

        # Use reasoning LLM for reflection
        client = self.reasoning_llm if isinstance(self.reasoning_llm, AsyncOpenAI) else self.llm

        async with self.telemetry.trace_llm_call(
            prompt=messages,
            model=self.reasoning_model,
            call_type="reflection",
            llm_name="tot_reflection"
        ):
            response = await client.chat.completions.create(
                messages=messages,
                **self._get_generation_kwargs(for_reasoning=True)
            )

            response_text = response.choices[0].message.content
            self.logger.debug(f"[TOT-REFLECT] Raw reflection: {response_text}")

            # Parse JSON response
            try:
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    reflection_data = json.loads(json_match.group())
                else:
                    reflection_data = json.loads(response_text)

                reflection = ReflectionResult(
                    refined_reasoning=reflection_data.get("refined_reasoning", initial_thought.reasoning),
                    refined_steps=reflection_data.get("refined_steps", initial_thought.planned_steps),
                    tool_names=reflection_data.get("tool_names", initial_thought.planned_tools),
                    alternative_approaches=reflection_data.get("alternative_approaches", [])
                )

                self.logger.info(f"[TOT-REFLECT] Refined plan with {len(reflection.refined_steps)} steps")
                self.logger.debug(f"[TOT-REFLECT] Refined tools: {reflection.tool_names}")

                # Update telemetry
                if hasattr(self.telemetry, 'update_generation'):
                    usage = None
                    if hasattr(response, 'usage') and response.usage:
                        usage = {
                            "prompt_tokens": response.usage.prompt_tokens,
                            "completion_tokens": response.usage.completion_tokens,
                            "total_tokens": response.usage.total_tokens
                        }
                    self.telemetry.update_generation(output=reflection_data, usage=usage)

                # Track metrics
                self._update_token_usage(response)

                return reflection

            except (json.JSONDecodeError, KeyError) as e:
                self.logger.exception(f"[TOT-REFLECT] Error parsing reflection: {e}")
                # Fallback to initial thought
                return ReflectionResult(
                    refined_reasoning=initial_thought.reasoning,
                    refined_steps=initial_thought.planned_steps,
                    tool_names=initial_thought.planned_tools,
                    alternative_approaches=[]
                )

    async def _filter_tools(self, tool_names: List[str]) -> List[BaseTool]:
        """Filter tools based on the planned tool names.

        Args:
            tool_names: List of tool names to use

        Returns:
            List of filtered BaseTool instances
        """
        # Normalize tool names (lowercase, strip whitespace)
        normalized_names = {name.lower().strip() for name in tool_names}

        filtered_tools = []
        for tool in self.tools:
            if tool.name.lower().strip() in normalized_names:
                filtered_tools.append(tool)

        # If no tools matched, return all tools (fallback)
        if not filtered_tools:
            self.logger.warning(f"[TOT-FILTER] No tools matched, using all {len(self.tools)} tools")
            return self.tools

        return filtered_tools

    async def _execute_plan(
        self,
        input_text: str,
        plan: ReflectionResult,
        tools: List[BaseTool],
        execution_history: List[Dict[str, Any]],
        citations: List[Citation]
    ) -> str:
        """Execute the refined plan with selected tools.

        Args:
            input_text: Original user request
            plan: The refined plan to execute
            tools: Selected tools for execution
            execution_history: List to track execution history
            citations: List to collect citations

        Returns:
            Final execution result
        """
        self.logger.info(f"[TOT-EXEC] Executing plan with {len(plan.refined_steps)} steps")

        # Build tool map from selected tools
        tool_map = {tool.name: tool for tool in tools}

        # Context accumulator
        context = f"Original request: {input_text}\n\nPlan reasoning: {plan.refined_reasoning}\n\n"

        step_results = []

        for step in plan.refined_steps:
            step_num = step.get("step_number", 0)
            step_desc = step.get("description", "")
            tool_name = step.get("tool", "")
            rationale = step.get("rationale", "")

            self.logger.info(f"[TOT-EXEC] Step {step_num}: {step_desc}")
            self.logger.debug(f"[TOT-EXEC] Tool: {tool_name}, Rationale: {rationale}")

            if tool_name and tool_name in tool_map:
                # Execute tool
                tool = tool_map[tool_name]

                # Generate tool arguments
                tool_args = await self._generate_tool_arguments(
                    tool=tool,
                    task_description=step_desc,
                    context=context
                )

                if tool_args:
                    try:
                        # Execute tool
                        async with self.telemetry.trace_tool_execution(tool_name, tool_args):
                            result = await tool.arun(tool_args)

                            self.logger.info(f"[TOT-EXEC] Step {step_num} completed")
                            self.logger.debug(f"[TOT-EXEC] Result: {str(result)[:200]}")

                            # Track in execution history
                            execution_history.append({
                                "step": step_num,
                                "description": step_desc,
                                "tool": tool_name,
                                "result": str(result),
                                "status": "completed"
                            })

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

                            step_results.append(f"Step {step_num} ({tool_name}): {result}")
                            context += f"\nStep {step_num} result: {result}\n"

                            # Track metrics
                            if self.current_metrics:
                                self.current_metrics.tool_executions += 1
                                self.current_metrics.successful_tool_calls += 1

                    except Exception as e:
                        self.logger.exception(f"[TOT-EXEC] Error executing step {step_num}: {e}")
                        error_msg = f"Error: {str(e)}"
                        execution_history.append({
                            "step": step_num,
                            "description": step_desc,
                            "tool": tool_name,
                            "result": error_msg,
                            "status": "failed"
                        })
                        step_results.append(f"Step {step_num} ({tool_name}): {error_msg}")

                        if self.current_metrics:
                            self.current_metrics.tool_executions += 1
                            self.current_metrics.failed_tool_calls += 1
                else:
                    self.logger.warning(f"[TOT-EXEC] Failed to generate arguments for step {step_num}")
                    execution_history.append({
                        "step": step_num,
                        "description": step_desc,
                        "tool": tool_name,
                        "result": "Failed to generate tool arguments",
                        "status": "failed"
                    })
            else:
                # No tool specified or tool not found - generate direct response
                self.logger.debug(f"[TOT-EXEC] Step {step_num} has no tool, generating direct response")
                response = await self._generate_direct_response(step_desc, context)

                execution_history.append({
                    "step": step_num,
                    "description": step_desc,
                    "tool": None,
                    "result": response,
                    "status": "completed"
                })

                step_results.append(f"Step {step_num}: {response}")
                context += f"\nStep {step_num} result: {response}\n"

        # Generate final synthesized response
        final_response = await self._synthesize_final_response(
            input_text=input_text,
            plan_reasoning=plan.refined_reasoning,
            step_results=step_results,
            citations=citations
        )

        return final_response

    async def _generate_tool_arguments(
        self,
        tool: BaseTool,
        task_description: str,
        context: str
    ) -> Optional[Dict[str, Any]]:
        """Generate arguments for a tool call.

        Args:
            tool: The tool to generate arguments for
            task_description: Description of what the tool should do
            context: Current execution context

        Returns:
            Dictionary of tool arguments or None if generation failed
        """
        # Get tool schema
        tool_schema = {}
        if hasattr(tool, 'args_schema') and tool.args_schema:
            if hasattr(tool.args_schema, 'model_json_schema'):
                tool_schema = tool.args_schema.model_json_schema()

        prompt = f"""Generate JSON arguments for the following tool call.

Tool: {tool.name}
Description: {tool.description}
Parameters: {json.dumps(tool_schema, indent=2)}

Task: {task_description}
Context: {context}

Generate ONLY a valid JSON object with the tool arguments. No explanation, no markdown.

Example format: {{"query": "search term", "limit": 10}}

JSON arguments:"""

        messages = [
            {"role": "system", "content": "You generate tool arguments in JSON format."},
            {"role": "user", "content": prompt}
        ]

        async with self.telemetry.trace_llm_call(
            prompt=messages,
            model=self.model,
            call_type="tool_args",
            llm_name="tot_tool_args"
        ):
            response = await self.llm.chat.completions.create(
                messages=messages,
                **self._get_generation_kwargs()
            )

            response_text = response.choices[0].message.content

            try:
                # Try to extract JSON
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    args = json.loads(json_match.group())
                else:
                    args = json.loads(response_text.strip())

                self.logger.debug(f"[TOT-ARGS] Generated args: {args}")

                # Update telemetry
                if hasattr(self.telemetry, 'update_generation'):
                    usage = None
                    if hasattr(response, 'usage') and response.usage:
                        usage = {
                            "prompt_tokens": response.usage.prompt_tokens,
                            "completion_tokens": response.usage.completion_tokens,
                            "total_tokens": response.usage.total_tokens
                        }
                    self.telemetry.update_generation(output=args, usage=usage)

                # Track metrics
                self._update_token_usage(response)

                return args

            except (json.JSONDecodeError, KeyError) as e:
                self.logger.exception(f"[TOT-ARGS] Error parsing tool arguments: {e}")
                return None

    async def _generate_direct_response(
        self,
        task_description: str,
        context: str
    ) -> str:
        """Generate a direct response without using tools.

        Args:
            task_description: What to respond to
            context: Current execution context

        Returns:
            Generated response
        """
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Context: {context}\n\nTask: {task_description}\n\nResponse:"}
        ]

        async with self.telemetry.trace_llm_call(
            prompt=messages,
            model=self.model,
            call_type="direct_response",
            llm_name="tot_direct_response"
        ):
            response = await self.llm.chat.completions.create(
                messages=messages,
                **self._get_generation_kwargs()
            )

            response_text = response.choices[0].message.content

            # Track metrics
            self._update_token_usage(response)

            return response_text

    async def _synthesize_final_response(
        self,
        input_text: str,
        plan_reasoning: str,
        step_results: List[str],
        citations: List[Citation]
    ) -> str:
        """Synthesize a final coherent response from all step results.

        Args:
            input_text: Original user request
            plan_reasoning: The reasoning from the planning phase
            step_results: Results from each execution step
            citations: Citations to include

        Returns:
            Final synthesized response
        """
        results_text = "\n\n".join(step_results)

        citation_info = ""
        if citations:
            citation_info = "\n\nAvailable citations:\n"
            for idx, citation in enumerate(citations, 1):
                citation_info += f"[{idx}] {citation.document_id}, chunk {citation.chunk_number}\n"

        system_prompt = """You are an assistant that creates comprehensive, well-structured responses.
Synthesize the step-by-step results into a clear, coherent answer that directly addresses the user's question."""

        if citations:
            system_prompt += "\n\nINCLUDE inline citations [1], [2] in your response and add a References section at the end."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"""Original question: {input_text}

Planning reasoning: {plan_reasoning}

Execution results:
{results_text}{citation_info}

Provide a comprehensive answer:"""}
        ]

        async with self.telemetry.trace_llm_call(
            prompt=messages,
            model=self.model,
            call_type="final_synthesis",
            llm_name="tot_final_synthesis"
        ):
            response = await self.llm.chat.completions.create(
                messages=messages,
                **self._get_generation_kwargs()
            )

            final_text = response.choices[0].message.content

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
                    output={"response": final_text},
                    usage=usage
                )

            # Track metrics
            self._update_token_usage(response)

            return final_text

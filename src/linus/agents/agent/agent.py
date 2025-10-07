"""Custom Agent implementation for Gemma3:27b without native tool support."""

from typing import List, Dict, Any, Optional, Tuple, Type, Union
from dataclasses import dataclass
import json
import re
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_community.chat_models import ChatOpenAI
from pydantic import BaseModel, Field
from loguru import logger


@dataclass
class ReasoningResult:
    """Result from the reasoning phase."""
    has_sufficient_info: bool
    tasks: List[Dict[str, Any]]
    reasoning: str


@dataclass
class TaskExecution:
    """Represents a task to be executed."""
    description: str
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    completed: bool = False
    result: Optional[Any] = None


class Agent:
    """Base Agent class with langchain integration."""

    def __init__(
        self,
        llm: ChatOpenAI,
        tools: List[BaseTool],
        verbose: bool = False,
        input_schema: Optional[Type[BaseModel]] = None,
        output_schema: Optional[Type[BaseModel]] = None,
        output_key: Optional[str] = None,
        state: Optional[Dict[str, Any]] = None
    ):
        """Initialize the base agent.

        Args:
            llm: LangChain ChatOpenAI instance (can be configured for Gemma)
            tools: List of available tools
            verbose: Whether to print debug information
            input_schema: Optional Pydantic BaseModel for structured input validation
            output_schema: Optional Pydantic BaseModel for structured output
            output_key: Optional key to save output in shared state
            state: Optional shared state dictionary between agents
        """
        self.llm = llm
        self.tools = tools
        self.tool_map = {tool.name: tool for tool in tools}
        self.verbose = verbose
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.output_key = output_key
        self.state = state if state is not None else {}

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
                    self.state[self.output_key] = output_obj
                    self._log(f"Saved output to state['{self.output_key}']")

                return output_obj
            except (json.JSONDecodeError, Exception) as e:
                self._log(f"Error parsing output with schema: {e}")
                # Fall back to string result
                if self.output_key:
                    self.state[self.output_key] = result
                return result

        # No output schema, save raw result if output_key is provided
        if self.output_key:
            self.state[self.output_key] = result
            self._log(f"Saved output to state['{self.output_key}']")

        return result

    def _log(self, message: str):
        """Log a message if verbose mode is enabled."""
        if self.verbose:
            logger.info(message)


class ReasoningAgent(Agent):
    """Agent that uses two-call approach for Gemma3:27b without tool support.

    First call: Reasoning phase to analyze the task and plan actions
    Second call: Execution phase to generate tool arguments and execute
    """

    def __init__(
        self,
        llm: ChatOpenAI,
        tools: List[BaseTool],
        verbose: bool = False,
        input_schema: Optional[Type[BaseModel]] = None,
        output_schema: Optional[Type[BaseModel]] = None,
        output_key: Optional[str] = None,
        state: Optional[Dict[str, Any]] = None
    ):
        """Initialize the reasoning agent."""
        super().__init__(llm, tools, verbose, input_schema, output_schema, output_key, state)
        self.reasoning_prompt = self._create_reasoning_prompt()
        self.execution_prompt = self._create_execution_prompt()
    
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

    def run(self, input_data: Union[str, BaseModel, Dict[str, Any]]) -> Union[str, BaseModel]:
        """Run the agent using the two-call approach.

        Args:
            input_data: The user's request (string, Pydantic model, or dict)

        Returns:
            The final response or result (string or Pydantic model)
        """
        # Validate and convert input
        input_text = self._validate_and_convert_input(input_data)
        logger.debug(f"[RUN] Processing request: {input_text}")

        # Phase 1: Reasoning
        reasoning_result = self._reasoning_call(input_text)
        logger.debug(f"[RUN] Reasoning result: {reasoning_result}")

        if not reasoning_result.has_sufficient_info:
            result = f"I need more information to complete this task. {reasoning_result.reasoning}"
            return self._format_output(result)

        # Phase 2: Execution
        results = []
        context = input_text

        # Add state context if available
        if self.state:
            state_context = f"\n\nShared state: {json.dumps({k: str(v) for k, v in self.state.items()})}"
            context = context + state_context
            logger.debug(f"[RUN] Added state context: {state_context}")

        for task_data in reasoning_result.tasks:
            task = TaskExecution(
                description=task_data["description"],
                tool_name=task_data.get("tool_name")
            )

            if task.tool_name:
                # Generate tool arguments and execute
                task_result = self._execute_task_with_tool(task, context)
                results.append(task_result)
                # Update context with results for subsequent tasks
                context = f"{context}\n\nPrevious result: {task_result}"
            else:
                # Direct LLM response without tool
                response = self._generate_response(task.description, context)
                results.append(response)
                context = f"{context}\n\nPrevious result: {response}"

        # Combine results into final response
        final_result = self._format_final_response(input_text, results)

        # Format output according to schema and save to state
        return self._format_output(final_result)
    
    def _reasoning_call(self, input_text: str) -> ReasoningResult:
        """Perform the reasoning phase.

        Args:
            input_text: The user's request

        Returns:
            ReasoningResult containing the analysis and planned tasks
        """
        prompt = self.reasoning_prompt + input_text

        messages = [
            SystemMessage(content=self.reasoning_prompt),
            HumanMessage(content=input_text)
        ]

        logger.debug(f"[REASONING] Input text: {input_text}")
        logger.debug(f"[REASONING] System prompt: {self.reasoning_prompt}")
        logger.debug(f"[REASONING] Messages: {messages}")

        response = self.llm.invoke(messages)
        logger.debug(f"[REASONING] Raw response: {response.content}")
        
        try:
            # Parse JSON response
            response_text = response.content
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
            logger.debug(f"[REASONING] Failed to parse: {response.content}")
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
            result = tool.run(tool_args)
            logger.debug(f"[EXECUTION] Tool result: {result}")
            task.completed = True
            task.result = result
            return str(result)
        except Exception as e:
            logger.error(f"[EXECUTION] Error executing tool {task.tool_name}: {e}")
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
            SystemMessage(content="You are a helpful assistant that generates tool arguments."),
            HumanMessage(content=prompt)
        ]

        response = self.llm.invoke(messages)
        logger.debug(f"[TOOL_ARGS] Raw response: {response.content}")
        
        try:
            # Parse JSON arguments
            response_text = response.content
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
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content=f"Context: {context}\n\nTask: {task_description}")
        ]

        response = self.llm.invoke(messages)
        logger.debug(f"[GENERATE] Response: {response.content}")
        return response.content
    
    def _format_final_response(self, original_request: str, results: List[str]) -> str:
        """Format the final response from all task results.

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
            SystemMessage(content="Summarize the following task results into a coherent response."),
            HumanMessage(content=f"Original request: {original_request}\n\nResults:\n{combined}")
        ]

        response = self.llm.invoke(messages)
        logger.debug(f"[FINAL] Final response: {response.content}")
        return response.content


# Example usage function
def create_gemma_agent(
    api_base: str = "http://localhost:11434/v1",  # Ollama OpenAI-compatible endpoint
    model: str = "gemma3:27b",
    tools: Optional[List[BaseTool]] = None,
    verbose: bool = True,
    input_schema: Optional[Type[BaseModel]] = None,
    output_schema: Optional[Type[BaseModel]] = None,
    output_key: Optional[str] = None,
    state: Optional[Dict[str, Any]] = None
) -> ReasoningAgent:
    """Create a ReasoningAgent configured for Gemma3:27b.

    Args:
        api_base: The OpenAI-compatible API endpoint
        model: The model name
        tools: List of tools available to the agent
        verbose: Whether to enable verbose logging
        input_schema: Optional Pydantic BaseModel for structured input validation
        output_schema: Optional Pydantic BaseModel for structured output
        output_key: Optional key to save output in shared state
        state: Optional shared state dictionary between agents

    Returns:
        Configured ReasoningAgent instance
    """
    # Configure LLM for Gemma through OpenAI-compatible API
    llm = ChatOpenAI(
        base_url=api_base,
        model=model,
        temperature=0.7,
        api_key="not-needed"  # Ollama doesn't require an API key
    )

    if tools is None:
        tools = []

    return ReasoningAgent(
        llm=llm,
        tools=tools,
        verbose=verbose,
        input_schema=input_schema,
        output_schema=output_schema,
        output_key=output_key,
        state=state
    )
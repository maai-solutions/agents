"""Factory functions for creating agents."""

from typing import List, Optional, Type, Any, Union
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel
from loguru import logger

from .reasoning_agent import ReasoningAgent
from .coordinator_agent import CoordinatorAgent, SubAgent
from .tot import TreeOfThoughtAgent
from .light_agent import LightAgent
from .swarm import Swarm
from .tool_base import BaseTool
from .config import AgentParams, LLMConfig, MemoryConfig, StateConfig
from ..graph.state import SharedState, ConversationMemoryBackend


# Example usage function
def Agent(
    params: Optional[AgentParams] = None,
    tools: Optional[List[BaseTool]] = None,
    verbose: bool = True,
    input_schema: Optional[Type[BaseModel]] = None,
    output_schema: Optional[Type[BaseModel]] = None,
    output_key: Optional[str] = None,
    state: Optional[SharedState] = None,
    max_iterations: int = 10,
    memory_context_ratio: float = 0.3,
    use_async: bool = True,
    use_json_format: bool = False,
    tracer: Optional[Any] = None,
    session_id: Optional[str] = None,
    agent_name: Optional[str] = None
) -> ReasoningAgent:
    """Create a ReasoningAgent configured for Gemma3:27b or other OpenAI-compatible models.

    Args:
        params: AgentParams configuration object (if not provided, uses defaults)
        tools: List of tools available to the agent
        verbose: Whether to enable verbose logging
        input_schema: Optional Pydantic BaseModel for structured input validation
        output_schema: Optional Pydantic BaseModel for structured output
        output_key: Optional key to save output in shared state
        state: Optional SharedState instance for state management
        max_iterations: Maximum number of reasoning-execution loops (default: 10)
        memory_context_ratio: Ratio of context to use for memory (0.0 to 1.0)
        use_async: Whether to use AsyncOpenAI client (default: True)
        use_json_format: Whether to use response_format={"type": "json_object"} (default: False)
        tracer: Optional telemetry tracer (AgentTracer or LangfuseTracer)
        session_id: Optional session ID for Langfuse session grouping
        agent_name: Optional name for the agent (used in hierarchical tracing)

    Returns:
        Configured ReasoningAgent instance

    Examples:
        # Using AgentParams (recommended):
        from linus.agents.agent.config import AgentParams, LLMConfig, MemoryConfig

        params = AgentParams(
            llm_config=LLMConfig(
                api_base="http://localhost:11434/v1",
                model="gemma3:27b",
                api_key="not-needed"
            ),
            temperature=0.7,
            max_tokens=2048,
            top_k=40,
            memory_config=MemoryConfig(
                enable_memory=True,
                memory_backend="in_memory",
                max_memory_size=100
            )
        )
        agent = Agent(params=params, tools=get_default_tools())

        # Using defaults (Ollama):
        agent = Agent(tools=get_default_tools())
    """
    # Use default params if not provided
    if params is None:
        params = AgentParams()

    # Configure OpenAI client
    if use_async:
        llm = AsyncOpenAI(
            base_url=params.llm_config.api_base,
            api_key=params.llm_config.api_key
        )
    else:
        llm = OpenAI(
            base_url=params.llm_config.api_base,
            api_key=params.llm_config.api_key
        )

    if tools is None:
        tools = []

    # Create memory if enabled (using SharedState with ConversationMemoryBackend)
    memory = None
    if params.memory_config.enable_memory:
        memory = SharedState(
            backend=ConversationMemoryBackend(max_size=params.memory_config.max_memory_size),
            max_context_tokens=params.memory_config.max_context_tokens,
            llm_client=llm,
            model=params.llm_config.model
        )
        logger.info(f"[MEMORY] Initialized conversation memory backend with max_size={params.memory_config.max_memory_size}")

    agent = ReasoningAgent(
        llm=llm,
        model=params.llm_config.model,
        tools=tools,
        verbose=verbose,
        input_schema=input_schema,
        output_schema=output_schema,
        output_key=output_key,
        state=state,
        max_iterations=max_iterations,
        memory=memory,
        memory_context_ratio=memory_context_ratio,
        temperature=params.temperature,
        max_tokens=params.max_tokens,
        top_p=params.top_p,
        top_k=params.top_k,
        api_base=params.llm_config.api_base,
        use_json_format=use_json_format,
        agent_name=agent_name
    )

    # Override tracer if provided
    if tracer is not None:
        agent.telemetry = tracer
        # Update tracer with agent_name if supported
        if agent_name is not None and hasattr(tracer, 'agent_name'):
            tracer.agent_name = agent_name

    # If session_id is provided, update the agent's tracer if it's a LangfuseTracer
    if session_id is not None and hasattr(agent.telemetry, 'session_id'):
        agent.telemetry.session_id = session_id

    return agent


def Coordinator(
    params: Optional[AgentParams] = None,
    subagents: Optional[List[SubAgent]] = None,
    tools: Optional[List[BaseTool]] = None,
    verbose: bool = True,
    input_schema: Optional[Type[BaseModel]] = None,
    output_schema: Optional[Type[BaseModel]] = None,
    output_key: Optional[str] = None,
    state: Optional[SharedState] = None,
    max_iterations: int = 15,
    memory_context_ratio: float = 0.3,
    use_async: bool = True,
    use_json_format: bool = False,
    tracer: Optional[Any] = None,
    session_id: Optional[str] = None,
    agent_name: Optional[str] = None
) -> CoordinatorAgent:
    """Create a CoordinatorAgent that orchestrates multiple subagents.

    Args:
        params: AgentParams configuration object (if not provided, uses defaults)
        subagents: List of SubAgent instances to coordinate
        tools: Optional list of tools for the coordinator (not subagents)
        verbose: Whether to enable verbose logging
        input_schema: Optional Pydantic BaseModel for input validation
        output_schema: Optional Pydantic BaseModel for output
        output_key: Optional key to save output in shared state
        state: Optional SharedState instance
        max_iterations: Maximum number of plan-execute-evaluate loops (default: 15)
        memory_context_ratio: Ratio of context to use for memory
        use_async: Whether to use AsyncOpenAI client (default: True)
        use_json_format: Whether to use JSON response format
        tracer: Optional telemetry tracer
        session_id: Optional session ID for Langfuse
        agent_name: Optional name for the agent

    Returns:
        Configured CoordinatorAgent instance

    Examples:
        # Using AgentParams:
        from linus.agents.agent.config import AgentParams, LLMConfig

        params = AgentParams(
            llm_config=LLMConfig(model="gemma3:27b"),
            temperature=0.7
        )

        # Create specialized subagents
        research_agent = Agent(tools=[SearchTool()])
        calc_agent = Agent(tools=[CalculatorTool()])

        # Wrap them as SubAgents
        subagents = [
            SubAgent(
                agent=research_agent,
                name="researcher",
                description="Searches for information",
                capabilities=["search", "web_research"]
            ),
            SubAgent(
                agent=calc_agent,
                name="calculator",
                description="Performs calculations",
                capabilities=["math", "calculator"]
            )
        ]

        # Create coordinator
        coordinator = Coordinator(
            params=params,
            subagents=subagents,
            verbose=True
        )
    """
    # Use default params if not provided
    if params is None:
        params = AgentParams()

    # Configure OpenAI client
    if use_async:
        llm = AsyncOpenAI(
            base_url=params.llm_config.api_base,
            api_key=params.llm_config.api_key
        )
    else:
        llm = OpenAI(
            base_url=params.llm_config.api_base,
            api_key=params.llm_config.api_key
        )

    if subagents is None:
        subagents = []
        logger.warning("[COORDINATOR] No subagents provided to coordinator")

    if tools is None:
        tools = []

    # Create memory if enabled (using SharedState with ConversationMemoryBackend)
    memory = None
    if params.memory_config.enable_memory:
        memory = SharedState(
            backend=ConversationMemoryBackend(max_size=params.memory_config.max_memory_size),
            max_context_tokens=params.memory_config.max_context_tokens,
            llm_client=llm,
            model=params.llm_config.model
        )
        logger.info(f"[MEMORY] Initialized conversation memory backend with max_size={params.memory_config.max_memory_size}")

    coordinator = CoordinatorAgent(
        llm=llm,
        model=params.llm_config.model,
        subagents=subagents,
        tools=tools,
        verbose=verbose,
        input_schema=input_schema,
        output_schema=output_schema,
        output_key=output_key,
        state=state,
        max_iterations=max_iterations,
        memory=memory,
        memory_context_ratio=memory_context_ratio,
        temperature=params.temperature,
        max_tokens=params.max_tokens,
        top_p=params.top_p,
        top_k=params.top_k,
        api_base=params.llm_config.api_base,
        use_json_format=use_json_format,
        agent_name=agent_name
    )

    # Override tracer if provided
    if tracer is not None:
        coordinator.telemetry = tracer
        if agent_name is not None and hasattr(tracer, 'agent_name'):
            tracer.agent_name = agent_name

    # If session_id is provided, update the tracer
    if session_id is not None and hasattr(coordinator.telemetry, 'session_id'):
        coordinator.telemetry.session_id = session_id

    return coordinator


def TreeOfThought(
    params: Optional[AgentParams] = None,
    tools: Optional[List[BaseTool]] = None,
    verbose: bool = True,
    input_schema: Optional[Type[BaseModel]] = None,
    output_schema: Optional[Type[BaseModel]] = None,
    output_key: Optional[str] = None,
    state: Optional[SharedState] = None,
    max_iterations: int = 10,
    # ToT-specific parameters
    reasoning_model: Optional[str] = None,
    reasoning_api_base: Optional[str] = None,
    reasoning_api_key: Optional[str] = None,
    enable_tool_filtering: bool = True,
    enable_reflection: bool = True,
    max_reflection_depth: int = 2,
    reasoning_temperature: float = 0.8,
    use_async: bool = True,
    tracer: Optional[Any] = None,
    session_id: Optional[str] = None,
    agent_name: Optional[str] = None
) -> TreeOfThoughtAgent:
    """Create a TreeOfThoughtAgent for complex reasoning tasks.

    The Tree of Thought agent uses a multi-phase approach:
    1. Initial thought generation with task analysis
    2. Reflection on the initial plan to refine it
    3. Tool selection and filtering based on task requirements
    4. Execution with the refined plan

    Args:
        params: AgentParams configuration object (if not provided, uses defaults)
        tools: List of tools available to the agent
        verbose: Whether to enable verbose logging
        input_schema: Optional Pydantic BaseModel for input validation
        output_schema: Optional Pydantic BaseModel for output
        output_key: Optional key to save output in shared state
        state: Optional SharedState instance for state management
        max_iterations: Maximum number of execution iterations
        reasoning_model: Model name for reasoning phase (defaults to main model)
        reasoning_api_base: API base for reasoning model (defaults to main api_base)
        reasoning_api_key: API key for reasoning model (defaults to main api_key)
        enable_tool_filtering: Whether to filter tools based on task analysis
        enable_reflection: Whether to enable reflection phase
        max_reflection_depth: Maximum number of reflection iterations
        reasoning_temperature: Temperature for reasoning phase (higher = more creative)
        use_async: Whether to use AsyncOpenAI client (default: True)
        tracer: Optional telemetry tracer
        session_id: Optional session ID for Langfuse
        agent_name: Optional name for the agent

    Returns:
        Configured TreeOfThoughtAgent instance

    Examples:
        # Using AgentParams
        from linus.agents.agent.config import AgentParams, LLMConfig

        params = AgentParams(
            llm_config=LLMConfig(model="gemma3:27b"),
            temperature=0.5
        )

        # Basic ToT agent with Ollama
        tot_agent = TreeOfThought(
            params=params,
            reasoning_model="deepseek-r1",  # Use DeepSeek-R1 for reasoning
            tools=[SearchTool(), CalculatorTool()],
            enable_reflection=True,
            enable_tool_filtering=True
        )

        # ToT agent with separate reasoning model
        tot_agent = TreeOfThought(
            reasoning_model="qwen2.5:32b",
            reasoning_temperature=0.9,  # Higher creativity for planning
            enable_reflection=True
        )

        # Run the agent
        result = await tot_agent.run("Analyze the market trends and calculate ROI")
    """
    # Use default params if not provided
    if params is None:
        params = AgentParams()

    # Configure execution LLM
    if use_async:
        llm = AsyncOpenAI(
            base_url=params.llm_config.api_base,
            api_key=params.llm_config.api_key
        )
    else:
        llm = OpenAI(
            base_url=params.llm_config.api_base,
            api_key=params.llm_config.api_key
        )

    # Configure reasoning LLM (may be different from execution LLM)
    reasoning_llm = None
    if reasoning_model or reasoning_api_base or reasoning_api_key:
        reasoning_base = reasoning_api_base or params.llm_config.api_base
        reasoning_key = reasoning_api_key or params.llm_config.api_key

        if use_async:
            reasoning_llm = AsyncOpenAI(
                base_url=reasoning_base,
                api_key=reasoning_key
            )
        else:
            reasoning_llm = OpenAI(
                base_url=reasoning_base,
                api_key=reasoning_key
            )

    if tools is None:
        tools = []

    # Create memory if enabled (using SharedState with ConversationMemoryBackend)
    memory = None
    if params.memory_config.enable_memory:
        memory = SharedState(
            backend=ConversationMemoryBackend(max_size=params.memory_config.max_memory_size),
            max_context_tokens=params.memory_config.max_context_tokens,
            llm_client=llm,
            model=params.llm_config.model
        )
        logger.info(f"[MEMORY] Initialized conversation memory backend for ToT agent with max_size={params.memory_config.max_memory_size}")

    agent = TreeOfThoughtAgent(
        llm=llm,
        model=params.llm_config.model,
        tools=tools,
        verbose=verbose,
        input_schema=input_schema,
        output_schema=output_schema,
        output_key=output_key,
        state=state,
        max_iterations=max_iterations,
        memory=memory,
        reasoning_model=reasoning_model or params.llm_config.model,
        reasoning_llm=reasoning_llm,
        enable_tool_filtering=enable_tool_filtering,
        enable_reflection=enable_reflection,
        max_reflection_depth=max_reflection_depth,
        temperature=params.temperature,
        max_tokens=params.max_tokens,
        top_p=params.top_p,
        top_k=params.top_k,
        reasoning_temperature=reasoning_temperature,
        agent_name=agent_name
    )

    # Override tracer if provided
    if tracer is not None:
        agent.telemetry = tracer
        if agent_name is not None and hasattr(tracer, 'agent_name'):
            tracer.agent_name = agent_name

    # If session_id is provided, update the tracer
    if session_id is not None and hasattr(agent.telemetry, 'session_id'):
        agent.telemetry.session_id = session_id

    return agent


def Light(
    params: Optional[AgentParams] = None,
    tools: Optional[List[BaseTool]] = None,
    verbose: bool = True,
    input_schema: Optional[Type[BaseModel]] = None,
    output_schema: Optional[Type[BaseModel]] = None,
    output_key: Optional[str] = None,
    state: Optional[SharedState] = None,
    # LightAgent-specific parameters
    instructions: str = "You are a helpful AI assistant.",
    role: Optional[str] = None,
    max_tool_iterations: int = 10,
    stream: bool = False,
    use_async: bool = True,
    tracer: Optional[Any] = None,
    session_id: Optional[str] = None,
    agent_name: Optional[str] = None
) -> LightAgent:
    """Create a LightAgent with native OpenAI function calling support.

    LightAgent is a simpler, more direct agent that uses OpenAI's native
    function calling instead of custom reasoning phases. It's ideal for:
    - Direct interaction with models that have good native function calling
    - Simpler tasks that don't require multi-phase reasoning
    - Streaming responses
    - Multi-agent swarms

    Args:
        params: AgentParams configuration object (if not provided, uses defaults)
        tools: List of tools available to the agent
        verbose: Whether to enable verbose logging
        input_schema: Optional Pydantic BaseModel for input validation
        output_schema: Optional Pydantic BaseModel for output
        output_key: Optional key to save output in shared state
        state: Optional SharedState instance
        instructions: System instructions for the agent
        role: Optional role description
        max_tool_iterations: Maximum tool calling iterations (default: 10)
        stream: Enable streaming responses (default: False)
        use_async: Whether to use AsyncOpenAI client (default: True)
        tracer: Optional telemetry tracer
        session_id: Optional session ID for Langfuse
        agent_name: Optional name for the agent

    Returns:
        Configured LightAgent instance

    Examples:
        # Using AgentParams
        from linus.agents.agent.config import AgentParams, LLMConfig

        params = AgentParams(
            llm_config=LLMConfig(model="gpt-4"),
            temperature=0.7
        )

        # Basic LightAgent
        agent = Light(
            params=params,
            instructions="You are a helpful research assistant.",
            tools=[SearchTool(), CalculatorTool()]
        )

        # With streaming (using defaults)
        agent = Light(
            stream=True,
            instructions="You are a coding assistant."
        )

        # In a swarm
        researcher = Light(
            instructions="You are a research specialist.",
            tools=[SearchTool()],
            agent_name="researcher"
        )

        calculator = Light(
            instructions="You are a math specialist.",
            tools=[CalculatorTool()],
            agent_name="calculator"
        )

        from linus.agents.agent.swarm import Swarm
        swarm = Swarm()
        swarm.register(researcher, calculator)

        result = await swarm.run("Calculate 42 * 17")
    """
    # Use default params if not provided
    if params is None:
        params = AgentParams()

    # Configure OpenAI client
    if use_async:
        llm = AsyncOpenAI(
            base_url=params.llm_config.api_base,
            api_key=params.llm_config.api_key
        )
    else:
        llm = OpenAI(
            base_url=params.llm_config.api_base,
            api_key=params.llm_config.api_key
        )

    if tools is None:
        tools = []

    # Create memory if enabled (using SharedState with ConversationMemoryBackend)
    memory = None
    if params.memory_config.enable_memory:
        memory = SharedState(
            backend=ConversationMemoryBackend(max_size=params.memory_config.max_memory_size),
            max_context_tokens=params.memory_config.max_context_tokens,
            llm_client=llm,
            model=params.llm_config.model
        )
        logger.info(f"[MEMORY] Initialized conversation memory backend for LightAgent with max_size={params.memory_config.max_memory_size}")

    agent = LightAgent(
        llm=llm,
        model=params.llm_config.model,
        tools=tools,
        verbose=verbose,
        input_schema=input_schema,
        output_schema=output_schema,
        output_key=output_key,
        state=state,
        memory=memory,
        instructions=instructions,
        role=role,
        max_tool_iterations=max_tool_iterations,
        temperature=params.temperature,
        max_tokens=params.max_tokens,
        top_p=params.top_p,
        top_k=params.top_k,
        stream=stream,
        agent_name=agent_name
    )

    # Override tracer if provided
    if tracer is not None:
        agent.telemetry = tracer
        if agent_name is not None and hasattr(tracer, 'agent_name'):
            tracer.agent_name = agent_name

    # If session_id is provided, update the tracer
    if session_id is not None and hasattr(agent.telemetry, 'session_id'):
        agent.telemetry.session_id = session_id

    return agent
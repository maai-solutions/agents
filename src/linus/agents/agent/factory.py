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
from ..graph.state import SharedState

# Import memory components
try:
    from .memory import MemoryManager, create_memory_manager
    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False
    logger.warning("Memory module not available")


# Example usage function
def Agent(
    api_base: str = "http://localhost:11434/v1",  # Ollama OpenAI-compatible endpoint
    model: str = "gemma3:27b",
    api_key: str = "not-needed",
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    tools: Optional[List[BaseTool]] = None,
    verbose: bool = True,
    input_schema: Optional[Type[BaseModel]] = None,
    output_schema: Optional[Type[BaseModel]] = None,
    output_key: Optional[str] = None,
    state: Optional[SharedState] = None,
    max_iterations: int = 10,
    enable_memory: bool = False,
    memory_backend: str = "in_memory",
    max_context_tokens: int = 4096,
    memory_context_ratio: float = 0.3,
    max_memory_size: Optional[int] = 100,
    use_async: bool = False,
    use_json_format: bool = False,
    tracer: Optional[Any] = None,
    session_id: Optional[str] = None,
    agent_name: Optional[str] = None
) -> ReasoningAgent:
    """Create a ReasoningAgent configured for Gemma3:27b or other OpenAI-compatible models.

    Args:
        api_base: The OpenAI-compatible API endpoint (e.g., "http://localhost:11434/v1" for Ollama)
        model: The model name (e.g., "gemma3:27b" for Ollama, "gpt-4" for OpenAI)
        api_key: API key for authentication (default: "not-needed" for Ollama, required for OpenAI)
        temperature: Sampling temperature (0.0 to 2.0). Higher = more random (default: 0.7)
        max_tokens: Maximum tokens to generate in completion (default: None = model default)
        top_p: Nucleus sampling parameter (0.0 to 1.0). Alternative to temperature (default: None)
        top_k: Top-k sampling parameter. Only available on some models like Ollama (default: None)
        tools: List of tools available to the agent
        verbose: Whether to enable verbose logging
        input_schema: Optional Pydantic BaseModel for structured input validation
        output_schema: Optional Pydantic BaseModel for structured output
        output_key: Optional key to save output in shared state
        state: Optional SharedState instance for state management
        max_iterations: Maximum number of reasoning-execution loops (default: 10)
        enable_memory: Whether to enable memory management
        memory_backend: Type of memory backend ("in_memory" or "vector_store")
        max_context_tokens: Maximum tokens for context window (for memory management, not generation)
        memory_context_ratio: Ratio of context to use for memory (0.0 to 1.0)
        max_memory_size: Maximum number of memories to keep (None for unlimited)
        use_async: Whether to use AsyncOpenAI client (default: False for OpenAI client)
        use_json_format: Whether to use response_format={"type": "json_object"} (default: False, not all models support this)
        tracer: Optional telemetry tracer (AgentTracer or LangfuseTracer)
        session_id: Optional session ID for Langfuse session grouping
        agent_name: Optional name for the agent (used in hierarchical tracing like agent.<name>)

    Returns:
        Configured ReasoningAgent instance

    Examples:
        # For Ollama (local):
        agent = Agent(
            api_base="http://localhost:11434/v1",
            model="gemma3:27b",
            api_key="not-needed",
            temperature=0.7,
            max_tokens=2048,
            top_k=40
        )

        # For OpenAI:
        agent = Agent(
            api_base="https://api.openai.com/v1",
            model="gpt-4",
            api_key="sk-...",
            temperature=0.5,
            max_tokens=1000,
            top_p=0.9
        )
    """
    # Configure OpenAI client for Gemma through OpenAI-compatible API
    if use_async:
        llm = AsyncOpenAI(
            base_url=api_base,
            api_key=api_key
        )
    else:
        llm = OpenAI(
            base_url=api_base,
            api_key=api_key
        )

    if tools is None:
        tools = []

    # Create memory manager if enabled
    memory_manager = None
    if enable_memory and MEMORY_AVAILABLE:
        memory_manager = create_memory_manager(
            backend_type=memory_backend,
            max_context_tokens=max_context_tokens,
            summary_threshold_tokens=int(max_context_tokens * 0.5),
            llm=llm,
            model=model,
            max_size=max_memory_size
        )
        logger.info(f"[MEMORY] Initialized {memory_backend} memory backend with OpenAI client")
    elif enable_memory and not MEMORY_AVAILABLE:
        logger.warning("[MEMORY] Memory requested but module not available")

    agent = ReasoningAgent(
        llm=llm,
        model=model,
        tools=tools,
        verbose=verbose,
        input_schema=input_schema,
        output_schema=output_schema,
        output_key=output_key,
        state=state,
        max_iterations=max_iterations,
        memory_manager=memory_manager,
        memory_context_ratio=memory_context_ratio,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        top_k=top_k,
        api_base=api_base,
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
    api_base: str = "http://localhost:11434/v1",
    model: str = "gemma3:27b",
    api_key: str = "not-needed",
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    subagents: Optional[List[SubAgent]] = None,
    tools: Optional[List[BaseTool]] = None,
    verbose: bool = True,
    input_schema: Optional[Type[BaseModel]] = None,
    output_schema: Optional[Type[BaseModel]] = None,
    output_key: Optional[str] = None,
    state: Optional[SharedState] = None,
    max_iterations: int = 15,
    enable_memory: bool = False,
    memory_backend: str = "in_memory",
    max_context_tokens: int = 4096,
    memory_context_ratio: float = 0.3,
    max_memory_size: Optional[int] = 100,
    use_async: bool = True,
    use_json_format: bool = False,
    tracer: Optional[Any] = None,
    session_id: Optional[str] = None,
    agent_name: Optional[str] = None
) -> CoordinatorAgent:
    """Create a CoordinatorAgent that orchestrates multiple subagents.

    Args:
        api_base: The OpenAI-compatible API endpoint
        model: The model name (e.g., "gemma3:27b", "gpt-4")
        api_key: API key for authentication
        temperature: Sampling temperature (0.0 to 2.0)
        max_tokens: Maximum tokens to generate
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        subagents: List of SubAgent instances to coordinate
        tools: Optional list of tools for the coordinator (not subagents)
        verbose: Whether to enable verbose logging
        input_schema: Optional Pydantic BaseModel for input validation
        output_schema: Optional Pydantic BaseModel for output
        output_key: Optional key to save output in shared state
        state: Optional SharedState instance
        max_iterations: Maximum number of plan-execute-evaluate loops (default: 15)
        enable_memory: Whether to enable memory management
        memory_backend: Type of memory backend
        max_context_tokens: Maximum tokens for context window
        memory_context_ratio: Ratio of context to use for memory
        max_memory_size: Maximum number of memories to keep
        use_async: Whether to use AsyncOpenAI client (default: True)
        use_json_format: Whether to use JSON response format
        tracer: Optional telemetry tracer
        session_id: Optional session ID for Langfuse
        agent_name: Optional name for the agent

    Returns:
        Configured CoordinatorAgent instance

    Examples:
        # Create specialized subagents
        research_agent = Agent(model="gemma3:27b", tools=[SearchTool()])
        calc_agent = Agent(model="gemma3:27b", tools=[CalculatorTool()])

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
            model="gemma3:27b",
            subagents=subagents,
            verbose=True
        )
    """
    # Configure OpenAI client
    if use_async:
        llm = AsyncOpenAI(
            base_url=api_base,
            api_key=api_key
        )
    else:
        llm = OpenAI(
            base_url=api_base,
            api_key=api_key
        )

    if subagents is None:
        subagents = []
        logger.warning("[COORDINATOR] No subagents provided to coordinator")

    if tools is None:
        tools = []

    # Create memory manager if enabled
    memory_manager = None
    if enable_memory and MEMORY_AVAILABLE:
        memory_manager = create_memory_manager(
            backend_type=memory_backend,
            max_context_tokens=max_context_tokens,
            summary_threshold_tokens=int(max_context_tokens * 0.5),
            llm=llm,
            model=model,
            max_size=max_memory_size
        )
        logger.info(f"[MEMORY] Initialized {memory_backend} memory backend")
    elif enable_memory and not MEMORY_AVAILABLE:
        logger.warning("[MEMORY] Memory requested but module not available")

    coordinator = CoordinatorAgent(
        llm=llm,
        model=model,
        subagents=subagents,
        tools=tools,
        verbose=verbose,
        input_schema=input_schema,
        output_schema=output_schema,
        output_key=output_key,
        state=state,
        max_iterations=max_iterations,
        memory_manager=memory_manager,
        memory_context_ratio=memory_context_ratio,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        top_k=top_k,
        api_base=api_base,
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
    api_base: str = "http://localhost:11434/v1",
    model: str = "gemma3:27b",
    api_key: str = "not-needed",
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    tools: Optional[List[BaseTool]] = None,
    verbose: bool = True,
    input_schema: Optional[Type[BaseModel]] = None,
    output_schema: Optional[Type[BaseModel]] = None,
    output_key: Optional[str] = None,
    state: Optional[SharedState] = None,
    max_iterations: int = 10,
    enable_memory: bool = False,
    memory_backend: str = "in_memory",
    max_context_tokens: int = 4096,
    memory_context_ratio: float = 0.3,
    max_memory_size: Optional[int] = 100,
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
        api_base: The OpenAI-compatible API endpoint for execution
        model: The model name for execution (e.g., "gemma3:27b", "gpt-4")
        api_key: API key for authentication
        temperature: Sampling temperature for execution (0.0 to 2.0)
        max_tokens: Maximum tokens to generate in completion
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter (Ollama-specific)
        tools: List of tools available to the agent
        verbose: Whether to enable verbose logging
        input_schema: Optional Pydantic BaseModel for input validation
        output_schema: Optional Pydantic BaseModel for output
        output_key: Optional key to save output in shared state
        state: Optional SharedState instance for state management
        max_iterations: Maximum number of execution iterations
        enable_memory: Whether to enable memory management
        memory_backend: Type of memory backend
        max_context_tokens: Maximum tokens for context window
        memory_context_ratio: Ratio of context to use for memory
        max_memory_size: Maximum number of memories to keep
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
        # Basic ToT agent with Ollama
        tot_agent = TreeOfThought(
            api_base="http://localhost:11434/v1",
            model="gemma3:27b",
            reasoning_model="deepseek-r1",  # Use DeepSeek-R1 for reasoning
            tools=[SearchTool(), CalculatorTool()],
            enable_reflection=True,
            enable_tool_filtering=True
        )

        # ToT agent with separate reasoning model
        tot_agent = TreeOfThought(
            api_base="http://localhost:11434/v1",
            model="gemma3:27b",
            reasoning_model="qwen2.5:32b",
            reasoning_temperature=0.9,  # Higher creativity for planning
            temperature=0.5,  # Lower temperature for execution
            enable_reflection=True
        )

        # Run the agent
        result = await tot_agent.run("Analyze the market trends and calculate ROI")
    """
    # Configure execution LLM
    if use_async:
        llm = AsyncOpenAI(
            base_url=api_base,
            api_key=api_key
        )
    else:
        llm = OpenAI(
            base_url=api_base,
            api_key=api_key
        )

    # Configure reasoning LLM (may be different from execution LLM)
    reasoning_llm = None
    if reasoning_model or reasoning_api_base or reasoning_api_key:
        reasoning_base = reasoning_api_base or api_base
        reasoning_key = reasoning_api_key or api_key

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

    # Create memory manager if enabled
    memory_manager = None
    if enable_memory and MEMORY_AVAILABLE:
        memory_manager = create_memory_manager(
            backend_type=memory_backend,
            max_context_tokens=max_context_tokens,
            summary_threshold_tokens=int(max_context_tokens * 0.5),
            llm=llm,
            model=model,
            max_size=max_memory_size
        )
        logger.info(f"[MEMORY] Initialized {memory_backend} memory backend for ToT agent")
    elif enable_memory and not MEMORY_AVAILABLE:
        logger.warning("[MEMORY] Memory requested but module not available")

    agent = TreeOfThoughtAgent(
        llm=llm,
        model=model,
        tools=tools,
        verbose=verbose,
        input_schema=input_schema,
        output_schema=output_schema,
        output_key=output_key,
        state=state,
        max_iterations=max_iterations,
        memory_manager=memory_manager,
        reasoning_model=reasoning_model or model,
        reasoning_llm=reasoning_llm,
        enable_tool_filtering=enable_tool_filtering,
        enable_reflection=enable_reflection,
        max_reflection_depth=max_reflection_depth,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        top_k=top_k,
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
    api_base: str = "http://localhost:11434/v1",
    model: str = "gemma3:27b",
    api_key: str = "not-needed",
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    tools: Optional[List[BaseTool]] = None,
    verbose: bool = True,
    input_schema: Optional[Type[BaseModel]] = None,
    output_schema: Optional[Type[BaseModel]] = None,
    output_key: Optional[str] = None,
    state: Optional[SharedState] = None,
    enable_memory: bool = False,
    memory_backend: str = "in_memory",
    max_context_tokens: int = 4096,
    max_memory_size: Optional[int] = 100,
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
        api_base: The OpenAI-compatible API endpoint
        model: The model name (e.g., "gpt-4", "gemma3:27b")
        api_key: API key for authentication
        temperature: Sampling temperature (0.0 to 2.0)
        max_tokens: Maximum tokens to generate
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter (Ollama-specific)
        tools: List of tools available to the agent
        verbose: Whether to enable verbose logging
        input_schema: Optional Pydantic BaseModel for input validation
        output_schema: Optional Pydantic BaseModel for output
        output_key: Optional key to save output in shared state
        state: Optional SharedState instance
        enable_memory: Whether to enable memory management
        memory_backend: Type of memory backend
        max_context_tokens: Maximum tokens for context window
        max_memory_size: Maximum number of memories to keep
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
        # Basic LightAgent
        agent = Light(
            model="gpt-4",
            instructions="You are a helpful research assistant.",
            tools=[SearchTool(), CalculatorTool()]
        )

        # With streaming
        agent = Light(
            model="gemma3:27b",
            stream=True,
            instructions="You are a coding assistant."
        )

        # In a swarm
        researcher = Light(
            model="gemma3:27b",
            instructions="You are a research specialist.",
            tools=[SearchTool()],
            agent_name="researcher"
        )

        calculator = Light(
            model="gemma3:27b",
            instructions="You are a math specialist.",
            tools=[CalculatorTool()],
            agent_name="calculator"
        )

        from linus.agents.agent.swarm import Swarm
        swarm = Swarm()
        swarm.register(researcher, calculator)

        result = await swarm.run("Calculate 42 * 17")
    """
    # Configure OpenAI client
    if use_async:
        llm = AsyncOpenAI(
            base_url=api_base,
            api_key=api_key
        )
    else:
        llm = OpenAI(
            base_url=api_base,
            api_key=api_key
        )

    if tools is None:
        tools = []

    # Create memory manager if enabled
    memory_manager = None
    if enable_memory and MEMORY_AVAILABLE:
        memory_manager = create_memory_manager(
            backend_type=memory_backend,
            max_context_tokens=max_context_tokens,
            summary_threshold_tokens=int(max_context_tokens * 0.5),
            llm=llm,
            model=model,
            max_size=max_memory_size
        )
        logger.info(f"[MEMORY] Initialized {memory_backend} memory backend for LightAgent")
    elif enable_memory and not MEMORY_AVAILABLE:
        logger.warning("[MEMORY] Memory requested but module not available")

    agent = LightAgent(
        llm=llm,
        model=model,
        tools=tools,
        verbose=verbose,
        input_schema=input_schema,
        output_schema=output_schema,
        output_key=output_key,
        state=state,
        memory_manager=memory_manager,
        instructions=instructions,
        role=role,
        max_tool_iterations=max_tool_iterations,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        top_k=top_k,
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
"""Factory functions for creating agents."""

from typing import List, Optional, Type, Any
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel
from loguru import logger

from .reasoning_agent import ReasoningAgent
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
    session_id: Optional[str] = None
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
        use_json_format=use_json_format
    )

    # Override tracer if provided
    if tracer is not None:
        agent.tracer = tracer

    # If session_id is provided but no tracer, update the agent's tracer if it's a LangfuseTracer
    if session_id is not None and hasattr(agent.tracer, 'session_id'):
        agent.tracer.session_id = session_id

    return agent
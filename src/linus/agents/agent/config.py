"""Configuration classes for agents."""

from typing import Optional
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """Configuration for LLM connection and model settings.

    Attributes:
        api_base: The OpenAI-compatible API endpoint (e.g., "http://localhost:11434/v1" for Ollama)
        model: The model name (e.g., "gemma3:27b", "gpt-4")
        api_key: API key for authentication (default: "not-needed" for Ollama)
    """
    api_base: str = Field(default="http://localhost:11434/v1", description="OpenAI-compatible API endpoint")
    model: str = Field(default="gemma3:27b", description="Model name to use")
    api_key: str = Field(default="not-needed", description="API key for authentication")


class MemoryConfig(BaseModel):
    """Configuration for agent memory management.

    Attributes:
        enable_memory: Whether to enable memory management
        memory_backend: Type of memory backend ("in_memory" or "vector_store")
        max_context_tokens: Maximum tokens for context window (for memory management, not generation)
        max_memory_size: Maximum number of memories to keep (None for unlimited)
    """
    enable_memory: bool = Field(default=False, description="Whether to enable memory management")
    memory_backend: str = Field(default="in_memory", description="Type of memory backend")
    max_context_tokens: int = Field(default=4096, description="Maximum tokens for context window")
    max_memory_size: Optional[int] = Field(default=100, description="Maximum number of memories to keep")


class AgentParams(BaseModel):
    """Consolidated configuration for agent parameters.

    This class includes all configuration needed for an agent:
    - LLM generation parameters (temperature, max_tokens, top_p, top_k)
    - Memory configuration (enable_memory, memory_backend, max_context_tokens, max_memory_size)
    - LLM connection settings (api_base, model, api_key)

    Attributes:
        temperature: Sampling temperature (0.0 to 2.0). Higher = more random/creative
        max_tokens: Maximum tokens to generate in completion (None = model default)
        top_p: Nucleus sampling parameter (0.0 to 1.0). Alternative to temperature
        top_k: Top-k sampling parameter. Only available on some models like Ollama
        memory_config: Memory management configuration
        llm_config: LLM connection and model settings
    """
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens to generate")
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    top_k: Optional[int] = Field(default=None, description="Top-k sampling parameter")
    memory_config: MemoryConfig = Field(default_factory=MemoryConfig, description="Memory configuration")
    llm_config: LLMConfig = Field(default_factory=LLMConfig, description="LLM connection settings")

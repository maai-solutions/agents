"""Application settings configuration."""

from pydantic_settings import BaseSettings, CliSettingsSource, SettingsConfigDict
from typing import Optional, Tuple, Type
from pydantic.fields import FieldInfo


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    )

    app_name: str = "ReasoningAgent API"
    app_version: str = "1.0.0"

    # Gemma/Ollama configuration
    llm_api_base: str = "http://localhost:11434/v1"
    llm_model: str = "gemma3:27b"
    llm_api_key: str = "not-needed"
    llm_temperature: float = 0.7
    llm_max_tokens: Optional[int] = 4096
    llm_top_p: Optional[float] = 1.0
    llm_top_k: Optional[int] = 50
    llm_use_json_format: bool = False

    # Agent configuration
    agent_verbose: bool = True
    agent_timeout: int = 300  # 5 minutes timeout

    # Telemetry configuration
    telemetry_enabled: bool = True
    telemetry_exporter: str = "langfuse"  # console, otlp, jaeger, langfuse
    telemetry_otlp_endpoint: str = "http://localhost:4317"
    telemetry_jaeger_endpoint: str = "localhost"
    telemetry_public_key: str = ""
    telemetry_secret_key: str = ""

    # Langfuse configuration (aliases for telemetry keys)
    langfuse_public_key: Optional[str] = None
    langfuse_secret_key: Optional[str] = None
    langfuse_host: str = "https://cloud.langfuse.com"

    # API configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Weaviate configuration
    wv_url: str = "localhost"
    wv_port: int = 18080
    wv_scheme: str = "http"
    wv_collection: str = "arag_dev"
    wv_max_distance: float = 0.7
    wv_alpha: float = 0.5
    wv_limit: int = 5
    wv_embedding_model: str = "nomic-embed-text:latest"

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ) -> Tuple:
        """Customize the priority of settings sources.

        Priority (highest to lowest):
        1. CLI arguments
        2. Environment variables
        3. .env file
        4. Default values
        """
        return (
            init_settings,
            CliSettingsSource(
                settings_cls,
                cli_parse_args=True,
                cli_prog_name="app",
            ),
            env_settings,
            dotenv_settings,
            file_secret_settings,
        )

"""Example custom tool: Text translation tool."""

from pydantic import BaseModel, Field
import sys
import os

# Add parent directory to path to import BaseTool
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from linus.agents.agent.tool_base import BaseTool


class TranslatorInput(BaseModel):
    """Input schema for translator tool."""
    text: str = Field(description="Text to translate")
    target_language: str = Field(description="Target language code (e.g., 'es', 'fr', 'de')")
    source_language: str = Field(default="auto", description="Source language code (auto-detect if not specified)")


class TranslatorTool(BaseTool):
    """Tool for translating text between languages.

    This is a mock implementation for demonstration purposes.
    In production, you would call an actual translation API.
    """

    name: str = "translator"
    description: str = "Translate text from one language to another"
    args_schema = TranslatorInput

    async def _arun(
        self,
        text: str,
        target_language: str,
        source_language: str = "auto"
    ) -> str:
        """Translate text.

        Args:
            text: Text to translate
            target_language: Target language code
            source_language: Source language code

        Returns:
            Translated text string
        """
        # Mock translation
        language_names = {
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "it": "Italian",
            "pt": "Portuguese"
        }

        lang_name = language_names.get(target_language, target_language)

        return f"[Mock translation to {lang_name}]: {text} (translated)"

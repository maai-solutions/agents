"""Example custom tool: Weather lookup tool."""

from pydantic import BaseModel, Field
import sys
import os

# Add parent directory to path to import BaseTool
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from linus.agents.agent.tool_base import BaseTool


class WeatherInput(BaseModel):
    """Input schema for weather tool."""
    city: str = Field(description="City name to get weather for")
    units: str = Field(default="celsius", description="Temperature units (celsius or fahrenheit)")


class WeatherTool(BaseTool):
    """Tool for getting weather information for a city.

    This is a mock implementation for demonstration purposes.
    In production, you would call an actual weather API.
    """

    name: str = "weather"
    description: str = "Get current weather information for a specified city"
    args_schema = WeatherInput

    async def _arun(self, city: str, units: str = "celsius") -> str:
        """Get weather for a city.

        Args:
            city: City name
            units: Temperature units (celsius or fahrenheit)

        Returns:
            Weather information string
        """
        # Mock weather data
        temp_c = 22
        temp_f = 72

        temp_str = f"{temp_c}°C" if units == "celsius" else f"{temp_f}°F"

        return f"Weather in {city}: {temp_str}, partly cloudy with light winds"

"""Base tool classes - pure Python implementation without LangChain."""

import os
import importlib
import importlib.util
from typing import Optional, Type, Dict, Any, Callable, List
from pydantic import BaseModel
from abc import ABC, abstractmethod
from copy import deepcopy
import json


class BaseTool(ABC):
    """Base class for all tools.

    This is a pure Python implementation replacing LangChain's BaseTool.
    """

    name: str
    description: str
    args_schema: Optional[Type[BaseModel]] = None

    def __init__(self):
        """Initialize the tool."""
        if not hasattr(self, 'name'):
            self.name = self.__class__.__name__.lower().replace('tool', '')
        if not hasattr(self, 'description'):
            self.description = self.__class__.__doc__ or "No description available"

    @abstractmethod
    async def _arun(self, *args, **kwargs) -> str:
        """Async execution of the tool.

        Must be implemented by subclasses.
        """
        raise NotImplementedError(f"Async execution not implemented for {self.name}")

    async def arun(self, tool_input: Dict[str, Any]) -> str:
        """Async version of run.

        Args:
            tool_input: Dictionary of arguments for the tool

        Returns:
            Result of the tool execution as a string
        """
        # Validate input against schema if provided
        if self.args_schema:
            try:
                validated_input = self.args_schema(**tool_input)
                tool_input = validated_input.model_dump()
            except Exception as e:
                return f"Error validating tool input: {str(e)}"

        # Execute the tool
        try:
            return await self._arun(**tool_input)
        except Exception as e:
            return f"Error executing tool '{self.name}': {str(e)}"


class StructuredTool(BaseTool):
    """A tool created from a function with structured arguments."""

    def __init__(
        self,
        name: str,
        description: str,
        func: Callable,
        args_schema: Optional[Type[BaseModel]] = None
    ):
        """Initialize a structured tool.

        Args:
            name: Name of the tool
            description: Description of what the tool does
            func: Function to execute
            args_schema: Pydantic model for argument validation
        """
        self.name = name
        self.description = description
        self.func = func
        self.args_schema = args_schema

    @classmethod
    def from_function(
        cls,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
        args_schema: Optional[Type[BaseModel]] = None
    ) -> 'StructuredTool':
        """Create a StructuredTool from a function.

        Args:
            func: Function to wrap
            name: Tool name (defaults to function name)
            description: Tool description (defaults to function docstring)
            args_schema: Pydantic model for argument validation

        Returns:
            StructuredTool instance
        """
        tool_name = name or func.__name__
        tool_description = description or func.__doc__ or f"Tool {tool_name}"
        return cls(
            name=tool_name,
            description=tool_description,
            func=func,
            args_schema=args_schema
        )

    async def _arun(self, **kwargs) -> str:
        """Async execution."""
        # If the function is async, await it
        import inspect
        if inspect.iscoroutinefunction(self.func):
            result = await self.func(**kwargs)
        else:
            result = self.func(**kwargs)
        return str(result)


def tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    args_schema: Optional[Type[BaseModel]] = None
):
    """Decorator to create a tool from a function.

    Usage:
        @tool(name="my_tool", description="Does something cool")
        def my_function(arg1: str, arg2: int) -> str:
            return f"Result: {arg1} {arg2}"

    Args:
        name: Tool name (defaults to function name)
        description: Tool description (defaults to function docstring)
        args_schema: Pydantic model for arguments

    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> StructuredTool:
        tool_name = name or func.__name__
        tool_description = description or func.__doc__ or f"Tool {tool_name}"
        return StructuredTool(
            name=tool_name,
            description=tool_description,
            func=func,
            args_schema=args_schema
        )
    return decorator


class ToolRegistry:
    """Centralized tool registry for managing and organizing tools."""

    def __init__(self):
        self.tool_map: Dict[str, BaseTool] = {}  # Tool name -> Tool instance
        self.tool_schemas: List[Dict[str, Any]] = []  # OpenAI format schemas

    def register_tool(self, tool: BaseTool) -> bool:
        """Register a single tool.

        Args:
            tool: BaseTool instance to register

        Returns:
            True if registration successful, False otherwise
        """
        if not isinstance(tool, BaseTool):
            return False

        tool_name = tool.name
        self.tool_map[tool_name] = tool

        # Build OpenAI-compatible schema
        schema = self._build_openai_schema(tool)
        self.tool_schemas.append(schema)
        return True

    def register_tools(self, tools: List[BaseTool]) -> bool:
        """Batch register multiple tools.

        Args:
            tools: List of BaseTool instances

        Returns:
            True if all registrations successful, False otherwise
        """
        success = True
        for tool in tools:
            if not self.register_tool(tool):
                success = False
        return success

    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """Get a tool by name.

        Args:
            tool_name: Name of the tool

        Returns:
            BaseTool instance or None if not found
        """
        return self.tool_map.get(tool_name)

    def get_tools(self) -> List[Dict[str, Any]]:
        """Get all tool schemas in OpenAI format.

        Returns:
            List of tool schemas
        """
        return deepcopy(self.tool_schemas)

    def get_tools_str(self) -> str:
        """Convert tool schemas to formatted JSON string.

        Returns:
            JSON string of tool schemas
        """
        return json.dumps(self.tool_schemas, indent=4, ensure_ascii=False)

    def _build_openai_schema(self, tool: BaseTool) -> Dict[str, Any]:
        """Build OpenAI-compatible tool schema.

        Args:
            tool: BaseTool instance

        Returns:
            OpenAI format tool schema
        """
        schema = {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        }

        # Extract schema from Pydantic model if available
        if tool.args_schema:
            json_schema = tool.args_schema.model_json_schema()
            properties = json_schema.get("properties", {})
            required = json_schema.get("required", [])

            schema["function"]["parameters"]["properties"] = properties
            schema["function"]["parameters"]["required"] = required

        return schema


class ToolLoader:
    """Dynamic tool loader supporting file-based tool loading."""

    def __init__(self, tools_directory: str = "tools"):
        """Initialize tool loader.

        Args:
            tools_directory: Directory containing tool Python files
        """
        self.tools_directory = tools_directory
        self.loaded_tools: Dict[str, BaseTool] = {}

    def load_tool(self, tool_name: str) -> BaseTool:
        """Load a single tool from file.

        Args:
            tool_name: Name of the tool (filename without .py)

        Returns:
            BaseTool instance

        Raises:
            FileNotFoundError: If tool file doesn't exist
            AttributeError: If tool is not properly defined
        """
        # Return cached tool if already loaded
        if tool_name in self.loaded_tools:
            return self.loaded_tools[tool_name]

        # Construct file path
        tool_path = os.path.join(self.tools_directory, f"{tool_name}.py")
        if not os.path.exists(tool_path):
            raise FileNotFoundError(f"Tool '{tool_name}' not found at {tool_path}")

        # Dynamic module loading
        spec = importlib.util.spec_from_file_location(tool_name, tool_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Try to find the tool in the module
        # Convention: look for a class ending with 'Tool' or matching tool_name
        tool_instance = None

        # First, try to find a class matching tool_name (case-insensitive)
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and issubclass(attr, BaseTool) and attr is not BaseTool:
                # Instantiate the tool class
                tool_instance = attr()
                break

        if tool_instance is None:
            raise AttributeError(
                f"No BaseTool subclass found in {tool_path}. "
                f"Ensure the file contains a class inheriting from BaseTool."
            )

        # Cache the loaded tool
        self.loaded_tools[tool_name] = tool_instance
        return tool_instance

    def load_tools(self, tool_names: List[str]) -> Dict[str, BaseTool]:
        """Batch load multiple tools.

        Args:
            tool_names: List of tool names to load

        Returns:
            Dictionary mapping tool names to BaseTool instances
        """
        for tool_name in tool_names:
            if tool_name not in self.loaded_tools:
                self.load_tool(tool_name)
        return self.loaded_tools

    def discover_tools(self) -> List[str]:
        """Discover all available tools in the tools directory.

        Returns:
            List of discovered tool names
        """
        if not os.path.exists(self.tools_directory):
            return []

        tool_names = []
        for filename in os.listdir(self.tools_directory):
            if filename.endswith('.py') and not filename.startswith('_'):
                tool_name = filename[:-3]  # Remove .py extension
                tool_names.append(tool_name)
        return tool_names

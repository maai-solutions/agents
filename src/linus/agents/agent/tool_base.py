"""Base tool classes - pure Python implementation without LangChain."""

from typing import Optional, Type, Dict, Any, Callable
from pydantic import BaseModel
from abc import ABC, abstractmethod


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
    def _run(self, *args, **kwargs) -> str:
        """Execute the tool's action.

        Must be implemented by subclasses.
        """
        raise NotImplementedError("Tool must implement _run method")

    def run(self, tool_input: Dict[str, Any]) -> str:
        """Run the tool with the given input.

        Args:
            tool_input: Dictionary of arguments for the tool

        Returns:
            Result of the tool execution as a string
        """
        # Validate input against schema if provided
        if self.args_schema:
            try:
                validated_input = self.args_schema(**tool_input)
                # Convert back to dict for _run method
                tool_input = validated_input.model_dump()
            except Exception as e:
                return f"Error validating tool input: {str(e)}"

        # Execute the tool
        try:
            return self._run(**tool_input)
        except Exception as e:
            return f"Error executing tool '{self.name}': {str(e)}"

    async def _arun(self, *args, **kwargs) -> str:
        """Async execution of the tool.

        Optional - can be overridden by subclasses.
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

    def _run(self, **kwargs) -> str:
        """Execute the function."""
        result = self.func(**kwargs)
        return str(result)

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

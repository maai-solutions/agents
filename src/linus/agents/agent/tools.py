"""Example tools for the ReasoningAgent."""

from typing import Optional, Type, Any
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field
from langchain_core.callbacks import CallbackManagerForToolRun
import subprocess
import requests
import json


class SearchInput(BaseModel):
    """Input for the search tool."""
    query: str = Field(description="The search query")
    limit: int = Field(default=5, description="Maximum number of results to return")


class SearchTool(BaseTool):
    """Tool for searching information."""
    name: str = "search"
    description: str = "Search for information on a given topic"
    args_schema: Type[BaseModel] = SearchInput
    
    def _run(
        self,
        query: str,
        limit: int = 5,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute the search."""
        # This is a mock implementation - replace with actual search API
        return f"Search results for '{query}' (limited to {limit} results):\n1. Result about {query}\n2. Another result\n3. More information"
    
    async def _arun(self, *args, **kwargs):
        """Async version not implemented."""
        raise NotImplementedError("Async search not supported")


class CalculatorInput(BaseModel):
    """Input for the calculator tool."""
    expression: str = Field(description="Mathematical expression to evaluate")


class CalculatorTool(BaseTool):
    """Tool for performing calculations."""
    name: str = "calculator"
    description: str = "Perform mathematical calculations"
    args_schema: Type[BaseModel] = CalculatorInput
    
    def _run(
        self,
        expression: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute the calculation."""
        try:
            # SECURITY WARNING: eval is dangerous in production!
            # Use a proper math expression parser instead
            result = eval(expression, {"__builtins__": {}}, {})
            return f"Result: {result}"
        except Exception as e:
            return f"Error calculating '{expression}': {str(e)}"
    
    async def _arun(self, *args, **kwargs):
        """Async version not implemented."""
        raise NotImplementedError("Async calculation not supported")


class FileReaderInput(BaseModel):
    """Input for the file reader tool."""
    file_path: str = Field(description="Path to the file to read")
    encoding: str = Field(default="utf-8", description="File encoding")


class FileReaderTool(BaseTool):
    """Tool for reading files."""
    name: str = "read_file"
    description: str = "Read contents of a file"
    args_schema: Type[BaseModel] = FileReaderInput
    
    def _run(
        self,
        file_path: str,
        encoding: str = "utf-8",
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Read the file."""
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
            return f"File contents:\n{content}"
        except FileNotFoundError:
            return f"Error: File '{file_path}' not found"
        except Exception as e:
            return f"Error reading file '{file_path}': {str(e)}"
    
    async def _arun(self, *args, **kwargs):
        """Async version not implemented."""
        raise NotImplementedError("Async file reading not supported")


class ShellCommandInput(BaseModel):
    """Input for shell command execution."""
    command: str = Field(description="Shell command to execute")
    timeout: int = Field(default=30, description="Command timeout in seconds")


class ShellCommandTool(BaseTool):
    """Tool for executing shell commands."""
    name: str = "shell_command"
    description: str = "Execute a shell command (use with caution)"
    args_schema: Type[BaseModel] = ShellCommandInput
    
    def _run(
        self,
        command: str,
        timeout: int = 30,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute the shell command."""
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            output = result.stdout if result.stdout else result.stderr
            return f"Command output:\n{output}"
        except subprocess.TimeoutExpired:
            return f"Error: Command '{command}' timed out after {timeout} seconds"
        except Exception as e:
            return f"Error executing command '{command}': {str(e)}"
    
    async def _arun(self, *args, **kwargs):
        """Async version not implemented."""
        raise NotImplementedError("Async shell execution not supported")


class APIRequestInput(BaseModel):
    """Input for API requests."""
    url: str = Field(description="API endpoint URL")
    method: str = Field(default="GET", description="HTTP method")
    headers: dict = Field(default_factory=dict, description="Request headers")
    data: Optional[dict] = Field(default=None, description="Request body data")


class APIRequestTool(BaseTool):
    """Tool for making API requests."""
    name: str = "api_request"
    description: str = "Make HTTP API requests"
    args_schema: Type[BaseModel] = APIRequestInput
    
    def _run(
        self,
        url: str,
        method: str = "GET",
        headers: dict = None,
        data: dict = None,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Make the API request."""
        try:
            headers = headers or {}
            
            if method.upper() == "GET":
                response = requests.get(url, headers=headers, params=data)
            elif method.upper() == "POST":
                response = requests.post(url, headers=headers, json=data)
            elif method.upper() == "PUT":
                response = requests.put(url, headers=headers, json=data)
            elif method.upper() == "DELETE":
                response = requests.delete(url, headers=headers)
            else:
                return f"Error: Unsupported HTTP method '{method}'"
            
            response.raise_for_status()
            
            try:
                result = response.json()
                return f"API Response:\n{json.dumps(result, indent=2)}"
            except json.JSONDecodeError:
                return f"API Response:\n{response.text}"
                
        except requests.RequestException as e:
            return f"Error making API request to '{url}': {str(e)}"
    
    async def _arun(self, *args, **kwargs):
        """Async version not implemented."""
        raise NotImplementedError("Async API requests not supported")


def get_default_tools() -> list[BaseTool]:
    """Get a list of default tools for the agent."""
    return [
        SearchTool(),
        CalculatorTool(),
        FileReaderTool(),
        ShellCommandTool(),
        APIRequestTool()
    ]


def create_custom_tool(
    name: str,
    description: str,
    func: callable,
    args_schema: Optional[Type[BaseModel]] = None
) -> StructuredTool:
    """Create a custom tool from a function.
    
    Args:
        name: Tool name
        description: Tool description
        func: The function to wrap
        args_schema: Optional Pydantic model for arguments
        
    Returns:
        A StructuredTool instance
    """
    return StructuredTool.from_function(
        func=func,
        name=name,
        description=description,
        args_schema=args_schema
    )
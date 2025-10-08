"""Data models for agent system."""

from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from pydantic import BaseModel


@dataclass
class ReasoningResult:
    """Result from the reasoning phase."""
    has_sufficient_info: bool
    tasks: List[Dict[str, Any]]
    reasoning: str


@dataclass
class TaskExecution:
    """Represents a task to be executed."""
    description: str
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    completed: bool = False
    result: Optional[Any] = None


@dataclass
class AgentMetrics:
    """Metrics collected during agent execution."""
    total_iterations: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    execution_time_seconds: float = 0.0
    reasoning_calls: int = 0
    tool_executions: int = 0
    completion_checks: int = 0
    llm_calls: int = 0
    successful_tool_calls: int = 0
    failed_tool_calls: int = 0
    task_completed: bool = False
    iterations_to_completion: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "total_iterations": self.total_iterations,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "execution_time_seconds": round(self.execution_time_seconds, 3),
            "reasoning_calls": self.reasoning_calls,
            "tool_executions": self.tool_executions,
            "completion_checks": self.completion_checks,
            "llm_calls": self.llm_calls,
            "successful_tool_calls": self.successful_tool_calls,
            "failed_tool_calls": self.failed_tool_calls,
            "task_completed": self.task_completed,
            "iterations_to_completion": self.iterations_to_completion,
            "avg_tokens_per_llm_call": round(self.total_tokens / self.llm_calls, 2) if self.llm_calls > 0 else 0,
            "success_rate": round(self.successful_tool_calls / self.tool_executions, 2) if self.tool_executions > 0 else 0
        }


@dataclass
class AgentResponse:
    """Complete agent response with result and metrics."""
    result: Union[str, BaseModel]
    metrics: AgentMetrics
    execution_history: List[Dict[str, Any]] = field(default_factory=list)
    completion_status: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary."""
        result_value = self.result
        if isinstance(self.result, BaseModel):
            result_value = self.result.model_dump()

        return {
            "result": result_value,
            "metrics": self.metrics.to_dict(),
            "execution_history": self.execution_history,
            "completion_status": self.completion_status
        }

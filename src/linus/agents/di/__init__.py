"""Dependency Injection module for the agent framework."""

from .container import Container, get_container
from .interfaces import ILogger, ITelemetry
from .providers import LoggerProvider, TelemetryProvider

__all__ = [
    "Container",
    "get_container",
    "ILogger",
    "ITelemetry",
    "LoggerProvider",
    "TelemetryProvider",
]

"""Dependency injection container.

This module provides a simple DI container for managing service instances
and their lifecycles across the agent framework.
"""

from typing import Any, Dict, Optional, Callable, TypeVar
from .interfaces import ILogger, ITelemetry, NoOpLogger, NoOpTelemetry
from .providers import LoggerProvider, TelemetryProvider

T = TypeVar('T')


class Container:
    """Simple dependency injection container.

    This container manages singleton instances of services and provides
    a centralized way to configure and access them throughout the framework.

    Example:
        # Setup container
        container = Container()
        container.register_logger(LoggerProvider.create_loguru_logger())
        container.register_telemetry(TelemetryProvider.create_from_config(
            exporter_type="langfuse",
            enabled=True
        ))

        # Use in components
        logger = container.get_logger()
        telemetry = container.get_telemetry()
    """

    def __init__(self):
        """Initialize the container."""
        self._services: Dict[type, Any] = {}
        self._factories: Dict[type, Callable] = {}

        # Register default no-op implementations
        self._services[ILogger] = NoOpLogger()
        self._services[ITelemetry] = NoOpTelemetry()

    def register(self, interface: type, instance: Any) -> None:
        """Register a service instance.

        Args:
            interface: Interface type (e.g., ILogger, ITelemetry)
            instance: Service instance implementing the interface
        """
        self._services[interface] = instance

    def register_factory(self, interface: type, factory: Callable[[], Any]) -> None:
        """Register a factory function for lazy instantiation.

        Args:
            interface: Interface type
            factory: Factory function that creates the service instance
        """
        self._factories[interface] = factory

    def get(self, interface: type) -> Any:
        """Get a service instance.

        Args:
            interface: Interface type to retrieve

        Returns:
            Service instance implementing the interface
        """
        # Check if we have an instance
        if interface in self._services:
            return self._services[interface]

        # Check if we have a factory
        if interface in self._factories:
            instance = self._factories[interface]()
            self._services[interface] = instance
            return instance

        raise ValueError(f"No service registered for interface: {interface}")

    def has(self, interface: type) -> bool:
        """Check if a service is registered.

        Args:
            interface: Interface type to check

        Returns:
            True if service is registered, False otherwise
        """
        return interface in self._services or interface in self._factories

    def clear(self) -> None:
        """Clear all registered services."""
        self._services.clear()
        self._factories.clear()

        # Re-register defaults
        self._services[ILogger] = NoOpLogger()
        self._services[ITelemetry] = NoOpTelemetry()

    # Convenience methods for common services

    def register_logger(self, logger: ILogger) -> None:
        """Register a logger service.

        Args:
            logger: Logger instance implementing ILogger
        """
        self.register(ILogger, logger)

    def register_telemetry(self, telemetry: ITelemetry) -> None:
        """Register a telemetry service.

        Args:
            telemetry: Telemetry instance implementing ITelemetry
        """
        self.register(ITelemetry, telemetry)

    def get_logger(self) -> ILogger:
        """Get the logger service.

        Returns:
            Logger instance implementing ILogger
        """
        return self.get(ILogger)

    def get_telemetry(self) -> ITelemetry:
        """Get the telemetry service.

        Returns:
            Telemetry instance implementing ITelemetry
        """
        return self.get(ITelemetry)

    def configure_defaults(
        self,
        use_loguru: bool = True,
        telemetry_enabled: bool = False,
        telemetry_config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Configure default services.

        Args:
            use_loguru: Whether to use Loguru for logging (default: True)
            telemetry_enabled: Whether to enable telemetry (default: False)
            telemetry_config: Configuration dict for telemetry (optional)
        """
        # Configure logger
        if use_loguru:
            self.register_logger(LoggerProvider.create_loguru_logger())
        else:
            self.register_logger(LoggerProvider.create_noop_logger())

        # Configure telemetry
        if telemetry_enabled and telemetry_config:
            telemetry = TelemetryProvider.create_from_config(**telemetry_config)
            self.register_telemetry(telemetry)
        elif not telemetry_enabled:
            self.register_telemetry(TelemetryProvider.create_noop_telemetry())


# Global container instance
_global_container: Optional[Container] = None


def get_container() -> Container:
    """Get the global container instance.

    Returns:
        Global Container instance
    """
    global _global_container
    if _global_container is None:
        _global_container = Container()
        # Configure with sensible defaults
        _global_container.configure_defaults()
    return _global_container


def set_container(container: Container) -> None:
    """Set the global container instance.

    Args:
        container: Container instance to set as global
    """
    global _global_container
    _global_container = container


def reset_container() -> None:
    """Reset the global container to a new instance."""
    global _global_container
    _global_container = None

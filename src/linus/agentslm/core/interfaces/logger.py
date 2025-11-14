from abc import ABC, abstractmethod


class ILogger(ABC):
    """Interface for logging service."""

    @abstractmethod
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        pass

    @abstractmethod
    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        pass

    @abstractmethod
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        pass

    @abstractmethod
    def error(self, message: str, **kwargs) -> None:
        """Log error message."""
        pass

    @abstractmethod
    def exception(self, message: str, **kwargs) -> None:
        """Log exception with traceback."""
        pass

class NoOpLogger(ILogger):
    """No-op logger implementation for when logging is disabled."""

    def debug(self, message: str, **kwargs) -> None:
        pass

    def info(self, message: str, **kwargs) -> None:
        pass

    def warning(self, message: str, **kwargs) -> None:
        pass

    def error(self, message: str, **kwargs) -> None:
        pass

    def exception(self, message: str, **kwargs) -> None:
        pass
"""Custom exception types for NirCLS components."""


class DataError(Exception):
    """Exception raised when input data is missing or invalid."""


class ModelError(Exception):
    """Exception raised when model configuration is invalid or incompatible."""


class ConfigError(Exception):
    """Exception raised when configuration values are invalid or inconsistent."""


class PipelineError(Exception):
    """Exception raised when a processing pipeline step fails."""


class DatabaseError(Exception):
    """Exception raised when database operations fail."""


class AuthError(Exception):
    """Exception raised when authentication fails."""


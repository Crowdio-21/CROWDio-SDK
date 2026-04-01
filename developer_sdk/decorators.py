"""
Declarative checkpointing SDK for CROWDio

Provides decorators and metadata classes for task definition with
automatic checkpoint support. Developers declare:
- Whether checkpointing is enabled
- Checkpoint frequency (interval in seconds)
- Which state variables should be checkpointed

The system automatically handles:
- State capture and serialization
- Async incremental checkpointing
- Recovery and resumption on worker failure
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from functools import wraps
import inspect
from .constants import CROWDioConstant


@dataclass
class CROWDioTaskMetadata:
    """
    Metadata for a task's checkpointing configuration

    Stored alongside task submission and used by workers
    to manage checkpoint capture and recovery.

    Attributes:
        checkpoint_enabled: Whether checkpointing is active for this task
        checkpoint_interval: Seconds between checkpoint captures
        checkpoint_state: List of variable names to checkpoint
        parallel: Whether task supports parallel execution
        retry_on_failure: Whether to retry task on failure
        max_retries: Maximum number of retry attempts
        timeout: Task timeout in seconds (None = no timeout)
    """

    checkpoint_enabled: bool = False
    checkpoint_interval: float = 10.0
    checkpoint_state: List[str] = field(default_factory=list)
    parallel: bool = True
    retry_on_failure: bool = True
    max_retries: int = 3
    timeout: Optional[float] = None

    # Internal tracking
    _func_name: str = ""
    _validated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize metadata to dictionary for transmission"""
        return {
            "checkpoint_enabled": self.checkpoint_enabled,
            "checkpoint_interval": self.checkpoint_interval,
            "checkpoint_state": list(self.checkpoint_state),
            "parallel": self.parallel,
            "retry_on_failure": self.retry_on_failure,
            "max_retries": self.max_retries,
            "timeout": self.timeout,
            "_func_name": self._func_name,
            "_validated": self._validated,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CROWDioTaskMetadata":
        """Deserialize metadata from dictionary"""
        return cls(
            checkpoint_enabled=data.get("checkpoint_enabled", False),
            checkpoint_interval=data.get("checkpoint_interval", 10.0),
            checkpoint_state=data.get("checkpoint_state", []),
            parallel=data.get("parallel", True),
            retry_on_failure=data.get("retry_on_failure", True),
            max_retries=data.get("max_retries", 3),
            timeout=data.get("timeout"),
            _func_name=data.get("_func_name", ""),
            _validated=data.get("_validated", False),
        )

    def validate_state_variables(self, state: Dict[str, Any]) -> bool:
        """
        Validate that all declared checkpoint variables exist in state

        Args:
            state: Current state dictionary

        Returns:
            True if all declared variables are present

        Raises:
            ValueError: If any declared variable is missing from state
        """
        if not self.checkpoint_enabled:
            return True

        missing = set(self.checkpoint_state) - set(state.keys())
        if missing:
            raise ValueError(
                f"Checkpoint validation failed for task '{self._func_name}': "
                f"Declared checkpoint variables not found in state: {missing}. "
                f"Available state variables: {set(state.keys())}"
            )

        self._validated = True
        return True

    def filter_checkpoint_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter state to only include declared checkpoint variables

        Args:
            state: Full state dictionary

        Returns:
            Filtered state with only declared checkpoint variables
        """
        if not self.checkpoint_enabled or not self.checkpoint_state:
            # If no specific variables declared, checkpoint all
            return dict(state)

        return {key: state[key] for key in self.checkpoint_state if key in state}


class CROWDioTaskConfig:
    """
    Configuration wrapper for task functions

    Attached to decorated functions via __crowdio_config__ attribute.
    """

    def __init__(self, metadata: CROWDioTaskMetadata, original_func: Callable):
        self.metadata = metadata
        self.original_func = original_func
        self._wrapper_func: Optional[Callable] = None

    def get_metadata(self) -> CROWDioTaskMetadata:
        """Get task metadata"""
        return self.metadata

    def is_checkpoint_enabled(self) -> bool:
        """Check if checkpointing is enabled"""
        return self.metadata.checkpoint_enabled


def CROWDio_task(
    parallel: bool = True,
    checkpoint: bool = False,
    checkpoint_interval: float = 10.0,
    checkpoint_state: Optional[List[str]] = None,
    retry_on_failure: bool = True,
    max_retries: int = 3,
    timeout: Optional[float] = None,
) -> Callable:
    """
    Decorator for declaring a CROWDio task with checkpoint support

    Usage:
        @CROWDio.task(
            parallel=True,
            checkpoint=True,
            checkpoint_interval=5,
            checkpoint_state=["i", "partial_result", "progress_percent"]
        )
        def my_task(state, data):
            for i in range(state.get("i", 0), len(data)):
                state["i"] = i
                state["partial_result"] = process(data[i])
                state["progress_percent"] = (i + 1) / len(data) * 100
            return state["partial_result"]

    Args:
        parallel: Whether task supports parallel execution across workers
        checkpoint: Enable automatic checkpointing
        checkpoint_interval: Seconds between checkpoint captures
        checkpoint_state: List of state variable names to checkpoint.
                         If None and checkpoint=True, all state is checkpointed.
        retry_on_failure: Automatically retry failed tasks
        max_retries: Maximum retry attempts
        timeout: Task timeout in seconds (None = no timeout)

    Returns:
        Decorated function with CROWDioTaskConfig attached
    """

    def decorator(func: Callable) -> Callable:
        # Create metadata for this task
        metadata = CROWDioTaskMetadata(
            checkpoint_enabled=checkpoint,
            checkpoint_interval=checkpoint_interval,
            checkpoint_state=checkpoint_state or [],
            parallel=parallel,
            retry_on_failure=retry_on_failure,
            max_retries=max_retries,
            timeout=timeout,
            _func_name=func.__name__,
        )

        # Create task config
        config = CROWDioTaskConfig(metadata, func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            """
            Task wrapper that handles state injection and checkpoint support

            When executed on a worker:
            1. If resuming: state is pre-loaded from checkpoint
            2. State is automatically tracked for checkpointing
            3. Only declared variables are captured
            """
            return func(*args, **kwargs)

        # Attach config to wrapper for SDK access
        wrapper.__crowdio_config__ = config
        wrapper.__crowdio_metadata__ = metadata

        # Store original function for serialization
        wrapper.__crowdio_original__ = func

        return wrapper

    return decorator


def CROWDio_get_task_metadata(func: Callable) -> Optional[CROWDioTaskMetadata]:
    """
    Extract CROWDioTaskMetadata from a decorated function

    Args:
        func: Potentially decorated function

    Returns:
        CROWDioTaskMetadata if function was decorated with @CROWDio_task, else None
    """
    if hasattr(func, "__crowdio_metadata__"):
        return func.__crowdio_metadata__
    return None


def CROWDio_get_task_config(func: Callable) -> Optional[CROWDioTaskConfig]:
    """
    Extract CROWDioTaskConfig from a decorated function

    Args:
        func: Potentially decorated function

    Returns:
        CROWDioTaskConfig if function was decorated with @CROWDio_task, else None
    """
    if hasattr(func, "__crowdio_config__"):
        return func.__crowdio_config__
    return None


def CROWDio_is_checkpoint_task(func: Callable) -> bool:
    """
    Check if a function has checkpointing enabled

    Args:
        func: Function to check

    Returns:
        True if function has checkpoint=True decorator
    """
    metadata = CROWDio_get_task_metadata(func)
    return metadata is not None and metadata.checkpoint_enabled


def CROWDio_create_state_dict(checkpoint_state: List[str]) -> Dict[str, Any]:
    """
    Create an initial state dictionary with declared checkpoint variables

    Args:
        checkpoint_state: List of variable names to track

    Returns:
        Dictionary with all variables initialized to None
    """
    return {var: None for var in checkpoint_state}


class CROWDioNamespace:
    """Namespace class for @CROWDio.task decorator style"""

    task = staticmethod(CROWDio_task)
    Constant = CROWDioConstant


CROWDio = CROWDioNamespace()

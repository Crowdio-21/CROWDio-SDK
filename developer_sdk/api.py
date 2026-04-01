from .client import CrowdComputeClient
from .decorators import (
    CROWDio_task,
    CROWDioTaskMetadata,
    CROWDioTaskConfig,
    CROWDio_get_task_metadata,
    CROWDio_get_task_config,
    CROWDio_is_checkpoint_task,
    CROWDio_create_state_dict,
    CROWDio,
)
from .constants import CROWDioConstant
from typing import Any, Callable, List, Optional, Dict

# Global client instance
_client = CrowdComputeClient()

# ============================================================================
# CROWDio Distributed Computing API
# ============================================================================

async def CROWDio_connect(host: str, port: int = 9000):
    """
    CROWDio_connect - Initialize the CROWDio environment and connect to foreman

    Establishes connection to the distributed computing cluster (foreman server).
    Must be called before any distributed operations.

    Args:
        host: Hostname/IP of foreman server
        port: Port number (default 9000)
    """
    await _client.connect(host, port)


async def CROWDio_disconnect():
    """
    CROWDio_disconnect - Finalize CROWDio environment and disconnect

    Cleanly closes the connection to the foreman server.
    Should be called when all distributed work is complete.
    """
    await _client.disconnect()


async def CROWDio_map(func: Callable, iterable: List[Any], **kwargs) -> List[Any]:
    """
    CROWDio_map - Distribute function across all workers and gather results

    Sends tasks to all available workers in a collective operation.
    If function is decorated with @CROWDio_task, checkpoint metadata is
    automatically extracted and sent to workers for checkpoint-aware execution.

    Args:
        func: Function to execute (optionally decorated with @CROWDio_task)
        iterable: List of arguments (one per worker process)
        **kwargs: Additional options:
            - checkpoint: Override checkpoint setting (bool)
            - checkpoint_interval: Override checkpoint interval (float)

    Returns:
        List of results from all workers (same order as iterable)
    """
    return await _client.map(func, iterable, **kwargs)


async def CROWDio_run(func: Callable, *args, **kwargs) -> Any:
    """
    CROWDio_run - Send a task to a single worker and receive result

    Sends a single task to an available worker and waits for the result.
    If function is decorated with @CROWDio_task, checkpoint metadata is
    automatically used for checkpoint-aware execution.

    Args:
        func: Function to execute (optionally decorated with @CROWDio_task)
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function

    Returns:
        Result from the worker
    """
    return await _client.run(func, *args, **kwargs)


async def CROWDio_submit(func: Callable, iterable: List[Any], **kwargs) -> str:
    """
    CROWDio_submit - Non-blocking submission of a job to workers

    Asynchronous job submission. Sends tasks and returns immediately with a
    job handle for later result retrieval via CROWDio_get().

    Args:
        func: Function to execute (optionally decorated with @CROWDio_task)
        iterable: List of arguments for tasks
        **kwargs: Additional options

    Returns:
        job_id: Handle to retrieve results later with CROWDio_get()
    """
    return await _client.submit(func, iterable, **kwargs)


async def CROWDio_get(job_id: str, timeout: Optional[float] = None) -> Any:
    """
    CROWDio_get - Wait for results from a non-blocking submission

    Waits for results from a job submitted via CROWDio_submit().
    Blocks until results are available or timeout is reached.

    Args:
        job_id: Job identifier from CROWDio_submit()
        timeout: Maximum seconds to wait (None = wait indefinitely)

    Returns:
        List of results from the job

    Raises:
        TimeoutError: If timeout is exceeded before results arrive
    """
    return await _client.get_results(job_id, timeout)


async def CROWDio_pipeline(
    stages: List[Dict],
    dependency_map: Optional[Dict[str, List[str]]] = None,
    **kwargs,
) -> List[Any]:
    """
    CROWDio_pipeline - Execute a DAG of dependent computation stages

    Implements pipeline parallelism with automatic dependency management.
    Tasks are blocked until their upstream dependencies complete
    (tracked via a dependency counter).

    Args:
        stages: Ordered list of stage dicts, each with:
            - func: Callable (optionally decorated with @CROWDio_task)
            - args_list: List of arguments (one per task in this stage)
            - pass_upstream_results: bool (default False) – if True,
              upstream results are injected into downstream args
            - name: Optional human-readable stage label
        dependency_map: Optional explicit task→[deps] mapping for
            arbitrary DAG topologies. If omitted, stages form
            sequential barriers.
        **kwargs: Extra options (checkpoint, checkpoint_interval, …)

    Returns:
        List of results from the final stage.

    Example::

        results = await CROWDio_pipeline([
            {"func": preprocess,  "args_list": raw_data},
            {"func": compute,     "args_list": [None]*len(raw_data),
             "pass_upstream_results": True},
            {"func": aggregate,   "args_list": [None],
             "pass_upstream_results": True},
        ])
    """
    return await _client.pipeline(stages, dependency_map=dependency_map, **kwargs)


# ============================================================================
# Preferred lowercase API aliases
# ============================================================================

crowdio_connect = CROWDio_connect
crowdio_disconnect = CROWDio_disconnect
crowdio_map = CROWDio_map
crowdio_run = CROWDio_run
crowdio_get = CROWDio_get
crowdio_submit = CROWDio_submit
crowdio_pipeline = CROWDio_pipeline
task = CROWDio_task

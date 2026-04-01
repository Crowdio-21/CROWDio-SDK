"""
CROWDio - Distributed Python Computing SDK

Provides declarative checkpointing support for distributed task execution.

Usage:
    import developer_sdk as CROWDio

    @CROWDio.task(
        checkpoint=True,
        checkpoint_interval=5,
        checkpoint_state=["i", "result", "progress_percent"]
    )
    def my_task(state, data):
        for i in range(state.get("i", 0), len(data)):
            state["i"] = i
            state["result"] = process(data[i])
            state["progress_percent"] = (i + 1) / len(data) * 100
        return state["result"]

    async def main():
        await CROWDio_connect("localhost", 9000)
        results = await CROWDio_map(my_task, data_list)
        await CROWDio_disconnect()
"""

from .api import (
    # Preferred lowercase API
    crowdio_connect,
    crowdio_disconnect,
    crowdio_map,
    crowdio_run,
    crowdio_get,
    crowdio_submit,
    crowdio_pipeline,
    task,
    # Connection API
    CROWDio_connect,
    CROWDio_disconnect,
    # Execution API
    CROWDio_map,
    CROWDio_run,
    CROWDio_get,
    CROWDio_submit,
    CROWDio_pipeline,
    # Declarative task API
    CROWDio_task,
    CROWDioTaskMetadata,
    CROWDioTaskConfig,
    CROWDio_get_task_metadata,
    CROWDio_get_task_config,
    CROWDio_is_checkpoint_task,
    CROWDio_create_state_dict,
    CROWDioConstant,
    CROWDio,
)

__version__ = "0.3.0"
__all__ = [
    # Preferred lowercase API
    "crowdio_connect",
    "crowdio_disconnect",
    "crowdio_map",
    "crowdio_run",
    "crowdio_get",
    "crowdio_submit",
    "crowdio_pipeline",
    "task",
    # Connection API
    "CROWDio_connect",
    "CROWDio_disconnect",
    # Execution API
    "CROWDio_map",
    "CROWDio_run",
    "CROWDio_get",
    "CROWDio_submit",
    "CROWDio_pipeline",
    # Declarative task API
    "CROWDio_task",
    "CROWDioTaskMetadata",
    "CROWDioTaskConfig",
    "CROWDio_get_task_metadata",
    "CROWDio_get_task_config",
    "CROWDio_is_checkpoint_task",
    "CROWDio_create_state_dict",
    "CROWDioConstant",
    "CROWDio",
]

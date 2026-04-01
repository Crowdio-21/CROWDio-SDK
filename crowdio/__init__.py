"""CROWDio public SDK package."""

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
    # Declarative task namespace
    CROWDio,
)
from .decorators import (
    # Declarative task API
    CROWDio_task,
    CROWDioTaskMetadata,
    CROWDioTaskConfig,
    CROWDio_get_task_metadata,
    CROWDio_get_task_config,
    CROWDio_is_checkpoint_task,
    CROWDio_create_state_dict,
)
from .constants import CROWDioConstant

__version__ = "0.3.1"
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

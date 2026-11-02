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
    # Declarative model jobs
    crowdio_submit_job,
    crowdio_list_models,
    crowdio_describe_model,
    crowdio_capabilities,
    CROWDio_submit_job,
    CROWDio_list_models,
    CROWDio_describe_model,
    CROWDio_capabilities,
    submit_job,
    list_models,
    describe_model,
)
from .declarative import (
    Capabilities,
    DeclarativeError,
    InvalidRequest,
    JobCancelled,
    JobFailed,
    JobHandle,
    JobRejected,
    ModelDescription,
    ModelNotFound,
    PlanEpoch,
    RequestResult,
    StageTiming,
    Timings,
    UnsupportedByForeman,
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
from .shard_store import ShardKey, ShardManifest, ShardStore
from .session import SessionManager, SessionState, SessionTransitionError
from .transfer import TransferClient
from .telemetry import TelemetryEvent, TelemetryEstimator
from .cost_profile import CostProfile, Distribution
from .structured_logging import StructuredLogger

__version__ = "0.4.0"
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
    "crowdio_submit_job",
    "crowdio_list_models",
    "crowdio_describe_model",
    "crowdio_capabilities",
    "CROWDio_submit_job",
    "CROWDio_list_models",
    "CROWDio_describe_model",
    "CROWDio_capabilities",
    "submit_job",
    "list_models",
    "describe_model",
    "Capabilities",
    "DeclarativeError",
    "InvalidRequest",
    "JobCancelled",
    "JobFailed",
    "JobHandle",
    "JobRejected",
    "ModelDescription",
    "ModelNotFound",
    "RequestResult",
    "PlanEpoch",
    "StageTiming",
    "Timings",
    "UnsupportedByForeman",
    "ShardKey",
    "ShardManifest",
    "ShardStore",
    "SessionManager",
    "SessionState",
    "SessionTransitionError",
    "TransferClient",
    "TelemetryEvent",
    "TelemetryEstimator",
    "CostProfile",
    "Distribution",
    "StructuredLogger",
]

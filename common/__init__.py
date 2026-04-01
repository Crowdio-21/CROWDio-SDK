"""
Common utilities for CrowdCompute

Exports:
- Protocol messages and types
- Serialization utilities
- Worker type detection and code instrumentation for mobile
"""

# Protocol exports
from .protocol import (
    Message,
    MessageType,
    CheckpointType,
    RecoveryStatus,
    create_worker_ready_message,
    create_submit_job_message,
    create_assign_task_message,
    create_assign_task_message_with_metadata,
    create_resume_task_message,
    create_resume_task_message_with_metadata,
    create_task_result_message,
)

# Code instrumentation exports for mobile workers
from .code_instrumenter import (
    WorkerType,
    detect_worker_capabilities,
    instrument_for_mobile,
    prepare_code_for_mobile_resume,
    generate_mobile_checkpoint_wrapper,
    CheckpointInstrumenter,
    ResumeInstrumenter,
)

# Serialization exports
from .serializer import (
    serialize_function,
    deserialize_function_for_PC,
    get_runtime_info,
)

__all__ = [
    # Protocol
    "Message",
    "MessageType",
    "CheckpointType",
    "RecoveryStatus",
    "create_worker_ready_message",
    "create_submit_job_message",
    "create_assign_task_message",
    "create_assign_task_message_with_metadata",
    "create_resume_task_message",
    "create_resume_task_message_with_metadata",
    "create_task_result_message",
    # Code instrumentation
    "WorkerType",
    "detect_worker_capabilities",
    "instrument_for_mobile",
    "prepare_code_for_mobile_resume",
    "generate_mobile_checkpoint_wrapper",
    "CheckpointInstrumenter",
    "ResumeInstrumenter",
    # Serialization
    "serialize_function",
    "deserialize_function_for_PC",
    "get_runtime_info",
]

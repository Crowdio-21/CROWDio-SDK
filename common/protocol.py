"""
Network protocol definitions for CrowdCompute

Extended with declarative checkpointing support:
- Task metadata in job submissions
- Checkpoint type (base/delta) in checkpoint messages
- Recovery status in resume messages
"""

import json
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class MessageType(Enum):
    """Message types for WebSocket communication"""
    # Client -> Foreman
    SUBMIT_JOB = "submit_job"
    SUBMIT_PIPELINE_JOB = "submit_pipeline_job"
    GET_RESULTS = "get_results"
    DISCONNECT = "disconnect"
    
    # Foreman -> Worker
    ASSIGN_TASK = "assign_task"
    PING = "ping"
    RESUME_TASK = "resume_task"
    
    # Worker -> Foreman
    TASK_RESULT = "task_result"
    TASK_ERROR = "task_error"
    WORKER_READY = "worker_ready"
    WORKER_HEARTBEAT = "worker_heartbeat"
    PONG = "pong"
    TASK_CHECKPOINT = "task_checkpoint"
    TASK_PROGRESS = "task_progress" 
    
    # Foreman -> Client
    JOB_RESULTS = "job_results"
    JOB_ERROR = "job_error"
    JOB_ACCEPTED = "job_accepted"
    JOB_PROGRESS = "job_progress"  
    
    # Checkpoint Acks
    CHECKPOINT_ACK = "checkpoint_ack"


class CheckpointType(Enum):
    """Types of checkpoints for incremental state saving"""
    BASE = "base"           # Full state checkpoint
    DELTA = "delta"         # Incremental changes only
    COMPACTED = "compacted" # Merged base + deltas


class RecoveryStatus(Enum):
    """Status of task recovery from checkpoint"""
    FRESH = "fresh"         # New task, no checkpoint
    RESUMED = "resumed"     # Resumed from checkpoint
    FAILED = "failed"       # Recovery failed, restarting fresh


class Message:
    """Base message class"""
    
    def __init__(self, msg_type: MessageType, data: Dict[str, Any], job_id: Optional[str] = None):
        self.type = msg_type
        self.data = data
        self.job_id = job_id
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "data": self.data,
            "job_id": self.job_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        return cls(
            msg_type=MessageType(data["type"]),
            data=data["data"],
            job_id=data.get("job_id")
        )
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Message':
        return cls.from_dict(json.loads(json_str))


# Message factory functions
def create_submit_job_message(func_code: str, args_list: List[Any], job_id: str) -> Message:
    """Create a job submission message"""
    return Message(
        msg_type=MessageType.SUBMIT_JOB,
        data={
            "func_code": func_code,
            "args_list": args_list,
            "total_tasks": len(args_list)
        },
        job_id=job_id
    )


def create_submit_pipeline_job_message(
    job_id: str,
    stages: List[Dict[str, Any]],
    dependency_map: Optional[Dict[str, List[str]]] = None,
    task_metadata: Optional[Dict[str, Any]] = None,
) -> Message:
    """
    Create a pipeline job submission message.

    A pipeline job consists of multiple stages where tasks in later stages
    depend on the completion of tasks in earlier stages.  The system uses
    a dependency counter per task: when all upstream tasks complete, the
    counter reaches zero and the task becomes eligible for scheduling.

    Args:
        job_id: Unique job identifier
        stages: List of stage dicts, each containing:
            - func_code (str): Serialised function code for this stage
            - args_list (list): Arguments for each task in the stage
            - pass_upstream_results (bool): Inject upstream results into args
            - task_metadata (dict, optional): Checkpoint config for stage
            - name (str, optional): Human-readable stage label
        dependency_map: Optional explicit task→[dependency] mapping for
                       arbitrary DAG topologies.  If omitted, sequential
                       stage-to-stage barriers are used.
        task_metadata: Optional global checkpoint metadata for all stages.

    Returns:
        Message ready to send over WebSocket
    """
    total_tasks = sum(len(s["args_list"]) for s in stages)
    return Message(
        msg_type=MessageType.SUBMIT_PIPELINE_JOB,
        data={
            "stages": stages,
            "total_tasks": total_tasks,
            "total_stages": len(stages),
            "dependency_map": dependency_map,
            "task_metadata": task_metadata,
        },
        job_id=job_id,
    )


def create_job_accepted_message(job_id: str) -> Message:
    """Create a job accepted message"""
    return Message(
        MessageType.JOB_ACCEPTED,
        {"job_id": job_id},
        job_id
    )

def create_assign_task_message(func_code: str, task_args: List[Any], task_id: str, job_id: str) -> Message:
    """Create a task assignment message"""
    return Message(
        msg_type=MessageType.ASSIGN_TASK,
        data={
            "func_code": func_code,
            "task_args": task_args,
            "task_id": task_id
        },
        job_id=job_id
    )


def create_task_result_message(result: Any, task_id: str, job_id: str) -> Message:
    """Create a task result message"""
    return Message(
        msg_type=MessageType.TASK_RESULT,
        data={
            "result": result,
            "task_id": task_id
        },
        job_id=job_id
    )


def create_task_error_message(error: str, task_id: str, job_id: str) -> Message:
    """Create a task error message"""
    return Message(
        msg_type=MessageType.TASK_ERROR,
        data={
            "error": error,
            "task_id": task_id
        },
        job_id=job_id
    )


def create_job_results_message(results: List[Any], job_id: str) -> Message:
    """Create a job results message"""
    return Message(
        msg_type=MessageType.JOB_RESULTS,
        data={"results": results},
        job_id=job_id
    )


def create_worker_ready_message(worker_id: str) -> Message:
    """Create a worker ready message"""
    return Message(
        msg_type=MessageType.WORKER_READY,
        data={"worker_id": worker_id}
    )


def create_ping_message() -> Message:
    """Create a ping message"""
    return Message(
        msg_type=MessageType.PING,
        data={}
    )


def create_pong_message() -> Message:
    """Create a pong message"""
    return Message(
        msg_type=MessageType.PONG,
        data={}
    )


def create_task_checkpoint_message(
    task_id: str, 
    job_id: str, 
    is_base: bool, 
    delta_data_hex: str, 
    progress_percent: float, 
    checkpoint_id: int,
    compression_type: str = "gzip",
    checkpoint_type: str = None,
    checkpoint_state_vars: List[str] = None,
    state_size_bytes: int = None
) -> Message:
    """Create a task checkpoint message from worker to foreman
    
    Args:
        task_id: Task identifier
        job_id: Job identifier
        is_base: True if this is the base checkpoint, False if delta
        delta_data_hex: Hex-encoded checkpoint data (compressed)
        progress_percent: Task progress as percentage (0-100)
        checkpoint_id: Sequential checkpoint number
        compression_type: Type of compression applied (gzip, zstd, etc)
        checkpoint_type: Type of checkpoint (base, delta, compacted) - from declarative API
        checkpoint_state_vars: List of state variables being checkpointed
        state_size_bytes: Uncompressed size of checkpoint state
    """
    data = {
        "task_id": task_id,
        "is_base": is_base,
        "delta_data_hex": delta_data_hex,
        "progress_percent": progress_percent,
        "checkpoint_id": checkpoint_id,
        "compression_type": compression_type
    }
    
    # Add optional declarative checkpointing fields
    if checkpoint_type is not None:
        data["checkpoint_type"] = checkpoint_type
    if checkpoint_state_vars is not None:
        data["checkpoint_state_vars"] = checkpoint_state_vars
    if state_size_bytes is not None:
        data["state_size_bytes"] = state_size_bytes
    
    return Message(
        msg_type=MessageType.TASK_CHECKPOINT,
        data=data,
        job_id=job_id
    )


def create_checkpoint_ack_message(task_id: str, job_id: str, checkpoint_id: int) -> Message:
    """Create a checkpoint acknowledgment message from foreman to worker"""
    return Message(
        msg_type=MessageType.CHECKPOINT_ACK,
        data={
            "task_id": task_id,
            "checkpoint_id": checkpoint_id
        },
        job_id=job_id
    )


def create_resume_task_message(
    task_id: str, 
    job_id: str, 
    func_code: str,
    checkpoint_state: dict,
    task_args: List[Any] = None,
    task_kwargs: Dict[str, Any] = None,
    progress_percent: float = 0.0,
    checkpoint_count: int = 0
) -> Message:
    """Create a task resumption message from foreman to worker
    
    Sends checkpoint state to worker so it can continue from where 
    the previous worker left off.
    
    Args:
        task_id: Task identifier
        job_id: Job identifier
        func_code: Function code to execute
        checkpoint_state: Dictionary containing the saved state from last checkpoint
        task_args: Original task arguments (required for function execution)
        task_kwargs: Original task keyword arguments
        progress_percent: Progress percentage at last checkpoint
        checkpoint_count: Total checkpoints available for this task
    """
    return Message(
        msg_type=MessageType.RESUME_TASK,
        data={
            "task_id": task_id,
            "func_code": func_code,
            "checkpoint_state": checkpoint_state,
            "task_args": task_args or [],
            "task_kwargs": task_kwargs or {},
            "progress_percent": progress_percent,
            "checkpoint_count": checkpoint_count
        },
        job_id=job_id
    )


def create_assign_task_message_with_metadata(
    func_code: str, 
    task_args: List[Any], 
    task_id: str, 
    job_id: str,
    task_metadata: Optional[Dict[str, Any]] = None
) -> Message:
    """
    Create a task assignment message with checkpoint metadata
    
    Extended version that includes task configuration for checkpoint-aware execution.
    
    Args:
        func_code: Serialized function code
        task_args: Arguments for this specific task
        task_id: Task identifier
        job_id: Job identifier
        task_metadata: Optional checkpoint configuration from @task decorator
        
    Returns:
        Message object for task assignment
    """
    data = {
        "func_code": func_code,
        "task_args": task_args,
        "task_id": task_id
    }
    
    if task_metadata:
        data["task_metadata"] = task_metadata
    
    return Message(
        msg_type=MessageType.ASSIGN_TASK,
        data=data,
        job_id=job_id
    )


def create_resume_task_message_with_metadata(
    task_id: str, 
    job_id: str, 
    func_code: str,
    checkpoint_state: dict,
    task_args: List[Any] = None,
    task_kwargs: Dict[str, Any] = None,
    task_metadata: Optional[Dict[str, Any]] = None,
    progress_percent: float = 0.0,
    checkpoint_count: int = 0,
    recovery_status: str = "resumed",
    base_checkpoint_count: int = 0,
    delta_checkpoint_count: int = 0
) -> Message:
    """
    Create an extended task resumption message with full metadata
    
    Includes recovery status and detailed checkpoint information for
    transparent task continuation on a new worker.
    
    Args:
        task_id: Task identifier
        job_id: Job identifier
        func_code: Function code to execute
        checkpoint_state: Reconstructed state from base + deltas
        task_args: Original task arguments
        task_kwargs: Original task keyword arguments
        task_metadata: Checkpoint configuration from @task decorator
        progress_percent: Progress at last checkpoint
        checkpoint_count: Total checkpoints taken
        recovery_status: 'resumed', 'fresh', or 'failed'
        base_checkpoint_count: Number of base checkpoints
        delta_checkpoint_count: Number of delta checkpoints
        
    Returns:
        Message object for task resumption
    """
    return Message(
        msg_type=MessageType.RESUME_TASK,
        data={
            "task_id": task_id,
            "func_code": func_code,
            "checkpoint_state": checkpoint_state,
            "task_args": task_args or [],
            "task_kwargs": task_kwargs or {},
            "task_metadata": task_metadata or {},
            "progress_percent": progress_percent,
            "checkpoint_count": checkpoint_count,
            "recovery_status": recovery_status,
            "base_checkpoint_count": base_checkpoint_count,
            "delta_checkpoint_count": delta_checkpoint_count
        },
        job_id=job_id
    )


def create_task_checkpoint_message_extended(
    task_id: str, 
    job_id: str,
    checkpoint_type: str,  # "base", "delta", or "compacted"
    checkpoint_data_hex: str, 
    progress_percent: float, 
    checkpoint_id: int,
    checkpoint_state_vars: List[str] = None,
    compression_type: str = "gzip",
    state_size_bytes: int = 0
) -> Message:
    """
    Create an extended checkpoint message with metadata
    
    Includes checkpoint type and tracked state variables for better
    dashboard visibility and recovery management.
    
    Args:
        task_id: Task identifier
        job_id: Job identifier
        checkpoint_type: "base", "delta", or "compacted"
        checkpoint_data_hex: Hex-encoded checkpoint data
        progress_percent: Task progress (0-100)
        checkpoint_id: Sequential checkpoint number
        checkpoint_state_vars: List of state variables captured
        compression_type: Compression algorithm used
        state_size_bytes: Uncompressed state size for metrics
        
    Returns:
        Message object for checkpoint
    """
    return Message(
        msg_type=MessageType.TASK_CHECKPOINT,
        data={
            "task_id": task_id,
            "checkpoint_type": checkpoint_type,
            "is_base": checkpoint_type == "base",
            "delta_data_hex": checkpoint_data_hex,
            "progress_percent": progress_percent,
            "checkpoint_id": checkpoint_id,
            "checkpoint_state_vars": checkpoint_state_vars or [],
            "compression_type": compression_type,
            "state_size_bytes": state_size_bytes
        },
        job_id=job_id
    )


def create_task_progress_message(
    task_id: str,
    job_id: str,
    progress_percent: float,
    status: str = "running",
    eta_seconds: Optional[float] = None
) -> Message:
    """
    Create a lightweight progress update message (no checkpoint data)
    
    Used for dashboard updates between checkpoint intervals.
    
    Args:
        task_id: Task identifier
        job_id: Job identifier
        progress_percent: Current progress (0-100)
        status: Task status string
        eta_seconds: Estimated time to completion
        
    Returns:
        Message object for progress update
    """
    return Message(
        msg_type=MessageType.TASK_PROGRESS,
        data={
            "task_id": task_id,
            "progress_percent": progress_percent,
            "status": status,
            "eta_seconds": eta_seconds
        },
        job_id=job_id
    )


def create_job_progress_message(
    job_id: str,
    completed_tasks: int,
    total_tasks: int,
    tasks_with_checkpoints: int = 0,
    average_progress: float = 0.0
) -> Message:
    """
    Create a job-level progress message for client updates
    
    Args:
        job_id: Job identifier
        completed_tasks: Number of completed tasks
        total_tasks: Total tasks in job
        tasks_with_checkpoints: Tasks with at least one checkpoint
        average_progress: Average progress across running tasks
        
    Returns:
        Message object for job progress
    """
    return Message(
        msg_type=MessageType.JOB_PROGRESS,
        data={
            "completed_tasks": completed_tasks,
            "total_tasks": total_tasks,
            "tasks_with_checkpoints": tasks_with_checkpoints,
            "average_progress": average_progress,
            "percent_complete": (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
        },
        job_id=job_id
    )

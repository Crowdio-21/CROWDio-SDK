import asyncio
import json
import uuid
import websockets
from typing import Any, Callable, List, Optional, Dict
from common.protocol import (
    Message,
    MessageType,
    create_submit_job_message,
    create_submit_pipeline_job_message,
    create_ping_message,
    create_pong_message,
)
from common.serializer import serialize_function
from common.code_instrumenter import (
    instrument_for_task_control,
    generate_runtime_wrappers,
)
from .decorators import CROWDio_get_task_metadata, CROWDioTaskMetadata


class CrowdComputeClient:
    """Main client for interacting with CrowdCompute foreman"""

    def __init__(self):
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.foreman_host: Optional[str] = None
        self.foreman_port: Optional[int] = None
        self.connected = False
        self.pending_jobs: Dict[str, asyncio.Future] = {}
        self._listen_task: Optional[asyncio.Task] = None
        self._submitted_jobs: Dict[str, Dict[str, Any]] = {}

    async def connect(self, host: str, port: int = 9000):
        """Connect to foreman server"""
        try:
            self.foreman_host = host
            self.foreman_port = port
            uri = f"ws://{host}:{port}"

            print(f"Connecting to foreman at {uri}...")
            # Increase max_size and disable built-in ping to match foreman server settings.
            # Large job results (e.g., base64 encoded images) routinely exceed the
            # 1 MiB default limit, which previously caused the foreman to close
            # the connection while streaming results back to the client.
            self.websocket = await websockets.connect(
                uri,
                max_size=None,  # Allow arbitrarily large result payloads
                ping_interval=None,
                ping_timeout=None,
                close_timeout=30,
            )
            self.connected = True

            # Start listening for responses
            self._listen_task = asyncio.create_task(self._listen_for_messages())

            print(f"Connected to foreman at {uri}")

        except Exception as e:
            print(f"Failed to connect to foreman: {e}")
            raise

    async def disconnect(self):
        """Disconnect from foreman server"""
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
            self.connected = False

        if self._listen_task:
            self._listen_task.cancel()
            self._listen_task = None

        print("Disconnected from foreman")

    async def _listen_for_messages(self):
        """Listen for messages from foreman"""
        try:
            async for message in self.websocket:
                await self._handle_message(message)
        except websockets.exceptions.ConnectionClosed:
            print("Connection to foreman closed")
            self.connected = False
        except Exception as e:
            print(f"Error in message listener: {e}")
            self.connected = False

    async def _handle_message(self, message_str: str):
        """Handle incoming message from foreman"""
        try:
            message = Message.from_json(message_str)

            if message.type == MessageType.JOB_RESULTS:
                job_id = message.job_id
                if job_id in self.pending_jobs:
                    future = self.pending_jobs.pop(job_id)
                    future.set_result(message.data["results"])

            elif message.type == MessageType.JOB_ERROR:
                job_id = message.job_id
                if job_id in self.pending_jobs:
                    future = self.pending_jobs.pop(job_id)
                    future.set_exception(Exception(message.data["error"]))

            elif message.type == MessageType.PING:
                # Respond to ping
                pong = create_pong_message()
                await self.websocket.send(pong.to_json())

        except Exception as e:
            print(f"Error handling message: {e}")

    async def map(self, func: Callable, iterable: List[Any], **kwargs) -> List[Any]:
        """
        Map function over iterable using distributed workers

        Extracts checkpoint metadata from @task decorated functions and
        sends it with the job submission for checkpoint-aware execution.

        Args:
            func: Function to execute (optionally decorated with @task)
            iterable: List of arguments to map over
            **kwargs: Additional options:
                - checkpoint: Override checkpoint setting
                - checkpoint_interval: Override checkpoint interval

        Returns:
            List of results from workers
        """
        if not self.connected:
            raise RuntimeError("Not connected to foreman. Call connect() first.")

        # Generate job ID
        job_id = str(uuid.uuid4())

        # Create future for this job
        future = asyncio.Future()
        self.pending_jobs[job_id] = future

        try:
            # Extract metadata from decorated function (if present)
            metadata = CROWDio_get_task_metadata(func)

            # Apply any overrides from kwargs
            if metadata:
                task_metadata = CROWDioTaskMetadata(
                    checkpoint_enabled=kwargs.get(
                        "checkpoint", metadata.checkpoint_enabled
                    ),
                    checkpoint_interval=kwargs.get(
                        "checkpoint_interval", metadata.checkpoint_interval
                    ),
                    checkpoint_state=kwargs.get(
                        "checkpoint_state", metadata.checkpoint_state
                    ),
                    parallel=kwargs.get("parallel", metadata.parallel),
                    retry_on_failure=kwargs.get(
                        "retry_on_failure", metadata.retry_on_failure
                    ),
                    max_retries=kwargs.get("max_retries", metadata.max_retries),
                    timeout=kwargs.get("timeout", metadata.timeout),
                    _func_name=metadata._func_name,
                )
            else:
                # Create default metadata if not decorated
                task_metadata = CROWDioTaskMetadata(
                    checkpoint_enabled=kwargs.get("checkpoint", False),
                    checkpoint_interval=kwargs.get("checkpoint_interval", 10.0),
                    checkpoint_state=kwargs.get("checkpoint_state", []),
                    parallel=kwargs.get("parallel", True),
                    retry_on_failure=kwargs.get("retry_on_failure", True),
                    max_retries=kwargs.get("max_retries", 3),
                    timeout=kwargs.get("timeout", None),
                    _func_name=func.__name__,
                )

            # Serialize function (use original if decorated)
            if hasattr(func, "__crowdio_original__"):
                func_code = serialize_function(func.__crowdio_original__)
            else:
                func_code = serialize_function(func)

            # Pre-instrument with task control (pause/kill) at SDK level
            # This is a static transformation that reduces foreman load
            checkpoint_state_vars = task_metadata.checkpoint_state or []
            func_code, num_funcs, num_ctrl_loops = instrument_for_task_control(
                func_code, checkpoint_state_vars=checkpoint_state_vars
            )
            runtime_wrapper = generate_runtime_wrappers(
                include_checkpoint=True,
                include_task_control=True,
            )
            func_code = runtime_wrapper + "\n" + func_code
            if num_funcs > 0:
                print(
                    f"[SDK] Task control: instrumented {num_funcs} functions, "
                    f"{num_ctrl_loops} loops"
                )
                
            # Create submission message with metadata
            message = self._create_submit_job_message_with_metadata(
                func_code, iterable, job_id, task_metadata
            )

            # Store job metadata for tracking
            self._submitted_jobs[job_id] = {
                "metadata": task_metadata,
                "func_name": task_metadata._func_name,
                "total_tasks": len(iterable),
                "checkpoint_enabled": task_metadata.checkpoint_enabled,
            }

            # Send to foreman
            await self.websocket.send(message.to_json())

            if task_metadata.checkpoint_enabled:
                print(
                    f"[SDK] Job {job_id[:8]}... submitted with checkpointing enabled "
                    f"(interval: {task_metadata.checkpoint_interval}s, "
                    f"state vars: {task_metadata.checkpoint_state or 'all'})"
                )
            else:
                print(f"[SDK] Job {job_id[:8]}... submitted (checkpointing disabled)")

            # Wait for results
            results = await future
            return results

        except Exception as e:
            # Clean up on error
            if job_id in self.pending_jobs:
                del self.pending_jobs[job_id]
            if job_id in self._submitted_jobs:
                del self._submitted_jobs[job_id]
            raise

    def _create_submit_job_message_with_metadata(
        self,
        func_code: str,
        args_list: List[Any],
        job_id: str,
        metadata: CROWDioTaskMetadata,
    ) -> Message:
        """
        Create a job submission message with checkpoint metadata

        Args:
            func_code: Serialized function code
            args_list: List of task arguments
            job_id: Job identifier
            metadata: Task checkpoint metadata

        Returns:
            Message object ready to send
        """
        return Message(
            msg_type=MessageType.SUBMIT_JOB,
            data={
                "func_code": func_code,
                "args_list": args_list,
                "total_tasks": len(args_list),
                "task_metadata": metadata.to_dict(),
            },
            job_id=job_id,
        )

    async def submit(self, func: Callable, iterable: List[Any], **kwargs) -> str:
        """
        Submit a job asynchronously without waiting for results

        Args:
            func: Function to execute
            iterable: List of arguments for tasks
            **kwargs: Additional options

        Returns:
            job_id: Identifier to retrieve results later
        """
        if not self.connected:
            raise RuntimeError("Not connected to foreman. Call connect() first.")

        job_id = str(uuid.uuid4())

        # Extract metadata
        metadata = CROWDio_get_task_metadata(func)
        if metadata:
            task_metadata = CROWDioTaskMetadata(
                checkpoint_enabled=kwargs.get(
                    "checkpoint", metadata.checkpoint_enabled
                ),
                checkpoint_interval=kwargs.get(
                    "checkpoint_interval", metadata.checkpoint_interval
                ),
                checkpoint_state=kwargs.get(
                    "checkpoint_state", metadata.checkpoint_state
                ),
                parallel=metadata.parallel,
                retry_on_failure=metadata.retry_on_failure,
                max_retries=metadata.max_retries,
                timeout=metadata.timeout,
                _func_name=metadata._func_name,
            )
        else:
            task_metadata = CROWDioTaskMetadata(
                checkpoint_enabled=kwargs.get("checkpoint", False),
                checkpoint_interval=kwargs.get("checkpoint_interval", 10.0),
                _func_name=func.__name__,
            )

        # Serialize function
        if hasattr(func, "__crowdio_original__"):
            func_code = serialize_function(func.__crowdio_original__)
        else:
            func_code = serialize_function(func)

        # Pre-instrument with task control (pause/kill) at SDK level
        checkpoint_state_vars = task_metadata.checkpoint_state or []
        func_code, num_funcs, num_ctrl_loops = instrument_for_task_control(
            func_code, checkpoint_state_vars=checkpoint_state_vars
        )
        runtime_wrapper = generate_runtime_wrappers(
            include_checkpoint=True,
            include_task_control=True,
        )
        func_code = runtime_wrapper + "\n" + func_code
        if num_funcs > 0:
            print(
                f"[SDK] Task control: instrumented {num_funcs} functions, "
                f"{num_ctrl_loops} loops"
            )

        # Create future for this job
        future = asyncio.Future()
        self.pending_jobs[job_id] = future

        # Store job metadata
        self._submitted_jobs[job_id] = {
            "metadata": task_metadata,
            "func_name": task_metadata._func_name,
            "total_tasks": len(iterable),
            "checkpoint_enabled": task_metadata.checkpoint_enabled,
            "future": future,
        }

        # Create and send message
        message = self._create_submit_job_message_with_metadata(
            func_code, iterable, job_id, task_metadata
        )
        await self.websocket.send(message.to_json())

        print(f"[SDK] Job {job_id[:8]}... submitted asynchronously")
        return job_id

    async def get_results(self, job_id: str, timeout: Optional[float] = None) -> Any:
        """
        Get results for a submitted job

        Args:
            job_id: Job identifier from submit()
            timeout: Maximum seconds to wait (None = wait forever)

        Returns:
            List of results

        Raises:
            TimeoutError: If timeout exceeded
            KeyError: If job_id not found
        """
        if job_id not in self.pending_jobs:
            raise KeyError(f"Job {job_id} not found")

        future = self.pending_jobs[job_id]

        if timeout:
            try:
                results = await asyncio.wait_for(future, timeout=timeout)
            except asyncio.TimeoutError:
                raise TimeoutError(
                    f"Job {job_id} did not complete within {timeout} seconds"
                )
        else:
            results = await future

        # Clean up
        del self.pending_jobs[job_id]
        if job_id in self._submitted_jobs:
            del self._submitted_jobs[job_id]

        return results

    async def pipeline(
        self,
        stages: List[Dict[str, Any]],
        dependency_map: Optional[Dict[str, List[str]]] = None,
        **kwargs,
    ) -> List[Any]:
        """
        Execute a pipeline of dependent stages on distributed workers.

        Each stage defines a function and its input arguments.  Tasks in
        later stages are automatically blocked until their upstream
        dependencies complete.  The system tracks a *dependency counter*
        per task and decrements it as upstream tasks finish; when the
        counter reaches zero the task becomes eligible for scheduling.

        This is the pipeline equivalent of ``map()`` – it submits the
        whole pipeline as a single job and waits for the final results.

        Args:
            stages: Ordered list of stage definitions.  Each stage is a dict:
                {
                    "func": <callable>,        # function for this stage
                    "args_list": [arg1, ...],  # one entry per task
                    "pass_upstream_results": False,  # inject upstream results?
                    "name": "stage_name",      # optional human label
                }
            dependency_map: Optional explicit mapping of
                task_id → [upstream_task_ids] for arbitrary DAGs.
                If omitted, sequential barriers between stages are used.
            **kwargs: Additional options (checkpoint, checkpoint_interval, etc.)

        Returns:
            List of results from the final stage.

        Example (sequential pipeline)::

            results = await client.pipeline([
                {"func": preprocess,  "args_list": raw_data},
                {"func": compute,     "args_list": [None]*len(raw_data),
                 "pass_upstream_results": True},
                {"func": aggregate,   "args_list": [None]*1,
                 "pass_upstream_results": True},
            ])
        """
        if not self.connected:
            raise RuntimeError("Not connected to foreman. Call connect() first.")

        job_id = str(uuid.uuid4())

        # Create future
        future = asyncio.Future()
        self.pending_jobs[job_id] = future

        try:
            # Build stage definitions for the protocol message
            stage_defs = []
            for stage_def in stages:
                func = stage_def["func"]
                args_list = stage_def["args_list"]

                # Serialise function
                if hasattr(func, "__crowdio_original__"):
                    func_code = serialize_function(func.__crowdio_original__)
                else:
                    func_code = serialize_function(func)

                # Extract per-stage task metadata
                metadata = CROWDio_get_task_metadata(func)

                # Pre-instrument with task control (pause/kill) at SDK level
                stage_ckpt_vars = []
                if metadata:
                    stage_ckpt_vars = metadata.checkpoint_state or []
                func_code, _, _ = instrument_for_task_control(
                    func_code, checkpoint_state_vars=stage_ckpt_vars
                )
                runtime_wrapper = generate_runtime_wrappers(
                    include_checkpoint=True,
                    include_task_control=True,
                )
                func_code = runtime_wrapper + "\n" + func_code
                task_meta_dict = None
                if metadata:
                    task_meta_dict = metadata.to_dict()

                stage_defs.append({
                    "func_code": func_code,
                    "args_list": args_list,
                    "pass_upstream_results": stage_def.get(
                        "pass_upstream_results", False
                    ),
                    "task_metadata": task_meta_dict,
                    "name": stage_def.get("name", func.__name__),
                })

            # Global task metadata from kwargs
            global_meta = None
            if kwargs.get("checkpoint"):
                global_meta = {
                    "checkpoint_enabled": True,
                    "checkpoint_interval": kwargs.get("checkpoint_interval", 10.0),
                    "checkpoint_state": kwargs.get("checkpoint_state", []),
                }

            # Build and send protocol message
            message = create_submit_pipeline_job_message(
                job_id=job_id,
                stages=stage_defs,
                dependency_map=dependency_map,
                task_metadata=global_meta,
            )

            self._submitted_jobs[job_id] = {
                "type": "pipeline",
                "stages": len(stage_defs),
                "total_tasks": sum(len(s["args_list"]) for s in stage_defs),
            }

            await self.websocket.send(message.to_json())

            total_tasks = sum(len(s["args_list"]) for s in stage_defs)
            print(
                f"[SDK] Pipeline job {job_id[:8]}... submitted "
                f"({len(stage_defs)} stages, {total_tasks} tasks)"
            )

            # Wait for results (same flow as map)
            results = await future
            return results

        except Exception as e:
            if job_id in self.pending_jobs:
                del self.pending_jobs[job_id]
            if job_id in self._submitted_jobs:
                del self._submitted_jobs[job_id]
            raise

    async def run(self, func: Callable, *args, **kwargs) -> Any:
        """Run a single function with arguments in one worker"""
        # Like AWS lambda
        pass

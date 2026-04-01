# Common Package

The `common` package contains shared modules used across all CrowdCompute components (developer SDK, foreman, and worker).

## Overview

This package provides the foundational communication protocol and serialization utilities that enable distributed computing across the CrowdCompute network.

## Modules

### `protocol.py`
Defines the message types and communication protocol for WebSocket-based communication between CrowdCompute components.

**Key Classes:**
- `MessageType` - Enumeration of all message types
- `Message` - Base message class for all communications

**Message Types:**
- `SUBMIT_JOB` - Client submits a job to foreman
- `JOB_ACCEPTED` - Foreman acknowledges job submission
- `JOB_RESULTS` - Foreman sends completed job results to client
- `JOB_ERROR` - Error occurred during job processing
- `WORKER_READY` - Worker announces availability
- `ASSIGN_TASK` - Foreman assigns task to worker
- `TASK_RESULT` - Worker sends task result to foreman
- `TASK_ERROR` - Worker reports task execution error
- `PING/PONG` - Heartbeat messages
- `DISCONNECT` - Graceful disconnection
- `WORKER_HEARTBEAT` - Worker status updates

**Helper Functions:**
- `create_job_submission_message()` - Create job submission messages
- `create_assign_task_message()` - Create task assignment messages
- `create_task_result_message()` - Create task result messages
- `create_worker_ready_message()` - Create worker ready messages

### `serializer.py`
Provides utilities for serializing and deserializing Python functions and data for network transmission.

**Key Functions:**
- `serialize_function(func)` - Serialize Python function to string *(Status: Stable)*
- `deserialize_function_for_PC(string_func)` - Deserialize string back to function *(Status: Stable)*
- `serialize_data(data)` - Serialize data to JSON string *(Status: Planning)*
- `deserialize_data(json_data)` - Deserialize JSON string back to data *(Status: Planning)*
- `hex_to_bytes(hex_data)` - Convert hex string to bytes  *(Status: Available, Unused)*
- `bytes_to_hex(data)` - Convert bytes to hex string  *(Status: Available, Unused)*

## Usage

```python
from common.protocol import Message, MessageType, create_job_submission_message
from common.serializer import serialize_function, deserialize_function_for_PC

# Serialize a function
def my_function(x):
    return x * 2

func_code = serialize_function(my_function)

# Create a message
message = create_job_submission_message(func_code, [1, 2, 3], "job-123")

# Send message as JSON
json_data = message.to_json()

# Deserialize function
restored_func = deserialize_function_for_PC(func_code)
```

## Dependencies

- `inspect` - For function serialization
- `json` - For data serialization

## Design Principles

1. **JSON Compatibility** - All messages are JSON-serializable for WebSocket transmission
2. **Function Serialization** - Uses inspect to convert function to string 
3. **Type Safety** - Uses Python type hints for better code clarity
4. **Extensibility** - Easy to add new message types and serialization methods

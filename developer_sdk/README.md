# Client Package

The `deveoper_sdk` package provides the CrowdCompute SDK for users to submit distributed computing jobs to the foreman.

## Overview

This package implements the client-side SDK that allows users to easily connect to a CrowdCompute foreman and submit Python functions for distributed execution across worker nodes.

## Modules

### `client.py`
Main client implementation providing the CrowdCompute SDK interface.

**Key Classes:**
- `CrowdComputeClient` - Main client class for connecting to foreman

**Key Methods:**
- `connect(host, port)` - Connect to foreman server
- `disconnect()` - Disconnect from foreman
- `map(func, iterable)` - Map function over iterable using distributed workers
- `run(func, *args, **kwargs)` - Run single function with arguments *(Status: Todo)*

### `api.py`

**Public API Functions:**
- `connect(host, port)` - Connect to foreman
- `disconnect()` - Disconnect from foreman
- `map(func, iterable)` - Distributed map operation
- `run(func, *args, **kwargs)` - Single function execution *(Status: Todo)*

### `__init__.py`
Package initialization that exports the main SDK functions.

## Usage

### Basic Usage

```python
import asyncio
from api import connect, map, disconnect

async def main():
    # Connect to foreman
    await connect("localhost", 9000)
    
    # Define a function to distribute
    def square(x):
        return x ** 2
    
    # Map function over data
    numbers = [1, 2, 3, 4, 5]
    results = await map(square, numbers)
    print(f"Results: {results}")  # [1, 4, 9, 16, 25]
    
    # Disconnect
    await disconnect()

# Run the example
asyncio.run(main())
```

### Advanced Usage *(Status: Todo)*

```python
import asyncio
from api import connect, map, run, disconnect

async def advanced_example():
    await connect("192.168.1.100", 9000)
    
    # Complex function
    def process_data(data):
        import time
        time.sleep(0.1)  # Simulate work
        return sum(data) * 2
    
    # Process multiple data arrays
    data_arrays = [
        [1, 2, 3, 4, 5],
        [10, 20, 30, 40, 50],
        [100, 200, 300, 400, 500]
    ]
    
    results = await map(process_data, data_arrays)
    print(f"Processed results: {results}")
    
    # Single function execution
    result = await run(lambda x: x * 10, 42)
    print(f"Single result: {result}")  # 420
    
    await disconnect()

asyncio.run(advanced_example())
```

## Features

### Distributed Map Operation
- Automatically splits data across available workers
- Handles function serialization and transmission
- Collects and returns results in order
- Supports any JSON-serializable data

### Error Handling
- Automatic reconnection on connection loss
- Graceful error reporting
- Timeout handling for long-running tasks

### Connection Management
- Automatic WebSocket connection management
- Heartbeat monitoring
- Clean disconnection

## Dependencies

- `websockets` - WebSocket client for communication
- `asyncio` - Asynchronous I/O support
- `uuid` - Unique job ID generation
- `common` package - Protocol and serialization utilities

## Architecture

```
User Code → Client SDK → WebSocket → Foreman → Workers
```

1. **Job Submission** - User calls `map()` or `run()`
2. **Function Serialization** - Function is serialized using inspect python package
3. **Message Creation** - Job submission message is created
4. **WebSocket Transmission** - Message sent to foreman
5. **Result Collection** - Results collected and returned to user

## Error Handling

The client handles various error scenarios:
- **Connection Errors** - Automatic retry with exponential backoff
- **Job Failures** - Exceptions are propagated to user code
- **Worker Failures** - Foreman automatically reassigns failed tasks
- **Network Issues** - Graceful degradation and error reporting

## Best Practices

1. **Always disconnect** - Use `await disconnect()` when done
2. **Handle exceptions** - Wrap calls in try-catch blocks
3. **Use async/await** - All operations are asynchronous
4. **Keep functions simple** - Complex functions may have serialization issues
5. **Monitor performance** - Use timing to optimize task distribution

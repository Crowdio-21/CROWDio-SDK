"""
Serialization utilities for CrowdCompute
"""

import inspect
import re
import sys
import types 
from typing import Any, Callable, List


def _env_info() -> str:
    """Return a concise runtime environment string for diagnostics"""
    return f"python={sys.version.split()[0]}"


def get_runtime_info() -> str:
    """Public helper to expose runtime info to other modules"""
    return _env_info()


def _strip_decorators(source: str) -> str:
    """
    Strip decorator lines from function source code.
    
    Removes @decorator(...) lines that precede function definitions,
    including multi-line decorators with parentheses.
    
    Args:
        source: Function source code string
        
    Returns:
        Source code with decorators removed
    """
    lines = source.split('\n')
    result_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        # Check if this is a decorator line
        if stripped.startswith('@'):
            # Skip decorator lines (including multi-line decorators)
            paren_count = stripped.count('(') - stripped.count(')')
            i += 1
            
            # Continue skipping if we're inside parentheses
            while paren_count > 0 and i < len(lines):
                paren_count += lines[i].count('(') - lines[i].count(')')
                i += 1
        else:
            result_lines.append(line)
            i += 1
    
    return '\n'.join(result_lines)


def serialize_function(func: Callable) -> str:
    """
    Serialize a Python function as a str
    
    Strips any decorators from the source code so the function
    can be executed on workers without needing decorator dependencies.
    
    Args:
        func: Function to serialize
        
    Returns:
        Function source code string (without decorators)
    """
    try:
        source = inspect.getsource(func)
        # Strip decorators so workers don't need decorator dependencies
        source = _strip_decorators(source)
        return source
    except Exception as e:
        raise ValueError(f"Failed to serialize function ({_env_info()}): {e}")

 
def deserialize_function_for_PC(func_code: str):
    """
    Turn function source code string into a callable function.
    
    Handles code that includes a task control wrapper (pause/resume/kill 
    functions prepended by the SDK) by using a shared namespace and 
    selecting the user's task function (skipping internal wrapper functions).
    """
    
    # Use a single namespace so wrapper globals (paused, killed, time)
    # are accessible from the function's __globals__
    namespace = {"__builtins__": __builtins__}
    exec(func_code, namespace)

    # Internal wrapper function names to skip
    _internal_names = {'pause', 'resume', 'kill'}

    # Find the user's function (skip wrapper functions)
    func = None
    for name, val in namespace.items():
        if isinstance(val, types.FunctionType) and name not in _internal_names:
            func = val
            break

    if func is None:
        raise ValueError("No function could be deserialized from code string")

    return func


def serialize_data(data: Any) -> bytes:
    """Serialize arbitrary data using _"""
    pass


def deserialize_data(data_bytes: bytes) -> Any:
    """Deserialize arbitrary data using _"""
    pass


def hex_to_bytes(hex_str: str) -> bytes:
    """Convert hex string back to bytes"""
    return bytes.fromhex(hex_str)


def bytes_to_hex(data_bytes: bytes) -> str:
    """Convert bytes to hex string"""
    return data_bytes.hex()

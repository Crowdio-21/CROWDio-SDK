"""
CROWDio path constants for mobile runtime path injection.

These string tokens are intentionally stable. Mobile runtimes can replace
or resolve them to real device-specific paths before task execution.
"""


class CROWDioConstant:
    """Stable symbolic path aliases used in distributed task configs."""

    FILE_DIR = "@CROWDIO:FILE_DIR"
    CACHE_DIR = "@CROWDIO:CACHE_DIR"
    OUTPUT_DIR = "@CROWDIO:OUTPUT_DIR"

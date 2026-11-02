"""Offline model export tool (``pip install crowdio-sdk[export]``).

The planner is importable without torch; the exporter and verifier import
torch / onnx / onnxruntime lazily.
"""

from .planner import (
    ModelSizes,
    PlannedShard,
    StorageBudgetExceeded,
    format_storage_report,
    parse_ranges,
    plan_shards,
    storage_report,
)

__all__ = [
    "ModelSizes",
    "PlannedShard",
    "StorageBudgetExceeded",
    "format_storage_report",
    "parse_ranges",
    "plan_shards",
    "storage_report",
    "export_model",
    "verify_export",
    "VerificationError",
]


def __getattr__(name):
    if name in ("export_model", "ExportResult"):
        from . import exporter

        return getattr(exporter, name)
    if name in ("verify_export", "VerificationError", "make_sample_inputs"):
        from . import verify

        return getattr(verify, name)
    raise AttributeError(name)

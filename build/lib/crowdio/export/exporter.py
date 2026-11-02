"""Offline shard export: plan, budget check, per-shard ONNX export, verify.

Output layout (one directory per model)::

    <out_root>/<model_id>/
        manifest.json              written only after verification passes
        storage_report.json
        verification_report.json
        shards/<shard_id>.onnx     (+ <shard_id>.onnx.data for external data)
        tokenizer/                 when a tokenizer is available
"""

from __future__ import annotations

import datetime as _dt
import json
import shutil
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from ..catalogue.manifest import (
    HIDDEN_IN,
    HIDDEN_OUT,
    IDS_IN,
    LOGITS_OUT,
    MASK_IN,
    MASK_OUT,
    ExternalDataFile,
    ModelManifest,
    ShardSpec,
    TensorContract,
    TensorSpec,
    sha256_file,
)
from .adapters import FLOPS_METHOD, EncoderAdapter, get_adapter, load_model
from .modules import EmbedShard, EncoderRangeShard, FullShard, HeadShard
from .planner import (
    DEFAULT_BUDGET_RATIO,
    DEFAULT_EXTERNAL_DATA_THRESHOLD,
    DEFAULT_MAX_SHARD_BYTES,
    DEFAULT_PHONE_SHARD_BUDGET_BYTES,
    MiB,
    PlannedShard,
    StorageBudgetExceeded,
    format_storage_report,
    plan_shards,
    storage_report,
)
from .verify import (
    DEFAULT_MAX_COMPOSITIONS,
    DEFAULT_TOLERANCE,
    VerificationError,
    make_sample_inputs,
    tokenize_samples,
    verify_export,
)

DEFAULT_OPSET = 18
DEFAULT_REFERENCE_SEQ_LEN = 128


@dataclass
class ExportResult:
    out_dir: Path
    storage_report: Dict[str, Any]
    manifest: Optional[ModelManifest] = None
    verification_report: Optional[Dict[str, Any]] = None


def exporter_version() -> str:
    from .. import __version__

    try:
        import onnx

        onnx_version = onnx.__version__
    except Exception:  # pragma: no cover
        onnx_version = "unavailable"
    return f"crowdio-export/{__version__} torch/{torch.__version__} onnx/{onnx_version}"


def _contract(role: str, hidden: int, num_labels: int) -> TensorContract:
    ids = TensorSpec(IDS_IN, "int64", ["batch", "seq"])
    mask_in = TensorSpec(MASK_IN, "int64", ["batch", "seq"])
    hidden_in = TensorSpec(HIDDEN_IN, "float32", ["batch", "seq", hidden])
    hidden_out = TensorSpec(HIDDEN_OUT, "float32", ["batch", "seq", hidden], binds_to=HIDDEN_IN)
    mask_out = TensorSpec(MASK_OUT, "int64", ["batch", "seq"], binds_to=MASK_IN)
    logits = TensorSpec(LOGITS_OUT, "float32", ["batch", num_labels])
    if role == "embed":
        return TensorContract([ids, mask_in], [hidden_out, mask_out])
    if role == "encoder":
        return TensorContract([hidden_in, mask_in], [hidden_out, mask_out])
    if role == "head":
        return TensorContract([hidden_in], [logits])
    return TensorContract([ids, mask_in], [logits])


def _activation_bytes(role: str, hidden: int, num_labels: int) -> Tuple[int, int, int]:
    """(in per token, out per token, out per sequence) for fp32 hidden, int64 ids/mask."""
    hidden_tok = hidden * 4
    if role == "embed":
        return 8 + 8, hidden_tok + 8, 0
    if role == "encoder":
        return hidden_tok + 8, hidden_tok + 8, 0
    if role == "head":
        return hidden_tok, 0, num_labels * 4
    return 8 + 8, 0, num_labels * 4


def _module_for(adapter: EncoderAdapter, planned: PlannedShard) -> nn.Module:
    if planned.role == "embed":
        return EmbedShard(adapter)
    if planned.role == "encoder":
        return EncoderRangeShard(adapter, planned.start_layer, planned.end_layer)
    if planned.role == "head":
        return HeadShard(adapter)
    return FullShard(adapter)


def _trace_args(adapter: EncoderAdapter, role: str, ids: torch.Tensor, mask: torch.Tensor):
    if role in ("embed", "full"):
        return (ids, mask)
    with torch.no_grad():
        hidden, _ = EmbedShard(adapter)(ids, mask)
    if role == "encoder":
        return (hidden, mask)
    return (hidden,)


def _export_onnx(module: nn.Module, args: tuple, contract: TensorContract, path: Path,
                 opset: int, external: bool) -> List[Path]:
    """Export one shard; returns the external data files written (may be empty)."""
    dynamic_axes = {}
    for spec in contract.inputs + contract.outputs:
        dynamic_axes[spec.name] = {i: d for i, d in enumerate(spec.shape) if isinstance(d, str)}
    target = path
    tmp_dir = None
    if external:
        tmp_dir = Path(tempfile.mkdtemp(prefix="crowdio-export-"))
        target = tmp_dir / path.name
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            module,
            args,
            target.as_posix(),
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=contract.input_names(),
            output_names=contract.output_names(),
            dynamic_axes=dynamic_axes,
            dynamo=False,
        )
    written: List[Path] = []
    if external:
        import onnx

        model = onnx.load(target.as_posix(), load_external_data=True)
        location = f"{path.name}.data"
        onnx.save_model(model, path.as_posix(), save_as_external_data=True,
                        all_tensors_to_one_file=True, location=location, size_threshold=1024)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        written.append(path.parent / location)
    import onnx

    onnx.checker.check_model(path.as_posix())
    return written


def export_model(
    model: Union[str, nn.Module],
    model_id: str,
    out_root: Union[str, Path],
    tokenizer: Any = None,
    cut_grid: int = 1,
    ranges: Optional[Sequence[Tuple[int, int]]] = None,
    include_full: bool = True,
    version: int = 1,
    sample_inputs: Optional[Sequence[Dict[str, np.ndarray]]] = None,
    sample_texts: Optional[Sequence[str]] = None,
    tolerance: float = DEFAULT_TOLERANCE,
    max_compositions: Optional[int] = DEFAULT_MAX_COMPOSITIONS,
    seed: int = 0,
    budget_ratio: Optional[float] = DEFAULT_BUDGET_RATIO,
    budget_bytes: Optional[int] = None,
    max_shard_bytes: int = DEFAULT_MAX_SHARD_BYTES,
    phone_shard_budget_bytes: int = DEFAULT_PHONE_SHARD_BUDGET_BYTES,
    external_data_threshold: int = DEFAULT_EXTERNAL_DATA_THRESHOLD,
    opset: int = DEFAULT_OPSET,
    reference_seq_len: int = DEFAULT_REFERENCE_SEQ_LEN,
    dry_run: bool = False,
    overwrite: bool = False,
    log=print,
) -> ExportResult:
    """Export verified fp32 ONNX shards over the cut grid plus a manifest.

    Raises :class:`StorageBudgetExceeded` before writing anything when the plan
    is over budget, and :class:`VerificationError` when any checked
    composition disagrees with the original model (no manifest is written).
    """
    if isinstance(model, str):
        model, loaded_tokenizer = load_model(model)
        tokenizer = tokenizer or loaded_tokenizer
    adapter = get_adapter(model)
    sizes = adapter.sizes(model_id, reference_seq_len)

    report = storage_report(sizes, cut_grid, ranges, include_full, budget_ratio, budget_bytes,
                            max_shard_bytes, phone_shard_budget_bytes)
    log(format_storage_report(report))
    out_dir = Path(out_root) / model_id
    if not report["within_budget"]:
        raise StorageBudgetExceeded(report)
    if dry_run:
        return ExportResult(out_dir, report)

    if (out_dir / "manifest.json").exists() and not overwrite:
        raise FileExistsError(
            f"{out_dir / 'manifest.json'} already exists; pass overwrite=True or choose another out_root"
        )
    if out_dir.exists() and overwrite:
        shutil.rmtree(out_dir)
    shard_dir = out_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "storage_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    atoms, planned, _ = plan_shards(sizes, cut_grid, ranges, include_full, max_shard_bytes)
    trace = make_sample_inputs(adapter.vocab_size, adapter.pad_token_id, shapes=((2, 16),), seed=seed + 1)[0]
    trace_ids = torch.from_numpy(trace["input_ids"])
    trace_mask = torch.from_numpy(trace["attention_mask"])
    version_string = exporter_version()

    specs: List[ShardSpec] = []
    for index, shard in enumerate(planned, 1):
        contract = _contract(shard.role, sizes.hidden_size, sizes.num_labels)
        path = out_dir / shard.file
        external = shard.est_bytes >= external_data_threshold
        log(f"[{index}/{len(planned)}] exporting {shard.shard_id} "
            f"({shard.est_bytes / MiB:.1f} MiB{', external data' if external else ''})")
        module = _module_for(adapter, shard).eval()
        with torch.no_grad():
            extra = _export_onnx(module, _trace_args(adapter, shard.role, trace_ids, trace_mask),
                                 contract, path, opset, external)
        size = path.stat().st_size
        if size + sum(p.stat().st_size for p in extra) > phone_shard_budget_bytes:
            log(f"  WARNING: {shard.shard_id} may not fit the smallest target phone "
                f"(> {phone_shard_budget_bytes / MiB:.0f} MiB)")
        in_tok, out_tok, out_seq = _activation_bytes(shard.role, sizes.hidden_size, sizes.num_labels)
        specs.append(ShardSpec(
            shard_id=shard.shard_id,
            model_id=model_id,
            role=shard.role,
            start_atom=shard.start_atom,
            end_atom=shard.end_atom,
            start_layer=shard.start_layer,
            end_layer=shard.end_layer,
            pinned=shard.pinned,
            file=shard.file,
            bytes=size,
            flops=shard.flops,
            in_act_bytes_per_token=in_tok,
            out_act_bytes_per_token=out_tok,
            out_act_bytes_per_sequence=out_seq,
            tensors=contract,
            sha256=sha256_file(path),
            precision="fp32",
            runtime_format="onnx",
            external_data_files=[
                ExternalDataFile(p.relative_to(out_dir).as_posix(), p.stat().st_size, sha256_file(p))
                for p in extra
            ],
            exporter_version=version_string,
        ))

    source = adapter.source_info()
    if tokenizer is not None:
        tokenizer.save_pretrained((out_dir / "tokenizer").as_posix())
        source["tokenizer_dir"] = "tokenizer"
    manifest = ModelManifest(
        model_id=model_id,
        version=version,
        source=source,
        num_layers=sizes.num_layers,
        hidden_size=sizes.hidden_size,
        cut_grid=cut_grid,
        atom_map=atoms,
        shards=specs,
        flops_estimation=dict(FLOPS_METHOD, reference_seq_len=reference_seq_len),
        exporter_version=version_string,
        created_at=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        complete_encoder_ranges=ranges is None,
    )
    report["actual_total_bytes"] = sum(s.total_bytes() for s in specs)
    report["actual_ratio_to_one_copy"] = round(report["actual_total_bytes"] / sizes.one_copy_bytes, 3)
    (out_dir / "storage_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    inputs = list(sample_inputs) if sample_inputs else make_sample_inputs(
        adapter.vocab_size, adapter.pad_token_id, seed=seed)
    if sample_texts and tokenizer is not None:
        inputs += tokenize_samples(tokenizer, sample_texts)
    log(f"verifying {model_id}: {len(inputs)} input batch(es), tolerance {tolerance:g}")
    verification = verify_export(manifest, out_dir, adapter, inputs, tolerance, max_compositions, seed)
    (out_dir / "verification_report.json").write_text(
        json.dumps(verification, indent=2) + "\n", encoding="utf-8")
    if not verification["passed"]:
        raise VerificationError(verification)
    manifest.save(out_dir / "manifest.json")
    comps = verification["compositions"]
    log(f"verified {comps['checked']}/{comps['total']} compositions ({comps['mode']}), "
        f"max |diff| {verification['max_abs_diff']:.3g}; manifest written to {out_dir / 'manifest.json'}")
    return ExportResult(out_dir, report, manifest, verification)

"""Shard planning and storage report (pure Python, no torch).

The planner turns layer sizes into the list of shards to export and a storage
report, so the budget check runs before any ONNX is written.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..catalogue.manifest import AtomSpec, all_ranges, build_atom_map

MiB = 1 << 20
GiB = 1 << 30

DEFAULT_BUDGET_RATIO = 10.0
DEFAULT_MAX_SHARD_BYTES = 2 * GiB
DEFAULT_PHONE_SHARD_BUDGET_BYTES = 800 * MiB
DEFAULT_EXTERNAL_DATA_THRESHOLD = 1 * GiB


class StorageBudgetExceeded(RuntimeError):
    """The planned export is larger than the configured storage budget."""

    def __init__(self, report: Dict[str, Any]):
        self.report = report
        super().__init__(report["message"])


@dataclass
class ModelSizes:
    """Parameter bytes and FLOPs of the parts of one encoder model."""

    model_id: str
    num_layers: int
    hidden_size: int
    num_labels: int
    embed_bytes: int
    layer_bytes: List[int]
    head_bytes: int
    embed_flops: int = 0
    layer_flops: List[int] = field(default_factory=list)
    head_flops: int = 0

    @property
    def one_copy_bytes(self) -> int:
        return self.embed_bytes + sum(self.layer_bytes) + self.head_bytes


@dataclass
class PlannedShard:
    shard_id: str
    role: str
    start_atom: Optional[int]
    end_atom: Optional[int]
    start_layer: Optional[int]
    end_layer: Optional[int]
    est_bytes: int
    flops: int

    @property
    def pinned(self) -> bool:
        return self.role in ("embed", "head")

    @property
    def file(self) -> str:
        return f"shards/{self.shard_id}.onnx"


def shard_id_for(model_id: str, role: str, start_atom: Optional[int] = None,
                 end_atom: Optional[int] = None, precision: str = "fp32") -> str:
    suffix = "" if precision == "fp32" else f"@{precision}"
    if role == "encoder":
        return f"{model_id}.enc.a{start_atom}-a{end_atom}{suffix}"
    return f"{model_id}.{role}{suffix}"


def parse_ranges(text: str) -> List[Tuple[int, int]]:
    """Parse ``"0-2,3,4-5"`` into inclusive atom ranges."""
    ranges = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            ranges.append((int(start), int(end)))
        else:
            ranges.append((int(part), int(part)))
    return ranges


def plan_shards(
    sizes: ModelSizes,
    cut_grid: int,
    ranges: Optional[Sequence[Tuple[int, int]]] = None,
    include_full: bool = True,
    max_shard_bytes: int = DEFAULT_MAX_SHARD_BYTES,
) -> Tuple[List[AtomSpec], List[PlannedShard], List[str]]:
    """Return ``(atom_map, shards, notes)``.

    ``ranges`` selects a subset of encoder ranges; ``None`` means all
    ``A(A+1)/2``. The full shard is planned only when it fits
    ``max_shard_bytes``.
    """
    atoms = build_atom_map(sizes.num_layers, cut_grid)
    num_atoms = len(atoms)
    selected = list(ranges) if ranges is not None else all_ranges(num_atoms)
    for start, end in selected:
        if not (0 <= start <= end < num_atoms):
            raise ValueError(f"range {start}-{end} is outside atoms 0..{num_atoms - 1}")
    if len(set(selected)) != len(selected):
        raise ValueError("ranges contain duplicates")

    layer_flops = sizes.layer_flops or [0] * sizes.num_layers
    notes: List[str] = []
    shards = [
        PlannedShard(shard_id_for(sizes.model_id, "embed"), "embed", None, None, None, None,
                     sizes.embed_bytes, sizes.embed_flops),
    ]
    for start, end in sorted(selected):
        first, last = atoms[start].start_layer, atoms[end].end_layer
        layer_range = range(first, last + 1)
        shards.append(PlannedShard(
            shard_id_for(sizes.model_id, "encoder", start, end), "encoder", start, end, first, last,
            sum(sizes.layer_bytes[i] for i in layer_range),
            sum(layer_flops[i] for i in layer_range),
        ))
    shards.append(PlannedShard(shard_id_for(sizes.model_id, "head"), "head", None, None, None, None,
                               sizes.head_bytes, sizes.head_flops))
    if include_full:
        if sizes.one_copy_bytes <= max_shard_bytes:
            shards.append(PlannedShard(
                shard_id_for(sizes.model_id, "full"), "full", 0, num_atoms - 1, 0, sizes.num_layers - 1,
                sizes.one_copy_bytes, sizes.embed_flops + sum(layer_flops) + sizes.head_flops,
            ))
        else:
            notes.append(
                f"full shard skipped: {sizes.one_copy_bytes / MiB:.1f} MiB exceeds "
                f"max_shard_bytes {max_shard_bytes / MiB:.1f} MiB"
            )
    for shard in shards:
        if shard.est_bytes > max_shard_bytes and shard.role != "full":
            raise ValueError(
                f"shard {shard.shard_id} needs {shard.est_bytes / MiB:.1f} MiB, over max_shard_bytes "
                f"{max_shard_bytes / MiB:.1f} MiB; use a smaller cut_grid"
            )
    return atoms, shards, notes


def storage_report(
    sizes: ModelSizes,
    cut_grid: int,
    ranges: Optional[Sequence[Tuple[int, int]]] = None,
    include_full: bool = True,
    budget_ratio: Optional[float] = DEFAULT_BUDGET_RATIO,
    budget_bytes: Optional[int] = None,
    max_shard_bytes: int = DEFAULT_MAX_SHARD_BYTES,
    phone_shard_budget_bytes: int = DEFAULT_PHONE_SHARD_BUDGET_BYTES,
    suggest: bool = True,
) -> Dict[str, Any]:
    """Estimate export size from parameter bytes and check it against the budget."""
    atoms, shards, notes = plan_shards(sizes, cut_grid, ranges, include_full, max_shard_bytes)
    num_atoms = len(atoms)
    encoders = [s for s in shards if s.role == "encoder"]
    atom_copies = sum(s.end_atom - s.start_atom + 1 for s in encoders)
    layer_copies = sum(s.end_layer - s.start_layer + 1 for s in encoders)
    total = sum(s.est_bytes for s in shards)
    one_copy = sizes.one_copy_bytes
    ratio = total / one_copy if one_copy else 0.0
    complete = ranges is None

    limits = []
    if budget_ratio is not None:
        limits.append(budget_ratio * one_copy)
    if budget_bytes is not None:
        limits.append(budget_bytes)
    limit = min(limits) if limits else None
    within = limit is None or total <= limit

    report: Dict[str, Any] = {
        "model_id": sizes.model_id,
        "estimate_basis": "parameter bytes (ONNX graph overhead not included)",
        "num_layers": sizes.num_layers,
        "cut_grid": cut_grid,
        "num_atoms": num_atoms,
        "complete_encoder_ranges": complete,
        "encoder_shards": len(encoders),
        "expected_encoder_shards_if_complete": num_atoms * (num_atoms + 1) // 2,
        "pinned_shards": [s.shard_id for s in shards if s.pinned],
        "full_shard": any(s.role == "full" for s in shards),
        "atom_copies": atom_copies,
        "expected_atom_copies_if_complete": num_atoms * (num_atoms + 1) * (num_atoms + 2) // 6,
        "layer_copies": layer_copies,
        "one_copy_bytes": one_copy,
        "embed_bytes": sizes.embed_bytes,
        "head_bytes": sizes.head_bytes,
        "encoder_bytes": sum(s.est_bytes for s in encoders),
        "total_bytes": total,
        "ratio_to_one_copy": round(ratio, 3),
        "budget_ratio": budget_ratio,
        "budget_bytes": budget_bytes,
        "within_budget": within,
        "largest_shard": max(shards, key=lambda s: s.est_bytes).shard_id,
        "largest_shard_bytes": max(s.est_bytes for s in shards),
        "phone_shard_budget_bytes": phone_shard_budget_bytes,
        "over_phone_budget": [s.shard_id for s in shards if s.est_bytes > phone_shard_budget_bytes],
        "notes": notes,
        "shards": [dict(asdict(s), pinned=s.pinned) for s in shards],
    }
    if not within:
        message = (
            f"Planned export for {sizes.model_id} is {total / MiB:.1f} MiB "
            f"({ratio:.2f}x one model copy), over the budget of {limit / MiB:.1f} MiB."
        )
        if suggest:
            options = []
            for g in range(cut_grid + 1, sizes.num_layers + 1):
                if g != cut_grid and (sizes.num_layers % g == 0 or g == sizes.num_layers):
                    alt = storage_report(sizes, g, None, include_full, budget_ratio, budget_bytes,
                                         max_shard_bytes, phone_shard_budget_bytes, suggest=False)
                    options.append(f"cut_grid={g}: {alt['total_bytes'] / MiB:.1f} MiB "
                                   f"({alt['ratio_to_one_copy']:.2f}x)")
                    if alt["within_budget"]:
                        break
            if options:
                message += " Raise cut_grid; " + "; ".join(options) + "."
            report["suggestions"] = options
        report["message"] = message
    return report


def format_storage_report(report: Dict[str, Any]) -> str:
    lines = [
        f"Storage report for {report['model_id']} (cut_grid={report['cut_grid']}, "
        f"{report['num_atoms']} atoms over {report['num_layers']} layers)",
        f"  encoder shards : {report['encoder_shards']} "
        f"(complete grid = {report['expected_encoder_shards_if_complete']})",
        f"  pinned shards  : {', '.join(report['pinned_shards'])}",
        f"  full shard     : {'yes' if report['full_shard'] else 'no'}",
        f"  atom copies    : {report['atom_copies']} "
        f"(A(A+1)(A+2)/6 = {report['expected_atom_copies_if_complete']}), layer copies: {report['layer_copies']}",
        f"  one model copy : {report['one_copy_bytes'] / MiB:10.1f} MiB",
        f"  total export   : {report['total_bytes'] / MiB:10.1f} MiB  "
        f"({report['ratio_to_one_copy']:.2f}x one copy, {report['estimate_basis']})",
        f"  budget         : ratio {report['budget_ratio']}, bytes {report['budget_bytes']} -> "
        f"{'OK' if report['within_budget'] else 'EXCEEDED'}",
        f"  largest shard  : {report['largest_shard']} ({report['largest_shard_bytes'] / MiB:.1f} MiB)",
    ]
    if report["over_phone_budget"]:
        lines.append(
            f"  WARNING: {len(report['over_phone_budget'])} shard(s) exceed the phone shard budget of "
            f"{report['phone_shard_budget_bytes'] / MiB:.0f} MiB: {', '.join(report['over_phone_budget'])}"
        )
    for note in report["notes"]:
        lines.append(f"  note: {note}")
    if not report["within_budget"]:
        lines.append(f"  {report['message']}")
    return "\n".join(lines)

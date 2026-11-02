"""Shard catalogue manifest (shared SDK / Foreman / worker contract).

One manifest describes every exported shard of one model version. The schema
is mirrored in ``docs/manifest.schema.json``; this module is the reference
implementation used by the export tool and by ``describe_model``.

Shard roles:

* ``embed``   pinned embedding block, never cut or merged
* ``encoder`` a contiguous range of atoms ``start_atom..end_atom`` (inclusive)
* ``head``    pinned classifier / pooling head, never cut or merged
* ``full``    the whole model (embed + every layer + head) in one graph

Tensor binding: an output's ``binds_to`` names the input of the next shard it
feeds. A runtime feeds a shard only the inputs that shard declares, so the
``attention_mask_out`` of the last encoder shard is simply not consumed by the
head.

This module has no torch / onnx dependency.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

MANIFEST_VERSION = 1

ROLES = ("embed", "encoder", "head", "full")
PRECISIONS = ("fp32", "fp16", "int8")
RUNTIME_FORMATS = ("onnx", "ort")
DTYPES = ("float32", "float16", "int64", "int32", "int8", "uint8", "bool")

HIDDEN_IN = "hidden_states"
HIDDEN_OUT = "hidden_states_out"
MASK_IN = "attention_mask"
MASK_OUT = "attention_mask_out"
IDS_IN = "input_ids"
LOGITS_OUT = "logits"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")

Dim = Union[int, str]


class ManifestError(ValueError):
    """Raised when a manifest does not satisfy the shared schema."""

    def __init__(self, errors: Sequence[str]):
        self.errors = list(errors)
        super().__init__("Invalid manifest:\n  - " + "\n  - ".join(self.errors))


@dataclass
class TensorSpec:
    name: str
    dtype: str
    shape: List[Dim]
    binds_to: Optional[str] = None

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "TensorSpec":
        return cls(
            name=str(value["name"]),
            dtype=str(value["dtype"]),
            shape=list(value["shape"]),
            binds_to=value.get("binds_to"),
        )


@dataclass
class TensorContract:
    inputs: List[TensorSpec]
    outputs: List[TensorSpec]

    def input_names(self) -> List[str]:
        return [t.name for t in self.inputs]

    def output_names(self) -> List[str]:
        return [t.name for t in self.outputs]

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "TensorContract":
        return cls(
            inputs=[TensorSpec.from_dict(t) for t in value["inputs"]],
            outputs=[TensorSpec.from_dict(t) for t in value["outputs"]],
        )


@dataclass
class ExternalDataFile:
    path: str
    bytes: int
    sha256: str

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "ExternalDataFile":
        return cls(path=str(value["path"]), bytes=int(value["bytes"]), sha256=str(value["sha256"]))


@dataclass
class ShardSpec:
    shard_id: str
    model_id: str
    role: str
    start_atom: Optional[int]
    end_atom: Optional[int]
    start_layer: Optional[int]
    end_layer: Optional[int]
    pinned: bool
    file: str
    bytes: int
    flops: int
    in_act_bytes_per_token: int
    out_act_bytes_per_token: int
    tensors: TensorContract
    sha256: str
    precision: str = "fp32"
    runtime_format: str = "onnx"
    external_data_files: List[ExternalDataFile] = field(default_factory=list)
    exporter_version: str = ""
    # Extension (flagged in the shared contract): per-sequence outputs such as
    # logits are not proportional to tokens.
    out_act_bytes_per_sequence: int = 0

    @property
    def atom_range(self) -> Optional[Tuple[int, int]]:
        if self.start_atom is None or self.end_atom is None:
            return None
        return self.start_atom, self.end_atom

    def total_bytes(self) -> int:
        return self.bytes + sum(f.bytes for f in self.external_data_files)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "ShardSpec":
        return cls(
            shard_id=str(value["shard_id"]),
            model_id=str(value["model_id"]),
            role=str(value["role"]),
            start_atom=value.get("start_atom"),
            end_atom=value.get("end_atom"),
            start_layer=value.get("start_layer"),
            end_layer=value.get("end_layer"),
            pinned=bool(value["pinned"]),
            file=str(value["file"]),
            bytes=int(value["bytes"]),
            flops=int(value["flops"]),
            in_act_bytes_per_token=int(value["in_act_bytes_per_token"]),
            out_act_bytes_per_token=int(value["out_act_bytes_per_token"]),
            tensors=TensorContract.from_dict(value["tensors"]),
            sha256=str(value["sha256"]),
            precision=str(value.get("precision", "fp32")),
            runtime_format=str(value.get("runtime_format", "onnx")),
            external_data_files=[
                ExternalDataFile.from_dict(f) for f in value.get("external_data_files", [])
            ],
            exporter_version=str(value.get("exporter_version", "")),
            out_act_bytes_per_sequence=int(value.get("out_act_bytes_per_sequence", 0)),
        )


@dataclass
class AtomSpec:
    atom: int
    start_layer: int
    end_layer: int


@dataclass
class ModelManifest:
    model_id: str
    version: int
    source: Dict[str, Any]
    num_layers: int
    hidden_size: int
    cut_grid: int
    atom_map: List[AtomSpec]
    shards: List[ShardSpec]
    flops_estimation: Dict[str, Any]
    exporter_version: str
    created_at: str
    complete_encoder_ranges: bool = True
    manifest_version: int = MANIFEST_VERSION

    @property
    def num_atoms(self) -> int:
        return len(self.atom_map)

    @property
    def has_full_shard(self) -> bool:
        return any(s.role == "full" for s in self.shards)

    def shards_by_role(self, role: str, precision: Optional[str] = None) -> List[ShardSpec]:
        return [
            s for s in self.shards
            if s.role == role and (precision is None or s.precision == precision)
        ]

    def shard(self, shard_id: str) -> ShardSpec:
        for s in self.shards:
            if s.shard_id == shard_id:
                return s
        raise KeyError(shard_id)

    def encoder_shard(self, start_atom: int, end_atom: int, precision: str = "fp32") -> Optional[ShardSpec]:
        for s in self.shards_by_role("encoder", precision):
            if s.atom_range == (start_atom, end_atom):
                return s
        return None

    def available_ranges(self, precision: str = "fp32") -> List[Tuple[int, int]]:
        return sorted(s.atom_range for s in self.shards_by_role("encoder", precision))

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["num_atoms"] = self.num_atoms
        value["has_full_shard"] = self.has_full_shard
        return value

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "ModelManifest":
        return cls(
            model_id=str(value["model_id"]),
            version=int(value["version"]),
            source=dict(value.get("source", {})),
            num_layers=int(value["num_layers"]),
            hidden_size=int(value["hidden_size"]),
            cut_grid=int(value["cut_grid"]),
            atom_map=[AtomSpec(int(a["atom"]), int(a["start_layer"]), int(a["end_layer"])) for a in value["atom_map"]],
            shards=[ShardSpec.from_dict(s) for s in value["shards"]],
            flops_estimation=dict(value.get("flops_estimation", {})),
            exporter_version=str(value.get("exporter_version", "")),
            created_at=str(value.get("created_at", "")),
            complete_encoder_ranges=bool(value.get("complete_encoder_ranges", True)),
            manifest_version=int(value.get("manifest_version", MANIFEST_VERSION)),
        )

    @classmethod
    def from_json(cls, text: str) -> "ModelManifest":
        return cls.from_dict(json.loads(text))

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ModelManifest":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    def save(self, path: Union[str, Path]) -> None:
        Path(path).write_text(self.to_json() + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Atom / range arithmetic (shared with the planner)
# ---------------------------------------------------------------------------

def build_atom_map(num_layers: int, cut_grid: int) -> List[AtomSpec]:
    """Group ``num_layers`` layers into atoms of ``cut_grid`` layers.

    The last atom is shorter when ``num_layers`` is not a multiple of
    ``cut_grid``.
    """
    if num_layers < 1:
        raise ValueError("num_layers must be >= 1")
    if cut_grid < 1:
        raise ValueError("cut_grid must be >= 1")
    atoms = []
    for index, start in enumerate(range(0, num_layers, cut_grid)):
        atoms.append(AtomSpec(index, start, min(start + cut_grid, num_layers) - 1))
    return atoms


def all_ranges(num_atoms: int) -> List[Tuple[int, int]]:
    """Every contiguous inclusive atom range: ``A(A+1)/2`` of them."""
    return [(s, e) for s in range(num_atoms) for e in range(s, num_atoms)]


def iter_compositions(
    num_atoms: int, ranges: Optional[Iterable[Tuple[int, int]]] = None
) -> Iterator[Tuple[Tuple[int, int], ...]]:
    """Yield every way to cover atoms ``0..A-1`` with available contiguous ranges."""
    available = set(ranges) if ranges is not None else set(all_ranges(num_atoms))

    def walk(start: int) -> Iterator[Tuple[Tuple[int, int], ...]]:
        if start == num_atoms:
            yield ()
            return
        for end in range(start, num_atoms):
            if (start, end) in available:
                for rest in walk(end + 1):
                    yield ((start, end),) + rest

    yield from walk(0)


def sha256_file(path: Union[str, Path], chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_REQUIRED_IO = {
    "embed": ({IDS_IN, MASK_IN}, {HIDDEN_OUT, MASK_OUT}),
    "encoder": ({HIDDEN_IN, MASK_IN}, {HIDDEN_OUT, MASK_OUT}),
    "head": ({HIDDEN_IN}, {LOGITS_OUT}),
    "full": ({IDS_IN, MASK_IN}, {LOGITS_OUT}),
}


def _check_contract(shard: ShardSpec, errors: List[str]) -> None:
    where = f"shard {shard.shard_id}"
    required = _REQUIRED_IO.get(shard.role)
    if required is None:
        return
    inputs = set(shard.tensors.input_names())
    outputs = set(shard.tensors.output_names())
    for name in sorted(required[0] - inputs):
        errors.append(f"{where}: tensor contract is missing input '{name}'")
    for name in sorted(required[1] - outputs):
        errors.append(f"{where}: tensor contract is missing output '{name}'")
    for spec in shard.tensors.inputs + shard.tensors.outputs:
        if spec.dtype not in DTYPES:
            errors.append(f"{where}: tensor '{spec.name}' has unknown dtype '{spec.dtype}'")
        if not spec.shape:
            errors.append(f"{where}: tensor '{spec.name}' has no shape")
        for dim in spec.shape:
            if not (isinstance(dim, str) and dim) and not (isinstance(dim, int) and dim > 0):
                errors.append(f"{where}: tensor '{spec.name}' has invalid dim {dim!r}")
    for spec in shard.tensors.outputs:
        if spec.name in (HIDDEN_OUT, MASK_OUT) and spec.binds_to not in (HIDDEN_IN, MASK_IN):
            errors.append(f"{where}: output '{spec.name}' must bind to the next shard's input")


def validate_manifest(
    manifest: ModelManifest,
    root: Optional[Union[str, Path]] = None,
    check_files: bool = False,
) -> None:
    """Raise :class:`ManifestError` listing every violation found.

    With ``check_files`` every shard file under ``root`` is size- and
    sha256-checked.
    """
    errors: List[str] = []
    if manifest.manifest_version != MANIFEST_VERSION:
        errors.append(f"unsupported manifest_version {manifest.manifest_version}")
    if not manifest.model_id:
        errors.append("model_id is empty")
    if manifest.version < 1:
        errors.append("version must be >= 1")
    if manifest.cut_grid < 1:
        errors.append("cut_grid must be >= 1")
    if manifest.hidden_size < 1:
        errors.append("hidden_size must be >= 1")

    try:
        expected_atoms = build_atom_map(manifest.num_layers, manifest.cut_grid)
    except ValueError as exc:
        errors.append(str(exc))
        expected_atoms = []
    if [asdict(a) for a in manifest.atom_map] != [asdict(a) for a in expected_atoms]:
        errors.append("atom_map does not match num_layers and cut_grid")
    atoms = {a.atom: a for a in manifest.atom_map}
    num_atoms = len(manifest.atom_map)

    seen_ids = set()
    seen_ranges: Dict[Tuple[str, str, Tuple[int, int]], str] = {}
    pinned_counts: Dict[Tuple[str, str], int] = {}
    precisions = set()
    for shard in manifest.shards:
        where = f"shard {shard.shard_id}"
        if shard.shard_id in seen_ids:
            errors.append(f"{where}: duplicate shard_id")
        seen_ids.add(shard.shard_id)
        if shard.model_id != manifest.model_id:
            errors.append(f"{where}: model_id '{shard.model_id}' != '{manifest.model_id}'")
        if shard.role not in ROLES:
            errors.append(f"{where}: unknown role '{shard.role}'")
            continue
        if shard.precision not in PRECISIONS:
            errors.append(f"{where}: unknown precision '{shard.precision}'")
        precisions.add(shard.precision)
        if shard.runtime_format not in RUNTIME_FORMATS:
            errors.append(f"{where}: unknown runtime_format '{shard.runtime_format}'")
        if not _SHA256.match(shard.sha256):
            errors.append(f"{where}: sha256 must be 64 lowercase hex characters")
        if shard.bytes <= 0:
            errors.append(f"{where}: bytes must be > 0")
        if shard.flops < 0 or shard.in_act_bytes_per_token < 0 or shard.out_act_bytes_per_token < 0:
            errors.append(f"{where}: flops and activation sizes must be >= 0")
        for ext in shard.external_data_files:
            if not _SHA256.match(ext.sha256):
                errors.append(f"{where}: external file {ext.path} has an invalid sha256")

        if shard.role in ("embed", "head"):
            if not shard.pinned:
                errors.append(f"{where}: {shard.role} shards must be pinned")
            if any(v is not None for v in (shard.start_atom, shard.end_atom, shard.start_layer, shard.end_layer)):
                errors.append(f"{where}: {shard.role} shards must not carry atom or layer bounds")
            key = (shard.role, shard.precision)
            pinned_counts[key] = pinned_counts.get(key, 0) + 1
        else:
            if shard.pinned:
                errors.append(f"{where}: {shard.role} shards must not be pinned")
            rng = shard.atom_range
            if rng is None or shard.start_layer is None or shard.end_layer is None:
                errors.append(f"{where}: {shard.role} shards need start/end atom and layer")
            else:
                start, end = rng
                if not (0 <= start <= end < num_atoms):
                    errors.append(f"{where}: atom range {rng} outside 0..{num_atoms - 1}")
                else:
                    if (shard.start_layer, shard.end_layer) != (atoms[start].start_layer, atoms[end].end_layer):
                        errors.append(f"{where}: layer bounds do not match atom_map for atoms {rng}")
                    if shard.role == "full" and rng != (0, num_atoms - 1):
                        errors.append(f"{where}: full shard must span every atom")
                    key = (shard.role, shard.precision, rng)
                    if key in seen_ranges:
                        errors.append(f"{where}: duplicates range {rng} of {seen_ranges[key]}")
                    seen_ranges[key] = shard.shard_id
        _check_contract(shard, errors)

    for precision in sorted(precisions):
        for role in ("embed", "head"):
            count = pinned_counts.get((role, precision), 0)
            if count != 1:
                errors.append(f"precision {precision}: expected exactly one {role} shard, found {count}")
        present = {r for (role, p, r) in seen_ranges if role == "encoder" and p == precision}
        if manifest.complete_encoder_ranges:
            missing = [r for r in all_ranges(num_atoms) if r not in present]
            if missing:
                errors.append(
                    f"precision {precision}: complete_encoder_ranges is true but {len(missing)} "
                    f"range(s) are missing, e.g. {missing[:3]}"
                )
        if num_atoms and not any(True for _ in iter_compositions(num_atoms, present)):
            errors.append(f"precision {precision}: encoder ranges cannot cover every atom")

    if check_files:
        base = Path(root) if root is not None else Path(".")
        for shard in manifest.shards:
            files = [(shard.file, shard.bytes, shard.sha256)] + [
                (f.path, f.bytes, f.sha256) for f in shard.external_data_files
            ]
            for rel, size, digest in files:
                path = base / rel
                if not path.is_file():
                    errors.append(f"shard {shard.shard_id}: file {rel} not found")
                    continue
                if path.stat().st_size != size:
                    errors.append(f"shard {shard.shard_id}: {rel} is {path.stat().st_size} bytes, manifest says {size}")
                elif sha256_file(path) != digest:
                    errors.append(f"shard {shard.shard_id}: {rel} sha256 mismatch")

    if errors:
        raise ManifestError(errors)


def summarize_manifest(manifest: ModelManifest) -> Dict[str, Any]:
    """Compact summary returned by ``list_models`` / ``describe_model``."""
    return {
        "model_id": manifest.model_id,
        "version": manifest.version,
        "num_layers": manifest.num_layers,
        "cut_grid": manifest.cut_grid,
        "num_atoms": manifest.num_atoms,
        "atom_map": [asdict(a) for a in manifest.atom_map],
        "available_ranges": {
            p: manifest.available_ranges(p) for p in sorted({s.precision for s in manifest.shards})
        },
        "has_full_shard": manifest.has_full_shard,
        "shard_bytes": {s.shard_id: s.total_bytes() for s in manifest.shards},
        "tensor_contracts": {
            role: asdict(next(s for s in manifest.shards if s.role == role).tensors)
            for role in ROLES if any(s.role == role for s in manifest.shards)
        },
    }

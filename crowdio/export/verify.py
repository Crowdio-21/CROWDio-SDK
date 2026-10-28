"""Mandatory verification of exported shards against the original model.

Structural checks: the manifest validates, every file matches its size and
sha256, and every ONNX graph exposes exactly the tensors its contract names
(so a shard that drops ``attention_mask`` fails).

Numerical checks: on several padded inputs of varying batch and sequence
length, run ``embed -> encoder ranges -> head`` for each composition and the
full shard, and compare logits with the PyTorch model: max absolute
difference under ``tolerance`` and identical argmax.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..catalogue.manifest import (
    ManifestError,
    ModelManifest,
    ShardSpec,
    iter_compositions,
    validate_manifest,
)

DEFAULT_TOLERANCE = 1e-4
DEFAULT_MAX_COMPOSITIONS = 256
DEFAULT_INPUT_SHAPES: Tuple[Tuple[int, int], ...] = ((1, 8), (2, 16), (3, 33))

Composition = Tuple[Tuple[int, int], ...]


class VerificationError(RuntimeError):
    def __init__(self, report: Dict[str, Any]):
        self.report = report
        failures = report.get("structural_errors", []) + [
            f"{f['what']}: {f['reason']}" for f in report.get("failures", [])
        ]
        super().__init__("Shard verification failed:\n  - " + "\n  - ".join(failures[:20]))


def make_sample_inputs(
    vocab_size: int,
    pad_token_id: int,
    shapes: Sequence[Tuple[int, int]] = DEFAULT_INPUT_SHAPES,
    seed: int = 0,
) -> List[Dict[str, np.ndarray]]:
    """Random token batches; every batch contains padded rows."""
    rng = np.random.default_rng(seed)
    low = min(1000, max(1, vocab_size // 4))
    samples = []
    for batch, seq in shapes:
        ids = rng.integers(low, vocab_size, size=(batch, seq), dtype=np.int64)
        mask = np.ones((batch, seq), dtype=np.int64)
        for row in range(batch):
            # Row 0 of a multi-row batch is full length; every other row is padded.
            if batch > 1 and row == 0:
                continue
            length = int(rng.integers(max(1, seq // 3), seq)) if seq > 1 else 1
            ids[row, length:] = pad_token_id
            mask[row, length:] = 0
        samples.append({"input_ids": ids, "attention_mask": mask})
    return samples


def tokenize_samples(tokenizer: Any, texts: Iterable[str], max_length: int = 64) -> List[Dict[str, np.ndarray]]:
    encoded = tokenizer(list(texts), return_tensors="np", padding="longest", truncation=True, max_length=max_length)
    return [{
        "input_ids": np.asarray(encoded["input_ids"], dtype=np.int64),
        "attention_mask": np.asarray(encoded["attention_mask"], dtype=np.int64),
    }]


def select_compositions(
    num_atoms: int,
    ranges: Sequence[Tuple[int, int]],
    max_compositions: Optional[int],
    seed: int,
) -> Tuple[List[Composition], int, str]:
    """All compositions when they fit, else a covering set plus a random sample.

    The covering set has, for every available range, one composition that
    uses it, so every encoder shard is executed at least once.
    """
    available = set(ranges)
    every = list(iter_compositions(num_atoms, available))
    total = len(every)
    if max_compositions is None or total <= max_compositions:
        return every, total, "all"

    chosen: List[Composition] = []
    seen = set()

    def add(comp: Composition) -> None:
        if comp not in seen:
            seen.add(comp)
            chosen.append(comp)

    for start, end in sorted(available):
        for comp in every:
            if (start, end) in comp:
                add(comp)
                break
    rng = random.Random(seed)
    rest = [c for c in every if c not in seen]
    rng.shuffle(rest)
    for comp in rest[: max(0, max_compositions - len(chosen))]:
        add(comp)
    return chosen, total, f"sample(seed={seed}, covers every range)"


class _Runner:
    def __init__(self, manifest: ModelManifest, root: Path, providers: Sequence[str]):
        import onnxruntime as ort

        self._ort = ort
        self.manifest = manifest
        self.root = root
        self.providers = list(providers)
        self._sessions: Dict[str, Any] = {}

    def session(self, shard: ShardSpec):
        if shard.shard_id not in self._sessions:
            options = self._ort.SessionOptions()
            options.log_severity_level = 3
            self._sessions[shard.shard_id] = self._ort.InferenceSession(
                str(self.root / shard.file), sess_options=options, providers=self.providers
            )
        return self._sessions[shard.shard_id]

    def check_graph(self, shard: ShardSpec) -> List[str]:
        session = self.session(shard)
        errors = []
        graph_in = {i.name for i in session.get_inputs()}
        graph_out = {o.name for o in session.get_outputs()}
        want_in = set(shard.tensors.input_names())
        want_out = set(shard.tensors.output_names())
        if graph_in != want_in:
            errors.append(f"shard {shard.shard_id}: graph inputs {sorted(graph_in)} != contract {sorted(want_in)}")
        if graph_out != want_out:
            errors.append(f"shard {shard.shard_id}: graph outputs {sorted(graph_out)} != contract {sorted(want_out)}")
        return errors

    def run(self, shard: ShardSpec, tensors: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Run one shard, feeding only the inputs it declares; rename by ``binds_to``."""
        session = self.session(shard)
        feeds = {name: tensors[name] for name in shard.tensors.input_names()}
        values = session.run(shard.tensors.output_names(), feeds)
        return {
            (spec.binds_to or spec.name): value for spec, value in zip(shard.tensors.outputs, values)
        }


def verify_export(
    manifest: ModelManifest,
    root: Union[str, Path],
    adapter: Any,
    inputs: Sequence[Dict[str, np.ndarray]],
    tolerance: float = DEFAULT_TOLERANCE,
    max_compositions: Optional[int] = DEFAULT_MAX_COMPOSITIONS,
    seed: int = 0,
    precision: str = "fp32",
    providers: Sequence[str] = ("CPUExecutionProvider",),
) -> Dict[str, Any]:
    """Return a verification report; ``report["passed"]`` is the verdict."""
    import torch

    root = Path(root)
    report: Dict[str, Any] = {
        "model_id": manifest.model_id,
        "precision": precision,
        "tolerance": tolerance,
        "criteria": "max |logits - reference| < tolerance and identical argmax on every input",
        "reference": "PyTorch model forward (eager attention)",
        "inputs": [
            {
                "batch": int(x["input_ids"].shape[0]),
                "seq": int(x["input_ids"].shape[1]),
                "padded_rows": int((x["attention_mask"] == 0).any(axis=1).sum()),
            }
            for x in inputs
        ],
        "structural_errors": [],
        "failures": [],
    }

    try:
        validate_manifest(manifest, root, check_files=True)
    except ManifestError as exc:
        report["structural_errors"].extend(exc.errors)

    shards = [s for s in manifest.shards if s.precision == precision]
    embed = next((s for s in shards if s.role == "embed"), None)
    head = next((s for s in shards if s.role == "head"), None)
    full = next((s for s in shards if s.role == "full"), None)
    if embed is None or head is None:
        report["structural_errors"].append("embed or head shard missing")
    if report["structural_errors"]:
        report["passed"] = False
        return report

    runner = _Runner(manifest, root, providers)
    for shard in shards:
        try:
            report["structural_errors"].extend(runner.check_graph(shard))
        except Exception as exc:  # pragma: no cover - depends on onnxruntime errors
            report["structural_errors"].append(f"shard {shard.shard_id}: cannot load: {exc}")
    if report["structural_errors"]:
        report["passed"] = False
        return report

    ranges = manifest.available_ranges(precision)
    compositions, total, mode = select_compositions(manifest.num_atoms, ranges, max_compositions, seed)
    report["compositions"] = {"total": total, "checked": len(compositions), "mode": mode}

    references = []
    for sample in inputs:
        logits = adapter.reference_logits(
            torch.from_numpy(sample["input_ids"]), torch.from_numpy(sample["attention_mask"])
        )
        references.append(logits.detach().cpu().numpy().astype(np.float32))

    def compare(what: str, outputs: List[np.ndarray]) -> Dict[str, Any]:
        worst = 0.0
        argmax_ok = True
        for out, ref in zip(outputs, references):
            worst = max(worst, float(np.max(np.abs(out - ref))))
            argmax_ok = argmax_ok and bool(np.array_equal(out.argmax(-1), ref.argmax(-1)))
        entry = {"what": what, "max_abs_diff": worst, "argmax_match": argmax_ok}
        if not (worst < tolerance and argmax_ok):
            reason = []
            if worst >= tolerance:
                reason.append(f"max_abs_diff {worst:.3g} >= {tolerance:g}")
            if not argmax_ok:
                reason.append("argmax differs")
            report["failures"].append({"what": what, "reason": ", ".join(reason)})
        return entry

    if full is not None:
        outs = [runner.run(full, dict(x))["logits"] for x in inputs]
        report["full_shard"] = compare(full.shard_id, outs)

    # Depth-first over compositions with prefix memoisation per input.
    caches: List[Dict[Composition, Dict[str, np.ndarray]]] = [
        {(): runner.run(embed, dict(x))} for x in inputs
    ]

    def state_after(cache: Dict[Composition, Dict[str, np.ndarray]], prefix: Composition):
        if prefix not in cache:
            previous = state_after(cache, prefix[:-1])
            cache[prefix] = runner.run(manifest.encoder_shard(*prefix[-1], precision), previous)
        return cache[prefix]

    results = []
    for comp in compositions:
        outs = [runner.run(head, state_after(cache, comp))["logits"] for cache in caches]
        label = " > ".join(f"a{s}-a{e}" for s, e in comp)
        results.append(dict(compare(label, outs), ranges=[list(r) for r in comp]))
    report["composition_results"] = results
    report["max_abs_diff"] = max([r["max_abs_diff"] for r in results] + (
        [report["full_shard"]["max_abs_diff"]] if full is not None else []))
    report["shards_exercised"] = sorted(
        {embed.shard_id, head.shard_id}
        | ({full.shard_id} if full is not None else set())
        | {manifest.encoder_shard(s, e, precision).shard_id for comp in compositions for s, e in comp}
    )
    report["passed"] = not report["failures"]
    return report

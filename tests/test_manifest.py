import copy
import json
from pathlib import Path

import pytest

from crowdio.catalogue import (
    ManifestError,
    ModelManifest,
    all_ranges,
    build_atom_map,
    iter_compositions,
    summarize_manifest,
    validate_manifest,
)

SCHEMA = Path(__file__).resolve().parents[1] / "docs" / "manifest.schema.json"


def _mutated(manifest, fn):
    data = copy.deepcopy(manifest.to_dict())
    fn(data)
    return ModelManifest.from_dict(data)


def _errors(manifest, **kwargs):
    with pytest.raises(ManifestError) as info:
        validate_manifest(manifest, **kwargs)
    return "\n".join(info.value.errors)


def _encoder(data, start=0, end=0):
    return next(s for s in data["shards"] if s["role"] == "encoder"
                and (s["start_atom"], s["end_atom"]) == (start, end))


def test_range_arithmetic():
    assert len(all_ranges(6)) == 21
    assert len(all_ranges(12)) == 78
    assert sum(1 for _ in iter_compositions(6)) == 2 ** 5
    assert [(a.start_layer, a.end_layer) for a in build_atom_map(6, 4)] == [(0, 3), (4, 5)]
    assert list(iter_compositions(3, [(0, 1), (2, 2)])) == [((0, 1), (2, 2))]


def test_exported_manifest_is_valid_and_round_trips(tiny_export):
    manifest = tiny_export.manifest
    validate_manifest(manifest, tiny_export.out_dir, check_files=True)
    on_disk = ModelManifest.load(tiny_export.out_dir / "manifest.json")
    assert on_disk.to_dict() == manifest.to_dict()
    assert manifest.num_atoms == 4 and manifest.has_full_shard
    assert len(manifest.shards_by_role("encoder")) == 10
    for shard in manifest.shards_by_role("encoder"):
        assert "attention_mask" in shard.tensors.input_names()
        assert "attention_mask_out" in shard.tensors.output_names()
    summary = summarize_manifest(manifest)
    assert summary["available_ranges"]["fp32"] == all_ranges(4)
    assert set(summary["tensor_contracts"]) == {"embed", "encoder", "head", "full"}


def test_manifest_matches_json_schema(tiny_export):
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(json.loads((tiny_export.out_dir / "manifest.json").read_text()), schema)


def test_dropping_the_mask_from_a_contract_is_rejected(tiny_export):
    def drop(data):
        enc = _encoder(data)
        enc["tensors"]["inputs"] = [t for t in enc["tensors"]["inputs"] if t["name"] != "attention_mask"]

    assert "missing input 'attention_mask'" in _errors(_mutated(tiny_export.manifest, drop))


@pytest.mark.parametrize("mutate, expected", [
    (lambda d: d["shards"].remove(_encoder(d, 1, 2)), "range(s) are missing"),
    (lambda d: _encoder(d).update(end_layer=1), "layer bounds do not match"),
    (lambda d: _encoder(d).update(sha256="xyz"), "sha256 must be"),
    (lambda d: d["shards"][0].update(pinned=False), "must be pinned"),
    (lambda d: _encoder(d).update(pinned=True), "must not be pinned"),
    (lambda d: _encoder(d, 1, 1).update(shard_id=_encoder(d)["shard_id"]), "duplicate shard_id"),
    (lambda d: _encoder(d).update(role="middle"), "unknown role"),
    (lambda d: d.update(cut_grid=2), "atom_map does not match"),
    (lambda d: [s for s in d["shards"] if s["role"] == "full"][0].update(end_atom=2, end_layer=2),
     "full shard must span every atom"),
])
def test_structural_violations(tiny_export, mutate, expected):
    assert expected in _errors(_mutated(tiny_export.manifest, mutate))


def test_subset_must_still_cover_every_atom(tiny_export):
    def keep_prefix(data):
        data["complete_encoder_ranges"] = False
        data["shards"] = [s for s in data["shards"] if s["role"] != "encoder" or s["end_atom"] < 3]

    assert "cannot cover every atom" in _errors(_mutated(tiny_export.manifest, keep_prefix))


def test_file_checks_detect_tampering(tiny_export, tmp_path):
    import shutil

    copy_dir = tmp_path / "copy"
    shutil.copytree(tiny_export.out_dir, copy_dir)
    shard = tiny_export.manifest.shards_by_role("encoder")[0]
    path = copy_dir / shard.file
    data = bytearray(path.read_bytes())
    data[-1] ^= 0xFF
    path.write_bytes(bytes(data))
    assert "sha256 mismatch" in _errors(tiny_export.manifest, root=copy_dir, check_files=True)
    path.unlink()
    assert "not found" in _errors(tiny_export.manifest, root=copy_dir, check_files=True)

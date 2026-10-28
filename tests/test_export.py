import copy
import json

import pytest

from crowdio.catalogue import ModelManifest, all_ranges
from crowdio.export.exporter import export_model
from crowdio.export.modules import EncoderRangeShard
from crowdio.export.planner import StorageBudgetExceeded
from crowdio.export.verify import VerificationError, make_sample_inputs, verify_export

from conftest import tiny_bert, tiny_distilbert

quiet = dict(log=lambda *_: None)


def test_tiny_distilbert_export_is_verified(tiny_export):
    report = tiny_export.verification_report
    manifest = tiny_export.manifest
    assert report["passed"]
    assert report["compositions"] == {"total": 8, "checked": 8, "mode": "all"}
    assert report["max_abs_diff"] < 1e-4
    assert all(r["argmax_match"] for r in report["composition_results"])
    assert all(i["padded_rows"] > 0 for i in report["inputs"])
    assert {(i["batch"], i["seq"]) for i in report["inputs"]} == {(1, 8), (2, 16), (3, 33)}
    assert set(report["shards_exercised"]) == {s.shard_id for s in manifest.shards}
    for name in ("manifest.json", "storage_report.json", "verification_report.json"):
        assert (tiny_export.out_dir / name).is_file()


def test_tiny_bert_with_cut_grid_2(tmp_path):
    result = export_model(tiny_bert(4), "tiny-bert", tmp_path, cut_grid=2, **quiet)
    assert result.manifest.available_ranges() == all_ranges(2)
    assert result.verification_report["passed"]


def test_uneven_atoms(tmp_path):
    result = export_model(tiny_distilbert(6), "tiny6", tmp_path, cut_grid=4, **quiet)
    assert [(a.start_layer, a.end_layer) for a in result.manifest.atom_map] == [(0, 3), (4, 5)]
    assert result.verification_report["passed"]


def test_dropping_the_mask_fails_verification(tmp_path, monkeypatch):
    def ignore_mask(self, hidden_states, attention_mask):
        ones = attention_mask * 0 + 1
        layer_mask = self.adapter.layer_mask(ones, hidden_states.dtype)
        for layer in self.layers:
            hidden_states = self.adapter.run_layer(layer, hidden_states, layer_mask, ones)
        return hidden_states, attention_mask.clone()

    monkeypatch.setattr(EncoderRangeShard, "forward", ignore_mask)
    with pytest.raises(VerificationError) as info:
        export_model(tiny_distilbert(), "nomask", tmp_path, **quiet)
    assert info.value.report["failures"]
    assert not (tmp_path / "nomask" / "manifest.json").exists()
    saved = json.loads((tmp_path / "nomask" / "verification_report.json").read_text())
    assert saved["passed"] is False


def test_contract_without_mask_fails_verification(tiny_export):
    data = copy.deepcopy(tiny_export.manifest.to_dict())
    enc = next(s for s in data["shards"] if s["role"] == "encoder")
    enc["tensors"]["inputs"] = [t for t in enc["tensors"]["inputs"] if t["name"] != "attention_mask"]
    adapter_model = tiny_distilbert()
    from crowdio.export.adapters import get_adapter

    report = verify_export(ModelManifest.from_dict(data), tiny_export.out_dir, get_adapter(adapter_model),
                           make_sample_inputs(200, 0))
    assert not report["passed"]
    assert any("attention_mask" in e for e in report["structural_errors"])


def test_sampled_compositions_cover_every_range(tiny_export):
    from crowdio.export.adapters import get_adapter

    report = verify_export(tiny_export.manifest, tiny_export.out_dir, get_adapter(tiny_distilbert()),
                           make_sample_inputs(200, 0), max_compositions=3, seed=7)
    assert report["passed"]
    assert report["compositions"]["total"] == 8
    assert report["compositions"]["mode"].startswith("sample(seed=7")
    assert set(report["shards_exercised"]) == {s.shard_id for s in tiny_export.manifest.shards}


def test_subset_of_ranges(tmp_path):
    result = export_model(tiny_distilbert(), "subset", tmp_path, ranges=[(0, 1), (2, 3), (0, 3)], **quiet)
    assert result.manifest.complete_encoder_ranges is False
    assert result.verification_report["compositions"]["total"] == 2


def test_external_data_format(tmp_path):
    result = export_model(tiny_distilbert(2), "ext", tmp_path, external_data_threshold=1, **quiet)
    for shard in result.manifest.shards:
        assert [f.path for f in shard.external_data_files] == [f"{shard.file}.data"]
        assert (result.out_dir / shard.external_data_files[0].path).is_file()
    assert result.verification_report["passed"]


def test_refuses_to_overwrite(tmp_path):
    export_model(tiny_distilbert(2), "same", tmp_path, **quiet)
    with pytest.raises(FileExistsError):
        export_model(tiny_distilbert(2), "same", tmp_path, **quiet)
    export_model(tiny_distilbert(2), "same", tmp_path, overwrite=True, **quiet)


def test_budget_stops_before_writing(tmp_path):
    with pytest.raises(StorageBudgetExceeded):
        export_model(tiny_distilbert(), "big", tmp_path, budget_ratio=2.0, **quiet)
    assert not (tmp_path / "big").exists()


def test_cli(tmp_path, capsys):
    from crowdio.export.cli import main

    model_dir = tmp_path / "model"
    tiny_distilbert(2).save_pretrained(model_dir)
    args = ["--model", str(model_dir), "--model-id", "cli", "--out", str(tmp_path / "out")]
    assert main(args + ["--dry-run"]) == 0
    assert "encoder shards : 3" in capsys.readouterr().out
    assert main(args + ["--budget-ratio", "1.5"]) == 3
    assert main(args) == 0
    assert (tmp_path / "out" / "cli" / "manifest.json").is_file()
    assert main(args) == 5

import pytest

from crowdio.export.planner import (
    ModelSizes,
    StorageBudgetExceeded,
    parse_ranges,
    plan_shards,
    storage_report,
)


def sizes(num_layers, embed=100, layer=10, head=1):
    return ModelSizes("m", num_layers, 8, 2, embed, [layer] * num_layers, head)


@pytest.mark.parametrize("num_layers, cut_grid, encoder_shards", [(6, 1, 21), (12, 2, 21), (12, 1, 78), (24, 4, 21)])
def test_encoder_shard_counts(num_layers, cut_grid, encoder_shards):
    report = storage_report(sizes(num_layers), cut_grid, budget_ratio=None)
    assert report["encoder_shards"] == encoder_shards
    assert report["atom_copies"] == report["expected_atom_copies_if_complete"]


def test_storage_totals_follow_the_formula():
    s = sizes(12)
    report = storage_report(s, 2, budget_ratio=None)
    a = 6
    atom_copies = a * (a + 1) * (a + 2) // 6
    assert report["atom_copies"] == atom_copies == 56
    assert report["layer_copies"] == 112
    assert report["encoder_bytes"] == 112 * 10
    assert report["total_bytes"] == 100 + 1 + 112 * 10 + s.one_copy_bytes
    assert report["ratio_to_one_copy"] == round(report["total_bytes"] / s.one_copy_bytes, 3)


def test_budget_exceeded_suggests_a_coarser_grid():
    report = storage_report(sizes(12, embed=10, layer=10, head=1), 1, budget_ratio=10.0)
    assert not report["within_budget"]
    assert "Raise cut_grid" in report["message"]
    assert report["suggestions"][0].startswith("cut_grid=2")


def test_subset_and_full_shard_rules():
    atoms, shards, notes = plan_shards(sizes(6), 1, ranges=[(0, 2), (3, 5)])
    assert [s.role for s in shards] == ["embed", "encoder", "encoder", "head", "full"]
    _, shards, notes = plan_shards(sizes(6), 1, max_shard_bytes=150)
    assert not any(s.role == "full" for s in shards) and "full shard skipped" in notes[0]
    with pytest.raises(ValueError, match="outside atoms"):
        plan_shards(sizes(6), 2, ranges=[(0, 3)])
    assert parse_ranges("0-2, 3,4-5") == [(0, 2), (3, 3), (4, 5)]


def bert_base_adapter():
    import torch
    from transformers import BertConfig, BertForSequenceClassification
    from crowdio.export.adapters import get_adapter

    torch.manual_seed(0)
    return get_adapter(BertForSequenceClassification(BertConfig(num_labels=2)).eval())


def test_bert_base_g2_storage_report(tmp_path):
    """BERT-base geometry (random weights): g=2 gives 21 encoder shards and a correct report."""
    from crowdio.export.adapters import _module_bytes
    from crowdio.export.exporter import export_model

    adapter = bert_base_adapter()
    embed = _module_bytes(adapter.embed_module())
    layer = _module_bytes(adapter.layers()[0])
    head = _module_bytes(adapter.head_module())
    one_copy = embed + 12 * layer + head

    result = export_model(adapter.model, "bert-base", tmp_path, cut_grid=2, dry_run=True, log=lambda *_: None)
    report = result.storage_report
    assert report["encoder_shards"] == 21
    assert report["num_atoms"] == 6
    assert report["atom_copies"] == 56 and report["layer_copies"] == 112
    assert report["one_copy_bytes"] == one_copy
    assert report["total_bytes"] == embed + head + 112 * layer + one_copy
    assert report["within_budget"]
    assert not (tmp_path / "bert-base").exists()

    with pytest.raises(StorageBudgetExceeded, match="cut_grid=2"):
        export_model(adapter.model, "bert-base", tmp_path, cut_grid=1, dry_run=True, log=lambda *_: None)

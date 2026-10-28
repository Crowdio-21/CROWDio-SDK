"""Acceptance export of the real DistilBERT SST-2 model (needs --run-slow and the HF cache)."""

import os

import pytest

from crowdio.catalogue import all_ranges, validate_manifest

MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
TEXTS = [
    "I loved the app, the experience was smooth and fast.",
    "Terrible support, I want my money back.",
    "It works.",
]


@pytest.mark.slow
def test_distilbert_sst2_full_grid(tmp_path):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from crowdio.export.exporter import export_model

    result = export_model(MODEL, "distilbert-sst2", tmp_path, cut_grid=1, sample_texts=TEXTS)
    manifest, report = result.manifest, result.verification_report

    assert len(manifest.shards_by_role("encoder")) == 21
    assert manifest.available_ranges() == all_ranges(6)
    assert [s.role for s in manifest.shards if s.pinned] == ["embed", "head"]
    assert manifest.has_full_shard
    validate_manifest(manifest, result.out_dir, check_files=True)

    assert report["passed"]
    assert report["compositions"] == {"total": 32, "checked": 32, "mode": "all"}
    assert report["max_abs_diff"] < 1e-4
    assert all(r["argmax_match"] for r in report["composition_results"])
    assert report["full_shard"]["argmax_match"]

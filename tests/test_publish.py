import asyncio
import json
import shutil

import pytest

from crowdio.catalogue import ModelManifest
from crowdio.client import CrowdComputeClient
from crowdio.publish import PublishError, VersionConflict, content_digest, main, publish_model

from fake_foreman import FakeForeman

quiet = dict(log=lambda *_: None)


@pytest.fixture
def export_copy(tiny_export, tmp_path):
    target = tmp_path / "tiny-distil"
    shutil.copytree(tiny_export.out_dir, target)
    return target


def serve(fake_kwargs, body):
    """Run ``body(fake)`` (sync) while a fake Foreman is up; return its result and the fake."""
    async def main_():
        async with FakeForeman(**fake_kwargs) as fake:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, body, fake), fake

    return asyncio.run(main_())


def url(fake):
    return f"http://127.0.0.1:{fake.http_port}"


def test_publish_then_discover_and_republish(export_copy):
    def body(fake):
        first = publish_model(export_copy, url(fake), **quiet)
        again = publish_model(export_copy, url(fake), **quiet)
        return first, again

    (first, again), fake = serve({"mode": "new"}, body)
    manifest = ModelManifest.load(export_copy / "manifest.json")
    assert first.status == "published" and first.version == 1
    assert len(first.uploaded) == len(manifest.shards)
    assert again.status == "unchanged" and again.uploaded == []
    stored = fake.catalogue["tiny-distil"][1]
    assert content_digest(stored["manifest"]) == content_digest(manifest)
    assert {s.file for s in manifest.shards} == set(stored["files"])


def test_published_model_is_discoverable(export_copy):
    async def main_():
        async with FakeForeman("new") as fake:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, lambda: publish_model(export_copy, url(fake), **quiet))
            client = CrowdComputeClient()
            await client.connect("127.0.0.1", fake.ws_port, http_port=fake.http_port)
            description = await client.declarative.describe_model("tiny-distil")
            await client.disconnect()
            return description

    description = asyncio.run(main_())
    assert description.version == 1 and description.num_atoms == 4


def test_changed_content_needs_a_version_bump(export_copy):
    def body(fake):
        publish_model(export_copy, url(fake), **quiet)
        data = json.loads((export_copy / "manifest.json").read_text())
        data["source"]["note"] = "re-exported"
        (export_copy / "manifest.json").write_text(json.dumps(data))
        with pytest.raises(VersionConflict, match="--version 2"):
            publish_model(export_copy, url(fake), **quiet)
        return publish_model(export_copy, url(fake), version=2, **quiet)

    bumped, fake = serve({"mode": "new"}, body)
    assert bumped.status == "published" and sorted(fake.catalogue["tiny-distil"]) == [1, 2]
    assert fake.catalogue["tiny-distil"][1]["manifest"]["source"].get("note") is None
    assert json.loads((export_copy / "manifest.json").read_text())["version"] == 1  # export dir untouched


def test_checksum_mismatch_after_upload_is_detected(export_copy):
    def body(fake):
        fake.corrupt_uploads = True
        with pytest.raises(PublishError, match="checksum mismatch"):
            publish_model(export_copy, url(fake), **quiet)

    _, fake = serve({"mode": "new"}, body)
    assert "tiny-distil" not in fake.catalogue


def test_refuses_unverified_or_tampered_exports(export_copy):
    def body(fake):
        shard = next((export_copy / "shards").glob("*.onnx"))
        data = bytearray(shard.read_bytes())
        data[-1] ^= 0xFF
        shard.write_bytes(bytes(data))
        with pytest.raises(PublishError, match="does not match its manifest"):
            publish_model(export_copy, url(fake), **quiet)
        report = export_copy / "verification_report.json"
        report.write_text(json.dumps({"passed": False}))
        with pytest.raises(PublishError, match="did not pass"):
            publish_model(export_copy, url(fake), **quiet)
        (export_copy / "manifest.json").unlink()
        with pytest.raises(PublishError, match="run crowdio-export first"):
            publish_model(export_copy, url(fake), **quiet)

    _, fake = serve({"mode": "new"}, body)
    assert not fake.staging


def test_old_foreman_has_no_catalogue(export_copy):
    def body(fake):
        with pytest.raises(PublishError, match="predates the shard catalogue"):
            publish_model(export_copy, url(fake), **quiet)

    serve({"mode": "old"}, body)


def test_cli(export_copy, capsys):
    def body(fake):
        codes = [main([str(export_copy), "--foreman", url(fake), "--dry-run"]),
                 main([str(export_copy), "--foreman", url(fake)])]
        data = json.loads((export_copy / "manifest.json").read_text())
        data["source"]["note"] = "changed"
        (export_copy / "manifest.json").write_text(json.dumps(data))
        codes.append(main([str(export_copy), "--foreman", url(fake)]))
        return codes

    codes, fake = serve({"mode": "new"}, body)
    assert codes == [0, 0, 3]
    assert "Versions are never overwritten" in capsys.readouterr().err

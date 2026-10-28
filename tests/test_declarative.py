import asyncio
import time

import numpy as np
import pytest

from common.client_protocol import SUBMIT_MODEL_JOB, ClientEnvelope
from crowdio.client import CrowdComputeClient
from crowdio.declarative import (
    InvalidRequest,
    JobCancelled,
    JobFailed,
    ModelNotFound,
    UnsupportedByForeman,
    prepare_inputs,
    validate_job_arguments,
)

from conftest import tiny_distilbert
from fake_foreman import FakeForeman


def run(coro):
    return asyncio.run(coro)


def echo_stage(x):
    return x


@pytest.fixture(scope="module")
def tiny(tiny_export):
    """Catalogue entry, ORT executor for a fixed plan, and the PyTorch reference."""
    from crowdio.export.adapters import get_adapter
    from crowdio.export.verify import _Runner

    manifest = tiny_export.manifest
    runner = _Runner(manifest, tiny_export.out_dir, ["CPUExecutionProvider"])
    plan = [manifest.shards_by_role("embed")[0], manifest.encoder_shard(0, 1),
            manifest.encoder_shard(2, 3), manifest.shards_by_role("head")[0]]

    def executor(model_id, tensors):
        values = dict(tensors)
        for shard in plan:
            values.update(runner.run(shard, values))
        return {"logits": values["logits"]}

    return {"manifests": {"tiny-distil": manifest.to_dict()}, "executor": executor,
            "manifest": manifest, "adapter": get_adapter(tiny_distilbert())}


async def connected(fake, **kwargs):
    client = CrowdComputeClient()
    await client.connect("127.0.0.1", fake.ws_port, http_port=fake.http_port, **kwargs)
    return client


def requests(n=3, seq=12, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        ids = rng.integers(10, 200, size=seq).astype(np.int64)
        mask = np.ones(seq, dtype=np.int64)
        mask[seq - 1 - i * 2:] = 0
        ids[mask == 0] = 0
        out.append({"input_ids": ids, "attention_mask": mask})
    return out


# -- local validation ----------------------------------------------------------

@pytest.mark.parametrize("kwargs, match", [
    (dict(objective="fastest"), "unknown objective"),
    (dict(objective="min_latency_with_deadline"), "deadline_ms is required"),
    (dict(objective="min_latency_with_deadline", deadline_ms=-5), "positive number"),
    (dict(deadline_ms=100), "only applies to"),
    (dict(inputs=[]), "non-empty batch"),
    (dict(model_id=""), "model_id"),
    (dict(options={"dag": {}}), "never carry a DAG"),
    (dict(options={"workers": ["w1"]}), "never carry a DAG"),
    (dict(options={"priority": 1}), "unknown option"),
    (dict(options={"route_mode": "teleport"}), "route_mode"),
    (dict(options={"max_workers": 0}), "max_workers"),
])
def test_argument_validation(kwargs, match):
    args = dict(model_id="m", inputs=[[1, 2]], objective="min_completion_time", deadline_ms=None, options=None)
    args.update(kwargs)
    with pytest.raises(InvalidRequest, match=match):
        validate_job_arguments(**args)


def test_valid_arguments_pass():
    options = validate_job_arguments("m", [[1]], "min_latency_with_deadline", 250, {"route_mode": "direct"})
    assert options == {"route_mode": "direct"}


@pytest.mark.parametrize("item, match", [
    (np.zeros((1, 1, 4), dtype=np.int64), "has shape"),
    ({"input_ids": [1, 2, 3], "attention_mask": [1, 1]}, "attention_mask shape"),
    ({"input_ids": [1, 2], "attention_mask": [1, 2]}, "only 0 and 1"),
    ({"input_ids": [1.5, 2.0]}, "must be integer"),
    ({"input_ids": [1, 999]}, r"token ids must be in \[0, 200\)"),
    ({"input_ids": list(range(1, 70))}, "exceeds 64"),
    ({"input_ids": [1], "token_type_ids": [0]}, "unexpected tensor"),
    ({"attention_mask": [1]}, "missing 'input_ids'"),
    ([], "empty axis"),
])
def test_inputs_checked_against_manifest(tiny, item, match):
    with pytest.raises(InvalidRequest, match=match):
        prepare_inputs([item], tiny["manifest"], "job")


def test_inputs_default_mask_and_ids(tiny):
    from common.serializer import deserialize_tensor

    encoded = prepare_inputs([[5, 6, 7], {"input_ids": [[5, 6]], "request_id": "r2"}], tiny["manifest"], "j")
    assert [e["request_id"] for e in encoded] == ["j:0", "r2"]
    assert deserialize_tensor(encoded[0]["tensors"]["attention_mask"]).tolist() == [[1, 1, 1]]
    with pytest.raises(InvalidRequest, match="duplicate request_id"):
        prepare_inputs([{"input_ids": [1], "request_id": "x"}] * 2, tiny["manifest"], "j")


def test_validation_runs_before_any_network():
    client = CrowdComputeClient()
    with pytest.raises(InvalidRequest):
        run(client.declarative.submit_job("m", [[1]], objective="nope"))


# -- new Foreman -----------------------------------------------------------------

def test_negotiation_discovery_and_submit(tiny):
    async def main():
        async with FakeForeman("new", tiny["manifests"], tiny["executor"]) as fake:
            client = await connected(fake)
            caps = await client.declarative.capabilities()
            assert caps.negotiated and caps.proto_version == 1 and caps.declarative_jobs
            assert caps.features["adaptive_planning"] is True and caps.features["prefetch"] is False

            models = await client.declarative.list_models()
            assert [m["model_id"] for m in models] == ["tiny-distil"]
            desc = await client.declarative.describe_model("tiny-distil")
            assert (desc.num_layers, desc.cut_grid, desc.num_atoms, desc.has_full_shard) == (4, 1, 4, True)
            assert len(desc.available_ranges["fp32"]) == 10
            assert desc.tensor_contracts["encoder"]["inputs"][1]["name"] == "attention_mask"

            batch = requests()
            handle = await client.declarative.submit_job("tiny-distil", batch)
            results = await handle.results(timeout=30)
            status = await handle.status()
            await client.disconnect()
            return fake, handle, results, status, batch

    fake, handle, results, status, batch = run(main())
    submit = next(m for m in fake.received if m.get("type") == SUBMIT_MODEL_JOB)
    assert set(submit["data"]) == {"job_id", "model_id", "model_version", "objective",
                                   "deadline_ms", "options", "inputs"}
    assert submit["data"]["model_version"] == 1

    import torch

    assert [r.request_id for r in results] == handle.request_ids
    for request, result in zip(batch, results):
        reference = tiny["adapter"].reference_logits(
            torch.from_numpy(request["input_ids"][None]), torch.from_numpy(request["attention_mask"][None])
        ).numpy()
        assert np.max(np.abs(result.output - reference)) < 1e-4
        assert result.plan_epoch == 1 and result.route_mode_used == "via_foreman"
        assert result.timings.queued_ms == 1.5 and result.timings.total_ms > 0
        stage = result.timings.stages[0]
        assert (stage.shard_id, stage.worker_id, stage.compute_ms, stage.transfer_ms) == ("embed", "w1", 0.5, 0.25)
        assert stage.extra == {"queue_len": 0}
    [epoch] = handle.plan_epochs
    assert (epoch.epoch, epoch.reason, epoch.announced) == (1, "initial plan", True)
    assert epoch.stages[0]["next_hop"]["worker_id"] == "w2"
    assert handle.state == "completed" and status["state"] == "completed" and status["completed"] == 3


def test_stream_yields_every_result(tiny):
    async def main():
        async with FakeForeman("new", tiny["manifests"], tiny["executor"]) as fake:
            client = await connected(fake)
            handle = await client.declarative.submit_job("tiny-distil", requests(4))
            seen = [r.request_id async for r in handle.stream()]
            await client.disconnect()
            return handle, seen

    handle, seen = run(main())
    assert sorted(seen) == sorted(handle.request_ids)
    assert seen == list(reversed(handle.request_ids))  # completion order


def test_unknown_model_and_rejection(tiny):
    async def main():
        async with FakeForeman("new", tiny["manifests"], tiny["executor"]) as fake:
            client = await connected(fake)
            with pytest.raises(ModelNotFound):
                await client.declarative.describe_model("nope")
            await client.declarative.describe_model("tiny-distil")
            fake.manifests.clear()  # catalogue entry removed after discovery
            from crowdio.declarative import JobRejected

            with pytest.raises(JobRejected, match="MODEL_NOT_FOUND"):
                await client.declarative.submit_job("tiny-distil", requests(1))
            await client.disconnect()

    run(main())


def test_cancel_and_disconnect(tiny):
    async def main():
        async with FakeForeman("new", tiny["manifests"], hold=True) as fake:
            client = await connected(fake)
            handle = await client.declarative.submit_job("tiny-distil", requests(2))
            assert (await handle.status())["state"] == "running"
            assert await handle.cancel() is True
            with pytest.raises(JobCancelled):
                await handle.results(timeout=5)

            other = await client.declarative.submit_job("tiny-distil", requests(1))
            await client.disconnect()
            with pytest.raises(JobFailed, match="disconnected"):
                await other.results(timeout=5)

    run(main())


# -- compatibility -------------------------------------------------------------------

def test_new_client_against_old_foreman(tiny):
    async def main():
        async with FakeForeman("old") as fake:
            started = time.perf_counter()
            client = await connected(fake, hello_timeout=0.3)
            connect_s = time.perf_counter() - started

            legacy = await client.pipeline([{"func": echo_stage, "args_list": [1, 2]}])

            with pytest.raises(UnsupportedByForeman, match="crowdio_pipeline") as info:
                await client.declarative.submit_job("tiny-distil", requests(1))
            assert "did not answer the capability handshake" in str(info.value)
            with pytest.raises(UnsupportedByForeman):
                await client.declarative.list_models()
            caps = await client.declarative.capabilities()

            legacy_after = await client.pipeline([{"func": echo_stage, "args_list": [3]}])
            await client.disconnect()
            return fake, connect_s, legacy, legacy_after, caps

    fake, connect_s, legacy, legacy_after, caps = run(main())
    assert connect_s < 0.3  # the handshake never delays connect()
    assert legacy == [{"echo": 1}, {"echo": 2}] and legacy_after == [{"echo": 3}]
    assert not caps.negotiated
    assert len(fake.parse_errors) == 1  # the hello, logged and ignored like today's Foreman
    assert not any(m.get("type") == SUBMIT_MODEL_JOB for m in fake.received)


def test_legacy_client_against_new_foreman(tiny):
    async def main():
        async with FakeForeman("new", tiny["manifests"], tiny["executor"]) as fake:
            client = await connected(fake, negotiate=False)  # what an old SDK sends
            result = await client.pipeline([{"func": echo_stage, "args_list": [7]}])
            await client.disconnect()
            return fake, result

    fake, result = run(main())
    assert result == [{"echo": 7}]
    assert fake.sent_envelopes_without_hello == 0
    assert all(ClientEnvelope.from_dict(m) is None for m in fake.received)


def test_capability_without_declarative_jobs(tiny):
    async def main():
        caps = {"declarative_jobs": False, "catalogue": True, "features": {}}
        async with FakeForeman("new", tiny["manifests"], capabilities=caps) as fake:
            client = await connected(fake)
            assert len(await client.declarative.list_models()) == 1
            with pytest.raises(UnsupportedByForeman, match="does not advertise"):
                await client.declarative.submit_job("tiny-distil", requests(1))
            await client.disconnect()

    run(main())


def test_unknown_envelope_types_are_ignored():
    client = CrowdComputeClient()
    run(client._handle_message(ClientEnvelope("from_the_future", {"job_id": "x"}).to_json()))
    run(client._handle_message('{"protocol": "crowdio.client", "version": 9}'))


def test_public_api_wiring(tiny):
    import crowdio

    async def main():
        async with FakeForeman("new", tiny["manifests"], tiny["executor"]) as fake:
            await crowdio.crowdio_connect("127.0.0.1", fake.ws_port, http_port=fake.http_port)
            try:
                caps = await crowdio.crowdio_capabilities()
                desc = await crowdio.describe_model("tiny-distil")
                handle = await crowdio.submit_job("tiny-distil", requests(2), "min_latency_with_deadline",
                                                  deadline_ms=500, options={"route_mode": "direct"})
                results = await handle.results(timeout=30)
            finally:
                await crowdio.crowdio_disconnect()
            return fake, caps, desc, results

    fake, caps, desc, results = run(main())
    submit = next(m for m in fake.received if m.get("type") == SUBMIT_MODEL_JOB)["data"]
    assert submit["objective"] == "min_latency_with_deadline" and submit["deadline_ms"] == 500
    assert submit["options"] == {"route_mode": "direct"}
    assert caps.negotiated and desc.model_id == "tiny-distil" and len(results) == 2


# -- results and observability ---------------------------------------------------------

def test_plan_switch_is_recorded(tiny):
    async def main():
        async with FakeForeman("new", tiny["manifests"], tiny["executor"], switch_plan=True) as fake:
            client = await connected(fake)
            handle = await client.declarative.submit_job("tiny-distil", requests(3))
            results = await handle.results(timeout=30)
            await client.disconnect()
            return handle, results

    handle, results = run(main())
    assert [(e.epoch, e.reason) for e in handle.plan_epochs] == [(1, "initial plan"), (2, "SLOW_STAGE on w2")]
    # Reverse completion order: the last request ran first under epoch 1.
    assert [r.plan_epoch for r in results] == [2, 2, 1]
    record = results[0].to_record()
    assert record["plan_epoch"] == 2 and record["outputs"]["logits"]["shape"] == [1, 2]
    import json

    json.dumps([r.to_record() for r in results])


def _handle():
    from crowdio.declarative import JobHandle

    async def make():
        return JobHandle(None, "j", "m", "min_completion_time", ["j:0", "j:1"])

    return asyncio.run(make())


def test_missing_and_malformed_fields_become_none():
    from common.serializer import serialize_tensor

    handle = _handle()
    handle._on_result({"job_id": "j", "request_id": "j:0",
                       "outputs": {"logits": serialize_tensor(np.zeros((1, 2), np.float32))}})
    handle._on_result({"job_id": "j", "request_id": "j:1", "outputs": {}, "plan_epoch": "7",
                       "route_mode_used": "direct", "custom": 1,
                       "timings": {"queued_ms": True, "total_ms": "fast", "stages": ["bad", {"compute_ms": 2}]}})
    first, second = handle._results["j:0"], handle._results["j:1"]
    assert (first.plan_epoch, first.route_mode_used, first.timings.total_ms, first.timings.stages) == (None, None, None, ())
    assert second.plan_epoch is None and second.route_mode_used == "direct" and second.data == {"custom": 1}
    assert second.timings.queued_ms is None and second.timings.total_ms is None
    assert [s.compute_ms for s in second.timings.stages] == [2.0]
    assert handle.plan_epochs == ()


def test_unannounced_epoch_and_read_only_epochs():
    import dataclasses

    handle = _handle()
    handle._on_result({"request_id": "j:0", "outputs": {}, "plan_epoch": 4})
    handle._on_plan({"epoch": 4, "reason": "OOM on w3"})
    handle._on_plan({"reason": "no epoch"})  # ignored
    [epoch] = handle.plan_epochs
    assert (epoch.epoch, epoch.reason, epoch.announced) == (4, "OOM on w3", True)
    with pytest.raises(dataclasses.FrozenInstanceError):
        epoch.reason = "edited"
    handle._on_result({"request_id": "j:1", "outputs": {}, "plan_epoch": 5})
    assert [(e.epoch, e.announced) for e in handle.plan_epochs] == [(4, True), (5, False)]

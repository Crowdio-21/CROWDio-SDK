# CROWDio SDK

CROWDio SDK provides client-side tools for submitting distributed Python workloads to a CROWDio foreman.

## Included Packages

- `crowdio`: public client APIs, decorators, and image utility helpers.
- `common`: shared protocol and serialization utilities used by the SDK runtime.

## Install

```bash
pip install crowdio-sdk
```

Install optional image processing dependencies:

```bash
pip install "crowdio-sdk[image]"
```

## Quick Start

```python
import asyncio
from crowdio import crowdio_connect, crowdio_map, crowdio_disconnect


def square(x):
    return x * x


async def main():
    await crowdio_connect("localhost", 9000)
    results = await crowdio_map(square, [1, 2, 3, 4])
    print(results)
    await crowdio_disconnect()


asyncio.run(main())
```

## Notes

- The package requires a reachable CROWDio foreman endpoint.
- Image utilities under `crowdio.image_utils` require Pillow (install with `[image]` extra).
- Tensor helpers (`common.serializer.serialize_tensor` / `deserialize_tensor`, used for
  `tensor_transport` payloads in DNN pipelines) require numpy (install with `[tensor]` extra).

## Worker SDK mechanisms

The SDK also exposes policy-independent worker mechanisms:

- `ShardStore` maintains a checksum-verified, LRU-bounded on-disk shard cache from
  a model/cut/partition manifest. Writes use a temporary file and atomic rename.
- `SessionManager` enforces single shard residency, legal load/unload transitions,
  and request draining before unload.
- `TransferClient` provides resumable downloads, bounded concurrency, throughput
  samples, and optional bandwidth emulation.
- `TelemetryEvent`, `TelemetryEstimator`, `CostProfile`, and `StructuredLogger`
  share the JSONL event schema used to estimate transfer and initialization costs.

These components do not select a policy; callers decide whether to move between
shards while the SDK guarantees safe execution and consistent measurements.

## Protocol additions (0.4.0)

- `MessageType.SUBMIT_BROADCAST_JOB` and `create_submit_broadcast_job_message` (one task per worker).
- `MessageType.KILL_TASK` / `MessageType.KILL_ACK` with `create_kill_task_message` / `create_kill_ack_message`.
- Tensor transport helpers: `serialize_tensor`, `deserialize_tensor`, `encode_feature_payload`, `decode_feature_payload`.

## Model export (offline)

Install with `pip install "crowdio-sdk[export]"`, then:

```bash
crowdio-export --model distilbert-base-uncased-finetuned-sst-2-english --model-id distilbert-sst2 --cut-grid 1 --out exports
```

The tool prints a storage report and stops if the export exceeds the budget
(`--budget-ratio`, default 10x one model copy). It then exports one ONNX graph
per contiguous atom range plus pinned embed/head shards and a full-model shard,
verifies chained shards against the PyTorch model (max |diff| < 1e-4 and the
same argmax), and only then writes `manifest.json`. Use `--dry-run` for the report
only and `--ranges 0-2,3-5` for a subset. The manifest format is described in
[docs/shard-manifest.md](docs/shard-manifest.md). Supported architectures:
DistilBERT and BERT sequence classifiers.

## Declarative model jobs

With a Foreman that hosts a shard catalogue, submit by model id and objective;
the Foreman chooses the cut and the workers:

```python
from crowdio import crowdio_connect, describe_model, submit_job

await crowdio_connect("localhost", 9000)            # http_port=8000 for the catalogue
info = await describe_model("distilbert-sst2")      # layers, cut grid, ranges, contracts
handle = await submit_job("distilbert-sst2", [{"input_ids": ids, "attention_mask": mask}],
                          objective="min_latency_with_deadline", deadline_ms=500)
results = await handle.results()                    # or: async for r in handle.stream()
```

Requests are validated locally against the model manifest. Against an older
Foreman the call raises `UnsupportedByForeman`; the legacy `crowdio_pipeline`
path is unchanged. Wire format: [docs/client-protocol.md](docs/client-protocol.md).

## Publishing to the catalogue

```bash
crowdio-publish exports/distilbert-sst2 --foreman http://192.168.1.7:8000
```

Uploads shards, tokenizer and manifest, checks each upload's sha256 as reported
by the Foreman, commits the version and reads it back. Identical re-publishes
are no-ops; changed content under an existing version is refused (`--version N`).

Each declarative result carries `plan_epoch`, `route_mode_used` and `timings`
(`queued_ms`, `total_ms`, per-stage `compute_ms` / `transfer_ms`) exactly as the
Foreman reports them; `handle.plan_epochs` lists the epochs seen and the reason
for each switch. `RequestResult.to_record()` gives a JSON row for experiment logs.
# Client <-> Foreman protocol v1 (declarative jobs)

Defined in `common/client_protocol.py`. The legacy `MessageType` messages are
unchanged; declarative jobs use a separate envelope on the same WebSocket:

```json
{"protocol": "crowdio.client", "version": 1, "type": "<type>", "data": {}}
```

## Compatibility rules

- The SDK sends `hello` right after connecting and does **not** wait for it.
  Only the declarative API waits for `hello_ack`, for at most `hello_timeout`
  (2 s by default). Without an answer it reports `UnsupportedByForeman` and
  points to `crowdio_pipeline`. `connect(..., negotiate=False)` sends no hello.
- The current Foreman (checked against `foreman/core/ws_manager.py`) logs
  `'hello' is not a valid MessageType`, sends nothing and keeps the connection;
  later legacy messages still classify the client.
- A new Foreman must handle the envelope before `Message.from_dict`, treat
  `hello` as a client connection in `_determine_client_type`, and send envelope
  messages **only** on connections that sent `hello`.
- Unknown envelope types are ignored by the SDK.

## Messages

| direction | type | data |
|---|---|---|
| C→F | `hello` | `client`, `sdk_version`, `proto_versions: [1]` |
| F→C | `hello_ack` | `proto_version`, `foreman_version`, `capabilities: {declarative_jobs, catalogue, features: {adaptive_planning, direct_path, peer_fetch, prefetch, problem_reports}}` |
| C→F | `submit_model_job` | `job_id`, `model_id`, `model_version`, `objective` (`min_completion_time` \| `min_latency_with_deadline`), `deadline_ms`, `options` (`max_workers`, `allow_recut`, `route_mode`), `inputs: [{request_id, tensors: {input_ids, attention_mask}}]` |
| F→C | `job_accepted` | `job_id` |
| F→C | `job_rejected` | `job_id`, `error`, `code` |
| F→C | `request_result` | `job_id`, `request_id`, `outputs: {logits: <tensor>}`, `plan_epoch`, `route_mode_used`, `timings` (below) |
| F→C | `plan_changed` | `job_id` and either `epoch`, `reason` or `plan: {epoch, stages, created_at, reason}` (the shared Plan) |
| C→F | `job_status_request` | `job_id` |
| F→C | `job_status` | `job_id`, `state`, `completed`, `total` |
| C→F | `cancel_job` | `job_id` |
| F→C | `job_cancelled` | `job_id` |
| F→C | `job_done` | `job_id`, `state` (`completed`) |
| F→C | `job_error` | `job_id`, `error` |

Tensors use `common.serializer.serialize_tensor` (`dtype`, `shape`,
`compression`, `payload_b64`). `submit_model_job` never contains a DAG, cut
point or worker choice; the SDK rejects such options locally.

### Result timings

All fields optional; the SDK records them as given and derives nothing.

```json
"timings": {
  "queued_ms": 1.5,
  "total_ms": 42.0,
  "stages": [{"stage_index": 0, "shard_id": "...", "worker_id": "...",
              "compute_ms": 10.2, "transfer_ms": 3.1}]
}
```

Unknown keys are kept in `extra`; malformed numbers become `None`. An epoch
that appears on a result without a `plan_changed` is listed with
`announced=False` and no reason.

## Catalogue (REST, Foreman HTTP port, default 8000)

- `GET /api/catalogue` returns `{"models": [...]}`. Each entry is a manifest or a
  `summarize_manifest` summary.
- `GET /api/catalogue/{model_id}/manifest[?version=N]` returns the manifest
  (latest version when `version` is omitted), or 404.

### Publishing (`crowdio-publish`)

- `GET /api/catalogue/{model_id}/versions` returns `{"versions": [1, 2]}`, or 404 when there are none.
- `PUT /api/catalogue/{model_id}/versions/{n}/files/{path}`: raw bytes with
  `Content-Length` and `X-Content-SHA256`. Respond `201 {"path", "bytes", "sha256"}`
  computed from the bytes **as stored**; the SDK compares them with its own.
  Respond 409 if version `n` is already committed.
- `PUT /api/catalogue/{model_id}/versions/{n}/manifest`: the manifest JSON. Check
  that every shard and external data file was uploaded with the listed sha256,
  then commit; 400 if not, 409 if `n` is already committed.
- A committed version is immutable. The SDK re-reads it after commit, treats an
  identical re-publish as a no-op, and refuses changed content under an existing
  version (`--version` picks a new one).
- Tokenizer files (`tokenizer/...`) are uploaded with the shards.

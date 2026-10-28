"""In-process Foreman stand-in for client compatibility tests.

``mode="old"`` reproduces the current Foreman's handling of messages it
cannot parse (``foreman/core/ws_manager.py``): ``Message.from_dict`` raises,
the error is logged, nothing is sent, and the connection stays open.
``mode="new"`` speaks the ``crowdio.client`` envelope and only to connections
that sent ``hello``. Both modes answer legacy ``SUBMIT_PIPELINE_JOB`` by
echoing the final stage's arguments as results.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, List, Optional

import websockets

from common.client_protocol import (
    CANCEL_JOB,
    HELLO,
    HELLO_ACK,
    JOB_ACCEPTED,
    JOB_CANCELLED,
    JOB_DONE,
    JOB_REJECTED,
    JOB_STATUS,
    JOB_STATUS_REQUEST,
    PLAN_CHANGED,
    REQUEST_RESULT,
    SUBMIT_MODEL_JOB,
    ClientEnvelope,
)
from common.protocol import Message, MessageType, create_job_results_message
from common.serializer import deserialize_tensor, serialize_tensor

Executor = Callable[[str, Dict[str, Any]], Dict[str, Any]]


class FakeForeman:
    def __init__(self, mode: str = "new", manifests: Optional[Dict[str, dict]] = None,
                 executor: Optional[Executor] = None, hold: bool = False,
                 capabilities: Optional[dict] = None, switch_plan: bool = False):
        self.mode = mode
        self.manifests = dict(manifests or {})
        self.executor = executor
        self.hold = hold
        self.switch_plan = switch_plan
        self.capabilities = capabilities or {
            "declarative_jobs": True, "catalogue": True,
            "features": {"adaptive_planning": True, "direct_path": False},
        }
        self.received: List[dict] = []
        self.parse_errors: List[str] = []
        self.sent_envelopes_without_hello = 0
        self.jobs: Dict[str, dict] = {}
        self.catalogue: Dict[str, Dict[int, dict]] = {}
        self.staging: Dict[tuple, Dict[str, dict]] = {}
        self.corrupt_uploads = False
        self.ws_port = 0
        self.http_port = 0

    # -- lifecycle -------------------------------------------------------------
    async def __aenter__(self):
        self._server = await websockets.serve(self._connection, "127.0.0.1", 0, max_size=None)
        self.ws_port = self._server.sockets[0].getsockname()[1]
        self._http = ThreadingHTTPServer(("127.0.0.1", 0), self._http_handler())
        self.http_port = self._http.server_address[1]
        threading.Thread(target=self._http.serve_forever, daemon=True).start()
        return self

    async def __aexit__(self, *exc):
        self._server.close()
        await self._server.wait_closed()
        self._http.shutdown()

    def _http_handler(self):
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, code, body):
                data = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _parts(self):
                path, _, query = self.path.partition("?")
                parts = [urllib.parse.unquote(p) for p in path.strip("/").split("/")]
                return parts, urllib.parse.parse_qs(query)

            def do_GET(self):
                if fake.mode != "new":
                    return self._send(404, {"detail": "Not Found"})
                parts, query = self._parts()
                if parts == ["api", "catalogue"]:
                    return self._send(200, {"models": list(fake.manifests.values())})
                if len(parts) == 4 and parts[:2] == ["api", "catalogue"] and parts[3] == "versions":
                    versions = sorted(fake.catalogue.get(parts[2], {}))
                    return self._send(200, {"versions": versions}) if versions else self._send(404, {})
                if len(parts) == 4 and parts[:2] == ["api", "catalogue"] and parts[3] == "manifest":
                    if "version" in query:
                        entry = fake.catalogue.get(parts[2], {}).get(int(query["version"][0]))
                        manifest = entry["manifest"] if entry else None
                    else:
                        manifest = fake.manifests.get(parts[2])
                    if manifest is not None:
                        return self._send(200, manifest)
                return self._send(404, {"detail": "Model not found"})

            def do_PUT(self):
                parts, _ = self._parts()
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                if fake.mode != "new" or len(parts) < 6 or parts[:2] != ["api", "catalogue"] or parts[3] != "versions":
                    return self._send(404, {"detail": "Not Found"})
                model_id, version = parts[2], int(parts[4])
                if version in fake.catalogue.get(model_id, {}):
                    return self._send(409, {"detail": f"{model_id} v{version} is already committed"})
                staged = fake.staging.setdefault((model_id, version), {})
                if parts[5] == "files":
                    path = "/".join(parts[6:])
                    stored = bytearray(body)
                    if fake.corrupt_uploads and stored:
                        stored[0] ^= 0xFF
                    staged[path] = {"bytes": len(stored), "sha256": hashlib.sha256(stored).hexdigest()}
                    return self._send(201, dict(staged[path], path=path))
                if parts[5] == "manifest":
                    manifest = json.loads(body)
                    for shard in manifest["shards"]:
                        for item in [{"path": shard["file"], "sha256": shard["sha256"]}] + shard["external_data_files"]:
                            if staged.get(item["path"], {}).get("sha256") != item["sha256"]:
                                return self._send(400, {"detail": f"{item['path']} missing or corrupt"})
                    fake.catalogue.setdefault(model_id, {})[version] = {"manifest": manifest, "files": staged}
                    fake.manifests[model_id] = manifest
                    return self._send(201, {"model_id": model_id, "version": version, "status": "committed"})
                return self._send(404, {"detail": "Not Found"})

        return Handler

    # -- websocket ----------------------------------------------------------------
    async def _connection(self, websocket, *args):
        state = {"hello": False}
        async for text in websocket:
            raw = json.loads(text)
            self.received.append(raw)
            envelope = ClientEnvelope.from_dict(raw)
            if envelope is not None and self.mode == "new":
                await self._envelope(websocket, state, envelope)
                continue
            try:
                message = Message.from_dict(raw)
            except Exception as exc:  # what the current Foreman does: log, keep listening
                self.parse_errors.append(repr(exc))
                continue
            if message.type == MessageType.SUBMIT_PIPELINE_JOB:
                last = message.data["stages"][-1]["args_list"]
                await websocket.send(create_job_results_message(
                    [{"echo": a} for a in last], message.job_id).to_json())

    async def _send(self, websocket, state, kind, data):
        if not state["hello"]:
            self.sent_envelopes_without_hello += 1
        await websocket.send(ClientEnvelope(kind, data).to_json())

    async def _envelope(self, websocket, state, envelope):
        data = dict(envelope.data)
        if envelope.type == HELLO:
            state["hello"] = True
            await self._send(websocket, state, HELLO_ACK, {
                "proto_version": 1, "foreman_version": "fake-1", "capabilities": self.capabilities})
        elif envelope.type == SUBMIT_MODEL_JOB:
            job_id = data["job_id"]
            if data["model_id"] not in self.manifests:
                await self._send(websocket, state, JOB_REJECTED,
                                 {"job_id": job_id, "error": "unknown model", "code": "MODEL_NOT_FOUND"})
                return
            self.jobs[job_id] = {"state": "running", "data": data, "completed": 0}
            await self._send(websocket, state, JOB_ACCEPTED, {"job_id": job_id})
            if not self.hold:
                asyncio.ensure_future(self._run(websocket, state, job_id))
        elif envelope.type == JOB_STATUS_REQUEST:
            job = self.jobs.get(data["job_id"], {"state": "unknown", "completed": 0, "data": {"inputs": []}})
            await self._send(websocket, state, JOB_STATUS, {
                "job_id": data["job_id"], "state": job["state"], "completed": job["completed"],
                "total": len(job["data"]["inputs"])})
        elif envelope.type == CANCEL_JOB:
            self.jobs[data["job_id"]]["state"] = "cancelled"
            await self._send(websocket, state, JOB_CANCELLED, {"job_id": data["job_id"]})

    async def _run(self, websocket, state, job_id):
        job = self.jobs[job_id]
        data = job["data"]
        await self._send(websocket, state, PLAN_CHANGED, {"job_id": job_id, "plan": {
            "epoch": 1, "reason": "initial plan", "created_at": "2026-10-27T00:00:00Z",
            "stages": [{"stage_index": 0, "shard_id": "embed", "worker_id": "w1",
                        "next_hop": {"worker_id": "w2", "allowed_routes": ["via_foreman"]}, "backups": []}],
        }})
        epoch = 1
        # Answer in reverse order so ordering in results() is exercised.
        for position, request in enumerate(reversed(data["inputs"])):
            if position == 1 and self.switch_plan:
                epoch = 2
                await self._send(websocket, state, PLAN_CHANGED,
                                 {"job_id": job_id, "epoch": 2, "reason": "SLOW_STAGE on w2"})
            started = time.perf_counter()
            tensors = {k: deserialize_tensor(v) for k, v in request["tensors"].items()}
            outputs = self.executor(data["model_id"], tensors) if self.executor else {
                "logits": tensors["input_ids"].sum(axis=-1, keepdims=True).astype("float32")}
            job["completed"] += 1
            await self._send(websocket, state, REQUEST_RESULT, {
                "job_id": job_id,
                "request_id": request["request_id"],
                "outputs": {k: serialize_tensor(v) for k, v in outputs.items()},
                "plan_epoch": epoch,
                "route_mode_used": "via_foreman",
                "timings": {
                    "queued_ms": 1.5,
                    "total_ms": (time.perf_counter() - started) * 1000,
                    "stages": [{"stage_index": 0, "shard_id": "embed", "worker_id": "w1",
                                "compute_ms": 0.5, "transfer_ms": 0.25, "queue_len": 0}],
                },
            })
            await asyncio.sleep(0)
        job["state"] = "completed"
        await self._send(websocket, state, JOB_DONE, {"job_id": job_id, "state": "completed"})

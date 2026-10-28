"""Declarative job submission: run a catalogue model by id and objective.

The SDK states *what* to run and *what the developer wants*. It never sends a
DAG, a cut point or worker choices, and it does no planning: the Foreman picks
the plan. Everything here is optional; the legacy ``crowdio_pipeline`` path is
untouched.
"""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Mapping, Optional, Sequence, Tuple

from common.client_protocol import (
    FEATURE_FLAGS,
    HELLO_ACK,
    JOB_ACCEPTED,
    JOB_CANCELLED,
    JOB_DONE,
    JOB_ERROR,
    JOB_REJECTED,
    JOB_STATUS,
    OBJECTIVES,
    PLAN_CHANGED,
    REQUEST_RESULT,
    ROUTE_MODES,
    ClientEnvelope,
    create_cancel_job,
    create_hello,
    create_job_status_request,
    create_submit_model_job,
)

from .catalogue.manifest import ManifestError, ModelManifest, summarize_manifest, validate_manifest

LEGACY_HINT = (
    "Use the legacy fixed-DAG path instead: crowdio_pipeline(stages=[...]) "
    "with one stage per exported partition."
)
OPTION_KEYS = ("max_workers", "allow_recut", "route_mode")
FORBIDDEN_KEYS = ("dag", "stages", "dependency_map", "plan", "cut", "cut_point", "split_idx",
                  "workers", "worker_id", "worker_ids", "shards", "shard_id")


class DeclarativeError(Exception):
    """Base class for declarative-path errors."""


class InvalidRequest(DeclarativeError, ValueError):
    """The request failed local validation; nothing was sent."""


class UnsupportedByForeman(DeclarativeError, RuntimeError):
    """The connected Foreman does not offer this capability."""

    def __init__(self, what: str, capabilities: "Capabilities"):
        self.capabilities = capabilities
        reason = ("it did not answer the capability handshake (older Foreman)"
                  if not capabilities.negotiated else
                  f"it does not advertise it (protocol {capabilities.proto_version})")
        super().__init__(f"The Foreman does not support {what}: {reason}. {LEGACY_HINT}")


class ModelNotFound(DeclarativeError, LookupError):
    pass


class JobRejected(DeclarativeError):
    def __init__(self, job_id: str, error: str, code: Optional[str] = None):
        self.job_id, self.error, self.code = job_id, error, code
        super().__init__(f"Job {job_id} rejected by Foreman: {error}" + (f" [{code}]" if code else ""))


class JobFailed(DeclarativeError):
    pass


class JobCancelled(DeclarativeError):
    pass


@dataclass(frozen=True)
class Capabilities:
    """Result of the hello / hello_ack handshake (read-only)."""

    negotiated: bool
    proto_version: Optional[int] = None
    foreman_version: Optional[str] = None
    declarative_jobs: bool = False
    catalogue: bool = False
    features: Mapping[str, bool] = field(default_factory=lambda: {f: False for f in FEATURE_FLAGS})
    raw: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def legacy(cls) -> "Capabilities":
        return cls(negotiated=False)

    @classmethod
    def from_ack(cls, data: Mapping[str, Any]) -> "Capabilities":
        caps = dict(data.get("capabilities") or {})
        flags = dict(caps.get("features") or {})
        return cls(
            negotiated=True,
            proto_version=data.get("proto_version"),
            foreman_version=data.get("foreman_version"),
            declarative_jobs=bool(caps.get("declarative_jobs", False)),
            catalogue=bool(caps.get("catalogue", False)),
            features={f: bool(flags.get(f, False)) for f in FEATURE_FLAGS},
            raw=dict(data),
        )


@dataclass
class ModelDescription:
    """``describe_model`` result: manifest summary plus the full manifest."""

    model_id: str
    version: int
    num_layers: int
    cut_grid: int
    num_atoms: int
    available_ranges: Dict[str, List[List[int]]]
    has_full_shard: bool
    shard_bytes: Dict[str, int]
    tensor_contracts: Dict[str, Any]
    manifest: ModelManifest

    @classmethod
    def from_manifest(cls, manifest: ModelManifest) -> "ModelDescription":
        summary = summarize_manifest(manifest)
        return cls(
            model_id=summary["model_id"],
            version=summary["version"],
            num_layers=summary["num_layers"],
            cut_grid=summary["cut_grid"],
            num_atoms=summary["num_atoms"],
            available_ranges={k: [list(r) for r in v] for k, v in summary["available_ranges"].items()},
            has_full_shard=summary["has_full_shard"],
            shard_bytes=summary["shard_bytes"],
            tensor_contracts=summary["tensor_contracts"],
            manifest=manifest,
        )


def _ms(value: Any) -> Optional[float]:
    """A Foreman-reported duration, or None when absent or malformed."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _int(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


@dataclass(frozen=True)
class StageTiming:
    """One pipeline stage of one request, as reported by the Foreman."""

    stage_index: Optional[int]
    shard_id: Optional[str]
    worker_id: Optional[str]
    compute_ms: Optional[float]
    transfer_ms: Optional[float]
    extra: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StageTiming":
        known = ("stage_index", "shard_id", "worker_id", "compute_ms", "transfer_ms")
        return cls(
            stage_index=_int(value.get("stage_index")),
            shard_id=None if value.get("shard_id") is None else str(value["shard_id"]),
            worker_id=None if value.get("worker_id") is None else str(value["worker_id"]),
            compute_ms=_ms(value.get("compute_ms")),
            transfer_ms=_ms(value.get("transfer_ms")),
            extra={k: v for k, v in value.items() if k not in known},
        )


@dataclass(frozen=True)
class Timings:
    """Foreman-reported timings; any field may be None. The SDK derives nothing."""

    queued_ms: Optional[float] = None
    total_ms: Optional[float] = None
    stages: Tuple[StageTiming, ...] = ()
    extra: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: Any) -> "Timings":
        if not isinstance(value, Mapping):
            return cls()
        stages = value.get("stages") or ()
        return cls(
            queued_ms=_ms(value.get("queued_ms")),
            total_ms=_ms(value.get("total_ms")),
            stages=tuple(StageTiming.from_dict(s) for s in stages if isinstance(s, Mapping)),
            extra={k: v for k, v in value.items() if k not in ("queued_ms", "total_ms", "stages")},
        )


@dataclass(frozen=True)
class PlanEpoch:
    """A plan epoch seen for a job.

    ``announced`` is True when a ``plan_changed`` message described it; an
    epoch only seen on a result has no reason or stages.
    """

    epoch: int
    reason: Optional[str] = None
    created_at: Optional[str] = None
    stages: Tuple[Mapping[str, Any], ...] = ()
    announced: bool = False


@dataclass(frozen=True)
class RequestResult:
    """One request's output with the plan and timings the Foreman reported."""

    request_id: str
    index: int
    outputs: Mapping[str, Any]
    plan_epoch: Optional[int] = None
    route_mode_used: Optional[str] = None
    timings: Timings = field(default_factory=Timings)
    data: Mapping[str, Any] = field(default_factory=dict)

    @property
    def output(self) -> Any:
        if "logits" in self.outputs:
            return self.outputs["logits"]
        if len(self.outputs) == 1:
            return next(iter(self.outputs.values()))
        return self.outputs

    def to_record(self) -> Dict[str, Any]:
        """JSON-friendly row for experiment logs (tensor shapes, not values)."""
        return {
            "request_id": self.request_id,
            "index": self.index,
            "plan_epoch": self.plan_epoch,
            "route_mode_used": self.route_mode_used,
            "queued_ms": self.timings.queued_ms,
            "total_ms": self.timings.total_ms,
            "stages": [
                {"stage_index": s.stage_index, "shard_id": s.shard_id, "worker_id": s.worker_id,
                 "compute_ms": s.compute_ms, "transfer_ms": s.transfer_ms}
                for s in self.timings.stages
            ],
            "outputs": {
                name: {"shape": list(getattr(v, "shape", ())), "dtype": str(getattr(v, "dtype", ""))}
                for name, v in self.outputs.items()
            },
        }

    @classmethod
    def from_message(cls, data: Mapping[str, Any], index: int) -> "RequestResult":
        from common.serializer import deserialize_tensor

        known = ("job_id", "request_id", "outputs", "plan_epoch", "route_mode_used", "timings")
        route = data.get("route_mode_used")
        return cls(
            request_id=str(data.get("request_id")),
            index=index,
            outputs={name: deserialize_tensor(p) for name, p in (data.get("outputs") or {}).items()},
            plan_epoch=_int(data.get("plan_epoch")),
            route_mode_used=None if route is None else str(route),
            timings=Timings.from_dict(data.get("timings")),
            data={k: v for k, v in data.items() if k not in known},
        )


# ---------------------------------------------------------------------------
# Local validation (no network)
# ---------------------------------------------------------------------------

def validate_job_arguments(
    model_id: Any,
    inputs: Any,
    objective: Any,
    deadline_ms: Any,
    options: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Check everything that does not need the manifest; returns clean options."""
    if not isinstance(model_id, str) or not model_id:
        raise InvalidRequest("model_id must be a non-empty string")
    if objective not in OBJECTIVES:
        raise InvalidRequest(f"unknown objective {objective!r}; expected one of {', '.join(OBJECTIVES)}")
    if objective == "min_latency_with_deadline":
        if deadline_ms is None:
            raise InvalidRequest("deadline_ms is required for objective 'min_latency_with_deadline'")
        if isinstance(deadline_ms, bool) or not isinstance(deadline_ms, (int, float)) or deadline_ms <= 0:
            raise InvalidRequest("deadline_ms must be a positive number of milliseconds")
    elif deadline_ms is not None:
        raise InvalidRequest(f"deadline_ms only applies to 'min_latency_with_deadline', not {objective!r}")
    if isinstance(inputs, (str, bytes)) or not hasattr(inputs, "__len__") or len(inputs) == 0:
        raise InvalidRequest("inputs must be a non-empty batch (list) of requests")

    options = dict(options or {})
    forbidden = sorted(k for k in options if k in FORBIDDEN_KEYS)
    if forbidden:
        raise InvalidRequest(
            f"declarative jobs never carry a DAG, cut point or worker choice (got {', '.join(forbidden)}); "
            f"the Foreman plans. {LEGACY_HINT}"
        )
    unknown = sorted(k for k in options if k not in OPTION_KEYS)
    if unknown:
        raise InvalidRequest(f"unknown option(s) {', '.join(unknown)}; allowed: {', '.join(OPTION_KEYS)}")
    if "max_workers" in options:
        value = options["max_workers"]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise InvalidRequest("options.max_workers must be an integer >= 1")
    if "allow_recut" in options and not isinstance(options["allow_recut"], bool):
        raise InvalidRequest("options.allow_recut must be a bool")
    if "route_mode" in options and options["route_mode"] not in ROUTE_MODES:
        raise InvalidRequest(f"options.route_mode must be one of {', '.join(ROUTE_MODES)}")
    return options


def _np():
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise ImportError("declarative jobs need numpy: pip install 'crowdio-sdk[tensor]'") from exc
    return np


def prepare_inputs(inputs: Sequence[Any], manifest: ModelManifest, job_id: str) -> List[Dict[str, Any]]:
    """Validate requests against the manifest's entry contract and encode them.

    A request is an array of token ids (``[seq]`` or ``[batch, seq]``) or a dict
    with ``input_ids``, optional ``attention_mask`` (defaults to all ones) and
    optional ``request_id``.
    """
    from common.serializer import serialize_tensor

    np = _np()
    entry = next((s for s in manifest.shards if s.role == "embed"), None) or next(
        (s for s in manifest.shards if s.role == "full"), None)
    if entry is None:
        raise InvalidRequest(f"model {manifest.model_id} has no entry shard in its manifest")
    contract = {t.name: t for t in entry.tensors.inputs}
    vocab = manifest.source.get("vocab_size")
    max_len = manifest.source.get("max_position_embeddings")

    encoded, seen_ids = [], set()
    for index, item in enumerate(inputs):
        where = f"inputs[{index}]"
        request = dict(item) if isinstance(item, Mapping) else {"input_ids": item}
        request_id = str(request.pop("request_id", f"{job_id}:{index}"))
        if request_id in seen_ids:
            raise InvalidRequest(f"{where}: duplicate request_id {request_id!r}")
        seen_ids.add(request_id)
        if "input_ids" not in request:
            raise InvalidRequest(f"{where}: missing 'input_ids'")
        unknown = sorted(set(request) - set(contract))
        if unknown:
            raise InvalidRequest(f"{where}: unexpected tensor(s) {', '.join(unknown)}; "
                                 f"model {manifest.model_id} takes {', '.join(contract)}")
        tensors = {}
        for name, spec in contract.items():
            if name not in request:
                if name == "attention_mask":
                    continue
                raise InvalidRequest(f"{where}: missing tensor '{name}'")
            value = np.asarray(request[name])
            if value.ndim == len(spec.shape) - 1:
                value = value[None, ...]
            if value.ndim != len(spec.shape):
                raise InvalidRequest(f"{where}: '{name}' has shape {list(value.shape)}, "
                                     f"manifest expects {spec.shape}")
            for axis, (got, want) in enumerate(zip(value.shape, spec.shape)):
                if isinstance(want, int) and got != want:
                    raise InvalidRequest(f"{where}: '{name}' axis {axis} is {got}, manifest expects {want}")
                if got == 0:
                    raise InvalidRequest(f"{where}: '{name}' has an empty axis {axis}")
            if spec.dtype.startswith("int") and not np.issubdtype(value.dtype, np.integer):
                raise InvalidRequest(f"{where}: '{name}' must be integer, got {value.dtype}")
            tensors[name] = value.astype(np.dtype(spec.dtype), copy=False)
        ids = tensors["input_ids"]
        if "attention_mask" in contract:
            mask = tensors.setdefault("attention_mask", np.ones_like(ids, dtype=np.int64))
            if mask.shape != ids.shape:
                raise InvalidRequest(f"{where}: attention_mask shape {list(mask.shape)} != "
                                     f"input_ids shape {list(ids.shape)}")
            if not np.isin(mask, (0, 1)).all():
                raise InvalidRequest(f"{where}: attention_mask must contain only 0 and 1")
        if vocab is not None and (ids.min() < 0 or ids.max() >= int(vocab)):
            raise InvalidRequest(f"{where}: token ids must be in [0, {vocab})")
        if max_len is not None and ids.shape[-1] > int(max_len):
            raise InvalidRequest(f"{where}: sequence length {ids.shape[-1]} exceeds {max_len}")
        encoded.append({
            "request_id": request_id,
            "tensors": {name: serialize_tensor(value) for name, value in tensors.items()},
        })
    return encoded


# ---------------------------------------------------------------------------
# Job handle
# ---------------------------------------------------------------------------

_DONE = object()


class JobHandle:
    """Handle for a declarative job. The SDK only records what the Foreman reports."""

    def __init__(self, session: "DeclarativeSession", job_id: str, model_id: str,
                 objective: str, request_ids: List[str]):
        self._session = session
        self.job_id = job_id
        self.model_id = model_id
        self.objective = objective
        self.request_ids = list(request_ids)
        self.state = "submitted"
        self.error: Optional[str] = None
        self._results: Dict[str, RequestResult] = {}
        self._epochs: Dict[int, PlanEpoch] = {}
        self._queue: asyncio.Queue = asyncio.Queue()
        self._done: asyncio.Future = asyncio.get_running_loop().create_future()

    @property
    def done(self) -> bool:
        return self._done.done()

    @property
    def plan_epochs(self) -> Tuple[PlanEpoch, ...]:
        """Plan epochs seen for this job, in the order first seen, with the Foreman's reasons."""
        return tuple(self._epochs.values())

    async def status(self, timeout: float = 10.0) -> Dict[str, Any]:
        """Ask the Foreman for this job's status."""
        return await self._session.request_status(self.job_id, timeout)

    async def results(self, timeout: Optional[float] = None) -> List[RequestResult]:
        """Wait for the job and return every result in submission order."""
        await asyncio.wait_for(asyncio.shield(self._done), timeout)
        self._raise_if_unsuccessful()
        missing = [r for r in self.request_ids if r not in self._results]
        if missing:
            raise JobFailed(f"job {self.job_id} finished without results for {len(missing)} request(s), "
                            f"e.g. {missing[:3]}")
        return [self._results[r] for r in self.request_ids]

    async def stream(self) -> AsyncIterator[RequestResult]:
        """Yield results as they arrive (completion order). Single consumer."""
        while not (self._done.done() and self._queue.empty()):
            item = await self._queue.get()
            if item is _DONE:
                break
            yield item
        self._raise_if_unsuccessful()

    async def cancel(self, timeout: float = 10.0) -> bool:
        """Ask the Foreman to cancel; True once it confirms."""
        if self.done:
            return self.state == "cancelled"
        await self._session.send(create_cancel_job(self.job_id))
        try:
            await asyncio.wait_for(asyncio.shield(self._done), timeout)
        except asyncio.TimeoutError:
            return False
        return self.state == "cancelled"

    # -- updates from the session -------------------------------------------
    def _raise_if_unsuccessful(self) -> None:
        if self.state == "cancelled":
            raise JobCancelled(f"job {self.job_id} was cancelled")
        if self.state == "failed":
            raise JobFailed(f"job {self.job_id} failed: {self.error}")

    def _on_result(self, data: Dict[str, Any]) -> None:
        request_id = str(data.get("request_id"))
        index = self.request_ids.index(request_id) if request_id in self.request_ids else -1
        result = RequestResult.from_message(data, index)
        if result.plan_epoch is not None and result.plan_epoch not in self._epochs:
            self._epochs[result.plan_epoch] = PlanEpoch(result.plan_epoch)
        if request_id not in self._results:
            self._results[request_id] = result
            self._queue.put_nowait(result)
        if self.state == "submitted" or self.state == "accepted":
            self.state = "running"

    def _on_plan(self, data: Dict[str, Any]) -> None:
        # Either flat {epoch, reason, ...} or {"plan": {epoch, stages, created_at, reason}}.
        plan = data.get("plan") if isinstance(data.get("plan"), Mapping) else data
        epoch = _int(plan.get("epoch"))
        if epoch is None:
            return
        reason = plan.get("reason", data.get("reason"))
        self._epochs[epoch] = PlanEpoch(
            epoch=epoch,
            reason=None if reason is None else str(reason),
            created_at=None if plan.get("created_at") is None else str(plan["created_at"]),
            stages=tuple(dict(s) for s in plan.get("stages") or () if isinstance(s, Mapping)),
            announced=True,
        )

    def _finish(self, state: str, error: Optional[str] = None) -> None:
        if self._done.done():
            return
        self.state = state
        self.error = error
        self._done.set_result(state)
        self._queue.put_nowait(_DONE)


# ---------------------------------------------------------------------------
# Session: handshake, catalogue, job bookkeeping for one connection
# ---------------------------------------------------------------------------

class DeclarativeSession:
    def __init__(self, client: Any):
        self._client = client
        self._caps_future: Optional[asyncio.Future] = None
        self._caps: Optional[Capabilities] = None
        self._jobs: Dict[str, JobHandle] = {}
        self._accept: Dict[str, asyncio.Future] = {}
        self._status: Dict[str, List[asyncio.Future]] = {}
        self._models: Dict[Any, ModelDescription] = {}
        self.http_base: Optional[str] = None
        self.hello_timeout = 2.0

    # -- transport -----------------------------------------------------------
    async def send(self, envelope: ClientEnvelope) -> None:
        if not self._client.connected or self._client.websocket is None:
            raise RuntimeError("Not connected to foreman. Call connect() first.")
        await self._client.websocket.send(envelope.to_json())

    async def start_negotiation(self) -> None:
        """Send ``hello`` without waiting; the answer is awaited on first use."""
        from . import __version__

        self._caps = None
        self._caps_future = asyncio.get_running_loop().create_future()
        await self.send(create_hello(__version__))

    async def capabilities(self, timeout: Optional[float] = None) -> Capabilities:
        if self._caps is not None:
            return self._caps
        if self._caps_future is None:
            await self.start_negotiation()
        try:
            self._caps = await asyncio.wait_for(asyncio.shield(self._caps_future),
                                                self.hello_timeout if timeout is None else timeout)
        except asyncio.TimeoutError:
            self._caps = Capabilities.legacy()
        return self._caps

    def close(self, reason: str = "connection closed") -> None:
        for handle in self._jobs.values():
            handle._finish("failed", reason)
        for future in list(self._accept.values()) + [f for fs in self._status.values() for f in fs]:
            if not future.done():
                future.set_exception(ConnectionError(reason))
        self._accept.clear()
        self._status.clear()
        self._caps = None
        self._caps_future = None
        self._models.clear()

    # -- incoming --------------------------------------------------------------
    def handle(self, envelope: ClientEnvelope) -> None:
        data = dict(envelope.data)
        job_id = data.get("job_id")
        if envelope.type == HELLO_ACK:
            caps = Capabilities.from_ack(data)
            self._caps = caps
            if self._caps_future is not None and not self._caps_future.done():
                self._caps_future.set_result(caps)
        elif envelope.type in (JOB_ACCEPTED, JOB_REJECTED):
            future = self._accept.pop(job_id, None)
            if future is not None and not future.done():
                future.set_result((envelope.type, data))
        elif envelope.type == JOB_STATUS:
            for future in self._status.pop(job_id, []):
                if not future.done():
                    future.set_result(data)
        elif job_id in self._jobs:
            handle = self._jobs[job_id]
            if envelope.type == REQUEST_RESULT:
                handle._on_result(data)
            elif envelope.type == PLAN_CHANGED:
                handle._on_plan(data)
            elif envelope.type == JOB_DONE:
                handle._finish(str(data.get("state", "completed")), data.get("error"))
            elif envelope.type == JOB_CANCELLED:
                handle._finish("cancelled")
            elif envelope.type == JOB_ERROR:
                handle._finish("failed", str(data.get("error", "unknown error")))

    # -- catalogue (REST) ------------------------------------------------------
    async def _http_json(self, path: str, query: Optional[Dict[str, Any]] = None) -> Any:
        if self.http_base is None:
            raise RuntimeError("Not connected to foreman. Call connect() first.")
        url = self.http_base.rstrip("/") + path
        if query:
            url += "?" + urllib.parse.urlencode({k: v for k, v in query.items() if v is not None})

        def fetch():
            with urllib.request.urlopen(url, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))

        return await asyncio.get_running_loop().run_in_executor(None, fetch)

    async def _require(self, attribute: str, what: str) -> Capabilities:
        caps = await self.capabilities()
        if not getattr(caps, attribute):
            raise UnsupportedByForeman(what, caps)
        return caps

    async def list_models(self) -> List[Dict[str, Any]]:
        await self._require("catalogue", "the shard catalogue")
        body = await self._http_json("/api/catalogue")
        models = body.get("models", []) if isinstance(body, dict) else body
        return [summarize_manifest(ModelManifest.from_dict(m)) if "shards" in m else dict(m) for m in models]

    async def describe_model(self, model_id: str, version: Optional[int] = None) -> ModelDescription:
        key = (model_id, version)
        if key in self._models:
            return self._models[key]
        await self._require("catalogue", "the shard catalogue")
        try:
            body = await self._http_json(f"/api/catalogue/{urllib.parse.quote(model_id, safe='')}/manifest",
                                         {"version": version})
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise ModelNotFound(f"model {model_id!r}"
                                    + (f" version {version}" if version else "")
                                    + " is not in the Foreman catalogue") from None
            raise
        manifest = ModelManifest.from_dict(body)
        try:
            validate_manifest(manifest)
        except ManifestError as exc:
            raise DeclarativeError(f"Foreman returned an invalid manifest for {model_id}: {exc}") from None
        description = ModelDescription.from_manifest(manifest)
        self._models[key] = description
        return description

    # -- jobs ------------------------------------------------------------------
    async def submit_job(
        self,
        model_id: str,
        inputs: Sequence[Any],
        objective: str = "min_completion_time",
        deadline_ms: Optional[float] = None,
        options: Optional[Mapping[str, Any]] = None,
        model_version: Optional[int] = None,
        accept_timeout: float = 30.0,
    ) -> JobHandle:
        options = validate_job_arguments(model_id, inputs, objective, deadline_ms, options)
        await self._require("declarative_jobs", "declarative jobs")
        description = await self.describe_model(model_id, model_version)
        job_id = str(uuid.uuid4())
        encoded = prepare_inputs(inputs, description.manifest, job_id)

        handle = JobHandle(self, job_id, model_id, objective, [r["request_id"] for r in encoded])
        accepted = asyncio.get_running_loop().create_future()
        self._accept[job_id] = accepted
        self._jobs[job_id] = handle
        try:
            await self.send(create_submit_model_job(
                job_id, model_id, encoded, objective, deadline_ms, options,
                model_version if model_version is not None else description.version,
            ))
            kind, data = await asyncio.wait_for(accepted, accept_timeout)
        except BaseException:
            self._jobs.pop(job_id, None)
            self._accept.pop(job_id, None)
            raise
        if kind == JOB_REJECTED:
            self._jobs.pop(job_id, None)
            raise JobRejected(job_id, str(data.get("error", "rejected")), data.get("code"))
        handle.state = "accepted"
        return handle

    async def request_status(self, job_id: str, timeout: float) -> Dict[str, Any]:
        future = asyncio.get_running_loop().create_future()
        self._status.setdefault(job_id, []).append(future)
        await self.send(create_job_status_request(job_id))
        try:
            return await asyncio.wait_for(future, timeout)
        finally:
            waiting = self._status.get(job_id, [])
            if future in waiting:
                waiting.remove(future)

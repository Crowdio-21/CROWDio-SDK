"""Versioned client <-> Foreman messages for declarative (model-id) jobs.

These use their own envelope instead of extending ``MessageType``, following
the reconfiguration protocol on the Foreman::

    {"protocol": "crowdio.client", "version": 1, "type": "...", "data": {...}}

Compatibility:

* An old Foreman fails to parse the envelope, logs it and sends nothing, so
  the SDK waits a bounded time for ``hello_ack`` and then reports that
  declarative jobs are unsupported. Legacy messages on the same connection
  keep working.
* A new Foreman must send envelope messages only on connections that sent
  ``hello``, so old clients never see them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

PROTOCOL = "crowdio.client"
PROTOCOL_VERSION = 1

# Client -> Foreman
HELLO = "hello"
SUBMIT_MODEL_JOB = "submit_model_job"
JOB_STATUS_REQUEST = "job_status_request"
CANCEL_JOB = "cancel_job"

# Foreman -> Client
HELLO_ACK = "hello_ack"
JOB_ACCEPTED = "job_accepted"
JOB_REJECTED = "job_rejected"
REQUEST_RESULT = "request_result"
PLAN_CHANGED = "plan_changed"
JOB_STATUS = "job_status"
JOB_DONE = "job_done"
JOB_CANCELLED = "job_cancelled"
JOB_ERROR = "job_error"

CLIENT_TYPES = {HELLO, SUBMIT_MODEL_JOB, JOB_STATUS_REQUEST, CANCEL_JOB}
FOREMAN_TYPES = {
    HELLO_ACK, JOB_ACCEPTED, JOB_REJECTED, REQUEST_RESULT, PLAN_CHANGED,
    JOB_STATUS, JOB_DONE, JOB_CANCELLED, JOB_ERROR,
}

FEATURE_FLAGS = ("adaptive_planning", "direct_path", "peer_fetch", "prefetch", "problem_reports")
OBJECTIVES = ("min_completion_time", "min_latency_with_deadline")
ROUTE_MODES = ("via_foreman", "direct", "opportunistic")


@dataclass(frozen=True)
class ClientEnvelope:
    type: str
    data: Mapping[str, Any] = field(default_factory=dict)
    version: int = PROTOCOL_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {"protocol": PROTOCOL, "version": self.version, "type": self.type, "data": dict(self.data)}

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Optional["ClientEnvelope"]:
        """Return ``None`` for anything that is not a client envelope."""
        if not isinstance(value, Mapping) or value.get("protocol") != PROTOCOL:
            return None
        return cls(str(value.get("type")), dict(value.get("data") or {}), int(value.get("version", 0)))


def create_hello(sdk_version: str) -> ClientEnvelope:
    return ClientEnvelope(HELLO, {"client": "crowdio-sdk", "sdk_version": sdk_version,
                                  "proto_versions": [PROTOCOL_VERSION]})


def create_submit_model_job(
    job_id: str,
    model_id: str,
    inputs: list,
    objective: str,
    deadline_ms: Optional[float] = None,
    options: Optional[Mapping[str, Any]] = None,
    model_version: Optional[int] = None,
) -> ClientEnvelope:
    return ClientEnvelope(SUBMIT_MODEL_JOB, {
        "job_id": job_id,
        "model_id": model_id,
        "model_version": model_version,
        "objective": objective,
        "deadline_ms": deadline_ms,
        "options": dict(options or {}),
        "inputs": inputs,
    })


def create_job_status_request(job_id: str) -> ClientEnvelope:
    return ClientEnvelope(JOB_STATUS_REQUEST, {"job_id": job_id})


def create_cancel_job(job_id: str) -> ClientEnvelope:
    return ClientEnvelope(CANCEL_JOB, {"job_id": job_id})

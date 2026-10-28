"""Shared event schema and lightweight online telemetry estimators."""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class TelemetryEvent:
    ts_mono: float
    ts_wall: str
    worker: str
    event: str
    phase: str = ""
    bytes: int = 0
    dur_ms: Optional[float] = None
    shard: str = ""
    cut: str = ""
    epoch: Optional[int] = None
    request_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "TelemetryEvent":
        fields = {name: value.get(name) for name in cls.__dataclass_fields__}
        fields["ts_mono"] = float(fields["ts_mono"])
        fields["bytes"] = int(fields["bytes"] or 0)
        return cls(**fields)


class TelemetryEstimator:
    def __init__(self, decay: float = 0.2) -> None:
        if not 0 < decay <= 1:
            raise ValueError("decay must be in (0, 1]")
        self.decay = decay
        self._throughput = 0.0
        self._latency = 0.0

    def observe(self, event: Any) -> None:
        item = event if isinstance(event, TelemetryEvent) else TelemetryEvent.from_dict(event)
        if item.dur_ms is not None and item.dur_ms >= 0:
            self._latency = self._update(self._latency, item.dur_ms)
            if item.bytes > 0 and item.dur_ms > 0:
                self._throughput = self._update(
                    self._throughput, item.bytes / (item.dur_ms / 1000.0)
                )

    def throughput_bytes_per_second(self) -> float:
        return self._throughput

    def latency_ms(self) -> float:
        return self._latency

    def snapshot(self) -> Dict[str, float]:
        return {"throughput_bytes_per_second": self._throughput, "latency_ms": self._latency}

    def _update(self, current: float, value: float) -> float:
        return value if current == 0 else (1 - self.decay) * current + self.decay * value

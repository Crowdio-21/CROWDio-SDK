"""Online initialization and bandwidth cost profiles."""

import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Distribution:
    samples: List[float] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.samples)

    @property
    def median(self) -> float:
        return statistics.median(self.samples) if self.samples else 0.0

    @property
    def p90(self) -> float:
        if not self.samples:
            return 0.0
        values = sorted(self.samples)
        return values[min(len(values) - 1, int(0.9 * (len(values) - 1)))]


class CostProfile:
    """Tracks T_init(device_class, shard, EP) and bandwidth samples."""

    def __init__(self) -> None:
        self._init: Dict[str, Distribution] = {}
        self._bandwidth: List[float] = []

    def record_load_complete(self, device_class: str, shard: str,
                             ep: int, duration_ms: float) -> None:
        if duration_ms < 0 or ep < 1:
            raise ValueError("duration_ms must be non-negative and EP positive")
        self._init.setdefault(self._key(device_class, shard, ep), Distribution()).samples.append(
            duration_ms
        )

    def record_bandwidth(self, bytes_per_second: float) -> None:
        if bytes_per_second <= 0:
            raise ValueError("bandwidth must be positive")
        self._bandwidth.append(bytes_per_second)

    def observe(self, event: Any) -> None:
        """Update the profile from a ``LOAD_COMPLETE`` event record."""
        value = event.to_dict() if hasattr(event, "to_dict") else event
        if value.get("event") != "LOAD_COMPLETE":
            return
        device_class = value.get("device_class")
        shard = value.get("shard")
        ep = value.get("ep")
        duration = value.get("dur_ms")
        if device_class is None or shard is None or ep is None or duration is None:
            raise ValueError("LOAD_COMPLETE requires device_class, shard, ep, and dur_ms")
        self.record_load_complete(str(device_class), str(shard), int(ep), float(duration))

    def t_init(self, device_class: str, shard: str, ep: int) -> Distribution:
        return self._init.get(self._key(device_class, shard, ep), Distribution())

    def bw_lower_bound(self, k: int = 1) -> float:
        if k < 1:
            raise ValueError("k must be positive")
        if not self._bandwidth:
            return 0.0
        values = sorted(self._bandwidth)
        lower = values[max(0, int(0.1 * (len(values) - 1)))]
        return lower / k

    @staticmethod
    def _key(device_class: str, shard: str, ep: int) -> str:
        return "%s|%s|%d" % (device_class, shard, ep)

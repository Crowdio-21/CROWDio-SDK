"""Buffered JSONL logging for SDK lifecycle and transfer events."""

import json
import queue
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


class StructuredLogger:
    """Write structured events on a background thread.

    ``log`` only enqueues an event, so filesystem work is not performed on the
    transfer or session critical path.
    """

    def __init__(self, path: Optional[str] = None, worker: str = "") -> None:
        self.path = Path(path) if path else None
        self.worker = worker
        self._queue = queue.Queue()
        self._stop = object()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def log(self, event: str, phase: str = "", bytes: int = 0,
            dur_ms: Optional[float] = None, shard: str = "", cut: str = "",
            epoch: Optional[int] = None, request_id: str = "",
            **fields: Any) -> None:
        record: Dict[str, Any] = {
            "ts_mono": time.monotonic(),
            "ts_wall": datetime.now(timezone.utc).isoformat(),
            "worker": self.worker,
            "event": event,
            "phase": phase,
            "bytes": bytes,
            "dur_ms": dur_ms,
            "shard": shard,
            "cut": cut,
            "epoch": epoch,
            "request_id": request_id,
        }
        record.update(fields)
        self._queue.put(record)

    def flush(self, timeout: float = 5.0) -> None:
        """Wait until currently queued records have been written."""
        done = threading.Event()
        self._queue.put(done)
        done.wait(timeout)

    def close(self) -> None:
        self.flush()
        self._queue.put(self._stop)
        self._thread.join(timeout=5)

    def _run(self) -> None:
        handle = None
        try:
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                handle = self.path.open("a", encoding="utf-8")
            while True:
                record = self._queue.get()
                try:
                    if record is self._stop:
                        return
                    if isinstance(record, threading.Event):
                        record.set()
                        continue
                    if handle:
                        handle.write(json.dumps(record, separators=(",", ":")) + "\n")
                        handle.flush()
                finally:
                    self._queue.task_done()
        finally:
            if handle:
                handle.close()

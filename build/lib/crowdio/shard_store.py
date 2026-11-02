"""Crash-safe on-disk shard cache and manifest index."""

import hashlib
import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple


ShardKey = Tuple[str, str, str]


@dataclass
class ShardManifest:
    model: str
    cut: str
    partition: str
    size: int
    checksum: str
    input_shape: Tuple[int, ...] = ()
    output_shape: Tuple[int, ...] = ()
    flops: Dict[str, float] = field(default_factory=dict)
    latency_ms: Dict[str, float] = field(default_factory=dict)

    @property
    def key(self) -> ShardKey:
        return self.model, self.cut, self.partition

    @staticmethod
    def _tensor_bytes(shape: Tuple[int, ...], dtype_size: int = 4) -> int:
        result = dtype_size
        for dimension in shape:
            result *= int(dimension)
        return result if shape else 0

    @property
    def input_bytes(self) -> int:
        return self._tensor_bytes(self.input_shape)

    @property
    def output_bytes(self) -> int:
        return self._tensor_bytes(self.output_shape)

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "ShardManifest":
        return cls(
            model=str(value["model"]), cut=str(value["cut"]),
            partition=str(value.get("partition", "0")),
            size=int(value["size"]), checksum=str(value["checksum"]).lower(),
            input_shape=tuple(value.get("input_shape", ())),
            output_shape=tuple(value.get("output_shape", ())),
            flops=dict(value.get("flops", {})),
            latency_ms=dict(value.get("latency_ms", {})),
        )


class ShardStore:
    """Store shards under *root* with an atomically replaced JSON index."""

    def __init__(self, root: str, max_bytes: int,
                 manifests: Optional[Iterable[Any]] = None) -> None:
        if max_bytes < 0:
            raise ValueError("max_bytes must be non-negative")
        self.root = Path(root)
        self.max_bytes = max_bytes
        self._index_path = self.root / "index.json"
        self._shard_dir = self.root / "shards"
        self._lock = threading.RLock()
        self._entries: Dict[str, Dict[str, Any]] = {}
        self._used_bytes = 0
        self.root.mkdir(parents=True, exist_ok=True)
        self._shard_dir.mkdir(exist_ok=True)
        self._load_index()
        if manifests:
            self.register_manifest(manifests)

    @staticmethod
    def _id(key: ShardKey) -> str:
        return json.dumps(list(key), separators=(",", ":"))

    def register_manifest(self, manifests: Iterable[Any]) -> None:
        if isinstance(manifests, dict):
            manifests = manifests.get("shards", [])
        with self._lock:
            for item in manifests:
                manifest = item if isinstance(item, ShardManifest) else ShardManifest.from_dict(item)
                entry = self._entries.setdefault(self._id(manifest.key), {})
                entry["manifest"] = asdict(manifest)
                entry.setdefault("last_used", 0.0)
            self._persist()

    def manifest(self, key: ShardKey) -> Optional[ShardManifest]:
        entry = self._entries.get(self._id(key))
        return ShardManifest.from_dict(entry["manifest"]) if entry and entry.get("manifest") else None

    def path_for(self, key: ShardKey) -> Path:
        return self._shard_dir / (hashlib.sha256(self._id(key).encode()).hexdigest() + ".bin")

    def has(self, key: ShardKey) -> bool:
        return self.path_for(key).is_file() and self._id(key) in self._entries

    def get(self, key: ShardKey) -> Optional[bytes]:
        with self._lock:
            if not self.has(key):
                return None
            path = self.path_for(key)
            data = path.read_bytes()
            entry = self._entries[self._id(key)]
            expected = entry.get("manifest", {}).get("checksum", "")
            if expected and hashlib.sha256(data).hexdigest() != expected:
                self._remove(key)
                raise IOError("cached shard checksum verification failed")
            entry["last_used"] = time.time()
            self._persist()
            return data

    def put(self, key: ShardKey, data: bytes) -> Path:
        with self._lock:
            manifest = self.manifest(key)
            if manifest is None:
                raise KeyError("shard is not present in the registered manifest")
            if len(data) != manifest.size:
                raise ValueError("shard size does not match manifest")
            if hashlib.sha256(data).hexdigest() != manifest.checksum.lower():
                raise ValueError("shard checksum does not match manifest")
            path = self.path_for(key)
            temp = path.with_suffix(".tmp")
            temp.write_bytes(data)
            os.replace(str(temp), str(path))
            entry = self._entries[self._id(key)]
            old_size = int(entry.get("stored_size", 0))
            self._used_bytes += len(data) - old_size
            entry.update({"stored_size": len(data), "last_used": time.time()})
            self._evict()
            self._persist()
            return path

    def _evict(self) -> None:
        candidates = sorted(
            ((float(v.get("last_used", 0)), k) for k, v in self._entries.items()
             if int(v.get("stored_size", 0)) > 0),
            key=lambda item: item[0],
        )
        for _, key_id in candidates:
            if self._used_bytes <= self.max_bytes:
                break
            self._remove(tuple(json.loads(key_id)))

    def _remove(self, key: ShardKey) -> None:
        key_id = self._id(key)
        entry = self._entries.get(key_id, {})
        self._used_bytes -= int(entry.get("stored_size", 0))
        try:
            self.path_for(key).unlink()
        except FileNotFoundError:
            pass
        entry["stored_size"] = 0
        entry["last_used"] = 0.0

    def _load_index(self) -> None:
        if self._index_path.exists():
            try:
                payload = json.loads(self._index_path.read_text(encoding="utf-8"))
                self._entries = dict(payload.get("entries", {}))
            except (OSError, ValueError, TypeError):
                self._entries = {}
        for key_id, entry in list(self._entries.items()):
            key = tuple(json.loads(key_id))
            path = self.path_for(key)
            if not path.is_file() or int(entry.get("stored_size", 0)) != path.stat().st_size:
                entry["stored_size"] = 0
                continue
            checksum = entry.get("manifest", {}).get("checksum")
            if checksum and hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
                path.unlink()
                entry["stored_size"] = 0
            else:
                self._used_bytes += path.stat().st_size
        self._persist()

    def _persist(self) -> None:
        payload = {"version": 1, "entries": self._entries}
        temp = self._index_path.with_suffix(".tmp")
        temp.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        os.replace(str(temp), str(self._index_path))

    @property
    def used_bytes(self) -> int:
        return self._used_bytes

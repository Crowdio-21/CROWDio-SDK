"""Resumable shard transfer with bounded concurrency and bandwidth emulation."""

import asyncio
import hashlib
import os
import time
import urllib.request
from urllib.parse import unquote, urlsplit
from pathlib import Path
from typing import Callable, List, Optional, Union


class TransferClient:
    def __init__(self, max_concurrency: int = 4,
                 bandwidth_bytes_per_second: Optional[float] = None) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be positive")
        if bandwidth_bytes_per_second is not None and bandwidth_bytes_per_second <= 0:
            raise ValueError("bandwidth_bytes_per_second must be positive")
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self.bandwidth_bytes_per_second = bandwidth_bytes_per_second
        self._samples: List[float] = []

    async def download(self, source: str, destination: Union[str, Path],
                       expected_size: Optional[int] = None,
                       checksum: Optional[str] = None,
                       progress: Optional[Callable[[int, float], None]] = None) -> Path:
        async with self._semaphore:
            return await asyncio.to_thread(
                self._download_sync, source, Path(destination), expected_size, checksum, progress
            )

    def _download_sync(self, source: str, destination: Path, expected_size: Optional[int],
                       checksum: Optional[str],
                       progress: Optional[Callable[[int, float], None]]) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp = destination.with_suffix(destination.suffix + ".part")
        offset = temp.stat().st_size if temp.exists() else 0
        started = time.monotonic()
        request = urllib.request.Request(source, headers={"Range": "bytes=%d-" % offset})
        if source.startswith("file://"):
            response = open(Path(unquote(urlsplit(source).path.lstrip("/"))), "rb")
            response.seek(offset)
        elif "://" not in source:
            response = open(source, "rb")
            response.seek(offset)
        else:
            response = urllib.request.urlopen(request, timeout=60)
            if offset and getattr(response, "status", 206) != 206:
                response.close()
                offset = 0
                response = urllib.request.urlopen(urllib.request.Request(source), timeout=60)
        try:
            mode = "ab" if offset else "wb"
            with open(temp, mode) as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
                    if self.bandwidth_bytes_per_second:
                        time.sleep(len(chunk) / self.bandwidth_bytes_per_second)
                    if progress:
                        progress(len(chunk), time.monotonic() - started)
        finally:
            response.close()
        size = temp.stat().st_size
        if expected_size is not None and size != expected_size:
            raise IOError("download size does not match expected size")
        if checksum:
            digest = hashlib.sha256(temp.read_bytes()).hexdigest()
            if digest.lower() != checksum.lower():
                raise IOError("download checksum does not match expected checksum")
        os.replace(str(temp), str(destination))
        elapsed = time.monotonic() - started
        if elapsed > 0:
            self._samples.append(size / elapsed)
        return destination

    @property
    def throughput_samples(self) -> List[float]:
        return list(self._samples)

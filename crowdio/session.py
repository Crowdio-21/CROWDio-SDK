"""Single-residency model session state machine."""

import asyncio
import inspect
import time
from enum import Enum
from typing import Any, Callable, Optional

from .structured_logging import StructuredLogger


class SessionState(str, Enum):
    IDLE = "IDLE"
    PREINIT = "PREINIT"
    LOADING = "LOADING"
    READY = "READY"
    DRAINING = "DRAINING"
    UNLOADING = "UNLOADING"


class SessionTransitionError(RuntimeError):
    pass


class SessionManager:
    """Owns legal transitions and drains requests before unloading."""

    def __init__(self, loader: Callable[..., Any], unloader: Callable[..., Any],
                 preinitializer: Optional[Callable[..., Any]] = None,
                 logger: Optional[StructuredLogger] = None) -> None:
        self.loader = loader
        self.unloader = unloader
        self.preinitializer = preinitializer
        self.logger = logger
        self.state = SessionState.IDLE
        self.loaded_shard: Optional[str] = None
        self._active = 0
        self._condition = asyncio.Condition()

    async def preinit(self, shard: str, *args: Any, **kwargs: Any) -> None:
        await self._transition(SessionState.PREINIT, {SessionState.IDLE})
        try:
            if self.preinitializer:
                await self._call(self.preinitializer, shard, *args, **kwargs)
        except Exception:
            await self._set_state(SessionState.IDLE)
            raise
        await self._set_state(SessionState.IDLE)
        self._emit("PREINIT_COMPLETE", shard)

    async def load(self, shard: str, *args: Any, **kwargs: Any) -> None:
        async with self._condition:
            if self.state == SessionState.READY and self.loaded_shard == shard:
                return
            if self.state != SessionState.IDLE:
                raise SessionTransitionError(
                    "load requires IDLE; current state is %s" % self.state.value
                )
            self.state = SessionState.LOADING
        self._emit("LOAD_START", shard)
        started = time.monotonic()
        try:
            await self._call(self.loader, shard, *args, **kwargs)
        except Exception:
            await self._set_state(SessionState.IDLE)
            raise
        async with self._condition:
            self.loaded_shard = shard
            self.state = SessionState.READY
            self._condition.notify_all()
        self._emit("LOAD_COMPLETE", shard, (time.monotonic() - started) * 1000)

    async def unload(self, *args: Any, **kwargs: Any) -> None:
        async with self._condition:
            if self.state != SessionState.READY:
                raise SessionTransitionError(
                    "unload requires READY; current state is %s" % self.state.value
                )
            shard = self.loaded_shard
            self.state = SessionState.DRAINING
            while self._active:
                await self._condition.wait()
            self.state = SessionState.UNLOADING
        self._emit("UNLOAD_START", shard or "")
        try:
            await self._call(self.unloader, shard, *args, **kwargs)
        except Exception:
            await self._set_state(SessionState.READY)
            raise
        async with self._condition:
            self.loaded_shard = None
            self.state = SessionState.IDLE
            self._condition.notify_all()
        self._emit("UNLOAD_COMPLETE", shard or "")

    async def acquire(self) -> None:
        async with self._condition:
            if self.state != SessionState.READY:
                raise SessionTransitionError("requests require READY state")
            self._active += 1

    async def release(self) -> None:
        async with self._condition:
            if self._active < 1:
                raise SessionTransitionError("release without a matching acquire")
            self._active -= 1
            if not self._active:
                self._condition.notify_all()

    async def _transition(self, target: SessionState, allowed: Any) -> None:
        async with self._condition:
            if self.state not in allowed:
                raise SessionTransitionError(
                    "%s requires %s; current state is %s" %
                    (target.value, ", ".join(s.value for s in allowed), self.state.value)
                )
            self.state = target
            self._condition.notify_all()
        self._emit(target.value, self.loaded_shard or "")

    async def _set_state(self, state: SessionState) -> None:
        async with self._condition:
            self.state = state
            self._condition.notify_all()

    async def _call(self, callback: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        value = callback(*args, **kwargs)
        return await value if inspect.isawaitable(value) else value

    def _emit(self, event: str, shard: str, dur_ms: Optional[float] = None) -> None:
        if self.logger:
            self.logger.log(event, phase="session", shard=shard, dur_ms=dur_ms)

    @property
    def active_requests(self) -> int:
        return self._active

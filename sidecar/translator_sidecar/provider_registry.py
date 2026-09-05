"""Event-loop-owned provider selection, session leases, and retirement."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from .cleanup import finish_cleanup
from .provider_contract import (
    CancelUtterance,
    CloseProviderSession,
    OpenProviderSession,
    ProviderHealth,
    ProviderId,
    ProviderInputFrame,
    ProviderProtocolError,
    ProviderSessionOpened,
    UpdateDebugText,
)


class RuntimeProvider(Protocol):
    async def open_session(
        self,
        request: OpenProviderSession,
        publish,
    ) -> tuple[ProviderSessionOpened, ProviderHealth]: ...
    async def submit_frame(self, frame: ProviderInputFrame) -> None: ...
    async def cancel_utterance(self, request: CancelUtterance) -> None: ...
    async def update_debug_text(self, request: UpdateDebugText) -> None: ...
    async def close_session(self, request: CloseProviderSession) -> None: ...
    async def wait_publications(self, session_id: UUID) -> None: ...
    async def shutdown(self) -> None: ...


@dataclass(eq=False)
class _Entry:
    provider: RuntimeProvider
    leases: int = 0
    disposal: asyncio.Task[None] | None = None


@dataclass(eq=False)
class ProviderLease:
    _registry: ProviderRegistry
    _entry: _Entry
    _released: bool = False

    @property
    def provider(self) -> RuntimeProvider:
        return self._entry.provider

    async def release(self) -> None:
        if not self._released:
            self._released = True
            self._entry.leases -= 1
        await self._registry._retire(self._entry)


class ProviderRegistry:
    """Selection and lease changes contain no awaits and are atomic on one loop."""

    def __init__(self, providers: dict[ProviderId, RuntimeProvider]) -> None:
        self._entries = {
            id(provider): _Entry(provider) for provider in providers.values()
        }
        self._selected = {
            key: self._entries[id(provider)] for key, provider in providers.items()
        }
        self._closed = False

    def get(self, provider_id: ProviderId) -> RuntimeProvider | None:
        entry = self._selected.get(provider_id)
        return entry.provider if entry is not None else None

    def acquire(self, provider_id: ProviderId) -> ProviderLease:
        entry = self._selected.get(provider_id)
        if self._closed or entry is None:
            raise ProviderProtocolError("provider is unavailable")
        entry.leases += 1
        return ProviderLease(self, entry)

    def replace(self, provider_id: ProviderId, provider: RuntimeProvider) -> None:
        if self._closed:
            raise ProviderProtocolError("provider registry is closed")
        entry = self._entries.get(id(provider))
        if entry is not None and entry.disposal is not None:
            raise ProviderProtocolError("provider is retiring")
        if entry is None:
            entry = self._entries[id(provider)] = _Entry(provider)
        self._selected[provider_id] = entry

    async def collect(self) -> None:
        await finish_cleanup(self._collect())

    async def _collect(self) -> None:
        outcomes = await asyncio.gather(
            *(self._retire(entry) for entry in tuple(self._entries.values())),
            return_exceptions=True,
        )
        if any(isinstance(outcome, BaseException) for outcome in outcomes):
            raise RuntimeError("provider_shutdown_failed")

    async def _retire(self, entry: _Entry) -> None:
        if self._entries.get(id(entry.provider)) is not entry:
            return
        if not self._closed and (entry.leases or entry in self._selected.values()):
            return
        if entry.disposal is None:
            entry.disposal = asyncio.create_task(entry.provider.shutdown())
        try:
            await finish_cleanup(entry.disposal)
        finally:
            if entry.disposal.done() and not entry.disposal.cancelled():
                if entry.disposal.exception() is None:
                    self._entries.pop(id(entry.provider), None)

    async def shutdown(self) -> None:
        self._closed = True
        self._selected.clear()
        for entry in self._entries.values():
            if (
                entry.disposal is not None
                and entry.disposal.done()
                and (
                    entry.disposal.cancelled() or entry.disposal.exception() is not None
                )
            ):
                entry.disposal = None
        await self.collect()

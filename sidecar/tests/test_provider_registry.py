import asyncio

import pytest

from translator_sidecar.provider_contract import ProviderId, ProviderProtocolError
from translator_sidecar.provider_registry import ProviderRegistry


class Backend:
    def __init__(self):
        self.calls = 0
        self.started = asyncio.Event()
        self.finish = asyncio.Event()
        self.fail = False

    async def shutdown(self):
        self.calls += 1
        self.started.set()
        await self.finish.wait()
        if self.fail:
            raise RuntimeError("synthetic cleanup failure")


@pytest.mark.parametrize("provider_id", list(ProviderId))
def test_lease_pins_retired_backend_and_release_is_idempotent(provider_id):
    async def scenario():
        original, replacement = Backend(), Backend()
        original.finish.set()
        replacement.finish.set()
        registry = ProviderRegistry({provider_id: original})
        lease = registry.acquire(provider_id)
        registry.replace(provider_id, replacement)
        await registry.collect()
        assert original.calls == 0
        await asyncio.gather(lease.release(), lease.release())
        assert original.calls == 1
        assert lease.provider is original
        await registry.shutdown()
        assert replacement.calls == 1
        assert not registry._entries

    asyncio.run(scenario())


def test_repeated_cancellation_joins_retirement_and_rejects_resurrection():
    async def scenario():
        original, replacement = Backend(), Backend()
        replacement.finish.set()
        registry = ProviderRegistry({ProviderId.LOCAL: original})
        lease = registry.acquire(ProviderId.LOCAL)
        registry.replace(ProviderId.LOCAL, replacement)
        release = asyncio.create_task(lease.release())
        await original.started.wait()
        for _ in range(3):
            release.cancel()
            await asyncio.sleep(0)
            assert not release.done()
        with pytest.raises(ProviderProtocolError, match="retiring"):
            registry.replace(ProviderId.LOCAL, original)
        original.finish.set()
        with pytest.raises(asyncio.CancelledError):
            await release
        assert original.calls == 1
        assert id(original) not in registry._entries
        await registry.shutdown()

    asyncio.run(scenario())


def test_concurrent_shutdown_retains_only_failed_cleanup_for_retry():
    async def scenario():
        local, cloud = Backend(), Backend()
        local.fail = True
        registry = ProviderRegistry({ProviderId.LOCAL: local, ProviderId.OPENAI: cloud})
        first = asyncio.create_task(registry.shutdown())
        await local.started.wait()
        second = asyncio.create_task(registry.shutdown())
        await asyncio.sleep(0)
        local.finish.set()
        cloud.finish.set()
        outcomes = await asyncio.gather(first, second, return_exceptions=True)
        assert all(isinstance(outcome, RuntimeError) for outcome in outcomes)
        assert local.calls == cloud.calls == 1
        assert set(registry._entries) == {id(local)}
        with pytest.raises(ProviderProtocolError):
            registry.acquire(ProviderId.LOCAL)
        with pytest.raises(ProviderProtocolError):
            registry.replace(ProviderId.LOCAL, Backend())
        local.fail = False
        await registry.shutdown()
        await registry.shutdown()
        assert local.calls == 2
        assert cloud.calls == 1
        assert not registry._entries

    asyncio.run(scenario())

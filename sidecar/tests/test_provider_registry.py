import asyncio

import pytest

from translator_sidecar import provider_registry as registry_module
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


@pytest.mark.parametrize("provider_id", list(ProviderId))
def test_explicit_release_retries_failed_retirement_without_releasing_peer(provider_id):
    async def scenario():
        original, replacement = Backend(), Backend()
        original.fail = True
        original.finish.set()
        replacement.finish.set()
        registry = ProviderRegistry({provider_id: original})
        lease = registry.acquire(provider_id)
        registry.replace(provider_id, replacement)
        retries = []
        try:
            with pytest.raises(RuntimeError, match="synthetic cleanup failure"):
                await lease.release()
            failed_task = registry._entries[id(original)].disposal
            assert original.calls == 1
            assert lease._entry.leases == 0
            assert not registry._closed

            # Collection observes a failed attempt; only an explicit retry may
            # start another disposal, without decrementing the released lease.
            with pytest.raises(RuntimeError, match="provider_shutdown_failed"):
                await registry.collect()
            assert original.calls == 1
            assert registry._entries[id(original)].disposal is failed_task

            original.fail = False
            original.finish.clear()
            original.started.clear()
            retries.append(asyncio.create_task(lease.release()))
            for _ in range(64):
                if original.started.is_set() or retries[0].done():
                    break
                await asyncio.sleep(0)
            retries.append(asyncio.create_task(lease.release()))
            for _ in range(8):
                await asyncio.sleep(0)
            calls_before_gate = original.calls
            retry_entered = original.started.is_set()
            pending_before_gate = [not task.done() for task in retries]
            leases_before_gate = lease._entry.leases
            original.finish.set()
            outcomes = await asyncio.gather(*retries, return_exceptions=True)
            removed_after_retry = id(original) not in registry._entries
            calls_after_retry = original.calls
            peer_lease = registry.acquire(provider_id)
            peer_is_selected = peer_lease.provider is replacement
            await peer_lease.release()
            peer_disposals = replacement.calls
            further_release = await asyncio.gather(
                lease.release(), return_exceptions=True
            )
            calls_after_repeat = original.calls
        finally:
            original.fail = False
            original.finish.set()
            replacement.finish.set()
            if retries:
                await asyncio.gather(*retries, return_exceptions=True)
            # Fixture recovery is deliberately after all causal observations:
            # old code can recover through terminal shutdown, but not release.
            await registry.shutdown()

        assert calls_before_gate == 2, "explicit release must start a new disposal"
        assert retry_entered
        assert pending_before_gate == [True, True]
        assert leases_before_gate == 0
        assert outcomes == [None, None]
        assert removed_after_retry
        assert calls_after_retry == calls_after_repeat == 2
        assert further_release == [None]
        assert peer_is_selected and peer_disposals == 0
        assert not registry._entries

    asyncio.run(scenario())


def test_failed_retirement_waiter_keeps_its_error_during_terminal_retry(monkeypatch):
    async def scenario():
        failure_seen, resume_waiter = asyncio.Event(), asyncio.Event()
        collection_entered, resume_collection = asyncio.Event(), asyncio.Event()
        first_attempt, original_errors = [], []
        finish_cleanup = registry_module.finish_cleanup

        async def paused_failed_waiter(operation):
            if isinstance(operation, asyncio.Task) and not first_attempt:
                first_attempt.append(operation)
                try:
                    return await finish_cleanup(operation)
                except RuntimeError as error:
                    original_errors.append(error)
                    failure_seen.set()
                    await resume_waiter.wait()
                    raise
            return await finish_cleanup(operation)

        class PausedCollectionRegistry(ProviderRegistry):
            async def _collect(self):
                collection_entered.set()
                await resume_collection.wait()
                await super()._collect()

        monkeypatch.setattr(registry_module, "finish_cleanup", paused_failed_waiter)
        original, replacement = Backend(), Backend()
        original.fail = True
        original.finish.set()
        replacement.finish.set()
        registry = PausedCollectionRegistry({ProviderId.LOCAL: original})
        lease = registry.acquire(ProviderId.LOCAL)
        registry.replace(ProviderId.LOCAL, replacement)
        release = asyncio.create_task(lease.release())
        terminal = None
        try:
            await asyncio.wait_for(failure_seen.wait(), 1)
            failed_attempt = lease._entry.disposal
            original.fail = False
            terminal = asyncio.create_task(registry.shutdown())
            await asyncio.wait_for(collection_entered.wait(), 1)
            slot_during_retry_admission = lease._entry.disposal
            calls_before_retry = original.calls
            leases_before_retry = lease._entry.leases
            resume_waiter.set()
            old_outcome = await asyncio.wait_for(
                asyncio.gather(release, return_exceptions=True), 1
            )
            resume_collection.set()
            await asyncio.wait_for(terminal, 1)
            calls_after_retry = original.calls, replacement.calls
        finally:
            original.fail = False
            original.finish.set()
            replacement.finish.set()
            resume_waiter.set()
            resume_collection.set()
            tasks = [release] if terminal is None else [release, terminal]
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 1)
            await registry.shutdown()

        assert first_attempt == [failed_attempt]
        assert slot_during_retry_admission is None
        assert calls_before_retry == 1 and leases_before_retry == 0
        assert len(original_errors) == 1
        assert old_outcome[0] is original_errors[0], "old waiter lost its exact error"
        assert calls_after_retry == (2, 1)
        assert lease._entry.leases == 0
        assert not registry._entries

    asyncio.run(scenario())

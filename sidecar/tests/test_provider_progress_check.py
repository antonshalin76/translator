"""Provider-progress oracle must require real non-silent terminal audio."""

import asyncio
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

CHECK = (
    Path(__file__).resolve().parents[2] / "scripts/translator-provider-progress-check"
)
pilot = SimpleNamespace(**runpy.run_path(str(CHECK)))


class Provider:
    def __init__(self, *, silent=False, blocked=False):
        self.silent, self.blocked = silent, blocked
        self.sessions = {}
        self.drained = self.closed = 0
        self.started = asyncio.Event()

    def reserve_session(self, request, publish):
        self.sessions[request.session_id] = publish

        async def open_session():
            return None, SimpleNamespace(state=pilot.ProviderState.READY, models=[])

        async def drain(_reason):
            self.drained += 1

        return SimpleNamespace(open=open_session, drain=drain)

    async def submit_frame(self, frame):
        self.started.set()
        if self.blocked:
            await asyncio.Event().wait()
        if frame.end_of_utterance:
            await self.sessions[frame.session_id](
                (
                    pilot.ProviderAudioDelta(
                        session_id=frame.session_id,
                        direction_id=frame.direction_id,
                        stream_id=frame.stream_id,
                        utterance_id=frame.utterance_id,
                        sequence=0,
                        event_sequence=3,
                        provider_monotonic_ns=1,
                        sample_rate_hz=16000,
                        channels=1,
                        sample_format=pilot.SampleFormat.S16LE,
                        frame_duration_ms=20,
                        pcm=bytes(640) if self.silent else b"\x01\x00" * 320,
                    ),
                    pilot.ProviderUtteranceFinal(
                        session_id=frame.session_id,
                        direction_id=frame.direction_id,
                        stream_id=frame.stream_id,
                        utterance_id=frame.utterance_id,
                        event_sequence=4,
                        final_audio_sequence=0,
                        outcome=pilot.UtteranceOutcome.COMPLETED,
                    ),
                ),
                lambda: None,
            )

    async def shutdown(self):
        self.closed += 1


def check(provider, **overrides):
    arguments = {
        "duration": 0,
        "max_calls": 2,
        "max_audio_ms": 80,
        "timeout": 1,
        "pace": False,
    }
    arguments.update(overrides)
    fixtures = {case[0]: [b"\x01\x00" * 320, bytes(640)] for case in pilot.CASES}
    return pilot.run_check(provider, pilot.ProviderId.OPENAI, fixtures, **arguments)


@pytest.mark.parametrize("silent", [False, True])
def test_terminal_alone_is_not_progress_and_cleanup_is_required(silent):
    async def scenario():
        provider = Provider(silent=silent)
        result = await check(provider)
        assert result["result"] == ("FAIL" if silent else "PASS")
        assert provider.closed == 1 and provider.drained == 2
        assert result["calls"] == 2
        assert result["input_audio_ms"] <= result["reserved_input_audio_ms"] == 80
        if not silent:
            assert all(
                item["completed"] == 1 and item["nonzero_output_frames"] == 1
                for item in result["directions"].values()
            )

    asyncio.run(scenario())


def test_cancellation_preserves_failed_result_and_drains_both_directions():
    async def scenario():
        provider = Provider(blocked=True)
        task = asyncio.create_task(check(provider))
        await provider.started.wait()
        task.cancel()
        result = await task
        assert result["result"] == "FAIL" and result["safe_failure"] == "cancelled"
        assert provider.drained == 2 and provider.closed == 1

    asyncio.run(scenario())


def test_timeout_cannot_pass_or_leave_provider_alive():
    async def scenario():
        provider = Provider(blocked=True)
        result = await check(provider, timeout=0.01)
        assert result["result"] == "FAIL" and result["safe_failure"] == "TimeoutError"
        assert provider.drained == 2 and provider.closed == 1

    asyncio.run(scenario())


def test_drain_delivery_error_cannot_pass():
    async def scenario():
        provider = Provider()
        reserve = provider.reserve_session

        def reserve_failed_drain(request, publish):
            reservation = reserve(request, publish)
            drain = reservation.drain

            async def failed(reason):
                await drain(reason)
                return SimpleNamespace(delivery_error="provider_unavailable")

            reservation.drain = failed
            return reservation

        provider.reserve_session = reserve_failed_drain
        result = await check(provider)
        assert result["result"] == "FAIL"
        assert provider.drained == 2 and provider.closed == 1

    asyncio.run(scenario())


def test_budget_rejects_before_mutating_counters():
    budget = pilot.Budget(2, 80)
    budget.claim(2)
    budget.claim(2)
    with pytest.raises(ValueError, match="live_budget_exhausted"):
        budget.claim(1)
    assert (budget.calls, budget.audio_ms) == (2, 80)


def test_credentials_are_private_unique_and_never_executed(tmp_path):
    path = tmp_path / ".env"
    path.write_text("OPENAI_API_KEY=synthetic-secret\n")
    path.chmod(0o600)
    assert pilot.credential(path) == "synthetic-secret"
    path.chmod(0o644)
    with pytest.raises(ValueError, match="unsafe_credential_file"):
        pilot.credential(path)
    path.chmod(0o600)
    path.write_text("OPENAI_API_KEY=$(touch marker)\n")
    with pytest.raises(ValueError, match="invalid_credential_file"):
        pilot.credential(path)
    assert not (tmp_path / "marker").exists()

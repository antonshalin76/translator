from __future__ import annotations

import asyncio
import base64
import json
import queue
from uuid import UUID, uuid4

import numpy as np

from translator_sidecar.openai_runtime import OpenAIRealtimeProvider
from translator_sidecar.openai_provider import OpenAIRealtimeConfig
from translator_sidecar.provider_contract import (
    AudioDirection,
    CloseProviderSession,
    CloseRequestReason,
    Language,
    OpenProviderSession,
    PcmFormat,
    ProviderAudioDelta,
    ProviderId,
    ProviderInputFrame,
    PrivacySafeProviderError,
    ProviderSessionClosed,
    ProviderTranslationDelta,
    ProviderUtteranceFinal,
    SampleFormat,
    TranslationMode,
    UtteranceOutcome,
    VoiceEngine,
    VoiceGender,
    VoiceProfile,
)


class FakeRealtimeWebSocket:
    def __init__(self) -> None:
        self.url = ""
        self.headers: list[str] = []
        self.timeout = 0
        self.sent: list[dict[str, object]] = []
        self.incoming: queue.Queue[str] = queue.Queue()
        self.closed = False

    def connect(self, url, *, header, timeout):
        self.url = url
        self.headers = list(header)
        self.timeout = timeout

    def send(self, payload: str) -> None:
        self.sent.append(json.loads(payload))

    def recv(self) -> str:
        item = self.incoming.get(timeout=1)
        return item

    def close(self) -> None:
        self.closed = True
        self.incoming.put(json.dumps({"type": "session.closed"}))

    def push(self, event: dict[str, object]) -> None:
        self.incoming.put(json.dumps(event))


def open_request() -> OpenProviderSession:
    return build_open_request(debug_text_enabled=False)


def build_open_request(*, debug_text_enabled: bool) -> OpenProviderSession:
    pcm_format = PcmFormat(
        sample_rate_hz=16_000,
        channels=1,
        sample_format=SampleFormat.S16LE,
        frame_duration_ms=20,
    )
    return OpenProviderSession(
        session_id=uuid4(),
        provider_id=ProviderId.OPENAI,
        direction_id=AudioDirection.MICROPHONE,
        source_language=Language.RU,
        target_language=Language.EN,
        mode=TranslationMode.STREAMING_FIRST,
        requested_input_format=pcm_format,
        requested_output_format=pcm_format,
        voice_profile=VoiceProfile(
            language=Language.EN,
            gender=VoiceGender.MALE,
            engine=VoiceEngine.OPENAI,
        ),
        debug_text_enabled=debug_text_enabled,
    )


def input_frame(
    session_id: UUID,
    *,
    stream_id: UUID,
    utterance_id: UUID,
) -> ProviderInputFrame:
    pcm = np.arange(320, dtype="<i2").tobytes()
    return ProviderInputFrame(
        session_id=session_id,
        direction_id=AudioDirection.MICROPHONE,
        stream_id=stream_id,
        utterance_id=utterance_id,
        sequence=0,
        capture_monotonic_ns=10_000_000,
        sample_rate_hz=16_000,
        channels=1,
        sample_format=SampleFormat.S16LE,
        frame_duration_ms=20,
        source_language=Language.RU,
        target_language=Language.EN,
        mode=TranslationMode.STREAMING_FIRST,
        pcm=pcm,
        end_of_utterance=True,
    )


async def wait_for_event(events, event_type):
    for _ in range(50):
        found = [event for event in events if isinstance(event, event_type)]
        if found:
            return found[-1]
        await asyncio.sleep(0.01)
    raise AssertionError(f"{event_type.__name__} was not published")


async def close_with_server_ack(
    provider: OpenAIRealtimeProvider,
    ws: FakeRealtimeWebSocket,
    session_id: UUID,
) -> None:
    close_task = asyncio.create_task(
        provider.close_session(
            CloseProviderSession(
                session_id=session_id,
                reason=CloseRequestReason.USER_STOP,
            )
        )
    )
    await asyncio.sleep(0.01)
    ws.push({"type": "session.closed"})
    await asyncio.wait_for(close_task, timeout=1)
    await provider.wait_publications(session_id)


def test_openai_runtime_resamples_audio_and_suppresses_debug_text() -> None:
    async def scenario() -> None:
        ws = FakeRealtimeWebSocket()
        published = []

        async def publish(batch, commit) -> None:
            published.extend(batch)
            commit()

        provider = OpenAIRealtimeProvider(
            OpenAIRealtimeConfig(cloud_opt_in=True),
            environ={"OPENAI_API_KEY": "secret-marker"},
            websocket_factory=lambda: ws,
            now_ns=lambda: 20_000_000,
        )
        request = open_request()

        opened, health = await provider.open_session(request, publish)

        assert opened.negotiated_input_format == request.requested_input_format
        assert opened.negotiated_output_format == request.requested_output_format
        assert opened.capabilities.cloud_egress is True
        assert health.provider_id is ProviderId.OPENAI
        assert "secret-marker" not in json.dumps(ws.sent)
        assert ws.sent[0]["type"] == "session.update"

        stream_id = uuid4()
        utterance_id = uuid4()
        frame = input_frame(
            request.session_id,
            stream_id=stream_id,
            utterance_id=utterance_id,
        )
        await provider.submit_frame(frame)
        append = [event for event in ws.sent if event["type"].endswith(".append")][-1]
        assert len(base64.b64decode(append["audio"], validate=True)) == 960

        output_pcm_24k = np.arange(480, dtype="<i2").tobytes()
        ws.push(
            {
                "type": "session.output_audio.delta",
                "delta": base64.b64encode(output_pcm_24k).decode("ascii"),
                "sample_rate": 24_000,
                "channels": 1,
                "format": "pcm16",
            }
        )
        ws.push({"type": "session.output_transcript.delta", "delta": "private text"})
        audio = await wait_for_event(published, ProviderAudioDelta)

        assert audio.sample_rate_hz == 16_000
        assert audio.channels == 1
        assert audio.frame_duration_ms == 20
        assert len(audio.pcm) == 640
        assert not any(isinstance(event, ProviderTranslationDelta) for event in published)

        await close_with_server_ack(provider, ws, request.session_id)

        assert ws.closed is True
        assert any(isinstance(event, ProviderSessionClosed) for event in published)
        await provider.shutdown()

    asyncio.run(scenario())


def test_openai_runtime_drains_close_until_session_closed_event() -> None:
    async def scenario() -> None:
        ws = FakeRealtimeWebSocket()
        published = []

        async def publish(batch, commit) -> None:
            published.extend(batch)
            commit()

        provider = OpenAIRealtimeProvider(
            OpenAIRealtimeConfig(cloud_opt_in=True),
            environ={"OPENAI_API_KEY": "secret-marker"},
            websocket_factory=lambda: ws,
            now_ns=lambda: 20_000_000,
        )
        request = open_request()
        await provider.open_session(request, publish)

        close_task = asyncio.create_task(
            provider.close_session(
                CloseProviderSession(
                    session_id=request.session_id,
                    reason=CloseRequestReason.USER_STOP,
                )
            )
        )
        await asyncio.sleep(0.05)
        assert ws.sent[-1]["type"] == "session.close"
        assert ws.closed is False
        assert not any(isinstance(event, ProviderSessionClosed) for event in published)

        ws.push({"type": "session.closed"})
        await asyncio.wait_for(close_task, timeout=1)
        await provider.wait_publications(request.session_id)

        assert any(isinstance(event, ProviderSessionClosed) for event in published)
        await provider.shutdown()

    asyncio.run(scenario())


def test_openai_runtime_assigns_pending_audio_sequence_at_publication_time() -> None:
    async def scenario() -> None:
        ws = FakeRealtimeWebSocket()
        published = []

        async def publish(batch, commit) -> None:
            published.extend(batch)
            commit()

        provider = OpenAIRealtimeProvider(
            OpenAIRealtimeConfig(cloud_opt_in=True),
            environ={"OPENAI_API_KEY": "secret-marker"},
            websocket_factory=lambda: ws,
            now_ns=lambda: 20_000_000,
        )
        request = build_open_request(debug_text_enabled=True)
        await provider.open_session(request, publish)
        stream_id = uuid4()
        utterance_id = uuid4()
        partial = input_frame(
            request.session_id,
            stream_id=stream_id,
            utterance_id=utterance_id,
        ).model_copy(update={"end_of_utterance": False})
        await provider.submit_frame(partial)

        output_pcm_24k = np.arange(480, dtype="<i2").tobytes()
        ws.push(
            {
                "type": "session.output_audio.delta",
                "delta": base64.b64encode(output_pcm_24k).decode("ascii"),
            }
        )
        ws.push({"type": "session.output_transcript.delta", "delta": "private text"})
        await wait_for_event(published, ProviderTranslationDelta)

        final = partial.model_copy(update={"sequence": 1, "end_of_utterance": True})
        await provider.submit_frame(final)
        audio = await wait_for_event(published, ProviderAudioDelta)
        text = next(event for event in published if isinstance(event, ProviderTranslationDelta))

        assert text.event_sequence < audio.event_sequence
        await close_with_server_ack(provider, ws, request.session_id)
        await provider.shutdown()

    asyncio.run(scenario())


def test_openai_runtime_drops_overlapping_utterance_without_reassigning_audio() -> None:
    async def scenario() -> None:
        ws = FakeRealtimeWebSocket()
        published = []

        async def publish(batch, commit) -> None:
            published.extend(batch)
            commit()

        provider = OpenAIRealtimeProvider(
            OpenAIRealtimeConfig(cloud_opt_in=True),
            environ={"OPENAI_API_KEY": "secret-marker"},
            websocket_factory=lambda: ws,
            now_ns=lambda: 20_000_000,
            final_idle_ms=10_000,
        )
        request = open_request()
        await provider.open_session(request, publish)
        stream_id = uuid4()
        first_utterance = uuid4()
        second_utterance = uuid4()
        await provider.submit_frame(
            input_frame(
                request.session_id,
                stream_id=stream_id,
                utterance_id=first_utterance,
            )
        )
        await provider.submit_frame(
            input_frame(
                request.session_id,
                stream_id=stream_id,
                utterance_id=second_utterance,
            )
        )

        dropped = [
            event
            for event in published
            if isinstance(event, ProviderUtteranceFinal)
            and event.utterance_id == second_utterance
        ]
        errors = [
            event
            for event in published
            if isinstance(event, PrivacySafeProviderError)
            and event.utterance_id == second_utterance
        ]
        assert errors
        assert dropped and dropped[-1].outcome is UtteranceOutcome.DROPPED

        output_pcm_24k = np.arange(480, dtype="<i2").tobytes()
        ws.push(
            {
                "type": "session.output_audio.delta",
                "delta": base64.b64encode(output_pcm_24k).decode("ascii"),
            }
        )
        audio = await wait_for_event(published, ProviderAudioDelta)
        assert audio.utterance_id == first_utterance

        await close_with_server_ack(provider, ws, request.session_id)
        await provider.shutdown()

    asyncio.run(scenario())

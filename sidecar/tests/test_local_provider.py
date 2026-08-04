from __future__ import annotations

import asyncio
from collections.abc import Iterator
import gc
import logging
from threading import Event as ThreadEvent
import traceback
from uuid import UUID, uuid4
import weakref

import pytest

import translator_sidecar.local.local_provider as local_provider_module
from translator_sidecar.local.inference_scheduler import InferenceScheduler
from translator_sidecar.local.local_provider import (
    LocalProvider,
    LocalProviderPublicationError,
    LocalProviderProtocolError,
)
from translator_sidecar.local.tts import TtsOutputLimit
from translator_sidecar.provider_contract import (
    AudioDirection,
    CancelReason,
    CancelUtterance,
    CloseProviderSession,
    CloseRequestReason,
    ComputeDevice,
    Language,
    ModelKind,
    ModelState,
    OpenProviderSession,
    PcmFormat,
    ProviderId,
    PrivacySafeProviderError,
    ProviderAudioDelta,
    ProviderHealth,
    ProviderInputFrame,
    ProviderLatency,
    ProviderSessionClosed,
    ProviderSessionOpened,
    ProviderState,
    ProviderTranscriptDelta,
    ProviderTranslationDelta,
    ProviderUtteranceFinal,
    SafeErrorCode,
    SampleFormat,
    SessionCloseReason,
    TranslationMode,
    UpdateDebugText,
    UtteranceOutcome,
    VoiceEngine,
    VoiceGender,
    VoiceProfile,
)


def run(coroutine):
    return asyncio.run(coroutine)


class FakeAsr:
    actual_device = "cuda"
    degraded = False
    unavailable = False
    resident_model_id = "small"

    def __init__(
        self,
        *,
        started: ThreadEvent | None = None,
        release: ThreadEvent | None = None,
        result: str = "final source",
        after_call=None,
        failure: Exception | None = None,
    ) -> None:
        self.calls: list[tuple[bytes, Language, TranslationMode]] = []
        self.started = started
        self.release = release
        self.result = result
        self.after_call = after_call
        self.failure = failure

    def transcribe(
        self,
        pcm: bytes,
        *,
        language: Language,
        mode: TranslationMode,
    ) -> str:
        self.calls.append((pcm, language, mode))
        if self.started is not None:
            self.started.set()
        if self.release is not None:
            assert self.release.wait(timeout=2)
        if self.after_call is not None:
            self.after_call()
        if self.failure is not None:
            raise self.failure
        return self.result


class FakeTranslator:
    model_path = "/models/nllb"
    actual_device = "cuda"
    unavailable = False

    def __init__(
        self,
        *,
        result: str = "final translation",
        token_count: int | None = None,
        after_call=None,
        failure: Exception | None = None,
        require_boundary=None,
    ) -> None:
        self.result = result
        self.token_count = token_count
        self.after_call = after_call
        self.failure = failure
        self.require_boundary = require_boundary
        self.calls: list[
            tuple[str, Language, Language, TranslationMode]
        ] = []

    def translate(
        self,
        text: str,
        *,
        source_language: Language,
        target_language: Language,
        mode: TranslationMode,
    ) -> str:
        self.calls.append(
            (text, source_language, target_language, mode)
        )
        if (
            self.require_boundary is not None
            and not self.require_boundary()
        ):
            raise AssertionError("MT ran outside source commit")
        if self.after_call is not None:
            self.after_call()
        if self.failure is not None:
            raise self.failure
        return self.result

    def count_tokens(self, text: str) -> int:
        return (
            self.token_count
            if self.token_count is not None
            else len(text.split())
        )


class FakeTts:
    actual_device = "cpu"
    unavailable = False

    def __init__(
        self,
        *,
        frame_count: int = 2,
        raise_output_limit: bool = False,
        before_first_frame=None,
        failure: Exception | None = None,
        frame_byte: int | None = None,
        require_boundary=None,
    ) -> None:
        self.frame_count = frame_count
        self.raise_output_limit = raise_output_limit
        self.before_first_frame = before_first_frame
        self.failure = failure
        self.frame_byte = frame_byte
        self.require_boundary = require_boundary
        self.calls: list[dict[str, object]] = []

    def synthesize_frames(
        self,
        text: str,
        *,
        target_language: Language,
        voice_profile: VoiceProfile,
        mode: TranslationMode,
        output_sample_rate_hz: int,
        output_channels: int,
        frame_duration_ms: int,
        cancelled,
        continuation: bool = False,
    ) -> Iterator[bytes]:
        self.calls.append(
            {
                "text": text,
                "target_language": target_language,
                "gender": voice_profile.gender,
                "mode": mode,
                "sample_rate_hz": output_sample_rate_hz,
                "channels": output_channels,
                "frame_duration_ms": frame_duration_ms,
                "continuation": continuation,
            }
        )
        if (
            self.require_boundary is not None
            and not self.require_boundary()
        ):
            raise AssertionError("TTS ran outside source commit")
        frame_bytes = (
            output_sample_rate_hz
            * output_channels
            * frame_duration_ms
            // 1000
            * 2
        )
        if self.failure is not None:
            raise self.failure
        for index in range(self.frame_count):
            if cancelled():
                return
            if index == 0 and self.before_first_frame is not None:
                self.before_first_frame()
            value = (
                self.frame_byte
                if self.frame_byte is not None
                else (index % 255) + 1
            )
            yield bytes([value]) * frame_bytes
        if self.raise_output_limit:
            raise TtsOutputLimit("synthesized audio exceeds 12 seconds")


def request(
    direction: AudioDirection,
    *,
    mode: TranslationMode = TranslationMode.QUALITY_FIRST,
    debug_text_enabled: bool = False,
    gender: VoiceGender = VoiceGender.FEMALE,
) -> OpenProviderSession:
    source, target = (
        (Language.RU, Language.EN)
        if direction is AudioDirection.MICROPHONE
        else (Language.EN, Language.RU)
    )
    input_format = PcmFormat(
        sample_rate_hz=16_000,
        channels=1,
        sample_format=SampleFormat.S16LE,
        frame_duration_ms=100,
    )
    output_format = PcmFormat(
        sample_rate_hz=24_000,
        channels=1,
        sample_format=SampleFormat.S16LE,
        frame_duration_ms=20,
    )
    return OpenProviderSession(
        session_id=uuid4(),
        provider_id=ProviderId.LOCAL,
        direction_id=direction,
        source_language=source,
        target_language=target,
        mode=mode,
        requested_input_format=input_format,
        requested_output_format=output_format,
        voice_profile=VoiceProfile(
            language=target,
            gender=gender,
            engine=VoiceEngine.PIPER,
        ),
        debug_text_enabled=debug_text_enabled,
    )


def input_frame(
    session: OpenProviderSession,
    *,
    sequence: int,
    utterance_id: UUID,
    end_of_utterance: bool,
    capture_monotonic_ns: int = 0,
    pcm_word: bytes = b"\x01\x02",
) -> ProviderInputFrame:
    pcm_format = session.requested_input_format
    sample_count = (
        pcm_format.sample_rate_hz
        * pcm_format.channels
        * pcm_format.frame_duration_ms
        // 1000
    )
    return ProviderInputFrame(
        session_id=session.session_id,
        direction_id=session.direction_id,
        stream_id=UUID(
            int=1
            if session.direction_id is AudioDirection.MICROPHONE
            else 2
        ),
        utterance_id=utterance_id,
        sequence=sequence,
        capture_monotonic_ns=capture_monotonic_ns,
        sample_rate_hz=pcm_format.sample_rate_hz,
        channels=pcm_format.channels,
        sample_format=pcm_format.sample_format,
        frame_duration_ms=pcm_format.frame_duration_ms,
        source_language=session.source_language,
        target_language=session.target_language,
        mode=session.mode,
        pcm=pcm_word * sample_count,
        end_of_utterance=end_of_utterance,
    )


def build_provider(
    *,
    asr: FakeAsr | None = None,
    translator: FakeTranslator | None = None,
    tts: FakeTts | None = None,
    now_ns=lambda: 0,
    mt_device: ComputeDevice = ComputeDevice.CUDA,
    asr_model_id: str = "faster-whisper-small",
    model_admission_probe=None,
) -> tuple[LocalProvider, FakeAsr, FakeTranslator, FakeTts]:
    effective_asr = asr or FakeAsr()
    effective_translator = translator or FakeTranslator()
    effective_tts = tts or FakeTts()
    return (
        LocalProvider(
            asr=effective_asr,
            translator=effective_translator,
            tts=effective_tts,
            scheduler=InferenceScheduler(),
            now_ns=now_ns,
            asr_model_id=asr_model_id,
            mt_model_id="nllb-200-distilled-600m-ct2-int8",
            tts_model_id="piper-presets-v1",
            mt_device=mt_device,
            model_admission_probe=model_admission_probe,
        ),
        effective_asr,
        effective_translator,
        effective_tts,
    )


def test_provider_marks_voiced_eou_tail_as_tts_continuation() -> None:
    async def scenario() -> None:
        provider, _asr, _translator, tts = build_provider(
            translator=FakeTranslator(result="Translated sentence.")
        )
        session = request(AudioDirection.MICROPHONE)
        collector = Collector()
        try:
            await provider.open_session(session, collector.publish)
            utterance_id = uuid4()
            for sequence in range(3):
                await provider.submit_frame(
                    input_frame(
                        session,
                        sequence=sequence,
                        utterance_id=utterance_id,
                        end_of_utterance=sequence == 2,
                        pcm_word=b"\x00\x20",
                    )
                )
            await provider.wait_idle()
        finally:
            await provider.shutdown()

        assert tts.calls[-1]["continuation"] is True

    run(scenario())


def test_provider_keeps_silent_eou_tail_as_terminal_tts_boundary() -> None:
    async def scenario() -> None:
        provider, _asr, _translator, tts = build_provider(
            translator=FakeTranslator(result="Translated sentence.")
        )
        session = request(AudioDirection.MICROPHONE)
        collector = Collector()
        try:
            await provider.open_session(session, collector.publish)
            utterance_id = uuid4()
            await provider.submit_frame(
                input_frame(
                    session,
                    sequence=0,
                    utterance_id=utterance_id,
                    end_of_utterance=False,
                    pcm_word=b"\x00\x20",
                )
            )
            await provider.submit_frame(
                input_frame(
                    session,
                    sequence=1,
                    utterance_id=utterance_id,
                    end_of_utterance=True,
                    pcm_word=b"\x00\x00",
                )
            )
            await provider.wait_idle()
        finally:
            await provider.shutdown()

        assert tts.calls[-1]["continuation"] is False

    run(scenario())


def test_local_provider_rejects_openai_provider_sessions() -> None:
    provider, _, _, _ = build_provider()

    async def scenario() -> None:
        async def publish(batch, commit) -> None:
            commit()

        openai_request = request(AudioDirection.MICROPHONE).model_copy(
            update={"provider_id": ProviderId.OPENAI}
        )
        with pytest.raises(LocalProviderProtocolError, match="unsupported provider"):
            await provider.open_session(openai_request, publish)
        await provider.shutdown()

    asyncio.run(scenario())


def test_duplex_sessions_translate_only_after_eou_and_keep_events_isolated() -> None:
    async def scenario() -> None:
        provider, asr, translator, tts = build_provider()
        sessions = [
            request(
                AudioDirection.MICROPHONE,
                mode=TranslationMode.BALANCED,
                gender=VoiceGender.MALE,
            ),
            request(
                AudioDirection.SPEAKER,
                mode=TranslationMode.STREAMING_FIRST,
                gender=VoiceGender.FEMALE,
            ),
        ]
        events: dict[UUID, list[object]] = {
            session.session_id: [] for session in sessions
        }
        opening: dict[UUID, tuple[object, object]] = {}

        async def publish(
            session_id: UUID,
            batch: tuple[object, ...],
            commit,
        ) -> None:
            events[session_id].extend(batch)
            commit()

        try:
            for session in sessions:
                opened, health = await provider.open_session(
                    session,
                    lambda batch, commit, session_id=session.session_id: publish(
                        session_id, batch, commit
                    ),
                )
                assert opened.session_id == session.session_id
                assert health.session_id == session.session_id
                opening[session.session_id] = (opened, health)

            utterances = [uuid4(), uuid4()]
            for session, utterance_id in zip(sessions, utterances):
                await provider.submit_frame(
                    input_frame(
                        session,
                        sequence=0,
                        utterance_id=utterance_id,
                        end_of_utterance=False,
                    )
                )
            assert asr.calls == []
            assert translator.calls == []
            assert tts.calls == []

            await asyncio.gather(
                *(
                    provider.submit_frame(
                        input_frame(
                            session,
                            sequence=1,
                            utterance_id=utterance_id,
                            end_of_utterance=True,
                        )
                    )
                    for session, utterance_id in zip(
                        sessions, utterances
                    )
                )
            )
            await provider.wait_idle()

            assert {call[1] for call in asr.calls} == {
                Language.RU,
                Language.EN,
            }
            assert {
                (call[1], call[2]) for call in translator.calls
            } == {
                (Language.RU, Language.EN),
                (Language.EN, Language.RU),
            }
            assert {call["target_language"] for call in tts.calls} == {
                Language.RU,
                Language.EN,
            }
            assert {call["mode"] for call in tts.calls} == {
                TranslationMode.BALANCED,
                TranslationMode.STREAMING_FIRST,
            }
            assert {call["gender"] for call in tts.calls} == {
                VoiceGender.MALE,
                VoiceGender.FEMALE,
            }
            assert all(
                call["sample_rate_hz"] == 24_000
                and call["channels"] == 1
                and call["frame_duration_ms"] == 20
                for call in tts.calls
            )
            for session, utterance_id in zip(sessions, utterances):
                session_events = events[session.session_id]
                assert all(
                    event.session_id == session.session_id
                    for event in session_events
                )
                assert [
                    type(event) for event in session_events
                ] == [
                    ProviderAudioDelta,
                    ProviderAudioDelta,
                    ProviderLatency,
                    ProviderUtteranceFinal,
                ]
                audio = session_events[:2]
                assert [event.sequence for event in audio] == [0, 1]
                assert all(
                    event.stream_id
                    == UUID(
                        int=1
                        if session.direction_id
                        is AudioDirection.MICROPHONE
                        else 2
                    )
                    and event.direction_id == session.direction_id
                    and event.sample_rate_hz == 24_000
                    and event.channels == 1
                    and event.frame_duration_ms == 20
                    and len(event.pcm) == 960
                    for event in audio
                )
                assert all(
                    event.utterance_id == utterance_id
                    for event in session_events
                )
                assert session_events[-1].outcome is UtteranceOutcome.COMPLETED
                assert session_events[-1].final_audio_sequence == 1
                opened, health = opening[session.session_id]
                sequences = [
                    opened.event_sequence,
                    health.event_sequence,
                    *(event.event_sequence for event in session_events),
                ]
                assert sequences == list(range(1, len(sequences) + 1))
                latency = session_events[-2]
                assert latency.asr_first_text_ms == 0
                assert latency.asr_final_text_ms == 0
                assert latency.mt_first_text_ms == 0
                assert latency.tts_first_audio_ms == 0
                assert latency.provider_total_ms == 0
            assert all(len(call[0]) == 6400 for call in asr.calls)
        finally:
            await provider.shutdown()

    run(scenario())


@pytest.mark.parametrize(
    "mode",
    [
        TranslationMode.QUALITY_FIRST,
        TranslationMode.BALANCED,
        TranslationMode.STREAMING_FIRST,
    ],
)
def test_source_overflow_discards_until_eou_then_emits_atomic_terminal(
    mode: TranslationMode,
) -> None:
    async def scenario() -> None:
        clock = iter((3_000_000_000,) * 20)
        provider, asr, translator, tts = build_provider(
            now_ns=lambda: next(clock)
        )
        session = request(AudioDirection.MICROPHONE, mode=mode)
        utterance_id = uuid4()
        try:
            collector = Collector()
            opened, health = await provider.open_session(
                session, collector.publish
            )
            for sequence in range(300):
                await provider.submit_frame(
                    input_frame(
                        session,
                        sequence=sequence,
                        utterance_id=utterance_id,
                        end_of_utterance=False,
                    )
                )
            await provider.submit_frame(
                input_frame(
                    session,
                    sequence=300,
                    utterance_id=utterance_id,
                    end_of_utterance=False,
                )
            )
            assert collector.batches == []
            overflow_health = await provider.health(session.session_id)
            assert (
                overflow_health.queues.provider_input_buffered_ms == 0
            )
            await provider.submit_frame(
                input_frame(
                    session,
                    sequence=301,
                    utterance_id=utterance_id,
                    end_of_utterance=True,
                )
            )
            await provider.wait_publications(session.session_id)
            assert len(collector.batches) == 1
            latency, error, final = collector.batches[0]
            assert isinstance(latency, ProviderLatency)
            assert isinstance(error, PrivacySafeProviderError)
            assert error.code is SafeErrorCode.QUEUE_OVERFLOW
            assert isinstance(final, ProviderUtteranceFinal)
            assert final.outcome is UtteranceOutcome.DROPPED
            assert final.final_audio_sequence is None
            assert latency.asr_first_text_ms is None
            assert latency.asr_final_text_ms is None
            assert latency.mt_first_text_ms is None
            assert latency.tts_first_audio_ms is None
            assert latency.provider_total_ms == 3000
            assert all(
                event.session_id == session.session_id
                and event.direction_id == session.direction_id
                and event.utterance_id == utterance_id
                for event in (latency, error, final)
            )
            assert [
                opened.event_sequence,
                health.event_sequence,
                latency.event_sequence,
                error.event_sequence,
                final.event_sequence,
            ] == [1, 2, 3, 4, 5]
            assert asr.calls == []
            assert translator.calls == []
            assert tts.calls == []
            snapshot = tuple(collector.events)
            await provider.wait_idle()
            await asyncio.sleep(0)
            assert tuple(collector.events) == snapshot
        finally:
            await provider.shutdown()

    run(scenario())


@pytest.mark.parametrize(
    ("mode", "accepted_ms"),
        [
        (mode, accepted_ms)
        for mode in TranslationMode
        for accepted_ms in (11_900, 12_000, 23_900, 24_000, 29_900, 30_000)
    ],
)
def test_source_at_or_just_below_cap_is_accepted_and_all_pcm_is_transcribed(
    mode: TranslationMode,
    accepted_ms: int,
) -> None:
    async def scenario() -> None:
        provider, asr, _, _ = build_provider()
        session = request(AudioDirection.MICROPHONE, mode=mode)
        collector = Collector()
        utterance_id = uuid4()
        try:
            await provider.open_session(session, collector.publish)
            frame_count = accepted_ms // 100
            expected_pcm = []
            for sequence in range(frame_count):
                pcm_word = bytes([(sequence % 255) + 1, 0])
                expected_pcm.append(pcm_word * 1600)
                await provider.submit_frame(
                    input_frame(
                        session,
                        sequence=sequence,
                        utterance_id=utterance_id,
                        end_of_utterance=sequence == frame_count - 1,
                        pcm_word=pcm_word,
                    )
                )
            await provider.wait_idle()
            assert len(asr.calls) == 1
            assert asr.calls[0][0] == b"".join(expected_pcm)
            assert asr.calls[0][2] is mode
            assert isinstance(
                collector.events[-1], ProviderUtteranceFinal
            )
            assert (
                collector.events[-1].outcome
                is UtteranceOutcome.COMPLETED
            )
        finally:
            await provider.shutdown()

    run(scenario())


def test_podcast_length_source_window_is_accepted_and_all_pcm_is_transcribed() -> None:
    async def scenario() -> None:
        provider, asr, _, _ = build_provider()
        session = request(
            AudioDirection.MICROPHONE,
            mode=TranslationMode.STREAMING_FIRST,
        )
        collector = Collector()
        utterance_id = uuid4()
        try:
            await provider.open_session(session, collector.publish)
            expected_pcm = []
            for sequence in range(240):
                pcm_word = bytes([(sequence % 255) + 1, 0])
                expected_pcm.append(pcm_word * 1600)
                await provider.submit_frame(
                    input_frame(
                        session,
                        sequence=sequence,
                        utterance_id=utterance_id,
                        end_of_utterance=sequence == 239,
                        pcm_word=pcm_word,
                    )
                )
            await provider.wait_idle()
            assert len(asr.calls) == 1
            assert asr.calls[0][0] == b"".join(expected_pcm)
            assert collector.events[-1].outcome is UtteranceOutcome.COMPLETED
        finally:
            await provider.shutdown()

    run(scenario())


def test_overflow_discarding_still_validates_sequence() -> None:
    async def scenario() -> None:
        provider, _, _, _ = build_provider()
        session = request(
            AudioDirection.MICROPHONE,
            mode=TranslationMode.STREAMING_FIRST,
        )
        collector = Collector()
        utterance_id = uuid4()
        try:
            await provider.open_session(session, collector.publish)
            for sequence in range(301):
                await provider.submit_frame(
                    input_frame(
                        session,
                        sequence=sequence,
                        utterance_id=utterance_id,
                        end_of_utterance=False,
                    )
                )
            with pytest.raises(
                LocalProviderProtocolError,
                match="sequence",
            ):
                await provider.submit_frame(
                    input_frame(
                        session,
                        sequence=302,
                        utterance_id=utterance_id,
                        end_of_utterance=True,
                    )
                )
            assert collector.events == []
        finally:
            await provider.shutdown()

    run(scenario())


def test_first_over_cap_frame_with_eou_drops_immediately() -> None:
    async def scenario() -> None:
        provider, asr, _, _ = build_provider()
        session = request(
            AudioDirection.MICROPHONE,
            mode=TranslationMode.STREAMING_FIRST,
        )
        collector = Collector()
        utterance_id = uuid4()
        try:
            await provider.open_session(session, collector.publish)
            for sequence in range(300):
                await provider.submit_frame(
                    input_frame(
                        session,
                        sequence=sequence,
                        utterance_id=utterance_id,
                        end_of_utterance=False,
                    )
                )
            await provider.submit_frame(
                input_frame(
                    session,
                    sequence=300,
                    utterance_id=utterance_id,
                    end_of_utterance=True,
                )
            )
            await provider.wait_publications(session.session_id)
            assert [
                type(event) for event in collector.events
            ] == [
                ProviderLatency,
                PrivacySafeProviderError,
                ProviderUtteranceFinal,
            ]
            assert collector.events[1].code is SafeErrorCode.QUEUE_OVERFLOW
            assert (
                collector.events[2].outcome
                is UtteranceOutcome.DROPPED
            )
            assert asr.calls == []
        finally:
            await provider.shutdown()

    run(scenario())


def test_debug_text_is_checked_at_publication_and_never_reaches_tts_early() -> None:
    async def scenario() -> None:
        started = ThreadEvent()
        release = ThreadEvent()
        provider, _, _, tts = build_provider(
            asr=FakeAsr(started=started, release=release)
        )
        session = request(
            AudioDirection.MICROPHONE,
            debug_text_enabled=True,
        )
        collector = Collector()
        utterance_id = uuid4()
        try:
            await provider.open_session(
                session, collector.publish
            )
            await provider.submit_frame(
                input_frame(
                    session,
                    sequence=0,
                    utterance_id=utterance_id,
                    end_of_utterance=True,
                )
            )
            assert await asyncio.to_thread(started.wait, 1)
            assert tts.calls == []
            await provider.update_debug_text(
                UpdateDebugText(
                    session_id=session.session_id,
                    enabled=False,
                )
            )
            release.set()
            await provider.wait_idle()
            assert not any(
                isinstance(
                    event,
                    (ProviderTranscriptDelta, ProviderTranslationDelta),
                )
                for event in collector.events
            )
            assert len(tts.calls) == 1
        finally:
            release.set()
            await provider.shutdown()

    run(scenario())


def test_debug_text_can_be_enabled_before_publication_per_session() -> None:
    async def scenario() -> None:
        started = ThreadEvent()
        release = ThreadEvent()
        provider, _, _, _ = build_provider(
            asr=FakeAsr(started=started, release=release)
        )
        debug_session = request(
            AudioDirection.MICROPHONE,
            debug_text_enabled=False,
        )
        normal_session = request(
            AudioDirection.SPEAKER,
            debug_text_enabled=False,
        )
        debug_collector = Collector()
        normal_collector = Collector()
        normal_utterance = uuid4()
        try:
            await provider.open_session(
                debug_session, debug_collector.publish
            )
            await provider.open_session(
                normal_session, normal_collector.publish
            )
            await provider.submit_frame(
                input_frame(
                    debug_session,
                    sequence=0,
                    utterance_id=uuid4(),
                    end_of_utterance=True,
                )
            )
            await provider.submit_frame(
                input_frame(
                    normal_session,
                    sequence=0,
                    utterance_id=normal_utterance,
                    end_of_utterance=True,
                )
            )
            assert await asyncio.to_thread(started.wait, 1)
            await provider.update_debug_text(
                UpdateDebugText(
                    session_id=debug_session.session_id,
                    enabled=True,
                )
            )
            release.set()
            await provider.wait_idle()
            assert [
                type(event) for event in debug_collector.events[:2]
            ] == [
                ProviderTranscriptDelta,
                ProviderTranslationDelta,
            ]
            transcript, translation = debug_collector.events[:2]
            assert transcript.text == "final source"
            assert transcript.is_final is True
            assert translation.text == "final translation"
            assert translation.stable_prefix is True
            assert translation.is_final is True
            assert not any(
                isinstance(
                    event,
                    (ProviderTranscriptDelta, ProviderTranslationDelta),
                )
                for event in normal_collector.events
            )
            assert (
                normal_collector.events[-1].outcome
                is UtteranceOutcome.COMPLETED
            )
        finally:
            release.set()
            await provider.shutdown()

    run(scenario())


def test_debug_content_does_not_reach_logs_or_terminal_metadata(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def scenario() -> None:
        caplog.set_level(logging.DEBUG)
        source_marker = "private-source-marker"
        translation_marker = "private-translation-marker"
        source_pcm_word = b"PM"
        provider, _, _, _ = build_provider(
            asr=FakeAsr(result=source_marker),
            translator=FakeTranslator(result=translation_marker),
            tts=FakeTts(frame_byte=0x5A),
        )
        session = request(
            AudioDirection.MICROPHONE,
            debug_text_enabled=False,
        )
        collector = Collector()
        try:
            await provider.open_session(session, collector.publish)
            await provider.submit_frame(
                input_frame(
                    session,
                    sequence=0,
                    utterance_id=uuid4(),
                    end_of_utterance=True,
                    pcm_word=source_pcm_word,
                )
            )
            await provider.wait_idle()
            normal_surface = repr(collector.events) + caplog.text
            assert source_marker not in normal_surface
            assert translation_marker not in normal_surface
            assert repr(source_pcm_word * 16) not in caplog.text
            output = next(
                event
                for event in collector.events
                if isinstance(event, ProviderAudioDelta)
            )
            assert output.pcm == b"Z" * 960
            assert repr(output.pcm[:32]) not in caplog.text
        finally:
            await provider.shutdown()

    run(scenario())


def test_cancel_during_native_asr_emits_one_terminal_and_no_late_events() -> None:
    async def scenario() -> None:
        started = ThreadEvent()
        release = ThreadEvent()
        provider, _, translator, tts = build_provider(
            asr=FakeAsr(started=started, release=release)
        )
        session = request(AudioDirection.MICROPHONE)
        collector = Collector()
        utterance_id = uuid4()
        try:
            await provider.open_session(
                session, collector.publish
            )
            await provider.submit_frame(
                input_frame(
                    session,
                    sequence=0,
                    utterance_id=utterance_id,
                    end_of_utterance=True,
                )
            )
            assert await asyncio.to_thread(started.wait, 1)
            await asyncio.wait_for(
                provider.cancel_utterance(
                    CancelUtterance(
                        session_id=session.session_id,
                        direction_id=session.direction_id,
                        utterance_id=utterance_id,
                        reason=CancelReason.USER_INTERRUPT,
                    )
                ),
                timeout=0.2,
            )
            await provider.wait_publications(session.session_id)
            assert len(collector.events) == 1
            assert isinstance(
                collector.events[0], ProviderUtteranceFinal
            )
            assert (
                collector.events[0].outcome
                is UtteranceOutcome.CANCELLED
            )
            release.set()
            await provider.wait_idle()
            await asyncio.sleep(0)
            finals = [
                event
                for event in collector.events
                if isinstance(event, ProviderUtteranceFinal)
            ]
            assert len(finals) == 1
            assert finals[0].outcome is UtteranceOutcome.CANCELLED
            assert collector.events == finals
            assert translator.calls == []
            assert tts.calls == []
        finally:
            release.set()
            await provider.shutdown()

    run(scenario())


@pytest.mark.parametrize("overflow", [False, True])
def test_cancel_purges_collecting_and_overflow_states(
    overflow: bool,
) -> None:
    async def scenario() -> None:
        provider, asr, _, _ = build_provider()
        session = request(
            AudioDirection.MICROPHONE,
            mode=TranslationMode.STREAMING_FIRST,
        )
        collector = Collector()
        utterance_id = uuid4()
        try:
            await provider.open_session(session, collector.publish)
            frame_count = 301 if overflow else 1
            for sequence in range(frame_count):
                await provider.submit_frame(
                    input_frame(
                        session,
                        sequence=sequence,
                        utterance_id=utterance_id,
                        end_of_utterance=False,
                    )
                )
            await provider.cancel_utterance(
                CancelUtterance(
                    session_id=session.session_id,
                    direction_id=session.direction_id,
                    utterance_id=utterance_id,
                    reason=CancelReason.USER_INTERRUPT,
                )
            )
            await provider.wait_publications(session.session_id)
            assert len(collector.batches) == 1
            assert len(collector.batches[0]) == 1
            assert (
                collector.batches[0][0].outcome
                is UtteranceOutcome.CANCELLED
            )
            assert asr.calls == []
            with pytest.raises(
                LocalProviderProtocolError,
                match="utterance",
            ):
                await provider.submit_frame(
                    input_frame(
                        session,
                        sequence=frame_count,
                        utterance_id=utterance_id,
                        end_of_utterance=True,
                    )
                )
        finally:
            await provider.shutdown()

    run(scenario())


def test_cancel_is_isolated_from_opposite_direction() -> None:
    async def scenario() -> None:
        started = ThreadEvent()
        release = ThreadEvent()
        provider, _, _, _ = build_provider(
            asr=FakeAsr(started=started, release=release)
        )
        microphone = request(AudioDirection.MICROPHONE)
        speaker = request(AudioDirection.SPEAKER)
        microphone_events = Collector()
        speaker_events = Collector()
        microphone_utterance = uuid4()
        try:
            await provider.open_session(
                microphone, microphone_events.publish
            )
            await provider.open_session(speaker, speaker_events.publish)
            await provider.submit_frame(
                input_frame(
                    microphone,
                    sequence=0,
                    utterance_id=microphone_utterance,
                    end_of_utterance=True,
                )
            )
            await provider.submit_frame(
                input_frame(
                    speaker,
                    sequence=0,
                    utterance_id=uuid4(),
                    end_of_utterance=True,
                )
            )
            assert await asyncio.to_thread(started.wait, 1)
            await asyncio.wait_for(
                provider.cancel_utterance(
                    CancelUtterance(
                        session_id=microphone.session_id,
                        direction_id=microphone.direction_id,
                        utterance_id=microphone_utterance,
                        reason=CancelReason.USER_INTERRUPT,
                    )
                ),
                timeout=0.2,
            )
            release.set()
            await provider.wait_idle()
            assert microphone_events.events[-1].outcome is UtteranceOutcome.CANCELLED
            assert speaker_events.events[-1].outcome is UtteranceOutcome.COMPLETED
        finally:
            release.set()
            await provider.shutdown()

    run(scenario())


def test_close_purges_session_and_prevents_late_native_events() -> None:
    async def scenario() -> None:
        started = ThreadEvent()
        release = ThreadEvent()
        provider, _, _, _ = build_provider(
            asr=FakeAsr(started=started, release=release)
        )
        session = request(AudioDirection.MICROPHONE)
        collector = Collector()
        try:
            opened, health = await provider.open_session(
                session, collector.publish
            )
            await provider.submit_frame(
                input_frame(
                    session,
                    sequence=0,
                    utterance_id=uuid4(),
                    end_of_utterance=True,
                )
            )
            assert await asyncio.to_thread(started.wait, 1)
            await asyncio.wait_for(
                provider.close_session(
                    CloseProviderSession(
                        session_id=session.session_id,
                        reason=CloseRequestReason.USER_STOP,
                    )
                ),
                timeout=0.2,
            )
            await provider.wait_publications(session.session_id)
            assert len(collector.events) == 1
            assert isinstance(
                collector.events[0], ProviderSessionClosed
            )
            closed = collector.events[0]
            assert closed.reason is SessionCloseReason.USER_STOP
            assert [
                opened.event_sequence,
                health.event_sequence,
                closed.event_sequence,
            ] == [1, 2, 3]
            release.set()
            await provider.wait_idle()
            assert len(collector.events) == 1
        finally:
            release.set()
            await provider.shutdown()

    run(scenario())


@pytest.mark.parametrize(
    ("translation", "token_count", "outcome"),
    [
        ("x" * 128, 96, UtteranceOutcome.COMPLETED),
        ("x" * 129, 1, UtteranceOutcome.DROPPED),
        ("short", 97, UtteranceOutcome.DROPPED),
    ],
    ids=["exact-limits", "characters", "tokens"],
)
def test_translation_limits_drop_without_tts_or_content_in_errors(
    translation: str,
    token_count: int,
    outcome: UtteranceOutcome,
) -> None:
    async def scenario() -> None:
        provider, _, _, tts = build_provider(
            translator=FakeTranslator(
                result=translation,
                token_count=token_count,
            )
        )
        session = request(
            AudioDirection.SPEAKER,
            debug_text_enabled=True,
        )
        collector = Collector()
        utterance_id = uuid4()
        try:
            opened, health = await provider.open_session(
                session, collector.publish
            )
            await provider.submit_frame(
                input_frame(
                    session,
                    sequence=0,
                    utterance_id=utterance_id,
                    end_of_utterance=True,
                )
            )
            await provider.wait_idle()
            assert collector.events[-1].outcome is outcome
            assert_session_sequence(
                opened, health, utterance_id, collector.events
            )
            if outcome is UtteranceOutcome.COMPLETED:
                assert len(tts.calls) == 1
            else:
                assert len(collector.batches) == 1
                assert [
                    type(event) for event in collector.batches[0]
                ] == [
                    ProviderLatency,
                    PrivacySafeProviderError,
                    ProviderUtteranceFinal,
                ]
                latency, error, final = collector.batches[0]
                assert latency.asr_first_text_ms is not None
                assert latency.asr_final_text_ms is not None
                assert latency.mt_first_text_ms is not None
                assert latency.tts_first_audio_ms is None
                assert error.code is SafeErrorCode.QUEUE_OVERFLOW
                assert final.final_audio_sequence is None
                assert translation not in repr(collector.batches)
                assert tts.calls == []
        finally:
            await provider.shutdown()

    run(scenario())


def test_podcast_length_translation_budget_scales_with_source_text() -> None:
    async def scenario() -> None:
        source_text = " ".join(f"source{i}" for i in range(80))
        translation = " ".join(f"translated{i}" for i in range(120))
        provider, _, _, tts = build_provider(
            asr=FakeAsr(result=source_text),
            translator=FakeTranslator(
                result=translation,
                token_count=120,
            ),
        )
        session = request(
            AudioDirection.SPEAKER,
            debug_text_enabled=True,
        )
        collector = Collector()
        utterance_id = uuid4()
        try:
            await provider.open_session(session, collector.publish)
            await provider.submit_frame(
                input_frame(
                    session,
                    sequence=0,
                    utterance_id=utterance_id,
                    end_of_utterance=True,
                )
            )
            await provider.wait_idle()
            assert collector.events[-1].outcome is UtteranceOutcome.COMPLETED
            assert len(tts.calls) == 1
            assert tts.calls[0]["text"] == translation
        finally:
            await provider.shutdown()

    run(scenario())


def test_scheduler_admission_overflow_is_an_atomic_drop() -> None:
    async def scenario() -> None:
        started = ThreadEvent()
        release = ThreadEvent()
        provider, _, _, _ = build_provider(
            asr=FakeAsr(started=started, release=release)
        )
        sessions = [
            request(AudioDirection.MICROPHONE) for _ in range(4)
        ]
        collectors = [Collector() for _ in sessions]
        openings: list[tuple[object, object]] = []
        utterances = [uuid4() for _ in sessions]
        try:
            for session, collector in zip(sessions, collectors):
                openings.append(
                    await provider.open_session(
                        session, collector.publish
                    )
                )
            for session, utterance_id in zip(
                sessions[:3], utterances[:3]
            ):
                await provider.submit_frame(
                    input_frame(
                        session,
                        sequence=0,
                        utterance_id=utterance_id,
                        end_of_utterance=True,
                    )
                )
            await provider.submit_frame(
                input_frame(
                    sessions[3],
                    sequence=0,
                    utterance_id=utterances[3],
                    end_of_utterance=True,
                )
            )
            await provider.wait_publications(sessions[3].session_id)
            assert [
                type(event) for event in collectors[3].events
            ] == [
                ProviderLatency,
                PrivacySafeProviderError,
                ProviderUtteranceFinal,
            ]
            assert (
                collectors[3].events[1].code
                is SafeErrorCode.QUEUE_OVERFLOW
            )
            assert (
                collectors[3].events[2].outcome
                is UtteranceOutcome.DROPPED
            )
            assert_session_sequence(
                *openings[3],
                utterances[3],
                collectors[3].events,
            )
            dropped_snapshot = tuple(collectors[3].events)
            for session, utterance_id in zip(
                sessions[:3], utterances[:3]
            ):
                await provider.cancel_utterance(
                    CancelUtterance(
                        session_id=session.session_id,
                        direction_id=session.direction_id,
                        utterance_id=utterance_id,
                        reason=CancelReason.USER_INTERRUPT,
                    )
                )
            release.set()
            await provider.wait_idle()
            await asyncio.sleep(0)
            assert tuple(collectors[3].events) == dropped_snapshot
        finally:
            release.set()
            await provider.shutdown()

    run(scenario())


def test_tts_output_limit_ends_with_atomic_drop_and_no_late_events() -> None:
    async def scenario() -> None:
        provider, _, _, _ = build_provider(
            tts=FakeTts(
                frame_count=600,
                raise_output_limit=True,
            )
        )
        session = request(AudioDirection.SPEAKER)
        collector = Collector()
        utterance_id = uuid4()
        try:
            opened, health = await provider.open_session(
                session, collector.publish
            )
            await provider.submit_frame(
                input_frame(
                    session,
                    sequence=0,
                    utterance_id=utterance_id,
                    end_of_utterance=True,
                )
            )
            await provider.wait_idle()
            assert len(
                [
                    event
                    for event in collector.events
                    if isinstance(event, ProviderAudioDelta)
                ]
            ) == 600
            assert [
                type(event) for event in collector.batches[-1]
            ] == [
                ProviderLatency,
                PrivacySafeProviderError,
                ProviderUtteranceFinal,
            ]
            assert (
                collector.batches[-1][1].code
                is SafeErrorCode.QUEUE_OVERFLOW
            )
            assert (
                collector.batches[-1][2].outcome
                is UtteranceOutcome.DROPPED
            )
            assert (
                collector.batches[-1][2].final_audio_sequence == 599
            )
            assert_session_sequence(
                opened, health, utterance_id, collector.events
            )
            terminal_count = len(collector.events)
            await asyncio.sleep(0)
            assert len(collector.events) == terminal_count
        finally:
            await provider.shutdown()

    run(scenario())


def test_nonzero_latency_fields_use_original_capture_onset() -> None:
    async def scenario() -> None:
        clock = MutableClock(1_000_000_000)
        provider, _, _, _ = build_provider(
            asr=FakeAsr(
                after_call=lambda: clock.set(1_100_000_000)
            ),
            translator=FakeTranslator(
                after_call=lambda: clock.set(1_200_000_000)
            ),
            tts=FakeTts(
                before_first_frame=lambda: clock.set(1_300_000_000)
            ),
            now_ns=clock,
        )
        session = request(AudioDirection.MICROPHONE)
        collector = Collector()
        try:
            await provider.open_session(session, collector.publish)
            await provider.submit_frame(
                input_frame(
                    session,
                    sequence=0,
                    utterance_id=uuid4(),
                    end_of_utterance=True,
                    capture_monotonic_ns=1_000_000_000,
                )
            )
            await provider.wait_idle()
            latency = next(
                event
                for event in collector.events
                if isinstance(event, ProviderLatency)
            )
            assert latency.asr_first_text_ms == 100
            assert latency.asr_final_text_ms == 100
            assert latency.mt_first_text_ms == 200
            assert latency.tts_first_audio_ms == 300
            assert latency.provider_total_ms == 300
        finally:
            await provider.shutdown()

    run(scenario())


@pytest.mark.parametrize(
    (
        "stage",
        "asr_ms",
        "mt_ms",
        "provider_total_ms",
        "native_call_counts",
    ),
    [
        (ModelKind.ASR, None, None, 100, (1, 0, 0)),
        (ModelKind.MT, 100, None, 200, (1, 1, 0)),
        (ModelKind.TTS, 100, 200, 200, (1, 1, 1)),
    ],
)
def test_runtime_model_failure_has_partial_latency_and_no_private_leak(
    stage: ModelKind,
    asr_ms: int | None,
    mt_ms: int | None,
    provider_total_ms: int,
    native_call_counts: tuple[int, int, int],
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def scenario() -> None:
        marker = f"private-{stage.value}-failure-marker"
        clock = MutableClock(1_000_000_000)
        asr = FakeAsr(
            after_call=lambda: clock.set(1_100_000_000),
            failure=(
                RuntimeError(marker)
                if stage is ModelKind.ASR
                else None
            ),
        )
        translator = FakeTranslator(
            after_call=lambda: clock.set(1_200_000_000),
            failure=(
                RuntimeError(marker)
                if stage is ModelKind.MT
                else None
            ),
        )
        tts = FakeTts(
            failure=(
                RuntimeError(marker)
                if stage is ModelKind.TTS
                else None
            )
        )
        provider, _, _, _ = build_provider(
            asr=asr,
            translator=translator,
            tts=tts,
            now_ns=clock,
        )
        session = request(AudioDirection.MICROPHONE)
        collector = Collector()
        utterance_id = uuid4()
        caplog.set_level(logging.DEBUG)
        try:
            opened, health = await provider.open_session(
                session, collector.publish
            )
            await provider.submit_frame(
                input_frame(
                    session,
                    sequence=0,
                    utterance_id=utterance_id,
                    end_of_utterance=True,
                    capture_monotonic_ns=1_000_000_000,
                )
            )
            await provider.wait_idle()
            assert [
                type(event) for event in collector.events
            ] == [
                ProviderLatency,
                PrivacySafeProviderError,
                ProviderUtteranceFinal,
            ]
            latency, error, final = collector.events
            assert latency.asr_first_text_ms == asr_ms
            assert latency.asr_final_text_ms == asr_ms
            assert latency.mt_first_text_ms == mt_ms
            assert latency.tts_first_audio_ms is None
            assert latency.provider_total_ms == provider_total_ms
            assert error.code is SafeErrorCode.PROVIDER_UNAVAILABLE
            assert final.outcome is UtteranceOutcome.DROPPED
            assert final.final_audio_sequence is None
            assert_session_sequence(
                opened, health, utterance_id, collector.events
            )
            assert (
                len(asr.calls),
                len(translator.calls),
                len(tts.calls),
            ) == native_call_counts
            failed_health = await provider.health(
                session.session_id
            )
            assert failed_health.state is ProviderState.UNAVAILABLE
            failed_model = next(
                model
                for model in failed_health.models
                if model.kind is stage
            )
            assert failed_model.state is ModelState.FAILED
            assert (
                failed_model.safe_error_code
                == SafeErrorCode.MODEL_NOT_LOADED.value
            )
            assert marker not in repr(collector.events)
            assert marker not in caplog.text
            snapshot = tuple(collector.events)
            await asyncio.sleep(0)
            assert tuple(collector.events) == snapshot
        finally:
            await provider.shutdown()

    run(scenario())


@pytest.mark.parametrize(
    (
        "asr_device",
        "asr_degraded",
        "asr_model_id",
        "mt_device",
        "mt_unavailable",
        "expected_state",
    ),
    [
        (
            "cuda",
            False,
            "faster-whisper-small",
            ComputeDevice.CUDA,
            False,
            ProviderState.READY,
        ),
        (
            "cuda",
            True,
            "faster-whisper-large-v3",
            ComputeDevice.CUDA,
            False,
            ProviderState.DEGRADED,
        ),
        (
            "cpu",
            True,
            "faster-whisper-small",
            ComputeDevice.CPU,
            False,
            ProviderState.DEGRADED,
        ),
        (
            "cuda",
            False,
            "faster-whisper-small",
            ComputeDevice.CUDA,
            True,
            ProviderState.UNAVAILABLE,
        ),
    ],
    ids=["ready", "large-to-small", "all-cpu", "missing-model"],
)
def test_health_has_separate_models_and_runtime_state(
    asr_device: str,
    asr_degraded: bool,
    asr_model_id: str,
    mt_device: ComputeDevice,
    mt_unavailable: bool,
    expected_state: ProviderState,
) -> None:
    async def scenario() -> None:
        asr = FakeAsr()
        asr.actual_device = asr_device
        asr.degraded = asr_degraded
        translator = FakeTranslator()
        translator.actual_device = mt_device.value
        translator.unavailable = mt_unavailable
        provider, _, _, _ = build_provider(
            asr=asr,
            translator=translator,
            mt_device=mt_device,
            asr_model_id=asr_model_id,
        )
        session = request(AudioDirection.SPEAKER)
        try:
            collector = Collector()
            _, health = await provider.open_session(
                session, collector.publish
            )
            assert isinstance(health, ProviderHealth)
            assert health.state is expected_state
            assert [model.kind for model in health.models] == [
                ModelKind.ASR,
                ModelKind.MT,
                ModelKind.TTS,
            ]
            assert [model.id for model in health.models] == [
                "small",
                "nllb-200-distilled-600m-ct2-int8",
                "piper-presets-v1",
            ]
            assert health.models[0].device is ComputeDevice(asr_device)
            assert health.models[1].device is mt_device
            assert health.models[2].device is ComputeDevice.CPU
            if mt_unavailable:
                assert health.models[1].state is ModelState.FAILED
                assert (
                    health.models[1].safe_error_code
                    == SafeErrorCode.MODEL_NOT_LOADED.value
                )
                assert (
                    health.safe_error.code
                    is SafeErrorCode.MODEL_NOT_LOADED
                )
            else:
                assert all(
                    model.state is ModelState.READY
                    and model.safe_error_code is None
                    for model in health.models
                )
                assert health.safe_error is None
            assert "final source" not in repr(health)
            assert "final translation" not in repr(health)
        finally:
            await provider.shutdown()

    run(scenario())


@pytest.mark.parametrize(
    "unavailable_kind",
    [ModelKind.ASR, ModelKind.MT, ModelKind.TTS],
)
def test_unavailable_model_fails_closed_without_native_calls(
    unavailable_kind: ModelKind,
) -> None:
    async def scenario() -> None:
        asr = FakeAsr()
        translator = FakeTranslator()
        tts = FakeTts()
        if unavailable_kind is ModelKind.ASR:
            asr.unavailable = True
        elif unavailable_kind is ModelKind.MT:
            translator.unavailable = True
        else:
            tts.unavailable = True
        provider, _, _, _ = build_provider(
            asr=asr,
            translator=translator,
            tts=tts,
        )
        session = request(AudioDirection.MICROPHONE)
        collector = Collector()
        utterance_id = uuid4()
        try:
            opened, health = await provider.open_session(
                session, collector.publish
            )
            assert health.state is ProviderState.UNAVAILABLE
            failed = next(
                model
                for model in health.models
                if model.kind is unavailable_kind
            )
            assert failed.state is ModelState.FAILED
            assert (
                failed.safe_error_code
                == SafeErrorCode.MODEL_NOT_LOADED.value
            )
            await provider.submit_frame(
                input_frame(
                    session,
                    sequence=0,
                    utterance_id=utterance_id,
                    end_of_utterance=True,
                )
            )
            await provider.wait_publications(session.session_id)
            assert [
                type(event) for event in collector.events
            ] == [
                ProviderLatency,
                PrivacySafeProviderError,
                ProviderUtteranceFinal,
            ]
            latency, error, final = collector.events
            assert latency.asr_first_text_ms is None
            assert latency.asr_final_text_ms is None
            assert latency.mt_first_text_ms is None
            assert latency.tts_first_audio_ms is None
            assert error.code is SafeErrorCode.MODEL_NOT_LOADED
            assert final.outcome is UtteranceOutcome.DROPPED
            assert final.final_audio_sequence is None
            assert_session_sequence(
                opened, health, utterance_id, collector.events
            )
            assert asr.calls == []
            assert translator.calls == []
            assert tts.calls == []
            snapshot = tuple(collector.events)
            await provider.wait_idle()
            await asyncio.sleep(0)
            assert tuple(collector.events) == snapshot
        finally:
            await provider.shutdown()

    run(scenario())


def test_sequential_utterances_reset_audio_sequence() -> None:
    async def scenario() -> None:
        provider, _, _, _ = build_provider()
        session = request(AudioDirection.MICROPHONE)
        collector = Collector()
        utterances = [uuid4(), uuid4()]
        try:
            await provider.open_session(session, collector.publish)
            for sequence, utterance_id in enumerate(utterances):
                await provider.submit_frame(
                    input_frame(
                        session,
                        sequence=sequence,
                        utterance_id=utterance_id,
                        end_of_utterance=True,
                    )
                )
            await provider.wait_idle()
            for utterance_id in utterances:
                assert [
                    event.sequence
                    for event in collector.events
                    if isinstance(event, ProviderAudioDelta)
                    and event.utterance_id == utterance_id
                ] == [0, 1]
        finally:
            await provider.shutdown()

    run(scenario())


def test_post_eou_frame_is_a_protocol_error() -> None:
    async def scenario() -> None:
        provider, _, _, _ = build_provider()
        session = request(AudioDirection.MICROPHONE)
        utterance_id = uuid4()
        try:
            await provider.open_session(
                session, Collector().publish
            )
            await provider.submit_frame(
                input_frame(
                    session,
                    sequence=0,
                    utterance_id=utterance_id,
                    end_of_utterance=True,
                )
            )
            with pytest.raises(
                LocalProviderProtocolError,
                match="utterance",
            ):
                await provider.submit_frame(
                    input_frame(
                        session,
                        sequence=1,
                        utterance_id=utterance_id,
                        end_of_utterance=False,
                    )
                )
        finally:
            await provider.shutdown()

    run(scenario())


class Collector:
    def __init__(self) -> None:
        self.batches: list[tuple[object, ...]] = []
        self.events: list[object] = []

    async def publish(
        self,
        batch: tuple[object, ...],
        commit,
    ) -> None:
        self.batches.append(batch)
        self.events.extend(batch)
        commit()


class GatedCollector(Collector):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self._gated = False

    async def publish(
        self,
        batch: tuple[object, ...],
        commit,
    ) -> None:
        if not self._gated:
            self._gated = True
            self.started.set()
            await self.release.wait()
        await super().publish(batch, commit)


class VisibleThenGatedCollector(Collector):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self._gated = False

    async def publish(
        self,
        batch: tuple[object, ...],
        commit,
    ) -> None:
        await super().publish(batch, commit)
        if not self._gated:
            self._gated = True
            self.started.set()
            await self.release.wait()


def test_gated_audio_publication_does_not_block_cancel() -> None:
    async def scenario() -> None:
        provider, _, _, _ = build_provider()
        session = request(AudioDirection.MICROPHONE)
        collector = GatedCollector()
        utterance_id = uuid4()
        try:
            opened, health = await provider.open_session(
                session, collector.publish
            )
            await provider.submit_frame(
                input_frame(
                    session,
                    sequence=0,
                    utterance_id=utterance_id,
                    end_of_utterance=True,
                )
            )
            await asyncio.wait_for(
                collector.started.wait(),
                timeout=1,
            )
            snapshot = await provider.health(session.session_id)
            assert snapshot.state in {
                ProviderState.READY,
                ProviderState.DEGRADED,
            }
            await asyncio.wait_for(
                provider.cancel_utterance(
                    CancelUtterance(
                        session_id=session.session_id,
                        direction_id=session.direction_id,
                        utterance_id=utterance_id,
                        reason=CancelReason.USER_INTERRUPT,
                    )
                ),
                timeout=0.2,
            )
            collector.release.set()
            await provider.wait_idle()
            assert [
                type(event) for event in collector.events
            ] == [
                ProviderUtteranceFinal,
            ]
            assert (
                collector.events[-1].outcome
                is UtteranceOutcome.CANCELLED
            )
            final = collector.events[-1]
            assert final.final_audio_sequence is None
            assert final.event_sequence > health.event_sequence
            assert opened.event_sequence == 1
            assert health.event_sequence == 2
            assert final.session_id == session.session_id
            assert final.direction_id == session.direction_id
            assert final.stream_id == UUID(int=1)
            assert final.utterance_id == utterance_id
        finally:
            collector.release.set()
            await provider.shutdown()

    run(scenario())


def test_visible_in_flight_batch_never_reuses_event_sequence() -> None:
    async def scenario() -> None:
        provider, _, _, _ = build_provider()
        session = request(AudioDirection.MICROPHONE)
        collector = VisibleThenGatedCollector()
        utterance_id = uuid4()
        try:
            await provider.open_session(session, collector.publish)
            await provider.submit_frame(
                input_frame(
                    session,
                    sequence=0,
                    utterance_id=utterance_id,
                    end_of_utterance=True,
                )
            )
            await asyncio.wait_for(
                collector.started.wait(),
                timeout=1,
            )
            await provider.cancel_utterance(
                CancelUtterance(
                    session_id=session.session_id,
                    direction_id=session.direction_id,
                    utterance_id=utterance_id,
                    reason=CancelReason.USER_INTERRUPT,
                )
            )
            collector.release.set()
            await provider.wait_idle()
            event_sequences = [
                event.event_sequence for event in collector.events
            ]
            assert event_sequences == sorted(set(event_sequences))
            assert [
                type(event) for event in collector.events
            ] == [
                ProviderAudioDelta,
                ProviderUtteranceFinal,
            ]
            assert collector.events[-1].final_audio_sequence == 0
            assert (
                collector.events[-1].event_sequence
                > collector.events[0].event_sequence
            )
        finally:
            collector.release.set()
            await provider.shutdown()

    run(scenario())


def test_publisher_failure_terminates_session_without_model_error_mapping(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def scenario() -> None:
        marker = "private-publisher-failure-marker"
        attempts: list[tuple[object, ...]] = []

        async def failing_publish(
            batch: tuple[object, ...],
            commit,
        ) -> None:
            attempts.append(batch)
            raise RuntimeError(marker)

        provider, asr, translator, tts = build_provider()
        session = request(AudioDirection.MICROPHONE)
        caplog.set_level(logging.DEBUG)
        try:
            await provider.open_session(session, failing_publish)
            await provider.submit_frame(
                input_frame(
                    session,
                    sequence=0,
                    utterance_id=uuid4(),
                    end_of_utterance=True,
                )
            )
            with pytest.raises(
                LocalProviderPublicationError,
                match="publication",
            ) as raised:
                await provider.wait_idle()
            await asyncio.wait_for(
                provider.wait_publications(session.session_id),
                timeout=0.2,
            )
            with pytest.raises(
                LocalProviderProtocolError,
                match="session",
            ):
                await provider.wait_publications(session.session_id)
            assert len(attempts) == 1
            assert len(attempts[0]) == 1
            assert isinstance(
                attempts[0][0], ProviderAudioDelta
            )
            assert not any(
                isinstance(event, PrivacySafeProviderError)
                for batch in attempts
                for event in batch
            )
            assert len(asr.calls) == 1
            assert len(translator.calls) == 1
            assert len(tts.calls) == 1
            assert marker not in caplog.text
            rendered = "".join(
                traceback.format_exception(
                    type(raised.value),
                    raised.value,
                    raised.value.__traceback__,
                )
            )
            assert marker not in rendered
            with pytest.raises(
                LocalProviderProtocolError,
                match="session",
            ):
                await provider.submit_frame(
                    input_frame(
                        session,
                        sequence=1,
                        utterance_id=uuid4(),
                        end_of_utterance=True,
                    )
                )
        finally:
            await provider.shutdown()

    run(scenario())


def test_failed_shared_model_rejects_already_queued_native_work() -> None:
    async def scenario() -> None:
        started = ThreadEvent()
        release = ThreadEvent()
        second_admission = asyncio.Event()
        admissions = 0

        def admission_probe(kind: ModelKind) -> None:
            nonlocal admissions
            if kind is ModelKind.ASR:
                admissions += 1
                if admissions == 2:
                    second_admission.set()

        asr = FakeAsr(
            started=started,
            release=release,
            failure=RuntimeError("private-asr-failure"),
        )
        provider, _, translator, tts = build_provider(
            asr=asr,
            model_admission_probe=admission_probe,
        )
        first = request(AudioDirection.MICROPHONE)
        second = request(AudioDirection.SPEAKER)
        first_events = Collector()
        second_events = Collector()
        try:
            await provider.open_session(first, first_events.publish)
            await provider.open_session(second, second_events.publish)
            await provider.submit_frame(
                input_frame(
                    first,
                    sequence=0,
                    utterance_id=uuid4(),
                    end_of_utterance=True,
                )
            )
            assert await asyncio.to_thread(started.wait, 1)
            await provider.submit_frame(
                input_frame(
                    second,
                    sequence=0,
                    utterance_id=uuid4(),
                    end_of_utterance=True,
                )
            )
            await asyncio.wait_for(
                second_admission.wait(),
                timeout=1,
            )
            release.set()
            await provider.wait_idle()
            assert len(asr.calls) == 1
            assert translator.calls == []
            assert tts.calls == []
            failed = await provider.health(second.session_id)
            assert failed.state is ProviderState.UNAVAILABLE
            assert failed.models[0].state is ModelState.FAILED
            assert all(
                events.events[-1].outcome
                is UtteranceOutcome.DROPPED
                for events in (first_events, second_events)
            )
        finally:
            release.set()
            await provider.shutdown()

    run(scenario())


def test_reopen_failed_session_does_not_replay_stale_publication_error() -> None:
    async def scenario() -> None:
        async def failing_publish(
            batch: tuple[object, ...],
            commit,
        ) -> None:
            raise RuntimeError("private-publisher-failure")

        provider, _, _, _ = build_provider()
        session = request(AudioDirection.MICROPHONE)
        try:
            await provider.open_session(session, failing_publish)
            await provider.submit_frame(
                input_frame(
                    session,
                    sequence=0,
                    utterance_id=uuid4(),
                    end_of_utterance=True,
                )
            )
            async with asyncio.timeout(1):
                while True:
                    try:
                        await provider.open_session(
                            session, Collector().publish
                        )
                    except LocalProviderProtocolError:
                        await asyncio.sleep(0)
                        continue
                    break
            await provider.wait_idle()
            health = await provider.health(session.session_id)
            assert health.state in {
                ProviderState.READY,
                ProviderState.DEGRADED,
            }
        finally:
            await provider.shutdown()

    run(scenario())


def test_failed_publication_errors_obey_completion_ledger_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        monkeypatch.setattr(
            local_provider_module,
            "_MAX_RETIRED_SESSIONS",
            2,
        )

        async def failing_publish(
            batch: tuple[object, ...],
            commit,
        ) -> None:
            raise RuntimeError("private-publisher-failure")

        provider, _, _, _ = build_provider()
        sessions = [
            request(AudioDirection.MICROPHONE) for _ in range(3)
        ]
        try:
            for session in sessions:
                await provider.open_session(session, failing_publish)
                await provider.submit_frame(
                    input_frame(
                        session,
                        sequence=0,
                        utterance_id=uuid4(),
                        end_of_utterance=True,
                    )
                )
                async with asyncio.timeout(1):
                    while True:
                        try:
                            await provider.health(session.session_id)
                        except LocalProviderProtocolError:
                            break
                        await asyncio.sleep(0)
            errors = 0
            for _ in range(3):
                try:
                    await provider.wait_idle()
                except LocalProviderPublicationError:
                    errors += 1
            assert errors == 2
        finally:
            await provider.shutdown()

    run(scenario())


def test_empty_asr_result_drops_only_utterance_without_model_failure() -> None:
    async def scenario() -> None:
        provider, asr, translator, tts = build_provider(
            asr=FakeAsr(result="  ")
        )
        session = request(AudioDirection.MICROPHONE)
        collector = Collector()
        try:
            await provider.open_session(session, collector.publish)
            await provider.submit_frame(
                input_frame(
                    session,
                    sequence=0,
                    utterance_id=uuid4(),
                    end_of_utterance=True,
                )
            )
            await provider.wait_idle()
            assert len(asr.calls) == 1
            assert translator.calls == []
            assert tts.calls == []
            assert [
                type(event) for event in collector.events
            ] == [
                ProviderLatency,
                PrivacySafeProviderError,
                ProviderUtteranceFinal,
            ]
            latency, error, final = collector.events
            assert latency.asr_first_text_ms is not None
            assert latency.asr_final_text_ms is not None
            assert latency.mt_first_text_ms is None
            assert latency.tts_first_audio_ms is None
            assert error.code.value == "no_speech"
            assert (
                final.outcome is UtteranceOutcome.DROPPED
            )
            health = await provider.health(session.session_id)
            assert health.state in {
                ProviderState.READY,
                ProviderState.DEGRADED,
            }
            assert health.models[0].state is ModelState.READY
        finally:
            await provider.shutdown()

    run(scenario())


def test_health_reports_cold_models_and_queued_source_truthfully() -> None:
    async def scenario() -> None:
        clock = MutableClock(1_000_000_000)
        started = ThreadEvent()
        release = ThreadEvent()
        asr = FakeAsr(started=started, release=release)
        asr.resident_model_id = None
        tts = FakeTts()
        tts.model_state = ModelState.NOT_LOADED
        provider, _, _, _ = build_provider(
            asr=asr,
            tts=tts,
            now_ns=clock,
        )
        first = request(AudioDirection.MICROPHONE)
        second = request(AudioDirection.MICROPHONE)
        first_events = Collector()
        second_events = Collector()
        try:
            _, cold = await provider.open_session(
                first, first_events.publish
            )
            await provider.open_session(second, second_events.publish)
            assert cold.state is ProviderState.STARTING
            assert [model.state for model in cold.models] == [
                ModelState.NOT_LOADED,
                ModelState.READY,
                ModelState.NOT_LOADED,
            ]
            await provider.submit_frame(
                input_frame(
                    first,
                    sequence=0,
                    utterance_id=uuid4(),
                    end_of_utterance=True,
                    capture_monotonic_ns=1_000_000_000,
                )
            )
            assert await asyncio.to_thread(started.wait, 1)
            loading = await provider.health(first.session_id)
            assert loading.models[0].state is ModelState.LOADING
            await provider.submit_frame(
                input_frame(
                    second,
                    sequence=0,
                    utterance_id=uuid4(),
                    end_of_utterance=True,
                    capture_monotonic_ns=1_000_000_000,
                )
            )
            clock.set(1_250_000_000)
            queued = await provider.health(second.session_id)
            assert queued.queues.provider_input_buffered_ms == 100
            assert queued.queues.queue_lag_ms == 250
            release.set()
            await provider.wait_idle()
            warm = await provider.health(second.session_id)
            assert warm.models[0].state is ModelState.READY
            assert warm.models[2].state is ModelState.READY
            third = request(AudioDirection.MICROPHONE)
            _, shared_warm = await provider.open_session(
                third, Collector().publish
            )
            assert shared_warm.models[0].state is ModelState.READY
            assert shared_warm.models[2].state is ModelState.READY
        finally:
            release.set()
            await provider.shutdown()

    run(scenario())


def test_local_provider_routes_actual_callbacks_through_source_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        calls = {"finalize_async": 0, "stream_once": 0}
        boundary = {"mt": False, "tts": False}
        real_source_commit = local_provider_module.SourceCommit

        class TrackingSourceCommit(real_source_commit):
            async def finalize_async(self, *args, **kwargs):
                calls["finalize_async"] += 1
                boundary["mt"] = True
                try:
                    return await super().finalize_async(
                        *args, **kwargs
                    )
                finally:
                    boundary["mt"] = False

            async def stream_once(self, *args, **kwargs):
                calls["stream_once"] += 1
                boundary["tts"] = True
                try:
                    async for frame in super().stream_once(
                        *args, **kwargs
                    ):
                        yield frame
                finally:
                    boundary["tts"] = False

        monkeypatch.setattr(
            local_provider_module,
            "SourceCommit",
            TrackingSourceCommit,
        )
        provider, _, _, _ = build_provider(
            translator=FakeTranslator(
                require_boundary=lambda: boundary["mt"]
            ),
            tts=FakeTts(
                require_boundary=lambda: boundary["tts"]
            ),
        )
        session = request(AudioDirection.MICROPHONE)
        collector = Collector()
        try:
            await provider.open_session(session, collector.publish)
            await provider.submit_frame(
                input_frame(
                    session,
                    sequence=0,
                    utterance_id=uuid4(),
                    end_of_utterance=True,
                )
            )
            await provider.wait_idle()
            assert calls == {
                "finalize_async": 1,
                "stream_once": 1,
            }
        finally:
            await provider.shutdown()

    run(scenario())


def test_terminal_id_capacity_fails_closed_without_eviction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        monkeypatch.setattr(
            local_provider_module,
            "_MAX_TERMINAL_IDS",
            2,
        )
        provider, _, _, _ = build_provider()
        session = request(AudioDirection.MICROPHONE)
        collector = Collector()
        utterance_ids = [uuid4(), uuid4(), uuid4()]
        try:
            await provider.open_session(session, collector.publish)
            for sequence, utterance_id in enumerate(
                utterance_ids[:2]
            ):
                await provider.submit_frame(
                    input_frame(
                        session,
                        sequence=sequence,
                        utterance_id=utterance_id,
                        end_of_utterance=False,
                    )
                )
                await provider.cancel_utterance(
                    CancelUtterance(
                        session_id=session.session_id,
                        direction_id=session.direction_id,
                        utterance_id=utterance_id,
                        reason=CancelReason.USER_INTERRUPT,
                    )
                )
                await provider.wait_publications(
                    session.session_id
                )
            with pytest.raises(
                LocalProviderProtocolError,
                match="terminal",
            ):
                await provider.submit_frame(
                    input_frame(
                        session,
                        sequence=2,
                        utterance_id=utterance_ids[2],
                        end_of_utterance=False,
                    )
                )
            with pytest.raises(
                LocalProviderProtocolError,
                match="session",
            ):
                await provider.health(session.session_id)
            assert len(collector.events) == 2
        finally:
            await provider.shutdown()

    run(scenario())


def test_terminal_capacity_cancels_other_active_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        monkeypatch.setattr(
            local_provider_module,
            "_MAX_TERMINAL_IDS",
            1,
        )
        started = ThreadEvent()
        release = ThreadEvent()
        provider, _, translator, tts = build_provider(
            asr=FakeAsr(started=started, release=release)
        )
        session = request(AudioDirection.MICROPHONE)
        collector = Collector()
        try:
            await provider.open_session(session, collector.publish)
            await provider.submit_frame(
                input_frame(
                    session,
                    sequence=0,
                    utterance_id=uuid4(),
                    end_of_utterance=True,
                )
            )
            assert await asyncio.to_thread(started.wait, 1)
            with pytest.raises(
                LocalProviderProtocolError,
                match="terminal",
            ):
                await provider.submit_frame(
                    input_frame(
                        session,
                        sequence=1,
                        utterance_id=uuid4(),
                        end_of_utterance=False,
                    )
                )
            release.set()
            await provider.wait_idle()
            assert translator.calls == []
            assert tts.calls == []
            assert collector.events == []
            with pytest.raises(
                LocalProviderProtocolError,
                match="session",
            ):
                await provider.health(session.session_id)
        finally:
            release.set()
            await provider.shutdown()

    run(scenario())


def test_retired_session_completion_ledger_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        monkeypatch.setattr(
            local_provider_module,
            "_MAX_RETIRED_SESSIONS",
            2,
            raising=False,
        )
        provider, _, _, _ = build_provider()
        sessions = [
            request(AudioDirection.MICROPHONE) for _ in range(3)
        ]
        collector_refs = []
        try:
            for session in sessions:
                collector = Collector()
                await provider.open_session(
                    session, collector.publish
                )
                collector_refs.append(weakref.ref(collector))
                await provider.close_session(
                    CloseProviderSession(
                        session_id=session.session_id,
                        reason=CloseRequestReason.USER_STOP,
                    )
                )
            await provider.wait_idle()
            with pytest.raises(
                LocalProviderProtocolError,
                match="session",
            ):
                await provider.wait_publications(
                    sessions[0].session_id
                )
            for session in sessions[1:]:
                await provider.wait_publications(session.session_id)
            del collector
            gc.collect()
            assert collector_refs[0]() is None
        finally:
            await provider.shutdown()

    run(scenario())


def test_total_publication_capacity_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        monkeypatch.setattr(
            local_provider_module,
            "_MAX_PENDING_EVENTS",
            4,
        )
        provider, asr, _, _ = build_provider()
        session = request(
            AudioDirection.MICROPHONE,
            mode=TranslationMode.STREAMING_FIRST,
        )
        collector = GatedCollector()
        sequence = 0
        try:
            await provider.open_session(session, collector.publish)
            for utterance_index in range(2):
                utterance_id = uuid4()
                for frame_index in range(301):
                    await provider.submit_frame(
                        input_frame(
                            session,
                            sequence=sequence,
                            utterance_id=utterance_id,
                            end_of_utterance=frame_index == 300,
                        )
                    )
                    sequence += 1
                if utterance_index == 0:
                    await asyncio.wait_for(
                        collector.started.wait(),
                        timeout=1,
                    )
            collector.release.set()
            with pytest.raises(LocalProviderPublicationError):
                await provider.wait_idle()
            assert asr.calls == []
            with pytest.raises(
                LocalProviderProtocolError,
                match="session",
            ):
                await provider.health(session.session_id)
        finally:
            collector.release.set()
            await provider.shutdown()

    run(scenario())


def test_close_releases_session_id_for_reopen() -> None:
    async def scenario() -> None:
        provider, _, _, _ = build_provider()
        session = request(AudioDirection.MICROPHONE)
        collector = Collector()
        try:
            await provider.open_session(session, collector.publish)
            await provider.close_session(
                CloseProviderSession(
                    session_id=session.session_id,
                    reason=CloseRequestReason.USER_STOP,
                )
            )
            await provider.wait_publications(session.session_id)
            reopened, _ = await provider.open_session(
                session, Collector().publish
            )
            assert reopened.session_id == session.session_id
        finally:
            await provider.shutdown()

    run(scenario())


def assert_session_sequence(
    opened: ProviderSessionOpened,
    health: ProviderHealth,
    utterance_id: UUID,
    events: list[object],
) -> None:
    sequence = [
        opened.event_sequence,
        health.event_sequence,
        *(event.event_sequence for event in events),
    ]
    assert sequence == list(range(1, len(sequence) + 1))
    assert all(
        event.session_id == opened.session_id
        and event.direction_id == opened.direction_id
        and event.utterance_id == utterance_id
        for event in events
    )


class MutableClock:
    def __init__(self, value: int) -> None:
        self.value = value

    def set(self, value: int) -> None:
        self.value = value

    def __call__(self) -> int:
        return self.value

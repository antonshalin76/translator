import asyncio
from collections.abc import Iterator
import os
import signal
import stat
import time
from typing import Any
from pathlib import Path
from uuid import UUID, uuid4

import grpc
import pytest

import translator_sidecar.__main__ as main_module
from translator_sidecar.generated.translator.provider.v1 import (
    provider_pb2,
    provider_pb2_grpc,
)
from translator_sidecar.provider_contract import (
    AudioDirection,
    CloseRequestReason,
    ModelHealth,
    ModelKind,
    ModelState,
    ProviderCapabilities,
    ProviderHealth,
    ProviderId,
    ProviderQueues,
    ProviderRetry,
    ProviderSessionOpened,
    ProviderState,
    ProviderUtteranceFinal,
    SafeErrorCode,
    SafeErrorSummary,
    UtteranceOutcome,
    make_provider_error,
)
from translator_sidecar.provider_engine import MockInjection, ProviderEngine
from translator_sidecar.local.inference_scheduler import InferenceScheduler
from translator_sidecar.local.local_provider import LocalProvider
from translator_sidecar.provider_contract import (
    ComputeDevice,
    Language,
    TranslationMode,
    VoiceProfile,
)
from translator_sidecar.grpc_server import (
    AUTH_METADATA_KEY,
    BoundedChannel,
    ChannelOverflow,
    ProviderGrpcServer,
    SidecarServerConfig,
    _event_to_proto,
)

TOKEN = "ab" * 32


class LocalGrpcAsr:
    actual_device = "cuda"
    degraded = False
    unavailable = False
    resident_model_id = "small"

    def __init__(self) -> None:
        self.calls: list[tuple[bytes, Language, TranslationMode]] = []

    def transcribe(
        self,
        pcm: bytes,
        *,
        language: Language,
        mode: TranslationMode,
    ) -> str:
        self.calls.append((pcm, language, mode))
        return {
            Language.RU: "russian source",
            Language.EN: "english source",
        }[language]


class LocalGrpcTranslator:
    unavailable = False

    def __init__(self) -> None:
        self.calls: list[tuple[str, Language, Language, TranslationMode]] = []

    def translate(
        self,
        text: str,
        *,
        source_language: Language,
        target_language: Language,
        mode: TranslationMode,
    ) -> str:
        self.calls.append((text, source_language, target_language, mode))
        return {
            (Language.RU, Language.EN): "english translation",
            (Language.EN, Language.RU): "russian translation",
        }[(source_language, target_language)]

    @staticmethod
    def count_tokens(text: str) -> int:
        return len(text.split())


class LocalGrpcTts:
    unavailable = False

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

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
    ) -> Iterator[bytes]:
        self.calls.append(
            {
                "text": text,
                "target_language": target_language,
                "voice_profile": voice_profile,
                "mode": mode,
            }
        )
        if cancelled():
            return
        frame_bytes = (
            output_sample_rate_hz * output_channels * frame_duration_ms // 1000 * 2
        )
        marker = b"\x11" if target_language is Language.EN else b"\x22"
        yield marker * frame_bytes


def local_provider_fixture(
    *,
    now_ns=lambda: 1_000_000,
) -> tuple[
    LocalProvider,
    LocalGrpcAsr,
    LocalGrpcTranslator,
    LocalGrpcTts,
]:
    asr = LocalGrpcAsr()
    translator = LocalGrpcTranslator()
    tts = LocalGrpcTts()
    provider = LocalProvider(
        asr=asr,
        translator=translator,
        tts=tts,
        scheduler=InferenceScheduler(),
        now_ns=now_ns,
        asr_model_id="faster-whisper-small",
        mt_model_id="nllb-200-distilled-600m-ct2-int8",
        tts_model_id="piper-medium",
        mt_device=ComputeDevice.CUDA,
    )
    return provider, asr, translator, tts


def run(coroutine):
    return asyncio.run(coroutine)


def open_request(
    session_id: UUID,
    direction: int = provider_pb2.AUDIO_DIRECTION_MICROPHONE,
    *,
    debug_text_enabled: bool = False,
    provider_id: int = provider_pb2.PROVIDER_ID_LOCAL,
    voice_engine: int = provider_pb2.VOICE_ENGINE_PIPER,
) -> provider_pb2.ProviderRequest:
    source, target = (
        (provider_pb2.LANGUAGE_RU, provider_pb2.LANGUAGE_EN)
        if direction == provider_pb2.AUDIO_DIRECTION_MICROPHONE
        else (provider_pb2.LANGUAGE_EN, provider_pb2.LANGUAGE_RU)
    )
    pcm_format = provider_pb2.PcmFormat(
        sample_rate_hz=16_000,
        channels=1,
        sample_format=provider_pb2.SAMPLE_FORMAT_S16LE,
        frame_duration_ms=100,
    )
    return provider_pb2.ProviderRequest(
        open_session=provider_pb2.OpenProviderSession(
            schema_version="translator.provider.open_session.v1",
            session_id=str(session_id),
            provider_id=provider_id,
            direction_id=direction,
            source_language=source,
            target_language=target,
            mode=provider_pb2.TRANSLATION_MODE_QUALITY_FIRST,
            requested_input_format=pcm_format,
            requested_output_format=pcm_format,
            voice_profile=provider_pb2.VoiceProfile(
                language=target,
                gender=provider_pb2.VOICE_GENDER_MALE,
                engine=voice_engine,
            ),
            debug_text_enabled=debug_text_enabled,
        )
    )


def frame_request(
    session_id: UUID,
    direction: int = provider_pb2.AUDIO_DIRECTION_MICROPHONE,
    *,
    sequence: int = 0,
    utterance_id: UUID | None = None,
    capture_monotonic_ns: int = 0,
    pcm: bytes = b"\x01\x02" * 1600,
) -> provider_pb2.ProviderRequest:
    source, target = (
        (provider_pb2.LANGUAGE_RU, provider_pb2.LANGUAGE_EN)
        if direction == provider_pb2.AUDIO_DIRECTION_MICROPHONE
        else (provider_pb2.LANGUAGE_EN, provider_pb2.LANGUAGE_RU)
    )
    return provider_pb2.ProviderRequest(
        input_frame=provider_pb2.ProviderInputFrame(
            schema_version="translator.provider.input.v1",
            session_id=str(session_id),
            direction_id=direction,
            stream_id=str(UUID(int=direction)),
            utterance_id=str(utterance_id or uuid4()),
            sequence=sequence,
            capture_monotonic_ns=capture_monotonic_ns,
            sample_rate_hz=16_000,
            channels=1,
            sample_format=provider_pb2.SAMPLE_FORMAT_S16LE,
            frame_duration_ms=100,
            source_language=source,
            target_language=target,
            mode=provider_pb2.TRANSLATION_MODE_QUALITY_FIRST,
            pcm=pcm,
            end_of_utterance=True,
        )
    )


def close_request(session_id: UUID) -> provider_pb2.ProviderRequest:
    return provider_pb2.ProviderRequest(
        close_session=provider_pb2.CloseProviderSession(
            schema_version="translator.provider.close_session.v1",
            session_id=str(session_id),
            reason=provider_pb2.CLOSE_REQUEST_REASON_USER_STOP,
        )
    )


async def requests(*items: provider_pb2.ProviderRequest):
    for item in items:
        yield item


class InteractiveRequests:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[provider_pb2.ProviderRequest | None] = (
            asyncio.Queue()
        )

    async def send(self, request: provider_pb2.ProviderRequest) -> None:
        await self._queue.put(request)

    async def close(self) -> None:
        await self._queue.put(None)

    async def __aiter__(self):
        while True:
            request = await self._queue.get()
            if request is None:
                return
            yield request


def secure_config(tmp_path: Path) -> SidecarServerConfig:
    parent = tmp_path / "sidecar"
    parent.mkdir(mode=0o700, parents=True)
    return SidecarServerConfig(
        socket_path=parent / "provider.sock",
        token=TOKEN,
        generation_id=uuid4(),
        now_ns=lambda: 0,
    )


async def collect_stream(
    config: SidecarServerConfig,
    *items: provider_pb2.ProviderRequest,
    metadata: tuple[tuple[str, str], ...] | None = None,
) -> list[provider_pb2.ProviderEvent]:
    channel = grpc.aio.insecure_channel(f"unix://{config.socket_path}")
    try:
        stub = provider_pb2_grpc.ProviderTransportStub(channel)
        effective_metadata = (
            ((AUTH_METADATA_KEY, f"Bearer {config.token}"),)
            if metadata is None
            else metadata
        )
        call = stub.Stream(
            requests(*items),
            metadata=effective_metadata,
            timeout=2,
        )
        return await asyncio.wait_for(
            _collect_events(call),
            timeout=3,
        )
    finally:
        await channel.close()


async def _collect_events(call) -> list[provider_pb2.ProviderEvent]:
    return [event async for event in call]


async def _read_remaining(call) -> list[provider_pb2.ProviderEvent]:
    events = []
    while True:
        event = await call.read()
        if event is grpc.aio.EOF:
            return events
        events.append(event)


def test_probe_requires_auth_and_returns_matching_generation(tmp_path: Path) -> None:
    async def scenario() -> None:
        config = secure_config(tmp_path)
        server = ProviderGrpcServer(config)
        await server.start()
        channel = grpc.aio.insecure_channel(f"unix://{config.socket_path}")
        try:
            stub = provider_pb2_grpc.ProviderTransportStub(channel)
            response = await stub.Probe(
                provider_pb2.ProviderProbeRequest(
                    schema_version="translator.provider.probe_request.v1"
                ),
                metadata=((AUTH_METADATA_KEY, f"Bearer {TOKEN}"),),
                timeout=1,
            )
            assert response.schema_version == "translator.provider.probe_response.v1"
            assert response.generation_id == str(config.generation_id)
            assert response.provider_ready is True
            with pytest.raises(grpc.aio.AioRpcError) as missing:
                await stub.Probe(
                    provider_pb2.ProviderProbeRequest(),
                    timeout=1,
                )
            assert missing.value.code() is grpc.StatusCode.UNAUTHENTICATED
            for schema_version in ("", "private-probe-schema-marker"):
                with pytest.raises(grpc.aio.AioRpcError) as invalid:
                    await stub.Probe(
                        provider_pb2.ProviderProbeRequest(
                            schema_version=schema_version
                        ),
                        metadata=((AUTH_METADATA_KEY, f"Bearer {TOKEN}"),),
                        timeout=1,
                    )
                assert invalid.value.code() is grpc.StatusCode.INVALID_ARGUMENT
                assert "private-probe-schema-marker" not in invalid.value.details()
        finally:
            await channel.close()
            await server.stop()

    run(scenario())


@pytest.mark.parametrize(
    "metadata",
    [
        ((AUTH_METADATA_KEY, "Bearer wrong"),),
        ((AUTH_METADATA_KEY, f"bearer {TOKEN}"),),
        ((AUTH_METADATA_KEY, f"Bearer  {TOKEN}"),),
        (
            (AUTH_METADATA_KEY, f"Bearer {TOKEN}"),
            (AUTH_METADATA_KEY, f"Bearer {TOKEN}"),
        ),
    ],
    ids=["wrong", "scheme", "whitespace", "duplicate"],
)
def test_invalid_probe_auth_uses_the_same_fail_closed_policy(
    tmp_path: Path, metadata: tuple[tuple[str, str], ...]
) -> None:
    async def scenario() -> None:
        config = secure_config(tmp_path)
        server = ProviderGrpcServer(config)
        await server.start()
        channel = grpc.aio.insecure_channel(f"unix://{config.socket_path}")
        try:
            stub = provider_pb2_grpc.ProviderTransportStub(channel)
            with pytest.raises(grpc.aio.AioRpcError) as rejected:
                await stub.Probe(
                    provider_pb2.ProviderProbeRequest(
                        schema_version="translator.provider.probe_request.v1"
                    ),
                    metadata=metadata,
                    timeout=1,
                )
            assert rejected.value.code() is grpc.StatusCode.UNAUTHENTICATED
        finally:
            await channel.close()
            await server.stop()

    run(scenario())


@pytest.mark.parametrize(
    "metadata",
    [
        (),
        ((AUTH_METADATA_KEY, "Bearer wrong"),),
        ((AUTH_METADATA_KEY, f"bearer {TOKEN}"),),
        ((AUTH_METADATA_KEY, f"Bearer  {TOKEN}"),),
        (
            (AUTH_METADATA_KEY, f"Bearer {TOKEN}"),
            (AUTH_METADATA_KEY, f"Bearer {TOKEN}"),
        ),
    ],
    ids=["missing", "wrong", "scheme", "whitespace", "duplicate"],
)
def test_invalid_stream_auth_is_rejected_before_request_iteration(
    tmp_path: Path, metadata: tuple[tuple[str, str], ...]
) -> None:
    async def scenario() -> None:
        config = secure_config(tmp_path)
        server = ProviderGrpcServer(config)
        await server.start()
        try:
            with pytest.raises(grpc.aio.AioRpcError) as rejected:
                await collect_stream(
                    config,
                    open_request(uuid4()),
                    metadata=metadata,
                )
            assert rejected.value.code() is grpc.StatusCode.UNAUTHENTICATED
            assert server.consumed_request_count == 0
        finally:
            await server.stop()

    run(scenario())


def test_uds_parent_and_existing_inode_checks_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    insecure_parent = tmp_path / "insecure"
    insecure_parent.mkdir(mode=0o755)
    insecure = SidecarServerConfig(
        socket_path=insecure_parent / "provider.sock",
        token=TOKEN,
        generation_id=uuid4(),
        now_ns=lambda: 0,
    )
    with pytest.raises(PermissionError, match="0700"):
        run(ProviderGrpcServer(insecure).start())

    config = secure_config(tmp_path)
    config.socket_path.write_text("foreign")
    with pytest.raises(FileExistsError):
        run(ProviderGrpcServer(config).start())
    assert config.socket_path.read_text() == "foreign"

    symlink_parent_target = tmp_path / "real-parent"
    symlink_parent_target.mkdir(mode=0o700)
    symlink_parent = tmp_path / "linked-parent"
    symlink_parent.symlink_to(symlink_parent_target, target_is_directory=True)
    linked_config = SidecarServerConfig(
        socket_path=symlink_parent / "provider.sock",
        token=TOKEN,
        generation_id=uuid4(),
        now_ns=lambda: 0,
    )
    with pytest.raises(PermissionError, match="real directory"):
        run(ProviderGrpcServer(linked_config).start())

    real_parent = tmp_path / "socket-symlink-parent"
    real_parent.mkdir(mode=0o700)
    target = tmp_path / "missing-target"
    socket_symlink = real_parent / "provider.sock"
    socket_symlink.symlink_to(target)
    symlink_config = SidecarServerConfig(
        socket_path=socket_symlink,
        token=TOKEN,
        generation_id=uuid4(),
        now_ns=lambda: 0,
    )
    with pytest.raises(FileExistsError):
        run(ProviderGrpcServer(symlink_config).start())
    assert socket_symlink.is_symlink()

    parent_file = tmp_path / "not-a-directory"
    parent_file.write_text("file")
    file_parent_config = SidecarServerConfig(
        socket_path=parent_file / "provider.sock",
        token=TOKEN,
        generation_id=uuid4(),
        now_ns=lambda: 0,
    )
    with pytest.raises(NotADirectoryError):
        run(ProviderGrpcServer(file_parent_config).start())

    owner_config = secure_config(tmp_path / "owner")
    monkeypatch.setattr(
        os, "getuid", lambda: owner_config.socket_path.parent.stat().st_uid + 1
    )
    with pytest.raises(PermissionError, match="owner"):
        run(ProviderGrpcServer(owner_config).start())


def test_server_binds_private_socket_and_rejects_invalid_config(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        config = secure_config(tmp_path)
        server = ProviderGrpcServer(config)
        await server.start()
        try:
            assert config.socket_path.stat().st_mode & 0o777 == 0o600
            assert stat.S_ISSOCK(config.socket_path.stat().st_mode)
            assert config.socket_path.stat().st_uid == os.getuid()
        finally:
            await server.stop()

    run(scenario())
    for token in ("a" * 63, "a" * 65, "A" * 64, "g" * 64):
        with pytest.raises(ValueError, match="token"):
            SidecarServerConfig(
                socket_path=tmp_path / "bad.sock",
                token=token,
                generation_id=uuid4(),
                now_ns=lambda: 0,
            )


def test_stream_enforces_first_open_and_single_session_identity(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        config = secure_config(tmp_path)
        server = ProviderGrpcServer(config)
        await server.start()
        try:
            with pytest.raises(grpc.aio.AioRpcError) as before_open:
                await collect_stream(config, frame_request(uuid4()))
            assert before_open.value.code() is grpc.StatusCode.INVALID_ARGUMENT
            assert before_open.value.details() == "protocol_error"

            first_id = uuid4()
            with pytest.raises(grpc.aio.AioRpcError) as second_open:
                await collect_stream(
                    config,
                    open_request(first_id),
                    open_request(uuid4()),
                )
            assert second_open.value.code() is grpc.StatusCode.INVALID_ARGUMENT
            assert second_open.value.details() == "protocol_error"

            with pytest.raises(grpc.aio.AioRpcError) as foreign:
                await collect_stream(
                    config,
                    open_request(first_id),
                    frame_request(uuid4()),
                )
            assert foreign.value.code() is grpc.StatusCode.INVALID_ARGUMENT
            assert foreign.value.details() == "protocol_error"

            for session_id in (first_id,):
                reopened = await collect_stream(
                    config,
                    open_request(session_id),
                    close_request(session_id),
                )
                assert reopened[-1].HasField("session_closed")
                reopened_after_close = await collect_stream(
                    config,
                    open_request(session_id),
                    close_request(session_id),
                )
                assert reopened_after_close[-1].HasField("session_closed")

            eof_id = uuid4()
            eof_events = await collect_stream(config, open_request(eof_id))
            assert eof_events[0].HasField("session_opened")
            reopened_after_eof = await collect_stream(
                config,
                open_request(eof_id),
                close_request(eof_id),
            )
            assert reopened_after_eof[-1].HasField("session_closed")
        finally:
            await server.stop()

    run(scenario())


def test_stream_rejects_empty_and_wrong_schema_requests_without_leaks(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        config = secure_config(tmp_path)
        server = ProviderGrpcServer(config)
        await server.start()
        try:
            marker = "private-wire-schema-marker"
            cases: list[tuple[UUID | None, list[provider_pb2.ProviderRequest]]] = []
            cases.append((None, [provider_pb2.ProviderRequest()]))

            open_id = uuid4()
            wrong_open = open_request(open_id)
            wrong_open.open_session.schema_version = marker
            cases.append((open_id, [wrong_open]))

            frame_id = uuid4()
            wrong_frame = frame_request(frame_id)
            wrong_frame.input_frame.schema_version = marker
            cases.append((frame_id, [open_request(frame_id), wrong_frame]))

            update_id = uuid4()
            wrong_update = provider_pb2.ProviderRequest(
                update_debug_text=provider_pb2.UpdateDebugText(
                    schema_version=marker,
                    session_id=str(update_id),
                    enabled=True,
                )
            )
            cases.append((update_id, [open_request(update_id), wrong_update]))

            close_id = uuid4()
            wrong_close = close_request(close_id)
            wrong_close.close_session.schema_version = marker
            cases.append((close_id, [open_request(close_id), wrong_close]))

            for session_id, request_items in cases:
                with pytest.raises(grpc.aio.AioRpcError) as rejected:
                    await collect_stream(config, *request_items)
                assert rejected.value.code() is grpc.StatusCode.INVALID_ARGUMENT
                assert rejected.value.details() == "invalid_request"
                assert marker not in rejected.value.details()
                if session_id is not None:
                    reopened = await collect_stream(
                        config,
                        open_request(session_id),
                        close_request(session_id),
                    )
                    assert reopened[-1].HasField("session_closed")
        finally:
            await server.stop()

    run(scenario())


def test_two_streams_emit_independent_ordered_audio_and_close(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        config = secure_config(tmp_path)
        server = ProviderGrpcServer(config)
        await server.start()
        channel = grpc.aio.insecure_channel(f"unix://{config.socket_path}")
        try:
            microphone_id = uuid4()
            speaker_id = uuid4()
            microphone_requests = InteractiveRequests()
            speaker_requests = InteractiveRequests()
            stub = provider_pb2_grpc.ProviderTransportStub(channel)
            metadata = ((AUTH_METADATA_KEY, f"Bearer {TOKEN}"),)
            microphone_call = stub.Stream(
                microphone_requests,
                metadata=metadata,
                timeout=3,
            )
            speaker_call = stub.Stream(
                speaker_requests,
                metadata=metadata,
                timeout=3,
            )
            await microphone_requests.send(open_request(microphone_id))
            await speaker_requests.send(
                open_request(speaker_id, provider_pb2.AUDIO_DIRECTION_SPEAKER)
            )
            microphone_opened, speaker_opened = await asyncio.wait_for(
                asyncio.gather(
                    microphone_call.read(),
                    speaker_call.read(),
                ),
                timeout=2,
            )
            assert microphone_opened.HasField("session_opened")
            assert speaker_opened.HasField("session_opened")

            await microphone_requests.send(frame_request(microphone_id))
            await speaker_requests.send(
                frame_request(speaker_id, provider_pb2.AUDIO_DIRECTION_SPEAKER)
            )
            await microphone_requests.send(close_request(microphone_id))
            await speaker_requests.send(close_request(speaker_id))
            await microphone_requests.close()
            await speaker_requests.close()
            microphone_tail, speaker_tail = await asyncio.wait_for(
                asyncio.gather(
                    _read_remaining(microphone_call),
                    _read_remaining(speaker_call),
                ),
                timeout=3,
            )
            microphone = [microphone_opened, *microphone_tail]
            speaker = [speaker_opened, *speaker_tail]
            for events, session_id in (
                (microphone, microphone_id),
                (speaker, speaker_id),
            ):
                assert events[0].WhichOneof("event") == "session_opened"
                assert any(event.HasField("audio_delta") for event in events)
                assert events[-1].WhichOneof("event") == "session_closed"
                sequences = [
                    getattr(event, event.WhichOneof("event")).event_sequence
                    for event in events
                ]
                assert sequences == sorted(set(sequences))
                assert {
                    getattr(event, event.WhichOneof("event")).session_id
                    for event in events
                } == {str(session_id)}
            assert server.created_channel_capacities == [
                ("control", 64),
                ("event", 64),
                ("control", 64),
                ("event", 64),
            ]
        finally:
            await channel.close()
            await server.stop()

    run(scenario())


def test_authenticated_duplex_stream_uses_local_provider_pipeline(
    tmp_path: Path,
) -> None:
    async def read_open(call) -> list[provider_pb2.ProviderEvent]:
        opened = [await call.read(), await call.read()]
        assert [event.WhichOneof("event") for event in opened] == [
            "session_opened",
            "health",
        ]
        return opened

    async def read_utterance(call) -> list[provider_pb2.ProviderEvent]:
        events = []
        while True:
            event = await call.read()
            assert event is not grpc.aio.EOF
            events.append(event)
            if event.HasField("utterance_final"):
                return events

    async def scenario() -> None:
        config = secure_config(tmp_path)
        provider, asr, translator, tts = local_provider_fixture()
        server = ProviderGrpcServer(config, local_provider=provider)
        await server.start()
        channel = grpc.aio.insecure_channel(f"unix://{config.socket_path}")
        microphone_requests = InteractiveRequests()
        speaker_requests = InteractiveRequests()
        try:
            stub = provider_pb2_grpc.ProviderTransportStub(channel)
            metadata = ((AUTH_METADATA_KEY, f"Bearer {TOKEN}"),)
            microphone_call = stub.Stream(
                microphone_requests,
                metadata=metadata,
                timeout=5,
            )
            speaker_call = stub.Stream(
                speaker_requests,
                metadata=metadata,
                timeout=5,
            )
            microphone_id = uuid4()
            speaker_id = uuid4()
            await microphone_requests.send(open_request(microphone_id))
            await speaker_requests.send(
                open_request(
                    speaker_id,
                    provider_pb2.AUDIO_DIRECTION_SPEAKER,
                )
            )
            microphone_open, speaker_open = await asyncio.gather(
                read_open(microphone_call),
                read_open(speaker_call),
            )

            microphone_utterance = uuid4()
            speaker_utterance = uuid4()
            await microphone_requests.send(
                frame_request(
                    microphone_id,
                    utterance_id=microphone_utterance,
                    pcm=b"\x31\x32" * 1600,
                )
            )
            await speaker_requests.send(
                frame_request(
                    speaker_id,
                    provider_pb2.AUDIO_DIRECTION_SPEAKER,
                    utterance_id=speaker_utterance,
                    pcm=b"\x41\x42" * 1600,
                )
            )
            microphone_events, speaker_events = await asyncio.gather(
                read_utterance(microphone_call),
                read_utterance(speaker_call),
            )

            for events, opened, session_id, utterance_id, marker in (
                (
                    microphone_events,
                    microphone_open,
                    microphone_id,
                    microphone_utterance,
                    b"\x11",
                ),
                (
                    speaker_events,
                    speaker_open,
                    speaker_id,
                    speaker_utterance,
                    b"\x22",
                ),
            ):
                assert [event.WhichOneof("event") for event in events] == [
                    "audio_delta",
                    "latency",
                    "utterance_final",
                ]
                audio = [
                    event.audio_delta
                    for event in events
                    if event.HasField("audio_delta")
                ]
                assert len(audio) == 1
                assert audio[0].pcm == marker * 3200
                assert audio[0].sequence == 0
                terminal = events[-1].utterance_final
                assert terminal.utterance_id == str(utterance_id)
                assert terminal.final_audio_sequence == 0
                assert terminal.outcome == (provider_pb2.UTTERANCE_OUTCOME_COMPLETED)
                latency = events[-2].latency
                assert latency.session_id == str(session_id)
                assert latency.utterance_id == str(utterance_id)
                combined = [*opened, *events]
                sequences = [
                    getattr(event, event.WhichOneof("event")).event_sequence
                    for event in combined
                ]
                assert sequences == sorted(set(sequences))
                assert {
                    getattr(event, event.WhichOneof("event")).session_id
                    for event in combined
                } == {str(session_id)}

            await microphone_requests.send(close_request(microphone_id))
            await speaker_requests.send(close_request(speaker_id))
            await microphone_requests.close()
            await speaker_requests.close()
            microphone_closed, speaker_closed = await asyncio.gather(
                microphone_call.read(),
                speaker_call.read(),
            )
            assert microphone_closed.HasField("session_closed")
            assert speaker_closed.HasField("session_closed")
            assert await microphone_call.read() is grpc.aio.EOF
            assert await speaker_call.read() is grpc.aio.EOF

            assert {(call[0], call[1]) for call in asr.calls} == {
                (b"\x31\x32" * 1600, Language.RU),
                (b"\x41\x42" * 1600, Language.EN),
            }
            assert {(call[1], call[2]) for call in translator.calls} == {
                (Language.RU, Language.EN),
                (Language.EN, Language.RU),
            }
            assert {call["target_language"] for call in tts.calls} == {
                Language.RU,
                Language.EN,
            }

            reopened_microphone, reopened_speaker = await asyncio.gather(
                collect_stream(
                    config,
                    open_request(microphone_id),
                    close_request(microphone_id),
                ),
                collect_stream(
                    config,
                    open_request(
                        speaker_id,
                        provider_pb2.AUDIO_DIRECTION_SPEAKER,
                    ),
                    close_request(speaker_id),
                ),
            )
            assert reopened_microphone[-1].HasField("session_closed")
            assert reopened_speaker[-1].HasField("session_closed")
        finally:
            await microphone_requests.close()
            await speaker_requests.close()
            await channel.close()
            await server.stop()

    run(scenario())


def test_provider_swap_keeps_existing_session_on_original_backend(
    tmp_path: Path,
) -> None:
    async def open_stream(
        stub,
        request_stream,
        session_id,
        direction=provider_pb2.AUDIO_DIRECTION_MICROPHONE,
    ):
        call = stub.Stream(
            request_stream,
            metadata=((AUTH_METADATA_KEY, f"Bearer {TOKEN}"),),
            timeout=5,
        )
        await request_stream.send(open_request(session_id, direction))
        assert (await call.read()).HasField("session_opened")
        assert (await call.read()).HasField("health")
        return call

    async def complete_utterance(call) -> None:
        kinds = []
        while "utterance_final" not in kinds:
            event = await call.read()
            assert event is not grpc.aio.EOF
            kinds.append(event.WhichOneof("event"))
        assert kinds == [
            "audio_delta",
            "latency",
            "utterance_final",
        ]

    async def scenario() -> None:
        base = secure_config(tmp_path)
        processing_gate = asyncio.Event()
        processing_gate.set()
        config = SidecarServerConfig(
            socket_path=base.socket_path,
            token=base.token,
            generation_id=base.generation_id,
            now_ns=base.now_ns,
            control_processing_gate=processing_gate,
        )
        original, original_asr, _, _ = local_provider_fixture()
        replacement, replacement_asr, _, _ = local_provider_fixture()
        server = ProviderGrpcServer(config, local_provider=original)
        await server.start()
        channel = grpc.aio.insecure_channel(f"unix://{config.socket_path}")
        original_requests = InteractiveRequests()
        replacement_requests = InteractiveRequests()
        reopened_requests = InteractiveRequests()
        try:
            stub = provider_pb2_grpc.ProviderTransportStub(channel)
            original_id = uuid4()
            original_call = await open_stream(
                stub,
                original_requests,
                original_id,
            )
            replacement_call = stub.Stream(
                replacement_requests,
                metadata=((AUTH_METADATA_KEY, f"Bearer {TOKEN}"),),
                timeout=5,
            )
            replacement_id = uuid4()
            processing_gate.clear()
            await replacement_requests.send(
                open_request(
                    replacement_id,
                    provider_pb2.AUDIO_DIRECTION_SPEAKER,
                )
            )

            async def wait_until_consumed() -> None:
                while server.consumed_request_count < 2:
                    await asyncio.sleep(0.001)

            await asyncio.wait_for(wait_until_consumed(), timeout=1)
            await server.replace_local_provider(replacement)
            processing_gate.set()
            assert (await replacement_call.read()).HasField("session_opened")
            assert (await replacement_call.read()).HasField("health")
            original_pcm = b"\x51\x52" * 1600
            replacement_pcm = b"\x61\x62" * 1600
            await original_requests.send(frame_request(original_id, pcm=original_pcm))
            await replacement_requests.send(
                frame_request(
                    replacement_id,
                    provider_pb2.AUDIO_DIRECTION_SPEAKER,
                    pcm=replacement_pcm,
                )
            )
            await asyncio.gather(
                complete_utterance(original_call),
                complete_utterance(replacement_call),
            )

            assert {(call[0], call[1]) for call in original_asr.calls} == {
                (original_pcm, Language.RU)
            }
            assert {(call[0], call[1]) for call in replacement_asr.calls} == {
                (replacement_pcm, Language.EN)
            }

            await original_requests.close()
            assert await original_call.read() is grpc.aio.EOF
            for _ in range(100):
                if original._closed:
                    break
                await asyncio.sleep(0)
            assert original._closed is True
            assert replacement._closed is False

            reopened_call = await open_stream(
                stub,
                reopened_requests,
                original_id,
            )
            await reopened_requests.send(close_request(original_id))
            await reopened_requests.close()
            assert (await reopened_call.read()).HasField("session_closed")

            await replacement_requests.send(close_request(replacement_id))
            await replacement_requests.close()
            assert (await replacement_call.read()).HasField("session_closed")
        finally:
            await original_requests.close()
            await replacement_requests.close()
            await reopened_requests.close()
            await channel.close()
            await server.stop()
        assert original._closed is True
        assert replacement._closed is True

    run(scenario())


def test_production_bootstrap_injects_local_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert callable(main_module.build_unavailable_local_provider)
    assert callable(main_module._build_server)
    config = secure_config(tmp_path)
    local_provider = object()
    openai_provider = object()
    captured: dict[str, object] = {}

    def fake_unavailable_local_provider(*, now_ns):
        assert now_ns is config.now_ns
        return local_provider

    def fake_openai_provider(openai_config, *, now_ns):
        assert openai_config.cloud_opt_in is True
        assert now_ns is config.now_ns
        return openai_provider

    class FakeServer:
        def __init__(
            self,
            received_config,
            *,
            local_provider,
            openai_provider,
            provider_ready,
        ) -> None:
            captured["config"] = received_config
            captured["local_provider"] = local_provider
            captured["openai_provider"] = openai_provider
            captured["provider_ready"] = provider_ready

    monkeypatch.setattr(
        main_module,
        "build_unavailable_local_provider",
        fake_unavailable_local_provider,
    )
    monkeypatch.setattr(
        main_module,
        "OpenAIRealtimeProvider",
        fake_openai_provider,
    )
    monkeypatch.setattr(main_module, "ProviderGrpcServer", FakeServer)

    server = main_module._build_server(config)

    assert isinstance(server, FakeServer)
    assert captured == {
        "config": config,
        "local_provider": local_provider,
        "openai_provider": openai_provider,
        "provider_ready": False,
    }


def test_serve_delegates_to_production_server_builder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert callable(main_module._build_server)
    config = secure_config(tmp_path)
    handlers = []
    built_configs = []
    order = []
    loaded_provider = object()

    class LoopProxy:
        def add_signal_handler(self, caught_signal, callback) -> None:
            handlers.append((caught_signal, callback))

    class FakeServer:
        async def start(self) -> None:
            order.append("start")
            assert {item[0] for item in handlers} == {
                signal.SIGINT,
                signal.SIGTERM,
            }

        async def replace_local_provider(self, provider) -> None:
            assert provider is loaded_provider
            order.append("replace")
            handlers[0][1]()

        async def stop(self) -> None:
            order.append("stop")
            built_configs.append("stopped")

    def fake_build_server(received_config):
        built_configs.append(received_config)
        return FakeServer()

    class UnexpectedDirectServer:
        def __init__(self, *args, **kwargs) -> None:
            raise AssertionError("serve bypassed production server builder")

    def fake_build_local_provider(*, now_ns):
        assert order == ["start", "to_thread"]
        assert callable(now_ns)
        order.append("build")
        return loaded_provider

    async def fake_to_thread(function, **kwargs):
        order.append("to_thread")
        return function(**kwargs)

    monkeypatch.setattr(
        main_module.asyncio,
        "get_running_loop",
        lambda: LoopProxy(),
    )
    monkeypatch.setattr(main_module, "_build_server", fake_build_server)
    monkeypatch.setattr(
        main_module,
        "build_local_provider",
        fake_build_local_provider,
    )
    monkeypatch.setattr(main_module.asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr(
        main_module,
        "ProviderGrpcServer",
        UnexpectedDirectServer,
    )
    monkeypatch.setenv(
        "TRANSLATOR_SIDECAR_SOCKET",
        str(config.socket_path),
    )
    monkeypatch.setenv("TRANSLATOR_SIDECAR_TOKEN", config.token)
    monkeypatch.setenv(
        "TRANSLATOR_SIDECAR_GENERATION",
        str(config.generation_id),
    )

    run(asyncio.wait_for(main_module._serve(), timeout=1))

    assert len(built_configs) == 2
    built_config = built_configs[0]
    assert isinstance(built_config, SidecarServerConfig)
    assert built_config.socket_path == config.socket_path
    assert built_config.token == config.token
    assert built_config.generation_id == config.generation_id
    assert built_configs[1] == "stopped"
    assert order == [
        "start",
        "to_thread",
        "build",
        "replace",
        "stop",
    ]


@pytest.mark.parametrize(
    ("failure_stage", "shutdown_raises"),
    [
        ("build", False),
        ("replace", False),
        ("replace", True),
    ],
)
def test_serve_cleans_up_after_model_activation_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
    shutdown_raises: bool,
) -> None:
    config = secure_config(tmp_path)
    handlers = []
    order = []

    class LoopProxy:
        def add_signal_handler(self, caught_signal, callback) -> None:
            handlers.append((caught_signal, callback))

    class BuiltProvider:
        shutdown_count = 0

        async def shutdown(self) -> None:
            self.shutdown_count += 1
            order.append("provider_shutdown")
            if shutdown_raises:
                raise RuntimeError("safe-provider-shutdown-failure")

    built_provider = BuiltProvider()

    class FakeServer:
        async def start(self) -> None:
            order.append("start")

        async def replace_local_provider(self, provider) -> None:
            assert provider is built_provider
            order.append("replace")
            raise RuntimeError("safe-replace-failure")

        async def stop(self) -> None:
            order.append("stop")

    def fake_build_server(received_config):
        assert received_config.socket_path == config.socket_path
        return FakeServer()

    def fake_build_local_provider(*, now_ns):
        assert callable(now_ns)
        order.append("build")
        if failure_stage == "build":
            raise RuntimeError("safe-build-failure")
        return built_provider

    async def fake_to_thread(function, **kwargs):
        return function(**kwargs)

    monkeypatch.setattr(
        main_module.asyncio,
        "get_running_loop",
        lambda: LoopProxy(),
    )
    monkeypatch.setattr(main_module.asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr(main_module, "_build_server", fake_build_server)
    monkeypatch.setattr(
        main_module,
        "build_local_provider",
        fake_build_local_provider,
    )
    monkeypatch.setenv(
        "TRANSLATOR_SIDECAR_SOCKET",
        str(config.socket_path),
    )
    monkeypatch.setenv("TRANSLATOR_SIDECAR_TOKEN", config.token)
    monkeypatch.setenv(
        "TRANSLATOR_SIDECAR_GENERATION",
        str(config.generation_id),
    )

    with pytest.raises(
        RuntimeError,
        match=f"safe-{failure_stage}-failure",
    ):
        run(asyncio.wait_for(main_module._serve(), timeout=1))

    if failure_stage == "build":
        assert order == ["start", "build", "stop"]
        assert built_provider.shutdown_count == 0
    else:
        assert order == [
            "start",
            "build",
            "replace",
            "provider_shutdown",
            "stop",
        ]
        assert built_provider.shutdown_count == 1


def test_runtime_debug_disable_suppresses_text_events(tmp_path: Path) -> None:
    async def scenario() -> None:
        config = secure_config(tmp_path)
        server = ProviderGrpcServer(config)
        await server.start()
        try:
            disabled_id = uuid4()
            preserved_id = uuid4()
            first_utterance_id = uuid4()
            second_utterance_id = uuid4()
            disabled, preserved = await asyncio.wait_for(
                asyncio.gather(
                    collect_stream(
                        config,
                        open_request(disabled_id, debug_text_enabled=True),
                        frame_request(
                            disabled_id,
                            sequence=0,
                            utterance_id=first_utterance_id,
                        ),
                        provider_pb2.ProviderRequest(
                            update_debug_text=provider_pb2.UpdateDebugText(
                                schema_version=(
                                    "translator.provider.update_debug_text.v1"
                                ),
                                session_id=str(disabled_id),
                                enabled=False,
                            )
                        ),
                        frame_request(
                            disabled_id,
                            sequence=1,
                            utterance_id=second_utterance_id,
                        ),
                        close_request(disabled_id),
                    ),
                    collect_stream(
                        config,
                        open_request(preserved_id, debug_text_enabled=True),
                        frame_request(preserved_id),
                        close_request(preserved_id),
                    ),
                ),
                timeout=3,
            )
            disabled_text = [
                event
                for event in disabled
                if event.WhichOneof("event")
                in {"transcript_delta", "translation_delta"}
            ]
            preserved_text = [
                event
                for event in preserved
                if event.WhichOneof("event")
                in {"transcript_delta", "translation_delta"}
            ]
            assert len(disabled_text) == 2
            assert len(preserved_text) == 2
            assert all(
                getattr(event, event.WhichOneof("event")).utterance_id
                == str(first_utterance_id)
                for event in disabled_text
            )
            second_audio = [
                event.audio_delta
                for event in disabled
                if event.HasField("audio_delta")
                and event.audio_delta.utterance_id == str(second_utterance_id)
            ]
            second_final = [
                event.utterance_final
                for event in disabled
                if event.HasField("utterance_final")
                and event.utterance_final.utterance_id == str(second_utterance_id)
            ]
            assert len(second_audio) == 1
            assert len(second_final) == 1
            assert all(
                getattr(event, event.WhichOneof("event")).utterance_id
                != str(second_utterance_id)
                for event in disabled_text
            )
        finally:
            await server.stop()

    run(scenario())


def test_delayed_frame_completes_without_a_follow_up_request(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        base = secure_config(tmp_path)
        config = SidecarServerConfig(
            socket_path=base.socket_path,
            token=base.token,
            generation_id=base.generation_id,
            now_ns=time.monotonic_ns,
        )
        server = ProviderGrpcServer(
            config,
            engine=ProviderEngine(injection=MockInjection(process_delay_ms=50)),
        )
        await server.start()
        try:
            session_id = uuid4()
            events = await collect_stream(
                config,
                open_request(session_id),
                frame_request(
                    session_id,
                    capture_monotonic_ns=time.monotonic_ns(),
                ),
                close_request(session_id),
            )
            assert any(event.HasField("audio_delta") for event in events)
            assert any(event.HasField("utterance_final") for event in events)
        finally:
            await server.stop()

    run(scenario())


class FakeOpenAIGrpcProvider:
    def __init__(self) -> None:
        self.opens = []
        self.frames = []
        self.closes = []
        self.waits = []
        self.cancelled = []
        self.debug_updates = []
        self.shutdown_count = 0

    async def open_session(self, request, publish):
        self.opens.append((request, publish))
        opened = ProviderSessionOpened(
            session_id=request.session_id,
            direction_id=request.direction_id,
            event_sequence=1,
            negotiated_input_format=request.requested_input_format,
            negotiated_output_format=request.requested_output_format,
            capabilities=ProviderCapabilities(
                transcript_delta=False,
                translation_delta=False,
                cancellation=True,
                cloud_egress=True,
            ),
        )
        health = ProviderHealth(
            session_id=request.session_id,
            direction_id=request.direction_id,
            event_sequence=2,
            provider_id=ProviderId.OPENAI,
            provider_name="openai-realtime-translation",
            state=ProviderState.READY,
            models=(
                ModelHealth(
                    kind=ModelKind.SPEECH_TO_SPEECH,
                    id="gpt-realtime-translate",
                    state=ModelState.READY,
                    device=ComputeDevice.CLOUD,
                ),
            ),
            queues=ProviderQueues(
                provider_input_buffered_ms=0,
                provider_output_buffered_ms=0,
                queue_lag_ms=0,
            ),
        )
        return opened, health

    async def submit_frame(self, frame) -> None:
        self.frames.append(frame)

    async def cancel_utterance(self, request) -> None:
        self.cancelled.append(request)

    async def update_debug_text(self, request) -> None:
        self.debug_updates.append(request)

    async def close_session(self, request) -> None:
        self.closes.append(request)

    async def wait_publications(self, session_id) -> None:
        self.waits.append(session_id)

    async def shutdown(self) -> None:
        self.shutdown_count += 1


def test_openai_stream_dispatches_to_openai_provider_when_local_provider_is_loaded(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        config = secure_config(tmp_path)
        local_provider, asr, _, _ = local_provider_fixture()
        openai_provider = FakeOpenAIGrpcProvider()
        server = ProviderGrpcServer(
            config,
            local_provider=local_provider,
            openai_provider=openai_provider,
        )
        await server.start()
        try:
            session_id = uuid4()
            events = await collect_stream(
                config,
                open_request(
                    session_id,
                    provider_id=provider_pb2.PROVIDER_ID_OPENAI,
                    voice_engine=provider_pb2.VOICE_ENGINE_OPENAI,
                ),
                frame_request(session_id),
                close_request(session_id),
            )
            kinds = [event.WhichOneof("event") for event in events]
            assert kinds[:2] == ["session_opened", "health"]
            assert events[0].session_opened.capabilities.cloud_egress is True
            assert events[0].session_opened.negotiated_input_format.sample_rate_hz == 16_000
            assert events[1].health.provider_id == provider_pb2.PROVIDER_ID_OPENAI
            assert [item[0].provider_id for item in openai_provider.opens] == [
                ProviderId.OPENAI
            ]
            assert len(openai_provider.frames) == 1
            assert len(openai_provider.closes) == 1
            assert openai_provider.waits == [session_id]
            assert not asr.calls
        finally:
            await server.stop()

    run(scenario())


class OverflowEngine(ProviderEngine):
    def enqueue_frame(self, frame, *, now_ns):
        return (
            make_provider_error(
                session_id=frame.session_id,
                direction_id=frame.direction_id,
                event_sequence=3,
                code=SafeErrorCode.QUEUE_OVERFLOW,
                retryable=True,
                utterance_id=frame.utterance_id,
            ),
            ProviderUtteranceFinal(
                session_id=frame.session_id,
                direction_id=frame.direction_id,
                stream_id=frame.stream_id,
                utterance_id=frame.utterance_id,
                event_sequence=4,
                final_audio_sequence=None,
                outcome=UtteranceOutcome.DROPPED,
            ),
        )


def test_no_speech_error_serializes_to_stable_proto() -> None:
    session_id = uuid4()
    utterance_id = uuid4()
    event = make_provider_error(
        session_id=session_id,
        direction_id=AudioDirection.MICROPHONE,
        event_sequence=3,
        code=SafeErrorCode.NO_SPEECH,
        retryable=True,
        stream_id=UUID(int=1),
        utterance_id=utterance_id,
    )

    encoded = _event_to_proto(event).error

    assert encoded.code == provider_pb2.SAFE_ERROR_CODE_NO_SPEECH == 8
    assert encoded.safe_message == "No speech was detected"
    assert encoded.session_id == str(session_id)
    assert encoded.stream_id == str(UUID(int=1))
    assert encoded.utterance_id == str(utterance_id)


class RetryHealthEngine(ProviderEngine):
    def health(self, session_id, *, now_ns):
        return (
            super()
            .health(session_id, now_ns=now_ns)
            .model_copy(
                update={
                    "retry": ProviderRetry(
                        attempt=2,
                        next_retry_after_ms=250,
                        reason_code="mock_retry",
                    ),
                    "safe_error": SafeErrorSummary(
                        code=SafeErrorCode.PROVIDER_UNAVAILABLE,
                        message="Provider is unavailable",
                        retryable=True,
                    ),
                }
            )
        )


def test_stream_health_preserves_retry_and_safe_error(tmp_path: Path) -> None:
    async def scenario() -> None:
        config = secure_config(tmp_path)
        server = ProviderGrpcServer(config, engine=RetryHealthEngine())
        await server.start()
        try:
            session_id = uuid4()
            events = await collect_stream(
                config,
                open_request(session_id),
                close_request(session_id),
            )
            health_event = next(
                event.health for event in events if event.HasField("health")
            )
            assert health_event.retry.attempt == 2
            assert health_event.retry.next_retry_after_ms == 250
            assert health_event.retry.reason_code == "mock_retry"
            assert health_event.safe_error.code == "provider_unavailable"
            assert health_event.safe_error.message == "Provider is unavailable"
            assert health_event.safe_error.retryable is True
        finally:
            await server.stop()

    run(scenario())


class RaisingEngine(ProviderEngine):
    def enqueue_frame(self, frame, *, now_ns):
        raise RuntimeError("private-unexpected-engine-marker")


def test_unexpected_engine_failure_is_private_and_releases_session(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        config = secure_config(tmp_path)
        server = ProviderGrpcServer(config, engine=RaisingEngine())
        await server.start()
        try:
            session_id = uuid4()
            with pytest.raises(grpc.aio.AioRpcError) as failed:
                await collect_stream(
                    config,
                    open_request(session_id),
                    frame_request(session_id),
                )
            assert failed.value.code() is grpc.StatusCode.INTERNAL
            assert failed.value.details() == "internal_error"
            assert "private-unexpected-engine-marker" not in failed.value.details()

            reopened = await collect_stream(
                config,
                open_request(session_id),
                close_request(session_id),
            )
            assert reopened[-1].HasField("session_closed")
        finally:
            await server.stop()

    run(scenario())


@pytest.mark.parametrize(
    ("capacity", "expected_status"),
    [(2, None), (1, grpc.StatusCode.RESOURCE_EXHAUSTED)],
)
def test_server_atomically_delivers_terminal_pair_or_exhausts(
    tmp_path: Path,
    capacity: int,
    expected_status: grpc.StatusCode | None,
) -> None:
    async def scenario() -> None:
        config = secure_config(tmp_path)
        config = SidecarServerConfig(
            socket_path=config.socket_path,
            token=config.token,
            generation_id=config.generation_id,
            now_ns=config.now_ns,
            channel_capacity=capacity,
        )
        server = ProviderGrpcServer(config, engine=OverflowEngine())
        await server.start()
        try:
            session_id = uuid4()
            if expected_status is None:
                events = await collect_stream(
                    config,
                    open_request(session_id),
                    frame_request(session_id),
                )
                kinds = [event.WhichOneof("event") for event in events]
                assert kinds[-2:] == ["error", "utterance_final"]
                assert (
                    events[-2].error.code == provider_pb2.SAFE_ERROR_CODE_QUEUE_OVERFLOW
                )
            else:
                with pytest.raises(grpc.aio.AioRpcError) as exhausted:
                    await collect_stream(
                        config,
                        open_request(session_id),
                        frame_request(session_id),
                    )
                assert exhausted.value.code() is expected_status
                assert exhausted.value.details() == "resource_exhausted"
        finally:
            await server.stop()

    run(scenario())


def test_control_channel_applies_backpressure_without_aborting_valid_burst(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        base = secure_config(tmp_path)
        processing_gate = asyncio.Event()
        processing_gate.set()
        config = SidecarServerConfig(
            socket_path=base.socket_path,
            token=base.token,
            generation_id=base.generation_id,
            now_ns=base.now_ns,
            channel_capacity=1,
            control_processing_gate=processing_gate,
        )
        server = ProviderGrpcServer(config)
        await server.start()
        channel = grpc.aio.insecure_channel(f"unix://{config.socket_path}")
        request_stream = InteractiveRequests()
        session_id = uuid4()
        try:
            stub = provider_pb2_grpc.ProviderTransportStub(channel)
            call = stub.Stream(
                request_stream,
                metadata=((AUTH_METADATA_KEY, f"Bearer {TOKEN}"),),
                timeout=3,
            )
            await request_stream.send(open_request(session_id))
            opened = await asyncio.wait_for(call.read(), timeout=1)
            assert opened.HasField("session_opened")
            health_event = await asyncio.wait_for(call.read(), timeout=1)
            assert health_event.HasField("health")
            processing_gate.clear()
            await request_stream.send(frame_request(session_id))
            await request_stream.send(
                provider_pb2.ProviderRequest(
                    update_debug_text=provider_pb2.UpdateDebugText(
                        schema_version="translator.provider.update_debug_text.v1",
                        session_id=str(session_id),
                        enabled=False,
                    )
                )
            )
            await request_stream.send(close_request(session_id))
            await asyncio.sleep(0.05)
            processing_gate.set()
            while True:
                event = await asyncio.wait_for(call.read(), timeout=2)
                if event.HasField("session_closed"):
                    break
        finally:
            processing_gate.set()
            await request_stream.close()
            await channel.close()
            await server.stop()

    run(scenario())


def test_client_cancellation_releases_open_session(tmp_path: Path) -> None:
    async def scenario() -> None:
        config = secure_config(tmp_path)
        server = ProviderGrpcServer(config)
        await server.start()
        channel = grpc.aio.insecure_channel(f"unix://{config.socket_path}")
        request_stream = InteractiveRequests()
        session_id = uuid4()
        try:
            stub = provider_pb2_grpc.ProviderTransportStub(channel)
            call = stub.Stream(
                request_stream,
                metadata=((AUTH_METADATA_KEY, f"Bearer {TOKEN}"),),
                timeout=3,
            )
            await request_stream.send(open_request(session_id))
            opened = await asyncio.wait_for(call.read(), timeout=1)
            assert opened.HasField("session_opened")
            assert call.cancel() is True
            await request_stream.close()
            await asyncio.sleep(0)

            reopened = await collect_stream(
                config,
                open_request(session_id),
                close_request(session_id),
            )
            assert reopened[-1].HasField("session_closed")
        finally:
            await channel.close()
            await server.stop()

    run(scenario())


def test_server_stop_cancellation_finishes_stop_and_clears_owned_state(
    tmp_path: Path,
) -> None:
    class GatedServer:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.finished = False

        async def stop(self, grace: int) -> None:
            assert grace == 0
            self.started.set()
            await self.release.wait()
            self.finished = True

    async def scenario() -> None:
        config = secure_config(tmp_path)
        server = ProviderGrpcServer(config)
        fake = GatedServer()
        parent_fd = os.open(config.socket_path.parent, os.O_RDONLY)
        server._server = fake  # type: ignore[assignment]
        server._parent_fd = parent_fd

        stopping = asyncio.create_task(server.stop())
        await fake.started.wait()
        stopping.cancel()
        fake.release.set()
        with pytest.raises(asyncio.CancelledError):
            await stopping

        assert fake.finished is True
        assert server._server is None
        assert server._parent_fd is None
        with pytest.raises(OSError):
            os.fstat(parent_fd)

    run(scenario())


def test_server_stop_defers_repeated_cancellation_until_stop_finishes(
    tmp_path: Path,
) -> None:
    class GatedServer:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.finished = False

        async def stop(self, grace: int) -> None:
            assert grace == 0
            self.started.set()
            await self.release.wait()
            self.finished = True

    async def scenario() -> None:
        config = secure_config(tmp_path)
        server = ProviderGrpcServer(config)
        fake = GatedServer()
        parent_fd = os.open(config.socket_path.parent, os.O_RDONLY)
        server._server = fake  # type: ignore[assignment]
        server._parent_fd = parent_fd

        stopping = asyncio.create_task(server.stop())
        await fake.started.wait()
        stopping.cancel()
        await asyncio.sleep(0)
        stopping.cancel()
        await asyncio.sleep(0)
        released_early = (
            stopping.done() or server._server is None or server._parent_fd is None
        )
        fake.release.set()
        with pytest.raises(asyncio.CancelledError):
            await stopping

        assert released_early is False
        assert fake.finished is True
        assert server._server is None
        assert server._parent_fd is None
        with pytest.raises(OSError):
            os.fstat(parent_fd)

    run(scenario())


class CleanupRaisingEngine(ProviderEngine):
    def __init__(self) -> None:
        super().__init__()
        self.release_calls: list[UUID] = []

    def close_session(self, request):
        if request.reason is CloseRequestReason.DAEMON_SHUTDOWN:
            raise RuntimeError("private-cleanup-close-marker")
        return super().close_session(request)

    def release_session(self, session_id):
        self.release_calls.append(session_id)
        return super().release_session(session_id)


def test_cleanup_releases_session_even_when_close_raises(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        config = secure_config(tmp_path)
        engine = CleanupRaisingEngine()
        server = ProviderGrpcServer(config, engine=engine)
        await server.start()
        session_id = uuid4()
        try:
            opened = await collect_stream(config, open_request(session_id))
            assert [event.WhichOneof("event") for event in opened] == [
                "session_opened",
                "health",
            ]
            assert engine.release_calls == [session_id]

            reopened = await collect_stream(
                config,
                open_request(session_id),
                close_request(session_id),
            )
            assert reopened[0].HasField("session_opened")
        finally:
            await server.stop()

    run(scenario())


def test_local_provider_lease_is_released_when_cleanup_raises(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def scenario() -> None:
        config = secure_config(tmp_path)
        provider, *_ = local_provider_fixture()
        original_close = provider.close_session

        async def raising_close(request):
            if request.reason is CloseRequestReason.DAEMON_SHUTDOWN:
                await original_close(request)
                raise RuntimeError("private-local-cleanup-marker")
            return await original_close(request)

        provider.close_session = raising_close  # type: ignore[method-assign]
        server = ProviderGrpcServer(config, local_provider=provider)
        await server.start()
        session_id = uuid4()
        try:
            opened = await collect_stream(config, open_request(session_id))
            assert opened[0].HasField("session_opened")
            assert server._provider_leases[id(provider)] == 0
            assert "private-local-cleanup-marker" not in caplog.text

            provider.close_session = original_close  # type: ignore[method-assign]
            reopened = await collect_stream(
                config,
                open_request(session_id),
                close_request(session_id),
            )
            assert reopened[0].HasField("session_opened")
        finally:
            await server.stop()

    run(scenario())


def test_control_and_event_channels_are_bounded_to_64() -> None:
    async def scenario() -> None:
        channel: BoundedChannel[int] = BoundedChannel()
        for value in range(64):
            channel.put_nowait(value)
        assert channel.qsize() == 64
        with pytest.raises(ChannelOverflow):
            channel.put_nowait(64)

        terminal_channel: BoundedChannel[int] = BoundedChannel()
        for value in range(62):
            terminal_channel.put_nowait(value)
        terminal_channel.put_terminal(100, 101)
        assert terminal_channel.qsize() == 64

        exhausted: BoundedChannel[int] = BoundedChannel()
        for value in range(63):
            exhausted.put_nowait(value)
        with pytest.raises(ChannelOverflow, match="resource_exhausted"):
            exhausted.put_terminal(100, 101)
        assert exhausted.qsize() == 63

    run(scenario())

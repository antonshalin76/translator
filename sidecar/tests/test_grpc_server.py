import asyncio
import gc
import inspect
import os
import signal
import stat
import time
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import grpc
import pytest
from engine_runtime_provider import EngineRuntimeProvider, MockInjection, ProviderEngine
from test_openai_runtime import FakeRealtimeWebSocket
from test_provider_contract import VOICE_OVERRIDE_CASES

import translator_sidecar.__main__ as main_module
from translator_sidecar.cleanup import finish_cleanup
from translator_sidecar.generated.translator.provider.v1 import (
    provider_pb2,
    provider_pb2_grpc,
)
from translator_sidecar.grpc_server import (
    AUTH_METADATA_KEY,
    BoundedChannel,
    ChannelOverflow,
    ProviderGrpcServer,
    SidecarServerConfig,
    _AuthInterceptor,
    _event_to_proto,
    _ProviderServicer,
)
from translator_sidecar.local.inference_scheduler import InferenceScheduler
from translator_sidecar.local.local_provider import (
    LocalProvider,
    LocalProviderPublicationError,
)
from translator_sidecar.openai_provider import OpenAIRealtimeConfig
from translator_sidecar.openai_runtime import (
    OpenAIProviderProtocolError,
    OpenAIRealtimeProvider,
)
from translator_sidecar.provider_contract import (
    AudioDirection,
    CloseProviderSession,
    CloseRequestReason,
    ComputeDevice,
    Language,
    ModelHealth,
    ModelKind,
    ModelState,
    ProviderCapabilities,
    ProviderHealth,
    ProviderId,
    ProviderQueues,
    ProviderRetry,
    ProviderSessionClosed,
    ProviderSessionOpened,
    ProviderState,
    ProviderUtteranceFinal,
    SafeErrorCode,
    SafeErrorSummary,
    SessionCloseReason,
    TranslationMode,
    UtteranceOutcome,
    VoiceProfile,
    make_provider_error,
)
from translator_sidecar.provider_registry import SessionDrainReceipt

TOKEN = "ab" * 32


@pytest.mark.parametrize("provider_id", [ProviderId.LOCAL, ProviderId.OPENAI])
@pytest.mark.parametrize("model_path,provider_voice_id", VOICE_OVERRIDE_CASES)
def test_actual_ipc_voice_override_reaches_provider_and_refuses_without_effects(
    tmp_path,
    caplog,
    provider_id,
    model_path,
    provider_voice_id,
):
    async def scenario():
        config = secure_config(tmp_path)
        local, asr, mt, tts = local_provider_fixture()
        sockets = []

        async def connect(_uri, **_options):
            socket = FakeRealtimeWebSocket()
            sockets.append(socket)
            return socket

        cloud = OpenAIRealtimeProvider(
            OpenAIRealtimeConfig(cloud_opt_in=True),
            environ={"OPENAI_API_KEY": "synthetic-secret"},
            websocket_factory=connect,
        )
        seen = {ProviderId.LOCAL: [], ProviderId.OPENAI: []}
        for key, provider in ((ProviderId.LOCAL, local), (ProviderId.OPENAI, cloud)):
            original = provider.reserve_session

            def reserve(value, publish, *, key=key, original=original):
                seen[key].append(value)
                return original(value, publish)

            provider.reserve_session = reserve
        server = ProviderGrpcServer(config, local_provider=local, openai_provider=cloud)
        await server.start()
        identity = uuid4()
        value = open_request(
            identity,
            provider_id=provider_pb2.PROVIDER_ID_LOCAL
            if provider_id is ProviderId.LOCAL
            else provider_pb2.PROVIDER_ID_OPENAI,
            voice_engine=provider_pb2.VOICE_ENGINE_PIPER
            if provider_id is ProviderId.LOCAL
            else provider_pb2.VOICE_ENGINE_OPENAI,
        )
        value.open_session.requested_input_format.frame_duration_ms = 20
        value.open_session.requested_output_format.frame_duration_ms = 20
        for field, argument in (
            ("model_path", model_path),
            ("provider_voice_id", provider_voice_id),
        ):
            if argument is not None:
                setattr(value.open_session.voice_profile, field, argument)
        events, code, detail = [], None, None
        try:
            async with grpc.aio.insecure_channel(
                f"unix://{config.socket_path}"
            ) as channel:
                call = provider_pb2_grpc.ProviderTransportStub(channel).Stream(
                    requests(value),
                    metadata=((AUTH_METADATA_KEY, f"Bearer {TOKEN}"),),
                    timeout=3,
                )
                try:
                    async for event in call:
                        events.append(event)
                except grpc.aio.AioRpcError as error:
                    code, detail = error.code(), error.details()
            async with asyncio.timeout(1):
                while any(not state.ending for state in server._streams):
                    await asyncio.sleep(0)
            await server.retry_stream_cleanup()
            assert len(seen[provider_id]) == 1
            parsed = seen[provider_id][0].voice_profile
            assert (parsed.model_path, parsed.provider_voice_id) == (
                model_path,
                provider_voice_id,
            )
            assert sum(len(items) for items in seen.values()) == 1
            assert (code, detail, events) == (
                grpc.StatusCode.INVALID_ARGUMENT,
                "protocol_error",
                [],
            )
            assert not sockets and not asr.calls and not mt.calls and not tts.calls
            assert (
                not local._sessions
                and not local._retired_sessions
                and not local._scheduler._sessions
            )
            assert (
                not cloud._sessions and not cloud._failed_opens and not cloud._opening
            )
            assert not server._streams
            assert all(
                entry.leases == 0 for entry in server.providers._entries.values()
            )
            assert not local._closed and not cloud._closed
            assert "private-voice-override" not in caplog.text
            value.open_session.voice_profile.ClearField("model_path")
            value.open_session.voice_profile.ClearField("provider_voice_id")
            repaired = await collect_stream(config, value, close_request(identity))
            assert [event.WhichOneof("event") for event in repaired] == [
                "session_opened",
                "health",
                "session_closed",
            ]
            assert repaired[-1].session_closed.session_id == str(identity)
        finally:
            await asyncio.wait_for(server.stop(), timeout=3)

    run(asyncio.wait_for(scenario(), timeout=6))


class LocalGrpcAsr:
    def close(self) -> None:
        pass

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
    def close(self) -> None:
        pass

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
    def close(self) -> None:
        pass

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
        continuation: bool = False,
    ) -> Iterator[bytes]:
        self.calls.append(
            {
                "text": text,
                "target_language": target_language,
                "voice_profile": voice_profile,
                "mode": mode,
                "continuation": continuation,
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


def engine_runtime_provider(
    config: SidecarServerConfig,
    engine: ProviderEngine | None = None,
) -> EngineRuntimeProvider:
    return EngineRuntimeProvider(engine, now_ns=config.now_ns)


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
            assert response.provider_ready is False
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

    real_ancestor = tmp_path / "real-ancestor"
    real_ancestor.mkdir(mode=0o700)
    ancestor_parent = real_ancestor / "private-parent"
    ancestor_parent.mkdir(mode=0o700)
    linked_ancestor = tmp_path / "linked-ancestor"
    linked_ancestor.symlink_to(real_ancestor, target_is_directory=True)
    ancestor_config = SidecarServerConfig(
        socket_path=linked_ancestor / ancestor_parent.name / "provider.sock",
        token=TOKEN,
        generation_id=uuid4(),
        now_ns=lambda: 0,
    )
    with pytest.raises(PermissionError, match="real directory"):
        run(ProviderGrpcServer(ancestor_config).start())

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
        assert server.provider_ready is False
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


@pytest.mark.parametrize(
    "failure",
    ["constructor", "registration", "bind_raise", "bind_zero", "start", "chmod"],
)
def test_startup_failure_rolls_back_native_server_and_parent_fd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    instances = []
    native_factory = grpc.aio.server
    register_native = provider_pb2_grpc.add_ProviderTransportServicer_to_server

    class NativeServer:
        def __init__(self, native) -> None:
            self.native = native
            self.stop_calls = []
            instances.append(self)

        def add_insecure_port(self, address: str) -> int:
            assert address.startswith("unix:/proc/self/fd/")
            if failure == "bind_raise":
                raise RuntimeError("synthetic_bind_failure")
            return 0 if failure == "bind_zero" else 1

        async def start(self) -> None:
            if failure == "start":
                raise RuntimeError("synthetic_start_failure")
            await self.native.start()

        async def stop(self, grace: float) -> None:
            self.stop_calls.append(grace)
            native = self.native
            try:
                await native.stop(grace)
            finally:
                self.native = None

    def construct_server(**kwargs):
        assert kwargs["interceptors"]
        if failure == "constructor":
            raise RuntimeError("synthetic_constructor_failure")
        return NativeServer(native_factory(**kwargs))

    def register_servicer(servicer, server) -> None:
        assert servicer is not None
        assert server in instances
        if failure == "registration":
            raise RuntimeError("synthetic_registration_failure")
        register_native(servicer, server.native)

    def chmod(*args, **kwargs) -> None:
        if failure == "chmod":
            raise PermissionError("synthetic_chmod_failure")
        os.chmod(*args, **kwargs)

    monkeypatch.setattr(grpc.aio, "server", construct_server)
    monkeypatch.setattr(
        provider_pb2_grpc,
        "add_ProviderTransportServicer_to_server",
        register_servicer,
    )
    if failure == "chmod":
        monkeypatch.setattr(os, "chmod", chmod)

    async def scenario() -> None:
        config = secure_config(tmp_path)
        warmup = native_factory(interceptors=(_AuthInterceptor(config.token),))
        try:
            await warmup.start()
            gc.collect()
            await asyncio.sleep(0)
            descriptor_count = len(os.listdir("/proc/self/fd"))
            pending_tasks = asyncio.all_tasks()
            for _ in range(100):
                server = ProviderGrpcServer(config)
                with pytest.raises((PermissionError, RuntimeError)):
                    await server.start()
                assert server._server is None
                assert server._parent_fd is None
            await asyncio.sleep(0)
            assert len(os.listdir("/proc/self/fd")) == descriptor_count
            assert asyncio.all_tasks() == pending_tasks
            expected_instances = 0 if failure == "constructor" else 100
            assert len(instances) == expected_instances
            assert all(server.stop_calls == [0] for server in instances)
        finally:
            await warmup.stop(grace=0)
            del warmup

    run(scenario())


def test_startup_rollback_defers_repeated_cancellation_until_native_stop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start_entered = asyncio.Event()
    stop_entered = asyncio.Event()
    release_stop = asyncio.Event()

    class NativeServer:
        stopped = False

        def add_insecure_port(self, address: str) -> int:
            assert address.startswith("unix:/proc/self/fd/")
            return 1

        async def start(self) -> None:
            start_entered.set()
            await asyncio.Event().wait()

        async def stop(self, grace: float) -> None:
            assert grace == 0
            stop_entered.set()
            await release_stop.wait()
            self.stopped = True

    native_server = NativeServer()
    monkeypatch.setattr(grpc.aio, "server", lambda **_: native_server)
    monkeypatch.setattr(
        provider_pb2_grpc,
        "add_ProviderTransportServicer_to_server",
        lambda *_: None,
    )

    async def scenario() -> None:
        config = secure_config(tmp_path)
        descriptor_count = len(os.listdir("/proc/self/fd"))
        pending_tasks = asyncio.all_tasks()
        server = ProviderGrpcServer(config)
        starting = asyncio.create_task(server.start())
        await start_entered.wait()
        starting.cancel()
        await stop_entered.wait()
        for _ in range(3):
            starting.cancel()
            await asyncio.sleep(0)
            assert not starting.done()
        release_stop.set()
        with pytest.raises(asyncio.CancelledError):
            await starting
        assert native_server.stopped is True
        assert server._server is None
        assert server._parent_fd is None
        await asyncio.sleep(0)
        assert len(os.listdir("/proc/self/fd")) == descriptor_count
        assert asyncio.all_tasks() == pending_tasks

    run(scenario())


def test_stream_enforces_first_open_and_single_session_identity(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        config = secure_config(tmp_path)
        server = ProviderGrpcServer(
            config,
            local_provider=engine_runtime_provider(config),
        )
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
        server = ProviderGrpcServer(
            config,
            local_provider=engine_runtime_provider(config),
        )
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
        server = ProviderGrpcServer(
            config,
            local_provider=engine_runtime_provider(config),
        )
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


def test_empty_provider_registry_fails_closed_then_accepts_installed_provider(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        config = secure_config(tmp_path)
        server = ProviderGrpcServer(config)
        assert server.provider_ready is False
        await server.start()
        try:
            session_id = uuid4()
            with pytest.raises(grpc.aio.AioRpcError) as unavailable:
                await collect_stream(config, open_request(session_id))
            assert unavailable.value.code() is grpc.StatusCode.INVALID_ARGUMENT
            assert unavailable.value.details() == "protocol_error"

            provider, *_ = local_provider_fixture()
            server.install_local_provider(provider)
            assert server.provider_ready is True
            recovered = await collect_stream(
                config,
                open_request(session_id),
                close_request(session_id),
            )
            assert [event.WhichOneof("event") for event in recovered] == [
                "session_opened",
                "health",
                "session_closed",
            ]
        finally:
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
                    pcm=b"\x00\x00" * 1600,
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
                (b"\x00\x00" * 1600, Language.EN),
            }
            assert {(call[1], call[2]) for call in translator.calls} == {
                (Language.RU, Language.EN),
                (Language.EN, Language.RU),
            }
            assert {call["target_language"] for call in tts.calls} == {
                Language.RU,
                Language.EN,
            }
            assert {
                (call["target_language"], call["continuation"]) for call in tts.calls
            } == {
                (Language.EN, True),
                (Language.RU, False),
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
            server.install_local_provider(replacement)
            await server.providers.collect()
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
        @property
        def providers(self):
            return self

        async def collect(self) -> None:
            pass

        async def start(self) -> None:
            order.append("start")
            assert {item[0] for item in handlers} == {
                signal.SIGINT,
                signal.SIGTERM,
            }

        def install_local_provider(self, provider) -> None:
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

        def install_local_provider(self, provider) -> None:
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


def test_bootstrap_cancellation_reclaims_completed_model_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario():
        config = secure_config(tmp_path)
        loading = asyncio.Event()
        release = asyncio.Event()
        order = []

        class LoopProxy:
            def add_signal_handler(self, *args):
                pass

        class BuiltProvider:
            async def shutdown(self):
                order.append("model_closed")

        class FakeServer:
            async def start(self):
                pass

            async def stop(self):
                order.append("server_stopped")

        async def load(*args, **kwargs):
            loading.set()
            await release.wait()
            order.append("model_loaded")
            return BuiltProvider()

        monkeypatch.setattr(
            main_module.asyncio, "get_running_loop", lambda: LoopProxy()
        )
        monkeypatch.setattr(main_module.asyncio, "to_thread", load)
        monkeypatch.setattr(main_module, "_build_server", lambda _: FakeServer())
        monkeypatch.setenv("TRANSLATOR_SIDECAR_SOCKET", str(config.socket_path))
        monkeypatch.setenv("TRANSLATOR_SIDECAR_TOKEN", config.token)
        monkeypatch.setenv("TRANSLATOR_SIDECAR_GENERATION", str(config.generation_id))
        serving = asyncio.create_task(main_module._serve())
        await loading.wait()
        for _ in range(3):
            serving.cancel()
            await asyncio.sleep(0)
            assert not serving.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await serving
        assert order == ["model_loaded", "model_closed", "server_stopped"]

    run(scenario())


def test_retirement_failure_after_transfer_does_not_double_close_new_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario():
        config = secure_config(tmp_path)

        class LoopProxy:
            def add_signal_handler(self, *args):
                pass

        class Backend:
            def __init__(self, fail=False):
                self.calls = 0
                self.fail = fail

            async def shutdown(self):
                self.calls += 1
                if self.fail and self.calls == 1:
                    raise RuntimeError("synthetic retired backend failure")

        original, replacement = Backend(True), Backend()
        server = ProviderGrpcServer(config, local_provider=original)

        async def start():
            pass

        async def load(*args, **kwargs):
            return replacement

        server.start = start
        monkeypatch.setattr(
            main_module.asyncio, "get_running_loop", lambda: LoopProxy()
        )
        monkeypatch.setattr(main_module.asyncio, "to_thread", load)
        monkeypatch.setattr(main_module, "_build_server", lambda _: server)
        monkeypatch.setenv("TRANSLATOR_SIDECAR_SOCKET", str(config.socket_path))
        monkeypatch.setenv("TRANSLATOR_SIDECAR_TOKEN", config.token)
        monkeypatch.setenv("TRANSLATOR_SIDECAR_GENERATION", str(config.generation_id))
        with pytest.raises(RuntimeError, match=r"^provider_shutdown_failed$"):
            await main_module._serve()
        assert original.calls == 2
        assert replacement.calls == 1
        assert not server.providers._entries

    run(scenario())


def test_runtime_debug_disable_suppresses_text_events(tmp_path: Path) -> None:
    async def scenario() -> None:
        config = secure_config(tmp_path)
        server = ProviderGrpcServer(
            config,
            local_provider=engine_runtime_provider(config),
        )
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
            local_provider=engine_runtime_provider(
                config,
                ProviderEngine(injection=MockInjection(process_delay_ms=50)),
            ),
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


@dataclass
class FakeGrpcReservation:
    provider: Any
    request: Any
    publish: Any
    task: Any = None

    async def open(self):
        return await self.provider.open_effect(self.request, self.publish)

    async def drain(self, reason):
        if self.task is None or (
            self.task.done()
            and (self.task.cancelled() or self.task.exception() is not None)
        ):
            self.task = asyncio.create_task(self._drain(reason))
        return await finish_cleanup(self.task)

    async def _drain(self, reason):
        error = await self.provider.close_effect(
            CloseProviderSession(session_id=self.request.session_id, reason=reason)
        )
        self.provider.waits.append(self.request.session_id)
        return SessionDrainReceipt(self.request.session_id, error)


class FakeOpenAIGrpcProvider:
    def __init__(self) -> None:
        self.opens = []
        self.frames = []
        self.closes = []
        self.waits = []
        self.cancelled = []
        self.debug_updates = []
        self.shutdown_count = 0

    def reserve_session(self, request, publish):
        return FakeGrpcReservation(self, request, publish)

    async def open_effect(self, request, publish):
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

    async def close_effect(self, request) -> None:
        self.closes.append(request)

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
            assert (
                events[0].session_opened.negotiated_input_format.sample_rate_hz
                == 16_000
            )
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
        server = ProviderGrpcServer(
            config,
            local_provider=engine_runtime_provider(config, RetryHealthEngine()),
        )
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
        server = ProviderGrpcServer(
            config,
            local_provider=engine_runtime_provider(config, RaisingEngine()),
        )
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
        server = ProviderGrpcServer(
            config,
            local_provider=engine_runtime_provider(config, OverflowEngine()),
        )
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
        server = ProviderGrpcServer(
            config,
            local_provider=engine_runtime_provider(config),
        )
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
        server = ProviderGrpcServer(
            config,
            local_provider=engine_runtime_provider(config),
        )
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

        async def stop(self, grace: float) -> None:
            assert grace == 0.25
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

        async def stop(self, grace: float) -> None:
            assert grace == 0.25
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
        server = ProviderGrpcServer(
            config,
            local_provider=engine_runtime_provider(config, engine),
        )
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


def test_local_provider_lease_is_retained_when_scheduler_close_fails(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def scenario() -> None:
        config = secure_config(tmp_path)
        provider, *_ = local_provider_fixture()
        original_close = provider._scheduler.close_session

        def raising_close(identity):
            original_close(identity)
            raise RuntimeError("private-local-cleanup-marker")

        provider._scheduler.close_session = raising_close
        server = ProviderGrpcServer(config, local_provider=provider)
        await server.start()
        session_id = uuid4()
        try:
            with pytest.raises(grpc.aio.AioRpcError) as raised:
                await collect_stream(config, open_request(session_id))
            assert raised.value.code() is grpc.StatusCode.INTERNAL
            (record,) = server._streams
            reservation = record.reservation
            worker = reservation.publication_task
            assert record.session_id == session_id
            assert provider._sessions[session_id] is reservation
            assert session_id not in provider._scheduler._sessions
            assert worker is not None and not worker.done()
            assert server.providers._entries[id(provider)].leases == 1
            assert "private-local-cleanup-marker" not in caplog.text

            provider._scheduler.close_session = original_close
            await server.retry_stream_cleanup()
            assert worker.done()
            assert record.receipt.session_id == session_id
            assert server.providers._entries[id(provider)].leases == 0
            assert record not in server._streams
            reopened = await collect_stream(
                config,
                open_request(session_id),
                close_request(session_id),
            )
            assert reopened[0].HasField("session_opened")
        finally:
            provider._scheduler.close_session = original_close
            await server.stop()

    run(scenario())


def test_reselecting_a_leased_provider_does_not_shutdown_the_current_backend(
    tmp_path: Path,
) -> None:
    class Provider:
        shutdown_count = 0

        async def shutdown(self) -> None:
            self.shutdown_count += 1

    async def scenario() -> None:
        original, replacement = Provider(), Provider()
        server = ProviderGrpcServer(secure_config(tmp_path), local_provider=original)
        leased = server.providers.acquire(ProviderId.LOCAL)
        server.install_local_provider(replacement)
        await server.providers.collect()
        server.install_local_provider(original)
        await server.providers.collect()
        await leased.release()
        assert original.shutdown_count == 0
        assert replacement.shutdown_count == 1
        await server.stop()
        assert server.provider_ready is False
        assert original.shutdown_count == 1

    run(scenario())


def test_shutdown_attempts_both_providers_when_local_shutdown_fails(
    tmp_path: Path,
) -> None:
    class Provider:
        def __init__(self, fail: bool = False) -> None:
            self.fail = fail
            self.shutdown_count = 0

        async def shutdown(self) -> None:
            self.shutdown_count += 1
            if self.fail:
                raise RuntimeError("synthetic-shutdown-failure")

    async def scenario() -> None:
        local, cloud = Provider(fail=True), Provider()
        server = ProviderGrpcServer(
            secure_config(tmp_path), local_provider=local, openai_provider=cloud
        )
        with pytest.raises(RuntimeError):
            await server.stop()
        assert cloud.shutdown_count == 1
        local.fail = False
        await server.stop()
        assert local.shutdown_count == 2
        assert cloud.shutdown_count == 1

    run(scenario())


def test_control_and_event_channels_are_bounded_to_64() -> None:
    async def scenario() -> None:
        channel: BoundedChannel[int] = BoundedChannel()
        for value in range(64):
            channel.put_nowait(value)
        assert channel.qsize() == 64
        with pytest.raises(ChannelOverflow):
            channel.put_nowait(64)

        batch_channel: BoundedChannel[int] = BoundedChannel()
        for value in range(62):
            batch_channel.put_nowait(value)
        batch_channel.put_many_nowait((100, 101))
        assert batch_channel.qsize() == 64

        exhausted: BoundedChannel[int] = BoundedChannel()
        for value in range(63):
            exhausted.put_nowait(value)
        with pytest.raises(ChannelOverflow, match="resource_exhausted"):
            exhausted.put_many_nowait((100, 101))
        assert exhausted.qsize() == 63

    run(scenario())


class FailedOpenCleanupSocket:
    """Only websocket effects; actual OpenAI provider owns failed-open retirement."""

    def __init__(self):
        self.transport = self
        self.cleanup_fails = True
        self.closed = False
        self.reads = 0
        self.closes = 0
        self.aborts = 0

    async def recv(self):
        self.reads += 1
        raise OSError("synthetic handshake failure")

    async def close(self):
        self.closes += 1
        if self.cleanup_fails:
            raise OSError("synthetic close failure")
        self.closed = True

    def abort(self):
        self.aborts += 1
        if self.cleanup_fails:
            raise OSError("synthetic abort failure")
        self.closed = True


def test_grpc_failed_open_pins_actual_openai_backend_until_resource_drain(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def scenario():
        config = secure_config(tmp_path)
        socket = FailedOpenCleanupSocket()
        connects = []

        async def connect_socket(uri, **options):
            connects.append(uri)
            return socket

        provider = OpenAIRealtimeProvider(
            OpenAIRealtimeConfig(cloud_opt_in=True),
            environ={"OPENAI_API_KEY": "synthetic-key"},
            websocket_factory=connect_socket,
            now_ns=lambda: 0,
        )
        shutdown_calls = []
        original_shutdown = provider.shutdown

        async def observed_shutdown():
            shutdown_calls.append(provider)
            await original_shutdown()

        provider.shutdown = observed_shutdown
        replacement = FakeOpenAIGrpcProvider()
        server = ProviderGrpcServer(config, openai_provider=provider)
        await server.start()
        session_id = uuid4()
        request = open_request(
            session_id,
            provider_id=provider_pb2.PROVIDER_ID_OPENAI,
            voice_engine=provider_pb2.VOICE_ENGINE_OPENAI,
        )
        request.open_session.requested_input_format.frame_duration_ms = 20
        request.open_session.requested_output_format.frame_duration_ms = 20
        entry = server.providers._entries[id(provider)]
        try:
            with pytest.raises(grpc.aio.AioRpcError) as error:
                await collect_stream(config, request)
            after_open = (
                len(connects),
                socket.reads,
                socket.closes,
                socket.aborts,
                socket.closed,
                entry.leases,
                session_id in provider._failed_opens,
            )
            server.providers.replace(ProviderId.OPENAI, replacement)
            collect_error = None
            try:
                await server.providers.collect()
            except RuntimeError as failure:
                collect_error = type(failure)
            after_collect = (
                entry.leases,
                len(shutdown_calls),
                replacement.shutdown_count,
                socket.closed,
                collect_error,
            )
            rpc_code = error.value.code()
        finally:
            # Repair independently of the assertions, including the old forbidden disposal.
            socket.cleanup_fails = False
            await asyncio.wait_for(server.stop(), timeout=3)

        assert rpc_code in (grpc.StatusCode.INTERNAL, grpc.StatusCode.INVALID_ARGUMENT)
        assert after_open[:2] == (1, 1), "the real OpenAI handshake must be entered"
        assert after_open[2] >= 1 and after_open[3] >= 1
        assert after_open[4:] == (False, 1, True), (
            "failed close AND abort leave an actual live failed-open owner and its lease",
            after_open,
        )
        assert after_collect == (1, 0, 0, False, None), (
            "replacement/collection cannot dispose the backend pinned by failed Open",
            after_collect,
        )
        assert socket.closed
        assert "Abort already called" not in caplog.text
        assert "UsageError" not in caplog.text

    run(asyncio.wait_for(scenario(), timeout=6))


class CleanupFailingGrpcProvider(FakeOpenAIGrpcProvider):
    def __init__(self):
        super().__init__()
        self.cleanup_fails = True
        self.resource_live = False

    async def open_effect(self, request, publish):
        self.resource_live = True
        return await super().open_effect(request, publish)

    async def close_effect(self, request):
        self.closes.append(request)
        if self.cleanup_fails:
            raise RuntimeError("synthetic unresolved session cleanup")
        self.resource_live = False

    async def shutdown(self):
        self.shutdown_count += 1
        if self.cleanup_fails:
            raise RuntimeError("synthetic unresolved backend cleanup")
        self.resource_live = False


def test_grpc_established_cleanup_failure_keeps_lease_through_replacement(
    tmp_path: Path,
) -> None:
    async def scenario():
        config = secure_config(tmp_path)
        provider = CleanupFailingGrpcProvider()
        replacement = FakeOpenAIGrpcProvider()
        server = ProviderGrpcServer(config, openai_provider=provider)
        await server.start()
        entry = server.providers._entries[id(provider)]
        try:
            try:
                await collect_stream(
                    config,
                    open_request(
                        uuid4(),
                        provider_id=provider_pb2.PROVIDER_ID_OPENAI,
                        voice_engine=provider_pb2.VOICE_ENGINE_OPENAI,
                    ),
                )
            except grpc.aio.AioRpcError:
                pass
            after_close = (
                len(provider.opens),
                len(provider.closes),
                provider.resource_live,
                entry.leases,
            )
            server.providers.replace(ProviderId.OPENAI, replacement)
            collect_error = None
            try:
                await server.providers.collect()
            except RuntimeError as failure:
                collect_error = type(failure)
            after_collect = (
                entry.leases,
                provider.shutdown_count,
                replacement.shutdown_count,
                provider.resource_live,
                collect_error,
            )
        finally:
            provider.cleanup_fails = False
            await asyncio.wait_for(server.stop(), timeout=3)

        assert after_close[:2] == (1, 1), "established stream cleanup must be entered"
        assert after_close[2:] == (True, 1), after_close
        assert after_collect == (1, 0, 0, True, None), after_collect
        assert not provider.resource_live

    run(asyncio.wait_for(scenario(), timeout=6))


def test_grpc_local_close_waits_for_its_scheduler_job_but_not_healthy_peer(
    tmp_path: Path,
) -> None:
    async def scenario():
        config = secure_config(tmp_path)
        provider, *_ = local_provider_fixture()
        a, b = uuid4(), uuid4()
        entered = {identity: asyncio.Event() for identity in (a, b)}
        release = {identity: asyncio.Event() for identity in (a, b)}
        returned = set()
        close_entered = asyncio.Event()
        closed = []
        original_scheduler_close = provider._scheduler.close_session

        async def held_pipeline(session, _utterance, context):
            identity = session.request.session_id
            entered[identity].set()
            try:
                await release[identity].wait()
                context.ensure_current()
            finally:
                returned.add(identity)

        def observed_scheduler_close(identity):
            closed.append(identity)
            if identity == a:
                close_entered.set()
            return original_scheduler_close(identity)

        provider._run_pipeline = held_pipeline
        provider._scheduler.close_session = observed_scheduler_close
        server = ProviderGrpcServer(config, local_provider=provider)
        await server.start()
        entry = server.providers._entries[id(provider)]
        channel = grpc.aio.insecure_channel(f"unix://{config.socket_path}")
        stub = provider_pb2_grpc.ProviderTransportStub(channel)
        streams = {identity: InteractiveRequests() for identity in (a, b)}
        calls = {
            identity: stub.Stream(
                streams[identity],
                metadata=((AUTH_METADATA_KEY, f"Bearer {config.token}"),),
                timeout=5,
            )
            for identity in (a, b)
        }
        a_terminal = None
        try:
            for identity, direction in (
                (a, provider_pb2.AUDIO_DIRECTION_MICROPHONE),
                (b, provider_pb2.AUDIO_DIRECTION_SPEAKER),
            ):
                await streams[identity].send(open_request(identity, direction))
                assert (await calls[identity].read()).HasField("session_opened")
                assert (await calls[identity].read()).HasField("health")
                await streams[identity].send(frame_request(identity, direction))
                await asyncio.wait_for(entered[identity].wait(), timeout=1)
            before_close = (entry.leases, len(provider._futures), set(returned))
            await streams[a].send(close_request(a))
            a_terminal = asyncio.create_task(_read_remaining(calls[a]))
            await asyncio.wait_for(close_entered.wait(), timeout=1)
            # Observe completion without cancelling a correctly pending drain.
            await asyncio.wait({a_terminal}, timeout=0.05)
            before_a_release = (
                entry.leases,
                a_terminal.done(),
                set(returned),
                tuple(closed),
            )
            release[a].set()
            a_events = await asyncio.wait_for(asyncio.shield(a_terminal), timeout=1)
            a_closed_events = [
                event.session_closed.session_id
                for event in a_events
                if event.HasField("session_closed")
            ]
            after_a_release = (
                entry.leases,
                set(returned),
                release[b].is_set(),
                calls[b].done(),
                len(provider._futures),
            )
        finally:
            for event in release.values():
                event.set()
            for call in calls.values():
                call.cancel()
            await channel.close()
            if a_terminal is not None:
                await asyncio.gather(a_terminal, return_exceptions=True)
            await asyncio.wait_for(server.stop(), timeout=3)

        assert before_close == (2, 2, set()), "both real scheduler jobs entered"
        assert before_a_release == (2, False, set(), (a,)), before_a_release
        assert after_a_release == (1, {a}, False, False, 1), after_a_release
        assert len(a_events) == 1 and a_closed_events == [str(a)]
        assert returned == {a, b}

    run(asyncio.wait_for(scenario(), timeout=8))


def test_grpc_duplicate_uuid_never_closes_healthy_local_owner_and_eof_reuses_backend(
    tmp_path: Path,
):
    async def scenario():
        config = secure_config(tmp_path)
        provider, *_ = local_provider_fixture()
        closes, shutdowns = [], []
        original_close, original_shutdown = (
            provider._scheduler.close_session,
            provider.shutdown,
        )

        def close(identity):
            closes.append(identity)
            return original_close(identity)

        async def shutdown():
            shutdowns.append(True)
            await original_shutdown()

        provider._scheduler.close_session, provider.shutdown = close, shutdown
        server = ProviderGrpcServer(config, local_provider=provider)
        await server.start()
        entry = server.providers._entries[id(provider)]
        channel = grpc.aio.insecure_channel(f"unix://{config.socket_path}")
        stub = provider_pb2_grpc.ProviderTransportStub(channel)
        stream = InteractiveRequests()
        identity = uuid4()
        call = stub.Stream(
            stream, metadata=((AUTH_METADATA_KEY, f"Bearer {TOKEN}"),), timeout=5
        )
        try:
            await stream.send(open_request(identity))
            assert (await call.read()).HasField("session_opened")
            assert (await call.read()).HasField("health")
            duplicate_events = []
            duplicate = stub.Stream(
                requests(open_request(identity)),
                metadata=((AUTH_METADATA_KEY, f"Bearer {TOKEN}"),),
                timeout=2,
            )
            with pytest.raises(grpc.aio.AioRpcError) as rejected:
                async for event in duplicate:
                    duplicate_events.append(event)
            assert rejected.value.code() is grpc.StatusCode.INVALID_ARGUMENT
            assert duplicate_events == []
            assert entry.leases == 1 and closes == [] and shutdowns == []
            await stream.send(frame_request(identity))
            observed = []
            while not any(event.HasField("utterance_final") for event in observed):
                observed.append(await asyncio.wait_for(call.read(), timeout=1))
            assert all(
                getattr(event, event.WhichOneof("event")).session_id == str(identity)
                for event in observed
            )
            await stream.send(close_request(identity))
            closed = await _read_remaining(call)
            assert len(closed) == 1 and closed[0].HasField("session_closed")
            assert entry.leases == 0 and closes == [identity]
            for _ in range(100):
                events = await collect_stream(config, open_request(identity))
                assert [event.WhichOneof("event") for event in events] == [
                    "session_opened",
                    "health",
                ]
                assert entry.leases == 0 and not shutdowns
            assert len(closes) >= 101 and set(closes) == {identity}
        finally:
            call.cancel()
            await channel.close()
            await asyncio.wait_for(server.stop(), timeout=3)
        assert len(shutdowns) == 1

    run(asyncio.wait_for(scenario(), timeout=12))


def test_failed_transport_stop_retains_exact_server_and_fd_until_same_owner_retry(
    tmp_path: Path,
):
    class FailingTransport:
        def __init__(self):
            self.attempts = 0
            self.fail = True
            self.closed = False

        async def stop(self, grace):
            assert grace == 0.25
            self.attempts += 1
            if self.fail:
                raise OSError("synthetic transport stop failure")
            self.closed = True

    async def scenario():
        config = secure_config(tmp_path)
        provider = FakeOpenAIGrpcProvider()
        server = ProviderGrpcServer(config, openai_provider=provider)
        transport = FailingTransport()
        descriptor = os.open(config.socket_path.parent, os.O_RDONLY)
        server._server, server._parent_fd = transport, descriptor

        def descriptor_live():
            try:
                os.fstat(descriptor)
            except OSError:
                return False
            return True

        try:
            with pytest.raises(OSError):
                await server.stop()
            failed_snapshot = (
                server._server is transport,
                server._parent_fd == descriptor,
                descriptor_live(),
                transport.attempts,
                provider.shutdown_count,
                transport.closed,
            )
            transport.fail = False
            await server.stop()
            repaired_snapshot = (
                server._server,
                server._parent_fd,
                descriptor_live(),
                transport.attempts,
                provider.shutdown_count,
                transport.closed,
            )
            await server.stop()
            repeated_snapshot = (transport.attempts, provider.shutdown_count)
        finally:
            transport.fail = False
            if not transport.closed:
                await transport.stop(0.25)
            await asyncio.wait_for(server.stop(), timeout=3)
            if descriptor_live():
                os.close(descriptor)

        assert failed_snapshot == (True, True, True, 1, 0, False), failed_snapshot
        assert repaired_snapshot == (None, None, False, 2, 1, True), repaired_snapshot
        assert repeated_snapshot == (2, 1)

    run(asyncio.wait_for(scenario(), timeout=6))


def test_actual_stream_closure_wakes_blocked_publication_without_queued_dispatch(
    tmp_path: Path,
):
    class Context:
        async def abort(self, code, detail):
            raise AssertionError((code, detail))

    async def scenario():
        config = secure_config(tmp_path)
        provider, asr, _, _ = local_provider_fixture()
        owner = ProviderGrpcServer(config, local_provider=provider)
        servicer = _ProviderServicer(config, owner)
        source = InteractiveRequests()
        stream = servicer.Stream(source.__aiter__(), Context())
        identity = uuid4()
        commits, frames = [], []
        original_submit = provider.submit_frame
        blocked = closing = None

        async def submit(value):
            frames.append(value.session_id)
            await original_submit(value)

        provider.submit_frame = submit
        try:
            await source.send(open_request(identity))
            assert (await anext(stream)).HasField("session_opened")
            assert (await anext(stream)).HasField("health")
            publish = provider._sessions[identity].publish
            drained = inspect.getclosurevars(publish).nonlocals["event_drained"]

            def terminal(sequence):
                return ProviderUtteranceFinal(
                    session_id=identity,
                    stream_id=identity,
                    direction_id=AudioDirection.MICROPHONE,
                    utterance_id=uuid4(),
                    event_sequence=sequence,
                    outcome=UtteranceOutcome.CANCELLED,
                )

            await publish((terminal(3), terminal(4)), lambda: commits.append("first"))
            assert (await anext(stream)).HasField("utterance_final")
            assert not drained.is_set(), (
                "one queued event must keep the real wait closed"
            )
            blocked = asyncio.create_task(
                publish((terminal(5),), lambda: commits.append("late"))
            )
            await asyncio.sleep(0)
            assert len(drained._waiters) == 1 and not blocked.done()
            await source.send(frame_request(identity))
            await source.send(open_request(uuid4()))
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert owner.consumed_request_count == 3
            assert not frames and not asr.calls
            closing = asyncio.create_task(stream.aclose())
            await asyncio.wait({closing, blocked}, timeout=0.05)
            publication_error = None
            if blocked.done():
                publication_error = type(blocked.exception())
            snapshot = (
                closing.done(),
                blocked.done(),
                publication_error,
                tuple(commits),
                tuple(frames),
                len(asr.calls),
            )
        finally:
            if blocked is not None:
                blocked.cancel()
                await asyncio.gather(blocked, return_exceptions=True)
            if closing is not None:
                await asyncio.wait_for(asyncio.shield(closing), timeout=1)
            else:
                await stream.aclose()
            await asyncio.wait_for(owner.stop(), timeout=3)

        assert snapshot == (
            True,
            True,
            LocalProviderPublicationError,
            ("first",),
            (),
            0,
        ), snapshot

    run(asyncio.wait_for(scenario(), timeout=6))


@pytest.mark.parametrize(
    "disconnect", [False, True], ids=["orderly-close", "disconnected"]
)
def test_grpc_orderly_close_and_cancelled_retry_join_one_drain_and_terminal_event(
    tmp_path: Path,
    disconnect,
):
    class GatedClose(FakeOpenAIGrpcProvider):
        def __init__(self):
            super().__init__()
            self.entered, self.release = asyncio.Event(), asyncio.Event()
            self.commits = []

        async def close_effect(self, request):
            self.closes.append(request)
            self.entered.set()
            await self.release.wait()
            opened, publish = self.opens[0]
            try:
                await publish(
                    (
                        ProviderSessionClosed(
                            session_id=opened.session_id,
                            direction_id=opened.direction_id,
                            event_sequence=3,
                            reason=SessionCloseReason.USER_STOP,
                        ),
                    ),
                    lambda: self.commits.append(opened.session_id),
                )
            except LocalProviderPublicationError:
                return SafeErrorCode.PROVIDER_UNAVAILABLE

    async def scenario():
        config = secure_config(tmp_path)
        provider = GatedClose()
        local, *_ = local_provider_fixture()
        server = ProviderGrpcServer(
            config, openai_provider=provider, local_provider=local
        )
        await server.start()
        entry = server.providers._entries[id(provider)]
        source = InteractiveRequests()
        identity = uuid4()
        channel = grpc.aio.insecure_channel(f"unix://{config.socket_path}")
        call = provider_pb2_grpc.ProviderTransportStub(channel).Stream(
            source, metadata=((AUTH_METADATA_KEY, f"Bearer {TOKEN}"),), timeout=5
        )
        retries, terminal = [], None
        later = None

        async def read_terminal():
            try:
                return await _read_remaining(call)
            except asyncio.CancelledError:
                if disconnect:
                    return []
                raise

        try:
            await source.send(
                open_request(
                    identity,
                    provider_id=provider_pb2.PROVIDER_ID_OPENAI,
                    voice_engine=provider_pb2.VOICE_ENGINE_OPENAI,
                )
            )
            assert (await call.read()).HasField("session_opened")
            assert (await call.read()).HasField("health")
            if disconnect:
                call.cancel()
            else:
                await source.send(close_request(identity))
            terminal = asyncio.create_task(read_terminal())
            await asyncio.wait_for(provider.entered.wait(), timeout=1)
            retries.append(asyncio.create_task(server.retry_stream_cleanup()))
            retries.append(asyncio.create_task(server.retry_stream_cleanup()))
            await asyncio.sleep(0)
            retries[0].cancel()
            await asyncio.sleep(0)
            later_source = InteractiveRequests()
            later_id = uuid4()
            later = provider_pb2_grpc.ProviderTransportStub(channel).Stream(
                later_source,
                metadata=((AUTH_METADATA_KEY, f"Bearer {TOKEN}"),),
                timeout=3,
            )
            await later_source.send(open_request(later_id))
            assert (await later.read()).HasField("session_opened")
            assert (await later.read()).HasField("health")
            local_entry = server.providers._entries[id(local)]
            assert local_entry.leases == 1 and not later.done()
            retries[0].cancel()
            await asyncio.sleep(0)
            before_release = (
                entry.leases,
                len(provider.closes),
                terminal.done(),
                tuple(task.done() for task in retries),
            )
            provider.release.set()
            events = await asyncio.wait_for(asyncio.shield(terminal), timeout=1)
            outcomes = await asyncio.wait_for(
                asyncio.gather(*retries, return_exceptions=True), timeout=1
            )
            assert before_release == (1, 1, disconnect, (False, False))
            assert (
                isinstance(outcomes[0], asyncio.CancelledError) and outcomes[1] is None
            )
            if disconnect:
                assert events == [] and provider.commits == []
            else:
                assert len(events) == 1 and events[0].session_closed.session_id == str(
                    identity
                )
                assert provider.commits == [identity]
            assert entry.leases == 0 and len(provider.closes) == 1
            await server.retry_stream_cleanup()
            assert len(provider.closes) == 1 and entry.leases == 0
            assert local_entry.leases == 1 and not later.done()
            await later_source.send(close_request(later_id))
            later_events = await _read_remaining(later)
            assert len(later_events) == 1 and later_events[
                0
            ].session_closed.session_id == str(later_id)
        finally:
            provider.release.set()
            call.cancel()
            if later is not None:
                later.cancel()
            await channel.close()
            await asyncio.gather(*retries, return_exceptions=True)
            if terminal is not None:
                await asyncio.gather(terminal, return_exceptions=True)
            await asyncio.wait_for(server.stop(), timeout=3)

    run(asyncio.wait_for(scenario(), timeout=6))


@pytest.mark.parametrize(
    "cancel_open", [False, True], ids=["failed-open", "cancelled-open"]
)
def test_grpc_retained_retry_cycles_preserve_healthy_peer_and_original_disposal_owner(
    tmp_path: Path,
    cancel_open,
):
    class OwnedFailure(FailedOpenCleanupSocket):
        def __init__(self):
            super().__init__()
            self.entered, self.aborted = asyncio.Event(), asyncio.Event()
            self.release = asyncio.Event()

        async def recv(self):
            self.entered.set()
            if cancel_open:
                await self.release.wait()
            return await super().recv()

        def abort(self):
            try:
                super().abort()
            finally:
                self.aborted.set()

    def cloud_open(identity):
        value = open_request(
            identity,
            provider_id=provider_pb2.PROVIDER_ID_OPENAI,
            voice_engine=provider_pb2.VOICE_ENGINE_OPENAI,
        )
        value.open_session.requested_input_format.frame_duration_ms = 20
        value.open_session.requested_output_format.frame_duration_ms = 20
        return value

    async def terminal(call, events):
        try:
            while True:
                event = await call.read()
                if event is grpc.aio.EOF:
                    break
                events.append(event)
        except grpc.aio.AioRpcError as error:
            return error.code()
        except asyncio.CancelledError:
            return asyncio.CancelledError
        return None

    async def scenario():
        config = secure_config(tmp_path)
        healthy_socket = FakeRealtimeWebSocket()
        sockets = []

        async def connect(_uri, **_options):
            socket = healthy_socket if not sockets else OwnedFailure()
            sockets.append(socket)
            return socket

        provider = OpenAIRealtimeProvider(
            OpenAIRealtimeConfig(cloud_opt_in=True),
            environ={"OPENAI_API_KEY": "synthetic-secret"},
            websocket_factory=connect,
        )
        replacement = FakeOpenAIGrpcProvider()
        shutdowns = []
        fail_disposal = False
        original_shutdown = provider.shutdown

        async def shutdown():
            shutdowns.append(True)
            if fail_disposal:
                raise RuntimeError("synthetic disposal failure")
            await original_shutdown()

        provider.shutdown = shutdown
        server = ProviderGrpcServer(config, openai_provider=provider)
        await server.start()
        entry = server.providers._entries[id(provider)]
        channel = grpc.aio.insecure_channel(f"unix://{config.socket_path}")
        stub = provider_pb2_grpc.ProviderTransportStub(channel)
        metadata = ((AUTH_METADATA_KEY, f"Bearer {TOKEN}"),)
        healthy_source = InteractiveRequests()
        healthy_id = uuid4()
        healthy = stub.Stream(healthy_source, metadata=metadata, timeout=15)
        calls, collectors = [healthy], []
        try:
            await healthy_source.send(cloud_open(healthy_id))
            assert (await healthy.read()).HasField("session_opened")
            assert (await healthy.read()).HasField("health")
            healthy_records = frozenset(id(record) for record in server._streams)
            assert len(healthy_records) == 1 and entry.leases == 1
            for index in range(100):
                identity = uuid4()
                source = InteractiveRequests()
                call = stub.Stream(source, metadata=metadata, timeout=3)
                calls.append(call)
                events = []
                collecting = asyncio.create_task(terminal(call, events))
                collectors.append(collecting)
                await source.send(cloud_open(identity))
                async with asyncio.timeout(1):
                    while len(sockets) != index + 2:
                        await asyncio.sleep(0)
                assert len(sockets) == index + 2
                socket = sockets[-1]
                await asyncio.wait_for(socket.entered.wait(), timeout=1)
                opening_owner = provider._opening.get(identity)
                if cancel_open:
                    assert opening_owner is not None and not opening_owner.done()
                    call.cancel()
                    call.cancel()
                await asyncio.wait_for(socket.aborted.wait(), timeout=1)
                outcome = await asyncio.wait_for(asyncio.shield(collecting), timeout=1)
                assert events == []
                assert outcome is (
                    asyncio.CancelledError
                    if cancel_open
                    else grpc.StatusCode.INVALID_ARGUMENT
                )
                async with asyncio.timeout(1):
                    while identity in provider._opening or (
                        opening_owner is not None and not opening_owner.done()
                    ):
                        await asyncio.sleep(0)
                assert identity in provider._failed_opens
                before_failed_retry = frozenset(
                    id(record) for record in server._streams
                )
                with pytest.raises(OpenAIProviderProtocolError):
                    await server.retry_stream_cleanup()
                assert (
                    entry.leases == 2
                    and not socket.closed
                    and not healthy_socket.closed
                )
                assert (
                    frozenset(id(record) for record in server._streams)
                    == before_failed_retry
                )
                if index == 0:
                    retained_records = frozenset(
                        id(record) for record in server._streams
                    )
                    retained_provider = tuple(
                        frozenset((key, id(record)) for key, record in records.items())
                        for records in (
                            provider._sessions,
                            provider._failed_opens,
                            provider._opening,
                        )
                    )
                    retained_effects = (len(sockets), socket.closes, socket.aborts)
                    for _ in range(100):
                        refused_events = []
                        refused = stub.Stream(
                            requests(cloud_open(uuid4())), metadata=metadata, timeout=2
                        )
                        calls.append(refused)
                        refused_status = await terminal(refused, refused_events)
                        async with asyncio.timeout(1):
                            while entry.leases != 2 or len(server._streams) != 2:
                                await asyncio.sleep(0)
                        assert refused_status is grpc.StatusCode.INVALID_ARGUMENT
                        assert refused_events == []
                        assert (
                            frozenset(id(record) for record in server._streams)
                            == retained_records
                        )
                        assert (
                            tuple(
                                frozenset(
                                    (key, id(record)) for key, record in records.items()
                                )
                                for records in (
                                    provider._sessions,
                                    provider._failed_opens,
                                    provider._opening,
                                )
                            )
                            == retained_provider
                        )
                        assert (
                            len(sockets),
                            socket.closes,
                            socket.aborts,
                        ) == retained_effects
                if index == 99:
                    server.providers.replace(ProviderId.OPENAI, replacement)
                    await server.providers.collect()
                before_repair = (
                    entry.leases,
                    len(server._streams),
                    socket.closed,
                    tuple(shutdowns),
                    healthy_socket.closed,
                )
                socket.cleanup_fails = False
                socket.release.set()
                await server.retry_stream_cleanup()
                assert before_repair == (2, 2, False, (), False), before_repair
                assert entry.leases == 1 and socket.closed and not healthy_socket.closed
                assert (
                    frozenset(id(record) for record in server._streams)
                    == healthy_records
                )
                assert not provider._failed_opens and not provider._opening
                assert set(provider._sessions) == {healthy_id}
            b_events = await collect_stream(config, cloud_open(uuid4()))
            assert [event.WhichOneof("event") for event in b_events] == [
                "session_opened",
                "health",
            ]
            assert replacement.shutdown_count == 0 and len(replacement.closes) == 1
            frame = frame_request(healthy_id)
            frame.input_frame.frame_duration_ms = 20
            frame.input_frame.pcm = bytes(640)
            frame.input_frame.end_of_utterance = False
            await healthy_source.send(frame)
            async with asyncio.timeout(1):
                while (
                    healthy_socket.sent[-1]["type"]
                    != "session.input_audio_buffer.append"
                ):
                    await asyncio.sleep(0)
            assert (
                healthy_socket.sent[-1]["type"] == "session.input_audio_buffer.append"
            )
            fail_disposal = True
            await healthy_source.send(close_request(healthy_id))
            ending_events = []
            await terminal(healthy, ending_events)
            assert [event.WhichOneof("event") for event in ending_events] == [
                "utterance_final",
                "session_closed",
            ]
            final = ending_events[0].utterance_final
            closed = ending_events[1].session_closed
            assert [final.session_id, closed.session_id] == [str(healthy_id)] * 2
            assert [final.direction_id, closed.direction_id] == [
                frame.input_frame.direction_id
            ] * 2
            assert [final.event_sequence, closed.event_sequence] == [3, 4]
            assert final.stream_id == frame.input_frame.stream_id
            assert final.utterance_id == frame.input_frame.utterance_id
            assert final.outcome == provider_pb2.UTTERANCE_OUTCOME_CANCELLED
            assert not final.HasField("final_audio_sequence")
            assert closed.reason == provider_pb2.SESSION_CLOSE_REASON_USER_STOP
            assert entry.leases == 0 and len(shutdowns) == 1 and healthy_socket.closed
            assert len(server._streams) == 1
            before_retry = tuple(
                (
                    socket.closed,
                    getattr(socket, "closes", None),
                    getattr(socket, "aborts", None),
                )
                for socket in sockets
            )
            fail_disposal = False
            await server.retry_stream_cleanup()
            await server.retry_stream_cleanup()
            assert len(shutdowns) == 2 and entry.leases == 0
            assert not server._streams and id(provider) not in server.providers._entries
            assert (
                tuple(
                    (
                        socket.closed,
                        getattr(socket, "closes", None),
                        getattr(socket, "aborts", None),
                    )
                    for socket in sockets
                )
                == before_retry
            )
            assert replacement.shutdown_count == 0
        finally:
            fail_disposal = False
            for socket in sockets[1:]:
                socket.cleanup_fails = False
                socket.release.set()
            for call in calls:
                call.cancel()
            await channel.close()
            await asyncio.gather(*collectors, return_exceptions=True)
            await asyncio.wait_for(server.stop(), timeout=3)

    run(asyncio.wait_for(scenario(), timeout=20))


def test_terminal_stop_intent_rejects_new_open_and_queued_frame_before_transport_ack(
    tmp_path: Path,
):
    async def scenario():
        control_gate, stop_entered, stop_release = (asyncio.Event() for _ in range(3))
        control_gate.set()
        config = replace(secure_config(tmp_path), control_processing_gate=control_gate)
        provider = FakeOpenAIGrpcProvider()
        server = ProviderGrpcServer(config, openai_provider=provider)
        await server.start()
        transport, descriptor = server._server, server._parent_fd
        original_stop = transport.stop
        fail_stop = True

        async def stop(grace):
            stop_entered.set()
            await stop_release.wait()
            if fail_stop:
                raise OSError("synthetic live transport stop failure")
            await original_stop(grace)

        transport.stop = stop
        channel = grpc.aio.insecure_channel(f"unix://{config.socket_path}")
        stub = provider_pb2_grpc.ProviderTransportStub(channel)
        metadata = ((AUTH_METADATA_KEY, f"Bearer {TOKEN}"),)
        source = InteractiveRequests()
        identity = uuid4()
        healthy = stub.Stream(source, metadata=metadata, timeout=5)
        opening = stopping = None
        new_events = []
        try:
            await source.send(
                open_request(
                    identity,
                    provider_id=provider_pb2.PROVIDER_ID_OPENAI,
                    voice_engine=provider_pb2.VOICE_ENGINE_OPENAI,
                )
            )
            assert (await healthy.read()).HasField("session_opened")
            assert (await healthy.read()).HasField("health")
            control_gate.clear()
            await source.send(frame_request(identity))
            for _ in range(1000):
                if server.consumed_request_count == 2:
                    break
                await asyncio.sleep(0)
            assert server.consumed_request_count == 2 and provider.frames == []
            stopping = asyncio.create_task(server.stop())
            await asyncio.wait_for(stop_entered.wait(), timeout=1)
            opening = stub.Stream(
                requests(
                    open_request(
                        uuid4(),
                        provider_id=provider_pb2.PROVIDER_ID_OPENAI,
                        voice_engine=provider_pb2.VOICE_ENGINE_OPENAI,
                    )
                ),
                metadata=metadata,
                timeout=2,
            )
            control_gate.set()
            rejected = None
            try:
                while True:
                    event = await opening.read()
                    if event is grpc.aio.EOF:
                        break
                    new_events.append(event)
            except grpc.aio.AioRpcError as error:
                rejected = error.code()
            before_stop_failure = (
                len(provider.opens),
                len(provider.frames),
                provider.shutdown_count,
                tuple(event.WhichOneof("event") for event in new_events),
                rejected,
            )
            stop_release.set()
            with pytest.raises(OSError):
                await stopping
            retained = server._server is transport and server._parent_fd == descriptor
            fail_stop = False
            await server.stop()
        finally:
            fail_stop = False
            stop_release.set()
            control_gate.set()
            healthy.cancel()
            if opening is not None:
                opening.cancel()
            await channel.close()
            if stopping is not None:
                await asyncio.gather(stopping, return_exceptions=True)
            await asyncio.wait_for(original_stop(0), timeout=2)
            await asyncio.wait_for(server.stop(), timeout=3)
        assert before_stop_failure == (1, 0, 0, (), grpc.StatusCode.UNAVAILABLE), (
            before_stop_failure
        )
        assert retained

    run(asyncio.wait_for(scenario(), timeout=8))

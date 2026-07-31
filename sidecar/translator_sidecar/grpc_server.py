"""Authenticated grpc.aio transport for the local provider sidecar."""

from __future__ import annotations

import asyncio
import hmac
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generic, Protocol, TypeVar
from uuid import UUID

import grpc
from pydantic import ValidationError

from .generated.translator.provider.v1 import provider_pb2, provider_pb2_grpc
from .local.local_provider import (
    LocalProvider,
    LocalProviderPublicationError,
    LocalProviderProtocolError,
)
from .provider_contract import (
    AudioDirection,
    CancelReason,
    CancelUtterance,
    CloseProviderSession,
    CloseRequestReason,
    ComputeDevice,
    Language,
    OpenProviderSession,
    PcmFormat,
    PrivacySafeProviderError,
    ProviderAudioDelta,
    ProviderHealth,
    ProviderId,
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
from .provider_engine import ProviderEngine, ProviderProtocolError

AUTH_METADATA_KEY = "authorization"
CHANNEL_CAPACITY = 64

_OPEN_VERSION = "translator.provider.open_session.v1"
_INPUT_VERSION = "translator.provider.input.v1"
_CANCEL_VERSION = "translator.provider.cancel_utterance.v1"
_CLOSE_VERSION = "translator.provider.close_session.v1"
_DEBUG_VERSION = "translator.provider.update_debug_text.v1"
_PROBE_REQUEST_VERSION = "translator.provider.probe_request.v1"
_PROBE_RESPONSE_VERSION = "translator.provider.probe_response.v1"

T = TypeVar("T")


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


class ChannelOverflow(RuntimeError):
    pass


class BoundedChannel(Generic[T]):
    def __init__(self, capacity: int = CHANNEL_CAPACITY) -> None:
        if not 1 <= capacity <= CHANNEL_CAPACITY:
            raise ValueError("channel capacity must be between 1 and 64")
        self.capacity = capacity
        self._queue: asyncio.Queue[T] = asyncio.Queue(maxsize=capacity)

    def put_nowait(self, item: T) -> None:
        try:
            self._queue.put_nowait(item)
        except asyncio.QueueFull as error:
            raise ChannelOverflow("resource_exhausted") from error

    async def put(self, item: T) -> None:
        await self._queue.put(item)

    def put_terminal(self, error: T, terminal: T) -> None:
        if self.capacity - self.qsize() < 2:
            raise ChannelOverflow("resource_exhausted")
        self._queue.put_nowait(error)
        self._queue.put_nowait(terminal)

    def put_many_nowait(self, items: tuple[T, ...]) -> None:
        if self.capacity - self.qsize() < len(items):
            raise ChannelOverflow("resource_exhausted")
        for item in items:
            self._queue.put_nowait(item)

    async def get(self) -> T:
        return await self._queue.get()

    def get_nowait(self) -> T:
        return self._queue.get_nowait()

    def qsize(self) -> int:
        return self._queue.qsize()

    def empty(self) -> bool:
        return self._queue.empty()


@dataclass(frozen=True, slots=True)
class SidecarServerConfig:
    socket_path: Path
    token: str
    generation_id: UUID
    now_ns: Callable[[], int]
    channel_capacity: int = CHANNEL_CAPACITY
    control_processing_gate: asyncio.Event | None = None

    def __post_init__(self) -> None:
        if re.fullmatch(r"[0-9a-f]{64}", self.token) is None:
            raise ValueError("token must be exactly 64 lowercase hex characters")
        if not 1 <= self.channel_capacity <= CHANNEL_CAPACITY:
            raise ValueError("channel capacity must be between 1 and 64")


class _AuthInterceptor(grpc.aio.ServerInterceptor):
    def __init__(self, token: str) -> None:
        self._expected = f"Bearer {token}"

    async def intercept_service(self, continuation, handler_call_details):
        handler = await continuation(handler_call_details)
        if handler is None or self._authorized(handler_call_details):
            return handler

        if handler.request_streaming and handler.response_streaming:

            async def abort_stream(request_iterator, context):
                await context.abort(
                    grpc.StatusCode.UNAUTHENTICATED,
                    "authentication_failed",
                )
                yield provider_pb2.ProviderEvent()

            return grpc.stream_stream_rpc_method_handler(
                abort_stream,
                request_deserializer=handler.request_deserializer,
                response_serializer=handler.response_serializer,
            )

        async def abort_unary(request, context):
            await context.abort(
                grpc.StatusCode.UNAUTHENTICATED,
                "authentication_failed",
            )

        return grpc.unary_unary_rpc_method_handler(
            abort_unary,
            request_deserializer=handler.request_deserializer,
            response_serializer=handler.response_serializer,
        )

    def _authorized(self, handler_call_details) -> bool:
        values = [
            value
            for key, value in handler_call_details.invocation_metadata
            if key.lower() == AUTH_METADATA_KEY
        ]
        return len(values) == 1 and hmac.compare_digest(values[0], self._expected)


class _InvalidRequest(ValueError):
    pass


@dataclass(slots=True)
class _StreamState:
    session_id: UUID | None = None
    closed: bool = False
    runtime_provider: RuntimeProvider | None = None
    runtime_provider_id: ProviderId | None = None
    provider_acquired: bool = False


_CONTROL_END = object()


class _ProviderServicer(provider_pb2_grpc.ProviderTransportServicer):
    def __init__(
        self,
        config: SidecarServerConfig,
        engine: ProviderEngine,
        local_provider: LocalProvider | None,
        openai_provider: RuntimeProvider | None,
        owner: ProviderGrpcServer,
    ) -> None:
        self._config = config
        self._engine = engine
        self._uses_runtime_provider = (
            local_provider is not None or openai_provider is not None
        )
        self._owner = owner

    async def Probe(self, request, context):
        if request.schema_version != _PROBE_REQUEST_VERSION:
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "invalid_request",
            )
        return provider_pb2.ProviderProbeResponse(
            schema_version=_PROBE_RESPONSE_VERSION,
            generation_id=str(self._config.generation_id),
            provider_ready=self._owner.provider_ready,
        )

    async def Stream(self, request_iterator, context):
        capacity = self._config.channel_capacity
        control: BoundedChannel[object] = BoundedChannel(capacity)
        events: BoundedChannel[provider_pb2.ProviderEvent] = BoundedChannel(capacity)
        self._owner.created_channel_capacities.extend(
            (("control", capacity), ("event", capacity))
        )
        state = _StreamState()
        failure: list[tuple[grpc.StatusCode, str]] = []
        signal = asyncio.Event()
        event_drained = asyncio.Event()
        event_drained.set()
        transport_closed = asyncio.Event()
        consumer_done = False

        def fail(code: grpc.StatusCode, detail: str) -> None:
            if not failure:
                failure.append((code, detail))
            signal.set()

        async def produce_requests() -> None:
            try:
                async for request in request_iterator:
                    self._owner.consumed_request_count += 1
                    await control.put(request)
                await control.put(_CONTROL_END)
            except asyncio.CancelledError:
                raise
            except Exception:
                fail(grpc.StatusCode.INVALID_ARGUMENT, "invalid_request")

        async def enqueue_event(event: provider_pb2.ProviderEvent) -> None:
            await event_drained.wait()
            try:
                events.put_nowait(event)
                event_drained.clear()
                signal.set()
            except ChannelOverflow:
                fail(
                    grpc.StatusCode.RESOURCE_EXHAUSTED,
                    "resource_exhausted",
                )

        async def enqueue_terminal_pair(
            error: provider_pb2.ProviderEvent,
            terminal: provider_pb2.ProviderEvent,
        ) -> None:
            await event_drained.wait()
            try:
                events.put_terminal(error, terminal)
                event_drained.clear()
                signal.set()
            except ChannelOverflow:
                fail(
                    grpc.StatusCode.RESOURCE_EXHAUSTED,
                    "resource_exhausted",
                )

        async def publish_local(
            batch,
            commit: Callable[[], None],
        ) -> None:
            if transport_closed.is_set():
                raise LocalProviderPublicationError("provider transport is closed")
            encoded = tuple(_event_to_proto(event) for event in batch)
            await event_drained.wait()
            if transport_closed.is_set():
                raise LocalProviderPublicationError("provider transport is closed")
            try:
                events.put_many_nowait(encoded)
            except ChannelOverflow:
                fail(
                    grpc.StatusCode.RESOURCE_EXHAUSTED,
                    "resource_exhausted",
                )
                raise LocalProviderPublicationError(
                    "provider transport capacity reached"
                ) from None
            event_drained.clear()
            commit()
            signal.set()

        async def consume_requests() -> None:
            nonlocal consumer_done
            try:
                while not failure and not state.closed:
                    item = await control.get()
                    if item is _CONTROL_END:
                        break
                    await event_drained.wait()
                    gate = self._config.control_processing_gate
                    if gate is not None:
                        await gate.wait()
                    try:
                        if not self._uses_runtime_provider:
                            produced = self._handle_request(item, state)
                        else:
                            produced = await self._handle_runtime_request(
                                item,
                                state,
                                publish_local,
                            )
                    except _InvalidRequest:
                        fail(
                            grpc.StatusCode.INVALID_ARGUMENT,
                            "invalid_request",
                        )
                        break
                    except (
                        LocalProviderProtocolError,
                        ProviderProtocolError,
                        ValidationError,
                        ValueError,
                    ):
                        fail(
                            grpc.StatusCode.INVALID_ARGUMENT,
                            "protocol_error",
                        )
                        break
                    except Exception:
                        fail(
                            grpc.StatusCode.INTERNAL,
                            "internal_error",
                        )
                        break

                    if (
                        not self._uses_runtime_provider
                        and len(produced) == 2
                        and produced[0].HasField("error")
                        and produced[1].HasField("utterance_final")
                    ):
                        await enqueue_terminal_pair(produced[0], produced[1])
                    else:
                        for event in produced:
                            await enqueue_event(event)
                            if failure:
                                break
                    if (
                        not self._uses_runtime_provider
                        and not failure
                        and item.WhichOneof("request") == "input_frame"
                        and state.session_id is not None
                    ):
                        while True:
                            wakeup_ms = self._engine.next_wakeup_ms(
                                state.session_id,
                                now_ns=self._config.now_ns(),
                            )
                            if wakeup_ms is None:
                                break
                            await asyncio.sleep(max(1, wakeup_ms) / 1000)
                            pending = self._process_pending(state.session_id)
                            for event in pending:
                                await enqueue_event(event)
                                if failure:
                                    break
                            if failure:
                                break
            finally:
                consumer_done = True
                signal.set()

        producer_task = asyncio.create_task(produce_requests())
        consumer_task = asyncio.create_task(consume_requests())
        try:
            while True:
                if failure:
                    code, detail = failure[0]
                    await context.abort(code, detail)
                if not events.empty():
                    event = events.get_nowait()
                    if events.empty():
                        event_drained.set()
                    yield event
                    continue
                if consumer_done:
                    break
                signal.clear()
                if failure or not events.empty() or consumer_done:
                    signal.set()
                    continue
                await signal.wait()
        finally:
            transport_closed.set()
            producer_task.cancel()
            consumer_task.cancel()
            await asyncio.gather(
                producer_task,
                consumer_task,
                return_exceptions=True,
            )
            try:
                await self._cleanup_session(state)
            finally:
                if state.provider_acquired:
                    state.provider_acquired = False
                    if state.runtime_provider_id is ProviderId.LOCAL:
                        await self._owner.release_local_provider(
                            state.runtime_provider
                        )

    def _handle_request(
        self,
        request: provider_pb2.ProviderRequest,
        state: _StreamState,
    ) -> tuple[provider_pb2.ProviderEvent, ...]:
        kind = request.WhichOneof("request")
        if kind is None:
            raise _InvalidRequest("missing request")
        if state.session_id is None and kind != "open_session":
            raise ProviderProtocolError("open_session_required")
        if state.session_id is not None and kind == "open_session":
            raise ProviderProtocolError("duplicate_open_session")

        if kind == "open_session":
            model = _open_from_proto(request.open_session)
            opened = self._engine.open_session(model)
            state.session_id = model.session_id
            health = self._engine.health(
                model.session_id,
                now_ns=self._config.now_ns(),
            )
            return (_event_to_proto(opened), _event_to_proto(health))

        if kind == "input_frame":
            frame = _frame_from_proto(request.input_frame)
            if frame.session_id != state.session_id:
                raise ProviderProtocolError("session_identity_mismatch")
            admission = self._engine.enqueue_frame(
                frame,
                now_ns=self._config.now_ns(),
            )
            if isinstance(admission, tuple):
                return tuple(_event_to_proto(event) for event in admission)
            produced = []
            if admission is not None:
                produced.append(_event_to_proto(admission))
            else:
                produced.extend(self._process_pending(frame.session_id))
            return tuple(produced)

        if kind == "cancel_utterance":
            model = _cancel_from_proto(request.cancel_utterance)
            if model.session_id != state.session_id:
                raise ProviderProtocolError("session_identity_mismatch")
            return (_event_to_proto(self._engine.cancel_utterance(model)),)

        if kind == "update_debug_text":
            model = _debug_from_proto(request.update_debug_text)
            if model.session_id != state.session_id:
                raise ProviderProtocolError("session_identity_mismatch")
            self._engine.update_debug_text(model)
            return ()

        if kind == "close_session":
            model = _close_from_proto(request.close_session)
            if model.session_id != state.session_id:
                raise ProviderProtocolError("session_identity_mismatch")
            closed = self._engine.close_session(model)
            state.closed = True
            return (_event_to_proto(closed),)

        raise _InvalidRequest("unknown request")

    def _process_pending(
        self, session_id: UUID
    ) -> tuple[provider_pb2.ProviderEvent, ...]:
        produced = [
            _event_to_proto(event)
            for event in self._engine.process_next(
                session_id,
                now_ns=self._config.now_ns(),
            )
        ]
        produced.extend(
            _event_to_proto(event)
            for event in self._engine.drain_output(
                session_id,
                now_ns=self._config.now_ns(),
            )
        )
        return tuple(produced)

    async def _handle_runtime_request(
        self,
        request: provider_pb2.ProviderRequest,
        state: _StreamState,
        publish,
    ) -> tuple[provider_pb2.ProviderEvent, ...]:
        kind = request.WhichOneof("request")
        if kind is None:
            raise _InvalidRequest("missing request")
        if state.session_id is None and kind != "open_session":
            raise LocalProviderProtocolError("open_session_required")
        if state.session_id is not None and kind == "open_session":
            raise LocalProviderProtocolError("duplicate_open_session")

        if kind == "open_session":
            model = _open_from_proto(request.open_session)
            provider = await self._owner.acquire_runtime_provider(model.provider_id)
            try:
                opened, health = await provider.open_session(
                    model,
                    publish,
                )
            except BaseException:
                if model.provider_id is ProviderId.LOCAL:
                    await self._owner.release_local_provider(provider)
                raise
            state.runtime_provider = provider
            state.runtime_provider_id = model.provider_id
            state.provider_acquired = model.provider_id is ProviderId.LOCAL
            state.session_id = model.session_id
            return (_event_to_proto(opened), _event_to_proto(health))

        provider = state.runtime_provider
        if provider is None:
            raise LocalProviderProtocolError("open_session_required")

        if kind == "input_frame":
            frame = _frame_from_proto(request.input_frame)
            if frame.session_id != state.session_id:
                raise LocalProviderProtocolError("session_identity_mismatch")
            await provider.submit_frame(frame)
            return ()

        if kind == "cancel_utterance":
            model = _cancel_from_proto(request.cancel_utterance)
            if model.session_id != state.session_id:
                raise LocalProviderProtocolError("session_identity_mismatch")
            await provider.cancel_utterance(model)
            return ()

        if kind == "update_debug_text":
            model = _debug_from_proto(request.update_debug_text)
            if model.session_id != state.session_id:
                raise LocalProviderProtocolError("session_identity_mismatch")
            await provider.update_debug_text(model)
            return ()

        if kind == "close_session":
            model = _close_from_proto(request.close_session)
            if model.session_id != state.session_id:
                raise LocalProviderProtocolError("session_identity_mismatch")
            await provider.close_session(model)
            await provider.wait_publications(model.session_id)
            state.closed = True
            return ()

        raise _InvalidRequest("unknown request")

    async def _cleanup_session(self, state: _StreamState) -> None:
        if state.session_id is None:
            return
        if self._uses_runtime_provider:
            provider = state.runtime_provider
            if provider is None:
                return
            if state.closed:
                return
            try:
                await provider.close_session(
                    CloseProviderSession(
                        session_id=state.session_id,
                        reason=CloseRequestReason.DAEMON_SHUTDOWN,
                    )
                )
                await provider.wait_publications(state.session_id)
            except Exception:
                pass
            return
        try:
            if state.closed:
                return
            try:
                self._engine.close_session(
                    CloseProviderSession(
                        session_id=state.session_id,
                        reason=CloseRequestReason.DAEMON_SHUTDOWN,
                    )
                )
            except Exception:
                pass
        finally:
            try:
                self._engine.release_session(state.session_id)
            except ProviderProtocolError:
                pass


class ProviderGrpcServer:
    def __init__(
        self,
        config: SidecarServerConfig,
        *,
        engine: ProviderEngine | None = None,
        local_provider: LocalProvider | None = None,
        openai_provider: RuntimeProvider | None = None,
        provider_ready: bool = True,
    ) -> None:
        if engine is not None and (
            local_provider is not None or openai_provider is not None
        ):
            raise ValueError("provider backends are mutually exclusive")
        self.config = config
        self.engine = engine or ProviderEngine()
        self.local_provider = local_provider
        self.openai_provider = openai_provider
        self.provider_ready = provider_ready
        self.consumed_request_count = 0
        self.created_channel_capacities: list[tuple[str, int]] = []
        self._server: grpc.aio.Server | None = None
        self._parent_fd: int | None = None
        self._lifecycle_lock = asyncio.Lock()
        self._provider_lock = asyncio.Lock()
        self._provider_leases: dict[int, int] = {}
        self._provider_instances: dict[int, LocalProvider] = {}
        self._retired_providers: set[int] = set()
        if local_provider is not None:
            identity = id(local_provider)
            self._provider_leases[identity] = 0
            self._provider_instances[identity] = local_provider

    async def acquire_runtime_provider(
        self,
        provider_id: ProviderId,
    ) -> RuntimeProvider:
        if provider_id is ProviderId.LOCAL:
            return await self.acquire_local_provider()
        if provider_id is ProviderId.OPENAI and self.openai_provider is not None:
            return self.openai_provider
        raise LocalProviderProtocolError("provider is unavailable")

    async def acquire_local_provider(self) -> LocalProvider:
        async with self._provider_lock:
            provider = self.local_provider
            if provider is None:
                raise RuntimeError("local provider is unavailable")
            identity = id(provider)
            self._provider_instances.setdefault(identity, provider)
            self._provider_leases[identity] = self._provider_leases.get(identity, 0) + 1
            return provider

    async def release_local_provider(
        self,
        provider: LocalProvider | None,
    ) -> None:
        if provider is None:
            return
        shutdown: LocalProvider | None = None
        async with self._provider_lock:
            identity = id(provider)
            leases = self._provider_leases.get(identity)
            if leases is None or leases <= 0:
                return
            leases -= 1
            self._provider_leases[identity] = leases
            if leases == 0 and identity in self._retired_providers:
                self._retired_providers.remove(identity)
                self._provider_leases.pop(identity, None)
                shutdown = self._provider_instances.pop(
                    identity,
                    None,
                )
        if shutdown is not None:
            await shutdown.shutdown()

    async def replace_local_provider(
        self,
        provider: LocalProvider,
    ) -> None:
        shutdown: LocalProvider | None = None
        async with self._provider_lock:
            previous = self.local_provider
            if previous is provider:
                return
            identity = id(provider)
            self._provider_instances[identity] = provider
            self._provider_leases.setdefault(identity, 0)
            self.local_provider = provider
            self.provider_ready = True
            if previous is not None:
                previous_identity = id(previous)
                self._retired_providers.add(previous_identity)
                if self._provider_leases.get(previous_identity, 0) == 0:
                    self._retired_providers.remove(previous_identity)
                    self._provider_leases.pop(previous_identity, None)
                    shutdown = self._provider_instances.pop(
                        previous_identity,
                        None,
                    )
        if shutdown is not None:
            await shutdown.shutdown()

    async def _shutdown_local_providers(self) -> None:
        async with self._provider_lock:
            providers = tuple(self._provider_instances.values())
            self._provider_instances.clear()
            self._provider_leases.clear()
            self._retired_providers.clear()
            self.local_provider = None
        first_error: Exception | None = None
        for provider in providers:
            try:
                await provider.shutdown()
            except Exception as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error

    async def _shutdown_openai_provider(self) -> None:
        provider = self.openai_provider
        self.openai_provider = None
        if provider is not None:
            await provider.shutdown()

    async def start(self) -> None:
        async with self._lifecycle_lock:
            if self._server is not None:
                raise RuntimeError("server_already_started")
            parent_fd = self._open_verified_parent()
            server = grpc.aio.server(
                interceptors=(_AuthInterceptor(self.config.token),)
            )
            provider_pb2_grpc.add_ProviderTransportServicer_to_server(
                _ProviderServicer(
                    self.config,
                    self.engine,
                    self.local_provider,
                    self.openai_provider,
                    self,
                ),
                server,
            )
            bound_path = f"/proc/self/fd/{parent_fd}/{self.config.socket_path.name}"
            bound = server.add_insecure_port(f"unix:{bound_path}")
            if bound == 0:
                os.close(parent_fd)
                raise RuntimeError("uds_bind_failed")
            try:
                await server.start()
                os.chmod(
                    self.config.socket_path.name,
                    0o600,
                    dir_fd=parent_fd,
                    follow_symlinks=False,
                )
                socket_stat = os.stat(
                    self.config.socket_path.name,
                    dir_fd=parent_fd,
                    follow_symlinks=False,
                )
                if not stat.S_ISSOCK(socket_stat.st_mode):
                    raise RuntimeError("bound_path_is_not_socket")
                if socket_stat.st_uid != os.getuid():
                    raise PermissionError("socket owner mismatch")
                self._server = server
                self._parent_fd = parent_fd
            except BaseException:
                await asyncio.shield(server.stop(grace=0))
                os.close(parent_fd)
                raise

    async def stop(self) -> None:
        async with self._lifecycle_lock:
            if self._server is None:
                await self._shutdown_local_providers()
                await self._shutdown_openai_provider()
                return
            server = self._server
            stop_task = asyncio.ensure_future(server.stop(grace=0))
            cancelled: BaseException | None = None
            try:
                while not stop_task.done():
                    try:
                        await asyncio.shield(stop_task)
                    except BaseException as error:
                        if not isinstance(error, asyncio.CancelledError):
                            raise
                        cancelled = error
                        current = asyncio.current_task()
                        if current is not None:
                            current.uncancel()
                stop_task.result()
            finally:
                self._server = None
                if self._parent_fd is not None:
                    os.close(self._parent_fd)
                    self._parent_fd = None
                await self._shutdown_local_providers()
                await self._shutdown_openai_provider()
            if cancelled is not None:
                raise cancelled

    def _open_verified_parent(self) -> int:
        parent = self.config.socket_path.parent
        try:
            parent_fd = os.open(
                parent,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
            )
        except OSError as error:
            try:
                parent_stat = os.lstat(parent)
            except NotADirectoryError:
                raise
            if stat.S_ISLNK(parent_stat.st_mode):
                raise PermissionError(
                    "socket parent must be a real directory"
                ) from error
            if not stat.S_ISDIR(parent_stat.st_mode):
                raise NotADirectoryError(parent) from error
            raise
        parent_stat = os.fstat(parent_fd)
        try:
            if not stat.S_ISDIR(parent_stat.st_mode):
                raise NotADirectoryError(parent)
            if parent_stat.st_uid != os.getuid():
                raise PermissionError("socket parent owner mismatch")
            if stat.S_IMODE(parent_stat.st_mode) != 0o700:
                raise PermissionError("socket parent mode must be 0700")
            try:
                os.stat(
                    self.config.socket_path.name,
                    dir_fd=parent_fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                return parent_fd
            raise FileExistsError(self.config.socket_path)
        except NotADirectoryError:
            os.close(parent_fd)
            raise
        except BaseException:
            os.close(parent_fd)
            raise


_DIRECTION_FROM_PROTO = {
    provider_pb2.AUDIO_DIRECTION_MICROPHONE: AudioDirection.MICROPHONE,
    provider_pb2.AUDIO_DIRECTION_SPEAKER: AudioDirection.SPEAKER,
}
_DIRECTION_TO_PROTO = {value: key for key, value in _DIRECTION_FROM_PROTO.items()}
_MODE_FROM_PROTO = {
    provider_pb2.TRANSLATION_MODE_QUALITY_FIRST: TranslationMode.QUALITY_FIRST,
    provider_pb2.TRANSLATION_MODE_BALANCED: TranslationMode.BALANCED,
    provider_pb2.TRANSLATION_MODE_STREAMING_FIRST: TranslationMode.STREAMING_FIRST,
}
_MODE_TO_PROTO = {value: key for key, value in _MODE_FROM_PROTO.items()}
_LANGUAGE_FROM_PROTO = {
    provider_pb2.LANGUAGE_RU: Language.RU,
    provider_pb2.LANGUAGE_EN: Language.EN,
}
_LANGUAGE_TO_PROTO = {value: key for key, value in _LANGUAGE_FROM_PROTO.items()}
_SAMPLE_FROM_PROTO = {
    provider_pb2.SAMPLE_FORMAT_S16LE: SampleFormat.S16LE,
}
_SAMPLE_TO_PROTO = {value: key for key, value in _SAMPLE_FROM_PROTO.items()}
_VOICE_GENDER_FROM_PROTO = {
    provider_pb2.VOICE_GENDER_MALE: VoiceGender.MALE,
    provider_pb2.VOICE_GENDER_FEMALE: VoiceGender.FEMALE,
}
_VOICE_ENGINE_FROM_PROTO = {
    provider_pb2.VOICE_ENGINE_PIPER: VoiceEngine.PIPER,
    provider_pb2.VOICE_ENGINE_SILERO: VoiceEngine.SILERO,
    provider_pb2.VOICE_ENGINE_OPENAI: VoiceEngine.OPENAI,
}
_PROVIDER_FROM_PROTO = {
    provider_pb2.PROVIDER_ID_LOCAL: ProviderId.LOCAL,
    provider_pb2.PROVIDER_ID_OPENAI: ProviderId.OPENAI,
}
_CLOSE_FROM_PROTO = {
    provider_pb2.CLOSE_REQUEST_REASON_USER_STOP: CloseRequestReason.USER_STOP,
    provider_pb2.CLOSE_REQUEST_REASON_ROUTE_REMOVED: CloseRequestReason.ROUTE_REMOVED,
    provider_pb2.CLOSE_REQUEST_REASON_DEVICE_UNAVAILABLE: CloseRequestReason.DEVICE_UNAVAILABLE,
    provider_pb2.CLOSE_REQUEST_REASON_PROVIDER_SWITCH: CloseRequestReason.PROVIDER_SWITCH,
    provider_pb2.CLOSE_REQUEST_REASON_DAEMON_SHUTDOWN: CloseRequestReason.DAEMON_SHUTDOWN,
}
_CANCEL_FROM_PROTO = {
    provider_pb2.CANCEL_REASON_LATENCY_POLICY: CancelReason.LATENCY_POLICY,
    provider_pb2.CANCEL_REASON_ROUTE_REMOVED: CancelReason.ROUTE_REMOVED,
    provider_pb2.CANCEL_REASON_USER_INTERRUPT: CancelReason.USER_INTERRUPT,
    provider_pb2.CANCEL_REASON_QUEUE_OVERFLOW: CancelReason.QUEUE_OVERFLOW,
}


def _enum(mapping: dict[int, T], value: int) -> T:
    try:
        return mapping[value]
    except KeyError as error:
        raise _InvalidRequest("unsupported enum") from error


def _uuid(value: str) -> UUID:
    try:
        return UUID(value)
    except (ValueError, AttributeError) as error:
        raise _InvalidRequest("invalid identifier") from error


def _pcm_from_proto(value) -> PcmFormat:
    return PcmFormat(
        sample_rate_hz=value.sample_rate_hz,
        channels=value.channels,
        sample_format=_enum(_SAMPLE_FROM_PROTO, value.sample_format),
        frame_duration_ms=value.frame_duration_ms,
    )


def _open_from_proto(value) -> OpenProviderSession:
    if value.schema_version != _OPEN_VERSION:
        raise _InvalidRequest("invalid open version")
    return OpenProviderSession(
        session_id=_uuid(value.session_id),
        provider_id=_enum(_PROVIDER_FROM_PROTO, value.provider_id),
        direction_id=_enum(_DIRECTION_FROM_PROTO, value.direction_id),
        source_language=_enum(_LANGUAGE_FROM_PROTO, value.source_language),
        target_language=_enum(_LANGUAGE_FROM_PROTO, value.target_language),
        mode=_enum(_MODE_FROM_PROTO, value.mode),
        requested_input_format=_pcm_from_proto(value.requested_input_format),
        requested_output_format=_pcm_from_proto(value.requested_output_format),
        voice_profile=VoiceProfile(
            language=_enum(
                _LANGUAGE_FROM_PROTO,
                value.voice_profile.language,
            ),
            gender=_enum(
                _VOICE_GENDER_FROM_PROTO,
                value.voice_profile.gender,
            ),
            engine=_enum(
                _VOICE_ENGINE_FROM_PROTO,
                value.voice_profile.engine,
            ),
            model_path=(
                value.voice_profile.model_path
                if value.voice_profile.HasField("model_path")
                else None
            ),
            provider_voice_id=(
                value.voice_profile.provider_voice_id
                if value.voice_profile.HasField("provider_voice_id")
                else None
            ),
        ),
        debug_text_enabled=value.debug_text_enabled,
    )


def _frame_from_proto(value) -> ProviderInputFrame:
    if value.schema_version != _INPUT_VERSION:
        raise _InvalidRequest("invalid input version")
    return ProviderInputFrame(
        session_id=_uuid(value.session_id),
        direction_id=_enum(_DIRECTION_FROM_PROTO, value.direction_id),
        stream_id=_uuid(value.stream_id),
        utterance_id=_uuid(value.utterance_id),
        sequence=value.sequence,
        capture_monotonic_ns=value.capture_monotonic_ns,
        sample_rate_hz=value.sample_rate_hz,
        channels=value.channels,
        sample_format=_enum(_SAMPLE_FROM_PROTO, value.sample_format),
        frame_duration_ms=value.frame_duration_ms,
        source_language=_enum(
            _LANGUAGE_FROM_PROTO,
            value.source_language,
        ),
        target_language=_enum(
            _LANGUAGE_FROM_PROTO,
            value.target_language,
        ),
        mode=_enum(_MODE_FROM_PROTO, value.mode),
        pcm=value.pcm,
        end_of_utterance=value.end_of_utterance,
    )


def _cancel_from_proto(value) -> CancelUtterance:
    if value.schema_version != _CANCEL_VERSION:
        raise _InvalidRequest("invalid cancel version")
    return CancelUtterance(
        session_id=_uuid(value.session_id),
        direction_id=_enum(_DIRECTION_FROM_PROTO, value.direction_id),
        utterance_id=_uuid(value.utterance_id),
        reason=_enum(_CANCEL_FROM_PROTO, value.reason),
    )


def _close_from_proto(value) -> CloseProviderSession:
    if value.schema_version != _CLOSE_VERSION:
        raise _InvalidRequest("invalid close version")
    return CloseProviderSession(
        session_id=_uuid(value.session_id),
        reason=_enum(_CLOSE_FROM_PROTO, value.reason),
    )


def _debug_from_proto(value) -> UpdateDebugText:
    if value.schema_version != _DEBUG_VERSION:
        raise _InvalidRequest("invalid debug version")
    return UpdateDebugText(
        session_id=_uuid(value.session_id),
        enabled=value.enabled,
    )


def _pcm_to_proto(value: PcmFormat) -> provider_pb2.PcmFormat:
    return provider_pb2.PcmFormat(
        sample_rate_hz=value.sample_rate_hz,
        channels=value.channels,
        sample_format=_SAMPLE_TO_PROTO[value.sample_format],
        frame_duration_ms=value.frame_duration_ms,
    )


def _event_to_proto(event) -> provider_pb2.ProviderEvent:
    if isinstance(event, ProviderSessionOpened):
        return provider_pb2.ProviderEvent(
            session_opened=provider_pb2.ProviderSessionOpened(
                schema_version=event.schema_version,
                session_id=str(event.session_id),
                direction_id=_DIRECTION_TO_PROTO[event.direction_id],
                negotiated_input_format=_pcm_to_proto(event.negotiated_input_format),
                negotiated_output_format=_pcm_to_proto(event.negotiated_output_format),
                capabilities=provider_pb2.ProviderCapabilities(
                    audio_output=event.capabilities.audio_output,
                    transcript_delta=event.capabilities.transcript_delta,
                    translation_delta=event.capabilities.translation_delta,
                    cancellation=event.capabilities.cancellation,
                    cloud_egress=event.capabilities.cloud_egress,
                ),
                event_sequence=event.event_sequence,
            )
        )
    if isinstance(event, ProviderAudioDelta):
        return provider_pb2.ProviderEvent(
            audio_delta=provider_pb2.ProviderAudioDelta(
                schema_version=event.schema_version,
                session_id=str(event.session_id),
                direction_id=_DIRECTION_TO_PROTO[event.direction_id],
                stream_id=str(event.stream_id),
                utterance_id=str(event.utterance_id),
                sequence=event.sequence,
                event_sequence=event.event_sequence,
                provider_monotonic_ns=event.provider_monotonic_ns,
                sample_rate_hz=event.sample_rate_hz,
                channels=event.channels,
                sample_format=_SAMPLE_TO_PROTO[event.sample_format],
                frame_duration_ms=event.frame_duration_ms,
                pcm=event.pcm,
            )
        )
    if isinstance(event, ProviderTranscriptDelta):
        return provider_pb2.ProviderEvent(
            transcript_delta=provider_pb2.ProviderTranscriptDelta(
                schema_version=event.schema_version,
                session_id=str(event.session_id),
                direction_id=_DIRECTION_TO_PROTO[event.direction_id],
                stream_id=str(event.stream_id),
                utterance_id=str(event.utterance_id),
                event_sequence=event.event_sequence,
                text=event.text,
                is_final=event.is_final,
            )
        )
    if isinstance(event, ProviderTranslationDelta):
        return provider_pb2.ProviderEvent(
            translation_delta=provider_pb2.ProviderTranslationDelta(
                schema_version=event.schema_version,
                session_id=str(event.session_id),
                direction_id=_DIRECTION_TO_PROTO[event.direction_id],
                stream_id=str(event.stream_id),
                utterance_id=str(event.utterance_id),
                event_sequence=event.event_sequence,
                text=event.text,
                stable_prefix=event.stable_prefix,
                is_final=event.is_final,
            )
        )
    if isinstance(event, ProviderUtteranceFinal):
        value = provider_pb2.ProviderUtteranceFinal(
            schema_version=event.schema_version,
            session_id=str(event.session_id),
            direction_id=_DIRECTION_TO_PROTO[event.direction_id],
            stream_id=str(event.stream_id),
            utterance_id=str(event.utterance_id),
            event_sequence=event.event_sequence,
            outcome={
                UtteranceOutcome.COMPLETED: provider_pb2.UTTERANCE_OUTCOME_COMPLETED,
                UtteranceOutcome.CANCELLED: provider_pb2.UTTERANCE_OUTCOME_CANCELLED,
                UtteranceOutcome.DROPPED: provider_pb2.UTTERANCE_OUTCOME_DROPPED,
            }[event.outcome],
        )
        if event.final_audio_sequence is not None:
            value.final_audio_sequence = event.final_audio_sequence
        return provider_pb2.ProviderEvent(utterance_final=value)
    if isinstance(event, ProviderSessionClosed):
        return provider_pb2.ProviderEvent(
            session_closed=provider_pb2.ProviderSessionClosed(
                schema_version=event.schema_version,
                session_id=str(event.session_id),
                direction_id=_DIRECTION_TO_PROTO[event.direction_id],
                event_sequence=event.event_sequence,
                reason={
                    SessionCloseReason.USER_STOP: provider_pb2.SESSION_CLOSE_REASON_USER_STOP,
                    SessionCloseReason.ROUTE_REMOVED: provider_pb2.SESSION_CLOSE_REASON_ROUTE_REMOVED,
                    SessionCloseReason.DEVICE_UNAVAILABLE: provider_pb2.SESSION_CLOSE_REASON_DEVICE_UNAVAILABLE,
                    SessionCloseReason.PROVIDER_SWITCH: provider_pb2.SESSION_CLOSE_REASON_PROVIDER_SWITCH,
                    SessionCloseReason.DAEMON_SHUTDOWN: provider_pb2.SESSION_CLOSE_REASON_DAEMON_SHUTDOWN,
                    SessionCloseReason.PROVIDER_FAILURE: provider_pb2.SESSION_CLOSE_REASON_PROVIDER_FAILURE,
                    SessionCloseReason.CLOSE_TIMEOUT: provider_pb2.SESSION_CLOSE_REASON_CLOSE_TIMEOUT,
                }[event.reason],
            )
        )
    if isinstance(event, ProviderHealth):
        return _health_to_proto(event)
    if isinstance(event, ProviderLatency):
        value = provider_pb2.ProviderLatency(
            schema_version=event.schema_version,
            session_id=str(event.session_id),
            direction_id=_DIRECTION_TO_PROTO[event.direction_id],
            stream_id=str(event.stream_id),
            event_sequence=event.event_sequence,
        )
        _set_optional(value, "utterance_id", event.utterance_id)
        for field in (
            "asr_first_text_ms",
            "asr_final_text_ms",
            "mt_first_text_ms",
            "tts_first_audio_ms",
            "provider_total_ms",
        ):
            _set_optional(value, field, getattr(event, field))
        return provider_pb2.ProviderEvent(latency=value)
    if isinstance(event, PrivacySafeProviderError):
        value = provider_pb2.ProviderError(
            schema_version=event.schema_version,
            session_id=str(event.session_id),
            direction_id=_DIRECTION_TO_PROTO[event.direction_id],
            event_sequence=event.event_sequence,
            code={
                SafeErrorCode.PROVIDER_UNAVAILABLE: provider_pb2.SAFE_ERROR_CODE_PROVIDER_UNAVAILABLE,
                SafeErrorCode.MODEL_NOT_LOADED: provider_pb2.SAFE_ERROR_CODE_MODEL_NOT_LOADED,
                SafeErrorCode.UNSUPPORTED_LANGUAGE_PAIR: provider_pb2.SAFE_ERROR_CODE_UNSUPPORTED_LANGUAGE_PAIR,
                SafeErrorCode.QUEUE_OVERFLOW: provider_pb2.SAFE_ERROR_CODE_QUEUE_OVERFLOW,
                SafeErrorCode.CANCELLED: provider_pb2.SAFE_ERROR_CODE_CANCELLED,
                SafeErrorCode.NO_SPEECH: provider_pb2.SAFE_ERROR_CODE_NO_SPEECH,
                SafeErrorCode.CLOUD_NOT_ENABLED: provider_pb2.SAFE_ERROR_CODE_CLOUD_NOT_ENABLED,
                SafeErrorCode.PROVIDER_AUTH_FAILED: provider_pb2.SAFE_ERROR_CODE_PROVIDER_AUTH_FAILED,
            }[event.code],
            retryable=event.retryable,
            safe_message=event.safe_message,
        )
        if event.stream_id is not None:
            value.stream_id = str(event.stream_id)
        if event.utterance_id is not None:
            value.utterance_id = str(event.utterance_id)
        return provider_pb2.ProviderEvent(error=value)
    raise TypeError("unsupported provider event")


def _health_to_proto(event: ProviderHealth) -> provider_pb2.ProviderEvent:
    value = provider_pb2.ProviderHealth(
        schema_version=event.schema_version,
        session_id=str(event.session_id),
        direction_id=_DIRECTION_TO_PROTO[event.direction_id],
        event_sequence=event.event_sequence,
        provider_id={
            ProviderId.LOCAL: provider_pb2.PROVIDER_ID_LOCAL,
            ProviderId.OPENAI: provider_pb2.PROVIDER_ID_OPENAI,
        }[event.provider_id],
        provider_name=event.provider_name,
        state={
            ProviderState.STARTING: provider_pb2.PROVIDER_STATE_STARTING,
            ProviderState.READY: provider_pb2.PROVIDER_STATE_READY,
            ProviderState.DEGRADED: provider_pb2.PROVIDER_STATE_DEGRADED,
            ProviderState.BACKPRESSURE: provider_pb2.PROVIDER_STATE_BACKPRESSURE,
            ProviderState.RESTARTING: provider_pb2.PROVIDER_STATE_RESTARTING,
            ProviderState.UNAVAILABLE: provider_pb2.PROVIDER_STATE_UNAVAILABLE,
            ProviderState.CLOSED: provider_pb2.PROVIDER_STATE_CLOSED,
        }[event.state],
        queues=provider_pb2.ProviderQueues(
            provider_input_buffered_ms=event.queues.provider_input_buffered_ms,
            provider_output_buffered_ms=event.queues.provider_output_buffered_ms,
            queue_lag_ms=event.queues.queue_lag_ms,
        ),
    )
    for model in event.models:
        value.models.add(
            kind={
                "asr": provider_pb2.MODEL_KIND_ASR,
                "mt": provider_pb2.MODEL_KIND_MT,
                "tts": provider_pb2.MODEL_KIND_TTS,
                "speech_to_speech": provider_pb2.MODEL_KIND_SPEECH_TO_SPEECH,
            }[model.kind.value],
            id=model.id,
            state={
                "not_loaded": provider_pb2.MODEL_STATE_NOT_LOADED,
                "loading": provider_pb2.MODEL_STATE_LOADING,
                "ready": provider_pb2.MODEL_STATE_READY,
                "failed": provider_pb2.MODEL_STATE_FAILED,
            }[model.state.value],
            device=(
                {
                    ComputeDevice.CUDA: provider_pb2.COMPUTE_DEVICE_CUDA,
                    ComputeDevice.CPU: provider_pb2.COMPUTE_DEVICE_CPU,
                    ComputeDevice.CLOUD: provider_pb2.COMPUTE_DEVICE_CLOUD,
                }[model.device]
                if model.device is not None
                else provider_pb2.COMPUTE_DEVICE_UNSPECIFIED
            ),
            safe_error_code=model.safe_error_code or "",
        )
    if event.retry is not None:
        value.retry.CopyFrom(
            provider_pb2.ProviderRetry(
                attempt=event.retry.attempt,
                next_retry_after_ms=event.retry.next_retry_after_ms,
                reason_code=event.retry.reason_code,
            )
        )
    if event.safe_error is not None:
        value.safe_error.CopyFrom(
            provider_pb2.SafeErrorSummary(
                code=event.safe_error.code.value,
                message=event.safe_error.message,
                retryable=event.safe_error.retryable,
            )
        )
    return provider_pb2.ProviderEvent(health=value)


def _set_optional(message, field: str, value) -> None:
    if value is not None:
        setattr(message, field, str(value) if isinstance(value, UUID) else value)

"""Private Rust/UDS fixture: observe accepted-open cleanup without stopping it."""

import asyncio
import sys
import time
from pathlib import Path
from uuid import UUID, uuid4

from translator_sidecar.grpc_server import (
    ProviderGrpcServer,
    SidecarServerConfig,
    _ProviderServicer,
)
from translator_sidecar.local.runtime import build_unavailable_local_provider


async def scenario(socket: Path, session_id: UUID) -> None:
    provider = build_unavailable_local_provider(now_ns=time.monotonic_ns)
    server = ProviderGrpcServer(
        SidecarServerConfig(socket, "ab" * 32, uuid4(), time.monotonic_ns),
        local_provider=provider,
        provider_ready=False,
    )
    original = _ProviderServicer._handle_runtime_request
    cancelled = asyncio.Event()
    injected = False

    async def accepted_then_gate(self, request, state, publish):
        nonlocal injected
        events = await original(self, request, state, publish)
        if request.WhichOneof("request") == "open_session" and not injected:
            injected = True
            if not (
                state.session_id == session_id
                and session_id in provider._sessions
                and session_id in provider._scheduler._sessions
                and server.providers._entries[id(provider)].leases == 1
            ):
                raise RuntimeError("accepted ownership evidence missing")
            print("accepted", flush=True)
            try:
                await asyncio.Future()
            finally:
                cancelled.set()
        return events

    _ProviderServicer._handle_runtime_request = accepted_then_gate
    stdin = asyncio.StreamReader()
    pipe, _ = await asyncio.get_running_loop().connect_read_pipe(
        lambda: asyncio.StreamReaderProtocol(stdin), sys.stdin
    )
    try:
        await server.start()
        print("ready", flush=True)
        await cancelled.wait()
        while (
            session_id in provider._sessions
            or session_id in provider._retired_sessions
            or session_id in provider._scheduler._sessions
            or server.providers._entries[id(provider)].leases != 0
        ):
            await asyncio.sleep(0.005)
        print("drained", flush=True)
        if await stdin.readline() != b"stop\n":
            raise RuntimeError("fixture lifetime control missing")
    finally:
        pipe.close()
        await server.stop()
        socket.unlink(missing_ok=True)


async def main() -> None:
    async with asyncio.timeout(10):
        await scenario(Path(sys.argv[1]), UUID(sys.argv[2]))


if __name__ == "__main__":
    asyncio.run(main())

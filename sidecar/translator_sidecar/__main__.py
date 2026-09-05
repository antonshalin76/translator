"""Executable entrypoint for the supervised provider sidecar."""

from __future__ import annotations

import asyncio
import os
import signal
import time
from pathlib import Path
from uuid import UUID

from .cleanup import finish_cleanup
from .grpc_server import ProviderGrpcServer, SidecarServerConfig
from .local.runtime import (
    build_local_provider,
    build_unavailable_local_provider,
)
from .openai_provider import OpenAIRealtimeConfig
from .openai_runtime import OpenAIRealtimeProvider


def _required_environment(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"missing_{name.lower()}")
    return value


def _build_server(config: SidecarServerConfig) -> ProviderGrpcServer:
    return ProviderGrpcServer(
        config,
        local_provider=build_unavailable_local_provider(now_ns=config.now_ns),
        openai_provider=OpenAIRealtimeProvider(
            OpenAIRealtimeConfig(cloud_opt_in=True),
            now_ns=config.now_ns,
        ),
        provider_ready=False,
    )


async def _serve() -> None:
    config = SidecarServerConfig(
        socket_path=Path(_required_environment("TRANSLATOR_SIDECAR_SOCKET")),
        token=_required_environment("TRANSLATOR_SIDECAR_TOKEN"),
        generation_id=UUID(_required_environment("TRANSLATOR_SIDECAR_GENERATION")),
        now_ns=time.monotonic_ns,
    )
    stopped = asyncio.Event()
    loop = asyncio.get_running_loop()
    for caught_signal in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(caught_signal, stopped.set)

    server = _build_server(config)
    try:
        await server.start()
        loading = asyncio.create_task(
            asyncio.to_thread(
                build_local_provider,
                now_ns=config.now_ns,
            )
        )
        try:
            loaded_provider = await finish_cleanup(loading)
        except asyncio.CancelledError:
            if not loading.cancelled() and loading.exception() is None:
                await finish_cleanup(loading.result().shutdown())
            raise
        try:
            server.install_local_provider(loaded_provider)
        except BaseException:
            try:
                await finish_cleanup(loaded_provider.shutdown())
            except Exception:
                pass
            raise
        await server.providers.collect()
        await stopped.wait()
    finally:
        await finish_cleanup(server.stop())


def main() -> None:
    asyncio.run(_serve())


if __name__ == "__main__":
    main()

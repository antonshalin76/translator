"""Own an isolated model run through reap and atomic report publication."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, current_thread, main_thread
from typing import Any, Literal, NoReturn

from pydantic import BaseModel, ConfigDict, Field

from translator_sidecar.cleanup import finish_cleanup

type BenchmarkKind = Literal["task6", "podcast"]


class RunLimits(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_run_seconds: float = Field(
        default=3600, gt=0, allow_inf_nan=False, strict=True
    )
    terminate_grace_seconds: float = Field(
        default=5, gt=0, allow_inf_nan=False, strict=True
    )


DEFAULT_LIMITS = RunLimits()


class BenchmarkRunError(RuntimeError):
    pass


class _Cancellation(Event):
    interrupted = False

    def is_set(self) -> bool:
        return self.interrupted or super().is_set()


@contextmanager
def _defer_sigint(cancel: _Cancellation) -> Iterator[None]:
    previous = signal.getsignal(signal.SIGINT)
    if (
        current_thread() is not main_thread()
        or previous is not signal.default_int_handler
    ):
        yield
        return

    def record_interrupt(_signum, _frame):
        # Signal handlers must not acquire the Event's lock.
        cancel.interrupted = True

    signal.signal(signal.SIGINT, record_interrupt)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)
        if cancel.interrupted:
            previous(signal.SIGINT, None)


class _Task6Report(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    schema_version: Literal["translator.task6-benchmark.v2"]
    generated_at_unix_ns: int
    environment: dict
    fixture: dict
    asr_candidates: list[dict]
    voice_profiles: list[dict]
    quality: dict
    duplex_candidates: list[dict]
    normal_runtime: dict


class _PodcastReport(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    schema_version: Literal["translator.podcast-quality-debug.v1"]
    generated_at: str
    inputs: dict
    candidate_matrix: dict
    models: list[dict]


def _reject_constant(_value: str) -> NoReturn:
    raise ValueError("non-finite report value")


def _validate_report(kind: BenchmarkKind, raw: str) -> dict:
    payload = json.loads(raw, parse_constant=_reject_constant)
    schema = _Task6Report if kind == "task6" else _PodcastReport
    return schema.model_validate(payload).model_dump()


def _retire(worker: subprocess.Popen, grace: float) -> None:
    try:
        worker.terminate()
    finally:
        try:
            worker.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            try:
                worker.kill()
            finally:
                # Keep ownership even if the OS cannot yet reap this process.
                worker.wait()


def _run_isolated(
    kind: BenchmarkKind,
    request: dict[str, Any],
    output_path: Path,
    limits: RunLimits,
    cancel: Event,
) -> dict:
    try:
        if kind not in ("task6", "podcast") or cancel.is_set():
            raise BenchmarkRunError("benchmark run is unavailable")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(
            prefix=".benchmark-", dir=output_path.parent
        ) as directory:
            request_path = Path(directory) / "request.json"
            staging_path = Path(directory) / "result.json"
            request_path.write_text(
                json.dumps(request, allow_nan=False), encoding="utf-8"
            )
            deadline = time.monotonic() + limits.model_run_seconds
            worker = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "translator_sidecar.benchmark.process_run",
                    kind,
                    str(request_path),
                    str(staging_path),
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
                shell=False,
                start_new_session=True,
            )
            try:
                while True:
                    remaining = deadline - time.monotonic()
                    if cancel.is_set() or remaining <= 0:
                        raise BenchmarkRunError("benchmark run did not complete")
                    try:
                        worker.wait(timeout=min(remaining, 0.05))
                        break
                    except subprocess.TimeoutExpired:
                        continue
                if worker.returncode != 0:
                    raise BenchmarkRunError("benchmark worker failed")
                payload = _validate_report(
                    kind, staging_path.read_text(encoding="utf-8")
                )
                if cancel.is_set() or time.monotonic() >= deadline:
                    raise BenchmarkRunError("benchmark run did not complete")
                # Cancellation after this checkpoint does not roll back publication.
                os.replace(staging_path, output_path)
                return payload
            finally:
                if worker.returncode is None:
                    _retire(worker, limits.terminate_grace_seconds)
    except BenchmarkRunError:
        raise
    except Exception:
        raise BenchmarkRunError("benchmark run is unavailable") from None


def run_benchmark(
    kind: BenchmarkKind,
    request: dict[str, Any],
    output_path: Path,
    *,
    limits: RunLimits = DEFAULT_LIMITS,
) -> dict:
    cancel = _Cancellation()
    interrupted = None
    with _defer_sigint(cancel), ThreadPoolExecutor(max_workers=1) as executor:
        try:
            supervisor = executor.submit(
                _run_isolated, kind, request, output_path, limits, cancel
            )
        except BaseException:
            cancel.set()
            raise
        while True:
            try:
                result = supervisor.result()
                break
            except KeyboardInterrupt as error:
                interrupted = interrupted or error
                cancel.set()
            except BaseException:
                if interrupted is None:
                    raise
                break
    if interrupted is not None:
        raise interrupted
    return result


async def run_benchmark_async(
    kind: BenchmarkKind,
    request: dict[str, Any],
    output_path: Path,
    *,
    limits: RunLimits = DEFAULT_LIMITS,
) -> dict:
    cancel = Event()
    operation = asyncio.to_thread(
        _run_isolated, kind, request, output_path, limits, cancel
    )
    try:
        supervisor = asyncio.create_task(operation)
    except BaseException:
        operation.close()
        raise
    try:
        return await asyncio.shield(supervisor)
    except asyncio.CancelledError:
        cancel.set()
        try:
            await finish_cleanup(supervisor)
        except BaseException:
            pass
        raise


def _fatal_cleanup(_error: BaseException) -> NoReturn:
    os._exit(70)


def _worker_main(kind: str, request_path: Path, staging_path: Path) -> None:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if not isinstance(request, dict):
        raise ValueError("invalid benchmark request")
    if kind == "task6" and not request:
        from translator_sidecar.benchmark import task6_live

        payload = task6_live._run_owned(fatal_cleanup=_fatal_cleanup)
    elif kind == "podcast":
        from translator_sidecar.benchmark import podcast_quality

        payload = asyncio.run(
            podcast_quality._run_owned(request, fatal_cleanup=_fatal_cleanup)
        )
    else:
        raise ValueError("invalid benchmark kind")
    staging_path.write_text(
        json.dumps(
            payload, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    try:
        _worker_main(sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]))
    except BaseException:
        os._exit(70)

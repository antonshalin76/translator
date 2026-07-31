"""Production-pipeline E2E benchmark harness for Task 7."""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
from typing import Any

import numpy as np
import psutil

from translator_sidecar.benchmark.task7 import (
    BenchmarkClassification,
    BenchmarkConfig,
    BenchmarkDirection,
    BenchmarkProfile,
    BoundaryObservation,
    ProfileSpec,
    ResourceLimits,
    ResourceSample,
    RunContext,
    run_task7_benchmark,
)
from translator_sidecar.benchmark.task6 import load_quality_corpus
from translator_sidecar.local.model_manifest import load_manifest
from translator_sidecar.local.tts import PiperTts, PiperVoiceRegistry
from translator_sidecar.provider_contract import (
    Language,
    TranslationMode,
    VoiceEngine,
    VoiceGender,
    VoiceProfile,
)


_BRIDGE_SCHEMA = "translator.task7-bridge.v1"
_REPORT_SCHEMA = "translator.task7-e2e.v1"
_CONTENT_KEYS = frozenset(
    {
        "audio",
        "content",
        "pcm",
        "text",
        "transcript",
        "translation",
    }
)
_DIRECTION_BY_BRIDGE = {
    "microphone": BenchmarkDirection.RU_TO_EN,
    "speaker": BenchmarkDirection.EN_TO_RU,
}
_PCM_ARGUMENTS = (
    "--raw",
    "--format=s16le",
    "--rate=16000",
    "--channels=1",
    "--channel-map=mono",
    "--latency-msec=20",
    "--process-time-msec=20",
    "--client-name=translator-task7-e2e",
    "--property=translator.task7_e2e=true",
    "--property=media.role=communication",
)
_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_MANIFEST = _ROOT / "models" / "manifest.json"
_DEFAULT_CORPUS = _ROOT / "sidecar" / "tests" / "quality_corpus" / "task6-v4.json"
_DEFAULT_QUALITY_EVIDENCE = _ROOT / "docs" / "benchmarks" / "task6-results.json"
_DEFAULT_PYTHON = _ROOT / "sidecar" / ".venv" / "bin" / "python"
_DEFAULT_SIDECAR_ROOT = _ROOT / "sidecar"
_TEMP_INPUT_SINK = "translator_task7_ru_in"
_FRAME_BYTES = 16_000 * 2 * 20 // 1_000
_TRAILING_SILENCE_BYTES = 16_000 * 2 * 2_000 // 1_000
_ONSET_RMS = 300.0
_QUIET_RMS = 180.0
_VOICE_IDS = {
    Language.RU: "piper-ru-dmitri-medium",
    Language.EN: "piper-en-ryan-medium",
}
_PRODUCTION_RESOURCE_LIMITS = ResourceLimits(
    cpu_percent=2_880.0,
    rss_bytes=4 * 1024**3,
    vram_mib=6_144,
)
_PRODUCTION_CPU_P95_LIMIT = 2_400.0
_MIN_STABILITY_SAMPLES = 20
_MIN_STABILITY_DURATION_SECONDS = 60.0
_MIN_GROWTH_BUDGET_BYTES = 256 * 1024**2
_MIN_GROWTH_BUDGET_MIB = 256
_RESOURCE_WINDOW_FRACTION = 0.10
_GRAPH_ENDPOINTS = frozenset(
    {
        "translator_mic_out",
        "translator_virtual_mic",
        "translator_remote_in",
        _TEMP_INPUT_SINK,
    }
)
_GRAPH_IDENTITY_PROPERTIES = frozenset(
    {
        "application.process.binary",
        "application.process.id",
        "translator.owner",
        "translator.session_id",
        "translator.test_profile",
    }
)


class Task7E2EError(RuntimeError):
    """The production-pipeline evidence could not be collected safely."""


@dataclass(frozen=True)
class ProfilePairPlan:
    cold_probe_pairs: int
    excluded_warmups: int
    measured_pairs: int

    @property
    def total_pairs(self) -> int:
        return self.cold_probe_pairs + self.excluded_warmups + self.measured_pairs

    @property
    def phases(self) -> tuple[tuple[str, int], ...]:
        phases = []
        if self.cold_probe_pairs:
            phases.append(("cold_probe", self.cold_probe_pairs))
        phases.extend(
            (
                ("warmup", self.excluded_warmups),
                ("measured", self.measured_pairs),
            )
        )
        return tuple(phases)


def profile_pair_plan(
    profile: BenchmarkProfile,
    *,
    measured_count: int,
) -> ProfilePairPlan:
    if measured_count < 100:
        raise ValueError("measured pair count must be at least 100")
    return ProfilePairPlan(
        cold_probe_pairs=1 if profile is BenchmarkProfile.COLD else 0,
        excluded_warmups=10,
        measured_pairs=measured_count,
    )


@dataclass(frozen=True)
class BridgeEvent:
    event: str
    direction: BenchmarkDirection | None
    utterance_id: str | None
    monotonic_ns: int
    sequence: int | None
    queue_lag_ms: float | None
    provider_latency_ms: float | None
    error_code: str | None
    retryable: bool | None
    terminal_outcome: str | None
    restart_attempt: int | None
    raw: Mapping[str, Any]

    @classmethod
    def parse(cls, line: str) -> BridgeEvent:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError("bridge event is not JSON") from error
        if not isinstance(payload, dict):
            raise ValueError("bridge event root is invalid")
        if any(key.lower() in _CONTENT_KEYS for key in payload):
            raise ValueError("bridge event violates privacy contract")
        if payload.get("schema_version") != _BRIDGE_SCHEMA:
            raise ValueError("bridge event schema is invalid")
        event = payload.get("event")
        if not isinstance(event, str) or not event:
            raise ValueError("bridge event name is invalid")
        raw_direction = payload.get("direction")
        if raw_direction is None:
            direction = None
        else:
            try:
                direction = _DIRECTION_BY_BRIDGE[raw_direction]
            except (KeyError, TypeError) as error:
                raise ValueError("bridge event direction is invalid") from error
        monotonic_ns = payload.get("monotonic_ns")
        if (
            not isinstance(monotonic_ns, int)
            or isinstance(monotonic_ns, bool)
            or monotonic_ns < 0
        ):
            raise ValueError("bridge event clock is invalid")
        utterance_id = payload.get("utterance_id")
        if utterance_id is not None and not isinstance(utterance_id, str):
            raise ValueError("bridge utterance identity is invalid")
        sequence = _optional_nonnegative_int(payload, "sequence")
        queue_lag_ms = _optional_nonnegative_float(payload, "queue_lag_ms")
        provider_latency_ms = _optional_nonnegative_float(
            payload,
            "provider_latency_ms",
        )
        if provider_latency_ms is None:
            provider_latency_ms = _optional_nonnegative_float(
                payload,
                "provider_total_ms",
            )
        if provider_latency_ms is None:
            provider_latency_ms = _optional_nonnegative_float(
                payload,
                "tts_first_audio_ms",
            )
        error_code = payload.get("code")
        if error_code is not None and not isinstance(error_code, str):
            raise ValueError("bridge safe error code is invalid")
        retryable = payload.get("retryable")
        if retryable is not None and not isinstance(retryable, bool):
            raise ValueError("bridge retryability is invalid")
        terminal_outcome = payload.get("outcome")
        if terminal_outcome is not None and terminal_outcome not in {
            "completed",
            "cancelled",
            "dropped",
        }:
            raise ValueError("bridge terminal outcome is invalid")
        restart_attempt = _optional_nonnegative_int(payload, "attempt")
        if event == "generation_restart" and (
            restart_attempt is None or restart_attempt == 0
        ):
            raise ValueError("bridge restart attempt is invalid")
        return cls(
            event=event,
            direction=direction,
            utterance_id=utterance_id,
            monotonic_ns=monotonic_ns,
            sequence=sequence,
            queue_lag_ms=queue_lag_ms,
            provider_latency_ms=provider_latency_ms,
            error_code=error_code,
            retryable=retryable,
            terminal_outcome=terminal_outcome,
            restart_attempt=restart_attempt,
            raw=payload,
        )


class BridgeEventStream:
    """Thread-safe directional event demultiplexer for blocking NDJSON."""

    def __init__(self, lines: Iterable[str]) -> None:
        self._condition = threading.Condition()
        self._events: dict[BenchmarkDirection | None, deque[BridgeEvent]] = defaultdict(
            deque
        )
        self._error: Exception | None = None
        self._ended = False
        self._restart_generation = 0
        self._restart_event: BridgeEvent | None = None
        self._restart_consumed: set[tuple[int, BenchmarkDirection]] = set()
        self._reader = threading.Thread(
            target=self._read,
            args=(lines,),
            name="translator-task7-bridge-events",
            daemon=True,
        )
        self._reader.start()

    def _read(self, lines: Iterable[str]) -> None:
        try:
            for line in lines:
                if not line.strip():
                    continue
                event = BridgeEvent.parse(line)
                with self._condition:
                    if event.event == "generation_restart":
                        self._restart_generation += 1
                        self._restart_event = event
                    else:
                        self._events[event.direction].append(event)
                    self._condition.notify_all()
        except Exception as error:
            with self._condition:
                self._error = error
                self._condition.notify_all()
        finally:
            with self._condition:
                self._ended = True
                self._condition.notify_all()

    def next_global(self, expected: set[str], *, timeout_s: float) -> BridgeEvent:
        return self._next(None, expected, timeout_s=timeout_s)

    def next_for(
        self,
        direction: BenchmarkDirection,
        expected: set[str],
        *,
        timeout_s: float,
        after_restart_generation: int | None = None,
    ) -> BridgeEvent:
        return self._next(
            direction,
            expected,
            timeout_s=timeout_s,
            after_restart_generation=after_restart_generation,
        )

    def restart_generation(self) -> int:
        with self._condition:
            return self._restart_generation

    def _next(
        self,
        direction: BenchmarkDirection | None,
        expected: set[str],
        *,
        timeout_s: float,
        after_restart_generation: int | None = None,
    ) -> BridgeEvent:
        if not expected or not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("bridge wait contract is invalid")
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while True:
                restart_key = (
                    self._restart_generation,
                    direction,
                )
                if (
                    direction is not None
                    and after_restart_generation is not None
                    and self._restart_generation > after_restart_generation
                    and restart_key not in self._restart_consumed
                ):
                    assert self._restart_event is not None
                    self._restart_consumed.add(restart_key)
                    return self._restart_event
                failures = self._events[None]
                for index, failure in enumerate(failures):
                    if failure.event == "failure":
                        del failures[index]
                        stage = failure.raw.get("stage", "unknown")
                        code = failure.raw.get("code", "unknown")
                        raise Task7E2EError(f"bridge failed: {stage}:{code}")
                events = self._events[direction]
                for index, event in enumerate(events):
                    if event.event in expected:
                        del events[index]
                        return event
                if self._error is not None:
                    raise Task7E2EError("bridge event stream failed") from self._error
                remaining = deadline - time.monotonic()
                if remaining <= 0 or self._ended:
                    scope = "global" if direction is None else direction.value
                    expected_names = ",".join(sorted(expected))
                    buffered_names = (
                        ",".join(sorted({event.event for event in events})) or "none"
                    )
                    raise Task7E2EError(
                        "bridge event wait timed out: "
                        f"direction={scope} expected={expected_names} "
                        f"buffered={buffered_names} ended={self._ended}"
                    )
                self._condition.wait(remaining)


@dataclass
class _PendingPair:
    contexts: dict[BenchmarkDirection, RunContext]
    running: bool = False
    results: dict[BenchmarkDirection, BoundaryObservation] | None = None
    error: BaseException | None = None
    readers: int = 0


class E2EPairAdapter:
    """Coalesces the benchmark core's two calls into one duplex injection."""

    def __init__(
        self,
        measure_pair: Callable[
            [Mapping[BenchmarkDirection, RunContext]],
            dict[BenchmarkDirection, BoundaryObservation],
        ],
    ) -> None:
        self._measure_pair = measure_pair
        self._condition = threading.Condition()
        self._pairs: dict[int, _PendingPair] = {}

    def measure_direction(self, context: RunContext) -> BoundaryObservation:
        execute = False
        with self._condition:
            pair = self._pairs.setdefault(
                context.pair_index,
                _PendingPair(contexts={}),
            )
            if context.direction in pair.contexts:
                raise Task7E2EError("duplicate direction in E2E pair")
            pair.contexts[context.direction] = context
            pair.readers += 1
            if len(pair.contexts) == 2 and not pair.running:
                pair.running = True
                execute = True
            else:
                while not pair.running:
                    self._condition.wait()
        if execute:
            try:
                results = self._measure_pair(dict(pair.contexts))
                if set(results) != set(BenchmarkDirection):
                    raise Task7E2EError("E2E pair result is incomplete")
                with self._condition:
                    pair.results = results
                    self._condition.notify_all()
            except BaseException as error:
                with self._condition:
                    pair.error = error
                    self._condition.notify_all()
        with self._condition:
            while pair.results is None and pair.error is None:
                self._condition.wait()
            try:
                if pair.error is not None:
                    raise pair.error
                assert pair.results is not None
                return pair.results[context.direction]
            finally:
                pair.readers -= 1
                if pair.readers == 0:
                    del self._pairs[context.pair_index]


class PidTreeResourceSampler:
    """Samples only the bridge process tree and its CUDA compute PIDs."""

    def __init__(
        self,
        root_pid: int,
        *,
        process_factory: Callable[[int], Any] = psutil.Process,
        runner: Callable[..., Any] = subprocess.run,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self._root_pid = root_pid
        self._process_factory = process_factory
        self._runner = runner
        self._clock_ns = clock_ns
        self._last_ns = -1
        self._lock = threading.Lock()
        self._processes: dict[int, Any] = {}

    def __call__(self) -> ResourceSample:
        with self._lock:
            return self._sample()

    def _sample(self) -> ResourceSample:
        root = self._processes.get(self._root_pid)
        if root is None:
            root = self._process_factory(self._root_pid)
        discovered = [root, *root.children(recursive=True)]
        processes = [
            self._processes.get(process.pid, process) for process in discovered
        ]
        self._processes = {process.pid: process for process in processes}
        pids = {process.pid for process in processes}
        cpu_percent = sum(
            max(0.0, float(process.cpu_percent(interval=None))) for process in processes
        )
        rss_bytes = sum(max(0, int(process.memory_info().rss)) for process in processes)
        result = self._runner(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
        vram_mib = 0
        for line in result.stdout.splitlines():
            fields = [field.strip() for field in line.split(",", 1)]
            if len(fields) != 2:
                continue
            try:
                pid, memory = int(fields[0]), int(fields[1])
            except ValueError:
                continue
            if pid in pids:
                vram_mib += max(0, memory)
        monotonic_ns = self._clock_ns()
        if monotonic_ns < self._last_ns:
            raise Task7E2EError("resource clock moved backwards")
        self._last_ns = monotonic_ns
        return ResourceSample(
            monotonic_ns=monotonic_ns,
            cpu_percent=cpu_percent,
            rss_bytes=rss_bytes,
            vram_mib=vram_mib,
        )


@dataclass(frozen=True)
class FixtureIdentity:
    fixture_id: str
    direction: BenchmarkDirection
    pcm_sha256: str
    duration_ms: int

    def to_report_dict(self) -> dict[str, str | int]:
        if not self.fixture_id or len(self.pcm_sha256) != 64 or self.duration_ms <= 0:
            raise ValueError("fixture identity is invalid")
        return {
            "fixture_id": self.fixture_id,
            "direction": self.direction.value,
            "audio_sha256": self.pcm_sha256,
            "duration_ms": self.duration_ms,
        }


@dataclass(frozen=True)
class QualityEvidence:
    sha256: str
    corpus_id: str
    passes_thresholds: bool

    def to_report_dict(self) -> dict[str, str | bool]:
        if len(self.sha256) != 64 or not self.corpus_id:
            raise ValueError("quality evidence identity is invalid")
        return {
            "schema_version": "translator.task6-benchmark.v2",
            "sha256": self.sha256,
            "corpus_id": self.corpus_id,
            "passes_thresholds": self.passes_thresholds,
        }


@dataclass
class AudioFixture:
    identity: FixtureIdentity
    audio: bytearray

    def zeroize(self) -> None:
        self.audio[:] = b"\0" * len(self.audio)


class PulseOnsetMonitor:
    """Detects the first audible 20 ms frame after a quiet boundary."""

    def __init__(
        self,
        device: str,
        *,
        process_factory: Callable[..., Any] = subprocess.Popen,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self._clock_ns = clock_ns
        self._process = process_factory(
            build_pulse_capture_command(device),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )
        if self._process.stdout is None:
            raise Task7E2EError("audible monitor pipe is unavailable")
        self._condition = threading.Condition()
        self._armed_generation = 0
        self._detected: dict[int, int] = {}
        self._last_rms = 0.0
        self._quiet_frames = 0
        self._stopped = False
        self._thread = threading.Thread(
            target=self._read,
            name="translator-task7-audible-monitor",
            daemon=True,
        )
        self._thread.start()

    def _read(self) -> None:
        assert self._process.stdout is not None
        consecutive_loud = 0
        first_loud_ns = 0
        while not self._stopped:
            frame = self._process.stdout.read(_FRAME_BYTES)
            if len(frame) != _FRAME_BYTES:
                break
            samples = np.frombuffer(frame, dtype="<i2").astype(np.float64)
            rms = float(np.sqrt(np.mean(np.square(samples))))
            frame_ns = self._clock_ns()
            with self._condition:
                self._last_rms = rms
                if rms <= _QUIET_RMS:
                    self._quiet_frames += 1
                else:
                    self._quiet_frames = 0
                generation = self._armed_generation
                if generation > 0 and generation not in self._detected:
                    if rms >= _ONSET_RMS:
                        if consecutive_loud == 0:
                            first_loud_ns = frame_ns
                        consecutive_loud += 1
                        if consecutive_loud >= 2:
                            self._detected[generation] = first_loud_ns
                            self._condition.notify_all()
                    else:
                        consecutive_loud = 0
                else:
                    consecutive_loud = 0
                self._condition.notify_all()
        with self._condition:
            self._stopped = True
            self._condition.notify_all()

    def arm_after_quiet(self, *, timeout_s: float = 3.0) -> int:
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while self._quiet_frames < 15:
                if self._stopped:
                    raise Task7E2EError("audible monitor stopped")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise Task7E2EError(
                        f"audio output is not quiet (rms={self._last_rms:.1f})"
                    )
                self._condition.wait(remaining)
            self._armed_generation += 1
            return self._armed_generation

    def wait_onset(self, generation: int, *, timeout_s: float) -> int:
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while generation not in self._detected:
                if self._stopped:
                    raise Task7E2EError("audible monitor stopped")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise Task7E2EError(
                        f"audible onset timed out (last_rms={self._last_rms:.1f})"
                    )
                self._condition.wait(remaining)
            return self._detected.pop(generation)

    def stop(self) -> None:
        self._stopped = True
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=2)
        self._thread.join(timeout=2)


class TemporaryInputSink:
    def __init__(
        self,
        *,
        runner: Callable[..., Any] = subprocess.run,
    ) -> None:
        self._runner = runner
        self.module_id: int | None = None

    @property
    def monitor_source(self) -> str:
        return f"{_TEMP_INPUT_SINK}.monitor"

    def start(self) -> None:
        if self.module_id is not None:
            raise Task7E2EError("temporary input sink is already active")
        result = self._runner(
            [
                "pactl",
                "load-module",
                "module-null-sink",
                f"sink_name={_TEMP_INPUT_SINK}",
                "rate=16000",
                "channels=1",
                "channel_map=mono",
                (
                    "sink_properties=device.description=Translator_Task7_Input"
                    " translator.task7_e2e=true"
                ),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
        try:
            self.module_id = int(result.stdout.strip())
        except ValueError as error:
            raise Task7E2EError("temporary sink module id is invalid") from error

    def stop(self) -> None:
        if self.module_id is None:
            return
        module_id = self.module_id
        self.module_id = None
        self._runner(
            ["pactl", "unload-module", str(module_id)],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )


class BridgeProcess:
    def __init__(
        self,
        command: Sequence[str],
        *,
        process_factory: Callable[..., Any] = subprocess.Popen,
    ) -> None:
        self._process = process_factory(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        if self._process.stdin is None or self._process.stdout is None:
            raise Task7E2EError("bridge control pipes are unavailable")
        self.events = BridgeEventStream(self._process.stdout)
        self._stderr: deque[str] = deque(maxlen=20)
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr,
            name="translator-task7-bridge-stderr",
            daemon=True,
        )
        self._stderr_thread.start()

    @property
    def pid(self) -> int:
        return int(self._process.pid)

    def _drain_stderr(self) -> None:
        if self._process.stderr is None:
            return
        for line in self._process.stderr:
            self._stderr.append(line.rstrip())

    def wait_ready(self, *, timeout_s: float = 180.0) -> BridgeEvent:
        try:
            return self.events.next_global({"ready"}, timeout_s=timeout_s)
        except Task7E2EError as error:
            detail = self.diagnostic_tail()
            raise Task7E2EError(f"bridge did not become ready: {detail}") from error

    def diagnostic_tail(self) -> str:
        return "; ".join(self._stderr)

    def stop(self) -> None:
        if self._process.poll() is not None:
            return
        assert self._process.stdin is not None
        try:
            self._process.stdin.write("stop\n")
            self._process.stdin.flush()
            self._process.wait(timeout=15)
        except (BrokenPipeError, subprocess.TimeoutExpired):
            self.kill()
            raise Task7E2EError("bridge stop timed out") from None
        finally:
            self._stderr_thread.join(timeout=2)
        if self._process.returncode != 0:
            raise Task7E2EError("bridge stop failed: " + "; ".join(self._stderr))

    def kill(self) -> None:
        if self._process.poll() is not None:
            return
        try:
            os.killpg(self._process.pid, signal.SIGTERM)
            self._process.wait(timeout=3)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(self._process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self._process.wait(timeout=3)


class ContinuousResourceCollector:
    def __init__(self, sampler: Callable[[], ResourceSample]) -> None:
        self._sampler = sampler
        self._samples: list[ResourceSample] = []
        self._error: BaseException | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="translator-task7-resource-sampler",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                self._samples.append(self._sampler())
            except BaseException as error:
                self._error = error
                return
            self._stop.wait(max(0.0, 1.0 - (time.monotonic() - started)))

    def stop(self) -> tuple[ResourceSample, ...]:
        self._stop.set()
        self._thread.join(timeout=3)
        if self._thread.is_alive():
            raise Task7E2EError("resource collector did not stop")
        if self._error is not None:
            raise Task7E2EError("resource collection failed") from self._error
        try:
            self._samples.append(self._sampler())
        except BaseException as error:
            raise Task7E2EError("resource collection failed") from error
        return tuple(self._samples)


class LivePipelineMeasurement:
    def __init__(
        self,
        *,
        events: BridgeEventStream,
        fixtures: Mapping[BenchmarkDirection, Sequence[AudioFixture]],
        monitors: Mapping[BenchmarkDirection, PulseOnsetMonitor],
        timeout_s: float,
        profile: BenchmarkProfile,
        total_pairs: int,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self._events = events
        self._fixtures = fixtures
        self._monitors = monitors
        self._timeout_s = timeout_s
        self._profile = profile
        self._total_pairs = total_pairs
        self._clock_ns = clock_ns
        self._sleeper = sleeper
        self._started_ns: int | None = None

    def measure_pair(
        self,
        contexts: Mapping[BenchmarkDirection, RunContext],
    ) -> dict[BenchmarkDirection, BoundaryObservation]:
        if set(contexts) != set(BenchmarkDirection):
            raise Task7E2EError("live pair context is incomplete")
        if self._started_ns is None:
            self._started_ns = self._clock_ns()
        pair_index = next(iter(contexts.values())).pair_index
        if any(context.pair_index != pair_index for context in contexts.values()):
            raise Task7E2EError("live pair indices differ")
        generations = {
            direction: self._monitors[direction].arm_after_quiet()
            for direction in BenchmarkDirection
        }
        selected = {
            direction: self._fixture_for(context)
            for direction, context in contexts.items()
        }
        restart_generation = self._events.restart_generation()
        playback_targets = {
            BenchmarkDirection.RU_TO_EN: _TEMP_INPUT_SINK,
            BenchmarkDirection.EN_TO_RU: "translator_remote_in",
        }
        injection_ns = self._clock_ns()
        with ThreadPoolExecutor(
            max_workers=4,
            thread_name_prefix="translator-task7-e2e-pair",
        ) as executor:
            playback = {
                direction: executor.submit(
                    _play_fixture,
                    playback_targets[direction],
                    selected[direction].audio,
                    self._timeout_s,
                )
                for direction in BenchmarkDirection
            }
            measurements = {
                direction: executor.submit(
                    self._collect_direction,
                    direction,
                    injection_ns,
                    generations[direction],
                    restart_generation,
                )
                for direction in BenchmarkDirection
            }
            for future in playback.values():
                future.result()
            results = {
                direction: future.result() for direction, future in measurements.items()
            }
        self._pace(pair_index)
        return results

    def _fixture_for(self, context: RunContext) -> AudioFixture:
        values = self._fixtures[context.direction]
        if not values:
            raise Task7E2EError("fixture pool is empty")
        index = context.pair_index % len(values)
        return values[index]

    def _collect_direction(
        self,
        direction: BenchmarkDirection,
        injection_ns: int,
        generation: int,
        restart_generation: int,
    ) -> BoundaryObservation:
        speech = self._events.next_for(
            direction,
            {"generation_restart", "speech_started"},
            timeout_s=self._timeout_s,
            after_restart_generation=restart_generation,
        )
        if speech.event == "generation_restart":
            return _restarted_observation(injection_ns)
        utterance_id = speech.utterance_id
        first_audio_ns: int | None = None
        last_audio_ns: int | None = None
        provider_latency_ms: float | None = None
        queue_lag_ms = 0.0
        terminal = False
        dropped = False
        event_names = {
            "audio_frame",
            "asr_final",
            "first_audio_expired",
            "generation_restart",
            "provider_error",
            "provider_latency",
            "transcript_final",
            "translation_final",
            "utterance_terminal",
            "utterance_terminal_outcome",
        }
        deadline = time.monotonic() + self._timeout_s
        while not terminal:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise Task7E2EError("utterance terminal timed out")
            event = self._events.next_for(
                direction,
                event_names,
                timeout_s=remaining,
                after_restart_generation=restart_generation,
            )
            if event.event == "generation_restart":
                return _restarted_observation(injection_ns)
            if (
                event.utterance_id is not None
                and utterance_id is not None
                and event.utterance_id != utterance_id
            ):
                continue
            if event.event == "audio_frame":
                first_audio_ns = first_audio_ns or event.monotonic_ns
                last_audio_ns = event.monotonic_ns
                queue_lag_ms = max(queue_lag_ms, event.queue_lag_ms or 0.0)
            elif event.event == "first_audio_expired":
                dropped = True
            elif event.event == "provider_latency":
                provider_latency_ms = event.provider_latency_ms
            elif event.event == "provider_error":
                raise Task7E2EError(
                    f"{direction.value} provider error: {event.error_code}"
                )
            elif event.event == "utterance_terminal_outcome":
                if event.terminal_outcome not in {None, "completed"}:
                    dropped = True
            elif event.event == "utterance_terminal":
                terminal = True
        if dropped:
            return BoundaryObservation(
                speech_onset_ns=injection_ns,
                capture_ns=speech.monotonic_ns,
                first_audio_ns=None,
                last_audio_ns=None,
                first_audible_ns=None,
                queue_lag_ms=queue_lag_ms,
                provider_latency_ms=provider_latency_ms,
                dropped=True,
            )
        try:
            first_audible_ns = self._monitors[direction].wait_onset(
                generation,
                timeout_s=max(0.1, deadline - time.monotonic()),
            )
        except Task7E2EError as error:
            raise Task7E2EError(
                f"{direction.value} audible evidence failed: {error}"
            ) from error
        if first_audio_ns is None or last_audio_ns is None:
            raise Task7E2EError("utterance completed without translated audio")
        return BoundaryObservation(
            speech_onset_ns=injection_ns,
            capture_ns=speech.monotonic_ns,
            first_audio_ns=first_audio_ns,
            last_audio_ns=last_audio_ns,
            first_audible_ns=first_audible_ns,
            queue_lag_ms=queue_lag_ms,
            provider_latency_ms=provider_latency_ms,
            quality_passed=True,
        )

    def _pace(self, pair_index: int) -> None:
        if self._profile is not BenchmarkProfile.SOAK_30_MINUTES:
            return
        assert self._started_ns is not None
        target_ns = self._started_ns + int(
            (pair_index + 1) * 1_800 * 1_000_000_000 / self._total_pairs
        )
        remaining_s = (target_ns - self._clock_ns()) / 1_000_000_000
        if remaining_s > 0:
            self._sleeper(remaining_s)


def _restarted_observation(injection_ns: int) -> BoundaryObservation:
    return BoundaryObservation(
        speech_onset_ns=injection_ns,
        capture_ns=injection_ns,
        first_audio_ns=None,
        last_audio_ns=None,
        first_audible_ns=None,
        queue_lag_ms=0.0,
        provider_latency_ms=None,
        dropped=True,
        restarted=True,
    )


def run_smoke_pairs(
    measure_pair: Callable[
        [Mapping[BenchmarkDirection, RunContext]],
        dict[BenchmarkDirection, BoundaryObservation],
    ],
    *,
    pair_count: int,
) -> tuple[dict[BenchmarkDirection, BoundaryObservation], ...]:
    if pair_count <= 0:
        raise ValueError("smoke pair count must be positive")
    observations = []
    for pair_index in range(pair_count):
        pair = measure_pair(
            {
                direction: RunContext(
                    direction=direction,
                    pair_index=pair_index,
                    is_warmup=True,
                    mode=TranslationMode.QUALITY_FIRST,
                )
                for direction in BenchmarkDirection
            }
        )
        if set(pair) != set(BenchmarkDirection):
            raise Task7E2EError("smoke pair result is incomplete")
        observations.append(pair)
    return tuple(observations)


def build_local_fixtures(
    manifest_path: Path,
    corpus_path: Path,
    *,
    measured_pool_size: int = 10,
) -> tuple[
    dict[BenchmarkDirection, tuple[AudioFixture, ...]], tuple[FixtureIdentity, ...]
]:
    if measured_pool_size <= 0:
        raise ValueError("measured fixture pool must be positive")
    manifest = load_manifest(manifest_path)
    corpus = load_quality_corpus(corpus_path)
    voice_paths = {}
    for language, model_id in _VOICE_IDS.items():
        model = manifest.models[model_id]
        for model_file in model.files:
            manifest.resolve_runtime_file(model.id, model_file.path)
        onnx = next(
            model.cache_path / model_file.path
            for model_file in model.files
            if model_file.path.endswith(".onnx")
        )
        voice_paths[(language, VoiceGender.MALE)] = onnx
    tts = PiperTts(PiperVoiceRegistry(voice_paths))
    fixture_sets: dict[BenchmarkDirection, list[AudioFixture]] = {
        direction: [] for direction in BenchmarkDirection
    }
    identities: list[FixtureIdentity] = []
    source_rows = [
        *((f"warmup-{index:03d}", value) for index, value in enumerate(corpus.warmups)),
        *(
            (f"measured-{index:03d}", value)
            for index, value in enumerate(corpus.cases[:measured_pool_size])
        ),
    ]
    for row_id, row in source_rows:
        for direction, language, text in (
            (BenchmarkDirection.RU_TO_EN, Language.RU, row.ru),
            (BenchmarkDirection.EN_TO_RU, Language.EN, row.en),
        ):
            audio = bytearray(
                b"".join(
                    tts.synthesize_frames(
                        text,
                        target_language=language,
                        voice_profile=VoiceProfile(
                            language=language,
                            gender=VoiceGender.MALE,
                            engine=VoiceEngine.PIPER,
                        ),
                        mode=TranslationMode.QUALITY_FIRST,
                        output_sample_rate_hz=16_000,
                        output_channels=1,
                        frame_duration_ms=100,
                    )
                )
            )
            audio.extend(b"\0" * _TRAILING_SILENCE_BYTES)
            identity = FixtureIdentity(
                fixture_id=f"{row_id}-{direction.value}",
                direction=direction,
                pcm_sha256=hashlib.sha256(audio).hexdigest(),
                duration_ms=len(audio) * 1_000 // (16_000 * 2),
            )
            fixture_sets[direction].append(AudioFixture(identity, audio))
            identities.append(identity)
    return (
        {direction: tuple(values) for direction, values in fixture_sets.items()},
        tuple(identities),
    )


def pulse_graph_summary(
    *,
    runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    endpoint_snapshot: dict[str, list[dict[str, Any]]] = {
        "sinks": [],
        "sources": [],
    }
    route_snapshot: dict[str, list[dict[str, Any]]] = {
        "sink_inputs": [],
        "source_outputs": [],
    }
    for kind in ("sinks", "sources"):
        result = runner(
            ["pactl", "--format=json", "list", kind],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
        endpoints = json.loads(result.stdout)
        if not isinstance(endpoints, list):
            raise Task7E2EError("PulseAudio endpoint graph is invalid")
        endpoint_snapshot[kind] = sorted(
            (
                {
                    "index": endpoint.get("index"),
                    "name": endpoint.get("name"),
                    "owner_module": endpoint.get("owner_module"),
                }
                for endpoint in endpoints
                if isinstance(endpoint, dict)
                and endpoint.get("name") in _GRAPH_ENDPOINTS
            ),
            key=lambda value: (
                str(value["name"]),
                str(value["index"]),
                str(value["owner_module"]),
            ),
        )
    for kind in ("sink-inputs", "source-outputs"):
        result = runner(
            ["pactl", "--format=json", "list", kind],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
        streams = json.loads(result.stdout)
        if not isinstance(streams, list):
            raise Task7E2EError("PulseAudio route graph is invalid")
        route_key = "sink_inputs" if kind == "sink-inputs" else "source_outputs"
        target_key = "sink" if kind == "sink-inputs" else "source"
        routes = []
        for stream in streams:
            if not isinstance(stream, dict):
                continue
            properties = stream.get("properties", {})
            if not isinstance(properties, dict) or not any(
                key.startswith("translator.") for key in properties
            ):
                continue
            identity_properties = {
                key: properties[key]
                for key in sorted(_GRAPH_IDENTITY_PROPERTIES)
                if key in properties
            }
            routes.append(
                {
                    "client": stream.get("client"),
                    "index": stream.get("index"),
                    "module": stream.get("module"),
                    "object_serial": properties.get("object.serial"),
                    "properties": identity_properties,
                    "target": stream.get(target_key),
                }
            )
        route_snapshot[route_key] = sorted(
            routes,
            key=lambda value: (
                str(value["index"]),
                str(value["object_serial"]),
                str(value["target"]),
            ),
        )
    counts = {
        "sinks": len(endpoint_snapshot["sinks"]),
        "sources": len(endpoint_snapshot["sources"]),
        "sink_inputs": len(route_snapshot["sink_inputs"]),
        "source_outputs": len(route_snapshot["source_outputs"]),
    }
    return {
        "counts": counts,
        "endpoints": endpoint_snapshot,
        "routes": route_snapshot,
    }


def _play_fixture(device: str, audio: bytearray, timeout_s: float) -> None:
    result = subprocess.run(
        build_pulse_playback_command(device),
        input=bytes(audio),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=timeout_s,
        check=False,
    )
    if result.returncode != 0:
        raise Task7E2EError("fixture playback failed")


def resource_payload(
    samples: Sequence[ResourceSample],
    *,
    limits: ResourceLimits,
    cpu_p95_limit: float,
    required_duration_seconds: float | None = None,
) -> dict[str, Any]:
    if not samples:
        raise Task7E2EError("continuous resource evidence is empty")
    if any(
        current.monotonic_ns < previous.monotonic_ns
        for previous, current in zip(samples, samples[1:], strict=False)
    ):
        raise Task7E2EError("resource clock moved backwards")
    duration_seconds = (
        samples[-1].monotonic_ns - samples[0].monotonic_ns
    ) / 1_000_000_000
    cpu_values = sorted(sample.cpu_percent for sample in samples)
    cpu_p95 = cpu_values[math.ceil(len(cpu_values) * 0.95) - 1]
    cpu_peak = cpu_values[-1]
    rss_peak = max(sample.rss_bytes for sample in samples)
    vram_peak = max(sample.vram_mib for sample in samples)
    enough_stability_evidence = (
        len(samples) >= _MIN_STABILITY_SAMPLES
        and duration_seconds >= _MIN_STABILITY_DURATION_SECONDS
    )
    stability: dict[str, Any] = {
        "minimum_samples": _MIN_STABILITY_SAMPLES,
        "minimum_duration_seconds": _MIN_STABILITY_DURATION_SECONDS,
        "sample_count": len(samples),
    }
    violations = []
    if (
        required_duration_seconds is not None
        and duration_seconds < required_duration_seconds
    ):
        violations.append("declared_duration")
    if limits.cpu_percent is not None and cpu_peak > limits.cpu_percent:
        violations.append("cpu_peak")
    if cpu_p95 > cpu_p95_limit:
        violations.append("cpu_p95")
    if limits.rss_bytes is not None and rss_peak > limits.rss_bytes:
        violations.append("rss_peak")
    if limits.vram_mib is not None and vram_peak > limits.vram_mib:
        violations.append("vram_peak")
    if not enough_stability_evidence:
        violations.append("insufficient_stability_evidence")
    else:
        window_size = max(
            2,
            math.ceil(len(samples) * _RESOURCE_WINDOW_FRACTION),
        )
        first_window = samples[:window_size]
        last_window = samples[-window_size:]
        first_rss = int(np.median([sample.rss_bytes for sample in first_window]))
        last_rss = int(np.median([sample.rss_bytes for sample in last_window]))
        first_vram = float(np.median([sample.vram_mib for sample in first_window]))
        last_vram = float(np.median([sample.vram_mib for sample in last_window]))
        rss_growth = max(0, last_rss - first_rss)
        vram_growth = max(0.0, last_vram - first_vram)
        rss_growth_limit = max(
            _MIN_GROWTH_BUDGET_BYTES,
            math.ceil(first_rss * 0.10),
        )
        vram_growth_limit = max(
            _MIN_GROWTH_BUDGET_MIB,
            math.ceil(first_vram * 0.10),
        )
        stability.update(
            {
                "window_fraction": _RESOURCE_WINDOW_FRACTION,
                "window_sample_count": window_size,
                "rss_baseline_bytes": first_rss,
                "rss_final_bytes": last_rss,
                "rss_growth_bytes": rss_growth,
                "rss_growth_limit_bytes": rss_growth_limit,
                "vram_baseline_mib": first_vram,
                "vram_final_mib": last_vram,
                "vram_growth_mib": vram_growth,
                "vram_growth_limit_mib": vram_growth_limit,
            }
        )
        if rss_growth > rss_growth_limit:
            violations.append("rss_growth")
        if vram_growth > vram_growth_limit:
            violations.append("vram_growth")
    return {
        "samples": [sample.to_dict() for sample in samples],
        "observed_duration_seconds": duration_seconds,
        "cpu_percent_peak": cpu_peak,
        "cpu_percent_p95": cpu_p95,
        "rss_bytes_peak": rss_peak,
        "vram_mib_peak": vram_peak,
        "limits": {
            "cpu_percent_peak": limits.cpu_percent,
            "cpu_percent_p95": cpu_p95_limit,
            "rss_bytes_peak": limits.rss_bytes,
            "vram_mib_peak": limits.vram_mib,
            "required_duration_seconds": required_duration_seconds,
        },
        "stability": stability,
        "violations": violations,
        "within_limits": not violations,
    }


def _smoke_direction_payload(
    observation: BoundaryObservation,
) -> dict[str, Any]:
    return {
        "dropped": observation.dropped,
        "restarted": observation.restarted,
        "speech_onset_to_first_audible_ms": (
            observation.first_audible_ns - observation.speech_onset_ns
        )
        / 1_000_000
        if observation.first_audible_ns is not None
        else None,
        "capture_to_first_audio_ms": (
            observation.first_audio_ns - observation.capture_ns
        )
        / 1_000_000
        if observation.first_audio_ns is not None
        else None,
        "capture_to_last_audio_ms": (observation.last_audio_ns - observation.capture_ns)
        / 1_000_000
        if observation.last_audio_ns is not None
        else None,
        "queue_lag_ms": observation.queue_lag_ms,
        "provider_latency_ms": observation.provider_latency_ms,
    }


def build_smoke_report_payload(
    observation_pairs: Sequence[Mapping[BenchmarkDirection, BoundaryObservation]],
    *,
    resources: Sequence[ResourceSample],
    quality_evidence: QualityEvidence,
    graph_before: Mapping[str, Any],
    graph_after: Mapping[str, Any],
) -> dict[str, Any]:
    if not observation_pairs:
        raise ValueError("smoke observations are empty")
    pairs = []
    for pair_index, observations in enumerate(observation_pairs):
        if set(observations) != set(BenchmarkDirection):
            raise ValueError("smoke observation pair is incomplete")
        pairs.append(
            {
                "pair_index": pair_index,
                "directions": {
                    direction.value: _smoke_direction_payload(observations[direction])
                    for direction in BenchmarkDirection
                },
            }
        )
    return {
        "schema_version": "translator.task7-e2e-smoke.v1",
        "evidence_scope": "production_pipeline_e2e_smoke",
        "release_eligible": False,
        "simultaneous": True,
        "pair_count": len(pairs),
        "pairs": pairs,
        "directions": pairs[-1]["directions"],
        "resources": {
            **resource_payload(
                resources,
                limits=_PRODUCTION_RESOURCE_LIMITS,
                cpu_p95_limit=_PRODUCTION_CPU_P95_LIMIT,
            ),
            "scope": "bridge_process_tree_and_filtered_gpu",
        },
        "quality_evidence": quality_evidence.to_report_dict(),
        "graph": {
            "before": dict(graph_before),
            "after": dict(graph_after),
            "unchanged_after_cleanup": graph_before == graph_after,
        },
    }


def load_task6_quality_evidence(path: Path) -> QualityEvidence:
    raw = path.read_bytes()
    payload = json.loads(raw)
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != "translator.task6-benchmark.v2"
    ):
        raise ValueError("Task 6 quality evidence schema is invalid")
    quality_wrapper = payload.get("quality")
    quality = (
        quality_wrapper.get("quality") if isinstance(quality_wrapper, dict) else None
    )
    if not isinstance(quality, dict):
        raise ValueError("Task 6 quality evidence is incomplete")
    corpus_id = quality.get("corpus_id")
    passes = quality.get("passes_thresholds")
    if not isinstance(corpus_id, str) or not corpus_id:
        raise ValueError("Task 6 quality corpus identity is invalid")
    if passes is not True:
        raise ValueError("Task 6 quality thresholds did not pass")
    return QualityEvidence(
        sha256=hashlib.sha256(raw).hexdigest(),
        corpus_id=corpus_id,
        passes_thresholds=True,
    )


def build_pulse_capture_command(device: str) -> tuple[str, ...]:
    _validate_pulse_name(device)
    return (
        "parec",
        "--record",
        f"--device={device}",
        *_PCM_ARGUMENTS,
        "--stream-name=translator-task7-e2e-detector",
    )


def build_pulse_playback_command(device: str) -> tuple[str, ...]:
    _validate_pulse_name(device)
    return (
        "pacat",
        "--playback",
        f"--device={device}",
        *_PCM_ARGUMENTS,
        "--stream-name=translator-task7-e2e-fixture",
    )


def build_e2e_report_payload(
    *,
    benchmark_payload: Mapping[str, Any],
    profile: str,
    bridge_sha256: str,
    corpus_sha256: str,
    fixtures: Sequence[FixtureIdentity],
    startup_ready_ms: float,
    graph_before: Mapping[str, Any],
    graph_after: Mapping[str, Any],
    quality_evidence: QualityEvidence,
    cold_probe: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if benchmark_payload.get("schema_version") != "translator.task7-benchmark.v1":
        raise ValueError("benchmark payload schema is invalid")
    if (
        len(bridge_sha256) != 64
        or len(corpus_sha256) != 64
        or not math.isfinite(startup_ready_ms)
        or startup_ready_ms < 0
    ):
        raise ValueError("E2E provenance is invalid")
    payload = dict(benchmark_payload)
    payload["schema_version"] = _REPORT_SCHEMA
    payload["evidence_scope"] = "production_pipeline_e2e"
    raw_profile = payload.get("profile")
    profile_payload = dict(raw_profile) if isinstance(raw_profile, dict) else {}
    profile_payload["kind"] = profile
    payload["profile"] = profile_payload
    resources = dict(payload.get("resources", {}))
    within_limits = resources.get("within_limits")
    if not isinstance(within_limits, bool):
        raise ValueError("E2E resource verdict is invalid")
    classification = payload.get("classification")
    allowed_classifications = {item.value for item in BenchmarkClassification}
    if classification not in allowed_classifications:
        raise ValueError("E2E classification is invalid")
    if not within_limits:
        classification = BenchmarkClassification.FAILS_USABLE_LIMIT.value
    payload["classification"] = classification
    payload["release_eligible"] = (
        quality_evidence.passes_thresholds
        and within_limits
        and classification != BenchmarkClassification.FAILS_USABLE_LIMIT.value
    )
    resources["scope"] = "bridge_process_tree_and_filtered_gpu"
    payload["resources"] = resources
    payload["runtime"] = {
        "bridge_sha256": bridge_sha256,
        "harness_sha256": _sha256_file(Path(__file__)),
        "startup_ready_ms": startup_ready_ms,
        "measurement_uncertainty_ms": 20,
    }
    payload["fixtures"] = {
        "corpus_sha256": corpus_sha256,
        "items": [fixture.to_report_dict() for fixture in fixtures],
    }
    payload["quality_evidence"] = quality_evidence.to_report_dict()
    if profile == BenchmarkProfile.COLD.value:
        if cold_probe is None:
            raise ValueError("cold E2E report lacks first-pair evidence")
        payload["cold_probe"] = dict(cold_probe)
    elif cold_probe is not None:
        raise ValueError("non-cold E2E report contains cold evidence")
    payload["graph"] = {
        "before": dict(graph_before),
        "after": dict(graph_after),
        "unchanged_after_cleanup": dict(graph_before) == dict(graph_after),
    }
    serialized = json.dumps(payload, sort_keys=True)
    if any(f'"{key}"' in serialized.lower() for key in _CONTENT_KEYS):
        raise ValueError("E2E report violates privacy contract")
    return payload


def build_cold_probe_payload(
    observations: Mapping[BenchmarkDirection, BoundaryObservation],
) -> dict[str, Any]:
    if set(observations) != set(BenchmarkDirection):
        raise ValueError("cold probe observations are incomplete")
    return {
        "excluded_from_percentiles": True,
        "directions": {
            direction.value: _smoke_direction_payload(observations[direction])
            for direction in BenchmarkDirection
        },
    }


def _validate_pulse_name(value: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or "\x00" in value
        or any(character.isspace() for character in value)
    ):
        raise ValueError("PulseAudio target is invalid")


def _optional_nonnegative_int(
    payload: Mapping[str, Any],
    key: str,
) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"bridge {key} is invalid")
    return value


def _optional_nonnegative_float(
    payload: Mapping[str, Any],
    key: str,
) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise ValueError(f"bridge {key} is invalid")
    return float(value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure the real Task 7 production duplex pipeline",
    )
    parser.add_argument(
        "--profile",
        choices=("cold", "warm", "soak_30_minutes"),
        required=True,
    )
    parser.add_argument("--physical-sink", required=True)
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=_DEFAULT_MANIFEST)
    parser.add_argument("--corpus", type=Path, default=_DEFAULT_CORPUS)
    parser.add_argument(
        "--quality-evidence",
        type=Path,
        default=_DEFAULT_QUALITY_EVIDENCE,
    )
    parser.add_argument("--python", type=Path, default=_DEFAULT_PYTHON)
    parser.add_argument(
        "--sidecar-root",
        type=Path,
        default=_DEFAULT_SIDECAR_ROOT,
    )
    parser.add_argument("--measured-count", type=int, default=100)
    parser.add_argument("--fixture-pool-size", type=int, default=10)
    parser.add_argument("--utterance-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-pairs", type=int, default=1)
    return parser.parse_args(argv)


def run_live_e2e(arguments: argparse.Namespace) -> dict[str, Any]:
    _validate_pulse_name(arguments.physical_sink)
    if arguments.smoke_pairs <= 0:
        raise ValueError("smoke pair count must be positive")
    if not arguments.smoke and arguments.smoke_pairs != 1:
        raise ValueError("--smoke-pairs requires --smoke")
    if not arguments.bridge.is_file():
        raise Task7E2EError("benchmark bridge binary is unavailable")
    if (
        (not arguments.smoke and arguments.measured_count < 100)
        or not math.isfinite(arguments.utterance_timeout_seconds)
        or arguments.utterance_timeout_seconds <= 0
    ):
        raise Task7E2EError("benchmark cardinality or timeout is invalid")
    fixtures, fixture_identities = build_local_fixtures(
        arguments.manifest,
        arguments.corpus,
        measured_pool_size=arguments.fixture_pool_size,
    )
    quality_evidence = load_task6_quality_evidence(arguments.quality_evidence)
    graph_before = pulse_graph_summary()
    temporary_sink = TemporaryInputSink()
    bridge: BridgeProcess | None = None
    monitors: dict[BenchmarkDirection, PulseOnsetMonitor] = {}
    collector: ContinuousResourceCollector | None = None
    resource_samples: tuple[ResourceSample, ...] = ()
    benchmark_payload: dict[str, Any] | None = None
    startup_ready_ms = 0.0
    runtime_parent = Path(os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}"))
    socket_path = (
        runtime_parent / "translator" / f"task7-e2e-sidecar-{os.getpid()}.sock"
    )
    profile = BenchmarkProfile(arguments.profile)
    pair_plan = profile_pair_plan(
        profile,
        measured_count=arguments.measured_count,
    )
    profile_spec = ProfileSpec(
        profile,
        duration_seconds=(
            1_800.0 if profile is BenchmarkProfile.SOAK_30_MINUTES else None
        ),
    )
    config = (
        None
        if arguments.smoke
        else BenchmarkConfig(
            profile=profile_spec,
            measured_count_per_direction=arguments.measured_count,
            resource_limits=_PRODUCTION_RESOURCE_LIMITS,
        )
    )
    total_pairs = arguments.smoke_pairs if config is None else pair_plan.total_pairs
    smoke_observation_pairs: (
        tuple[dict[BenchmarkDirection, BoundaryObservation], ...] | None
    ) = None
    cold_probe_observations: dict[BenchmarkDirection, BoundaryObservation] | None = None
    try:
        temporary_sink.start()
        bridge_command = (
            str(arguments.bridge.resolve()),
            "--microphone-capture",
            temporary_sink.monitor_source,
            "--speaker-playback",
            arguments.physical_sink,
            "--python",
            str(arguments.python.absolute()),
            "--sidecar-root",
            str(arguments.sidecar_root.resolve()),
            "--socket-path",
            str(socket_path),
        )
        startup_ns = time.monotonic_ns()
        bridge = BridgeProcess(bridge_command)
        bridge.wait_ready()
        startup_ready_ms = (time.monotonic_ns() - startup_ns) / 1_000_000
        monitors = {
            BenchmarkDirection.RU_TO_EN: PulseOnsetMonitor("translator_virtual_mic"),
            BenchmarkDirection.EN_TO_RU: PulseOnsetMonitor(
                f"{arguments.physical_sink}.monitor"
            ),
        }
        sampler = PidTreeResourceSampler(bridge.pid)
        sampler()
        collector = ContinuousResourceCollector(sampler)
        collector.start()
        measurement = LivePipelineMeasurement(
            events=bridge.events,
            fixtures=fixtures,
            monitors=monitors,
            timeout_s=arguments.utterance_timeout_seconds,
            profile=profile,
            total_pairs=total_pairs,
        )
        adapter = E2EPairAdapter(measurement.measure_pair)
        try:
            if config is None:
                smoke_observation_pairs = run_smoke_pairs(
                    measurement.measure_pair,
                    pair_count=arguments.smoke_pairs,
                )
            else:
                if pair_plan.cold_probe_pairs:
                    cold_probe_observations = measurement.measure_pair(
                        {
                            direction: RunContext(
                                direction=direction,
                                pair_index=0,
                                is_warmup=True,
                                mode=TranslationMode.QUALITY_FIRST,
                            )
                            for direction in BenchmarkDirection
                        }
                    )
                report = run_task7_benchmark(
                    config,
                    measure_direction=adapter.measure_direction,
                    sample_resources=sampler,
                )
                benchmark_payload = report.to_dict()
        except Task7E2EError as error:
            detail = bridge.diagnostic_tail()
            raise Task7E2EError(
                f"{error}; bridge diagnostics: {detail or 'none'}"
            ) from error
    finally:
        cleanup_errors: list[BaseException] = []
        if collector is not None:
            try:
                resource_samples = collector.stop()
            except BaseException as error:
                cleanup_errors.append(error)
        for monitor in monitors.values():
            try:
                monitor.stop()
            except BaseException as error:
                cleanup_errors.append(error)
        if bridge is not None:
            try:
                bridge.stop()
            except BaseException as error:
                bridge.kill()
                cleanup_errors.append(error)
        try:
            temporary_sink.stop()
        except BaseException as error:
            cleanup_errors.append(error)
        for values in fixtures.values():
            for fixture in values:
                fixture.zeroize()
        if cleanup_errors:
            raise Task7E2EError("E2E cleanup failed") from cleanup_errors[0]
    graph_after = pulse_graph_summary()
    if graph_after != graph_before:
        raise Task7E2EError("PulseAudio graph changed after E2E cleanup")
    if smoke_observation_pairs is not None:
        return build_smoke_report_payload(
            smoke_observation_pairs,
            resources=resource_samples,
            quality_evidence=quality_evidence,
            graph_before=graph_before,
            graph_after=graph_after,
        )
    if benchmark_payload is None:
        raise Task7E2EError("E2E benchmark did not produce a report")
    benchmark_payload["resources"] = resource_payload(
        resource_samples,
        limits=_PRODUCTION_RESOURCE_LIMITS,
        cpu_p95_limit=_PRODUCTION_CPU_P95_LIMIT,
        required_duration_seconds=(
            1_800.0 if profile is BenchmarkProfile.SOAK_30_MINUTES else None
        ),
    )
    return build_e2e_report_payload(
        benchmark_payload=benchmark_payload,
        profile=profile.value,
        bridge_sha256=_sha256_file(arguments.bridge),
        corpus_sha256=_sha256_file(arguments.corpus),
        fixtures=fixture_identities,
        startup_ready_ms=startup_ready_ms,
        graph_before=graph_before,
        graph_after=graph_after,
        quality_evidence=quality_evidence,
        cold_probe=(
            build_cold_probe_payload(cold_probe_observations)
            if cold_probe_observations is not None
            else None
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    try:
        payload = run_live_e2e(arguments)
        serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        arguments.output.write_text(serialized, encoding="utf-8")
    except (Task7E2EError, ValueError, OSError, subprocess.SubprocessError) as error:
        print(f"Task 7 E2E benchmark failed: {error}", file=os.sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

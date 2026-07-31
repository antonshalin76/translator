"""Workstation graph-boundary adapter for the Task 7 benchmark core."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import selectors
import subprocess
import sys
from threading import Lock
import time
from typing import Any, Protocol

import numpy as np
import psutil

from translator_sidecar.benchmark.task7 import (
    BenchmarkConfig,
    BenchmarkDirection,
    BenchmarkProfile,
    BenchmarkReport,
    BoundaryObservation,
    ProfileSpec,
    ResourceSample,
    RunContext,
    run_task7_benchmark,
)


_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_TASK6_EVIDENCE = _ROOT / "docs" / "benchmarks" / "task6-results.json"
_TASK6_SCHEMA = "translator.task6-benchmark.v2"
_SELECTED_MODEL = "faster-whisper-small"
_OUTGOING_PLAYBACK = "translator_mic_out"
_OUTGOING_CAPTURE = "translator_virtual_mic"
_SAMPLE_FORMAT = "s16le"
_SAMPLE_RATE_HZ = 16_000
_CHANNELS = 1
_BYTES_PER_SAMPLE = 2
_MARKER_SAMPLES = 1_024
_MARKER_AMPLITUDE = 3_000
_CORRELATION_THRESHOLD = 0.75
_DEFAULT_MAX_EVIDENCE_AGE_SECONDS = 48 * 60 * 60
_TASK7_PROPERTY = "translator.task7"


class LiveBenchmarkError(RuntimeError):
    """Graph-boundary evidence cannot be collected safely."""


@dataclass(frozen=True)
class ProviderLatencyEvidence:
    path: Path
    sha256: str
    generated_at_unix_ns: int
    model_id: str
    excluded_warmups: int
    measured_per_direction: int
    ru_to_en_latency_ms: tuple[float, ...]
    en_to_ru_latency_ms: tuple[float, ...]
    cold_inference_ms: float
    warm_asr_p95_ms: float

    def latency_ms(
        self,
        direction: BenchmarkDirection,
        pair_index: int,
        *,
        cold: bool = False,
    ) -> float:
        values = {
            BenchmarkDirection.RU_TO_EN: self.ru_to_en_latency_ms,
            BenchmarkDirection.EN_TO_RU: self.en_to_ru_latency_ms,
        }[direction]
        latency = values[pair_index % len(values)]
        if cold and pair_index == 0:
            latency += max(0.0, self.cold_inference_ms - self.warm_asr_p95_ms)
        return latency


@dataclass(frozen=True)
class AudioTransportMeasurement:
    playback_write_ns: int
    first_audible_ns: int
    last_audible_ns: int
    queue_lag_ms: float
    sample_format: str = _SAMPLE_FORMAT
    sample_rate_hz: int = _SAMPLE_RATE_HZ
    channels: int = _CHANNELS

    def __post_init__(self) -> None:
        if (
            self.sample_format != _SAMPLE_FORMAT
            or self.sample_rate_hz != _SAMPLE_RATE_HZ
            or self.channels != _CHANNELS
        ):
            raise LiveBenchmarkError(
                "audio transport format must be s16le/16000/mono"
            )
        if not (
            0
            <= self.playback_write_ns
            <= self.first_audible_ns
            <= self.last_audible_ns
        ):
            raise LiveBenchmarkError(
                "audio transport timestamps must be monotonic"
            )
        if not math.isfinite(self.queue_lag_ms) or self.queue_lag_ms < 0:
            raise LiveBenchmarkError(
                "audio transport queue lag must be finite and nonnegative"
            )


@dataclass(frozen=True)
class _EndpointIdentity:
    index: int
    name: str
    object_serial: str
    sample_specification: str
    active_port: str | None
    monitor_source: str | None


@dataclass(frozen=True)
class PulseGraphSnapshot:
    outgoing_playback: str
    outgoing_capture: str
    incoming_playback: str
    incoming_capture: str
    identities: tuple[_EndpointIdentity, ...]


class _AudioTransport(Protocol):
    def measure(
        self,
        *,
        playback_target: str,
        capture_target: str,
        marker: np.ndarray,
        timeout_s: float,
    ) -> AudioTransportMeasurement: ...

    def assert_no_processes(self) -> None: ...


class _GraphProbe(Protocol):
    def snapshot(self) -> PulseGraphSnapshot: ...

    def assert_unchanged(self, baseline: PulseGraphSnapshot) -> None: ...

    def assert_no_task7_streams(self) -> None: ...


def load_task6_provider_evidence(
    path: Path,
    *,
    now_ns: int | None = None,
    max_age_seconds: float = _DEFAULT_MAX_EVIDENCE_AGE_SECONDS,
) -> ProviderLatencyEvidence:
    if not math.isfinite(max_age_seconds) or max_age_seconds <= 0:
        raise LiveBenchmarkError("Task 6 evidence age limit must be positive")
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise LiveBenchmarkError("Task 6 evidence is unreadable") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != _TASK6_SCHEMA:
        raise LiveBenchmarkError("Task 6 evidence schema is unsupported")

    generated_at = _required_int(payload, "generated_at_unix_ns")
    current_ns = time.time_ns() if now_ns is None else now_ns
    age_ns = current_ns - generated_at
    if age_ns < 0 or age_ns > max_age_seconds * 1_000_000_000:
        raise LiveBenchmarkError("Task 6 evidence is stale or future-dated")

    normal_runtime = _required_dict(payload, "normal_runtime")
    if (
        normal_runtime.get("selected_asr") != _SELECTED_MODEL
        or normal_runtime.get("resident_model_id") != "small"
    ):
        raise LiveBenchmarkError(
            "Task 6 evidence does not select the small resident model"
        )

    duplex = _find_model_record(payload, "duplex_candidates", _SELECTED_MODEL)
    if duplex.get("simultaneous") is not True:
        raise LiveBenchmarkError(
            "Task 6 provider evidence is not simultaneous duplex"
        )
    excluded_warmups = _record_int(duplex, "excluded_warmups")
    measured_count = _record_int(duplex, "measured_per_direction")
    if excluded_warmups != 10 or measured_count < 100:
        raise LiveBenchmarkError(
            "Task 6 evidence lacks 10 warmups and 100 measured samples"
        )
    ru_to_en = _latency_series(duplex, "ru_to_en_latency_ms", measured_count)
    en_to_ru = _latency_series(duplex, "en_to_ru_latency_ms", measured_count)

    asr = _find_model_record(payload, "asr_candidates", _SELECTED_MODEL)
    if (
        _record_int(asr, "excluded_warmups") != 10
        or _record_int(asr, "measured_count") < 100
    ):
        raise LiveBenchmarkError("Task 6 small ASR evidence is incomplete")
    cold_inference_ms = _record_float(asr, "cold_inference_ms")
    warm_asr_p95_ms = _record_float(asr, "warm_p95_ms")

    return ProviderLatencyEvidence(
        path=path.resolve(),
        sha256=hashlib.sha256(raw).hexdigest(),
        generated_at_unix_ns=generated_at,
        model_id=_SELECTED_MODEL,
        excluded_warmups=excluded_warmups,
        measured_per_direction=measured_count,
        ru_to_en_latency_ms=ru_to_en,
        en_to_ru_latency_ms=en_to_ru,
        cold_inference_ms=cold_inference_ms,
        warm_asr_p95_ms=warm_asr_p95_ms,
    )


def deterministic_marker() -> np.ndarray:
    state = 0x6D2B79F5
    values = np.empty(_MARKER_SAMPLES, dtype=np.int16)
    for index in range(_MARKER_SAMPLES):
        state ^= (state << 13) & 0xFFFFFFFF
        state ^= state >> 17
        state ^= (state << 5) & 0xFFFFFFFF
        values[index] = _MARKER_AMPLITUDE if state & 1 else -_MARKER_AMPLITUDE
    return values


def correlate_marker(
    captured_pcm: bytes,
    marker: np.ndarray,
    *,
    threshold: float = _CORRELATION_THRESHOLD,
) -> int:
    if len(captured_pcm) % _BYTES_PER_SAMPLE:
        raise LiveBenchmarkError("captured audio is not valid s16le PCM")
    if (
        marker.dtype != np.int16
        or marker.ndim != 1
        or marker.size == 0
    ):
        raise LiveBenchmarkError("correlation marker format is invalid")
    captured = np.frombuffer(captured_pcm, dtype="<i2").astype(np.float64)
    reference = marker.astype(np.float64)
    if captured.size < reference.size:
        raise LiveBenchmarkError("correlation marker was not captured")

    numerator = np.correlate(captured, reference, mode="valid")
    energy = np.convolve(
        captured * captured,
        np.ones(reference.size, dtype=np.float64),
        mode="valid",
    )
    denominator = np.sqrt(energy * float(np.dot(reference, reference)))
    scores = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0,
    )
    onset = int(np.argmax(scores))
    if not math.isfinite(float(scores[onset])) or scores[onset] < threshold:
        raise LiveBenchmarkError("correlation marker was not detected")
    return onset


def build_capture_command(target: str) -> tuple[str, ...]:
    _validate_target(target)
    return (
        "parec",
        "--raw",
        f"--device={target}",
        f"--format={_SAMPLE_FORMAT}",
        f"--rate={_SAMPLE_RATE_HZ}",
        f"--channels={_CHANNELS}",
        "--latency-msec=20",
        "--property=media.role=communication",
        "--property=translator.owner=true",
        f"--property={_TASK7_PROPERTY}=true",
    )


def build_playback_command(target: str) -> tuple[str, ...]:
    _validate_target(target)
    return (
        "pacat",
        "--raw",
        "--playback",
        f"--device={target}",
        f"--format={_SAMPLE_FORMAT}",
        f"--rate={_SAMPLE_RATE_HZ}",
        f"--channels={_CHANNELS}",
        "--latency-msec=20",
        "--property=media.role=communication",
        "--property=translator.owner=true",
        f"--property={_TASK7_PROPERTY}=true",
    )


class PulseGraphProbe:
    def __init__(
        self,
        physical_sink: str,
        *,
        runner: Callable[..., Any] = subprocess.run,
    ) -> None:
        _validate_target(physical_sink)
        if physical_sink.startswith("translator_"):
            raise LiveBenchmarkError("incoming sink must be a physical sink")
        self._physical_sink = physical_sink
        self._runner = runner

    def snapshot(self) -> PulseGraphSnapshot:
        sinks = self._pactl_json("sinks")
        sources = self._pactl_json("sources")
        outgoing_sink = _named_endpoint(sinks, _OUTGOING_PLAYBACK, "sink")
        physical_sink = _named_endpoint(
            sinks,
            self._physical_sink,
            "physical sink",
        )
        outgoing_source = _named_endpoint(
            sources,
            _OUTGOING_CAPTURE,
            "source",
        )
        monitor_name = physical_sink.get("monitor_source")
        if not isinstance(monitor_name, str) or not monitor_name:
            raise LiveBenchmarkError("physical sink has no monitor source")
        monitor_source = _named_endpoint(
            sources,
            monitor_name,
            "physical sink monitor",
        )
        identities = tuple(
            _endpoint_identity(endpoint)
            for endpoint in (
                outgoing_sink,
                outgoing_source,
                physical_sink,
                monitor_source,
            )
        )
        if len(set(identities)) != 4:
            raise LiveBenchmarkError("audio endpoint identity is ambiguous")
        return PulseGraphSnapshot(
            outgoing_playback=_OUTGOING_PLAYBACK,
            outgoing_capture=_OUTGOING_CAPTURE,
            incoming_playback=self._physical_sink,
            incoming_capture=monitor_name,
            identities=identities,
        )

    def assert_unchanged(self, baseline: PulseGraphSnapshot) -> None:
        if self.snapshot() != baseline:
            raise LiveBenchmarkError("audio device changed during benchmark")

    def assert_no_task7_streams(self) -> None:
        leaked = []
        for kind in ("sink-inputs", "source-outputs"):
            for stream in self._pactl_json(kind):
                properties = stream.get("properties")
                if (
                    isinstance(properties, dict)
                    and str(properties.get(_TASK7_PROPERTY, "")).lower()
                    == "true"
                ):
                    leaked.append((kind, stream.get("index")))
        if leaked:
            raise LiveBenchmarkError("Task 7 Pulse stream leak detected")

    def _pactl_json(self, kind: str) -> list[dict[str, Any]]:
        environment = {**os.environ, "LC_ALL": "C", "LANG": "C"}
        try:
            result = self._runner(
                ("pactl", "--format=json", "list", kind),
                check=True,
                capture_output=True,
                text=True,
                timeout=3,
                env=environment,
            )
            payload = json.loads(result.stdout)
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            subprocess.SubprocessError,
        ) as error:
            raise LiveBenchmarkError(
                f"Pulse graph inspection failed for {kind}"
            ) from error
        if not isinstance(payload, list) or any(
            not isinstance(item, dict) for item in payload
        ):
            raise LiveBenchmarkError("Pulse graph response is malformed")
        return payload


class SubprocessAudioTransport:
    def __init__(
        self,
        *,
        popen: Callable[..., Any] = subprocess.Popen,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self._popen = popen
        self._clock_ns = clock_ns
        self._sleeper = sleeper
        self._active: list[Any] = []
        self._lock = Lock()

    def measure(
        self,
        *,
        playback_target: str,
        capture_target: str,
        marker: np.ndarray,
        timeout_s: float,
    ) -> AudioTransportMeasurement:
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise LiveBenchmarkError("correlation timeout must be positive")
        if marker.dtype != np.int16 or marker.ndim != 1:
            raise LiveBenchmarkError("marker must be one-dimensional int16")

        capture = None
        playback = None
        captured = bytearray()
        selector = selectors.DefaultSelector()
        try:
            capture = self._start(
                build_capture_command(capture_target),
                stdout=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
            )
            if capture.stdout is None:
                raise LiveBenchmarkError("capture process has no PCM output")
            selector.register(capture.stdout, selectors.EVENT_READ)
            self._sleeper(0.05)
            playback = self._start(
                build_playback_command(playback_target),
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
            )
            if playback.stdin is None:
                raise LiveBenchmarkError("playback process has no PCM input")

            playback_write_ns = self._clock_ns()
            playback.stdin.write(marker.astype("<i2", copy=False).tobytes())
            playback.stdin.close()

            deadline_ns = playback_write_ns + int(timeout_s * 1_000_000_000)
            onset = None
            detected_ns = None
            while self._clock_ns() < deadline_ns:
                remaining = max(
                    0.0,
                    (deadline_ns - self._clock_ns()) / 1_000_000_000,
                )
                events = selector.select(min(0.05, remaining))
                if not events:
                    continue
                chunk = os.read(capture.stdout.fileno(), 4_096)
                if not chunk:
                    raise LiveBenchmarkError(
                        "capture process ended before correlation"
                    )
                captured.extend(chunk)
                try:
                    onset = correlate_marker(bytes(captured), marker)
                except LiveBenchmarkError:
                    continue
                detected_ns = self._clock_ns()
                break
            if onset is None or detected_ns is None:
                raise LiveBenchmarkError("correlation timeout")

            trailing_samples = (
                len(captured) // _BYTES_PER_SAMPLE - onset
            )
            inferred_onset_ns = detected_ns - int(
                trailing_samples * 1_000_000_000 / _SAMPLE_RATE_HZ
            )
            marker_duration_ns = int(
                marker.size * 1_000_000_000 / _SAMPLE_RATE_HZ
            )
            if inferred_onset_ns < playback_write_ns:
                raise LiveBenchmarkError(
                    "correlated audio timestamp is non-monotonic"
                )
            _wait_success(playback, timeout_s=1.0)
            graph_latency_ms = (
                inferred_onset_ns - playback_write_ns
            ) / 1_000_000
            return AudioTransportMeasurement(
                playback_write_ns=playback_write_ns,
                first_audible_ns=inferred_onset_ns,
                last_audible_ns=inferred_onset_ns + marker_duration_ns,
                queue_lag_ms=graph_latency_ms,
            )
        finally:
            selector.close()
            self._stop(playback)
            self._stop(capture)

    def assert_no_processes(self) -> None:
        with self._lock:
            active = [
                process
                for process in self._active
                if process.poll() is None
            ]
        if active:
            raise LiveBenchmarkError("Task 7 audio process leak detected")

    def _start(self, command: Sequence[str], **streams: Any) -> Any:
        environment = {**os.environ, "LC_ALL": "C", "LANG": "C"}
        try:
            process = self._popen(
                command,
                stderr=subprocess.DEVNULL,
                env=environment,
                **streams,
            )
        except OSError as error:
            raise LiveBenchmarkError("Pulse PCM process failed to start") from error
        with self._lock:
            self._active.append(process)
        return process

    def _stop(self, process: Any | None) -> None:
        if process is None:
            return
        try:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=0.5)
        finally:
            if process.poll() is not None:
                with self._lock:
                    if process in self._active:
                        self._active.remove(process)


class ResourceSampler:
    def __init__(
        self,
        *,
        process: Any | None = None,
        runner: Callable[..., Any] = subprocess.run,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self._process = psutil.Process() if process is None else process
        self._runner = runner
        self._clock_ns = clock_ns
        self._last_ns: int | None = None
        self._lock = Lock()
        self._process.cpu_percent(interval=None)

    def __call__(self) -> ResourceSample:
        with self._lock:
            monotonic_ns = self._clock_ns()
            if self._last_ns is not None and monotonic_ns <= self._last_ns:
                raise LiveBenchmarkError(
                    "resource monotonic clock did not advance"
                )
            self._last_ns = monotonic_ns
            cpu_percent = float(self._process.cpu_percent(interval=None))
            rss_bytes = int(self._process.memory_info().rss)
            try:
                result = self._runner(
                    (
                        "nvidia-smi",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ),
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=2,
                )
                values = [
                    int(line.strip())
                    for line in result.stdout.splitlines()
                    if line.strip()
                ]
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                raise LiveBenchmarkError(
                    "GPU resource telemetry is unavailable"
                ) from error
            if not values:
                raise LiveBenchmarkError("GPU resource telemetry is empty")
            return ResourceSample(
                monotonic_ns=monotonic_ns,
                cpu_percent=cpu_percent,
                rss_bytes=rss_bytes,
                vram_mib=max(values),
            )


class LiveBoundaryAdapter:
    def __init__(
        self,
        evidence: ProviderLatencyEvidence,
        *,
        probe: _GraphProbe,
        audio: _AudioTransport,
        profile: BenchmarkProfile = BenchmarkProfile.WARM,
        timeout_s: float = 2.0,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise LiveBenchmarkError("correlation timeout must be positive")
        self.evidence = evidence
        self._probe = probe
        self._audio = audio
        self._profile = profile
        self._timeout_s = timeout_s
        self._clock_ns = clock_ns
        self._sleeper = sleeper
        self._baseline = probe.snapshot()
        self._marker = deterministic_marker()
        self._run_started_ns: int | None = None
        self._pair_interval_s = 0.0
        self._probe.assert_no_task7_streams()

    @property
    def baseline(self) -> PulseGraphSnapshot:
        return self._baseline

    def prepare(self, config: BenchmarkConfig) -> None:
        if config.profile.kind is not self._profile:
            raise LiveBenchmarkError("adapter and benchmark profiles differ")
        self._probe.assert_unchanged(self._baseline)
        self._probe.assert_no_task7_streams()
        self._run_started_ns = self._clock_ns()
        if config.profile.kind is BenchmarkProfile.SOAK_30_MINUTES:
            duration = config.profile.duration_seconds
            if duration is None or duration < 1_800:
                raise LiveBenchmarkError("30-minute soak duration is invalid")
            total_pairs = (
                config.excluded_warmups
                + config.measured_count_per_direction
            )
            self._pair_interval_s = duration / total_pairs

    def measure_direction(self, context: RunContext) -> BoundaryObservation:
        self._probe.assert_unchanged(self._baseline)
        playback_target, capture_target = {
            BenchmarkDirection.RU_TO_EN: (
                self._baseline.outgoing_playback,
                self._baseline.outgoing_capture,
            ),
            BenchmarkDirection.EN_TO_RU: (
                self._baseline.incoming_playback,
                self._baseline.incoming_capture,
            ),
        }[context.direction]
        measurement = self._audio.measure(
            playback_target=playback_target,
            capture_target=capture_target,
            marker=self._marker,
            timeout_s=self._timeout_s,
        )
        self._probe.assert_unchanged(self._baseline)
        provider_latency_ms = self.evidence.latency_ms(
            context.direction,
            context.pair_index,
            cold=self._profile is BenchmarkProfile.COLD,
        )
        speech_onset_ns = measurement.playback_write_ns - int(
            provider_latency_ms * 1_000_000
        )
        if speech_onset_ns < 0:
            raise LiveBenchmarkError(
                "combined boundary timestamp is non-monotonic"
            )
        self._pace_soak(context.pair_index)
        return BoundaryObservation(
            speech_onset_ns=speech_onset_ns,
            capture_ns=speech_onset_ns,
            first_audio_ns=measurement.playback_write_ns,
            last_audio_ns=measurement.last_audible_ns,
            first_audible_ns=measurement.first_audible_ns,
            queue_lag_ms=measurement.queue_lag_ms,
            provider_latency_ms=provider_latency_ms,
        )

    def teardown(self) -> None:
        self._audio.assert_no_processes()
        self._probe.assert_unchanged(self._baseline)
        self._probe.assert_no_task7_streams()

    def _pace_soak(self, pair_index: int) -> None:
        if self._pair_interval_s <= 0 or self._run_started_ns is None:
            return
        deadline_ns = self._run_started_ns + int(
            (pair_index + 1) * self._pair_interval_s * 1_000_000_000
        )
        remaining_s = (deadline_ns - self._clock_ns()) / 1_000_000_000
        if remaining_s > 0:
            self._sleeper(remaining_s)


def run_task7_live(
    config: BenchmarkConfig,
    *,
    adapter: LiveBoundaryAdapter,
    sample_resources: Callable[[], ResourceSample],
) -> BenchmarkReport:
    try:
        adapter.prepare(config)
        return run_task7_benchmark(
            config,
            measure_direction=adapter.measure_direction,
            sample_resources=sample_resources,
        )
    finally:
        adapter.teardown()


def build_live_report_payload(
    report: BenchmarkReport,
    adapter: LiveBoundaryAdapter,
) -> dict[str, Any]:
    payload = report.to_dict()
    payload["schema_version"] = "translator.task7-component-benchmark.v1"
    payload["evidence_scope"] = "hybrid_component_estimate"
    payload["release_eligible"] = False
    payload["release_classification"] = None
    payload["component_estimate_classification"] = payload.pop(
        "classification"
    )
    payload["resources"]["scope"] = "benchmark_process_and_total_gpu"
    payload["workstation_evidence"] = {
        "task6": {
            "schema_version": _TASK6_SCHEMA,
            "sha256": adapter.evidence.sha256,
            "generated_at_unix_ns": adapter.evidence.generated_at_unix_ns,
            "model_id": adapter.evidence.model_id,
            "measured_per_direction": (
                adapter.evidence.measured_per_direction
            ),
        },
        "graph": {
            "outgoing": {
                "playback": adapter.baseline.outgoing_playback,
                "capture": adapter.baseline.outgoing_capture,
            },
            "incoming": {
                "playback": adapter.baseline.incoming_playback,
                "capture": adapter.baseline.incoming_capture,
            },
            "sample_format": _SAMPLE_FORMAT,
            "sample_rate_hz": _SAMPLE_RATE_HZ,
            "channels": _CHANNELS,
            "correlation_marker": "deterministic-bpsk-v1",
        },
    }
    return payload


def _find_model_record(
    payload: dict[str, Any],
    key: str,
    model_id: str,
) -> dict[str, Any]:
    records = payload.get(key)
    if not isinstance(records, list):
        raise LiveBenchmarkError(f"Task 6 {key} is missing")
    matches = [
        record
        for record in records
        if isinstance(record, dict) and record.get("model_id") == model_id
    ]
    if len(matches) != 1:
        raise LiveBenchmarkError(
            f"Task 6 {key} must contain one selected small-model record"
        )
    return matches[0]


def _required_dict(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise LiveBenchmarkError(f"Task 6 {key} is missing")
    return value


def _required_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise LiveBenchmarkError(f"Task 6 {key} is invalid")
    return value


def _record_int(record: dict[str, Any], key: str) -> int:
    value = record.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise LiveBenchmarkError(f"Task 6 {key} is invalid")
    return value


def _record_float(record: dict[str, Any], key: str) -> float:
    value = record.get(key)
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise LiveBenchmarkError(f"Task 6 {key} is invalid")
    return float(value)


def _latency_series(
    record: dict[str, Any],
    key: str,
    measured_count: int,
) -> tuple[float, ...]:
    values = record.get(key)
    if not isinstance(values, list) or len(values) != measured_count:
        raise LiveBenchmarkError(f"Task 6 {key} count is invalid")
    measured = tuple(_finite_latency(value, key) for value in values)
    if len(measured) < 100:
        raise LiveBenchmarkError(f"Task 6 {key} has fewer than 100 samples")
    return measured


def _finite_latency(value: Any, key: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise LiveBenchmarkError(f"Task 6 {key} contains invalid latency")
    return float(value)


def _validate_target(target: str) -> None:
    if (
        not isinstance(target, str)
        or not target
        or any(character.isspace() for character in target)
        or "\x00" in target
    ):
        raise LiveBenchmarkError("Pulse target name is invalid")


def _named_endpoint(
    endpoints: list[dict[str, Any]],
    name: str,
    label: str,
) -> dict[str, Any]:
    matches = [endpoint for endpoint in endpoints if endpoint.get("name") == name]
    if len(matches) != 1:
        raise LiveBenchmarkError(f"required {label} endpoint is unavailable")
    return matches[0]


def _endpoint_identity(endpoint: dict[str, Any]) -> _EndpointIdentity:
    index = endpoint.get("index")
    name = endpoint.get("name")
    sample_specification = endpoint.get("sample_specification")
    properties = endpoint.get("properties")
    if (
        not isinstance(index, int)
        or not isinstance(name, str)
        or not isinstance(sample_specification, str)
        or not isinstance(properties, dict)
    ):
        raise LiveBenchmarkError("audio endpoint identity is malformed")
    serial = properties.get("object.serial")
    if not isinstance(serial, str) or not serial:
        raise LiveBenchmarkError("audio endpoint object serial is missing")
    active_port = endpoint.get("active_port")
    monitor_source = endpoint.get("monitor_source")
    if active_port is not None and not isinstance(active_port, str):
        raise LiveBenchmarkError("audio endpoint active port is malformed")
    if monitor_source is not None and not isinstance(monitor_source, str):
        raise LiveBenchmarkError("audio endpoint monitor is malformed")
    return _EndpointIdentity(
        index=index,
        name=name,
        object_serial=serial,
        sample_specification=sample_specification,
        active_port=active_port,
        monitor_source=monitor_source,
    )


def _wait_success(process: Any, *, timeout_s: float) -> None:
    try:
        return_code = process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired as error:
        raise LiveBenchmarkError("playback process did not drain") from error
    if return_code != 0:
        raise LiveBenchmarkError("playback process failed")


def _profile_spec(profile: BenchmarkProfile) -> ProfileSpec:
    if profile is BenchmarkProfile.SOAK_30_MINUTES:
        return ProfileSpec(profile, duration_seconds=1_800.0)
    return ProfileSpec(profile)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure Task 7 Pulse/PipeWire graph boundaries",
    )
    parser.add_argument(
        "--profile",
        choices=[profile.value for profile in BenchmarkProfile],
        required=True,
    )
    parser.add_argument("--physical-sink", required=True)
    parser.add_argument(
        "--task6-evidence",
        type=Path,
        default=_DEFAULT_TASK6_EVIDENCE,
    )
    parser.add_argument(
        "--max-task6-age-hours",
        type=float,
        default=48.0,
    )
    parser.add_argument(
        "--measured-count",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--correlation-timeout-seconds",
        type=float,
        default=2.0,
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--confirm-audible-marker",
        action="store_true",
        help="required because this benchmark plays correlation markers",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_args(argv)
    if not arguments.confirm_audible_marker:
        print(
            "refusing audible benchmark without --confirm-audible-marker",
            file=sys.stderr,
        )
        return 2
    try:
        profile = BenchmarkProfile(arguments.profile)
        evidence = load_task6_provider_evidence(
            arguments.task6_evidence,
            max_age_seconds=arguments.max_task6_age_hours * 60 * 60,
        )
        probe = PulseGraphProbe(arguments.physical_sink)
        audio = SubprocessAudioTransport()
        adapter = LiveBoundaryAdapter(
            evidence,
            probe=probe,
            audio=audio,
            profile=profile,
            timeout_s=arguments.correlation_timeout_seconds,
        )
        config = BenchmarkConfig(
            profile=_profile_spec(profile),
            measured_count_per_direction=arguments.measured_count,
        )
        report = run_task7_live(
            config,
            adapter=adapter,
            sample_resources=ResourceSampler(),
        )
        payload = build_live_report_payload(report, adapter)
        serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        if arguments.output is None:
            sys.stdout.write(serialized)
        else:
            arguments.output.write_text(serialized, encoding="utf-8")
    except (LiveBenchmarkError, ValueError) as error:
        print(f"Task 7 live benchmark failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

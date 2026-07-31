from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from threading import Barrier, Lock

import numpy as np
import pytest

from translator_sidecar.benchmark.task7 import (
    BenchmarkConfig,
    BenchmarkDirection,
    BenchmarkProfile,
    ProfileSpec,
    ResourceSample,
)
from translator_sidecar.benchmark.task7_live import (
    AudioTransportMeasurement,
    LiveBenchmarkError,
    LiveBoundaryAdapter,
    PulseGraphProbe,
    ResourceSampler,
    build_live_report_payload,
    build_capture_command,
    build_playback_command,
    correlate_marker,
    deterministic_marker,
    load_task6_provider_evidence,
    run_task7_live,
)


PHYSICAL_SINK = (
    "alsa_output.usb-Jieli_Technology_UACDemoV1.0-00.analog-stereo"
)
NOW_NS = 2_000_000_000_000


def _task6_payload(generated_at_ns: int = NOW_NS - 1_000_000_000) -> dict:
    return {
        "schema_version": "translator.task6-benchmark.v2",
        "generated_at_unix_ns": generated_at_ns,
        "normal_runtime": {
            "selected_asr": "faster-whisper-small",
            "resident_model_id": "small",
        },
        "asr_candidates": [
            {
                "model_id": "faster-whisper-small",
                "cold_inference_ms": 650.0,
                "warm_p95_ms": 90.0,
                "excluded_warmups": 10,
                "measured_count": 100,
            }
        ],
        "duplex_candidates": [
            {
                "model_id": "faster-whisper-small",
                "simultaneous": True,
                "excluded_warmups": 10,
                "measured_per_direction": 100,
                "ru_to_en_latency_ms": [400.0 + index for index in range(100)],
                "en_to_ru_latency_ms": [500.0 + index for index in range(100)],
            },
            {
                "model_id": "faster-whisper-large-v3",
                "simultaneous": True,
                "excluded_warmups": 10,
                "measured_per_direction": 100,
                "ru_to_en_latency_ms": [900.0] * 100,
                "en_to_ru_latency_ms": [1_000.0] * 100,
            },
        ],
    }


def _write_task6(tmp_path: Path, payload: dict | None = None) -> Path:
    path = tmp_path / "task6-results.json"
    path.write_text(
        json.dumps(payload or _task6_payload()),
        encoding="utf-8",
    )
    return path


def test_task6_evidence_selects_fresh_measured_small_model(
    tmp_path: Path,
) -> None:
    evidence = load_task6_provider_evidence(
        _write_task6(tmp_path),
        now_ns=NOW_NS,
        max_age_seconds=60,
    )

    assert evidence.model_id == "faster-whisper-small"
    assert evidence.latency_ms(BenchmarkDirection.RU_TO_EN, 3) == 403.0
    assert evidence.latency_ms(BenchmarkDirection.EN_TO_RU, 3) == 503.0
    assert evidence.measured_per_direction == 100
    assert len(evidence.sha256) == 64


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.update(
            generated_at_unix_ns=NOW_NS - 61_000_000_000
        ),
        lambda payload: payload["normal_runtime"].update(
            selected_asr="faster-whisper-large-v3"
        ),
        lambda payload: payload["duplex_candidates"][0].update(
            measured_per_direction=99
        ),
        lambda payload: payload["duplex_candidates"][0].update(
            simultaneous=False
        ),
    ],
)
def test_task6_evidence_fails_closed_when_not_reusable(
    tmp_path: Path,
    mutation,
) -> None:
    payload = _task6_payload()
    mutation(payload)

    with pytest.raises(LiveBenchmarkError):
        load_task6_provider_evidence(
            _write_task6(tmp_path, payload),
            now_ns=NOW_NS,
            max_age_seconds=60,
        )


def test_marker_correlation_detects_sample_onset_and_rejects_wrong_format() -> None:
    marker = deterministic_marker()
    prefix = np.zeros(257, dtype=np.int16)
    capture = np.concatenate((prefix, marker, np.zeros(64, dtype=np.int16)))

    assert correlate_marker(capture.tobytes(), marker) == 257
    with pytest.raises(LiveBenchmarkError, match="s16le"):
        correlate_marker(capture.tobytes() + b"\x00", marker)
    with pytest.raises(LiveBenchmarkError, match="correlation"):
        correlate_marker(np.zeros(2_000, dtype=np.int16).tobytes(), marker)


def test_marker_correlation_tolerates_duplex_resampling_distortion() -> None:
    marker = deterministic_marker()
    noise = np.random.default_rng(7).normal(0, 2_000, marker.size)
    distorted = np.clip(
        marker.astype(np.float64) + noise,
        np.iinfo(np.int16).min,
        np.iinfo(np.int16).max,
    ).astype(np.int16)
    capture = np.concatenate((np.zeros(193, dtype=np.int16), distorted))

    assert correlate_marker(capture.tobytes(), marker) == 193


def test_transport_measurement_rejects_wrong_format_and_clock_order() -> None:
    with pytest.raises(LiveBenchmarkError, match="format"):
        AudioTransportMeasurement(
            playback_write_ns=1,
            first_audible_ns=2,
            last_audible_ns=3,
            queue_lag_ms=1.0,
            sample_rate_hz=48_000,
        )
    with pytest.raises(LiveBenchmarkError, match="monotonic"):
        AudioTransportMeasurement(
            playback_write_ns=3,
            first_audible_ns=2,
            last_audible_ns=4,
            queue_lag_ms=1.0,
        )


def test_pulse_commands_use_exact_targets_and_pcm_contract() -> None:
    capture = build_capture_command("translator_virtual_mic")
    playback = build_playback_command("translator_mic_out")

    assert capture[:2] == ("parec", "--raw")
    assert "--device=translator_virtual_mic" in capture
    assert playback[:2] == ("pacat", "--raw")
    assert "--playback" in playback
    assert "--device=translator_mic_out" in playback
    for command in (capture, playback):
        assert "--format=s16le" in command
        assert "--rate=16000" in command
        assert "--channels=1" in command
        assert "--property=translator.task7=true" in command


def _pactl_runner(command, **kwargs):
    kind = command[-1]
    if kind == "sinks":
        payload = [
            {
                "index": 10,
                "name": "translator_mic_out",
                "sample_specification": "float32le 1ch 48000Hz",
                "monitor_source": "translator_mic_out.monitor",
                "properties": {"object.serial": "101"},
                "active_port": None,
            },
            {
                "index": 11,
                "name": PHYSICAL_SINK,
                "sample_specification": "s16le 2ch 48000Hz",
                "monitor_source": f"{PHYSICAL_SINK}.monitor",
                "properties": {"object.serial": "102"},
                "active_port": "analog-output",
            },
        ]
    elif kind == "sources":
        payload = [
            {
                "index": 20,
                "name": "translator_virtual_mic",
                "sample_specification": "float32le 1ch 48000Hz",
                "properties": {"object.serial": "201"},
                "active_port": None,
            },
            {
                "index": 21,
                "name": f"{PHYSICAL_SINK}.monitor",
                "sample_specification": "s16le 2ch 48000Hz",
                "properties": {"object.serial": "102"},
                "active_port": None,
            },
        ]
    else:
        payload = []
    return _Completed(json.dumps(payload))


class _Completed:
    def __init__(self, stdout: str) -> None:
        self.stdout = stdout


def test_graph_probe_validates_exact_endpoints_and_detects_device_change() -> None:
    probe = PulseGraphProbe(PHYSICAL_SINK, runner=_pactl_runner)
    baseline = probe.snapshot()

    assert baseline.incoming_capture == f"{PHYSICAL_SINK}.monitor"
    assert baseline.outgoing_playback == "translator_mic_out"
    assert baseline.outgoing_capture == "translator_virtual_mic"
    probe.assert_unchanged(baseline)

    def changed_runner(command, **kwargs):
        result = _pactl_runner(command, **kwargs)
        if command[-1] == "sinks":
            payload = json.loads(result.stdout)
            payload[1]["properties"]["object.serial"] = "replacement"
            return _Completed(json.dumps(payload))
        return result

    changed = PulseGraphProbe(PHYSICAL_SINK, runner=changed_runner)
    with pytest.raises(LiveBenchmarkError, match="device changed"):
        changed.assert_unchanged(baseline)


class _FakeProbe:
    def __init__(self) -> None:
        self.baseline = type(
            "Snapshot",
            (),
            {
                "outgoing_playback": "translator_mic_out",
                "outgoing_capture": "translator_virtual_mic",
                "incoming_playback": PHYSICAL_SINK,
                "incoming_capture": f"{PHYSICAL_SINK}.monitor",
            },
        )()
        self.unchanged_checks = 0
        self.teardown_checks = 0

    def snapshot(self):
        return self.baseline

    def assert_unchanged(self, baseline) -> None:
        assert baseline is self.baseline
        self.unchanged_checks += 1

    def assert_no_task7_streams(self) -> None:
        self.teardown_checks += 1


class _FakeAudio:
    def __init__(self, *, synchronize: bool = False) -> None:
        self.calls: list[tuple[str, str]] = []
        self._lock = Lock()
        self._barrier = Barrier(2) if synchronize else None
        self.process_checks = 0

    def measure(self, *, playback_target, capture_target, marker, timeout_s):
        if self._barrier is not None:
            self._barrier.wait(timeout=1)
        with self._lock:
            self.calls.append((playback_target, capture_target))
        return AudioTransportMeasurement(
            playback_write_ns=1_000_000_000_000,
            first_audible_ns=1_000_020_000_000,
            last_audible_ns=1_000_084_000_000,
            queue_lag_ms=2.0,
        )

    def assert_no_processes(self) -> None:
        self.process_checks += 1


class _FakeSampler:
    def __init__(self) -> None:
        self.ns = 0

    def __call__(self) -> ResourceSample:
        self.ns += 1_000_000
        return ResourceSample(self.ns, 5.0, 1_024, 64)


def test_live_adapter_combines_provider_latency_with_detected_graph_transport(
    tmp_path: Path,
) -> None:
    evidence = load_task6_provider_evidence(
        _write_task6(tmp_path),
        now_ns=NOW_NS,
        max_age_seconds=60,
    )
    probe = _FakeProbe()
    audio = _FakeAudio()
    adapter = LiveBoundaryAdapter(evidence, probe=probe, audio=audio)

    from translator_sidecar.benchmark.task7 import RunContext
    from translator_sidecar.provider_contract import TranslationMode

    observation = adapter.measure_direction(
        RunContext(
            direction=BenchmarkDirection.RU_TO_EN,
            pair_index=3,
            is_warmup=True,
            mode=TranslationMode.QUALITY_FIRST,
        )
    )

    assert observation.provider_latency_ms == 403.0
    assert (
        observation.first_audible_ns - observation.speech_onset_ns
    ) / 1_000_000 == pytest.approx(423.0)
    assert audio.calls == [
        ("translator_mic_out", "translator_virtual_mic")
    ]


def test_live_run_executes_exact_warmup_and_measured_counts_and_tears_down(
    tmp_path: Path,
) -> None:
    evidence = load_task6_provider_evidence(
        _write_task6(tmp_path),
        now_ns=NOW_NS,
        max_age_seconds=60,
    )
    probe = _FakeProbe()
    audio = _FakeAudio(synchronize=True)
    adapter = LiveBoundaryAdapter(evidence, probe=probe, audio=audio)
    checks_before_run = probe.teardown_checks
    report = run_task7_live(
        BenchmarkConfig(profile=ProfileSpec(BenchmarkProfile.WARM)),
        adapter=adapter,
        sample_resources=_FakeSampler(),
    )

    targets = Counter(audio.calls)
    assert targets[("translator_mic_out", "translator_virtual_mic")] == 110
    assert targets[(PHYSICAL_SINK, f"{PHYSICAL_SINK}.monitor")] == 110
    assert report.excluded_warmups == 10
    assert report.measured_count_per_direction == 100
    assert probe.teardown_checks == checks_before_run + 2
    assert audio.process_checks == 1

    payload = build_live_report_payload(report, adapter)
    assert payload["evidence_scope"] == "hybrid_component_estimate"
    assert payload["release_eligible"] is False
    assert payload["release_classification"] is None
    assert payload["component_estimate_classification"] == "meets_target"
    assert "classification" not in payload
    assert payload["resources"]["scope"] == "benchmark_process_and_total_gpu"
    serialized = json.dumps(payload, sort_keys=True)
    assert "PRIVATE_SPEECH_MARKER" not in serialized
    assert deterministic_marker().tobytes().hex() not in serialized
    assert "pcm" not in serialized.lower()
    assert evidence.sha256 in serialized
    assert "deterministic-bpsk-v1" in serialized


def test_live_run_checks_teardown_even_when_measurement_fails(
    tmp_path: Path,
) -> None:
    evidence = load_task6_provider_evidence(
        _write_task6(tmp_path),
        now_ns=NOW_NS,
        max_age_seconds=60,
    )
    probe = _FakeProbe()

    class FailingAudio(_FakeAudio):
        def measure(self, **kwargs):
            raise LiveBenchmarkError("correlation timeout")

    adapter = LiveBoundaryAdapter(
        evidence,
        probe=probe,
        audio=FailingAudio(),
        profile=BenchmarkProfile.COLD,
    )
    checks_before_run = probe.teardown_checks
    with pytest.raises(LiveBenchmarkError, match="correlation timeout"):
        run_task7_live(
            BenchmarkConfig(profile=ProfileSpec(BenchmarkProfile.COLD)),
            adapter=adapter,
            sample_resources=_FakeSampler(),
        )
    assert probe.teardown_checks == checks_before_run + 2


def test_live_run_fails_closed_on_audio_process_leak(tmp_path: Path) -> None:
    evidence = load_task6_provider_evidence(
        _write_task6(tmp_path),
        now_ns=NOW_NS,
        max_age_seconds=60,
    )
    probe = _FakeProbe()

    class LeakingAudio(_FakeAudio):
        def assert_no_processes(self) -> None:
            raise LiveBenchmarkError("audio process leak")

    adapter = LiveBoundaryAdapter(
        evidence,
        probe=probe,
        audio=LeakingAudio(synchronize=True),
    )
    with pytest.raises(LiveBenchmarkError, match="process leak"):
        run_task7_live(
            BenchmarkConfig(profile=ProfileSpec(BenchmarkProfile.WARM)),
            adapter=adapter,
            sample_resources=_FakeSampler(),
        )


def test_resource_sampler_uses_psutil_and_nvidia_smi_and_rejects_clock_rewind(
) -> None:
    class Process:
        def cpu_percent(self, interval=None):
            return 12.5

        def memory_info(self):
            return type("Memory", (), {"rss": 2_048})()

    runner_calls = []

    def runner(command, **kwargs):
        runner_calls.append((command, kwargs))
        return _Completed("512\n")

    clock_values = iter((100, 99))
    sampler = ResourceSampler(
        process=Process(),
        runner=runner,
        clock_ns=lambda: next(clock_values),
    )

    assert sampler().to_dict() == {
        "monotonic_ns": 100,
        "cpu_percent": 12.5,
        "rss_bytes": 2_048,
        "vram_mib": 512,
    }
    assert runner_calls[0][0][0] == "nvidia-smi"
    with pytest.raises(LiveBenchmarkError, match="monotonic"):
        sampler()

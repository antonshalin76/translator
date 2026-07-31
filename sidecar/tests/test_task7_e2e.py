from __future__ import annotations

import json
from pathlib import Path
from threading import Barrier, Event

import pytest

import translator_sidecar.benchmark.task7_e2e as task7_e2e
from translator_sidecar.benchmark.task7 import (
    BenchmarkClassification,
    BenchmarkDirection,
    BenchmarkProfile,
    BoundaryObservation,
    ResourceLimits,
    ResourceSample,
    RunContext,
)
from translator_sidecar.benchmark.task7_e2e import (
    BridgeEvent,
    BridgeEventStream,
    E2EPairAdapter,
    FixtureIdentity,
    LivePipelineMeasurement,
    PidTreeResourceSampler,
    QualityEvidence,
    Task7E2EError,
    build_e2e_report_payload,
    build_pulse_capture_command,
    build_pulse_playback_command,
    build_smoke_report_payload,
    load_task6_quality_evidence,
    parse_arguments,
    run_live_e2e,
    run_smoke_pairs,
)
from translator_sidecar.provider_contract import TranslationMode


def _context(direction: BenchmarkDirection, pair_index: int = 10) -> RunContext:
    return RunContext(
        direction=direction,
        pair_index=pair_index,
        is_warmup=False,
        mode=TranslationMode.QUALITY_FIRST,
    )


def test_bridge_event_stream_parses_privacy_safe_ndjson_and_routes_by_direction() -> (
    None
):
    lines = iter(
        (
            json.dumps(
                {
                    "schema_version": "translator.task7-bridge.v1",
                    "event": "speech_started",
                    "direction": "microphone",
                    "utterance_id": "00000000-0000-0000-0000-000000000001",
                    "monotonic_ns": 100,
                }
            )
            + "\n",
            json.dumps(
                {
                    "schema_version": "translator.task7-bridge.v1",
                    "event": "audio_frame",
                    "direction": "speaker",
                    "utterance_id": "00000000-0000-0000-0000-000000000002",
                    "monotonic_ns": 200,
                    "sequence": 3,
                    "queue_lag_ms": 20,
                }
            )
            + "\n",
        )
    )
    stream = BridgeEventStream(lines)

    microphone = stream.next_for(
        BenchmarkDirection.RU_TO_EN,
        {"speech_started"},
        timeout_s=0.1,
    )
    speaker = stream.next_for(
        BenchmarkDirection.EN_TO_RU,
        {"audio_frame"},
        timeout_s=0.1,
    )

    assert microphone.monotonic_ns == 100
    assert speaker.sequence == 3
    assert speaker.queue_lag_ms == 20
    assert "text" not in microphone.raw
    assert "pcm" not in speaker.raw


@pytest.mark.parametrize(
    "payload",
    [
        {
            "schema_version": "translator.task7-bridge.v1",
            "event": "transcript_final",
            "direction": "microphone",
            "utterance_id": "00000000-0000-0000-0000-000000000001",
            "monotonic_ns": 1,
            "text": "private",
        },
        {
            "schema_version": "translator.task7-bridge.v1",
            "event": "audio_frame",
            "direction": "microphone",
            "utterance_id": "00000000-0000-0000-0000-000000000001",
            "monotonic_ns": 1,
            "pcm": "00",
        },
    ],
)
def test_bridge_event_rejects_content_fields(payload: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="privacy"):
        BridgeEvent.parse(json.dumps(payload))


def test_bridge_event_stream_fails_immediately_on_global_failure() -> None:
    stream = BridgeEventStream(
        (
            json.dumps(
                {
                    "schema_version": "translator.task7-bridge.v1",
                    "event": "failure",
                    "stage": "runtime_start",
                    "code": "runtime_start_failed",
                    "monotonic_ns": 10,
                }
            )
            + "\n",
        )
    )

    with pytest.raises(
        RuntimeError,
        match="runtime_start:runtime_start_failed",
    ):
        stream.next_global({"ready"}, timeout_s=1)


def test_bridge_event_stream_delivers_global_restart_to_both_directions() -> None:
    stream = BridgeEventStream(
        (
            json.dumps(
                {
                    "schema_version": "translator.task7-bridge.v1",
                    "event": "generation_restart",
                    "attempt": 1,
                    "monotonic_ns": 10,
                }
            )
            + "\n",
        )
    )

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=2) as executor:
        events = tuple(
            executor.map(
                lambda direction: stream.next_for(
                    direction,
                    {"generation_restart"},
                    timeout_s=0.1,
                    after_restart_generation=0,
                ),
                BenchmarkDirection,
            )
        )

    assert {event.restart_attempt for event in events} == {1}
    assert {event.monotonic_ns for event in events} == {10}
    assert stream.restart_generation() == 1
    with pytest.raises(Task7E2EError, match="timed out"):
        stream.next_for(
            BenchmarkDirection.RU_TO_EN,
            {"generation_restart"},
            timeout_s=0.1,
            after_restart_generation=1,
        )


def test_live_direction_turns_generation_restart_into_restarted_observation() -> None:
    stream = BridgeEventStream(
        (
            json.dumps(
                {
                    "schema_version": "translator.task7-bridge.v1",
                    "event": "generation_restart",
                    "attempt": 1,
                    "monotonic_ns": 10,
                }
            )
            + "\n",
        )
    )
    measurement = LivePipelineMeasurement(
        events=stream,
        fixtures={direction: () for direction in BenchmarkDirection},
        monitors={direction: object() for direction in BenchmarkDirection},
        timeout_s=0.1,
        profile=BenchmarkProfile.WARM,
        total_pairs=1,
    )

    observation = measurement._collect_direction(
        BenchmarkDirection.RU_TO_EN,
        injection_ns=1,
        generation=0,
        restart_generation=0,
    )

    assert observation.dropped is True
    assert observation.restarted is True
    assert observation.first_audio_ns is None
    assert observation.first_audible_ns is None


def test_bridge_event_stream_timeout_names_direction_and_buffered_events() -> None:
    stream = BridgeEventStream(
        (
            json.dumps(
                {
                    "schema_version": "translator.task7-bridge.v1",
                    "event": "provider_latency",
                    "direction": "microphone",
                    "monotonic_ns": 10,
                }
            )
            + "\n",
        )
    )

    with pytest.raises(
        RuntimeError,
        match=("ru_to_en.*expected=speech_started.*buffered=provider_latency"),
    ):
        stream.next_for(
            BenchmarkDirection.RU_TO_EN,
            {"speech_started"},
            timeout_s=0.1,
        )


def test_bridge_event_parses_only_typed_safe_error_and_terminal_outcome() -> None:
    error = BridgeEvent.parse(
        json.dumps(
            {
                "schema_version": "translator.task7-bridge.v1",
                "event": "provider_error",
                "direction": "microphone",
                "utterance_id": "00000000-0000-0000-0000-000000000001",
                "code": "provider_unavailable",
                "retryable": True,
                "monotonic_ns": 20,
            }
        )
    )
    terminal = BridgeEvent.parse(
        json.dumps(
            {
                "schema_version": "translator.task7-bridge.v1",
                "event": "utterance_terminal",
                "direction": "microphone",
                "utterance_id": "00000000-0000-0000-0000-000000000001",
                "outcome": "dropped",
                "monotonic_ns": 30,
            }
        )
    )

    assert error.error_code == "provider_unavailable"
    assert error.retryable is True
    assert terminal.terminal_outcome == "dropped"


def test_pair_adapter_executes_one_simultaneous_pair_for_two_direction_calls() -> None:
    calls: list[tuple[int, set[BenchmarkDirection]]] = []

    def measure_pair(contexts):
        calls.append((next(iter(contexts.values())).pair_index, set(contexts)))
        return {
            direction: BoundaryObservation(
                speech_onset_ns=1,
                capture_ns=2,
                first_audio_ns=3,
                last_audio_ns=4,
                first_audible_ns=5,
                queue_lag_ms=0,
                provider_latency_ms=1,
            )
            for direction in BenchmarkDirection
        }

    adapter = E2EPairAdapter(measure_pair)
    barrier = Barrier(2)

    def invoke(direction: BenchmarkDirection) -> BoundaryObservation:
        barrier.wait(timeout=1)
        return adapter.measure_direction(_context(direction))

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(
            executor.map(
                invoke,
                (BenchmarkDirection.RU_TO_EN, BenchmarkDirection.EN_TO_RU),
            )
        )

    assert len(results) == 2
    assert calls == [
        (
            10,
            {BenchmarkDirection.RU_TO_EN, BenchmarkDirection.EN_TO_RU},
        )
    ]


def test_smoke_pairs_repeat_real_duplex_measurement_with_sequential_indices() -> None:
    calls: list[dict[BenchmarkDirection, RunContext]] = []

    def measure_pair(contexts):
        calls.append(dict(contexts))
        pair_index = next(iter(contexts.values())).pair_index
        return {
            direction: BoundaryObservation(
                speech_onset_ns=pair_index,
                capture_ns=2,
                first_audio_ns=3,
                last_audio_ns=4,
                first_audible_ns=5,
                queue_lag_ms=0,
                provider_latency_ms=1,
            )
            for direction in BenchmarkDirection
        }

    observations = run_smoke_pairs(measure_pair, pair_count=3)

    assert len(observations) == 3
    assert [
        {context.pair_index for context in contexts.values()} for contexts in calls
    ] == [{0}, {1}, {2}]
    assert [
        pair[BenchmarkDirection.RU_TO_EN].speech_onset_ns for pair in observations
    ] == [0, 1, 2]
    assert all(set(contexts) == set(BenchmarkDirection) for contexts in calls)
    assert all(
        context.is_warmup and context.mode is TranslationMode.QUALITY_FIRST
        for contexts in calls
        for context in contexts.values()
    )


def test_smoke_pairs_reject_invalid_count_and_cli_requires_smoke() -> None:
    with pytest.raises(ValueError, match="positive"):
        run_smoke_pairs(lambda contexts: {}, pair_count=0)

    arguments = parse_arguments(
        (
            "--profile",
            "warm",
            "--physical-sink",
            "sink",
            "--bridge",
            "/tmp/bridge",
            "--output",
            "/tmp/report.json",
            "--smoke-pairs",
            "3",
        )
    )
    with pytest.raises(ValueError, match="requires --smoke"):
        run_live_e2e(arguments)


def test_smoke_report_is_non_release_evidence_and_preserves_pair_count() -> None:
    observation = BoundaryObservation(
        speech_onset_ns=1,
        capture_ns=2,
        first_audio_ns=3,
        last_audio_ns=4,
        first_audible_ns=5,
        queue_lag_ms=0,
        provider_latency_ms=1,
    )

    payload = build_smoke_report_payload(
        (
            {direction: observation for direction in BenchmarkDirection},
            {direction: observation for direction in BenchmarkDirection},
        ),
        resources=(
            ResourceSample(
                monotonic_ns=1,
                cpu_percent=1,
                rss_bytes=2,
                vram_mib=3,
            ),
        ),
        quality_evidence=QualityEvidence(
            sha256="d" * 64,
            corpus_id="task6-v4",
            passes_thresholds=True,
        ),
        graph_before={"owned_streams": 0},
        graph_after={"owned_streams": 0},
    )

    assert payload["schema_version"] == "translator.task7-e2e-smoke.v1"
    assert payload["evidence_scope"] == "production_pipeline_e2e_smoke"
    assert payload["release_eligible"] is False
    assert payload["simultaneous"] is True
    assert payload["pair_count"] == 2
    assert len(payload["pairs"]) == 2


def test_pid_tree_sampler_filters_cpu_rss_and_gpu_to_bridge_tree() -> None:
    class Memory:
        def __init__(self, rss: int) -> None:
            self.rss = rss

    class Process:
        def __init__(self, pid: int, cpu: float, rss: int) -> None:
            self.pid = pid
            self._cpu = cpu
            self._rss = rss

        def children(self, recursive=True):
            assert recursive is True
            return [Process(12, 3.0, 200), Process(13, 4.0, 300)]

        def cpu_percent(self, interval=None):
            assert interval is None
            return self._cpu

        def memory_info(self):
            return Memory(self._rss)

    def process_factory(pid: int):
        assert pid == 11
        return Process(11, 2.0, 100)

    def runner(command, **kwargs):
        assert command[0] == "nvidia-smi"
        return type("Result", (), {"stdout": "11, 512\n13, 256\n99, 4096\n"})()

    sampler = PidTreeResourceSampler(
        11,
        process_factory=process_factory,
        runner=runner,
        clock_ns=lambda: 123,
    )

    assert sampler().to_dict() == {
        "monotonic_ns": 123,
        "cpu_percent": 9.0,
        "rss_bytes": 600,
        "vram_mib": 768,
    }


def test_pid_tree_sampler_keeps_process_handles_for_cpu_baselines() -> None:
    class Process:
        def __init__(self, pid: int) -> None:
            self.pid = pid
            self.calls = 0

        def children(self, recursive=True):
            assert recursive is True
            return []

        def cpu_percent(self, interval=None):
            assert interval is None
            self.calls += 1
            return 0.0 if self.calls == 1 else 12.5

        def memory_info(self):
            return type("Memory", (), {"rss": 100})()

    sampler = PidTreeResourceSampler(
        11,
        process_factory=Process,
        runner=lambda *args, **kwargs: type("Result", (), {"stdout": ""})(),
        clock_ns=lambda: 123,
    )

    assert sampler().cpu_percent == 0.0
    assert sampler().cpu_percent == 12.5


def test_continuous_resource_collector_adds_a_final_synchronous_sample() -> None:
    first_sampled = Event()
    calls = 0

    def sample() -> ResourceSample:
        nonlocal calls
        calls += 1
        first_sampled.set()
        return ResourceSample(
            monotonic_ns=calls,
            cpu_percent=1.0,
            rss_bytes=2,
            vram_mib=3,
        )

    collector = task7_e2e.ContinuousResourceCollector(sample)
    collector.start()
    assert first_sampled.wait(timeout=1)

    samples = collector.stop()

    assert len(samples) >= 2
    assert samples[-1].monotonic_ns > samples[0].monotonic_ns
    assert calls == len(samples)


def test_continuous_resource_collector_surfaces_final_sample_failure() -> None:
    first_sampled = Event()
    calls = 0

    def sample() -> ResourceSample:
        nonlocal calls
        calls += 1
        if calls > 1:
            raise RuntimeError("final sample failed")
        first_sampled.set()
        return ResourceSample(
            monotonic_ns=1,
            cpu_percent=1.0,
            rss_bytes=2,
            vram_mib=3,
        )

    collector = task7_e2e.ContinuousResourceCollector(sample)
    collector.start()
    assert first_sampled.wait(timeout=1)

    with pytest.raises(task7_e2e.Task7E2EError, match="collection failed"):
        collector.stop()


def test_resource_payload_fails_closed_on_sustained_growth() -> None:
    gib = 1024**3
    mib = 1024**2
    samples = tuple(
        ResourceSample(
            monotonic_ns=index * 1_000_000_000,
            cpu_percent=100.0,
            rss_bytes=gib + index * 4 * mib,
            vram_mib=1_800 + index * 4,
        )
        for index in range(100)
    )

    payload = task7_e2e.resource_payload(
        samples,
        limits=ResourceLimits(
            cpu_percent=2_880.0,
            rss_bytes=4 * gib,
            vram_mib=6_144,
        ),
        cpu_p95_limit=2_400.0,
    )

    assert payload["within_limits"] is False
    assert payload["limits"]["vram_mib_peak"] == 6_144
    assert payload["stability"]["rss_growth_bytes"] > 256 * mib
    assert payload["stability"]["vram_growth_mib"] > 256
    assert set(payload["violations"]) == {"rss_growth", "vram_growth"}


@pytest.mark.parametrize("shape", ["plateau", "transient_spike", "noise"])
def test_resource_payload_accepts_stable_post_warmup_windows(
    shape: str,
) -> None:
    gib = 1024**3
    values = []
    for index in range(100):
        rss = gib
        vram = 1_800
        if shape == "transient_spike" and index == 50:
            rss += 200 * 1024**2
            vram += 200
        if shape == "noise":
            rss += ((index % 5) - 2) * 4 * 1024**2
            vram += (index % 5) - 2
        values.append(
            ResourceSample(
                monotonic_ns=index * 1_000_000_000,
                cpu_percent=2_399.0,
                rss_bytes=rss,
                vram_mib=vram,
            )
        )

    payload = task7_e2e.resource_payload(
        tuple(values),
        limits=ResourceLimits(
            cpu_percent=2_880.0,
            rss_bytes=4 * gib,
            vram_mib=6_144,
        ),
        cpu_p95_limit=2_400.0,
    )

    assert payload["within_limits"] is True
    assert payload["violations"] == []


def test_resource_payload_fails_closed_without_stability_evidence() -> None:
    samples = tuple(
        ResourceSample(
            monotonic_ns=index * 1_000_000_000,
            cpu_percent=100.0,
            rss_bytes=1024**3,
            vram_mib=1_800,
        )
        for index in range(10)
    )

    payload = task7_e2e.resource_payload(
        samples,
        limits=ResourceLimits(
            cpu_percent=2_880.0,
            rss_bytes=4 * 1024**3,
            vram_mib=6_144,
        ),
        cpu_p95_limit=2_400.0,
    )

    assert payload["within_limits"] is False
    assert payload["violations"] == ["insufficient_stability_evidence"]


def test_resource_payload_reports_peak_limit_violations() -> None:
    samples = tuple(
        ResourceSample(
            monotonic_ns=index * 1_000_000_000,
            cpu_percent=2_881.0,
            rss_bytes=4 * 1024**3 + 1,
            vram_mib=6_145,
        )
        for index in range(100)
    )

    payload = task7_e2e.resource_payload(
        samples,
        limits=ResourceLimits(
            cpu_percent=2_880.0,
            rss_bytes=4 * 1024**3,
            vram_mib=6_144,
        ),
        cpu_p95_limit=2_400.0,
    )

    assert payload["within_limits"] is False
    assert set(payload["violations"]) == {
        "cpu_peak",
        "cpu_p95",
        "rss_peak",
        "vram_peak",
    }


def test_resource_payload_limit_boundaries_are_inclusive() -> None:
    samples = tuple(
        ResourceSample(
            monotonic_ns=index * 1_000_000_000,
            cpu_percent=2_400.0 if index < 95 else 2_880.0,
            rss_bytes=4 * 1024**3,
            vram_mib=6_144,
        )
        for index in range(100)
    )

    payload = task7_e2e.resource_payload(
        samples,
        limits=ResourceLimits(
            cpu_percent=2_880.0,
            rss_bytes=4 * 1024**3,
            vram_mib=6_144,
        ),
        cpu_p95_limit=2_400.0,
    )

    assert payload["within_limits"] is True


@pytest.mark.parametrize(
    ("last_ns", "expected_within_limits"),
    [
        (1_799_999_999_999, False),
        (1_800_000_000_000, True),
    ],
)
def test_resource_payload_enforces_declared_soak_duration(
    last_ns: int,
    expected_within_limits: bool,
) -> None:
    samples = tuple(
        ResourceSample(
            monotonic_ns=last_ns * index // 99,
            cpu_percent=100.0,
            rss_bytes=1024**3,
            vram_mib=1_800,
        )
        for index in range(100)
    )

    payload = task7_e2e.resource_payload(
        samples,
        limits=ResourceLimits(
            cpu_percent=2_880.0,
            rss_bytes=4 * 1024**3,
            vram_mib=6_144,
        ),
        cpu_p95_limit=2_400.0,
        required_duration_seconds=1_800.0,
    )

    assert payload["within_limits"] is expected_within_limits
    assert ("declared_duration" in payload["violations"]) is (
        not expected_within_limits
    )


def test_graph_snapshot_detects_route_change_with_unchanged_counts() -> None:
    def snapshot(sink: int, serial: str = "500") -> dict[str, object]:
        responses = {
            "sinks": [
                {
                    "index": 10,
                    "name": "physical_sink",
                    "owner_module": 1,
                }
            ],
            "sources": [
                {
                    "index": 20,
                    "name": "physical_source",
                    "owner_module": 2,
                }
            ],
            "sink-inputs": [
                {
                    "index": 30,
                    "sink": sink,
                    "client": 40,
                    "module": None,
                    "properties": {
                        "object.serial": serial,
                        "translator.owner": "translator-daemon",
                        "translator.session_id": "session-1",
                        "application.process.id": "123",
                        "application.process.binary": "peer",
                    },
                }
            ],
            "source-outputs": [],
        }

        def runner(command, **kwargs):
            assert kwargs["check"] is True
            kind = command[-1]
            return type(
                "Result",
                (),
                {"stdout": json.dumps(responses[kind])},
            )()

        return task7_e2e.pulse_graph_summary(runner=runner)

    before = snapshot(10)
    after = snapshot(11)
    replaced = snapshot(10, serial="501")

    assert before["counts"] == after["counts"]
    assert before != after
    assert before != replaced
    assert before["routes"]["sink_inputs"][0]["target"] == 10
    assert after["routes"]["sink_inputs"][0]["target"] == 11
    assert before["routes"]["sink_inputs"][0]["object_serial"] == "500"


def test_graph_snapshot_normalizes_order_and_ignores_unowned_churn() -> None:
    def snapshot(reverse: bool, extra_unowned: bool) -> dict[str, object]:
        sinks = [
            {"index": 11, "name": "translator_remote_in", "owner_module": 2},
            {"index": 10, "name": "translator_mic_out", "owner_module": 1},
        ]
        streams = [
            {
                "index": 31,
                "sink": 11,
                "client": 41,
                "module": None,
                "properties": {
                    "object.serial": "501",
                    "translator.owner": "translator-daemon",
                },
            },
            {
                "index": 30,
                "sink": 10,
                "client": 40,
                "module": None,
                "properties": {
                    "object.serial": "500",
                    "translator.task7_e2e": "true",
                },
            },
        ]
        if extra_unowned:
            streams.append(
                {
                    "index": 99,
                    "sink": 50,
                    "client": 60,
                    "module": None,
                    "properties": {"application.name": "unrelated"},
                }
            )
        if reverse:
            sinks.reverse()
            streams.reverse()
        responses = {
            "sinks": sinks,
            "sources": [],
            "sink-inputs": streams,
            "source-outputs": [],
        }

        def runner(command, **kwargs):
            return type(
                "Result",
                (),
                {"stdout": json.dumps(responses[command[-1]])},
            )()

        return task7_e2e.pulse_graph_summary(runner=runner)

    assert snapshot(False, False) == snapshot(True, True)


def test_graph_snapshot_covers_endpoint_and_source_output_identity() -> None:
    def snapshot(
        *,
        source_target: int,
        source_owner_module: int,
    ) -> dict[str, object]:
        responses = {
            "sinks": [
                {
                    "index": 10,
                    "name": "translator_mic_out",
                    "owner_module": 1,
                }
            ],
            "sources": [
                {
                    "index": 20,
                    "name": "translator_virtual_mic",
                    "owner_module": source_owner_module,
                }
            ],
            "sink-inputs": [],
            "source-outputs": [
                {
                    "index": 31,
                    "source": source_target,
                    "client": 41,
                    "module": 3,
                    "properties": {
                        "object.serial": "501",
                        "translator.owner": "translator-daemon",
                        "translator.session_id": "session-1",
                    },
                }
            ],
        }

        def runner(command, **kwargs):
            return type(
                "Result",
                (),
                {"stdout": json.dumps(responses[command[-1]])},
            )()

        return task7_e2e.pulse_graph_summary(runner=runner)

    before = snapshot(source_target=20, source_owner_module=2)
    moved = snapshot(source_target=21, source_owner_module=2)
    replaced_endpoint = snapshot(source_target=20, source_owner_module=9)

    assert before["counts"] == moved["counts"]
    assert before != moved
    assert before != replaced_endpoint
    assert set(before["routes"]) == {"sink_inputs", "source_outputs"}
    assert before["routes"]["source_outputs"][0] == {
        "client": 41,
        "index": 31,
        "module": 3,
        "object_serial": "501",
        "properties": {
            "translator.owner": "translator-daemon",
            "translator.session_id": "session-1",
        },
        "target": 20,
    }


def test_cold_profile_has_one_excluded_probe_before_warmups() -> None:
    cold = task7_e2e.profile_pair_plan(
        BenchmarkProfile.COLD,
        measured_count=100,
    )
    warm = task7_e2e.profile_pair_plan(
        BenchmarkProfile.WARM,
        measured_count=100,
    )

    assert cold.cold_probe_pairs == 1
    assert cold.excluded_warmups == 10
    assert cold.measured_pairs == 100
    assert cold.total_pairs == 111
    assert cold.phases == (
        ("cold_probe", 1),
        ("warmup", 10),
        ("measured", 100),
    )
    assert warm.cold_probe_pairs == 0
    assert warm.total_pairs == 110
    assert warm.phases == (("warmup", 10), ("measured", 100))


def test_cold_probe_payload_records_both_directions_and_is_excluded() -> None:
    observation = BoundaryObservation(
        speech_onset_ns=1,
        capture_ns=2,
        first_audio_ns=3,
        last_audio_ns=4,
        first_audible_ns=5,
        queue_lag_ms=6,
        provider_latency_ms=7,
    )

    payload = task7_e2e.build_cold_probe_payload(
        {direction: observation for direction in BenchmarkDirection}
    )

    assert payload["excluded_from_percentiles"] is True
    assert set(payload["directions"]) == {"ru_to_en", "en_to_ru"}
    assert (
        payload["directions"]["ru_to_en"]["speech_onset_to_first_audible_ms"]
        == 0.000004
    )


def test_pulse_commands_are_exact_and_content_report_is_privacy_safe(
    tmp_path: Path,
) -> None:
    capture = build_pulse_capture_command("translator_virtual_mic")
    playback = build_pulse_playback_command("translator_remote_in")
    for command in (capture, playback):
        assert "--raw" in command
        assert "--format=s16le" in command
        assert "--rate=16000" in command
        assert "--channels=1" in command
        assert "--property=translator.task7_e2e=true" in command

    fixtures = (
        FixtureIdentity(
            fixture_id="warmup-001",
            direction=BenchmarkDirection.RU_TO_EN,
            pcm_sha256="a" * 64,
            duration_ms=500,
        ),
    )
    payload = build_e2e_report_payload(
        benchmark_payload={
            "schema_version": "translator.task7-benchmark.v1",
            "classification": "meets_target",
            "resources": {"within_limits": True},
        },
        profile="warm",
        bridge_sha256="b" * 64,
        corpus_sha256="c" * 64,
        fixtures=fixtures,
        startup_ready_ms=900.0,
        graph_before={"owned_streams": 0},
        graph_after={"owned_streams": 0},
        quality_evidence=QualityEvidence(
            sha256="d" * 64,
            corpus_id="task6-v4",
            passes_thresholds=True,
        ),
    )
    serialized = json.dumps(payload, sort_keys=True)

    assert payload["schema_version"] == "translator.task7-e2e.v1"
    assert payload["evidence_scope"] == "production_pipeline_e2e"
    assert payload["release_eligible"] is True
    assert payload["resources"]["scope"] == "bridge_process_tree_and_filtered_gpu"
    assert payload["runtime"]["harness_sha256"] == task7_e2e._sha256_file(
        Path(task7_e2e.__file__)
    )
    assert payload["quality_evidence"]["passes_thresholds"] is True
    assert "private" not in serialized.lower()
    assert "pcm" not in serialized.lower()
    assert "transcript" not in serialized.lower()
    assert str(tmp_path) not in serialized


def test_e2e_report_forces_failed_classification_on_resource_failure() -> None:
    payload = build_e2e_report_payload(
        benchmark_payload={
            "schema_version": "translator.task7-benchmark.v1",
            "classification": BenchmarkClassification.MEETS_TARGET.value,
            "profile": {"kind": "warm"},
            "resources": {"within_limits": False},
        },
        profile="warm",
        bridge_sha256="b" * 64,
        corpus_sha256="c" * 64,
        fixtures=(),
        startup_ready_ms=1.0,
        graph_before={"routes": {}},
        graph_after={"routes": {}},
        quality_evidence=QualityEvidence(
            sha256="d" * 64,
            corpus_id="task6-v4",
            passes_thresholds=True,
        ),
    )

    assert payload["classification"] == BenchmarkClassification.FAILS_USABLE_LIMIT.value


def test_e2e_report_resources_cannot_upgrade_failed_latency() -> None:
    payload = build_e2e_report_payload(
        benchmark_payload={
            "schema_version": "translator.task7-benchmark.v1",
            "classification": (BenchmarkClassification.FAILS_USABLE_LIMIT.value),
            "profile": {"kind": "warm"},
            "resources": {"within_limits": True},
        },
        profile="warm",
        bridge_sha256="b" * 64,
        corpus_sha256="c" * 64,
        fixtures=(),
        startup_ready_ms=1.0,
        graph_before={"routes": {}},
        graph_after={"routes": {}},
        quality_evidence=QualityEvidence(
            sha256="d" * 64,
            corpus_id="task6-v4",
            passes_thresholds=True,
        ),
    )

    assert payload["classification"] == BenchmarkClassification.FAILS_USABLE_LIMIT.value


def test_task6_quality_evidence_is_hash_bound_and_fails_closed(
    tmp_path: Path,
) -> None:
    path = tmp_path / "task6.json"
    payload = {
        "schema_version": "translator.task6-benchmark.v2",
        "quality": {
            "quality": {
                "corpus_id": "task6-v4",
                "passes_thresholds": True,
            }
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    evidence = load_task6_quality_evidence(path)

    assert evidence.corpus_id == "task6-v4"
    assert evidence.passes_thresholds is True
    assert len(evidence.sha256) == 64
    payload["quality"]["quality"]["passes_thresholds"] = False
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="quality thresholds"):
        load_task6_quality_evidence(path)

from __future__ import annotations

import json
from threading import Barrier, Lock

import pytest

from translator_sidecar.benchmark.task7 import (
    BenchmarkClassification,
    BenchmarkConfig,
    BenchmarkDirection,
    BenchmarkProfile,
    BoundaryObservation,
    PolicyThresholds,
    ProfileSpec,
    ResourceLimits,
    ResourceSample,
    RunContext,
    percentile_nearest_rank,
    run_task7_benchmark,
)
from translator_sidecar.provider_contract import TranslationMode


def _observation(
    context: RunContext,
    *,
    first_audible_ms: float = 800.0,
    queue_lag_ms: float = 20.0,
    provider_latency_ms: float = 500.0,
    timed_out: bool = False,
    dropped: bool = False,
    restarted: bool = False,
    quality_passed: bool = True,
) -> BoundaryObservation:
    onset_ns = context.pair_index * 2_000_000_000
    capture_ns = onset_ns + 50_000_000
    if timed_out or dropped:
        return BoundaryObservation(
            speech_onset_ns=onset_ns,
            capture_ns=capture_ns,
            first_audio_ns=None,
            last_audio_ns=None,
            first_audible_ns=None,
            queue_lag_ms=queue_lag_ms,
            provider_latency_ms=provider_latency_ms,
            timed_out=timed_out,
            dropped=dropped,
            restarted=restarted,
            quality_passed=quality_passed,
        )
    return BoundaryObservation(
        speech_onset_ns=onset_ns,
        capture_ns=capture_ns,
        first_audio_ns=capture_ns + 400_000_000,
        last_audio_ns=capture_ns + 700_000_000,
        first_audible_ns=onset_ns + round(first_audible_ms * 1_000_000),
        queue_lag_ms=queue_lag_ms,
        provider_latency_ms=provider_latency_ms,
        timed_out=False,
        dropped=False,
        restarted=restarted,
        quality_passed=quality_passed,
    )


class _ResourceClock:
    def __init__(self, step_ns: int = 1_000_000) -> None:
        self._now_ns = 0
        self._step_ns = step_ns

    def sample(self) -> ResourceSample:
        self._now_ns += self._step_ns
        return ResourceSample(
            monotonic_ns=self._now_ns,
            cpu_percent=45.0,
            rss_bytes=600_000_000,
            vram_mib=4_200,
        )


def _config(
    *,
    measured_count: int = 100,
    profile: ProfileSpec | None = None,
) -> BenchmarkConfig:
    return BenchmarkConfig(
        profile=profile or ProfileSpec(BenchmarkProfile.WARM),
        excluded_warmups=10,
        measured_count_per_direction=measured_count,
        resource_limits=ResourceLimits(
            cpu_percent=800.0,
            rss_bytes=2_000_000_000,
            vram_mib=10_240,
        ),
    )


def test_duplex_runs_each_pair_concurrently_and_excludes_ten_warmups() -> None:
    barrier = Barrier(2)
    lock = Lock()
    active = 0
    peak_active = 0
    observed: list[RunContext] = []

    def measure(context: RunContext) -> BoundaryObservation:
        nonlocal active, peak_active
        with lock:
            active += 1
            peak_active = max(peak_active, active)
            observed.append(context)
        barrier.wait(timeout=2)
        with lock:
            active -= 1
        latency = 10_000.0 if context.is_warmup else 800.0
        return _observation(context, first_audible_ms=latency)

    report = run_task7_benchmark(
        _config(),
        measure_direction=measure,
        sample_resources=_ResourceClock().sample,
    )

    assert peak_active == 2
    assert len(observed) == 220
    assert {context.direction for context in observed} == {
        BenchmarkDirection.RU_TO_EN,
        BenchmarkDirection.EN_TO_RU,
    }
    assert all(
        context.mode is TranslationMode.QUALITY_FIRST
        for context in observed
        if context.pair_index == 0
    )
    assert report.simultaneous
    assert report.excluded_warmups == 10
    assert report.measured_count_per_direction == 100
    assert len(report.ru_to_en.samples) == 100
    assert len(report.en_to_ru.samples) == 100
    assert report.ru_to_en.speech_onset_to_first_audible_ms.p95 == 800.0
    assert report.en_to_ru.speech_onset_to_first_audible_ms.p95 == 800.0


@pytest.mark.parametrize("warmups", [0, 9, 11])
def test_exactly_ten_excluded_warmups_are_required(warmups: int) -> None:
    with pytest.raises(ValueError, match="excluded_warmups"):
        BenchmarkConfig(
            profile=ProfileSpec(BenchmarkProfile.WARM),
            excluded_warmups=warmups,
            measured_count_per_direction=100,
        )


@pytest.mark.parametrize("measured_count", [0, 1, 99])
def test_at_least_one_hundred_measured_samples_per_direction_are_required(
    measured_count: int,
) -> None:
    with pytest.raises(ValueError, match="measured_count_per_direction"):
        _config(measured_count=measured_count)


@pytest.mark.parametrize(
    ("profile", "duration_seconds"),
    [
        (BenchmarkProfile.COLD, None),
        (BenchmarkProfile.WARM, None),
        (BenchmarkProfile.SOAK_30_MINUTES, 1_800.0),
    ],
)
def test_cold_warm_and_thirty_minute_profile_schema(
    profile: BenchmarkProfile,
    duration_seconds: float | None,
) -> None:
    spec = ProfileSpec(profile, duration_seconds=duration_seconds)

    assert spec.kind is profile
    assert spec.duration_seconds == duration_seconds


def test_thirty_minute_profile_requires_declared_and_observed_duration() -> None:
    with pytest.raises(ValueError, match="1800"):
        ProfileSpec(BenchmarkProfile.SOAK_30_MINUTES, duration_seconds=1_799.0)

    clock = _ResourceClock(step_ns=1_000_000_000)
    with pytest.raises(ValueError, match="observed duration"):
        run_task7_benchmark(
            _config(
                profile=ProfileSpec(
                    BenchmarkProfile.SOAK_30_MINUTES,
                    duration_seconds=1_800.0,
                )
            ),
            measure_direction=_observation,
            sample_resources=clock.sample,
        )


def test_nearest_rank_percentiles_are_deterministic() -> None:
    values = tuple(float(value) for value in range(1, 101))

    assert percentile_nearest_rank(values, 0.50) == 50.0
    assert percentile_nearest_rank(values, 0.95) == 95.0
    assert percentile_nearest_rank((9.0, 1.0, 5.0), 0.50) == 5.0


def test_latency_metrics_are_derived_from_supplied_boundary_timestamps() -> None:
    report = run_task7_benchmark(
        _config(),
        measure_direction=lambda context: _observation(
            context,
            first_audible_ms=900.0,
            queue_lag_ms=75.0,
            provider_latency_ms=550.0,
        ),
        sample_resources=_ResourceClock().sample,
    )

    for direction in (report.ru_to_en, report.en_to_ru):
        assert direction.speech_onset_to_first_audible_ms.p50 == 900.0
        assert direction.capture_to_first_audio_ms.p95 == 400.0
        assert direction.capture_to_last_audio_ms.p95 == 700.0
        assert direction.queue_lag_ms.p95 == 75.0
        assert direction.provider_latency_ms.p95 == 550.0


def test_quality_first_degrades_to_balanced_then_streaming_on_thresholds() -> None:
    modes: dict[BenchmarkDirection, list[TranslationMode]] = {
        direction: [] for direction in BenchmarkDirection
    }

    def measure(context: RunContext) -> BoundaryObservation:
        modes[context.direction].append(context.mode)
        return _observation(
            context,
            first_audible_ms=1_600.0,
            queue_lag_ms=600.0,
        )

    report = run_task7_benchmark(
        BenchmarkConfig(
            profile=ProfileSpec(BenchmarkProfile.WARM),
            excluded_warmups=10,
            measured_count_per_direction=100,
            policy=PolicyThresholds(
                quality_first_first_audible_ms=1_000.0,
                balanced_first_audible_ms=1_250.0,
                streaming_first_first_audible_ms=1_500.0,
                queue_lag_ms=500.0,
                consecutive_breaches=3,
            ),
        ),
        measure_direction=measure,
        sample_resources=_ResourceClock().sample,
    )

    for direction_modes in modes.values():
        assert direction_modes[:7] == [
            TranslationMode.QUALITY_FIRST,
            TranslationMode.QUALITY_FIRST,
            TranslationMode.QUALITY_FIRST,
            TranslationMode.BALANCED,
            TranslationMode.BALANCED,
            TranslationMode.BALANCED,
            TranslationMode.STREAMING_FIRST,
        ]
    for direction in (report.ru_to_en, report.en_to_ru):
        assert [transition.to_mode for transition in direction.transitions] == [
            TranslationMode.BALANCED,
            TranslationMode.STREAMING_FIRST,
        ]


def test_latency_below_threshold_does_not_change_quality_first_mode() -> None:
    report = run_task7_benchmark(
        _config(),
        measure_direction=lambda context: _observation(
            context,
            first_audible_ms=800.0,
            queue_lag_ms=100.0,
        ),
        sample_resources=_ResourceClock().sample,
    )

    assert report.ru_to_en.transitions == ()
    assert report.en_to_ru.transitions == ()
    assert report.ru_to_en.final_mode is TranslationMode.QUALITY_FIRST
    assert report.en_to_ru.final_mode is TranslationMode.QUALITY_FIRST


@pytest.mark.parametrize(
    ("first_audible_ms", "expected"),
    [
        (1_000.0, BenchmarkClassification.MEETS_TARGET),
        (1_001.0, BenchmarkClassification.USABLE_DEGRADED),
        (1_500.0, BenchmarkClassification.USABLE_DEGRADED),
        (1_501.0, BenchmarkClassification.FAILS_USABLE_LIMIT),
    ],
)
def test_final_classification_uses_graph_boundary_p95(
    first_audible_ms: float,
    expected: BenchmarkClassification,
) -> None:
    report = run_task7_benchmark(
        _config(),
        measure_direction=lambda context: _observation(
            context,
            first_audible_ms=first_audible_ms,
        ),
        sample_resources=_ResourceClock().sample,
    )

    assert report.classification is expected


@pytest.mark.parametrize("failure", ["timeout", "drop"])
def test_timeout_and_drop_rates_are_separate_and_strictly_below_one_percent(
    failure: str,
) -> None:
    def measure(context: RunContext) -> BoundaryObservation:
        failed = not context.is_warmup and context.pair_index == 10
        return _observation(
            context,
            timed_out=failed and failure == "timeout",
            dropped=failed and failure == "drop",
        )

    report = run_task7_benchmark(
        _config(),
        measure_direction=measure,
        sample_resources=_ResourceClock().sample,
    )

    for direction in (report.ru_to_en, report.en_to_ru):
        assert direction.timeout_rate == (
            pytest.approx(0.01) if failure == "timeout" else 0.0
        )
        assert direction.drop_rate == (
            pytest.approx(0.01) if failure == "drop" else 0.0
        )
    assert report.classification is BenchmarkClassification.FAILS_USABLE_LIMIT


def test_failed_samples_do_not_contribute_partial_output_latency() -> None:
    def measure(context: RunContext) -> BoundaryObservation:
        observation = _observation(context, first_audible_ms=800.0)
        if not context.is_warmup and context.pair_index < 20:
            return BoundaryObservation(
                speech_onset_ns=observation.speech_onset_ns,
                capture_ns=observation.capture_ns,
                first_audio_ns=observation.capture_ns + 9_000_000_000,
                last_audio_ns=observation.capture_ns + 10_000_000_000,
                first_audible_ns=observation.speech_onset_ns
                + 10_000_000_000,
                queue_lag_ms=observation.queue_lag_ms,
                provider_latency_ms=observation.provider_latency_ms,
                dropped=True,
            )
        return observation

    report = run_task7_benchmark(
        _config(),
        measure_direction=measure,
        sample_resources=_ResourceClock().sample,
    )

    for direction in (report.ru_to_en, report.en_to_ru):
        assert direction.speech_onset_to_first_audible_ms.p95 == 800.0
        assert direction.capture_to_first_audio_ms.p95 == 400.0
        assert direction.capture_to_last_audio_ms.p95 == 700.0


def test_restart_quality_and_resource_failures_block_usable_classification() -> None:
    def measure(context: RunContext) -> BoundaryObservation:
        return _observation(
            context,
            restarted=not context.is_warmup and context.pair_index == 10,
            quality_passed=not (
                not context.is_warmup and context.pair_index == 11
            ),
        )

    report = run_task7_benchmark(
        _config(),
        measure_direction=measure,
        sample_resources=lambda: ResourceSample(
            monotonic_ns=1,
            cpu_percent=900.0,
            rss_bytes=600_000_000,
            vram_mib=4_200,
        ),
    )

    assert report.ru_to_en.restart_count == 1
    assert report.en_to_ru.restart_count == 1
    assert not report.quality_passed
    assert not report.resources.within_limits
    assert report.classification is BenchmarkClassification.FAILS_USABLE_LIMIT


def test_resource_clock_must_not_move_backwards() -> None:
    timestamps = iter([2, 1, *(3 + value for value in range(218))])

    with pytest.raises(ValueError, match="resource sample clock"):
        run_task7_benchmark(
            _config(),
            measure_direction=_observation,
            sample_resources=lambda: ResourceSample(
                monotonic_ns=next(timestamps),
                cpu_percent=1.0,
                rss_bytes=1,
                vram_mib=1,
            ),
        )


def test_serialization_contains_telemetry_but_no_content_text() -> None:
    private_marker = "never-serialize-spoken-content"

    def measure(context: RunContext) -> BoundaryObservation:
        assert private_marker
        return _observation(context)

    report = run_task7_benchmark(
        _config(),
        measure_direction=measure,
        sample_resources=_ResourceClock().sample,
    )
    payload = report.to_dict()
    serialized = json.dumps(payload, sort_keys=True)

    assert payload["schema_version"] == "translator.task7-benchmark.v1"
    assert len(payload["directions"]["ru_to_en"]["samples"]) == 100
    assert len(payload["resources"]["samples"]) == 220
    assert private_marker not in serialized
    assert not any(
        forbidden in serialized.lower()
        for forbidden in ("transcript", "translation_text", "pcm", "utterance")
    )
    allowed_strings = {
        "translator.task7-benchmark.v1",
        *(value.value for value in BenchmarkClassification),
        *(value.value for value in BenchmarkDirection),
        *(value.value for value in BenchmarkProfile),
        *(value.value for value in TranslationMode),
        "first_audible",
        "queue_lag",
        "timeout",
        "drop",
    }

    def string_values(value: object) -> list[str]:
        if isinstance(value, str):
            return [value]
        if isinstance(value, dict):
            return [
                item
                for nested in value.values()
                for item in string_values(nested)
            ]
        if isinstance(value, list):
            return [
                item for nested in value for item in string_values(nested)
            ]
        return []

    assert set(string_values(payload)) <= allowed_strings

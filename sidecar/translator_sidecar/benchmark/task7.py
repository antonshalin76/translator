"""Deterministic graph-boundary benchmark core for Task 7."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
import math

from translator_sidecar.provider_contract import TranslationMode


_SCHEMA_VERSION = "translator.task7-benchmark.v1"
_REQUIRED_WARMUPS = 10
_MINIMUM_MEASURED_PER_DIRECTION = 100
_THIRTY_MINUTES_SECONDS = 1_800.0


class BenchmarkDirection(str, Enum):
    RU_TO_EN = "ru_to_en"
    EN_TO_RU = "en_to_ru"


class BenchmarkProfile(str, Enum):
    COLD = "cold"
    WARM = "warm"
    SOAK_30_MINUTES = "soak_30_minutes"


class BenchmarkClassification(str, Enum):
    MEETS_TARGET = "meets_target"
    USABLE_DEGRADED = "usable_degraded"
    FAILS_USABLE_LIMIT = "fails_usable_limit"


class _TransitionReason(str, Enum):
    FIRST_AUDIBLE = "first_audible"
    QUEUE_LAG = "queue_lag"
    TIMEOUT = "timeout"
    DROP = "drop"


@dataclass(frozen=True)
class ProfileSpec:
    kind: BenchmarkProfile
    duration_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.duration_seconds is not None:
            _require_finite_nonnegative(
                self.duration_seconds,
                "duration_seconds",
            )
        if (
            self.kind is BenchmarkProfile.SOAK_30_MINUTES
            and (
                self.duration_seconds is None
                or self.duration_seconds < _THIRTY_MINUTES_SECONDS
            )
        ):
            raise ValueError(
                "soak_30_minutes duration_seconds must be at least 1800"
            )


@dataclass(frozen=True)
class ResourceLimits:
    cpu_percent: float | None = None
    rss_bytes: int | None = None
    vram_mib: int | None = None

    def __post_init__(self) -> None:
        if self.cpu_percent is not None:
            _require_finite_nonnegative(self.cpu_percent, "cpu_percent")
        if self.rss_bytes is not None and self.rss_bytes < 0:
            raise ValueError("rss_bytes must be nonnegative")
        if self.vram_mib is not None and self.vram_mib < 0:
            raise ValueError("vram_mib must be nonnegative")


@dataclass(frozen=True)
class PolicyThresholds:
    quality_first_first_audible_ms: float = 1_000.0
    balanced_first_audible_ms: float = 1_250.0
    streaming_first_first_audible_ms: float = 1_500.0
    queue_lag_ms: float = 500.0
    consecutive_breaches: int = 3

    def __post_init__(self) -> None:
        thresholds = (
            self.quality_first_first_audible_ms,
            self.balanced_first_audible_ms,
            self.streaming_first_first_audible_ms,
        )
        for name, value in (
            ("quality_first_first_audible_ms", thresholds[0]),
            ("balanced_first_audible_ms", thresholds[1]),
            ("streaming_first_first_audible_ms", thresholds[2]),
            ("queue_lag_ms", self.queue_lag_ms),
        ):
            _require_finite_nonnegative(value, name)
        if thresholds != tuple(sorted(thresholds)):
            raise ValueError("first-audible thresholds must be nondecreasing")
        if self.consecutive_breaches <= 0:
            raise ValueError("consecutive_breaches must be positive")

    def first_audible_ms(self, mode: TranslationMode) -> float:
        return {
            TranslationMode.QUALITY_FIRST: (
                self.quality_first_first_audible_ms
            ),
            TranslationMode.BALANCED: self.balanced_first_audible_ms,
            TranslationMode.STREAMING_FIRST: (
                self.streaming_first_first_audible_ms
            ),
        }[mode]


@dataclass(frozen=True)
class BenchmarkConfig:
    profile: ProfileSpec
    excluded_warmups: int = _REQUIRED_WARMUPS
    measured_count_per_direction: int = _MINIMUM_MEASURED_PER_DIRECTION
    policy: PolicyThresholds = field(default_factory=PolicyThresholds)
    resource_limits: ResourceLimits = field(default_factory=ResourceLimits)
    target_p95_ms: float = 1_000.0
    usable_limit_p95_ms: float = 1_500.0

    def __post_init__(self) -> None:
        if self.excluded_warmups != _REQUIRED_WARMUPS:
            raise ValueError("excluded_warmups must equal 10")
        if self.measured_count_per_direction < _MINIMUM_MEASURED_PER_DIRECTION:
            raise ValueError(
                "measured_count_per_direction must be at least 100"
            )
        _require_finite_nonnegative(self.target_p95_ms, "target_p95_ms")
        _require_finite_nonnegative(
            self.usable_limit_p95_ms,
            "usable_limit_p95_ms",
        )
        if self.target_p95_ms > self.usable_limit_p95_ms:
            raise ValueError("target_p95_ms must not exceed usable_limit_p95_ms")


@dataclass(frozen=True)
class RunContext:
    direction: BenchmarkDirection
    pair_index: int
    is_warmup: bool
    mode: TranslationMode


@dataclass(frozen=True)
class BoundaryObservation:
    speech_onset_ns: int
    capture_ns: int
    first_audio_ns: int | None
    last_audio_ns: int | None
    first_audible_ns: int | None
    queue_lag_ms: float
    provider_latency_ms: float | None
    timed_out: bool = False
    dropped: bool = False
    restarted: bool = False
    quality_passed: bool = True

    def __post_init__(self) -> None:
        for name, value in (
            ("speech_onset_ns", self.speech_onset_ns),
            ("capture_ns", self.capture_ns),
        ):
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.capture_ns < self.speech_onset_ns:
            raise ValueError("capture_ns must not precede speech_onset_ns")
        _require_finite_nonnegative(self.queue_lag_ms, "queue_lag_ms")
        if self.provider_latency_ms is not None:
            _require_finite_nonnegative(
                self.provider_latency_ms,
                "provider_latency_ms",
            )
        timestamps = (
            self.first_audio_ns,
            self.last_audio_ns,
            self.first_audible_ns,
        )
        if not self.timed_out and not self.dropped and any(
            value is None for value in timestamps
        ):
            raise ValueError(
                "successful observations require all output timestamps"
            )
        if any(value is not None and value < 0 for value in timestamps):
            raise ValueError("output timestamps must be nonnegative")
        if (
            self.first_audio_ns is not None
            and self.first_audio_ns < self.capture_ns
        ):
            raise ValueError("first_audio_ns must not precede capture_ns")
        if (
            self.first_audio_ns is not None
            and self.last_audio_ns is not None
            and self.last_audio_ns < self.first_audio_ns
        ):
            raise ValueError("last_audio_ns must not precede first_audio_ns")
        if (
            self.first_audible_ns is not None
            and self.first_audible_ns < self.speech_onset_ns
        ):
            raise ValueError(
                "first_audible_ns must not precede speech_onset_ns"
            )


@dataclass(frozen=True)
class ResourceSample:
    monotonic_ns: int
    cpu_percent: float
    rss_bytes: int
    vram_mib: int

    def __post_init__(self) -> None:
        if self.monotonic_ns < 0:
            raise ValueError("monotonic_ns must be nonnegative")
        _require_finite_nonnegative(self.cpu_percent, "cpu_percent")
        if self.rss_bytes < 0:
            raise ValueError("rss_bytes must be nonnegative")
        if self.vram_mib < 0:
            raise ValueError("vram_mib must be nonnegative")

    def to_dict(self) -> dict[str, int | float]:
        return {
            "monotonic_ns": self.monotonic_ns,
            "cpu_percent": self.cpu_percent,
            "rss_bytes": self.rss_bytes,
            "vram_mib": self.vram_mib,
        }


@dataclass(frozen=True)
class MetricPercentiles:
    p50: float | None
    p95: float | None

    def to_dict(self) -> dict[str, float | None]:
        return {"p50": self.p50, "p95": self.p95}


@dataclass(frozen=True)
class MeasuredSample:
    pair_index: int
    mode: TranslationMode
    speech_onset_to_first_audible_ms: float | None
    capture_to_first_audio_ms: float | None
    capture_to_last_audio_ms: float | None
    queue_lag_ms: float
    provider_latency_ms: float | None
    timed_out: bool
    dropped: bool
    restarted: bool
    quality_passed: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "pair_index": self.pair_index,
            "mode": self.mode.value,
            "speech_onset_to_first_audible_ms": (
                self.speech_onset_to_first_audible_ms
            ),
            "capture_to_first_audio_ms": self.capture_to_first_audio_ms,
            "capture_to_last_audio_ms": self.capture_to_last_audio_ms,
            "queue_lag_ms": self.queue_lag_ms,
            "provider_latency_ms": self.provider_latency_ms,
            "timed_out": self.timed_out,
            "dropped": self.dropped,
            "restarted": self.restarted,
            "quality_passed": self.quality_passed,
        }


@dataclass(frozen=True)
class ModeTransition:
    pair_index: int
    from_mode: TranslationMode
    to_mode: TranslationMode
    reason: _TransitionReason

    def to_dict(self) -> dict[str, int | str]:
        return {
            "pair_index": self.pair_index,
            "from_mode": self.from_mode.value,
            "to_mode": self.to_mode.value,
            "reason": self.reason.value,
        }


@dataclass(frozen=True)
class DirectionReport:
    direction: BenchmarkDirection
    samples: tuple[MeasuredSample, ...]
    transitions: tuple[ModeTransition, ...]
    final_mode: TranslationMode
    speech_onset_to_first_audible_ms: MetricPercentiles
    capture_to_first_audio_ms: MetricPercentiles
    capture_to_last_audio_ms: MetricPercentiles
    queue_lag_ms: MetricPercentiles
    provider_latency_ms: MetricPercentiles
    timeout_count: int
    timeout_rate: float
    drop_count: int
    drop_rate: float
    restart_count: int
    quality_passed: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "direction": self.direction.value,
            "samples": [sample.to_dict() for sample in self.samples],
            "transitions": [
                transition.to_dict() for transition in self.transitions
            ],
            "final_mode": self.final_mode.value,
            "speech_onset_to_first_audible_ms": (
                self.speech_onset_to_first_audible_ms.to_dict()
            ),
            "capture_to_first_audio_ms": (
                self.capture_to_first_audio_ms.to_dict()
            ),
            "capture_to_last_audio_ms": (
                self.capture_to_last_audio_ms.to_dict()
            ),
            "queue_lag_ms": self.queue_lag_ms.to_dict(),
            "provider_latency_ms": self.provider_latency_ms.to_dict(),
            "timeout_count": self.timeout_count,
            "timeout_rate": self.timeout_rate,
            "drop_count": self.drop_count,
            "drop_rate": self.drop_rate,
            "restart_count": self.restart_count,
            "quality_passed": self.quality_passed,
        }


@dataclass(frozen=True)
class ResourceReport:
    samples: tuple[ResourceSample, ...]
    observed_duration_seconds: float
    cpu_percent_peak: float
    rss_bytes_peak: int
    vram_mib_peak: int
    within_limits: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "samples": [sample.to_dict() for sample in self.samples],
            "observed_duration_seconds": self.observed_duration_seconds,
            "cpu_percent_peak": self.cpu_percent_peak,
            "rss_bytes_peak": self.rss_bytes_peak,
            "vram_mib_peak": self.vram_mib_peak,
            "within_limits": self.within_limits,
        }


@dataclass(frozen=True)
class BenchmarkReport:
    profile: ProfileSpec
    simultaneous: bool
    excluded_warmups: int
    measured_count_per_direction: int
    ru_to_en: DirectionReport
    en_to_ru: DirectionReport
    resources: ResourceReport
    quality_passed: bool
    classification: BenchmarkClassification

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "profile": {
                "kind": self.profile.kind.value,
                "duration_seconds": self.profile.duration_seconds,
            },
            "simultaneous": self.simultaneous,
            "excluded_warmups": self.excluded_warmups,
            "measured_count_per_direction": (
                self.measured_count_per_direction
            ),
            "directions": {
                BenchmarkDirection.RU_TO_EN.value: self.ru_to_en.to_dict(),
                BenchmarkDirection.EN_TO_RU.value: self.en_to_ru.to_dict(),
            },
            "resources": self.resources.to_dict(),
            "quality_passed": self.quality_passed,
            "classification": self.classification.value,
        }


class _ModePolicy:
    def __init__(self, thresholds: PolicyThresholds) -> None:
        self.mode = TranslationMode.QUALITY_FIRST
        self._thresholds = thresholds
        self._consecutive_breaches = 0
        self.transitions: list[ModeTransition] = []

    def observe(
        self,
        pair_index: int,
        observation: BoundaryObservation,
    ) -> None:
        reason = self._breach_reason(observation)
        if reason is None:
            self._consecutive_breaches = 0
            return
        self._consecutive_breaches += 1
        if (
            self._consecutive_breaches
            < self._thresholds.consecutive_breaches
        ):
            return
        next_mode = {
            TranslationMode.QUALITY_FIRST: TranslationMode.BALANCED,
            TranslationMode.BALANCED: TranslationMode.STREAMING_FIRST,
            TranslationMode.STREAMING_FIRST: None,
        }[self.mode]
        self._consecutive_breaches = 0
        if next_mode is None:
            return
        previous = self.mode
        self.mode = next_mode
        self.transitions.append(
            ModeTransition(
                pair_index=pair_index,
                from_mode=previous,
                to_mode=next_mode,
                reason=reason,
            )
        )

    def _breach_reason(
        self,
        observation: BoundaryObservation,
    ) -> _TransitionReason | None:
        if observation.timed_out:
            return _TransitionReason.TIMEOUT
        if observation.dropped:
            return _TransitionReason.DROP
        audible_ms = _duration_ms(
            observation.speech_onset_ns,
            observation.first_audible_ns,
        )
        if (
            audible_ms is not None
            and audible_ms > self._thresholds.first_audible_ms(self.mode)
        ):
            return _TransitionReason.FIRST_AUDIBLE
        if observation.queue_lag_ms > self._thresholds.queue_lag_ms:
            return _TransitionReason.QUEUE_LAG
        return None


def percentile_nearest_rank(
    values: Sequence[float],
    percentile: float,
) -> float:
    if not values:
        raise ValueError("percentile values must not be empty")
    if not math.isfinite(percentile) or not 0 < percentile <= 1:
        raise ValueError("percentile must be in the interval (0, 1]")
    checked = tuple(float(value) for value in values)
    if any(not math.isfinite(value) for value in checked):
        raise ValueError("percentile values must be finite")
    ordered = sorted(checked)
    return ordered[math.ceil(percentile * len(ordered)) - 1]


def run_task7_benchmark(
    config: BenchmarkConfig,
    *,
    measure_direction: Callable[[RunContext], BoundaryObservation],
    sample_resources: Callable[[], ResourceSample],
) -> BenchmarkReport:
    policies = {
        direction: _ModePolicy(config.policy)
        for direction in BenchmarkDirection
    }
    measured: dict[BenchmarkDirection, list[MeasuredSample]] = {
        direction: [] for direction in BenchmarkDirection
    }
    resources: list[ResourceSample] = []
    total_pairs = (
        config.excluded_warmups + config.measured_count_per_direction
    )

    with ThreadPoolExecutor(
        max_workers=2,
        thread_name_prefix="translator-task7",
    ) as executor:
        for pair_index in range(total_pairs):
            is_warmup = pair_index < config.excluded_warmups
            contexts = {
                direction: RunContext(
                    direction=direction,
                    pair_index=pair_index,
                    is_warmup=is_warmup,
                    mode=policies[direction].mode,
                )
                for direction in BenchmarkDirection
            }
            resources.append(sample_resources())
            futures = {
                direction: executor.submit(
                    measure_direction,
                    context,
                )
                for direction, context in contexts.items()
            }
            observations = {
                direction: future.result()
                for direction, future in futures.items()
            }
            resources.append(sample_resources())
            for direction, observation in observations.items():
                context = contexts[direction]
                policies[direction].observe(pair_index, observation)
                if not is_warmup:
                    measured[direction].append(
                        _measured_sample(context, observation)
                    )

    resource_report = _resource_report(
        tuple(resources),
        config.resource_limits,
    )
    if (
        config.profile.kind is BenchmarkProfile.SOAK_30_MINUTES
        and resource_report.observed_duration_seconds
        < (config.profile.duration_seconds or _THIRTY_MINUTES_SECONDS)
    ):
        raise ValueError(
            "soak_30_minutes observed duration is shorter than declared"
        )

    direction_reports = {
        direction: _direction_report(
            direction,
            tuple(measured[direction]),
            policies[direction],
        )
        for direction in BenchmarkDirection
    }
    quality_passed = all(
        report.quality_passed for report in direction_reports.values()
    )
    classification = _classify(
        config,
        direction_reports,
        resource_report,
        quality_passed,
    )
    return BenchmarkReport(
        profile=config.profile,
        simultaneous=True,
        excluded_warmups=config.excluded_warmups,
        measured_count_per_direction=config.measured_count_per_direction,
        ru_to_en=direction_reports[BenchmarkDirection.RU_TO_EN],
        en_to_ru=direction_reports[BenchmarkDirection.EN_TO_RU],
        resources=resource_report,
        quality_passed=quality_passed,
        classification=classification,
    )


def _measured_sample(
    context: RunContext,
    observation: BoundaryObservation,
) -> MeasuredSample:
    completed = not observation.timed_out and not observation.dropped
    return MeasuredSample(
        pair_index=context.pair_index,
        mode=context.mode,
        speech_onset_to_first_audible_ms=(
            _duration_ms(
                observation.speech_onset_ns,
                observation.first_audible_ns,
            )
            if completed
            else None
        ),
        capture_to_first_audio_ms=(
            _duration_ms(
                observation.capture_ns,
                observation.first_audio_ns,
            )
            if completed
            else None
        ),
        capture_to_last_audio_ms=(
            _duration_ms(
                observation.capture_ns,
                observation.last_audio_ns,
            )
            if completed
            else None
        ),
        queue_lag_ms=observation.queue_lag_ms,
        provider_latency_ms=observation.provider_latency_ms,
        timed_out=observation.timed_out,
        dropped=observation.dropped,
        restarted=observation.restarted,
        quality_passed=observation.quality_passed,
    )


def _direction_report(
    direction: BenchmarkDirection,
    samples: tuple[MeasuredSample, ...],
    policy: _ModePolicy,
) -> DirectionReport:
    count = len(samples)
    timeout_count = sum(sample.timed_out for sample in samples)
    drop_count = sum(sample.dropped for sample in samples)
    return DirectionReport(
        direction=direction,
        samples=samples,
        transitions=tuple(policy.transitions),
        final_mode=policy.mode,
        speech_onset_to_first_audible_ms=_percentiles(
            sample.speech_onset_to_first_audible_ms for sample in samples
        ),
        capture_to_first_audio_ms=_percentiles(
            sample.capture_to_first_audio_ms for sample in samples
        ),
        capture_to_last_audio_ms=_percentiles(
            sample.capture_to_last_audio_ms for sample in samples
        ),
        queue_lag_ms=_percentiles(
            sample.queue_lag_ms for sample in samples
        ),
        provider_latency_ms=_percentiles(
            sample.provider_latency_ms for sample in samples
        ),
        timeout_count=timeout_count,
        timeout_rate=timeout_count / count,
        drop_count=drop_count,
        drop_rate=drop_count / count,
        restart_count=sum(sample.restarted for sample in samples),
        quality_passed=all(sample.quality_passed for sample in samples),
    )


def _percentiles(values: Iterable[float | None]) -> MetricPercentiles:
    measured = tuple(float(value) for value in values if value is not None)
    if not measured:
        return MetricPercentiles(p50=None, p95=None)
    return MetricPercentiles(
        p50=percentile_nearest_rank(measured, 0.50),
        p95=percentile_nearest_rank(measured, 0.95),
    )


def _resource_report(
    samples: tuple[ResourceSample, ...],
    limits: ResourceLimits,
) -> ResourceReport:
    if not samples:
        raise ValueError("resource samples must not be empty")
    if any(
        current.monotonic_ns < previous.monotonic_ns
        for previous, current in zip(samples, samples[1:], strict=False)
    ):
        raise ValueError("resource sample clock must not move backwards")
    first_ns = min(sample.monotonic_ns for sample in samples)
    last_ns = max(sample.monotonic_ns for sample in samples)
    cpu_peak = max(sample.cpu_percent for sample in samples)
    rss_peak = max(sample.rss_bytes for sample in samples)
    vram_peak = max(sample.vram_mib for sample in samples)
    within_limits = (
        (limits.cpu_percent is None or cpu_peak <= limits.cpu_percent)
        and (limits.rss_bytes is None or rss_peak <= limits.rss_bytes)
        and (limits.vram_mib is None or vram_peak <= limits.vram_mib)
    )
    return ResourceReport(
        samples=samples,
        observed_duration_seconds=(last_ns - first_ns) / 1_000_000_000,
        cpu_percent_peak=cpu_peak,
        rss_bytes_peak=rss_peak,
        vram_mib_peak=vram_peak,
        within_limits=within_limits,
    )


def _classify(
    config: BenchmarkConfig,
    directions: dict[BenchmarkDirection, DirectionReport],
    resources: ResourceReport,
    quality_passed: bool,
) -> BenchmarkClassification:
    stable = resources.within_limits and quality_passed
    p95_values: list[float] = []
    for report in directions.values():
        stable = (
            stable
            and report.timeout_rate < 0.01
            and report.drop_rate < 0.01
            and report.restart_count == 0
        )
        p95 = report.speech_onset_to_first_audible_ms.p95
        if p95 is None:
            stable = False
        else:
            p95_values.append(p95)
    if not stable or len(p95_values) != len(directions):
        return BenchmarkClassification.FAILS_USABLE_LIMIT
    worst_p95 = max(p95_values)
    if worst_p95 <= config.target_p95_ms:
        return BenchmarkClassification.MEETS_TARGET
    if worst_p95 <= config.usable_limit_p95_ms:
        return BenchmarkClassification.USABLE_DEGRADED
    return BenchmarkClassification.FAILS_USABLE_LIMIT


def _duration_ms(start_ns: int, end_ns: int | None) -> float | None:
    if end_ns is None:
        return None
    return (end_ns - start_ns) / 1_000_000


def _require_finite_nonnegative(value: float, name: str) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")

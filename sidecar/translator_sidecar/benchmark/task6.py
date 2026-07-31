"""Versioned Task 6 quality corpus and metric evaluation."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from threading import Event, Thread
from typing import Mapping, Protocol, Sequence
from uuid import UUID, uuid4

from jiwer import wer
import regex
from sacrebleu.metrics import CHRF

from translator_sidecar.provider_contract import Language, TranslationMode


_SCHEMA_VERSION = "translator.quality-corpus.v4"
_REQUIRED_SCENARIOS = frozenset({"short", "long", "duplex_overlap"})
_REQUIRED_CRITICAL = frozenset({"negation", "number", "name"})
_CHRF2_FLOOR = 45.0
_WER_CEILING = 0.15
_RESOURCE_SAMPLE_INTERVAL_SECONDS = 0.02
_COUNT_WORDS = {
    Language.EN: {
        0: {"zero"},
        1: {"one"},
        2: {"two"},
        3: {"three"},
        4: {"four"},
        5: {"five"},
        6: {"six"},
        7: {"seven"},
        8: {"eight"},
        9: {"nine"},
        10: {"ten"},
        11: {"eleven"},
        12: {"twelve"},
    },
    Language.RU: {
        0: {"ноль"},
        1: {"один", "одна"},
        2: {"два", "две", "двое"},
        3: {"три"},
        4: {"четыре"},
        5: {"пять"},
        6: {"шесть", "шестеро"},
        7: {"семь"},
        8: {"восемь"},
        9: {"девять"},
        10: {"десять"},
        11: {"одиннадцать"},
        12: {"двенадцать"},
    },
}


class CorpusError(ValueError):
    """The versioned quality corpus or measured results are invalid."""


@dataclass(frozen=True)
class WarmupCase:
    ru: str
    en: str


@dataclass(frozen=True)
class QualityCase:
    case_id: str
    value: str
    ru: str
    en: str
    critical: tuple[str, ...]
    scenarios: tuple[str, ...]
    number_semantics: str | None
    number_role_ru_anchors: tuple[str, ...]
    number_role_en_anchors: tuple[str, ...]
    accepted_ru_names: tuple[str, ...]
    accepted_en_names: tuple[str, ...]
    rejected_ru_names: tuple[str, ...]
    rejected_en_names: tuple[str, ...]
    allowed_ru_name_initials: tuple[str, ...]
    allowed_en_name_initials: tuple[str, ...]
    negation_ru_anchors: tuple[str, ...]
    negation_en_anchors: tuple[str, ...]


@dataclass(frozen=True)
class QualityCorpus:
    schema_version: str
    corpus_id: str
    warmups: tuple[WarmupCase, ...]
    cases: tuple[QualityCase, ...]


@dataclass(frozen=True)
class CriticalViolation:
    case_id: str
    kind: str


@dataclass(frozen=True)
class DirectionQuality:
    chrf2: float
    synthesized_wer: float
    critical_violations: tuple[CriticalViolation, ...]

    @property
    def passes_thresholds(self) -> bool:
        return passes_quality_thresholds(
            chrf2=self.chrf2,
            synthesized_wer=self.synthesized_wer,
            critical_violation_count=len(self.critical_violations),
        )


@dataclass(frozen=True)
class QualityReport:
    corpus_id: str
    excluded_warmups: int
    measured_per_direction: int
    ru_to_en: DirectionQuality
    en_to_ru: DirectionQuality

    @property
    def passes_thresholds(self) -> bool:
        return self.ru_to_en.passes_thresholds and self.en_to_ru.passes_thresholds


@dataclass(frozen=True)
class DirectionRun:
    success_count: int
    drop_count: int
    drop_rate: float
    success_latency_ms: tuple[float, ...]

    @property
    def passes_drop_threshold(self) -> bool:
        return self.drop_rate < 0.01


@dataclass(frozen=True)
class QualityBenchmarkRun:
    quality: QualityReport
    critical_review_content_sha256: str
    excluded_warmups: int
    measured_per_direction: int
    ru_to_en: DirectionRun
    en_to_ru: DirectionRun

    @property
    def passes_thresholds(self) -> bool:
        return (
            self.quality.passes_thresholds
            and self.ru_to_en.passes_drop_threshold
            and self.en_to_ru.passes_drop_threshold
        )


@dataclass(frozen=True)
class AsrBenchmarkConfig:
    model_id: str
    audio_duration_ms: int
    warmup_count: int = 10
    measured_count: int = 100


@dataclass(frozen=True)
class AsrBenchmarkReport:
    model_id: str
    excluded_warmups: int
    measured_count: int
    cold_inference_ms: float
    warm_p95_ms: float
    audio_throughput_x: float
    cpu_percent_peak: float
    rss_bytes_peak: int
    gpu_percent_peak: float
    vram_mib_peak: int


@dataclass(frozen=True)
class DuplexBenchmarkConfig:
    model_id: str
    warmup_count: int = 10
    measured_count_per_direction: int = 100


@dataclass(frozen=True)
class DuplexBenchmarkReport:
    model_id: str
    simultaneous: bool
    excluded_warmups: int
    measured_per_direction: int
    ru_to_en_latency_ms: tuple[float, ...]
    en_to_ru_latency_ms: tuple[float, ...]
    cpu_percent_peak: float
    rss_bytes_peak: int
    gpu_percent_peak: float
    vram_mib_peak: int

    @property
    def vram_within_budget(self) -> bool:
        return within_vram_budget(self.vram_mib_peak)


class TranslatorAdapter(Protocol):
    def translate(
        self,
        text: str,
        *,
        source_language: Language,
        target_language: Language,
        mode: TranslationMode,
    ) -> str: ...


class AsrAdapter(Protocol):
    def transcribe(
        self,
        pcm: bytes,
        *,
        language: Language,
        mode: TranslationMode,
    ) -> str: ...


ResourceSample = tuple[float, int, float, int]


class _PeriodicResourceSampler:
    def __init__(
        self,
        sample: Callable[[], ResourceSample],
    ) -> None:
        self._sample = sample
        self._stop = Event()
        self._samples: list[ResourceSample] = []
        self._error: Exception | None = None
        self._thread = Thread(
            target=self._run,
            name="translator-resource-sampler",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> tuple[ResourceSample, ...]:
        self._stop.set()
        self._thread.join()
        if self._error is not None:
            raise self._error
        return tuple(self._samples)

    def _run(self) -> None:
        while not self._stop.wait(_RESOURCE_SAMPLE_INTERVAL_SECONDS):
            try:
                self._samples.append(self._sample())
            except Exception as error:
                self._error = error
                self._stop.set()
                return


def passes_quality_thresholds(
    *,
    chrf2: float,
    synthesized_wer: float,
    critical_violation_count: int,
) -> bool:
    return (
        chrf2 >= _CHRF2_FLOOR
        and synthesized_wer <= _WER_CEILING
        and critical_violation_count == 0
    )


def within_vram_budget(vram_mib: int) -> bool:
    return vram_mib <= 10_240


def load_quality_corpus(path: Path) -> QualityCorpus:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CorpusError("quality corpus is unreadable") from error
    if not isinstance(payload, dict):
        raise CorpusError("quality corpus root is invalid")
    if payload.get("schema_version") != _SCHEMA_VERSION:
        raise CorpusError("quality corpus schema is invalid")
    corpus_id = _required_text(payload, "corpus_id")
    name_aliases = _required_name_aliases(payload)

    warmup_payload = payload.get("warmups")
    if not isinstance(warmup_payload, list) or len(warmup_payload) != 10:
        raise CorpusError("quality corpus must contain ten warmups")
    warmups = tuple(
        WarmupCase(
            ru=_required_text(value, "ru"),
            en=_required_text(value, "en"),
        )
        for value in warmup_payload
        if isinstance(value, dict)
    )
    if len(warmups) != 10:
        raise CorpusError("quality corpus warmup is invalid")

    template_payload = payload.get("templates")
    if not isinstance(template_payload, list):
        raise CorpusError("quality corpus templates are invalid")
    cases: list[QualityCase] = []
    for template in template_payload:
        if not isinstance(template, dict):
            raise CorpusError("quality corpus template is invalid")
        template_id = _required_text(template, "id")
        ru_template = _required_template(template, "ru")
        en_template = _required_template(template, "en")
        critical = _required_labels(template, "critical")
        scenarios = _required_labels(template, "scenarios")
        number_semantics = template.get("number_semantics")
        if "number" in critical:
            if number_semantics not in {"identifier", "count", "time"}:
                raise CorpusError("quality corpus number semantics are invalid")
        elif number_semantics is not None:
            raise CorpusError("quality corpus number semantics are invalid")
        negation_anchors = _required_negation_anchors(
            template,
            required="negation" in critical,
        )
        number_role_anchors = _required_language_anchors(
            template,
            key="number_role_anchors",
            required=number_semantics == "identifier",
        )
        name_sentence_initials = _required_language_anchors(
            template,
            key="name_sentence_initials",
            required="name" in critical,
        )
        values = template.get("values")
        if not isinstance(values, list) or not values:
            raise CorpusError("quality corpus values are invalid")
        for value in values:
            if not isinstance(value, str) or not value.strip():
                raise CorpusError("quality corpus value is invalid")
            if "name" in critical and value not in name_aliases:
                raise CorpusError("quality corpus name alias is missing")
            cases.append(
                QualityCase(
                    case_id=f"{template_id}:{value}",
                    value=value,
                    ru=ru_template.format(value=value),
                    en=en_template.format(value=value),
                    critical=critical,
                    scenarios=scenarios,
                    number_semantics=number_semantics,
                    number_role_ru_anchors=number_role_anchors[Language.RU],
                    number_role_en_anchors=number_role_anchors[Language.EN],
                    accepted_ru_names=(
                        name_aliases[value][Language.RU] if "name" in critical else ()
                    ),
                    accepted_en_names=(
                        name_aliases[value][Language.EN] if "name" in critical else ()
                    ),
                    rejected_ru_names=(
                        tuple(
                            alias
                            for owner, languages in name_aliases.items()
                            if owner != value
                            for alias in languages[Language.RU]
                        )
                        if "name" in critical
                        else ()
                    ),
                    rejected_en_names=(
                        tuple(
                            alias
                            for owner, languages in name_aliases.items()
                            if owner != value
                            for alias in languages[Language.EN]
                        )
                        if "name" in critical
                        else ()
                    ),
                    allowed_ru_name_initials=name_sentence_initials[Language.RU],
                    allowed_en_name_initials=name_sentence_initials[Language.EN],
                    negation_ru_anchors=negation_anchors[Language.RU],
                    negation_en_anchors=negation_anchors[Language.EN],
                )
            )

    if len(cases) < 100:
        raise CorpusError("quality corpus has fewer than 100 cases")
    if len({case.case_id for case in cases}) != len(cases):
        raise CorpusError("quality corpus case identifiers are not unique")
    scenario_coverage = {scenario for case in cases for scenario in case.scenarios}
    critical_coverage = {label for case in cases for label in case.critical}
    name_values = {case.value for case in cases if "name" in case.critical}
    if set(name_aliases) != name_values:
        raise CorpusError("quality corpus name alias coverage is invalid")
    if not _REQUIRED_SCENARIOS <= scenario_coverage:
        raise CorpusError("quality corpus scenario coverage is incomplete")
    if not _REQUIRED_CRITICAL <= critical_coverage:
        raise CorpusError("quality corpus critical coverage is incomplete")
    return QualityCorpus(
        schema_version=_SCHEMA_VERSION,
        corpus_id=corpus_id,
        warmups=warmups,
        cases=tuple(cases),
    )


def evaluate_quality(
    corpus: QualityCorpus,
    *,
    outputs: Mapping[Language, Sequence[str]],
    synthesized_transcripts: Mapping[Language, Sequence[str]],
) -> QualityReport:
    expected_count = len(corpus.cases)
    _validate_measurements(outputs, expected_count)
    _validate_measurements(synthesized_transcripts, expected_count)
    ru_to_en = _evaluate_direction(
        corpus.cases,
        target_language=Language.EN,
        outputs=outputs[Language.EN],
        synthesized_transcripts=synthesized_transcripts[Language.EN],
    )
    en_to_ru = _evaluate_direction(
        corpus.cases,
        target_language=Language.RU,
        outputs=outputs[Language.RU],
        synthesized_transcripts=synthesized_transcripts[Language.RU],
    )
    return QualityReport(
        corpus_id=corpus.corpus_id,
        excluded_warmups=len(corpus.warmups),
        measured_per_direction=expected_count,
        ru_to_en=ru_to_en,
        en_to_ru=en_to_ru,
    )


def run_quality_benchmark(
    corpus: QualityCorpus,
    *,
    translator: TranslatorAdapter,
    synthesize_and_transcribe: Callable[[str, Language], str],
    now_ns: Callable[[], int],
) -> QualityBenchmarkRun:
    outputs: dict[Language, list[str]] = {
        Language.RU: [],
        Language.EN: [],
    }
    transcripts: dict[Language, list[str]] = {
        Language.RU: [],
        Language.EN: [],
    }
    direction_runs: dict[Language, DirectionRun] = {}

    for source_language, target_language in (
        (Language.RU, Language.EN),
        (Language.EN, Language.RU),
    ):
        for warmup in corpus.warmups:
            source = warmup.ru if source_language is Language.RU else warmup.en
            translator.translate(
                source,
                source_language=source_language,
                target_language=target_language,
                mode=TranslationMode.QUALITY_FIRST,
            )

        successful_latencies: list[float] = []
        drop_count = 0
        for case in corpus.cases:
            source = case.ru if source_language is Language.RU else case.en
            started_ns = now_ns()
            translated = ""
            transcript = ""
            try:
                translated = translator.translate(
                    source,
                    source_language=source_language,
                    target_language=target_language,
                    mode=TranslationMode.QUALITY_FIRST,
                )
                transcript = synthesize_and_transcribe(
                    translated,
                    target_language,
                )
            except Exception:
                drop_count += 1
            else:
                successful_latencies.append((now_ns() - started_ns) / 1_000_000)
            outputs[target_language].append(translated)
            transcripts[target_language].append(transcript)

        measured_count = len(corpus.cases)
        success_count = measured_count - drop_count
        direction_runs[target_language] = DirectionRun(
            success_count=success_count,
            drop_count=drop_count,
            drop_rate=drop_count / measured_count,
            success_latency_ms=tuple(successful_latencies),
        )

    quality = evaluate_quality(
        corpus,
        outputs=outputs,
        synthesized_transcripts=transcripts,
    )
    return QualityBenchmarkRun(
        quality=quality,
        critical_review_content_sha256=critical_review_content_sha256(
            corpus,
            outputs=outputs,
        ),
        excluded_warmups=len(corpus.warmups),
        measured_per_direction=len(corpus.cases),
        ru_to_en=direction_runs[Language.EN],
        en_to_ru=direction_runs[Language.RU],
    )


def critical_review_content_sha256(
    corpus: QualityCorpus,
    *,
    outputs: Mapping[Language, Sequence[str]],
) -> str:
    _validate_measurements(outputs, len(corpus.cases))
    rows = []
    for source_language, target_language in (
        (Language.RU, Language.EN),
        (Language.EN, Language.RU),
    ):
        for case, output in zip(
            corpus.cases,
            outputs[target_language],
            strict=True,
        ):
            rows.append(
                {
                    "case_id": case.case_id,
                    "critical": list(case.critical),
                    "direction": (
                        f"{source_language.value}_to_{target_language.value}"
                    ),
                    "output": output,
                    "reference": (
                        case.en if target_language is Language.EN else case.ru
                    ),
                    "source": (
                        case.ru if source_language is Language.RU else case.en
                    ),
                }
            )
    canonical = json.dumps(
        {"corpus_id": corpus.corpus_id, "rows": rows},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def benchmark_asr_candidate(
    config: AsrBenchmarkConfig,
    *,
    adapter_factory: Callable[[], AsrAdapter],
    pcm: bytes,
    language: Language,
    now_ns: Callable[[], int],
    resource_sample: Callable[[], ResourceSample],
) -> AsrBenchmarkReport:
    if (
        not config.model_id
        or config.audio_duration_ms <= 0
        or config.warmup_count < 0
        or config.measured_count <= 0
        or not pcm
    ):
        raise CorpusError("ASR benchmark configuration is invalid")
    resources = [resource_sample()]
    periodic = _PeriodicResourceSampler(resource_sample)
    periodic.start()
    cold_started_ns = now_ns()
    try:
        adapter = adapter_factory()
        adapter.transcribe(
            pcm,
            language=language,
            mode=TranslationMode.QUALITY_FIRST,
        )
        cold_inference_ms = (now_ns() - cold_started_ns) / 1_000_000
        resources.append(resource_sample())

        for _ in range(config.warmup_count):
            adapter.transcribe(
                pcm,
                language=language,
                mode=TranslationMode.QUALITY_FIRST,
            )
            resources.append(resource_sample())

        measured_ms: list[float] = []
        for _ in range(config.measured_count):
            started_ns = now_ns()
            adapter.transcribe(
                pcm,
                language=language,
                mode=TranslationMode.QUALITY_FIRST,
            )
            measured_ms.append((now_ns() - started_ns) / 1_000_000)
            resources.append(resource_sample())
    finally:
        resources.extend(periodic.stop())

    total_runtime_ms = sum(measured_ms)
    if total_runtime_ms <= 0:
        raise CorpusError("ASR benchmark clock did not advance")
    cpu, rss, gpu, vram = _resource_peaks(resources)
    return AsrBenchmarkReport(
        model_id=config.model_id,
        excluded_warmups=config.warmup_count,
        measured_count=config.measured_count,
        cold_inference_ms=cold_inference_ms,
        warm_p95_ms=_nearest_rank_p95(measured_ms),
        audio_throughput_x=(
            config.measured_count * config.audio_duration_ms / total_runtime_ms
        ),
        cpu_percent_peak=cpu,
        rss_bytes_peak=rss,
        gpu_percent_peak=gpu,
        vram_mib_peak=vram,
    )


def benchmark_simultaneous_duplex(
    config: DuplexBenchmarkConfig,
    *,
    run_direction: Callable[[Language, UUID], float],
    resource_sample: Callable[[], ResourceSample],
) -> DuplexBenchmarkReport:
    if (
        not config.model_id
        or config.warmup_count < 0
        or config.measured_count_per_direction <= 0
    ):
        raise CorpusError("duplex benchmark configuration is invalid")
    ru_to_en: list[float] = []
    en_to_ru: list[float] = []
    resources = [resource_sample()]
    periodic = _PeriodicResourceSampler(resource_sample)
    periodic.start()
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            total_pairs = config.warmup_count + config.measured_count_per_direction
            for pair_index in range(total_pairs):
                ru_future = executor.submit(
                    run_direction,
                    Language.RU,
                    uuid4(),
                )
                en_future = executor.submit(
                    run_direction,
                    Language.EN,
                    uuid4(),
                )
                resources.append(resource_sample())
                ru_latency = float(ru_future.result())
                en_latency = float(en_future.result())
                if pair_index >= config.warmup_count:
                    ru_to_en.append(ru_latency)
                    en_to_ru.append(en_latency)
                resources.append(resource_sample())
    finally:
        resources.extend(periodic.stop())
    cpu, rss, gpu, vram = _resource_peaks(resources)
    return DuplexBenchmarkReport(
        model_id=config.model_id,
        simultaneous=True,
        excluded_warmups=config.warmup_count,
        measured_per_direction=config.measured_count_per_direction,
        ru_to_en_latency_ms=tuple(ru_to_en),
        en_to_ru_latency_ms=tuple(en_to_ru),
        cpu_percent_peak=cpu,
        rss_bytes_peak=rss,
        gpu_percent_peak=gpu,
        vram_mib_peak=vram,
    )


def _nearest_rank_p95(values: Sequence[float]) -> float:
    if not values:
        raise CorpusError("latency samples are empty")
    ordered = sorted(values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


def _resource_peaks(
    samples: Sequence[ResourceSample],
) -> ResourceSample:
    if not samples:
        raise CorpusError("resource samples are empty")
    return (
        max(sample[0] for sample in samples),
        max(sample[1] for sample in samples),
        max(sample[2] for sample in samples),
        max(sample[3] for sample in samples),
    )


def _evaluate_direction(
    cases: Sequence[QualityCase],
    *,
    target_language: Language,
    outputs: Sequence[str],
    synthesized_transcripts: Sequence[str],
) -> DirectionQuality:
    references = [
        case.en if target_language is Language.EN else case.ru for case in cases
    ]
    chrf2 = CHRF(beta=2).corpus_score(list(outputs), [references]).score
    synthesized_wer = wer(list(outputs), list(synthesized_transcripts))
    violations = tuple(
        violation
        for case, output in zip(cases, outputs, strict=True)
        for violation in _critical_violations(
            case,
            output,
            target_language=target_language,
        )
    )
    return DirectionQuality(
        chrf2=chrf2,
        synthesized_wer=synthesized_wer,
        critical_violations=violations,
    )


def _critical_violations(
    case: QualityCase,
    output: str,
    *,
    target_language: Language,
) -> tuple[CriticalViolation, ...]:
    violations: list[CriticalViolation] = []
    for kind in case.critical:
        if kind == "number":
            if not _number_is_preserved(
                case,
                output,
                target_language=target_language,
            ):
                violations.append(CriticalViolation(case.case_id, kind))
        elif kind == "name":
            aliases = (
                case.accepted_en_names
                if target_language is Language.EN
                else case.accepted_ru_names
            )
            rejected_aliases = (
                case.rejected_en_names
                if target_language is Language.EN
                else case.rejected_ru_names
            )
            allowed_sentence_initials = (
                case.allowed_en_name_initials
                if target_language is Language.EN
                else case.allowed_ru_name_initials
            )
            if not _name_is_preserved(
                output,
                aliases=aliases,
                rejected_aliases=rejected_aliases,
                allowed_sentence_initials=allowed_sentence_initials,
                target_language=target_language,
            ):
                violations.append(CriticalViolation(case.case_id, kind))
        elif kind == "negation":
            anchors = (
                case.negation_en_anchors
                if target_language is Language.EN
                else case.negation_ru_anchors
            )
            has_negation = _negation_is_scoped(
                output,
                anchors=anchors,
                target_language=target_language,
            )
            if not has_negation:
                violations.append(CriticalViolation(case.case_id, kind))
    return tuple(violations)


def _digits(value: str) -> str:
    return "".join(regex.findall(r"\p{Nd}", value))


def _numeric_expressions(value: str) -> set[str]:
    expressions = regex.findall(
        r"(?<!\p{N})\p{N}+(?:\s*[:.,/-]\s*\p{N}+)*(?!\p{N})",
        value,
    )
    return {_digits(expression) for expression in expressions}


def _number_is_preserved(
    case: QualityCase,
    output: str,
    *,
    target_language: Language,
) -> bool:
    expressions = _numeric_expressions(output)
    if case.number_semantics == "identifier":
        anchors = (
            case.number_role_en_anchors
            if target_language is Language.EN
            else case.number_role_ru_anchors
        )
        output_tokens = _word_tokens(output)
        return expressions == {_digits(case.value)} and any(
            _contains_token_sequence(output_tokens, _word_tokens(anchor))
            for anchor in anchors
        )
    if case.number_semantics == "count":
        expected = int(_digits(case.value))
        words = set(_word_tokens(output))
        observed = {int(expression) for expression in expressions if expression}
        observed.update(
            value
            for value, aliases in _COUNT_WORDS[target_language].items()
            if aliases & words
        )
        return observed == {expected}
    if case.number_semantics == "time":
        expected_hour, expected_minute = (
            int(part) for part in case.value.split(":", maxsplit=1)
        )
        return _time_values(output) == {(expected_hour, expected_minute)}
    return False


def _time_values(value: str) -> set[tuple[int, int]]:
    matches = regex.finditer(
        r"(?<!\p{N})(?P<hour>[01]?\p{N}|2[0-3])"
        r"\s*:\s*(?P<minute>[0-5]\p{N})"
        r"(?:\s*(?P<meridiem>[ap])\.?m\.?)?(?!\p{N})",
        value.casefold(),
    )
    times: set[tuple[int, int]] = set()
    for match in matches:
        hour = int(match.group("hour"))
        minute = int(match.group("minute"))
        meridiem = match.group("meridiem")
        if meridiem == "p" and hour < 12:
            hour += 12
        elif meridiem == "a" and hour == 12:
            hour = 0
        times.add((hour, minute))
    return times


def _word_tokens(value: str) -> tuple[str, ...]:
    return tuple(regex.findall(r"\p{L}+", value.casefold()))


def _contains_token_sequence(
    tokens: tuple[str, ...],
    expected: tuple[str, ...],
) -> bool:
    return bool(expected) and any(
        tokens[index : index + len(expected)] == expected
        for index in range(len(tokens) - len(expected) + 1)
    )


def _name_is_preserved(
    output: str,
    *,
    aliases: tuple[str, ...],
    rejected_aliases: tuple[str, ...],
    allowed_sentence_initials: tuple[str, ...],
    target_language: Language,
) -> bool:
    output_tokens = tuple(regex.findall(r"\p{L}+", output))
    folded_tokens = tuple(token.casefold() for token in output_tokens)
    alias_tokens = tuple(_word_tokens(alias) for alias in aliases)
    alias_spans = tuple(
        (index, index + len(expected))
        for expected in alias_tokens
        for index in range(len(folded_tokens) - len(expected) + 1)
        if folded_tokens[index : index + len(expected)] == expected
    )
    if not alias_spans:
        return False
    if any(
        _contains_token_sequence(folded_tokens, _word_tokens(alias))
        for alias in rejected_aliases
    ):
        return False
    allowed_title_tokens = {token for expected in alias_tokens for token in expected}
    allowed_title_tokens.update(
        token
        for initial in allowed_sentence_initials
        for token in _word_tokens(initial)
    )
    if target_language is Language.EN:
        allowed_title_tokens.add("i")
    if any(
        token[:1].isupper() and token.casefold() not in allowed_title_tokens
        for token in output_tokens
    ):
        return False
    conjunctions = (
        {"and", "or"} if target_language is Language.EN else {"и", "или"}
    )
    return all(
        (start == 0 or folded_tokens[start - 1] not in conjunctions)
        and (end == len(folded_tokens) or folded_tokens[end] not in conjunctions)
        for start, end in alias_spans
    )


def _negation_is_scoped(
    output: str,
    *,
    anchors: tuple[str, ...],
    target_language: Language,
) -> bool:
    negation_words = (
        {"not", "no", "never"}
        if target_language is Language.EN
        else {"не", "нет", "никогда"}
    )
    anchor_tokens = tuple(_word_tokens(anchor) for anchor in anchors)
    for raw_clause in regex.split(r"[.;!?]+", output):
        normalized_clause = regex.sub(
            r"n['’]t\b",
            " not",
            raw_clause.casefold(),
        )
        tokens = _word_tokens(normalized_clause)
        for expected in anchor_tokens:
            for anchor_index in range(len(tokens) - len(expected) + 1):
                if tokens[anchor_index : anchor_index + len(expected)] != expected:
                    continue
                if anchor_index > 0 and tokens[anchor_index - 1] in negation_words:
                    return True
    return False


def _validate_measurements(
    values: Mapping[Language, Sequence[str]],
    expected_count: int,
) -> None:
    if set(values) != {Language.RU, Language.EN}:
        raise CorpusError("quality results must contain both directions")
    if any(len(values[language]) != expected_count for language in values):
        raise CorpusError("quality result cardinality is invalid")
    if any(
        not isinstance(value, str) for language in values for value in values[language]
    ):
        raise CorpusError("quality result value is invalid")


def _required_text(payload: object, key: str) -> str:
    if not isinstance(payload, dict):
        raise CorpusError("quality corpus object is invalid")
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise CorpusError(f"quality corpus {key} is invalid")
    return value


def _required_template(payload: dict[str, object], key: str) -> str:
    value = _required_text(payload, key)
    if value.count("{value}") != 1:
        raise CorpusError(f"quality corpus {key} template is invalid")
    return value


def _required_labels(
    payload: dict[str, object],
    key: str,
) -> tuple[str, ...]:
    value = payload.get(key)
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(label, str) or not label for label in value)
    ):
        raise CorpusError(f"quality corpus {key} labels are invalid")
    return tuple(value)


def _required_name_aliases(
    payload: dict[str, object],
) -> dict[str, dict[Language, tuple[str, ...]]]:
    raw_aliases = payload.get("name_aliases")
    if not isinstance(raw_aliases, dict) or not raw_aliases:
        raise CorpusError("quality corpus name aliases are invalid")
    aliases: dict[str, dict[Language, tuple[str, ...]]] = {}
    normalized_owners: dict[
        tuple[Language, tuple[str, ...]],
        str,
    ] = {}
    for value, raw_languages in raw_aliases.items():
        if (
            not isinstance(value, str)
            or not value
            or not isinstance(raw_languages, dict)
        ):
            raise CorpusError("quality corpus name alias is invalid")
        language_aliases: dict[Language, tuple[str, ...]] = {}
        for language in Language:
            raw_values = raw_languages.get(language.value)
            if (
                not isinstance(raw_values, list)
                or not raw_values
                or any(
                    not isinstance(alias, str) or not alias.strip()
                    for alias in raw_values
                )
            ):
                raise CorpusError("quality corpus name alias is invalid")
            language_aliases[language] = tuple(raw_values)
            for alias in raw_values:
                normalized = _word_tokens(alias)
                owner_key = (language, normalized)
                owner = normalized_owners.setdefault(owner_key, value)
                if not normalized or owner != value:
                    raise CorpusError("quality corpus name aliases collide")
        aliases[value] = language_aliases
    return aliases


def _required_negation_anchors(
    payload: dict[str, object],
    *,
    required: bool,
) -> dict[Language, tuple[str, ...]]:
    return _required_language_anchors(
        payload,
        key="negation_anchors",
        required=required,
    )


def _required_language_anchors(
    payload: dict[str, object],
    *,
    key: str,
    required: bool,
) -> dict[Language, tuple[str, ...]]:
    raw_anchors = payload.get(key)
    if not required:
        if raw_anchors is not None:
            raise CorpusError(f"quality corpus {key} are invalid")
        return {language: () for language in Language}
    if not isinstance(raw_anchors, dict):
        raise CorpusError(f"quality corpus {key} are invalid")
    anchors: dict[Language, tuple[str, ...]] = {}
    for language in Language:
        raw_values = raw_anchors.get(language.value)
        if (
            not isinstance(raw_values, list)
            or not raw_values
            or any(
                not isinstance(value, str) or not _word_tokens(value)
                for value in raw_values
            )
        ):
            raise CorpusError(f"quality corpus {key} are invalid")
        anchors[language] = tuple(raw_values)
    return anchors

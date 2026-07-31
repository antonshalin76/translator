from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
from threading import Barrier, Lock, current_thread
import time
from uuid import UUID

from jiwer import wer
import pytest
from sacrebleu.metrics import CHRF

from translator_sidecar.benchmark.task6 import (
    AsrBenchmarkConfig,
    CorpusError,
    DuplexBenchmarkConfig,
    benchmark_asr_candidate,
    benchmark_simultaneous_duplex,
    evaluate_quality,
    load_quality_corpus,
    passes_quality_thresholds,
    run_quality_benchmark,
    within_vram_budget,
)
from translator_sidecar.benchmark import task6_live
from translator_sidecar.benchmark.task6_live import (
    _build_payload,
    _run_voice_smokes,
)
from translator_sidecar.provider_contract import (
    Language,
    TranslationMode,
)


CORPUS_PATH = Path(__file__).parent / "quality_corpus" / "task6-v4.json"


def test_versioned_corpus_expands_to_ten_warmups_and_one_hundred_cases() -> None:
    corpus = load_quality_corpus(CORPUS_PATH)

    assert corpus.schema_version == "translator.quality-corpus.v4"
    assert corpus.corpus_id == "task6-v4"
    assert len(corpus.warmups) == 10
    assert len(corpus.cases) == 100
    assert len({case.case_id for case in corpus.cases}) == 100
    assert all(case.ru and case.en for case in corpus.cases)
    assert {"negation", "number", "name"} <= {
        label for case in corpus.cases for label in case.critical
    }
    assert {"short", "long", "duplex_overlap"} <= {
        scenario for case in corpus.cases for scenario in case.scenarios
    }
    assert max(len(case.en.split()) for case in corpus.cases) >= 20


def test_quality_metrics_use_measured_cases_only_and_accept_exact_references() -> None:
    corpus = load_quality_corpus(CORPUS_PATH)
    outputs = {
        Language.RU: [case.ru for case in corpus.cases],
        Language.EN: [case.en for case in corpus.cases],
    }
    synthesized_transcripts = {
        Language.RU: [case.ru for case in corpus.cases],
        Language.EN: [case.en for case in corpus.cases],
    }

    report = evaluate_quality(
        corpus,
        outputs=outputs,
        synthesized_transcripts=synthesized_transcripts,
    )

    assert report.measured_per_direction == 100
    assert report.excluded_warmups == 10
    assert report.ru_to_en.chrf2 == pytest.approx(100.0)
    assert report.en_to_ru.chrf2 == pytest.approx(100.0)
    assert report.ru_to_en.synthesized_wer == pytest.approx(0.0)
    assert report.en_to_ru.synthesized_wer == pytest.approx(0.0)
    assert report.ru_to_en.critical_violations == ()
    assert report.en_to_ru.critical_violations == ()
    assert report.passes_thresholds


def test_metrics_match_sacrebleu_chrf2_and_jiwer_oracles() -> None:
    corpus = load_quality_corpus(CORPUS_PATH)
    references = {
        Language.RU: [case.ru for case in corpus.cases],
        Language.EN: [case.en for case in corpus.cases],
    }
    outputs = {language: list(values) for language, values in references.items()}
    transcripts = {language: list(values) for language, values in references.items()}
    outputs[Language.EN][0] = "I confirm the order."
    transcripts[Language.EN][1] = "Do not mute microphone until 09:10."
    outputs[Language.RU][0] = "Подтверждаю заказ."
    transcripts[Language.RU][1] = "Не отключайте микрофон 09:10."

    report = evaluate_quality(
        corpus,
        outputs=outputs,
        synthesized_transcripts=transcripts,
    )
    for language, direction in [
        (Language.EN, report.ru_to_en),
        (Language.RU, report.en_to_ru),
    ]:
        expected_chrf2 = (
            CHRF(beta=2)
            .corpus_score(
                outputs[language],
                [references[language]],
            )
            .score
        )
        chrf1 = (
            CHRF(beta=1)
            .corpus_score(
                outputs[language],
                [references[language]],
            )
            .score
        )
        expected_wer = wer(outputs[language], transcripts[language])
        assert expected_chrf2 != pytest.approx(chrf1)
        assert direction.chrf2 == pytest.approx(expected_chrf2)
        assert direction.synthesized_wer == pytest.approx(expected_wer)


def test_quality_threshold_boundaries_are_inclusive() -> None:
    assert passes_quality_thresholds(
        chrf2=45.0,
        synthesized_wer=0.15,
        critical_violation_count=0,
    )
    assert not passes_quality_thresholds(
        chrf2=44.999,
        synthesized_wer=0.15,
        critical_violation_count=0,
    )
    assert not passes_quality_thresholds(
        chrf2=45.0,
        synthesized_wer=0.15001,
        critical_violation_count=0,
    )
    assert not passes_quality_thresholds(
        chrf2=45.0,
        synthesized_wer=0.15,
        critical_violation_count=1,
    )


def test_report_verdict_applies_chrf2_and_wer_thresholds() -> None:
    corpus = load_quality_corpus(CORPUS_PATH)
    references = {
        Language.RU: [case.ru for case in corpus.cases],
        Language.EN: [case.en for case in corpus.cases],
    }
    for failing_language in (Language.RU, Language.EN):
        low_quality = {
            language: list(values) for language, values in references.items()
        }
        low_quality[failing_language] = [
            f"{'не ' if 'negation' in case.critical and failing_language is Language.RU else ''}"
            f"{'not ' if 'negation' in case.critical and failing_language is Language.EN else ''}"
            f"{case.value}"
            for case in corpus.cases
        ]
        low_chrf = evaluate_quality(
            corpus,
            outputs=low_quality,
            synthesized_transcripts=references,
        )
        failing = (
            low_chrf.en_to_ru if failing_language is Language.RU else low_chrf.ru_to_en
        )
        passing = (
            low_chrf.ru_to_en if failing_language is Language.RU else low_chrf.en_to_ru
        )
        assert failing.chrf2 < 45
        assert passing.chrf2 == pytest.approx(100)
        assert not low_chrf.passes_thresholds

        transcripts = {
            language: list(values) for language, values in references.items()
        }
        transcripts[failing_language] = [
            "шум" if failing_language is Language.RU else "noise"
        ] * 100
        high_wer = evaluate_quality(
            corpus,
            outputs=references,
            synthesized_transcripts=transcripts,
        )
        failing = (
            high_wer.en_to_ru if failing_language is Language.RU else high_wer.ru_to_en
        )
        passing = (
            high_wer.ru_to_en if failing_language is Language.RU else high_wer.en_to_ru
        )
        assert failing.synthesized_wer > 0.15
        assert passing.synthesized_wer == pytest.approx(0)
        assert not high_wer.passes_thresholds


@pytest.mark.parametrize(
    ("language", "case_id", "corruption", "expected_kind"),
    [
        (Language.EN, "disagree:1", "I agree with option 1.", "negation"),
        (
            Language.EN,
            "order-number:104",
            "I confirm order number 999.",
            "number",
        ),
        (
            Language.RU,
            "send-file:Alex",
            "Передайте файл пользователю Boris.",
            "name",
        ),
        (
            Language.RU,
            "send-file:Alex",
            "Передайте файл пользователю Александру.",
            "name",
        ),
        (
            Language.EN,
            "participants:12",
            "There are 1 or 2 participants in the room.",
            "number",
        ),
        (
            Language.EN,
            "order-number:104",
            "I confirm order number 104 or 999.",
            "number",
        ),
        (
            Language.EN,
            "send-file:Alex",
            "Send the file to Alex and Boris.",
            "name",
        ),
        (
            Language.EN,
            "do-not-mute:09:10",
            "Do not wait; mute the microphone until 09:10.",
            "negation",
        ),
        (
            Language.EN,
            "order-number:104",
            "I confirm channel number 104.",
            "number",
        ),
        (
            Language.EN,
            "send-file:Alex",
            "Boris sends the file to Alex.",
            "name",
        ),
        (
            Language.EN,
            "send-file:Alex",
            "Send the file to Alex and boris.",
            "name",
        ),
        (
            Language.EN,
            "do-not-mute:09:10",
            "Do not fail to mute the microphone until 09:10.",
            "negation",
        ),
    ],
)
def test_critical_corruption_is_reported_by_case_and_kind(
    language: Language,
    case_id: str,
    corruption: str,
    expected_kind: str,
) -> None:
    corpus = load_quality_corpus(CORPUS_PATH)
    outputs = {
        Language.RU: [case.ru for case in corpus.cases],
        Language.EN: [case.en for case in corpus.cases],
    }
    index = next(
        index for index, case in enumerate(corpus.cases) if case.case_id == case_id
    )
    outputs[language][index] = corruption

    report = evaluate_quality(
        corpus,
        outputs=outputs,
        synthesized_transcripts={
            Language.RU: [case.ru for case in corpus.cases],
            Language.EN: [case.en for case in corpus.cases],
        },
    )
    direction = report.ru_to_en if language is Language.EN else report.en_to_ru

    assert any(
        violation.case_id == corpus.cases[index].case_id
        and violation.kind == expected_kind
        for violation in direction.critical_violations
    )
    assert not report.passes_thresholds


def test_critical_oracle_accepts_format_and_cross_script_equivalents() -> None:
    corpus = load_quality_corpus(CORPUS_PATH)
    outputs = {
        Language.RU: [case.ru for case in corpus.cases],
        Language.EN: [case.en for case in corpus.cases],
    }
    replacements = {
        (Language.EN, "do-not-mute:09:10"): (
            "Don't mute the microphone until 09 : 10."
        ),
        (Language.RU, "send-file:Alex"): ("Передайте файл пользователю Алексу."),
        (Language.EN, "participants:12"): (
            "There are twelve participants in the room."
        ),
        (Language.EN, "scheduled:13:15"): ("My meeting is scheduled for 1:15 p.m."),
        (Language.RU, "do-not-mute:10:00"): (
            "Не заглушай микрофон до 10:00."
        ),
    }
    for (language, case_id), value in replacements.items():
        index = next(
            index for index, case in enumerate(corpus.cases) if case.case_id == case_id
        )
        outputs[language][index] = value

    report = evaluate_quality(
        corpus,
        outputs=outputs,
        synthesized_transcripts={
            Language.RU: [case.ru for case in corpus.cases],
            Language.EN: [case.en for case in corpus.cases],
        },
    )

    assert not report.ru_to_en.critical_violations
    assert not report.en_to_ru.critical_violations


def test_corpus_and_metric_cardinality_fail_closed(tmp_path: Path) -> None:
    payload = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    for mutation in ("nine_warmups", "ninety_nine_cases"):
        invalid = tmp_path / f"{mutation}.json"
        mutated = json.loads(json.dumps(payload))
        if mutation == "nine_warmups":
            mutated["warmups"].pop()
        else:
            mutated["templates"][0]["values"].pop()
        invalid.write_text(json.dumps(mutated), encoding="utf-8")
        with pytest.raises(CorpusError):
            load_quality_corpus(invalid)
    eleven_warmups = json.loads(json.dumps(payload))
    eleven_warmups["warmups"].append(eleven_warmups["warmups"][0])
    invalid_warmups = tmp_path / "eleven-warmups.json"
    invalid_warmups.write_text(json.dumps(eleven_warmups), encoding="utf-8")
    with pytest.raises(CorpusError):
        load_quality_corpus(invalid_warmups)

    invalid_alias_payloads = []
    missing_language = json.loads(json.dumps(payload))
    del missing_language["name_aliases"]["Alex"]["ru"]
    invalid_alias_payloads.append(missing_language)
    empty_alias = json.loads(json.dumps(payload))
    empty_alias["name_aliases"]["Alex"]["ru"] = []
    invalid_alias_payloads.append(empty_alias)
    missing_canonical = json.loads(json.dumps(payload))
    del missing_canonical["name_aliases"]["Alex"]
    invalid_alias_payloads.append(missing_canonical)
    unknown_canonical = json.loads(json.dumps(payload))
    unknown_canonical["name_aliases"]["Unknown"] = {
        "ru": ["Неизвестный"],
        "en": ["Unknown"],
    }
    invalid_alias_payloads.append(unknown_canonical)
    colliding_alias = json.loads(json.dumps(payload))
    colliding_alias["name_aliases"]["Max"]["ru"].append("Алекс")
    invalid_alias_payloads.append(colliding_alias)
    for index, invalid_alias_payload in enumerate(invalid_alias_payloads):
        invalid_alias_path = tmp_path / f"invalid-alias-{index}.json"
        invalid_alias_path.write_text(
            json.dumps(invalid_alias_payload),
            encoding="utf-8",
        )
        with pytest.raises(CorpusError):
            load_quality_corpus(invalid_alias_path)

    missing_negation_anchor = json.loads(json.dumps(payload))
    negation_template = next(
        template
        for template in missing_negation_anchor["templates"]
        if "negation" in template["critical"]
    )
    del negation_template["negation_anchors"]["en"]
    invalid_negation = tmp_path / "missing-negation-anchor.json"
    invalid_negation.write_text(
        json.dumps(missing_negation_anchor),
        encoding="utf-8",
    )
    with pytest.raises(CorpusError):
        load_quality_corpus(invalid_negation)

    unexpected_negation_anchor = json.loads(json.dumps(payload))
    non_negation_template = next(
        template
        for template in unexpected_negation_anchor["templates"]
        if "negation" not in template["critical"]
    )
    non_negation_template["negation_anchors"] = {
        "ru": ["лишний"],
        "en": ["unexpected"],
    }
    invalid_negation = tmp_path / "unexpected-negation-anchor.json"
    invalid_negation.write_text(
        json.dumps(unexpected_negation_anchor),
        encoding="utf-8",
    )
    with pytest.raises(CorpusError):
        load_quality_corpus(invalid_negation)

    missing_number_role = json.loads(json.dumps(payload))
    identifier_template = next(
        template
        for template in missing_number_role["templates"]
        if template.get("number_semantics") == "identifier"
    )
    del identifier_template["number_role_anchors"]
    invalid_role = tmp_path / "missing-number-role.json"
    invalid_role.write_text(json.dumps(missing_number_role), encoding="utf-8")
    with pytest.raises(CorpusError):
        load_quality_corpus(invalid_role)

    missing_name_initials = json.loads(json.dumps(payload))
    name_template = next(
        template
        for template in missing_name_initials["templates"]
        if "name" in template["critical"]
    )
    del name_template["name_sentence_initials"]
    invalid_name = tmp_path / "missing-name-initials.json"
    invalid_name.write_text(json.dumps(missing_name_initials), encoding="utf-8")
    with pytest.raises(CorpusError):
        load_quality_corpus(invalid_name)

    one_hundred_one = json.loads(json.dumps(payload))
    one_hundred_one["templates"][0]["values"].append("1205")
    valid_101 = tmp_path / "one-hundred-one.json"
    valid_101.write_text(json.dumps(one_hundred_one), encoding="utf-8")
    assert len(load_quality_corpus(valid_101).cases) == 101

    corpus = load_quality_corpus(CORPUS_PATH)
    references = {
        Language.RU: [case.ru for case in corpus.cases],
        Language.EN: [case.en for case in corpus.cases],
    }
    invalid_metric_inputs = [
        (
            {**references, Language.EN: references[Language.EN][:-1]},
            references,
        ),
        (
            {**references, Language.EN: [*references[Language.EN], "extra"]},
            references,
        ),
        (
            references,
            {**references, Language.RU: references[Language.RU][:-1]},
        ),
        (
            {Language.RU: references[Language.RU]},
            references,
        ),
    ]
    for outputs, transcripts in invalid_metric_inputs:
        with pytest.raises(CorpusError):
            evaluate_quality(
                corpus,
                outputs=outputs,
                synthesized_transcripts=transcripts,
            )


def test_quality_runner_excludes_warmups_and_measures_every_case() -> None:
    corpus = load_quality_corpus(CORPUS_PATH)
    translations = {(case.ru, Language.EN): case.en for case in corpus.cases} | {
        (case.en, Language.RU): case.ru for case in corpus.cases
    }
    translations.update(
        {(warmup.ru, Language.EN): warmup.en for warmup in corpus.warmups}
    )
    translations.update(
        {(warmup.en, Language.RU): warmup.ru for warmup in corpus.warmups}
    )

    class FakeTranslator:
        def __init__(self) -> None:
            self.calls: list[tuple[str, Language, Language]] = []

        def translate(
            self,
            text: str,
            *,
            source_language: Language,
            target_language: Language,
            mode: TranslationMode,
        ) -> str:
            assert mode is TranslationMode.QUALITY_FIRST
            self.calls.append((text, source_language, target_language))
            return translations[(text, target_language)]

    translator = FakeTranslator()
    synthesized: list[tuple[str, Language]] = []

    def synthesize_and_transcribe(text: str, language: Language) -> str:
        synthesized.append((text, language))
        return text

    ticks = iter(range(0, 10_000_000_000, 1_000_000))
    run = run_quality_benchmark(
        corpus,
        translator=translator,
        synthesize_and_transcribe=synthesize_and_transcribe,
        now_ns=lambda: next(ticks),
    )

    assert len(translator.calls) == 220
    assert len(synthesized) == 200
    expected_translation_calls = [
        *((warmup.ru, Language.RU, Language.EN) for warmup in corpus.warmups),
        *((case.ru, Language.RU, Language.EN) for case in corpus.cases),
        *((warmup.en, Language.EN, Language.RU) for warmup in corpus.warmups),
        *((case.en, Language.EN, Language.RU) for case in corpus.cases),
    ]
    assert translator.calls == expected_translation_calls
    assert synthesized == [
        *((case.en, Language.EN) for case in corpus.cases),
        *((case.ru, Language.RU) for case in corpus.cases),
    ]
    assert run.excluded_warmups == 10
    assert run.measured_per_direction == 100
    assert run.ru_to_en.success_count == 100
    assert run.en_to_ru.success_count == 100
    assert run.ru_to_en.drop_rate == pytest.approx(0)
    assert run.en_to_ru.drop_rate == pytest.approx(0)
    assert run.quality.passes_thresholds


def test_quality_runner_counts_drops_per_direction_with_fixed_denominator() -> None:
    corpus = load_quality_corpus(CORPUS_PATH)
    translations = {(case.ru, Language.EN): case.en for case in corpus.cases} | {
        (case.en, Language.RU): case.ru for case in corpus.cases
    }
    translations.update(
        {(warmup.ru, Language.EN): warmup.en for warmup in corpus.warmups}
    )
    translations.update(
        {(warmup.en, Language.RU): warmup.ru for warmup in corpus.warmups}
    )
    failed_source = corpus.cases[50].ru

    class DroppingTranslator:
        def translate(self, text, *, source_language, target_language, mode):
            if text == failed_source:
                raise RuntimeError("private-quality-drop-marker")
            return translations[(text, target_language)]

    ticks = iter(range(0, 10_000_000_000, 1_000_000))
    synthesized: list[tuple[str, Language]] = []

    def synthesize_and_transcribe(text: str, language: Language) -> str:
        synthesized.append((text, language))
        return text

    run = run_quality_benchmark(
        corpus,
        translator=DroppingTranslator(),
        synthesize_and_transcribe=synthesize_and_transcribe,
        now_ns=lambda: next(ticks),
    )

    assert run.ru_to_en.success_count == 99
    assert run.ru_to_en.drop_count == 1
    assert run.ru_to_en.drop_rate == pytest.approx(0.01)
    assert len(run.ru_to_en.success_latency_ms) == 99
    assert not run.ru_to_en.passes_drop_threshold
    assert run.en_to_ru.success_count == 100
    assert run.en_to_ru.drop_count == 0
    assert run.en_to_ru.drop_rate == pytest.approx(0)
    assert run.en_to_ru.passes_drop_threshold
    assert len(synthesized) == 199
    assert (corpus.cases[50].en, Language.EN) not in synthesized
    assert not run.passes_thresholds


def test_asr_candidate_benchmark_has_cold_warm_and_resource_evidence() -> None:
    class ControlledClock:
        def __init__(self) -> None:
            self.ns = 0

        def now_ns(self) -> int:
            return self.ns

        def advance_ms(self, value: int) -> None:
            self.ns += value * 1_000_000

    clock = ControlledClock()
    measured_durations = list(range(1, 101))
    durations = iter([10, *([2] * 10), *measured_durations])

    class FakeAsr:
        def __init__(self) -> None:
            self.calls = 0

        def transcribe(self, pcm, *, language, mode) -> str:
            assert pcm == b"\x00\x00" * 16_000
            assert language is Language.EN
            assert mode is TranslationMode.QUALITY_FIRST
            self.calls += 1
            clock.advance_ms(next(durations))
            return "measured speech"

    factory_calls = 0
    adapter: FakeAsr | None = None

    def factory() -> FakeAsr:
        nonlocal factory_calls, adapter
        factory_calls += 1
        clock.advance_ms(40)
        adapter = FakeAsr()
        return adapter

    resource_call_count = 0

    def resource_sample() -> tuple[float, int, float, int]:
        nonlocal resource_call_count
        resource_call_count += 1
        if adapter is not None and adapter.calls == 50:
            return (30.0, 120_000_000, 40.0, 2_000)
        return (10.0, 100_000_000, 20.0, 1_000)

    report = benchmark_asr_candidate(
        AsrBenchmarkConfig(
            model_id="fake-asr",
            audio_duration_ms=1_000,
            warmup_count=10,
            measured_count=100,
        ),
        adapter_factory=factory,
        pcm=b"\x00\x00" * 16_000,
        language=Language.EN,
        now_ns=clock.now_ns,
        resource_sample=resource_sample,
    )

    assert factory_calls == 1
    assert adapter is not None
    assert adapter.calls == 111
    assert resource_call_count >= 112
    assert report.model_id == "fake-asr"
    assert report.excluded_warmups == 10
    assert report.measured_count == 100
    assert report.cold_inference_ms == pytest.approx(50)
    assert report.warm_p95_ms == pytest.approx(95)
    assert report.audio_throughput_x == pytest.approx(100_000 / 5_050)
    assert report.cpu_percent_peak == pytest.approx(30)
    assert report.rss_bytes_peak == 120_000_000
    assert report.gpu_percent_peak == pytest.approx(40)
    assert report.vram_mib_peak == 2_000


def test_asr_resource_sampler_observes_peak_during_inference() -> None:
    lock = Lock()
    active = False

    class BlockingAsr:
        def transcribe(self, pcm, *, language, mode) -> str:
            nonlocal active
            with lock:
                active = True
            time.sleep(0.08)
            with lock:
                active = False
            return "speech"

    def resource_sample() -> tuple[float, int, float, int]:
        with lock:
            is_active = active
        return (
            (200.0, 200_000_000, 90.0, 4_000)
            if is_active
            else (1.0, 100_000_000, 1.0, 1_000)
        )

    report = benchmark_asr_candidate(
        AsrBenchmarkConfig(
            model_id="blocking-asr",
            audio_duration_ms=1_000,
            warmup_count=0,
            measured_count=1,
        ),
        adapter_factory=BlockingAsr,
        pcm=b"\0\0" * 16_000,
        language=Language.EN,
        now_ns=time.monotonic_ns,
        resource_sample=resource_sample,
    )

    assert report.cpu_percent_peak == pytest.approx(200)
    assert report.rss_bytes_peak == 200_000_000
    assert report.gpu_percent_peak == pytest.approx(90)
    assert report.vram_mib_peak == 4_000


def test_simultaneous_duplex_uses_two_isolated_sessions_concurrently() -> None:
    barrier = Barrier(2)
    lock = Lock()
    active = 0
    peak_active = 0
    observed: list[tuple[Language, UUID]] = []

    def run_direction(language: Language, session_id: UUID) -> float:
        nonlocal active, peak_active
        with lock:
            active += 1
            peak_active = max(peak_active, active)
            observed.append((language, session_id))
        barrier.wait(timeout=1)
        with lock:
            active -= 1
        return 125.0 if language is Language.RU else 150.0

    report = benchmark_simultaneous_duplex(
        DuplexBenchmarkConfig(
            model_id="fake-small",
            warmup_count=2,
            measured_count_per_direction=3,
        ),
        run_direction=run_direction,
        resource_sample=lambda: (55.0, 500_000_000, 70.0, 9_500),
    )

    assert peak_active == 2
    assert report.model_id == "fake-small"
    assert {language for language, _ in observed} == {
        Language.RU,
        Language.EN,
    }
    assert len(observed) == 10
    assert len({session_id for _, session_id in observed}) == 10
    assert report.simultaneous
    assert report.excluded_warmups == 2
    assert report.measured_per_direction == 3
    assert report.ru_to_en_latency_ms == pytest.approx((125.0,) * 3)
    assert report.en_to_ru_latency_ms == pytest.approx((150.0,) * 3)
    assert report.vram_mib_peak == 9_500
    assert report.vram_within_budget
    assert within_vram_budget(10_240)
    assert not within_vram_budget(10_241)
    over_budget = benchmark_simultaneous_duplex(
        DuplexBenchmarkConfig(
            model_id="over-budget",
            warmup_count=0,
            measured_count_per_direction=1,
        ),
        run_direction=lambda language, session_id: 1.0,
        resource_sample=lambda: (1.0, 1, 1.0, 10_241),
    )
    assert not over_budget.vram_within_budget


def test_duplex_resource_sampler_observes_peak_during_active_pair() -> None:
    lock = Lock()
    active = 0

    def run_direction(language: Language, session_id: UUID) -> float:
        nonlocal active
        with lock:
            active += 1
        time.sleep(0.08)
        with lock:
            active -= 1
        return 10.0

    def resource_sample() -> tuple[float, int, float, int]:
        with lock:
            is_active = active > 0
        if (
            current_thread().name == "translator-resource-sampler"
            and is_active
        ):
            return (200.0, 200_000_000, 90.0, 4_000)
        return (1.0, 100_000_000, 1.0, 1_000)

    report = benchmark_simultaneous_duplex(
        DuplexBenchmarkConfig(
            model_id="blocking-duplex",
            warmup_count=0,
            measured_count_per_direction=1,
        ),
        run_direction=run_direction,
        resource_sample=resource_sample,
    )

    assert report.cpu_percent_peak == pytest.approx(200)
    assert report.rss_bytes_peak == 200_000_000
    assert report.gpu_percent_peak == pytest.approx(90)
    assert report.vram_mib_peak == 4_000


def test_live_report_persists_computed_acceptance_verdicts() -> None:
    corpus = load_quality_corpus(CORPUS_PATH)
    exact_translations = (
        {(case.ru, Language.EN): case.en for case in corpus.cases}
        | {(case.en, Language.RU): case.ru for case in corpus.cases}
        | {(warmup.ru, Language.EN): warmup.en for warmup in corpus.warmups}
        | {(warmup.en, Language.RU): warmup.ru for warmup in corpus.warmups}
    )
    ticks = iter(range(0, 10_000_000_000, 1_000_000))

    class ExactTranslator:
        def translate(self, text, *, source_language, target_language, mode):
            if text == corpus.cases[50].ru:
                raise RuntimeError("privacy-safe-drop")
            return exact_translations[(text, target_language)]

    run = run_quality_benchmark(
        corpus,
        translator=ExactTranslator(),
        synthesize_and_transcribe=lambda text, language: text,
        now_ns=lambda: next(ticks),
    )
    duplex = benchmark_simultaneous_duplex(
        DuplexBenchmarkConfig(
            model_id="faster-whisper-large-v3",
            warmup_count=0,
            measured_count_per_direction=1,
        ),
        run_direction=lambda language, session_id: 1.0,
        resource_sample=lambda: (1.0, 1, 1.0, 10_241),
    )
    selected_duplex = benchmark_simultaneous_duplex(
        DuplexBenchmarkConfig(
            model_id="faster-whisper-small",
            warmup_count=0,
            measured_count_per_direction=1,
        ),
        run_direction=lambda language, session_id: 1.0,
        resource_sample=lambda: (1.0, 1, 1.0, 1_000),
    )
    payload = _build_payload(
        generated_at_unix_ns=1,
        environment={"device": "cuda"},
        fixture={"sample_rate_hz": 16_000},
        asr_candidates=[{"model_id": "fake"}],
        voice_profiles=[],
        quality_run=run,
        duplex_candidates=(selected_duplex, duplex),
        normal_runtime={"selected_asr": "fake"},
    )
    serialized = json.loads(json.dumps(payload))

    assert serialized["schema_version"] == "translator.task6-benchmark.v2"
    assert serialized["quality"]["passes_thresholds"] is False
    assert serialized["quality"]["quality"]["passes_thresholds"] is False
    assert serialized["quality"]["ru_to_en"]["passes_drop_threshold"] is False
    assert serialized["quality"]["en_to_ru"]["passes_drop_threshold"] is True
    assert serialized["duplex_candidates"][0]["model_id"] == (
        "faster-whisper-small"
    )
    assert serialized["duplex_candidates"][0]["vram_within_budget"] is True
    assert serialized["duplex_candidates"][1]["model_id"] == (
        "faster-whisper-large-v3"
    )
    assert serialized["duplex_candidates"][1]["vram_within_budget"] is False


def test_committed_task6_evidence_is_complete_and_privacy_safe() -> None:
    root = Path(__file__).parents[2]
    corpus = load_quality_corpus(CORPUS_PATH)
    results = json.loads(
        (root / "docs/benchmarks/task6-results.json").read_text(encoding="utf-8")
    )
    review = json.loads(
        (root / "docs/benchmarks/task6-critical-review.json").read_text(
            encoding="utf-8"
        )
    )

    assert results["schema_version"] == "translator.task6-benchmark.v2"
    assert review["schema_version"] == "translator.task6-critical-review.v2"
    assert {
        candidate["model_id"] for candidate in results["duplex_candidates"]
    } == {"faster-whisper-small", "faster-whisper-large-v3"}
    for candidate in results["duplex_candidates"]:
        assert candidate["excluded_warmups"] == 10
        assert candidate["measured_per_direction"] == 100
        assert len(candidate["ru_to_en_latency_ms"]) == 100
        assert len(candidate["en_to_ru_latency_ms"]) == 100
    assert review["reviewed_rows"] == 200
    assert review["corpus_id"] == results["quality"]["quality"]["corpus_id"]
    assert review["meaning_changing_failures"] == 0
    assert review["ambiguities"] == 0
    assert review["reviewer"] == {
        "agent_id": "019faa32-26ba-7a23-a482-d7aecd537733",
        "kind": "independent_critic",
    }
    assert re.fullmatch(r"[0-9a-f]{64}", review["review_input_sha256"])
    assert re.fullmatch(r"[0-9a-f]{64}", review["review_content_sha256"])
    assert review["review_content_sha256"] == (
        results["quality"]["critical_review_content_sha256"]
    )
    assert review["critical_judgments"] == sum(
        len(row["critical"]) for row in review["rows"]
    )
    assert len(review["rows"]) == 200
    assert {
        (row["direction"], row["case_id"]) for row in review["rows"]
    } == {
        (direction, case.case_id)
        for direction in ("ru_to_en", "en_to_ru")
        for case in corpus.cases
    }
    expected_critical = {case.case_id: list(case.critical) for case in corpus.cases}
    assert all(
        row["critical"] == expected_critical[row["case_id"]]
        and row["verdict"] == "pass"
        for row in review["rows"]
    )
    assert review["verdict"] == "pass"
    assert all(
        {"case_id", "direction", "critical", "verdict"} == set(row)
        for row in review["rows"]
    )


def test_live_releases_quality_asr_before_creating_normal_residency(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []

    class QualityAsr:
        def release(self) -> bool:
            events.append("release-large")
            return True

    normal_asr = object()

    def create_normal(**kwargs):
        assert events == ["release-large"]
        events.append("create-small")
        return normal_asr

    monkeypatch.setattr(task6_live, "_asr_manager", create_normal)

    created = task6_live._release_quality_asr_then_create_normal(
        QualityAsr(),
        small_path=tmp_path / "small",
        large_path=tmp_path / "large",
        device="cuda",
    )

    assert created is normal_asr
    assert events == ["release-large", "create-small"]


def test_live_never_creates_normal_asr_when_quality_release_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class QualityAsr:
        def release(self) -> bool:
            return False

    monkeypatch.setattr(
        task6_live,
        "_asr_manager",
        lambda **kwargs: pytest.fail("small ASR created before proven release"),
    )

    with pytest.raises(RuntimeError, match="quality oracle release failed"):
        task6_live._release_quality_asr_then_create_normal(
            QualityAsr(),
            small_path=tmp_path / "small",
            large_path=tmp_path / "large",
            device="cuda",
        )


def test_live_voice_smoke_requires_nonempty_pcm_for_all_four_profiles() -> None:
    observed = []

    class FakeTts:
        def synthesize_frames(self, text, **kwargs):
            observed.append((text, kwargs))
            return iter((b"\0\1", b"\2\3"))

    profiles = _run_voice_smokes(FakeTts())

    assert {(profile["language"], profile["gender"]) for profile in profiles} == {
        ("ru", "male"),
        ("ru", "female"),
        ("en", "male"),
        ("en", "female"),
    }
    assert all(profile["frame_count"] == 2 for profile in profiles)
    assert all(profile["pcm_bytes"] == 4 for profile in profiles)
    assert all(
        set(profile)
        == {
            "language",
            "gender",
            "frame_count",
            "pcm_bytes",
        }
        for profile in profiles
    )


def test_live_resource_telemetry_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        task6_live.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            subprocess.CalledProcessError(1, "nvidia-smi")
        ),
    )

    with pytest.raises(
        task6_live.ResourceTelemetryError,
        match="GPU telemetry is unavailable",
    ):
        task6_live._resource_sample()


def test_live_resource_telemetry_reuses_primed_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeMemory:
        rss = 123_456

    class FakeProcess:
        def cpu_percent(self, *, interval):
            assert interval is None
            return 37.5

        def memory_info(self):
            return FakeMemory()

    class FakeResult:
        stdout = "42, 2048\n"

    monkeypatch.setattr(task6_live, "_PROCESS", FakeProcess())
    monkeypatch.setattr(
        task6_live.subprocess,
        "run",
        lambda *args, **kwargs: FakeResult(),
    )

    assert task6_live._resource_sample() == (
        37.5,
        123_456,
        42.0,
        2_048,
    )

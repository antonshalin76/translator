from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from threading import Barrier, Event, Lock, Thread, current_thread
from types import SimpleNamespace
from uuid import UUID

import pytest
from jiwer import wer
from sacrebleu.metrics import CHRF

from translator_sidecar.benchmark import task6_live
from translator_sidecar.benchmark.task6 import (
    AsrBenchmarkConfig,
    CorpusError,
    DuplexBenchmarkConfig,
    _resource_peaks,
    benchmark_asr_candidate,
    benchmark_simultaneous_duplex,
    evaluate_quality,
    load_quality_corpus,
    passes_quality_thresholds,
    run_quality_benchmark,
    within_vram_budget,
)
from translator_sidecar.benchmark.task6_live import (
    _build_payload,
    _run_voice_smokes,
)
from translator_sidecar.benchmark.task7_e2e import load_task6_quality_evidence
from translator_sidecar.provider_contract import (
    Language,
    TranslationMode,
)

CORPUS_PATH = Path(__file__).parent / "quality_corpus" / "task6-v4.json"


@pytest.mark.parametrize("phase", ["provider", "loop", "thread", "start", "started"])
def test_duplex_transfer_and_bridge_constructor_failure_keep_exact_owner(
    monkeypatch, phase
):
    closes, fatal_calls, loops, threads = [], [], [], []
    error = RuntimeError("synthetic constructor failure")
    real_loop, real_thread = asyncio.new_event_loop, Thread

    class Model:
        def close(self):
            closes.append(self)

    models = [Model() for _ in range(3)]

    def fatal(failure):
        fatal_calls.append((failure, list(closes)))
        raise _WorkerFatal()

    def provider(**kwargs):
        assert (kwargs["asr"], kwargs["translator"], kwargs["tts"]) == tuple(models)
        if phase == "provider":
            raise error
        return object()

    def loop():
        if phase == "loop":
            raise error
        created = real_loop()
        loops.append(created)
        return created

    def thread(**kwargs):
        if phase == "thread":
            raise error
        created = real_thread(**kwargs)
        threads.append(created)
        start = created.start

        def observed_start():
            if phase == "start":
                raise error
            start()
            raise error

        created.start = observed_start
        return created

    monkeypatch.setattr(task6_live, "LocalProvider", provider)
    monkeypatch.setattr(task6_live, "InferenceScheduler", lambda: object())
    monkeypatch.setattr(task6_live.asyncio, "new_event_loop", loop)
    monkeypatch.setattr(task6_live, "Thread", thread)
    monkeypatch.setattr(
        task6_live,
        "benchmark_simultaneous_duplex",
        lambda *_args, **kwargs: pytest.fail("benchmark after constructor failure"),
    )
    try:
        with pytest.raises(
            RuntimeError if phase == "provider" else _WorkerFatal
        ) as caught:
            with ExitStack() as resources:
                for model in models:
                    resources.callback(model.close)
                task6_live._benchmark_provider_duplex(
                    asr=models[0],
                    translator=models[1],
                    tts=models[2],
                    model_id=task6_live._LARGE_ID,
                    source_pcm={},
                    device="cpu",
                    resources=resources,
                    fatal_cleanup=fatal,
                )
        if phase == "provider":
            assert caught.value is error
            assert closes == list(reversed(models)) and not fatal_calls
        else:
            assert fatal_calls == [(error, [])]
            assert closes == []
            if phase == "started":
                assert threads[0].is_alive() and not loops[0].is_closed()
    finally:
        for item in loops:
            if not item.is_closed():
                item.call_soon_threadsafe(item.stop)
        for item in threads:
            if item.ident is not None:
                item.join(1)
                assert not item.is_alive()
        for item in loops:
            if not item.is_closed():
                item.close()


def test_task6_public_run_only_delegates_empty_request(monkeypatch, tmp_path):
    from translator_sidecar.benchmark import process_run

    expected, calls = {"synthetic": "parent-return"}, []
    limits = process_run.RunLimits(model_run_seconds=17, terminate_grace_seconds=2)

    def run(kind, request, output, *, limits):
        calls.append((kind, request, output, limits))
        return expected

    monkeypatch.setattr(process_run, "run_benchmark", run)
    monkeypatch.setattr(
        task6_live,
        "_run_owned",
        lambda **kwargs: pytest.fail("parent constructed benchmark models"),
    )
    output = tmp_path / "existing.json"
    output.write_text("previous")
    assert task6_live.run(output, limits=limits) is expected
    assert calls == [("task6", {}, output, limits)]
    assert output.read_text() == "previous"


class _WorkerFatal(BaseException):
    pass


def test_bridge_close_cannot_overtake_entered_submission(monkeypatch):
    fixture = _BridgeFixture(monkeypatch, hold_submit=True)
    direction_errors, close_errors = [], []

    def direction():
        try:
            fixture.bridge.run_direction(Language.RU, UUID(int=10))
        except BaseException as error:
            direction_errors.append(error)

    def close():
        try:
            fixture.bridge.close()
        except BaseException as error:
            close_errors.append(error)

    submitter, closer = (
        Thread(target=direction),
        Thread(target=close, name="fixture-close"),
    )
    try:
        submitter.start()
        assert fixture.submit_entered.wait(1) and fixture.entered.wait(1)
        closer.start()
        attempted = fixture.close_lock_attempted.wait(1)
        before_release = (
            fixture.shutdown_entered.is_set(),
            list(fixture.close_attempts),
            fixture.submissions[0].done(),
        )
        fixture.submit_release.set()
        submitter.join(1)
        fixture.release.set()
        closer.join(1)
        assert attempted and before_release == (False, [], False)
        assert not submitter.is_alive() and not closer.is_alive()
        assert len(direction_errors) == 1 and isinstance(
            direction_errors[0], TimeoutError
        )
        assert not close_errors and not fixture.fatal_calls
        assert fixture.finalized.is_set() and fixture.submissions[0].done()
        assert fixture.close_attempts == [True]
    finally:
        fixture.submit_release.set()
        fixture.release.set()
        for thread in (submitter, closer):
            if thread.ident is not None:
                thread.join(1)
        fixture.repair()


def test_bridge_joins_submissions_with_remaining_original_deadline(monkeypatch):
    fixture = _BridgeFixture(monkeypatch, elapsed_after_shutdown=29)
    try:
        with pytest.raises(TimeoutError, match="direction wait expired"):
            fixture.bridge.run_direction(Language.RU, UUID(int=9))
        fixture.bridge.close()
        assert fixture.waits == [("direction", 120), ("shutdown", 30), ("direction", 1)]
        assert fixture.join_waits == [1]
        assert fixture.finalized.is_set()
        assert not fixture.bridge._thread.is_alive() and not fixture.fatal_calls
        assert fixture.close_attempts == [True]
    finally:
        fixture.repair()


def test_bridge_thread_join_timeout_is_fatal_before_loop_close(monkeypatch):
    fixture = _BridgeFixture(
        monkeypatch, elapsed_after_shutdown=29, hold_thread_exit=True
    )
    try:
        with pytest.raises(_WorkerFatal):
            fixture.bridge.close()
        before_release = (
            fixture.thread_exit_entered.is_set(),
            fixture.bridge._thread.is_alive(),
            fixture.bridge._loop.is_running(),
            list(fixture.close_attempts),
        )
        assert before_release == (True, True, False, [])
        assert fixture.join_waits == [1]
        assert len(fixture.fatal_calls) == 1
        assert isinstance(fixture.fatal_calls[0], (RuntimeError, TimeoutError))
        assert fixture.waits == [("shutdown", 30)]
    finally:
        fixture.repair()


class _BridgeFixture:
    """Real loop, thread and submitted coroutine; repair is never the oracle."""

    def __init__(
        self,
        monkeypatch,
        *,
        shutdown_fails=False,
        elapsed_after_shutdown=None,
        hold_submit=False,
        hold_thread_exit=False,
    ):
        self.entered, self.release, self.finalized = Event(), Event(), Event()
        self.shutdown_entered = Event()
        self.failure = RuntimeError("synthetic shutdown failure")
        self.fatal_calls, self.close_attempts, self.submissions = [], [], []
        self.waits, self.elapsed = [], 0
        self.join_waits = []
        self.thread_exit_entered, self.thread_exit_release = Event(), Event()
        self.submit_entered, self.submit_release, self.close_lock_attempted = (
            Event(),
            Event(),
            Event(),
        )
        if hold_submit:

            class ObservedLock:
                def __init__(lock):
                    lock.actual = Lock()

                def __enter__(lock):
                    if current_thread().name == "fixture-close":
                        self.close_lock_attempted.set()
                    return lock.actual.__enter__()

                def __exit__(lock, *args):
                    return lock.actual.__exit__(*args)

            monkeypatch.setattr(task6_live, "Lock", ObservedLock, raising=False)
        if elapsed_after_shutdown is not None:
            monkeypatch.setattr(
                task6_live,
                "time",
                SimpleNamespace(
                    monotonic=lambda: self.elapsed,
                    monotonic_ns=lambda: self.elapsed * 1_000_000_000,
                ),
            )
        fixture = self

        class Provider:
            async def shutdown(self):
                fixture.shutdown_entered.set()
                if shutdown_fails:
                    raise fixture.failure

        if hold_thread_exit:
            original_run_loop = task6_live._ProviderDuplexBridge._run_loop

            def run_loop(bridge):
                original_run_loop(bridge)
                self.thread_exit_entered.set()
                assert self.thread_exit_release.wait(2)

            monkeypatch.setattr(task6_live._ProviderDuplexBridge, "_run_loop", run_loop)

        self.bridge = task6_live._ProviderDuplexBridge(
            Provider(), {}, fatal_cleanup=self.fatal
        )
        self.original_close = self.bridge._loop.close
        self.original_join = self.bridge._thread.join

        def join(timeout=None):
            self.join_waits.append(timeout)
            if hold_thread_exit:
                assert self.thread_exit_entered.wait(1)
                self.original_join(timeout=0.01)
                assert self.bridge._thread.is_alive()
                self.elapsed = 30
            else:
                self.original_join(timeout=timeout)

        monkeypatch.setattr(self.bridge._thread, "join", join)
        monkeypatch.setattr(
            self.bridge._loop, "close", lambda: self.close_attempts.append(True)
        )

        async def session(*_args):
            self.entered.set()
            try:
                while not self.release.is_set():
                    await asyncio.sleep(0.001)
                return 7.0
            finally:
                self.finalized.set()

        monkeypatch.setattr(self.bridge, "_run_session", session)
        original_submit = asyncio.run_coroutine_threadsafe

        def submit(coroutine, loop):
            future = original_submit(coroutine, loop)
            self.submissions.append(future)
            if coroutine.cr_code is session.__code__:
                original_result = future.result
                first = True

                def result(timeout=None):
                    nonlocal first
                    self.waits.append(("direction", timeout))
                    if first:
                        first = False
                        assert timeout == 120
                        assert self.entered.wait(1)
                        raise TimeoutError("synthetic direction wait expired")
                    if elapsed_after_shutdown is not None:
                        self.release.set()
                    return original_result(timeout=timeout)

                future.result = result
                future.cancel = lambda: pytest.fail("private submission cancelled")
                if hold_submit:
                    self.submit_entered.set()
                    assert self.submit_release.wait(2)
            elif elapsed_after_shutdown is not None:
                original_result = future.result

                def result(timeout=None):
                    self.waits.append(("shutdown", timeout))
                    value = original_result(timeout=timeout)
                    self.elapsed = elapsed_after_shutdown
                    return value

                future.result = result
            return future

        monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", submit)

    def fatal(self, error):
        self.fatal_calls.append(error)
        raise _WorkerFatal()

    def repair(self):
        self.submit_release.set()
        self.release.set()
        self.thread_exit_release.set()
        loop, thread = self.bridge._loop, self.bridge._thread
        if thread.is_alive():
            if loop.is_running():
                asyncio.run_coroutine_threadsafe(asyncio.sleep(0.02), loop).result(1)
                loop.call_soon_threadsafe(loop.stop)
            self.original_join(1)
        assert not thread.is_alive()
        pending = asyncio.all_tasks(loop)
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        self.original_close()


def test_bridge_failed_shutdown_keeps_loop_for_worker_fatal(monkeypatch):
    fixture = _BridgeFixture(monkeypatch, shutdown_fails=True)
    try:
        try:
            fixture.bridge.close()
        except BaseException as error:
            outcome = error
        else:
            outcome = None
        before_repair = (
            list(fixture.close_attempts),
            fixture.bridge._thread.is_alive(),
            fixture.bridge._loop.is_closed(),
        )
        assert fixture.shutdown_entered.is_set()
        assert before_repair == ([], True, False)
        assert isinstance(outcome, _WorkerFatal)
        assert fixture.fatal_calls == [fixture.failure]
    finally:
        fixture.repair()


def test_bridge_direction_timeout_retains_finalizer_before_loop_close(monkeypatch):
    fixture = _BridgeFixture(monkeypatch)
    finished, errors = Event(), []

    def close():
        try:
            fixture.bridge.close()
        except BaseException as error:
            errors.append(error)
        finally:
            finished.set()

    closer = Thread(target=close)
    try:
        with pytest.raises(TimeoutError, match="direction wait expired"):
            fixture.bridge.run_direction(Language.RU, UUID(int=7))
        direction = fixture.submissions[0]
        assert not direction.done() and fixture.entered.is_set()
        closer.start()
        assert fixture.shutdown_entered.wait(1)
        admitted = len(fixture.submissions)
        with pytest.raises(RuntimeError):
            fixture.bridge.run_direction(Language.EN, UUID(int=8))
        assert len(fixture.submissions) == admitted
        finished.wait(0.05)
        before_release = (
            finished.is_set(),
            list(fixture.close_attempts),
            fixture.bridge._thread.is_alive(),
            fixture.finalized.is_set(),
            direction.done(),
        )
        fixture.release.set()
        closer.join(1)
        assert not closer.is_alive()
        assert before_release == (False, [], True, False, False)
        assert fixture.finalized.is_set() and direction.done()
        assert direction.result(1) == 7.0
        assert not errors and not fixture.fatal_calls
        assert fixture.close_attempts == [True]
    finally:
        fixture.release.set()
        if closer.ident is not None:
            closer.join(1)
        fixture.repair()


def _owned_run_fixture(tmp_path, monkeypatch, failure=None):
    created = []
    duplex_calls = []
    published_before_cleanup = []
    events, fatal_calls = [], []
    terminal_error = RuntimeError("terminal provider failure")

    def fatal(error):
        fatal_calls.append(error)
        raise _WorkerFatal()

    models = {
        model_id: SimpleNamespace(
            id=model_id,
            cache_path=tmp_path / model_id,
            files=[SimpleNamespace(path="voice.onnx")],
        )
        for model_id in (
            task6_live._SMALL_ID,
            task6_live._LARGE_ID,
            task6_live._MT_ID,
            *task6_live._VOICE_IDS.values(),
        )
    }
    manifest = SimpleNamespace(models=models, resolve_runtime_file=lambda *_args: None)

    class Adapter:
        actual_device = "cpu"
        degraded = False

        def __init__(self, *_args, selected_id=None, **kwargs):
            self.closed = False
            self.close_count = 0
            self.resident_model_id = selected_id
            created.append(self)
            events.append(("create", selected_id, self))

        def check(self):
            assert not self.closed, "benchmark reused a closed model"

        def close(self):
            self.close_count += 1
            if (tmp_path / "out.json").exists():
                published_before_cleanup.append(self)
            self.closed = True
            self.resident_model_id = None

        def release(self):
            self.close()
            return True

    class Translator(Adapter):
        loads = 0

        @classmethod
        def load(cls, *_args, **kwargs):
            cls.loads += 1
            if failure == "second-mt" and cls.loads == 2:
                raise RuntimeError("second-mt")
            return cls()

    @dataclass
    class Report:
        model_id: str

    def candidate(config, *, adapter_factory, **kwargs):
        adapter_factory()
        return Report(config.model_id)

    def synthesize(tts, *_args):
        tts.check()
        return b"\0\0" * 160

    def quality(*_args, translator, **kwargs):
        translator.check()
        if failure == "quality":
            raise RuntimeError("quality")
        return "quality-report"

    def duplex(
        *,
        asr,
        translator,
        tts,
        model_id,
        resources,
        fatal_cleanup,
        on_complete=None,
        **kwargs,
    ):
        resources.pop_all()
        for adapter in (asr, translator, tts):
            adapter.check()
        duplex_calls.append((translator, tts))
        try:
            if failure == "duplex":
                raise RuntimeError("duplex")
            if on_complete is not None:
                on_complete()
            return Report(model_id)
        finally:
            if failure == "terminal":
                fatal_cleanup(terminal_error)
            for adapter in (asr, translator, tts):
                adapter.close()
            events.append(("terminal", model_id))

    monkeypatch.setattr(task6_live, "load_manifest", lambda *_args: manifest)
    monkeypatch.setattr(task6_live, "_cuda_available", lambda: False)
    monkeypatch.setattr(task6_live, "PiperVoiceRegistry", lambda *_args: object())
    monkeypatch.setattr(task6_live, "PiperTts", Adapter)
    monkeypatch.setattr(task6_live, "_run_voice_smokes", lambda *_args: [])
    monkeypatch.setattr(task6_live, "_synthesize_pcm", synthesize)
    monkeypatch.setattr(task6_live, "_asr_manager", Adapter)
    monkeypatch.setattr(task6_live, "benchmark_asr_candidate", candidate)
    monkeypatch.setattr(task6_live, "NllbTranslator", Translator)
    monkeypatch.setattr(task6_live, "load_quality_corpus", lambda *_args: object())
    monkeypatch.setattr(task6_live, "run_quality_benchmark", quality)
    monkeypatch.setattr(task6_live, "_benchmark_provider_duplex", duplex)
    monkeypatch.setattr(task6_live, "_resource_sample", lambda: (0, 0, 0, 0))
    monkeypatch.setattr(
        task6_live,
        "_build_payload",
        lambda **kwargs: {"normal_runtime": kwargs["normal_runtime"]},
    )
    return SimpleNamespace(
        run=lambda: task6_live._run_owned(fatal_cleanup=fatal),
        created=created,
        duplex_calls=duplex_calls,
        events=events,
        published_before_cleanup=published_before_cleanup,
        fatal_calls=fatal_calls,
        terminal_error=terminal_error,
    )


@pytest.mark.parametrize("failure", [None, "quality", "second-mt", "duplex"])
def test_live_run_owns_model_lifetimes_and_never_reuses_closed_adapters(
    tmp_path, monkeypatch, failure
):
    fixture = _owned_run_fixture(tmp_path, monkeypatch, failure)
    if failure is not None:
        with pytest.raises(RuntimeError, match=failure):
            fixture.run()
    else:
        payload = fixture.run()
        assert payload["normal_runtime"]["resident_model_id"] == task6_live._SMALL_ID
        assert len(fixture.duplex_calls) == 2
        assert fixture.duplex_calls[0][0] is not fixture.duplex_calls[1][0]
        assert fixture.duplex_calls[0][1] is not fixture.duplex_calls[1][1]
    assert fixture.created and all(adapter.closed for adapter in fixture.created)
    assert [adapter.close_count for adapter in fixture.created] == [1] * len(
        fixture.created
    )
    assert not fixture.published_before_cleanup and not (tmp_path / "out.json").exists()
    assert not fixture.fatal_calls


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
        (Language.RU, "do-not-mute:10:00"): ("Не заглушай микрофон до 10:00."),
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
        if current_thread().name == "translator-resource-sampler" and is_active:
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
    assert serialized["duplex_candidates"][0]["model_id"] == ("faster-whisper-small")
    assert serialized["duplex_candidates"][0]["vram_within_budget"] is True
    assert serialized["duplex_candidates"][1]["model_id"] == ("faster-whisper-large-v3")
    assert serialized["duplex_candidates"][1]["vram_within_budget"] is False


def test_task6_parser_publishes_only_hash_bound_synthetic_summary(
    tmp_path: Path,
) -> None:
    private_marker = "synthetic-private-human-review-text"
    payload = {
        "schema_version": "translator.task6-benchmark.v2",
        "quality": {
            "quality": {
                "corpus_id": "synthetic-task6-corpus",
                "passes_thresholds": True,
            },
            "review_rows": [{"source_text": private_marker}],
        },
    }
    path = tmp_path / "synthetic-task6-results.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    public_summary = load_task6_quality_evidence(path).to_report_dict()
    expected_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

    assert public_summary == {
        "schema_version": "translator.task6-benchmark.v2",
        "sha256": expected_sha256,
        "corpus_id": "synthetic-task6-corpus",
        "passes_thresholds": True,
    }
    assert set(public_summary) == {
        "schema_version",
        "sha256",
        "corpus_id",
        "passes_thresholds",
    }
    assert private_marker not in json.dumps(public_summary)


def test_live_releases_quality_asr_before_creating_normal_residency(
    monkeypatch, tmp_path
):
    fixture = _owned_run_fixture(tmp_path, monkeypatch)
    fixture.run()
    terminal = fixture.events.index(("terminal", task6_live._LARGE_ID))
    small_creations = [
        index
        for index, event in enumerate(fixture.events)
        if event[:2] == ("create", task6_live._SMALL_ID)
    ]
    assert len(small_creations) == 2
    assert small_creations[0] < terminal < small_creations[1]
    assert all(adapter.close_count == 1 for adapter in fixture.created)


def test_live_duplex_captures_residency_before_provider_shutdown(monkeypatch):
    events = []

    class Bridge:
        def __init__(self, *_args, fatal_cleanup):
            pass

        def run_direction(self, *_args):
            return 0

        def close(self):
            events.append("shutdown")

    def benchmark(*_args, **kwargs):
        events.append("measured")
        return "report"

    monkeypatch.setattr(task6_live, "LocalProvider", lambda **kwargs: object())
    monkeypatch.setattr(task6_live, "InferenceScheduler", lambda: object())
    monkeypatch.setattr(task6_live, "_ProviderDuplexBridge", Bridge)
    monkeypatch.setattr(task6_live, "benchmark_simultaneous_duplex", benchmark)
    result = task6_live._benchmark_provider_duplex(
        asr=object(),
        model_id=task6_live._SMALL_ID,
        translator=object(),
        tts=object(),
        source_pcm={},
        device="cpu",
        resources=ExitStack(),
        fatal_cleanup=lambda error: pytest.fail("unexpected fatal cleanup"),
        on_complete=lambda: events.append("snapshot"),
    )
    assert result == "report"
    assert events == ["measured", "snapshot", "shutdown"]


def test_live_never_creates_normal_asr_when_quality_release_fails(
    monkeypatch, tmp_path
):
    fixture = _owned_run_fixture(tmp_path, monkeypatch, "terminal")
    with pytest.raises(_WorkerFatal):
        fixture.run()
    assert fixture.fatal_calls == [fixture.terminal_error]
    assert (
        len(
            [
                event
                for event in fixture.events
                if event[:2] == ("create", task6_live._SMALL_ID)
            ]
        )
        == 1
    )
    assert not any(event[0] == "terminal" for event in fixture.events)
    transferred = [adapter for adapter in fixture.created if not adapter.closed]
    assert len(transferred) == 3
    assert all(adapter.close_count == 0 for adapter in transferred)
    assert all(
        adapter.close_count == 1 for adapter in fixture.created if adapter.closed
    )
    assert not (tmp_path / "out.json").exists()


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


def _telemetry_vendor(monkeypatch, *, fault=None, count=1, used=2_048 * 2**20):
    events = []

    class VendorError(Exception):
        pass

    def invoke(name, value=None):
        events.append(name)
        if name == fault:
            raise VendorError("synthetic-private-vendor-detail")
        return value

    handle = object()

    def get_handle(index):
        assert index == 0
        return invoke("handle", handle)

    def read(name, actual_handle, value):
        assert actual_handle is handle
        return invoke(name, value)

    vendor = SimpleNamespace(
        NVMLError=VendorError,
        nvmlInit=lambda: invoke("init"),
        nvmlShutdown=lambda: invoke("shutdown"),
        nvmlDeviceGetCount=lambda: invoke("count", count),
        nvmlDeviceGetHandleByIndex=get_handle,
        nvmlDeviceGetUtilizationRates=lambda device: read(
            "utilization", device, SimpleNamespace(gpu=42)
        ),
        nvmlDeviceGetMemoryInfo=lambda device: read(
            "memory", device, SimpleNamespace(used=used)
        ),
    )
    monkeypatch.setattr(task6_live, "pynvml", vendor, raising=False)

    def legacy_cli(*args, **kwargs):
        events.append("subprocess")
        return SimpleNamespace(stdout="42, 2048\n")

    monkeypatch.setattr(subprocess, "run", legacy_cli)
    return vendor, events


@pytest.mark.parametrize(
    "fault", ["init", "count", "handle", "utilization", "memory", "shutdown"]
)
def test_live_resource_telemetry_fails_closed(monkeypatch, fault) -> None:
    _vendor, events = _telemetry_vendor(monkeypatch, fault=fault)

    with pytest.raises(
        task6_live.ResourceTelemetryError,
        match="GPU telemetry is unavailable",
    ) as error:
        task6_live._resource_sample()
    assert "synthetic-private" not in str(error.value)
    calls = ["init", "count", "handle", "utilization", "memory", "shutdown"]
    expected = calls[: calls.index(fault) + 1]
    if fault not in {"init", "shutdown"}:
        expected.append("shutdown")
    assert events == expected


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

    monkeypatch.setattr(task6_live, "_PROCESS", FakeProcess())
    _vendor, events = _telemetry_vendor(monkeypatch)

    for _ in range(2):
        assert task6_live._resource_sample() == (37.5, 123_456, 42.0, 2_048)
    assert (
        events == ["init", "count", "handle", "utilization", "memory", "shutdown"] * 2
    )


def test_live_resource_telemetry_preserves_memory_when_utilization_is_unsupported(
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

    monkeypatch.setattr(task6_live, "_PROCESS", FakeProcess())
    vendor, events = _telemetry_vendor(monkeypatch)

    class UtilizationNotSupported(vendor.NVMLError):
        pass

    vendor.NVMLError_NotSupported = UtilizationNotSupported

    def unsupported(_device):
        events.append("utilization")
        raise UtilizationNotSupported()

    monkeypatch.setattr(vendor, "nvmlDeviceGetUtilizationRates", unsupported)

    assert task6_live._resource_sample() == (37.5, 123_456, None, 2_048)
    assert events == ["init", "count", "handle", "utilization", "memory", "shutdown"]


def test_resource_peaks_retains_unknown_gpu_utilization() -> None:
    assert _resource_peaks(((10.0, 100, None, 1_000), (20.0, 200, None, 2_000))) == (
        20.0,
        200,
        None,
        2_000,
    )


def test_resource_peaks_uses_available_gpu_utilization_samples() -> None:
    assert _resource_peaks(((10.0, 100, None, 1_000), (20.0, 200, 42.0, 2_000))) == (
        20.0,
        200,
        42.0,
        2_000,
    )


@pytest.mark.parametrize("count", [0, 2])
def test_live_resource_telemetry_requires_one_device(monkeypatch, count):
    _vendor, events = _telemetry_vendor(monkeypatch, count=count)
    with pytest.raises(task6_live.ResourceTelemetryError, match="unavailable"):
        task6_live._resource_sample()
    assert events == ["init", "count", "shutdown"]


@pytest.mark.parametrize("field", ["gpu", "used"])
@pytest.mark.parametrize("value", [None, "not-a-number", "missing-attribute"])
def test_live_resource_telemetry_rejects_unavailable_values(monkeypatch, field, value):
    vendor, events = _telemetry_vendor(monkeypatch)
    result = (
        SimpleNamespace()
        if value == "missing-attribute"
        else SimpleNamespace(**{field: value})
    )
    method = (
        "nvmlDeviceGetUtilizationRates" if field == "gpu" else "nvmlDeviceGetMemoryInfo"
    )
    original = getattr(vendor, method)

    def unavailable(handle):
        original(handle)
        return result

    monkeypatch.setattr(vendor, method, unavailable)
    with pytest.raises(task6_live.ResourceTelemetryError, match="unavailable"):
        task6_live._resource_sample()
    assert events[0] == "init" and events[-1] == "shutdown"
    assert events.count("shutdown") == 1 and "subprocess" not in events


@pytest.mark.parametrize("remainder", [0, 1, 2**20 - 1])
def test_live_resource_telemetry_converts_binary_mib(monkeypatch, remainder):
    _vendor, events = _telemetry_vendor(monkeypatch, used=2_048 * 2**20 + remainder)
    assert task6_live._resource_sample()[3] == 2_048
    assert events == ["init", "count", "handle", "utilization", "memory", "shutdown"]


def test_live_resource_telemetry_retains_held_vendor_call(monkeypatch):
    vendor, events = _telemetry_vendor(monkeypatch)
    entered, release, second_lock_attempted = Event(), Event(), Event()
    original = vendor.nvmlDeviceGetUtilizationRates
    original_lock = task6_live._RESOURCE_SAMPLE_LOCK

    class ObservedLock:
        def __enter__(self):
            if entered.is_set():
                second_lock_attempted.set()
            return original_lock.__enter__()

        def __exit__(self, *args):
            return original_lock.__exit__(*args)

    def held(handle):
        result = original(handle)
        if not entered.is_set():
            entered.set()
            assert release.wait(timeout=2)
        return result

    monkeypatch.setattr(task6_live, "_RESOURCE_SAMPLE_LOCK", ObservedLock())
    monkeypatch.setattr(vendor, "nvmlDeviceGetUtilizationRates", held)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(task6_live._resource_sample)
        second = None
        try:
            assert entered.wait(timeout=1)
            second = executor.submit(task6_live._resource_sample)
            assert second_lock_attempted.wait(timeout=1)
            assert not first.done() and not second.done()
            assert events == ["init", "count", "handle", "utilization"]
            release.set()
            assert first.result(timeout=1)[2:] == (42.0, 2_048)
            assert second.result(timeout=1)[2:] == (42.0, 2_048)
            assert (
                events
                == ["init", "count", "handle", "utilization", "memory", "shutdown"] * 2
            )
        finally:
            release.set()
            first.result(timeout=2)
            if second is not None:
                second.result(timeout=2)

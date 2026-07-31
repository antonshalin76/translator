from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
from threading import Event, Lock
import traceback
from uuid import uuid4

import pytest

from translator_sidecar.local.source_commit import (
    SourceCommit,
    SourceCommitProtocolError,
    SourceCommitUnavailable,
)


@pytest.mark.parametrize(
    "hypotheses",
    [
        ["private-repeat-A", "private-repeat-A", "private-repeat-B"],
        ["private-negation-can-go", "private-negation-cannot-go"],
        ["private-number-fifteen", "private-number-fifty"],
        ["private-punctuation-hello", "private-punctuation-hello-comma"],
        ["private-order-one-two", "private-order-two-one"],
    ],
)
def test_hypotheses_are_never_available_to_tts_before_eou(
    hypotheses: list[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    commit = SourceCommit(uuid4())
    tts_calls: list[str] = []
    caplog.set_level(logging.DEBUG)

    for hypothesis in hypotheses:
        commit.observe_asr_hypothesis(hypothesis)
        commit.observe_translation_hypothesis(hypothesis)
        with pytest.raises(SourceCommitProtocolError, match="not final"):
            commit.synthesize_once(tts_calls.append)

    assert tts_calls == []
    assert commit.committed_translation is None
    for hypothesis in hypotheses:
        assert hypothesis not in caplog.text

    translation_calls: list[str] = []

    def translate(source: str) -> str:
        translation_calls.append(source)
        return "committed final translation"

    commit.finalize(
        "distinct final source",
        end_of_utterance=True,
        translate=translate,
    )
    commit.synthesize_once(tts_calls.append)
    assert translation_calls == ["distinct final source"]
    assert tts_calls == ["committed final translation"]
    assert "distinct final source" not in caplog.text
    assert "committed final translation" not in caplog.text


def test_eou_commits_one_final_translation_and_synthesizes_once() -> None:
    utterance_id = uuid4()
    commit = SourceCommit(utterance_id)
    translation_calls: list[str] = []
    tts_calls: list[str] = []

    def translate(source: str) -> str:
        translation_calls.append(source)
        return "final translation"

    assert commit.utterance_id == utterance_id
    assert (
        commit.finalize(
            "final source",
            end_of_utterance=True,
            translate=translate,
        )
        == "final translation"
    )
    assert commit.committed_translation == "final translation"
    assert commit.synthesize_once(tts_calls.append) is None

    assert translation_calls == ["final source"]
    assert tts_calls == ["final translation"]
    with pytest.raises(SourceCommitProtocolError, match="consumed"):
        commit.synthesize_once(tts_calls.append)
    with pytest.raises(SourceCommitProtocolError, match="final"):
        commit.finalize(
            "revision",
            end_of_utterance=True,
            translate=translate,
        )
    assert translation_calls == ["final source"]
    assert tts_calls == ["final translation"]


def test_finalize_without_eou_never_calls_translation(
    caplog: pytest.LogCaptureFixture,
) -> None:
    source_marker = "private active speech marker"
    calls = 0
    caplog.set_level(logging.DEBUG)

    def translate(_source: str) -> str:
        nonlocal calls
        calls += 1
        return "translation"

    commit = SourceCommit(uuid4())
    with pytest.raises(SourceCommitProtocolError, match="end_of_utterance"):
        commit.finalize(
            source_marker,
            end_of_utterance=False,
            translate=translate,
        )
    assert calls == 0
    assert commit.committed_translation is None
    assert source_marker not in caplog.text


@pytest.mark.parametrize("after_finalize", [False, True])
def test_cancel_purges_text_and_blocks_translation_or_tts(
    after_finalize: bool,
    caplog: pytest.LogCaptureFixture,
) -> None:
    partial_source = "private cancel partial source marker"
    partial_translation = "private cancel partial translation marker"
    final_source = "private cancel final source marker"
    committed_translation = "private cancel committed translation marker"
    late_source = "private cancel late source marker"
    commit = SourceCommit(uuid4())
    translation_calls = 0
    tts_calls = 0
    caplog.set_level(logging.DEBUG)

    def translate(_source: str) -> str:
        nonlocal translation_calls
        translation_calls += 1
        return committed_translation

    def synthesize(_translation: str) -> None:
        nonlocal tts_calls
        tts_calls += 1

    commit.observe_asr_hypothesis(partial_source)
    commit.observe_translation_hypothesis(partial_translation)
    if after_finalize:
        commit.finalize(
            final_source,
            end_of_utterance=True,
            translate=translate,
        )
    commit.cancel()

    assert commit.committed_translation is None
    with pytest.raises(SourceCommitProtocolError, match="cancelled"):
        commit.finalize(
            late_source,
            end_of_utterance=True,
            translate=translate,
        )
    with pytest.raises(SourceCommitProtocolError, match="cancelled"):
        commit.synthesize_once(synthesize)
    assert translation_calls == int(after_finalize)
    assert tts_calls == 0
    for marker in (
        partial_source,
        partial_translation,
        final_source,
        committed_translation,
        late_source,
    ):
        assert marker not in caplog.text


def test_concurrent_finalize_calls_translate_once() -> None:
    commit = SourceCommit(uuid4())
    callback_entered = Event()
    release_callback = Event()
    calls = 0
    calls_lock = Lock()

    def translate(_source: str) -> str:
        nonlocal calls
        with calls_lock:
            calls += 1
        callback_entered.set()
        assert release_callback.wait(timeout=2)
        return "translation"

    def finalize() -> str:
        return commit.finalize(
            "source",
            end_of_utterance=True,
            translate=translate,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(finalize)
        assert callback_entered.wait(timeout=2)
        duplicate = pool.submit(finalize)
        with pytest.raises(SourceCommitProtocolError, match="final"):
            duplicate.result(timeout=1)
        release_callback.set()
        assert first.result(timeout=3) == "translation"
    assert calls == 1


def test_concurrent_synthesis_calls_tts_once() -> None:
    commit = SourceCommit(uuid4())
    commit.finalize(
        "source",
        end_of_utterance=True,
        translate=lambda _source: "translation",
    )
    callback_entered = Event()
    release_callback = Event()
    calls = 0
    calls_lock = Lock()

    def synthesize(_translation: str) -> str:
        nonlocal calls
        with calls_lock:
            calls += 1
        callback_entered.set()
        assert release_callback.wait(timeout=2)
        return "audio"

    def consume() -> str:
        return commit.synthesize_once(synthesize)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(consume)
        assert callback_entered.wait(timeout=2)
        duplicate = pool.submit(consume)
        with pytest.raises(SourceCommitProtocolError, match="consumed"):
            duplicate.result(timeout=1)
        release_callback.set()
        assert first.result(timeout=3) == "audio"
    assert calls == 1


@pytest.mark.parametrize("failure_at", ["translation", "synthesis"])
def test_native_failure_is_sanitized_and_never_retried(
    failure_at: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    marker = f"private {failure_at} marker"
    commit = SourceCommit(uuid4())
    calls = 0
    caplog.set_level(logging.DEBUG)

    def fail(_text: str) -> str:
        nonlocal calls
        calls += 1
        raise RuntimeError(marker)

    if failure_at == "translation":
        def operation() -> str:
            return commit.finalize(
                "private source",
                end_of_utterance=True,
                translate=fail,
            )
    else:
        commit.finalize(
            "source",
            end_of_utterance=True,
            translate=lambda _source: "private translation",
        )

        def operation() -> str:
            return commit.synthesize_once(fail)

    with pytest.raises(SourceCommitUnavailable, match="unavailable") as raised:
        operation()
    with pytest.raises(SourceCommitProtocolError, match="failed"):
        operation()
    assert calls == 1
    assert commit.committed_translation is None
    rendered = "".join(
        traceback.format_exception(
            type(raised.value), raised.value, raised.value.__traceback__
        )
    )
    assert marker not in rendered
    assert "private source" not in rendered
    assert "private translation" not in rendered
    assert marker not in caplog.text
    assert "private source" not in caplog.text
    assert "private translation" not in caplog.text


def test_async_callbacks_are_owned_exactly_once_by_commit_boundary() -> None:
    async def scenario() -> None:
        commit = SourceCommit(uuid4())
        translate_calls: list[str] = []
        synthesize_calls: list[str] = []

        async def translate(source: str) -> str:
            translate_calls.append(source)
            return "stable translation"

        async def synthesize(text: str):
            synthesize_calls.append(text)
            yield b"frame-1"
            yield b"frame-2"

        translated = await commit.finalize_async(
            "final source",
            end_of_utterance=True,
            translate=translate,
        )
        frames = [
            frame
            async for frame in commit.stream_once(synthesize)
        ]

        assert translated == "stable translation"
        assert frames == [b"frame-1", b"frame-2"]
        assert translate_calls == ["final source"]
        assert synthesize_calls == ["stable translation"]
        with pytest.raises(
            SourceCommitProtocolError,
            match="consumed",
        ):
            async for _frame in commit.stream_once(synthesize):
                pass

    asyncio.run(scenario())


def test_concurrent_async_callbacks_enter_once() -> None:
    async def scenario() -> None:
        commit = SourceCommit(uuid4())
        translate_entered = asyncio.Event()
        release_translate = asyncio.Event()
        translate_calls = 0

        async def translate(_source: str) -> str:
            nonlocal translate_calls
            translate_calls += 1
            translate_entered.set()
            await release_translate.wait()
            return "stable translation"

        first_finalize = asyncio.create_task(
            commit.finalize_async(
                "source",
                end_of_utterance=True,
                translate=translate,
            )
        )
        await asyncio.wait_for(translate_entered.wait(), timeout=1)
        with pytest.raises(
            SourceCommitProtocolError,
            match="final",
        ):
            await commit.finalize_async(
                "source",
                end_of_utterance=True,
                translate=translate,
            )
        release_translate.set()
        assert await first_finalize == "stable translation"
        assert translate_calls == 1

        synthesis_entered = asyncio.Event()
        release_synthesis = asyncio.Event()
        synthesis_calls = 0

        async def synthesize(_text: str):
            nonlocal synthesis_calls
            synthesis_calls += 1
            synthesis_entered.set()
            await release_synthesis.wait()
            yield b"frame"

        async def consume() -> list[bytes]:
            return [
                frame
                async for frame in commit.stream_once(synthesize)
            ]

        first_synthesis = asyncio.create_task(consume())
        await asyncio.wait_for(synthesis_entered.wait(), timeout=1)
        with pytest.raises(
            SourceCommitProtocolError,
            match="consumed",
        ):
            await consume()
        release_synthesis.set()
        assert await first_synthesis == [b"frame"]
        assert synthesis_calls == 1

    asyncio.run(scenario())

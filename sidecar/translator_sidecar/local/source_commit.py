"""Single-utterance final-source commit boundary."""

from __future__ import annotations

from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
)
from enum import Enum
from threading import Lock
from typing import TypeVar
from uuid import UUID


class SourceCommitProtocolError(RuntimeError):
    """The caller attempted an invalid source-commit transition."""


class SourceCommitUnavailable(RuntimeError):
    """A native final-stage callback failed without exposing text."""


class _State(str, Enum):
    ACTIVE = "active"
    FINALIZING = "finalizing"
    FINAL = "final"
    CONSUMING = "consuming"
    CONSUMED = "consumed"
    CANCELLED = "cancelled"
    FAILED = "failed"


_Result = TypeVar("_Result")
_SOURCE_COMMIT_UNAVAILABLE_MESSAGE = "source commit is unavailable"


class SourceCommit:
    """Own the immutable translation boundary for one wire utterance."""

    def __init__(self, utterance_id: UUID) -> None:
        self._utterance_id = utterance_id
        self._state = _State.ACTIVE
        self._committed_translation: str | None = None
        self._lock = Lock()

    @property
    def utterance_id(self) -> UUID:
        return self._utterance_id

    @property
    def committed_translation(self) -> str | None:
        with self._lock:
            if self._state is not _State.FINAL:
                return None
            return self._committed_translation

    def observe_asr_hypothesis(self, _text: str) -> None:
        self._observe_hypothesis()

    def observe_translation_hypothesis(self, _text: str) -> None:
        self._observe_hypothesis()

    def _observe_hypothesis(self) -> None:
        with self._lock:
            if self._state is _State.ACTIVE:
                return
            self._raise_for_state()

    def finalize(
        self,
        source_text: str,
        *,
        end_of_utterance: bool,
        translate: Callable[[str], str],
    ) -> str:
        with self._lock:
            if self._state is not _State.ACTIVE:
                self._raise_for_state(finalize=True)
            if not end_of_utterance:
                raise SourceCommitProtocolError(
                    "end_of_utterance is required for final source commit"
                )
            if not source_text.strip():
                raise SourceCommitUnavailable(_SOURCE_COMMIT_UNAVAILABLE_MESSAGE)
            self._state = _State.FINALIZING

        try:
            translation = translate(source_text).strip()
        except Exception:
            self._fail_if_in_progress(_State.FINALIZING)
            raise SourceCommitUnavailable(_SOURCE_COMMIT_UNAVAILABLE_MESSAGE) from None
        if not translation:
            self._fail_if_in_progress(_State.FINALIZING)
            raise SourceCommitUnavailable(_SOURCE_COMMIT_UNAVAILABLE_MESSAGE)

        with self._lock:
            if self._state is not _State.FINALIZING:
                self._raise_for_state(finalize=True)
            self._committed_translation = translation
            self._state = _State.FINAL
            return translation

    async def finalize_async(
        self,
        source_text: str,
        *,
        end_of_utterance: bool,
        translate: Callable[[str], Awaitable[str]],
    ) -> str:
        with self._lock:
            if self._state is not _State.ACTIVE:
                self._raise_for_state(finalize=True)
            if not end_of_utterance:
                raise SourceCommitProtocolError(
                    "end_of_utterance is required for final source commit"
                )
            if not source_text.strip():
                raise SourceCommitUnavailable(_SOURCE_COMMIT_UNAVAILABLE_MESSAGE)
            self._state = _State.FINALIZING

        try:
            translation = (await translate(source_text)).strip()
        except Exception:
            self._fail_if_in_progress(_State.FINALIZING)
            raise SourceCommitUnavailable(_SOURCE_COMMIT_UNAVAILABLE_MESSAGE) from None
        if not translation:
            self._fail_if_in_progress(_State.FINALIZING)
            raise SourceCommitUnavailable(_SOURCE_COMMIT_UNAVAILABLE_MESSAGE)

        with self._lock:
            if self._state is not _State.FINALIZING:
                self._raise_for_state(finalize=True)
            self._committed_translation = translation
            self._state = _State.FINAL
            return translation

    def synthesize_once(
        self,
        synthesize: Callable[[str], _Result],
    ) -> _Result:
        with self._lock:
            if self._state is not _State.FINAL:
                self._raise_for_state()
            translation = self._committed_translation
            if translation is None:
                self._state = _State.FAILED
                raise SourceCommitUnavailable(_SOURCE_COMMIT_UNAVAILABLE_MESSAGE)
            self._committed_translation = None
            self._state = _State.CONSUMING

        try:
            result = synthesize(translation)
        except Exception:
            self._fail_if_in_progress(_State.CONSUMING)
            raise SourceCommitUnavailable(_SOURCE_COMMIT_UNAVAILABLE_MESSAGE) from None

        with self._lock:
            if self._state is not _State.CONSUMING:
                self._raise_for_state()
            self._state = _State.CONSUMED
            return result

    async def stream_once(
        self,
        synthesize: Callable[[str], AsyncIterator[_Result]],
    ) -> AsyncIterator[_Result]:
        with self._lock:
            if self._state is not _State.FINAL:
                self._raise_for_state()
            translation = self._committed_translation
            if translation is None:
                self._state = _State.FAILED
                raise SourceCommitUnavailable(_SOURCE_COMMIT_UNAVAILABLE_MESSAGE)
            self._committed_translation = None
            self._state = _State.CONSUMING

        try:
            async for item in synthesize(translation):
                yield item
        except BaseException:
            self._fail_if_in_progress(_State.CONSUMING)
            raise

        with self._lock:
            if self._state is not _State.CONSUMING:
                self._raise_for_state()
            self._state = _State.CONSUMED

    def cancel(self) -> None:
        with self._lock:
            if self._state is _State.CANCELLED:
                raise SourceCommitProtocolError("source commit is already cancelled")
            self._committed_translation = None
            self._state = _State.CANCELLED

    def _fail_if_in_progress(self, expected: _State) -> None:
        with self._lock:
            if self._state is expected:
                self._committed_translation = None
                self._state = _State.FAILED

    def _raise_for_state(self, *, finalize: bool = False) -> None:
        if self._state is _State.ACTIVE:
            raise SourceCommitProtocolError("source is not final")
        if self._state is _State.CANCELLED:
            raise SourceCommitProtocolError("source commit is cancelled")
        if self._state is _State.FAILED:
            raise SourceCommitProtocolError("source commit failed")
        if finalize:
            raise SourceCommitProtocolError("source commit is already final")
        if self._state in {_State.CONSUMING, _State.CONSUMED}:
            raise SourceCommitProtocolError(
                "committed translation was already consumed"
            )
        raise SourceCommitProtocolError("source commit is already final")

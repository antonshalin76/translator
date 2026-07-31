"""OpenAI Realtime Translation provider boundary.

This module is intentionally offline-safe: preflight verifies consent and
credential availability, but does not open a network session.
"""

from __future__ import annotations

import base64
import binascii
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from .provider_contract import (
    SAFE_ERROR_MESSAGES,
    ComputeDevice,
    ModelHealth,
    ModelKind,
    ModelState,
    OpenProviderSession,
    PcmFormat,
    PrivacySafeProviderError,
    ProviderAudioDelta,
    ProviderCapabilities,
    ProviderHealth,
    ProviderId,
    ProviderQueues,
    ProviderSessionOpened,
    ProviderState,
    ProviderTranscriptDelta,
    ProviderTranslationDelta,
    SafeErrorCode,
    SafeErrorSummary,
    SampleFormat,
    make_provider_error,
)


OPENAI_REALTIME_TRANSLATION_MODEL = "gpt-realtime-translate"
OPENAI_REALTIME_TRANSLATION_ENDPOINT = (
    "wss://api.openai.com/v1/realtime/translations"
)
OPENAI_PROVIDER_NAME = "openai-realtime-translation"
_DEFAULT_CREDENTIAL_ENV = "OPENAI_API_KEY"
_DEFAULT_FRAME_DURATION_MS = 20


@dataclass(frozen=True, slots=True)
class OpenAIRealtimeConfig:
    cloud_opt_in: bool
    credential_env_name: str = _DEFAULT_CREDENTIAL_ENV
    model: str = OPENAI_REALTIME_TRANSLATION_MODEL
    endpoint: str = OPENAI_REALTIME_TRANSLATION_ENDPOINT

    def __post_init__(self) -> None:
        if not self.credential_env_name:
            raise ValueError("credential_env_name_required")
        if not self.model:
            raise ValueError("model_required")
        if self.endpoint != OPENAI_REALTIME_TRANSLATION_ENDPOINT:
            raise ValueError("translation_endpoint_must_be_official")


@dataclass(frozen=True, slots=True)
class OpenAIRealtimePreflightResult:
    can_start: bool
    audio_leaves_machine: bool
    network_session_started: bool
    credential_present: bool
    health: ProviderHealth
    opened: ProviderSessionOpened | None
    error: PrivacySafeProviderError | None
    connect_plan: dict[str, Any] | None

    def safe_report(self) -> dict[str, Any]:
        safe_error_code = self.error.code.value if self.error else None
        return {
            "can_start": self.can_start,
            "audio_leaves_machine": self.audio_leaves_machine,
            "network_session_started": self.network_session_started,
            "credential_present": self.credential_present,
            "provider_id": self.health.provider_id.value,
            "provider_state": self.health.state.value,
            "safe_error_code": safe_error_code,
            "cloud_egress": (
                self.opened.capabilities.cloud_egress
                if self.opened is not None
                else True
            ),
            "model": self.health.models[0].id,
        }


def openai_pcm_format() -> PcmFormat:
    return PcmFormat(
        sample_rate_hz=24_000,
        channels=1,
        sample_format=SampleFormat.S16LE,
        frame_duration_ms=_DEFAULT_FRAME_DURATION_MS,
    )


def build_session_update_event(
    request: OpenProviderSession,
) -> dict[str, Any]:
    return {
        "type": "session.update",
        "session": {
            "audio": {
                "output": {
                    "language": request.target_language.value,
                }
            }
        },
    }


def build_input_audio_append_event(pcm: bytes) -> dict[str, str]:
    return {
        "type": "session.input_audio_buffer.append",
        "audio": base64.b64encode(pcm).decode("ascii"),
    }


def build_session_close_event() -> dict[str, str]:
    return {"type": "session.close"}


class OpenAIRealtimeAdapter:
    def __init__(
        self,
        config: OpenAIRealtimeConfig,
        *,
        environ: Mapping[str, str] | None = None,
        start_network_session: Callable[[], None] | None = None,
    ) -> None:
        self._config = config
        self._environ = environ if environ is not None else os.environ
        self._start_network_session = start_network_session
        self._event_sequences: dict[UUID, int] = {}
        self._audio_sequences: dict[tuple[UUID, UUID], int] = {}
        self._audio_remainders: dict[tuple[UUID, UUID], bytes] = {}

    def preflight_open_session(
        self,
        request: OpenProviderSession,
    ) -> OpenAIRealtimePreflightResult:
        credential_present = self._credential_present()
        opened: ProviderSessionOpened | None = None
        error: PrivacySafeProviderError | None = None
        provider_matches = request.provider_id is ProviderId.OPENAI
        can_start = provider_matches and self._config.cloud_opt_in and credential_present

        if can_start:
            opened = ProviderSessionOpened(
                session_id=request.session_id,
                direction_id=request.direction_id,
                event_sequence=self._next_event_sequence(request.session_id),
                negotiated_input_format=openai_pcm_format(),
                negotiated_output_format=openai_pcm_format(),
                capabilities=ProviderCapabilities(
                    transcript_delta=True,
                    translation_delta=True,
                    cancellation=True,
                    cloud_egress=True,
                ),
            )
            state = ProviderState.READY
            model_state = ModelState.READY
            safe_error = None
        else:
            code = (
                SafeErrorCode.PROVIDER_UNAVAILABLE
                if not provider_matches
                else SafeErrorCode.CLOUD_NOT_ENABLED
                if not self._config.cloud_opt_in
                else SafeErrorCode.PROVIDER_AUTH_FAILED
            )
            error = make_provider_error(
                session_id=request.session_id,
                direction_id=request.direction_id,
                event_sequence=self._next_event_sequence(request.session_id),
                code=code,
                retryable=(code is SafeErrorCode.PROVIDER_AUTH_FAILED),
            )
            state = ProviderState.UNAVAILABLE
            model_state = ModelState.FAILED
            safe_error = SafeErrorSummary(
                code=code,
                message=SAFE_ERROR_MESSAGES[code],
                retryable=error.retryable,
            )

        health = ProviderHealth(
            session_id=request.session_id,
            direction_id=request.direction_id,
            event_sequence=self._next_event_sequence(request.session_id),
            provider_id=ProviderId.OPENAI,
            provider_name=OPENAI_PROVIDER_NAME,
            state=state,
            models=(
                ModelHealth(
                    kind=ModelKind.SPEECH_TO_SPEECH,
                    id=self._config.model,
                    state=model_state,
                    device=ComputeDevice.CLOUD,
                    safe_error_code=(
                        safe_error.code.value if safe_error is not None else None
                    ),
                ),
            ),
            queues=ProviderQueues(
                provider_input_buffered_ms=0,
                provider_output_buffered_ms=0,
                queue_lag_ms=0,
            ),
            safe_error=safe_error,
        )
        return OpenAIRealtimePreflightResult(
            can_start=can_start,
            audio_leaves_machine=True,
            network_session_started=False,
            credential_present=credential_present,
            health=health,
            opened=opened,
            error=error,
            connect_plan=(
                self._connect_plan(request) if can_start else None
            ),
        )

    def map_realtime_events(
        self,
        request: OpenProviderSession,
        event: Mapping[str, Any],
        *,
        stream_id: UUID,
        utterance_id: UUID,
        now_ns: int,
    ) -> tuple[
        ProviderAudioDelta | ProviderTranscriptDelta | ProviderTranslationDelta,
        ...,
    ]:
        event_type = event.get("type")
        if event_type == "session.output_audio.delta":
            delta = event.get("delta")
            if not isinstance(delta, str):
                return ()
            try:
                pcm = base64.b64decode(delta, validate=True)
            except binascii.Error as error:
                raise ValueError("invalid_openai_audio_delta") from error
            sequence_key = (request.session_id, utterance_id)
            combined = self._audio_remainders.get(sequence_key, b"") + pcm
            frame_bytes = (
                openai_pcm_format().sample_rate_hz
                * openai_pcm_format().channels
                * 2
                * openai_pcm_format().frame_duration_ms
                // 1000
            )
            complete_bytes = len(combined) - (len(combined) % frame_bytes)
            complete = combined[:complete_bytes]
            self._audio_remainders[sequence_key] = combined[complete_bytes:]
            events: list[ProviderAudioDelta] = []
            sequence = self._audio_sequences.get(sequence_key, 0)
            for offset in range(0, len(complete), frame_bytes):
                events.append(
                    ProviderAudioDelta(
                        session_id=request.session_id,
                        direction_id=request.direction_id,
                        stream_id=stream_id,
                        utterance_id=utterance_id,
                        sequence=sequence,
                        event_sequence=self._next_event_sequence(
                            request.session_id
                        ),
                        provider_monotonic_ns=now_ns,
                        sample_rate_hz=24_000,
                        channels=1,
                        sample_format=SampleFormat.S16LE,
                        frame_duration_ms=_DEFAULT_FRAME_DURATION_MS,
                        pcm=complete[offset : offset + frame_bytes],
                    )
                )
                sequence += 1
            self._audio_sequences[sequence_key] = sequence
            return tuple(events)

        if not request.debug_text_enabled:
            return ()
        delta = event.get("delta")
        if not isinstance(delta, str):
            return ()
        if event_type == "session.input_transcript.delta":
            return (
                ProviderTranscriptDelta(
                    session_id=request.session_id,
                    direction_id=request.direction_id,
                    stream_id=stream_id,
                    utterance_id=utterance_id,
                    event_sequence=self._next_event_sequence(request.session_id),
                    text=delta,
                    is_final=False,
                ),
            )
        if event_type == "session.output_transcript.delta":
            return (
                ProviderTranslationDelta(
                    session_id=request.session_id,
                    direction_id=request.direction_id,
                    stream_id=stream_id,
                    utterance_id=utterance_id,
                    event_sequence=self._next_event_sequence(request.session_id),
                    text=delta,
                    stable_prefix=False,
                    is_final=False,
                ),
            )
        return ()

    def _credential_present(self) -> bool:
        return bool(self._environ.get(self._config.credential_env_name, "").strip())

    def _connect_plan(self, request: OpenProviderSession) -> dict[str, Any]:
        return {
            "uri": f"{self._config.endpoint}?model={self._config.model}",
            "model": self._config.model,
            "target_language": request.target_language.value,
            "input_sample_rate_hz": openai_pcm_format().sample_rate_hz,
            "output_sample_rate_hz": openai_pcm_format().sample_rate_hz,
            "send_events": (
                build_session_update_event(request)["type"],
                build_input_audio_append_event(b"")["type"],
                build_session_close_event()["type"],
            ),
            "receive_events": (
                "session.output_audio.delta",
                "session.output_transcript.delta",
                "session.input_transcript.delta",
                "session.closed",
            ),
        }

    def _next_event_sequence(self, session_id: UUID) -> int:
        next_value = self._event_sequences.get(session_id, 0) + 1
        self._event_sequences[session_id] = next_value
        return next_value

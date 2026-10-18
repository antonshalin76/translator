"""Offline consent and credential preflight for the pinned OpenAI HTTPS pipeline."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from .openai_http import API_ORIGIN, ASR_MODEL, MT_MODEL, TTS_MODEL, VPN_INTERFACE
from .provider_contract import (
    SAFE_ERROR_MESSAGES,
    ComputeDevice,
    ModelHealth,
    ModelKind,
    ModelState,
    OpenProviderSession,
    PrivacySafeProviderError,
    ProviderCapabilities,
    ProviderHealth,
    ProviderId,
    ProviderQueues,
    ProviderSessionOpened,
    ProviderState,
    SafeErrorCode,
    SafeErrorSummary,
    make_provider_error,
)

OPENAI_TRANSCRIPTION_MODEL = ASR_MODEL
OPENAI_TRANSLATION_MODEL = MT_MODEL
OPENAI_SPEECH_MODEL = TTS_MODEL
OPENAI_REALTIME_TRANSLATION_MODEL = (
    f"{OPENAI_TRANSCRIPTION_MODEL}/{OPENAI_TRANSLATION_MODEL}/{OPENAI_SPEECH_MODEL}"
)
OPENAI_REALTIME_TRANSLATION_ENDPOINT = API_ORIGIN
OPENAI_PROVIDER_NAME = "openai-https-translation"


@dataclass(frozen=True, slots=True)
class OpenAIRealtimeConfig:
    # Existing callers retain their configuration type, not the old WS protocol.
    cloud_opt_in: bool
    credential_env_name: str = "OPENAI_API_KEY"
    model: str = OPENAI_REALTIME_TRANSLATION_MODEL
    endpoint: str = OPENAI_REALTIME_TRANSLATION_ENDPOINT
    vpn_interface: str = VPN_INTERFACE

    def __post_init__(self) -> None:
        if not self.credential_env_name:
            raise ValueError("credential_env_name_required")
        if self.model != OPENAI_REALTIME_TRANSLATION_MODEL:
            raise ValueError("translation_model_must_be_pinned")
        if self.endpoint != OPENAI_REALTIME_TRANSLATION_ENDPOINT:
            raise ValueError("translation_endpoint_must_be_official")
        if self.vpn_interface != VPN_INTERFACE:
            raise ValueError("vpn_interface_must_be_pinned")


def model_health(failure: SafeErrorCode | None = None) -> tuple[ModelHealth, ...]:
    return tuple(
        ModelHealth(
            kind=kind,
            id=model,
            state=ModelState.FAILED if failure else ModelState.READY,
            device=ComputeDevice.CLOUD,
            safe_error_code=failure.value if failure else None,
        )
        for kind, model in (
            (ModelKind.ASR, OPENAI_TRANSCRIPTION_MODEL),
            (ModelKind.MT, OPENAI_TRANSLATION_MODEL),
            (ModelKind.TTS, OPENAI_SPEECH_MODEL),
        )
    )


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
        return {
            "can_start": self.can_start,
            "audio_leaves_machine": self.audio_leaves_machine,
            "network_session_started": self.network_session_started,
            "credential_present": self.credential_present,
            "provider_id": self.health.provider_id.value,
            "provider_state": self.health.state.value,
            "safe_error_code": self.error.code.value if self.error else None,
            "connect_plan": self.connect_plan,
        }


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

    def preflight_open_session(
        self, request: OpenProviderSession
    ) -> OpenAIRealtimePreflightResult:
        credential_present = bool(
            self._environ.get(self._config.credential_env_name, "").strip()
        )
        code = (
            SafeErrorCode.PROVIDER_UNAVAILABLE
            if request.provider_id is not ProviderId.OPENAI
            or request.voice_profile.has_overrides
            else SafeErrorCode.CLOUD_NOT_ENABLED
            if not self._config.cloud_opt_in
            else SafeErrorCode.PROVIDER_AUTH_FAILED
            if not credential_present
            else None
        )
        error = (
            make_provider_error(
                session_id=request.session_id,
                direction_id=request.direction_id,
                event_sequence=1,
                code=code,
                retryable=False,
            )
            if code
            else None
        )
        opened = (
            ProviderSessionOpened(
                session_id=request.session_id,
                direction_id=request.direction_id,
                event_sequence=1,
                negotiated_input_format=request.requested_input_format,
                negotiated_output_format=request.requested_output_format,
                capabilities=ProviderCapabilities(
                    transcript_delta=request.debug_text_enabled,
                    translation_delta=request.debug_text_enabled,
                    cancellation=True,
                    cloud_egress=True,
                ),
            )
            if code is None
            else None
        )
        health = ProviderHealth(
            session_id=request.session_id,
            direction_id=request.direction_id,
            event_sequence=2,
            provider_id=ProviderId.OPENAI,
            provider_name=OPENAI_PROVIDER_NAME,
            state=ProviderState.UNAVAILABLE if code else ProviderState.READY,
            models=model_health(code),
            queues=ProviderQueues(
                provider_input_buffered_ms=0,
                provider_output_buffered_ms=0,
                queue_lag_ms=0,
            ),
            safe_error=SafeErrorSummary(
                code=code, message=SAFE_ERROR_MESSAGES[code], retryable=False
            )
            if code
            else None,
        )
        return OpenAIRealtimePreflightResult(
            can_start=code is None,
            audio_leaves_machine=True,
            network_session_started=False,
            credential_present=credential_present,
            health=health,
            opened=opened,
            error=error,
            connect_plan={
                "uri": self._config.endpoint,
                "model": self._config.model,
                "target_language": request.target_language.value,
                "endpoints": (
                    "audio/transcriptions",
                    "chat/completions",
                    "audio/speech",
                ),
                "vpn_interface": self._config.vpn_interface,
            }
            if code is None
            else None,
        )

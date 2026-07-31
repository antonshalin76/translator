"""Versioned provider contracts shared with the Rust daemon."""

from __future__ import annotations

from enum import Enum
from typing import Literal, TypedDict
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AudioDirection(str, Enum):
    MICROPHONE = "microphone"
    SPEAKER = "speaker"


class TranslationMode(str, Enum):
    QUALITY_FIRST = "quality_first"
    BALANCED = "balanced"
    STREAMING_FIRST = "streaming_first"


class Language(str, Enum):
    RU = "ru"
    EN = "en"


class VoiceGender(str, Enum):
    MALE = "male"
    FEMALE = "female"


class VoiceEngine(str, Enum):
    PIPER = "piper"
    SILERO = "silero"
    OPENAI = "openai"


class SampleFormat(str, Enum):
    S16LE = "s16le"


class ProviderId(str, Enum):
    LOCAL = "local"
    OPENAI = "openai"


class SafeErrorCode(str, Enum):
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    MODEL_NOT_LOADED = "model_not_loaded"
    UNSUPPORTED_LANGUAGE_PAIR = "unsupported_language_pair"
    QUEUE_OVERFLOW = "queue_overflow"
    CANCELLED = "cancelled"
    NO_SPEECH = "no_speech"
    CLOUD_NOT_ENABLED = "cloud_not_enabled"
    PROVIDER_AUTH_FAILED = "provider_auth_failed"


SAFE_ERROR_MESSAGES: dict[SafeErrorCode, str] = {
    SafeErrorCode.PROVIDER_UNAVAILABLE: "Provider is unavailable",
    SafeErrorCode.MODEL_NOT_LOADED: "Required model is not loaded",
    SafeErrorCode.UNSUPPORTED_LANGUAGE_PAIR: "Language pair is not supported",
    SafeErrorCode.QUEUE_OVERFLOW: "Provider queue limit was reached",
    SafeErrorCode.CANCELLED: "Provider operation was cancelled",
    SafeErrorCode.NO_SPEECH: "No speech was detected",
    SafeErrorCode.CLOUD_NOT_ENABLED: "Cloud provider is not enabled",
    SafeErrorCode.PROVIDER_AUTH_FAILED: "Provider authentication failed",
}


class VoiceProfile(ContractModel):
    language: Language
    gender: VoiceGender
    engine: VoiceEngine
    model_path: str | None = None
    provider_voice_id: str | None = None


class PcmFormat(ContractModel):
    sample_rate_hz: Literal[16000, 24000, 48000]
    channels: Literal[1, 2]
    sample_format: Literal[SampleFormat.S16LE]
    frame_duration_ms: Literal[20, 40, 60, 80, 100]


class OpenProviderSession(ContractModel):
    schema_version: Literal["translator.provider.open_session.v1"] = (
        "translator.provider.open_session.v1"
    )
    session_id: UUID
    provider_id: ProviderId
    direction_id: AudioDirection
    source_language: Language
    target_language: Language
    mode: TranslationMode
    requested_input_format: PcmFormat
    requested_output_format: PcmFormat
    voice_profile: VoiceProfile
    debug_text_enabled: bool = False


class ProviderCapabilities(ContractModel):
    audio_output: Literal[True] = True
    transcript_delta: bool
    translation_delta: bool
    cancellation: bool
    cloud_egress: bool


class ProviderSessionOpened(ContractModel):
    schema_version: Literal["translator.provider.session_opened.v1"] = (
        "translator.provider.session_opened.v1"
    )
    session_id: UUID
    direction_id: AudioDirection
    event_sequence: int = Field(ge=0)
    negotiated_input_format: PcmFormat
    negotiated_output_format: PcmFormat
    capabilities: ProviderCapabilities


class ProviderProbeRequest(ContractModel):
    schema_version: Literal["translator.provider.probe_request.v1"] = (
        "translator.provider.probe_request.v1"
    )


class ProviderProbeResponse(ContractModel):
    schema_version: Literal["translator.provider.probe_response.v1"] = (
        "translator.provider.probe_response.v1"
    )
    generation_id: UUID


class CloseRequestReason(str, Enum):
    USER_STOP = "user_stop"
    ROUTE_REMOVED = "route_removed"
    DEVICE_UNAVAILABLE = "device_unavailable"
    PROVIDER_SWITCH = "provider_switch"
    DAEMON_SHUTDOWN = "daemon_shutdown"


class SessionCloseReason(str, Enum):
    USER_STOP = "user_stop"
    ROUTE_REMOVED = "route_removed"
    DEVICE_UNAVAILABLE = "device_unavailable"
    PROVIDER_SWITCH = "provider_switch"
    DAEMON_SHUTDOWN = "daemon_shutdown"
    PROVIDER_FAILURE = "provider_failure"
    CLOSE_TIMEOUT = "close_timeout"


class CloseProviderSession(ContractModel):
    schema_version: Literal["translator.provider.close_session.v1"] = (
        "translator.provider.close_session.v1"
    )
    session_id: UUID
    reason: CloseRequestReason


class CancelReason(str, Enum):
    LATENCY_POLICY = "latency_policy"
    ROUTE_REMOVED = "route_removed"
    USER_INTERRUPT = "user_interrupt"
    QUEUE_OVERFLOW = "queue_overflow"


class CancelUtterance(ContractModel):
    schema_version: Literal["translator.provider.cancel_utterance.v1"] = (
        "translator.provider.cancel_utterance.v1"
    )
    session_id: UUID
    direction_id: AudioDirection
    utterance_id: UUID
    reason: CancelReason


class UpdateDebugText(ContractModel):
    schema_version: Literal["translator.provider.update_debug_text.v1"] = (
        "translator.provider.update_debug_text.v1"
    )
    session_id: UUID
    enabled: bool


class ProviderInputFrame(ContractModel):
    schema_version: Literal["translator.provider.input.v1"] = (
        "translator.provider.input.v1"
    )
    session_id: UUID
    direction_id: AudioDirection
    stream_id: UUID
    utterance_id: UUID
    sequence: int = Field(ge=0)
    capture_monotonic_ns: int = Field(ge=0)
    sample_rate_hz: Literal[16000, 24000, 48000]
    channels: Literal[1, 2]
    sample_format: Literal[SampleFormat.S16LE]
    frame_duration_ms: Literal[20, 40, 60, 80, 100]
    source_language: Language
    target_language: Language
    mode: TranslationMode
    pcm: bytes
    end_of_utterance: bool


class ProviderAudioDelta(ContractModel):
    schema_version: Literal["translator.provider.audio_delta.v1"] = (
        "translator.provider.audio_delta.v1"
    )
    session_id: UUID
    direction_id: AudioDirection
    stream_id: UUID
    utterance_id: UUID
    sequence: int = Field(ge=0)
    event_sequence: int = Field(ge=0)
    provider_monotonic_ns: int = Field(ge=0)
    sample_rate_hz: Literal[16000, 24000, 48000]
    channels: Literal[1, 2]
    sample_format: Literal[SampleFormat.S16LE]
    frame_duration_ms: Literal[20, 40, 60, 80, 100]
    pcm: bytes


class ProviderTranscriptDelta(ContractModel):
    schema_version: Literal["translator.provider.transcript_delta.v1"] = (
        "translator.provider.transcript_delta.v1"
    )
    session_id: UUID
    direction_id: AudioDirection
    stream_id: UUID
    utterance_id: UUID
    event_sequence: int = Field(ge=0)
    text: str
    is_final: bool


class ProviderTranslationDelta(ContractModel):
    schema_version: Literal["translator.provider.translation_delta.v1"] = (
        "translator.provider.translation_delta.v1"
    )
    session_id: UUID
    direction_id: AudioDirection
    stream_id: UUID
    utterance_id: UUID
    event_sequence: int = Field(ge=0)
    text: str
    stable_prefix: bool
    is_final: bool


class UtteranceOutcome(str, Enum):
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    DROPPED = "dropped"


class ProviderUtteranceFinal(ContractModel):
    schema_version: Literal["translator.provider.utterance_final.v1"] = (
        "translator.provider.utterance_final.v1"
    )
    session_id: UUID
    direction_id: AudioDirection
    stream_id: UUID
    utterance_id: UUID
    event_sequence: int = Field(ge=0)
    final_audio_sequence: int | None = Field(default=None, ge=0)
    outcome: UtteranceOutcome


class ProviderSessionClosed(ContractModel):
    schema_version: Literal["translator.provider.session_closed.v1"] = (
        "translator.provider.session_closed.v1"
    )
    session_id: UUID
    direction_id: AudioDirection
    event_sequence: int = Field(ge=0)
    reason: SessionCloseReason


class ProviderState(str, Enum):
    STARTING = "starting"
    READY = "ready"
    DEGRADED = "degraded"
    BACKPRESSURE = "backpressure"
    RESTARTING = "restarting"
    UNAVAILABLE = "unavailable"
    CLOSED = "closed"


class ModelKind(str, Enum):
    ASR = "asr"
    MT = "mt"
    TTS = "tts"
    SPEECH_TO_SPEECH = "speech_to_speech"


class ModelState(str, Enum):
    NOT_LOADED = "not_loaded"
    LOADING = "loading"
    READY = "ready"
    FAILED = "failed"


class ComputeDevice(str, Enum):
    CUDA = "cuda"
    CPU = "cpu"
    CLOUD = "cloud"


class ModelHealth(ContractModel):
    kind: ModelKind
    id: str
    state: ModelState
    device: ComputeDevice | None = None
    safe_error_code: str | None = None


class ProviderQueues(ContractModel):
    provider_input_buffered_ms: int = Field(ge=0)
    provider_output_buffered_ms: int = Field(ge=0)
    queue_lag_ms: int = Field(ge=0)


class ProviderRetry(ContractModel):
    attempt: int = Field(ge=0)
    next_retry_after_ms: int = Field(ge=0)
    reason_code: str


class SafeErrorSummary(ContractModel):
    code: SafeErrorCode
    message: str
    retryable: bool

    @model_validator(mode="after")
    def validate_static_message(self) -> SafeErrorSummary:
        if self.message != SAFE_ERROR_MESSAGES[self.code]:
            raise ValueError("message must be the static message for code")
        return self


class ProviderHealth(ContractModel):
    schema_version: Literal["translator.provider.health.v1"] = (
        "translator.provider.health.v1"
    )
    session_id: UUID
    direction_id: AudioDirection
    event_sequence: int = Field(ge=0)
    provider_id: ProviderId
    provider_name: str
    state: ProviderState
    models: tuple[ModelHealth, ...]
    queues: ProviderQueues
    retry: ProviderRetry | None = None
    safe_error: SafeErrorSummary | None = None


class ProviderLatency(ContractModel):
    schema_version: Literal["translator.provider.latency.v1"] = (
        "translator.provider.latency.v1"
    )
    session_id: UUID
    direction_id: AudioDirection
    stream_id: UUID
    event_sequence: int = Field(ge=0)
    utterance_id: UUID | None = None
    asr_first_text_ms: int | None = Field(default=None, ge=0)
    asr_final_text_ms: int | None = Field(default=None, ge=0)
    mt_first_text_ms: int | None = Field(default=None, ge=0)
    tts_first_audio_ms: int | None = Field(default=None, ge=0)
    provider_total_ms: int | None = Field(default=None, ge=0)


class LatencyPolicyState(ContractModel):
    direction_id: AudioDirection
    current_mode: TranslationMode
    rolling_window_seconds: Literal[60] = 60
    minimum_samples: Literal[20] = 20
    degrade_after_consecutive_windows: Literal[2] = 2
    recover_after_consecutive_windows: Literal[5] = 5
    cooldown_seconds_after_change: Literal[120] = 120
    p95_first_audio_ms: int = Field(ge=0)
    p95_last_audio_ms: int = Field(ge=0)
    p95_queue_lag_ms: int = Field(ge=0)
    last_mode_change_at: str | None = None
    reason: str | None = None


class PrivacySafeProviderError(ContractModel):
    schema_version: Literal["translator.provider.error.v1"] = (
        "translator.provider.error.v1"
    )
    session_id: UUID
    direction_id: AudioDirection
    stream_id: UUID | None = None
    utterance_id: UUID | None = None
    event_sequence: int = Field(ge=0)
    code: SafeErrorCode
    retryable: bool
    safe_message: str = Field(max_length=160)

    @model_validator(mode="after")
    def validate_static_message(self) -> PrivacySafeProviderError:
        if self.safe_message != SAFE_ERROR_MESSAGES[self.code]:
            raise ValueError("safe_message must be the static message for code")
        return self


def make_provider_error(
    *,
    session_id: UUID,
    direction_id: AudioDirection,
    event_sequence: int,
    code: SafeErrorCode,
    retryable: bool,
    stream_id: UUID | None = None,
    utterance_id: UUID | None = None,
) -> PrivacySafeProviderError:
    return PrivacySafeProviderError(
        session_id=session_id,
        direction_id=direction_id,
        event_sequence=event_sequence,
        stream_id=stream_id,
        utterance_id=utterance_id,
        code=code,
        retryable=retryable,
        safe_message=SAFE_ERROR_MESSAGES[code],
    )


class ProviderLogFields(TypedDict):
    event: Literal["provider_error"]
    code: str
    retryable: bool


def provider_log_fields(error: PrivacySafeProviderError) -> ProviderLogFields:
    """Project an error to fields permitted in normal operational logs."""
    return {
        "event": "provider_error",
        "code": error.code.value,
        "retryable": error.retryable,
    }

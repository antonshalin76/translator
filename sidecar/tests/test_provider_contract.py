import json
from pathlib import Path
from uuid import UUID, uuid4

from pydantic import ValidationError
import pytest

from translator_sidecar.provider_contract import (
    SAFE_ERROR_MESSAGES,
    AudioDirection,
    CloseProviderSession,
    CloseRequestReason,
    ComputeDevice,
    Language,
    LatencyPolicyState,
    ModelHealth,
    ModelKind,
    ModelState,
    OpenProviderSession,
    PcmFormat,
    ProviderCapabilities,
    ProviderAudioDelta,
    ProviderHealth,
    ProviderId,
    ProviderInputFrame,
    ProviderLatency,
    ProviderQueues,
    ProviderSessionClosed,
    ProviderSessionOpened,
    ProviderProbeRequest,
    ProviderProbeResponse,
    ProviderState,
    SafeErrorCode,
    SampleFormat,
    SessionCloseReason,
    TranslationMode,
    UpdateDebugText,
    VoiceEngine,
    VoiceGender,
    VoiceProfile,
    make_provider_error,
    provider_log_fields,
)
from translator_sidecar.generated.translator.provider.v1 import provider_pb2


def pcm_format() -> PcmFormat:
    return PcmFormat(
        sample_rate_hz=16_000,
        channels=1,
        sample_format=SampleFormat.S16LE,
        frame_duration_ms=40,
    )


def test_provider_health_matches_the_versioned_wire_shape() -> None:
    health = ProviderHealth(
        session_id=uuid4(),
        direction_id=AudioDirection.MICROPHONE,
        event_sequence=7,
        provider_id=ProviderId.LOCAL,
        provider_name="local-cascade",
        state=ProviderState.READY,
        models=[
            ModelHealth(
                kind=ModelKind.ASR,
                id="faster-whisper-small",
                state=ModelState.READY,
                device=ComputeDevice.CUDA,
            )
        ],
        queues=ProviderQueues(
            provider_input_buffered_ms=40,
            provider_output_buffered_ms=0,
            queue_lag_ms=12,
        ),
    )

    payload = health.model_dump(mode="json")

    assert payload["schema_version"] == "translator.provider.health.v1"
    assert payload["direction_id"] == "microphone"
    assert payload["event_sequence"] == 7
    assert payload["state"] == "ready"
    assert payload["queues"]["provider_input_buffered_ms"] == 40
    assert "pcm" not in payload
    assert "transcript" not in payload


def test_lifecycle_messages_use_distinct_versions_and_stable_identity() -> None:
    session_id = UUID("8ec8cb30-6881-4896-b413-7649b58cdfb2")
    opened_request = OpenProviderSession(
        session_id=session_id,
        provider_id=ProviderId.LOCAL,
        direction_id=AudioDirection.MICROPHONE,
        source_language=Language.RU,
        target_language=Language.EN,
        mode=TranslationMode.STREAMING_FIRST,
        requested_input_format=pcm_format(),
        requested_output_format=pcm_format(),
        voice_profile=VoiceProfile(
            language=Language.EN,
            gender=VoiceGender.MALE,
            engine=VoiceEngine.PIPER,
            model_path="models/en_US-lessac-medium.onnx",
        ),
        debug_text_enabled=False,
    )
    opened_event = ProviderSessionOpened(
        session_id=session_id,
        direction_id=AudioDirection.MICROPHONE,
        event_sequence=1,
        negotiated_input_format=pcm_format(),
        negotiated_output_format=pcm_format(),
        capabilities=ProviderCapabilities(
            audio_output=True,
            transcript_delta=False,
            translation_delta=False,
            cancellation=True,
            cloud_egress=False,
        ),
    )
    close_request = CloseProviderSession(
        session_id=session_id,
        reason=CloseRequestReason.USER_STOP,
    )
    closed_event = ProviderSessionClosed(
        session_id=session_id,
        direction_id=AudioDirection.MICROPHONE,
        event_sequence=9,
        reason=SessionCloseReason.USER_STOP,
    )

    assert opened_request.session_id == opened_event.session_id
    assert opened_request.session_id == close_request.session_id
    assert opened_request.provider_id is ProviderId.LOCAL
    assert close_request.session_id == closed_event.session_id
    assert opened_request.schema_version == "translator.provider.open_session.v1"
    assert opened_event.schema_version == "translator.provider.session_opened.v1"
    assert close_request.schema_version == "translator.provider.close_session.v1"
    assert closed_event.schema_version == "translator.provider.session_closed.v1"
    assert closed_event.event_sequence == 9
    assert opened_event.event_sequence == 1
    assert opened_event.capabilities.cloud_egress is False

    golden_path = (
        Path(__file__).resolve().parents[2]
        / "tests/contract-fixtures/open_session.json"
    )
    golden = json.loads(golden_path.read_text())
    assert opened_request.model_dump(mode="json", exclude_none=True) == golden
    with pytest.raises(ValidationError):
        OpenProviderSession.model_validate(
            {**golden, "schema_version": "translator.provider.health.v1"}
        )
    with pytest.raises(ValidationError):
        CloseProviderSession.model_validate(
            {
                "schema_version": "translator.provider.close_session.v1",
                "session_id": str(session_id),
                "reason": "provider_failure",
            }
        )


def test_runtime_debug_update_and_probe_have_distinct_versioned_contracts() -> None:
    session_id = uuid4()
    generation_id = uuid4()

    update = UpdateDebugText(session_id=session_id, enabled=False)
    probe = ProviderProbeRequest()
    response = ProviderProbeResponse(generation_id=generation_id)

    assert update.schema_version == "translator.provider.update_debug_text.v1"
    assert update.session_id == session_id
    assert update.enabled is False
    assert probe.schema_version == "translator.provider.probe_request.v1"
    assert response.schema_version == "translator.provider.probe_response.v1"
    assert response.generation_id == generation_id
    assert update.model_dump(mode="json") == {
        "schema_version": "translator.provider.update_debug_text.v1",
        "session_id": str(session_id),
        "enabled": False,
    }
    assert response.model_dump(mode="json") == {
        "schema_version": "translator.provider.probe_response.v1",
        "generation_id": str(generation_id),
    }
    with pytest.raises(ValidationError):
        UpdateDebugText.model_validate(
            {
                **update.model_dump(mode="json"),
                "schema_version": "translator.provider.close_session.v1",
            }
        )
    with pytest.raises(ValidationError):
        ProviderProbeRequest.model_validate(
            {"schema_version": "translator.provider.probe_response.v1"}
        )
    with pytest.raises(ValidationError):
        ProviderProbeResponse.model_validate(
            {
                **response.model_dump(mode="json"),
                "schema_version": "translator.provider.probe_request.v1",
            }
        )


def test_streaming_and_latency_contracts_preserve_ordering_and_policy_state() -> None:
    session_id = uuid4()
    stream_id = uuid4()
    utterance_id = uuid4()
    input_frame = ProviderInputFrame(
        session_id=session_id,
        direction_id=AudioDirection.MICROPHONE,
        stream_id=stream_id,
        utterance_id=utterance_id,
        sequence=3,
        capture_monotonic_ns=1_000_000,
        sample_rate_hz=16_000,
        channels=1,
        sample_format=SampleFormat.S16LE,
        frame_duration_ms=40,
        source_language=Language.RU,
        target_language=Language.EN,
        mode=TranslationMode.STREAMING_FIRST,
        end_of_utterance=True,
        pcm=b"\x00\x01\x02\x03",
    )
    audio = ProviderAudioDelta(
        session_id=session_id,
        direction_id=AudioDirection.MICROPHONE,
        stream_id=stream_id,
        utterance_id=utterance_id,
        sequence=0,
        event_sequence=4,
        provider_monotonic_ns=1_100_000,
        sample_rate_hz=16_000,
        channels=1,
        sample_format=SampleFormat.S16LE,
        frame_duration_ms=40,
        pcm=b"\x04\x05\x06\x07",
    )
    latency = ProviderLatency(
        session_id=session_id,
        direction_id=AudioDirection.MICROPHONE,
        stream_id=stream_id,
        event_sequence=5,
        utterance_id=utterance_id,
        asr_first_text_ms=180,
        mt_first_text_ms=75,
        tts_first_audio_ms=220,
        provider_total_ms=475,
    )
    policy = LatencyPolicyState(
        direction_id=AudioDirection.MICROPHONE,
        current_mode=TranslationMode.STREAMING_FIRST,
        p95_first_audio_ms=930,
        p95_last_audio_ms=1240,
        p95_queue_lag_ms=45,
        reason="first_audio_threshold",
    )

    assert input_frame.sequence == 3
    assert input_frame.end_of_utterance is True
    assert audio.event_sequence == 4
    assert latency.event_sequence == 5
    assert policy.current_mode is TranslationMode.STREAMING_FIRST
    assert input_frame.model_dump(mode="json")["sample_rate_hz"] == 16_000
    assert audio.model_dump(mode="json")["frame_duration_ms"] == 40
    assert "format" not in input_frame.model_dump(mode="json")
    assert "format" not in audio.model_dump(mode="json")


@pytest.mark.parametrize(
    ("sample_rate_hz", "channels", "frame_duration_ms"),
    [(44_100, 1, 40), (16_000, 3, 40), (16_000, 1, 30)],
)
def test_pcm_format_rejects_values_outside_the_negotiated_contract(
    sample_rate_hz: int,
    channels: int,
    frame_duration_ms: int,
) -> None:
    with pytest.raises(ValidationError):
        PcmFormat(
            sample_rate_hz=sample_rate_hz,
            channels=channels,
            sample_format=SampleFormat.S16LE,
            frame_duration_ms=frame_duration_ms,
        )


def test_provider_error_forbids_unknown_and_content_derived_fields() -> None:
    base = {
        "schema_version": "translator.provider.error.v1",
        "session_id": str(uuid4()),
        "direction_id": "speaker",
        "event_sequence": 1,
        "code": "provider_unavailable",
        "retryable": True,
        "safe_message": "Provider is unavailable",
    }

    with pytest.raises(ValidationError):
        from translator_sidecar.provider_contract import PrivacySafeProviderError

        PrivacySafeProviderError.model_validate({**base, "transcript": "private-spoken-marker"})

    with pytest.raises(ValidationError):
        PrivacySafeProviderError.model_validate(
            {**base, "safe_message": "private-spoken-marker"}
        )


def test_privacy_safe_logging_projects_only_operational_error_fields() -> None:
    error = make_provider_error(
        session_id=uuid4(),
        direction_id=AudioDirection.SPEAKER,
        event_sequence=11,
        code=SafeErrorCode.PROVIDER_UNAVAILABLE,
        retryable=True,
    )

    fields = provider_log_fields(error)

    assert fields == {
        "event": "provider_error",
        "code": "provider_unavailable",
        "retryable": True,
    }
    assert "private-spoken-marker" not in repr(fields)


def test_no_speech_has_stable_cross_language_contract() -> None:
    assert SafeErrorCode.NO_SPEECH.value == "no_speech"
    assert (
        SAFE_ERROR_MESSAGES[SafeErrorCode.NO_SPEECH]
        == "No speech was detected"
    )
    assert provider_pb2.SAFE_ERROR_CODE_NO_SPEECH == 8

    error = make_provider_error(
        session_id=uuid4(),
        direction_id=AudioDirection.MICROPHONE,
        event_sequence=1,
        code=SafeErrorCode.NO_SPEECH,
        retryable=True,
    )
    assert error.safe_message == "No speech was detected"

from __future__ import annotations

import json
import re
from uuid import UUID, uuid4

import pytest
from test_provider_contract import VOICE_OVERRIDE_CASES

from translator_sidecar.openai_provider import (
    OPENAI_REALTIME_TRANSLATION_ENDPOINT,
    OPENAI_REALTIME_TRANSLATION_MODEL,
    OpenAIRealtimeAdapter,
    OpenAIRealtimeConfig,
    build_input_audio_append_event,
    build_session_close_event,
    build_session_update_event,
    openai_pcm_format,
)
from translator_sidecar.provider_contract import (
    AudioDirection,
    Language,
    OpenProviderSession,
    PcmFormat,
    ProviderAudioDelta,
    ProviderId,
    ProviderTranscriptDelta,
    ProviderTranslationDelta,
    SafeErrorCode,
    SampleFormat,
    TranslationMode,
    VoiceEngine,
    VoiceGender,
    VoiceProfile,
)


def request(
    direction: AudioDirection = AudioDirection.MICROPHONE,
    *,
    debug_text_enabled: bool = False,
) -> OpenProviderSession:
    source, target = (
        (Language.RU, Language.EN)
        if direction is AudioDirection.MICROPHONE
        else (Language.EN, Language.RU)
    )
    input_format = PcmFormat(
        sample_rate_hz=16_000,
        channels=1,
        sample_format=SampleFormat.S16LE,
        frame_duration_ms=100,
    )
    output_format = PcmFormat(
        sample_rate_hz=24_000,
        channels=1,
        sample_format=SampleFormat.S16LE,
        frame_duration_ms=20,
    )
    return OpenProviderSession(
        session_id=uuid4(),
        provider_id=ProviderId.OPENAI,
        direction_id=direction,
        source_language=source,
        target_language=target,
        mode=TranslationMode.STREAMING_FIRST,
        requested_input_format=input_format,
        requested_output_format=output_format,
        voice_profile=VoiceProfile(
            language=target,
            gender=VoiceGender.MALE,
            engine=VoiceEngine.OPENAI,
        ),
        debug_text_enabled=debug_text_enabled,
    )


@pytest.mark.parametrize(
    "model_path,provider_voice_id",
    [*VOICE_OVERRIDE_CASES, pytest.param(None, "alloy", id="formerly-ignored-alloy")],
)
def test_preflight_refuses_voice_override_without_open_or_connect_plan(
    model_path, provider_voice_id
):
    value = request()
    value = value.model_copy(
        update={
            "voice_profile": value.voice_profile.model_copy(
                update={
                    "model_path": model_path,
                    "provider_voice_id": provider_voice_id,
                }
            )
        }
    )
    adapter = OpenAIRealtimeAdapter(
        OpenAIRealtimeConfig(cloud_opt_in=True),
        environ={"OPENAI_API_KEY": "synthetic-credential"},
        start_network_session=lambda: pytest.fail("preflight must not start network"),
    )
    result = adapter.preflight_open_session(value)
    assert (
        not result.can_start and result.opened is None and result.connect_plan is None
    )
    assert not result.network_session_started
    assert result.error.code is SafeErrorCode.PROVIDER_UNAVAILABLE
    assert result.error.retryable is False
    assert result.health.safe_error.code is SafeErrorCode.PROVIDER_UNAVAILABLE
    assert result.health.safe_error.retryable is False
    assert "private-voice-override" not in json.dumps(result.safe_report())


@pytest.mark.parametrize(
    "provider_id,consent,credential,expected,retryable",
    [
        (ProviderId.LOCAL, False, False, SafeErrorCode.PROVIDER_UNAVAILABLE, False),
        (ProviderId.LOCAL, True, True, SafeErrorCode.PROVIDER_UNAVAILABLE, False),
        (ProviderId.OPENAI, False, False, SafeErrorCode.CLOUD_NOT_ENABLED, False),
        (ProviderId.OPENAI, False, True, SafeErrorCode.CLOUD_NOT_ENABLED, False),
        (ProviderId.OPENAI, True, False, SafeErrorCode.PROVIDER_AUTH_FAILED, True),
    ],
)
def test_preflight_voice_override_keeps_existing_auth_error_precedence(
    provider_id, consent, credential, expected, retryable
):
    value = request()
    value = value.model_copy(
        update={
            "provider_id": provider_id,
            "voice_profile": value.voice_profile.model_copy(
                update={"provider_voice_id": "alloy"}
            ),
        }
    )
    adapter = OpenAIRealtimeAdapter(
        OpenAIRealtimeConfig(cloud_opt_in=consent),
        environ={"OPENAI_API_KEY": "synthetic-credential"} if credential else {},
    )
    result = adapter.preflight_open_session(value)
    assert (
        not result.can_start and result.opened is None and result.connect_plan is None
    )
    assert result.error.code is expected and result.error.retryable is retryable
    assert not result.network_session_started


@pytest.mark.parametrize("language", [Language.RU, Language.EN])
@pytest.mark.parametrize("gender", [VoiceGender.MALE, VoiceGender.FEMALE])
def test_preflight_builtin_profiles_preserve_existing_translation_negotiation(
    language, gender
):
    value = request(
        AudioDirection.MICROPHONE if language is Language.EN else AudioDirection.SPEAKER
    )
    value = value.model_copy(
        update={
            "voice_profile": value.voice_profile.model_copy(update={"gender": gender})
        }
    )
    adapter = OpenAIRealtimeAdapter(
        OpenAIRealtimeConfig(cloud_opt_in=True),
        environ={"OPENAI_API_KEY": "synthetic-secret"},
    )
    result = adapter.preflight_open_session(value)
    assert result.can_start and result.opened is not None and result.error is None
    assert result.connect_plan["target_language"] == language.value
    assert build_session_update_event(value)["session"]["audio"]["output"] == {
        "language": language.value
    }
    assert result.opened.negotiated_output_format == openai_pcm_format()


def test_openai_preflight_blocks_without_cloud_opt_in_and_never_starts_network() -> (
    None
):
    network_started = False

    def mark_network_started() -> None:
        nonlocal network_started
        network_started = True

    adapter = OpenAIRealtimeAdapter(
        OpenAIRealtimeConfig(cloud_opt_in=False),
        environ={"OPENAI_API_KEY": "credential-present-secret"},
        start_network_session=mark_network_started,
    )

    result = adapter.preflight_open_session(request())

    assert result.can_start is False
    assert result.network_session_started is False
    assert result.audio_leaves_machine is True
    assert result.opened is None
    assert result.error is not None
    assert result.error.code is SafeErrorCode.CLOUD_NOT_ENABLED
    assert result.health.provider_id is ProviderId.OPENAI
    assert result.health.state.value == "unavailable"
    assert result.health.safe_error is not None
    assert result.health.safe_error.code is SafeErrorCode.CLOUD_NOT_ENABLED
    assert network_started is False


def test_openai_config_pins_the_official_translation_endpoint() -> None:
    assert (
        OpenAIRealtimeConfig(
            cloud_opt_in=True,
            endpoint=OPENAI_REALTIME_TRANSLATION_ENDPOINT,
        ).endpoint
        == OPENAI_REALTIME_TRANSLATION_ENDPOINT
    )

    with pytest.raises(ValueError, match="translation_endpoint_must_be_official"):
        OpenAIRealtimeConfig(
            cloud_opt_in=True,
            endpoint="wss://example.test/v1/realtime/translations",
        )


def test_openai_preflight_blocks_missing_credentials_without_network_session() -> None:
    adapter = OpenAIRealtimeAdapter(
        OpenAIRealtimeConfig(cloud_opt_in=True),
        environ={},
        start_network_session=lambda: (_ for _ in ()).throw(
            AssertionError("network must not start without credentials")
        ),
    )

    result = adapter.preflight_open_session(request())

    assert result.can_start is False
    assert result.network_session_started is False
    assert result.error is not None
    assert result.error.code is SafeErrorCode.PROVIDER_AUTH_FAILED
    assert result.health.safe_error is not None
    assert result.health.safe_error.code is SafeErrorCode.PROVIDER_AUTH_FAILED
    assert result.health.models[0].state.value == "failed"
    rendered = result.safe_report()
    assert "credential-present-secret" not in json.dumps(rendered)
    assert rendered["credential_present"] is False


def test_openai_ready_preflight_negotiates_cloud_capabilities_without_secret_leak() -> (
    None
):
    adapter = OpenAIRealtimeAdapter(
        OpenAIRealtimeConfig(cloud_opt_in=True),
        environ={"OPENAI_API_KEY": "credential-present-secret"},
        start_network_session=lambda: (_ for _ in ()).throw(
            AssertionError("preflight must not open the socket")
        ),
    )

    result = adapter.preflight_open_session(request())

    assert result.can_start is True
    assert result.network_session_started is False
    assert result.error is None
    assert result.opened is not None
    assert result.opened.capabilities.cloud_egress is True
    assert result.opened.negotiated_input_format == openai_pcm_format()
    assert result.opened.negotiated_output_format == openai_pcm_format()
    assert result.health.state.value == "ready"
    assert result.connect_plan is not None
    assert result.connect_plan["uri"] == (
        f"{OPENAI_REALTIME_TRANSLATION_ENDPOINT}"
        f"?model={OPENAI_REALTIME_TRANSLATION_MODEL}"
    )
    rendered = json.dumps(result.safe_report(), sort_keys=True)
    assert "credential-present-secret" not in rendered
    assert not re.search(r"sk-[A-Za-z0-9_-]+", rendered)


def test_openai_preflight_rejects_provider_identity_mismatch_without_network_session() -> (
    None
):
    adapter = OpenAIRealtimeAdapter(
        OpenAIRealtimeConfig(cloud_opt_in=True),
        environ={"OPENAI_API_KEY": "credential-present-secret"},
        start_network_session=lambda: (_ for _ in ()).throw(
            AssertionError("provider mismatch must not open the socket")
        ),
    )
    local_request = request().model_copy(update={"provider_id": ProviderId.LOCAL})

    result = adapter.preflight_open_session(local_request)

    assert result.can_start is False
    assert result.network_session_started is False
    assert result.error is not None
    assert result.error.code is SafeErrorCode.PROVIDER_UNAVAILABLE
    assert result.health.safe_error is not None
    assert result.health.safe_error.code is SafeErrorCode.PROVIDER_UNAVAILABLE


def test_websocket_event_builders_match_translation_session_contract() -> None:
    session = request(AudioDirection.SPEAKER)
    pcm = b"\x00\x01\x02\x03"

    assert build_session_update_event(session) == {
        "type": "session.update",
        "session": {
            "audio": {"input": {"transcription": None}, "output": {"language": "ru"}}
        },
    }
    assert build_input_audio_append_event(pcm) == {
        "type": "session.input_audio_buffer.append",
        "audio": "AAECAw==",
    }
    assert build_session_close_event() == {"type": "session.close"}


def test_openai_event_mapping_respects_debug_text_gate_and_audio_contract() -> None:
    session = request(debug_text_enabled=False)
    adapter = OpenAIRealtimeAdapter(
        OpenAIRealtimeConfig(cloud_opt_in=True),
        environ={"OPENAI_API_KEY": "credential-present-secret"},
    )
    stream_id = UUID(int=10)
    utterance_id = UUID(int=11)

    pcm = b"\x01\x02" * (24_000 * 200 // 1000)
    audio_events = adapter.map_realtime_events(
        session,
        {
            "type": "session.output_audio.delta",
            "delta": build_input_audio_append_event(pcm)["audio"],
        },
        stream_id=stream_id,
        utterance_id=utterance_id,
        now_ns=123_000_000,
    )
    assert len(audio_events) == 10
    assert all(isinstance(event, ProviderAudioDelta) for event in audio_events)
    assert [event.sequence for event in audio_events] == list(range(10))
    assert all(len(event.pcm) == 960 for event in audio_events)
    assert all(event.sample_rate_hz == 24_000 for event in audio_events)
    assert all(event.frame_duration_ms == 20 for event in audio_events)

    assert (
        adapter.map_realtime_events(
            session,
            {"type": "session.output_transcript.delta", "delta": "private target"},
            stream_id=stream_id,
            utterance_id=utterance_id,
            now_ns=123_000_000,
        )
        == ()
    )

    debug_session = session.model_copy(update={"debug_text_enabled": True})
    translation = adapter.map_realtime_events(
        debug_session,
        {"type": "session.output_transcript.delta", "delta": "private target"},
        stream_id=stream_id,
        utterance_id=utterance_id,
        now_ns=123_000_000,
    )
    transcript = adapter.map_realtime_events(
        debug_session,
        {"type": "session.input_transcript.delta", "delta": "private source"},
        stream_id=stream_id,
        utterance_id=utterance_id,
        now_ns=123_000_000,
    )

    assert len(translation) == 1
    assert isinstance(translation[0], ProviderTranslationDelta)
    assert translation[0].text == "private target"
    assert len(transcript) == 1
    assert isinstance(transcript[0], ProviderTranscriptDelta)
    assert transcript[0].text == "private source"

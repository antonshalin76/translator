from __future__ import annotations

import json

import pytest
from test_openai_runtime import open_request
from test_provider_contract import VOICE_OVERRIDE_CASES

from translator_sidecar.openai_provider import (
    OpenAIRealtimeAdapter,
    OpenAIRealtimeConfig,
)
from translator_sidecar.provider_contract import ProviderId, SafeErrorCode


@pytest.mark.parametrize(
    "consent,key,provider,code",
    [
        (False, "synthetic", ProviderId.OPENAI, SafeErrorCode.CLOUD_NOT_ENABLED),
        (True, "", ProviderId.OPENAI, SafeErrorCode.PROVIDER_AUTH_FAILED),
        (True, "   ", ProviderId.OPENAI, SafeErrorCode.PROVIDER_AUTH_FAILED),
        (True, "synthetic", ProviderId.LOCAL, SafeErrorCode.PROVIDER_UNAVAILABLE),
        (True, "synthetic", ProviderId.OPENAI, None),
    ],
)
def test_preflight_is_offline_and_reports_safe_truth(consent, key, provider, code):
    value = open_request().model_copy(update={"provider_id": provider})
    adapter = OpenAIRealtimeAdapter(
        OpenAIRealtimeConfig(cloud_opt_in=consent),
        environ={"OPENAI_API_KEY": key},
        start_network_session=lambda: pytest.fail("network forbidden"),
    )
    result = adapter.preflight_open_session(value)
    assert result.can_start == (code is None)
    assert result.audio_leaves_machine and not result.network_session_started
    assert (result.error.code if result.error else None) is code
    assert "synthetic" not in json.dumps(result.safe_report())
    if code is None:
        assert result.opened.negotiated_input_format == value.requested_input_format
        assert not result.opened.capabilities.transcript_delta
        assert not result.opened.capabilities.translation_delta
        assert result.connect_plan["uri"] == "https://api.openai.com/v1"
        assert [model.id for model in result.health.models] == [
            "gpt-transcribe",
            "gpt-4.1",
            "gpt-4o-mini-tts",
        ]
    else:
        assert result.opened is None and result.connect_plan is None


@pytest.mark.parametrize("model_path,provider_voice_id", VOICE_OVERRIDE_CASES)
def test_preflight_refuses_voice_overrides(model_path, provider_voice_id):
    value = open_request()
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
    result = OpenAIRealtimeAdapter(
        OpenAIRealtimeConfig(cloud_opt_in=True), environ={"OPENAI_API_KEY": "synthetic"}
    ).preflight_open_session(value)
    assert (
        not result.can_start and result.error.code is SafeErrorCode.PROVIDER_UNAVAILABLE
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model": "other"},
        {"endpoint": "https://example.invalid"},
        {"credential_env_name": ""},
        {"vpn_interface": ""},
        {"vpn_interface": "eth0"},
        {"vpn_interface": "lo"},
    ],
)
def test_config_refuses_unpinned_models_endpoint_and_missing_owners(kwargs):
    with pytest.raises(ValueError):
        OpenAIRealtimeConfig(cloud_opt_in=True, **kwargs)

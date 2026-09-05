from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import pytest

import translator_sidecar.local.runtime as runtime_module
from translator_sidecar.local.model_lease import VerifiedModelSource
from translator_sidecar.local.mt import LocalTranslationError, NllbTranslator
from translator_sidecar.local.runtime import build_local_provider
from translator_sidecar.provider_contract import (
    AudioDirection,
    ComputeDevice,
    Language,
    ModelState,
    OpenProviderSession,
    PcmFormat,
    ProviderId,
    ProviderState,
    SampleFormat,
    TranslationMode,
    VoiceEngine,
    VoiceGender,
    VoiceProfile,
)


@dataclass(frozen=True)
class FakeFile:
    path: str


@dataclass(frozen=True)
class FakeModel:
    cache_path: Path
    files: tuple[FakeFile, ...]


class FakeManifest:
    def __init__(
        self,
        root: Path,
    ) -> None:
        self.root = root
        self.models = {
            model_id: FakeModel(
                root / model_id,
                (
                    FakeFile(f"{model_id}.bin"),
                    FakeFile(f"{model_id}.json"),
                ),
            )
            for model_id in (
                "faster-whisper-small",
                "faster-whisper-large-v3",
                "nllb-200-distilled-600m-ct2-int8",
                "piper-ru-dmitri-medium",
                "piper-en-ryan-medium",
                "piper-ru-irina-medium",
                "piper-en-hfc-female-medium",
            )
        }


class Closable:
    def close(self) -> None:
        unload = getattr(self, "unload_model", None)
        if unload is not None:
            unload()


@pytest.mark.parametrize("failure_phase", ["smoke", "asr"])
def test_bootstrap_retains_failed_native_cleanup_for_unavailable_provider_shutdown(
    tmp_path, monkeypatch, failure_phase
):
    manifest = FakeManifest(tmp_path)
    loads = []
    asr_closes = []

    class Native:
        attempts = 0

        def unload_model(self):
            self.attempts += 1
            if self.attempts == 1:
                raise RuntimeError("private-cleanup-marker")

    native = Native()

    class Adapter(NllbTranslator):
        @classmethod
        def load(cls, source, *, device):
            loads.append(device)
            return cls(tmp_path, translator=native, tokenizer=object())

        def translate(self, *_args, **kwargs):
            if self.unavailable:
                raise LocalTranslationError("local MT is unavailable")
            if failure_phase == "smoke":
                raise LocalTranslationError("bootstrap smoke failed")
            return "translated"

    class Asr:
        def __init__(self, **kwargs):
            pass

        def prepare(self):
            raise RuntimeError("ASR load failed")

        def close(self):
            asr_closes.append("close")

    monkeypatch.setattr(runtime_module, "load_manifest", lambda *_args: manifest)
    monkeypatch.setattr(runtime_module, "_cuda_available", lambda: True)
    monkeypatch.setattr(runtime_module, "NllbTranslator", Adapter)
    monkeypatch.setattr(runtime_module, "AsrModelManager", Asr)
    provider = build_local_provider(now_ns=lambda: 0, manifest_path=tmp_path)
    assert loads == ["cuda"]
    assert native.attempts == 1
    assert provider._translator._translator is native
    with pytest.raises(LocalTranslationError, match="unavailable"):
        provider._translator.translate("must not infer")

    async def scenario():
        async def publish(_batch, commit):
            commit()

        _, health = await provider.open_session(open_request(), publish)
        assert health.state is ProviderState.UNAVAILABLE
        assert all(model.state is ModelState.FAILED for model in health.models)
        await provider.shutdown()

    asyncio.run(scenario())
    assert native.attempts == 2
    assert provider._translator._translator is None
    assert asr_closes == (["close"] if failure_phase == "asr" else [])


def open_request() -> OpenProviderSession:
    pcm = PcmFormat(
        sample_rate_hz=16_000,
        channels=1,
        sample_format=SampleFormat.S16LE,
        frame_duration_ms=100,
    )
    return OpenProviderSession(
        session_id=uuid4(),
        provider_id=ProviderId.LOCAL,
        direction_id=AudioDirection.MICROPHONE,
        source_language=Language.RU,
        target_language=Language.EN,
        mode=TranslationMode.QUALITY_FIRST,
        requested_input_format=pcm,
        requested_output_format=pcm,
        voice_profile=VoiceProfile(
            language=Language.EN,
            gender=VoiceGender.MALE,
            engine=VoiceEngine.PIPER,
        ),
    )


@pytest.mark.parametrize(
    ("cuda_available", "device", "compute_device"),
    [
        (True, "cuda", ComputeDevice.CUDA),
        (False, "cpu", ComputeDevice.CPU),
    ],
)
def test_build_local_provider_uses_verified_manifest_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cuda_available: bool,
    device: str,
    compute_device: ComputeDevice,
) -> None:
    manifest = FakeManifest(tmp_path)
    captured = {}

    class FakeAsr(Closable):
        def __init__(self, **kwargs) -> None:
            captured["asr"] = kwargs
            captured["asr_instance"] = self

    class FakeTranslator(Closable):
        @classmethod
        def load(cls, path, *, device: str):
            captured["mt"] = (path, device)
            instance = cls()
            captured["mt_instance"] = instance
            return instance

        def translate(self, text, **kwargs):
            del text, kwargs
            return "translated"

    class FakeRegistry(Closable):
        def __init__(self, voice_paths) -> None:
            captured["voices"] = voice_paths
            captured["registry_instance"] = self

        def prepare(self) -> None:
            captured["registry_prepared"] = self

    class FakeTts:
        def __init__(self, registry) -> None:
            captured["tts_registry"] = registry
            captured["tts_instance"] = self

    class FakeScheduler:
        def __init__(self) -> None:
            captured["scheduler_instance"] = self

    class FakeProvider:
        def __init__(self, **kwargs) -> None:
            captured["provider"] = kwargs

    loaded_paths = []

    def fake_load_manifest(path: Path):
        loaded_paths.append(path)
        return manifest

    monkeypatch.setattr(runtime_module, "load_manifest", fake_load_manifest)
    monkeypatch.setattr(
        runtime_module,
        "_cuda_available",
        lambda: cuda_available,
    )
    monkeypatch.setattr(runtime_module, "AsrModelManager", FakeAsr)
    monkeypatch.setattr(runtime_module, "NllbTranslator", FakeTranslator)
    monkeypatch.setattr(runtime_module, "PiperVoiceRegistry", FakeRegistry)
    monkeypatch.setattr(runtime_module, "PiperTts", FakeTts)
    monkeypatch.setattr(runtime_module, "InferenceScheduler", FakeScheduler)
    monkeypatch.setattr(runtime_module, "LocalProvider", FakeProvider)
    manifest_path = tmp_path / "manifest.json"

    def now_ns() -> int:
        return 7

    provider = build_local_provider(
        now_ns=now_ns,
        manifest_path=manifest_path,
    )

    assert isinstance(provider, FakeProvider)
    assert loaded_paths == [manifest_path]
    assert captured["asr"] == {
        "selected_id": "small",
        "model_paths": {
            "small": VerifiedModelSource(manifest, "faster-whisper-small"),
        },
        "device": device,
    }
    assert captured["mt"] == (
        VerifiedModelSource(manifest, "nllb-200-distilled-600m-ct2-int8"),
        device,
    )
    assert captured["voices"] == {
        (Language.RU, VoiceGender.MALE): (
            VerifiedModelSource(manifest, "piper-ru-dmitri-medium")
        ),
        (Language.EN, VoiceGender.MALE): (
            VerifiedModelSource(manifest, "piper-en-ryan-medium")
        ),
        (Language.RU, VoiceGender.FEMALE): (
            VerifiedModelSource(manifest, "piper-ru-irina-medium")
        ),
        (Language.EN, VoiceGender.FEMALE): (
            VerifiedModelSource(manifest, "piper-en-hfc-female-medium")
        ),
    }
    assert captured["provider"]["now_ns"] is now_ns
    assert captured["provider"]["asr"] is captured["asr_instance"]
    assert captured["provider"]["translator"] is captured["mt_instance"]
    assert captured["provider"]["tts"] is captured["tts_instance"]
    assert captured["provider"]["scheduler"] is captured["scheduler_instance"]
    assert captured["provider"]["mt_device"] is compute_device
    assert captured["provider"]["asr_model_id"] == "faster-whisper-small"
    assert captured["provider"]["mt_model_id"] == "nllb-200-distilled-600m-ct2-int8"
    assert captured["provider"]["tts_model_id"] == "piper-medium"
    assert captured["registry_prepared"] is captured["registry_instance"]
    assert len(captured["asr"]["model_paths"]) == 1


def test_build_local_provider_uses_repository_manifest_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    observed = []
    captured = {}

    def stop_after_path(path: Path):
        observed.append(path)
        raise RuntimeError("stop-after-default-path")

    class FakeProvider:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    class FakeScheduler:
        pass

    monkeypatch.setattr(runtime_module, "load_manifest", stop_after_path)
    monkeypatch.setattr(runtime_module, "LocalProvider", FakeProvider)
    monkeypatch.setattr(runtime_module, "InferenceScheduler", FakeScheduler)

    provider = build_local_provider(now_ns=lambda: 0)

    assert isinstance(provider, FakeProvider)
    assert captured["asr"].unavailable is True
    assert captured["translator"].unavailable is True
    assert captured["tts"].unavailable is True
    assert "stop-after-default-path" not in caplog.text
    assert observed == [
        Path(runtime_module.__file__).resolve().parents[3] / "models" / "manifest.json"
    ]


def test_missing_runtime_starts_reachable_unavailable_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def missing_manifest(path: Path):
        raise RuntimeError("private-missing-model-marker")

    monkeypatch.setattr(runtime_module, "load_manifest", missing_manifest)
    provider = build_local_provider(
        now_ns=lambda: 0,
        manifest_path=tmp_path / "missing.json",
    )

    async def scenario() -> None:
        async def publish(batch, commit) -> None:
            commit()

        request = open_request()
        opened, health = await provider.open_session(request, publish)
        assert opened.session_id == request.session_id
        assert health.state is ProviderState.UNAVAILABLE
        assert {model.state for model in health.models} == {ModelState.FAILED}
        await provider.shutdown()

    asyncio.run(scenario())
    assert "private-missing-model-marker" not in caplog.text


def test_explicit_unavailable_runtime_mode_skips_model_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inventory_called = False

    def unexpected_runtime(*args, **kwargs):
        nonlocal inventory_called
        inventory_called = True
        raise AssertionError("unavailable mode touched model runtime")

    class UnexpectedTranslator:
        @classmethod
        def load(cls, *args, **kwargs):
            unexpected_runtime(*args, **kwargs)

    class UnexpectedAdapter:
        def __init__(self, *args, **kwargs) -> None:
            unexpected_runtime(*args, **kwargs)

    monkeypatch.setenv("TRANSLATOR_LOCAL_RUNTIME_MODE", "unavailable")
    monkeypatch.setattr(
        runtime_module,
        "load_manifest",
        unexpected_runtime,
    )
    monkeypatch.setattr(
        runtime_module,
        "_cuda_available",
        unexpected_runtime,
    )
    monkeypatch.setattr(
        runtime_module,
        "NllbTranslator",
        UnexpectedTranslator,
    )
    monkeypatch.setattr(
        runtime_module,
        "AsrModelManager",
        UnexpectedAdapter,
    )
    monkeypatch.setattr(
        runtime_module,
        "PiperVoiceRegistry",
        UnexpectedAdapter,
    )

    provider = build_local_provider(now_ns=lambda: 0)

    async def scenario() -> None:
        async def publish(batch, commit) -> None:
            commit()

        _, health = await provider.open_session(open_request(), publish)
        assert health.state is ProviderState.UNAVAILABLE
        assert {model.state for model in health.models} == {ModelState.FAILED}
        await provider.shutdown()

    asyncio.run(scenario())
    assert inventory_called is False


@pytest.mark.parametrize(
    "mode",
    ["", "UNAVAILABLE", "disabled", "0"],
)
def test_unknown_runtime_modes_do_not_bypass_inventory(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    inventory_called = False

    def observed_inventory(path: Path):
        nonlocal inventory_called
        inventory_called = True
        raise RuntimeError("observed-real-inventory")

    monkeypatch.setenv("TRANSLATOR_LOCAL_RUNTIME_MODE", mode)
    monkeypatch.setattr(
        runtime_module,
        "load_manifest",
        observed_inventory,
    )

    provider = build_local_provider(now_ns=lambda: 0)

    asyncio.run(provider.shutdown())
    assert inventory_called is True


def test_build_local_provider_requires_all_selected_manifest_entries_before_adapters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    constructed = []
    captured_providers = []

    class UnexpectedAdapter:
        def __init__(self, *args, **kwargs) -> None:
            constructed.append(type(self).__name__)

        @classmethod
        def load(cls, *args, **kwargs):
            constructed.append(cls.__name__)
            return cls()

    class FakeProvider:
        def __init__(self, **kwargs) -> None:
            captured_providers.append(kwargs)

    class FakeScheduler:
        pass

    monkeypatch.setattr(runtime_module, "AsrModelManager", UnexpectedAdapter)
    monkeypatch.setattr(runtime_module, "NllbTranslator", UnexpectedAdapter)
    monkeypatch.setattr(runtime_module, "PiperVoiceRegistry", UnexpectedAdapter)
    monkeypatch.setattr(runtime_module, "LocalProvider", FakeProvider)
    monkeypatch.setattr(runtime_module, "InferenceScheduler", FakeScheduler)
    selected_models = (
        "faster-whisper-small",
        "nllb-200-distilled-600m-ct2-int8",
        "piper-ru-dmitri-medium",
        "piper-en-ryan-medium",
        "piper-ru-irina-medium",
        "piper-en-hfc-female-medium",
    )
    for missing_model in selected_models:
        manifest = FakeManifest(tmp_path)
        del manifest.models[missing_model]
        monkeypatch.setattr(
            runtime_module,
            "load_manifest",
            lambda path, value=manifest: value,
        )
        provider = build_local_provider(
            now_ns=lambda: 0,
            manifest_path=tmp_path / "manifest.json",
        )

        assert isinstance(provider, FakeProvider)
        captured = captured_providers[-1]
        assert captured["asr"].unavailable is True
        assert captured["translator"].unavailable is True
        assert captured["tts"].unavailable is True

    assert constructed == []
    assert not caplog.text


def test_build_local_provider_fails_safely_after_cuda_and_cpu_mt_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    manifest = FakeManifest(tmp_path)
    mt_attempts = []
    asr_constructed = []
    captured = {}

    class FailingTranslator:
        @classmethod
        def load(cls, path, *, device):
            mt_attempts.append((path, device))
            raise RuntimeError("private-mt-load-marker")

    class UnexpectedAsr:
        def __init__(self, **kwargs) -> None:
            asr_constructed.append(kwargs)

    class FakeProvider:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    class FakeScheduler:
        pass

    monkeypatch.setattr(
        runtime_module,
        "load_manifest",
        lambda path: manifest,
    )
    monkeypatch.setattr(runtime_module, "_cuda_available", lambda: True)
    monkeypatch.setattr(runtime_module, "NllbTranslator", FailingTranslator)
    monkeypatch.setattr(runtime_module, "AsrModelManager", UnexpectedAsr)
    monkeypatch.setattr(runtime_module, "LocalProvider", FakeProvider)
    monkeypatch.setattr(runtime_module, "InferenceScheduler", FakeScheduler)

    provider = build_local_provider(
        now_ns=lambda: 0,
        manifest_path=tmp_path / "manifest.json",
    )

    mt_path = VerifiedModelSource(manifest, "nllb-200-distilled-600m-ct2-int8")
    assert isinstance(provider, FakeProvider)
    assert mt_attempts == [(mt_path, "cuda"), (mt_path, "cpu")]
    assert asr_constructed == []
    assert captured["asr"].unavailable is True
    assert captured["translator"].unavailable is True
    assert captured["tts"].unavailable is True
    assert captured["mt_device"] is ComputeDevice.CPU
    assert "private-mt-load-marker" not in caplog.text


@pytest.mark.parametrize(
    "failing_component",
    ["asr", "registry", "tts", "provider"],
)
def test_adapter_construction_failure_returns_reachable_unavailable_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    failing_component: str,
) -> None:
    manifest = FakeManifest(tmp_path)
    real_provider = runtime_module.LocalProvider
    closed = []

    class FakeTranslator(Closable):
        def close(self):
            closed.append("mt")

        @classmethod
        def load(cls, path, *, device):
            return cls()

        def translate(self, text, **kwargs):
            del text, kwargs
            return "translated"

    class MaybeFailAsr(Closable):
        def close(self):
            closed.append("asr")

        def __init__(self, **kwargs):
            if failing_component == "asr":
                raise RuntimeError("private-asr-construction-marker")

    class MaybeFailRegistry(Closable):
        def close(self):
            closed.append("registry")

        def __init__(self, voice_paths):
            if failing_component == "registry":
                raise RuntimeError("private-registry-construction-marker")

    class MaybeFailTts:
        def __init__(self, registry):
            self.registry = registry
            if failing_component == "tts":
                raise RuntimeError("private-tts-construction-marker")

        def close(self):
            self.registry.close()

    def maybe_fail_provider(**kwargs):
        if failing_component == "provider" and not getattr(
            kwargs["asr"], "unavailable", False
        ):
            raise RuntimeError("private-provider-construction-marker")
        return real_provider(**kwargs)

    monkeypatch.setattr(runtime_module, "load_manifest", lambda path: manifest)
    monkeypatch.setattr(runtime_module, "_cuda_available", lambda: False)
    monkeypatch.setattr(runtime_module, "NllbTranslator", FakeTranslator)
    monkeypatch.setattr(runtime_module, "AsrModelManager", MaybeFailAsr)
    monkeypatch.setattr(runtime_module, "PiperVoiceRegistry", MaybeFailRegistry)
    monkeypatch.setattr(runtime_module, "PiperTts", MaybeFailTts)
    monkeypatch.setattr(runtime_module, "LocalProvider", maybe_fail_provider)

    provider = build_local_provider(
        now_ns=lambda: 0,
        manifest_path=tmp_path / "manifest.json",
    )

    assert (
        closed
        == {
            "asr": ["mt"],
            "registry": ["asr", "mt"],
            "tts": ["registry", "asr", "mt"],
            "provider": ["registry", "asr", "mt"],
        }[failing_component]
    )

    async def scenario() -> None:
        async def publish(batch, commit) -> None:
            commit()

        try:
            _, health = await provider.open_session(open_request(), publish)
            assert health.state is ProviderState.UNAVAILABLE
            assert health.safe_error is not None
            assert health.safe_error.message == "Required model is not loaded"
            private_markers = (
                "private-asr-construction-marker",
                "private-registry-construction-marker",
                "private-tts-construction-marker",
                "private-provider-construction-marker",
            )
            assert not any(
                marker in health.safe_error.message for marker in private_markers
            )
            assert not any(marker in caplog.text for marker in private_markers)
        finally:
            await provider.shutdown()

    asyncio.run(scenario())


def test_build_local_provider_keeps_cuda_asr_after_cuda_mt_load_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = FakeManifest(tmp_path)
    captured = {}
    mt_attempts = []

    class FallbackTranslator(Closable):
        @classmethod
        def load(cls, path, *, device):
            mt_attempts.append((path, device))
            if device == "cuda":
                raise RuntimeError("private-cuda-mt-load-marker")
            instance = cls()
            captured["mt_instance"] = instance
            return instance

        def translate(self, text, **kwargs):
            del text, kwargs
            return "translated"

    class FakeAsr(Closable):
        def __init__(self, **kwargs) -> None:
            captured["asr"] = kwargs
            captured["asr_instance"] = self

    class FakeRegistry(Closable):
        def __init__(self, voice_paths) -> None:
            captured["registry"] = self

    class FakeTts:
        def __init__(self, registry) -> None:
            captured["tts_instance"] = self

    class FakeScheduler:
        pass

    class FakeProvider:
        def __init__(self, **kwargs) -> None:
            captured["provider"] = kwargs

    monkeypatch.setattr(
        runtime_module,
        "load_manifest",
        lambda path: manifest,
    )
    monkeypatch.setattr(runtime_module, "_cuda_available", lambda: True)
    monkeypatch.setattr(
        runtime_module,
        "NllbTranslator",
        FallbackTranslator,
    )
    monkeypatch.setattr(runtime_module, "AsrModelManager", FakeAsr)
    monkeypatch.setattr(runtime_module, "PiperVoiceRegistry", FakeRegistry)
    monkeypatch.setattr(runtime_module, "PiperTts", FakeTts)
    monkeypatch.setattr(runtime_module, "InferenceScheduler", FakeScheduler)
    monkeypatch.setattr(runtime_module, "LocalProvider", FakeProvider)

    provider = build_local_provider(
        now_ns=lambda: 0,
        manifest_path=tmp_path / "manifest.json",
    )

    mt_path = VerifiedModelSource(manifest, "nllb-200-distilled-600m-ct2-int8")
    assert isinstance(provider, FakeProvider)
    assert mt_attempts == [(mt_path, "cuda"), (mt_path, "cpu")]
    assert captured["asr"]["device"] == "cuda"
    assert captured["provider"]["asr"] is captured["asr_instance"]
    assert captured["provider"]["translator"] is captured["mt_instance"]
    assert captured["provider"]["mt_device"] is ComputeDevice.CPU


def test_cuda_mt_smoke_failure_reloads_and_verifies_cpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = FakeManifest(tmp_path)
    captured = {}
    loads = []
    smoke_calls = []

    class SmokeTranslator(Closable):
        def __init__(self, device: str) -> None:
            self.device = device
            self._translator = self
            self.unloaded = False

        @classmethod
        def load(cls, path, *, device):
            instance = cls(device)
            loads.append((path, device, instance))
            return instance

        def translate(
            self,
            text,
            *,
            source_language,
            target_language,
            mode,
        ):
            smoke_calls.append(
                (self.device, text, source_language, target_language, mode)
            )
            if self.device == "cuda" and source_language is Language.RU:
                raise RuntimeError("private-cuda-inference-marker")
            return "translated"

        def unload_model(self) -> None:
            self.unloaded = True

    class FakeAdapter(Closable):
        def __init__(self, *args, **kwargs) -> None:
            if "selected_id" in kwargs:
                captured["asr_kwargs"] = kwargs
            del args, kwargs

    class FakeRegistry(FakeAdapter):
        def prepare(self) -> None:
            pass

    class FakeProvider:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(runtime_module, "load_manifest", lambda path: manifest)
    monkeypatch.setattr(runtime_module, "_cuda_available", lambda: True)
    monkeypatch.setattr(runtime_module, "NllbTranslator", SmokeTranslator)
    monkeypatch.setattr(runtime_module, "AsrModelManager", FakeAdapter)
    monkeypatch.setattr(runtime_module, "PiperVoiceRegistry", FakeRegistry)
    monkeypatch.setattr(runtime_module, "PiperTts", FakeAdapter)
    monkeypatch.setattr(runtime_module, "LocalProvider", FakeProvider)
    monkeypatch.setattr(runtime_module, "InferenceScheduler", FakeAdapter)

    provider = build_local_provider(
        now_ns=lambda: 0,
        manifest_path=tmp_path / "manifest.json",
    )

    mt_path = VerifiedModelSource(manifest, "nllb-200-distilled-600m-ct2-int8")
    assert isinstance(provider, FakeProvider)
    assert [(path, device) for path, device, _ in loads] == [
        (mt_path, "cuda"),
        (mt_path, "cpu"),
    ]
    assert loads[0][2].unloaded is True
    assert [
        (device, source, target, mode)
        for device, _, source, target, mode in smoke_calls
    ] == [
        ("cuda", Language.RU, Language.EN, TranslationMode.QUALITY_FIRST),
        ("cuda", Language.EN, Language.RU, TranslationMode.QUALITY_FIRST),
        ("cpu", Language.RU, Language.EN, TranslationMode.QUALITY_FIRST),
        ("cpu", Language.EN, Language.RU, TranslationMode.QUALITY_FIRST),
    ]
    assert all(text.strip() for _, text, _, _, _ in smoke_calls)
    assert captured["translator"] is loads[1][2]
    assert captured["asr_kwargs"]["device"] == "cuda"
    assert captured["mt_device"] is ComputeDevice.CPU


def test_cuda_mt_smoke_success_does_not_load_cpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = FakeManifest(tmp_path)
    captured = {}
    loads = []
    smoke_calls = []

    class SmokeTranslator(Closable):
        @classmethod
        def load(cls, path, *, device):
            loads.append((path, device))
            return cls()

        def translate(self, text, **kwargs):
            smoke_calls.append((text, kwargs))
            return "translated"

    class FakeAdapter(Closable):
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

    class FakeRegistry(FakeAdapter):
        def prepare(self) -> None:
            pass

    class FakeProvider:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(runtime_module, "load_manifest", lambda path: manifest)
    monkeypatch.setattr(runtime_module, "_cuda_available", lambda: True)
    monkeypatch.setattr(runtime_module, "NllbTranslator", SmokeTranslator)
    monkeypatch.setattr(runtime_module, "AsrModelManager", FakeAdapter)
    monkeypatch.setattr(runtime_module, "PiperVoiceRegistry", FakeRegistry)
    monkeypatch.setattr(runtime_module, "PiperTts", FakeAdapter)
    monkeypatch.setattr(runtime_module, "LocalProvider", FakeProvider)
    monkeypatch.setattr(runtime_module, "InferenceScheduler", FakeAdapter)

    provider = build_local_provider(
        now_ns=lambda: 0,
        manifest_path=tmp_path / "manifest.json",
    )

    assert isinstance(provider, FakeProvider)
    assert [device for _, device in loads] == ["cuda"]
    assert [
        (
            call["source_language"],
            call["target_language"],
            call["mode"],
        )
        for _, call in smoke_calls
    ] == [
        (Language.RU, Language.EN, TranslationMode.QUALITY_FIRST),
        (Language.EN, Language.RU, TranslationMode.QUALITY_FIRST),
    ]
    assert captured["mt_device"] is ComputeDevice.CUDA


def test_cpu_mt_smoke_failure_returns_unavailable_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = FakeManifest(tmp_path)
    captured = {}
    smoke_calls = []
    loaded = []
    asr_constructed = False

    class SmokeTranslator(Closable):
        def __init__(self) -> None:
            self._translator = self
            self.unloaded = False

        @classmethod
        def load(cls, path, *, device):
            assert device == "cpu"
            instance = cls()
            loaded.append(instance)
            return instance

        def translate(
            self,
            text,
            *,
            source_language,
            target_language,
            mode,
        ):
            del text
            smoke_calls.append((source_language, target_language, mode))
            return "translated" if source_language is Language.RU else " "

        def unload_model(self) -> None:
            self.unloaded = True

    class UnexpectedAsr:
        def __init__(self, **kwargs) -> None:
            del kwargs
            nonlocal asr_constructed
            asr_constructed = True

    class FakeProvider:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    class FakeScheduler:
        pass

    monkeypatch.setattr(runtime_module, "load_manifest", lambda path: manifest)
    monkeypatch.setattr(runtime_module, "_cuda_available", lambda: False)
    monkeypatch.setattr(runtime_module, "NllbTranslator", SmokeTranslator)
    monkeypatch.setattr(runtime_module, "AsrModelManager", UnexpectedAsr)
    monkeypatch.setattr(runtime_module, "LocalProvider", FakeProvider)
    monkeypatch.setattr(runtime_module, "InferenceScheduler", FakeScheduler)

    provider = build_local_provider(
        now_ns=lambda: 0,
        manifest_path=tmp_path / "manifest.json",
    )

    assert isinstance(provider, FakeProvider)
    assert smoke_calls == [
        (Language.RU, Language.EN, TranslationMode.QUALITY_FIRST),
        (Language.EN, Language.RU, TranslationMode.QUALITY_FIRST),
    ]
    assert asr_constructed is False
    assert loaded[0].unloaded is True
    assert captured["translator"].unavailable is True
    assert captured["mt_device"] is ComputeDevice.CPU

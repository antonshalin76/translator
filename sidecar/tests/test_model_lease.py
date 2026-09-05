from __future__ import annotations

import fcntl
import gc
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

from translator_sidecar.local.asr import AsrModelManager, AsrUnavailable
from translator_sidecar.local.model_lease import VerifiedModelLease, VerifiedModelSource
from translator_sidecar.local.model_manifest import (
    ManifestError,
    ManifestPolicy,
    ModelEntry,
    ModelFile,
    ModelManifest,
    ModelSource,
    RuntimeFileOps,
)
from translator_sidecar.local.mt import (
    LocalTranslationCleanupPending,
    LocalTranslationError,
    NllbTranslator,
)
from translator_sidecar.local.tts import PiperVoiceRegistry, TtsUnavailable
from translator_sidecar.provider_contract import (
    Language,
    VoiceEngine,
    VoiceGender,
    VoiceProfile,
)


def model_source(
    directory: Path, payloads: dict[str, bytes], *, write: bool = True
) -> VerifiedModelSource:
    if write:
        for name, payload in payloads.items():
            (directory / name).write_bytes(payload)
    entry = ModelEntry(
        id="fixture-model",
        role="asr",
        source=ModelSource("fixture/model", "a" * 40, "MIT"),
        languages=("ru", "en"),
        cache_path=directory,
        acquisition="download",
        files=tuple(
            ModelFile(name, len(payload), hashlib.sha256(payload).hexdigest())
            for name, payload in payloads.items()
        ),
    )
    manifest = ModelManifest(
        1,
        ManifestPolicy(1, 1, "personal_noncommercial", False, False, directory, ()),
        {entry.id: entry},
        0,
        directory,
    )
    return VerifiedModelSource(manifest, entry.id)


@pytest.mark.parametrize("mutation", ["replace", "overwrite", "unlink"])
def test_snapshot_remains_verified_after_cache_mutation(tmp_path: Path, mutation: str):
    source = model_source(tmp_path, {"model.bin": b"approved"})
    with source.acquire() as lease:
        target = tmp_path / "model.bin"
        if mutation == "replace":
            replacement = tmp_path / "replacement"
            replacement.write_bytes(b"attacker")
            replacement.replace(target)
        elif mutation == "overwrite":
            target.write_bytes(b"attacker")
        else:
            target.unlink()
        assert lease.read_bytes("model.bin") == b"approved"
        with open(lease.path("model.bin"), "r+b") as writer:
            with pytest.raises(OSError):
                os.write(writer.fileno(), b"x")


def test_snapshot_rejects_in_place_mutation_during_acquisition(tmp_path: Path):
    source = model_source(tmp_path, {"model.bin": b"approved"})

    class MutatingFiles(RuntimeFileOps):
        def open_stable(self, path, **kwargs):
            result = super().open_stable(path, **kwargs)
            path.write_bytes(b"attacker")
            return result

    before = len(os.listdir("/proc/self/fd"))
    with pytest.raises(ManifestError, match="integrity"):
        VerifiedModelLease(source.manifest, source.model_id, filesystem=MutatingFiles())
    assert len(os.listdir("/proc/self/fd")) == before


def test_snapshot_rejects_partial_set_and_closes_all_descriptors(tmp_path: Path):
    source = model_source(tmp_path, {"model.bin": b"approved", "config.json": b"{}"})
    (tmp_path / "config.json").write_bytes(b"xx")
    before = len(os.listdir("/proc/self/fd"))
    with pytest.raises(ManifestError, match="integrity"):
        source.acquire()
    assert len(os.listdir("/proc/self/fd")) == before


def test_snapshot_verifies_sealed_destination_after_preseal_tampering(
    tmp_path, monkeypatch
):
    source = model_source(tmp_path, {"model.bin": b"approved"})
    original_fcntl = fcntl.fcntl

    def tamper_before_seal(descriptor, operation, seals):
        os.pwrite(descriptor, b"attacker", 0)
        return original_fcntl(descriptor, operation, seals)

    monkeypatch.setattr(fcntl, "fcntl", tamper_before_seal)
    before = len(os.listdir("/proc/self/fd"))
    with pytest.raises(ManifestError, match="integrity"):
        source.acquire()
    assert len(os.listdir("/proc/self/fd")) == before


def test_snapshot_readers_have_independent_offsets_and_cannot_resize(tmp_path: Path):
    source = model_source(tmp_path, {"model.bin": b"approved"})
    with source.acquire() as lease:
        first = lease.files()["model.bin"]
        second = lease.files()["model.bin"]
        assert first.read(3) == b"app"
        assert second.read() == b"approved"
        assert first.read() == b"roved"
        for size in (0, 100):
            with pytest.raises(OSError):
                os.truncate(lease.path("model.bin"), size)
        seals = fcntl.fcntl(first.fileno(), fcntl.F_GET_SEALS)
        assert seals & fcntl.F_SEAL_SEAL
    assert first.closed and second.closed
    with pytest.raises(ManifestError, match="closed"):
        lease.path("model.bin")


def test_snapshot_100_lifecycles_and_closed_lease_cannot_reuse_fd(tmp_path: Path):
    source = model_source(tmp_path, {"model.bin": b"approved"})
    before = len(os.listdir("/proc/self/fd"))
    for _ in range(100):
        lease = source.acquire()
        files = lease.files()
        assert files["model.bin"].read() == b"approved"
        lease.close()
        lease.close()
        with open(tmp_path / "model.bin", "rb"):
            with pytest.raises(ManifestError, match="closed"):
                lease.path("model.bin")
    assert len(os.listdir("/proc/self/fd")) == before


def test_snapshot_fdopen_failure_closes_new_memfd(tmp_path: Path, monkeypatch):
    source = model_source(tmp_path, {"model.bin": b"approved"})

    def fail(*args, **kwargs):
        raise OSError("descriptor wrapper failed")

    monkeypatch.setattr(os, "fdopen", fail)
    before = len(os.listdir("/proc/self/fd"))
    with pytest.raises(ManifestError):
        source.acquire()
    assert len(os.listdir("/proc/self/fd")) == before


@pytest.mark.parametrize("symlink", [False, True])
def test_runtime_admission_closes_descriptor_when_path_disappears_after_open(
    tmp_path, monkeypatch, symlink
):
    target = tmp_path / "model.bin"
    target.write_bytes(b"model")
    if symlink:
        link = tmp_path / "link"
        link.symlink_to(target)
        target = link
    original_open = os.open

    def disappearing_open(path, *args, **kwargs):
        descriptor = original_open(path, *args, **kwargs)
        Path(path).unlink()
        return descriptor

    before = len(os.listdir("/proc/self/fd"))
    monkeypatch.setattr(os, "open", disappearing_open)
    with pytest.raises(OSError):
        RuntimeFileOps().open_stable(
            target, allowed_symlink_root=tmp_path if symlink else None
        )
    assert len(os.listdir("/proc/self/fd")) == before


@pytest.mark.parametrize("failure", ["seal", "read"])
def test_snapshot_system_failure_closes_all_resources(
    tmp_path: Path, monkeypatch, failure
):
    source = model_source(tmp_path, {"model.bin": b"approved"})

    def fail(*args, **kwargs):
        raise OSError("snapshot operation failed")

    if failure == "seal":
        monkeypatch.setattr(fcntl, "fcntl", fail)
    else:
        monkeypatch.setattr(os, "read", fail)
    before = len(os.listdir("/proc/self/fd"))
    with pytest.raises(ManifestError):
        source.acquire()
    assert len(os.listdir("/proc/self/fd")) == before


class NativeModel:
    def unload_model(self):
        pass


_VOICE = VoiceProfile(
    language=Language.EN, gender=VoiceGender.MALE, engine=VoiceEngine.PIPER
)


def test_adapters_100_lifecycles_release_all_snapshot_and_reader_descriptors(tmp_path):
    source = model_source(
        tmp_path,
        {
            "model.bin": b"approved",
            "tokenizer.json": b"{}",
            "sentencepiece.bpe.model": b"sp",
            "voice.onnx": b"voice",
            "voice.onnx.json": b"{}",
        },
    )
    gc.collect()
    before = len(os.listdir("/proc/self/fd"))
    for _ in range(100):
        asr = AsrModelManager(
            selected_id="small",
            model_paths={"small": source},
            device="cpu",
            model_factory=lambda *_args, **_kwargs: NativeModel(),
        )
        asr.prepare()
        translator = NllbTranslator.load(
            source,
            device="cpu",
            translator_factory=lambda *_args, **_kwargs: NativeModel(),
            tokenizer_factory=lambda _payload: NativeModel(),
        )
        registry = PiperVoiceRegistry(
            {(Language.EN, VoiceGender.MALE): source},
            voice_factory=lambda *_args, **_kwargs: NativeModel(),
        )
        registry.get(_VOICE)
        registry.close()
        translator.close()
        asr.close()
        registry.close()
        translator.close()
        asr.close()
        assert len(os.listdir("/proc/self/fd")) == before


@pytest.mark.parametrize("component", ["asr", "mt", "tokenizer", "tts"])
def test_adapter_failed_constructor_releases_snapshots(tmp_path, component):
    source = model_source(
        tmp_path,
        {
            "model.bin": b"approved",
            "tokenizer.json": b"{}",
            "sentencepiece.bpe.model": b"sp",
            "voice.onnx": b"voice",
            "voice.onnx.json": b"{}",
        },
    )

    def fail(*args, **kwargs):
        raise RuntimeError("native constructor failed")

    gc.collect()
    before = len(os.listdir("/proc/self/fd"))
    if component == "asr":
        asr = AsrModelManager(
            selected_id="small",
            model_paths={"small": source},
            device="cpu",
            model_factory=fail,
        )
        with pytest.raises(AsrUnavailable):
            asr.prepare()
        asr.close()
    elif component in {"mt", "tokenizer"}:
        with pytest.raises(LocalTranslationError):
            NllbTranslator.load(
                source,
                device="cpu",
                translator_factory=fail
                if component == "mt"
                else lambda *_args, **_kwargs: NativeModel(),
                tokenizer_factory=fail,
            )
    else:
        registry = PiperVoiceRegistry(
            {(Language.EN, VoiceGender.MALE): source}, voice_factory=fail
        )
        with pytest.raises(TtsUnavailable):
            registry.get(_VOICE)
        registry.close()
    gc.collect()
    assert len(os.listdir("/proc/self/fd")) == before


def test_whisper_mapping_mutation_and_ambient_config_cannot_reopen_cache(tmp_path):
    source = model_source(tmp_path, {"model.bin": b"approved", "tokenizer.json": b"{}"})
    (tmp_path / "preprocessor_config.json").write_text('{"feature_size": 999}')
    observed = []

    def factory(identifier, *, files, **kwargs):
        (tmp_path / "model.bin").write_bytes(b"attacker")
        assert files.pop("tokenizer.json") == b"{}"
        assert files.pop("preprocessor_config.json") == b"{}"
        assert files["model.bin"].read() == b"approved"
        assert not Path(identifier).is_dir()
        observed.append(identifier)
        return NativeModel()

    asr = AsrModelManager(
        selected_id="small",
        model_paths={"small": source},
        device="cpu",
        model_factory=factory,
    )
    asr.prepare()
    assert Path(observed[0]).read_bytes() == b"approved"
    asr.close()
    assert not Path(observed[0]).exists()


def test_retained_native_voice_keeps_sealed_paths_alive_until_destroyed(tmp_path):
    source = model_source(tmp_path, {"voice.onnx": b"voice", "voice.onnx.json": b"{}"})
    paths = []

    def factory(path, *, config_path, **kwargs):
        paths.extend((path, config_path))
        return NativeModel()

    registry = PiperVoiceRegistry(
        {(Language.EN, VoiceGender.MALE): source}, voice_factory=factory
    )
    voice = registry.get(_VOICE)
    registry.close()
    assert Path(paths[0]).read_bytes() == b"voice"
    assert Path(paths[1]).read_bytes() == b"{}"
    with pytest.raises(TtsUnavailable):
        registry.get(_VOICE)
    del voice
    assert all(not Path(path).exists() for path in paths)


def test_retained_native_translator_keeps_snapshot_alive_until_destroyed(tmp_path):
    source = model_source(
        tmp_path, {"model.bin": b"model", "sentencepiece.bpe.model": b"sp"}
    )
    native = NativeModel()
    translator = NllbTranslator.load(
        source,
        device="cpu",
        translator_factory=lambda *_args, _native=native, **_kwargs: _native,
        tokenizer_factory=lambda _payload: NativeModel(),
    )
    path = translator.model_path
    translator.close()
    assert path.read_bytes() == b"model"
    del native
    assert not path.exists()


def test_asr_retained_inner_native_keeps_snapshot_and_blocks_release(tmp_path):
    source = model_source(tmp_path, {"model.bin": b"model", "tokenizer.json": b"{}"})
    retained = [NativeModel()]
    paths = []

    def factory(identifier, **kwargs):
        paths.append(identifier)
        wrapper = NativeModel()
        wrapper.model = retained[0]
        return wrapper

    asr = AsrModelManager(
        selected_id="small",
        model_paths={"small": source},
        device="cpu",
        model_factory=factory,
    )
    asr.prepare()
    assert asr.release() is False
    assert Path(paths[0]).read_bytes() == b"model"
    with pytest.raises(AsrUnavailable, match="cleanup"):
        asr.close()
    retained.clear()
    asr.close()
    assert not Path(paths[0]).exists()


def test_piper_retained_inner_session_keeps_snapshot_after_voice_destruction(tmp_path):
    source = model_source(tmp_path, {"voice.onnx": b"voice", "voice.onnx.json": b"{}"})
    retained = [NativeModel()]
    paths = []

    def factory(identifier, **kwargs):
        paths.append(identifier)
        wrapper = NativeModel()
        wrapper.session = retained[0]
        return wrapper

    registry = PiperVoiceRegistry(
        {(Language.EN, VoiceGender.MALE): source}, voice_factory=factory
    )
    registry.get(_VOICE)
    registry.close()
    assert Path(paths[0]).read_bytes() == b"voice"
    retained.clear()
    assert not Path(paths[0]).exists()


def test_mt_close_retries_native_unload_and_retains_owner_on_failure(tmp_path):
    class FailOnce(NativeModel):
        attempts = 0

        def unload_model(self):
            self.attempts += 1
            if self.attempts == 1:
                raise RuntimeError("private-unload-marker")

    source = model_source(
        tmp_path, {"model.bin": b"model", "sentencepiece.bpe.model": b"sp"}
    )
    native = FailOnce()
    translator = NllbTranslator.load(
        source,
        device="cpu",
        translator_factory=lambda *_args, **_kwargs: native,
        tokenizer_factory=lambda *_args: NativeModel(),
    )
    with pytest.raises(LocalTranslationError, match="cleanup"):
        translator.close()
    assert translator._translator is native
    assert translator.model_path.read_bytes() == b"model"
    translator.close()
    assert native.attempts == 2
    assert translator._translator is None


def test_mt_tokenizer_failure_transfers_failed_native_cleanup_to_owner(tmp_path):
    attempts = []

    class Native(NativeModel):
        def unload_model(self):
            attempts.append("unload")
            if len(attempts) == 1:
                raise RuntimeError("unload failed")

    def fail_tokenizer(_payload):
        raise RuntimeError("tokenizer failed")

    source = model_source(
        tmp_path, {"model.bin": b"model", "sentencepiece.bpe.model": b"sp"}
    )
    with pytest.raises(LocalTranslationCleanupPending) as raised:
        NllbTranslator.load(
            source,
            device="cpu",
            translator_factory=lambda *_args, **_kwargs: Native(),
            tokenizer_factory=fail_tokenizer,
        )
    adapter = raised.value.translator
    assert adapter.unavailable
    assert adapter._translator is not None
    assert adapter.model_path.read_bytes() == b"model"
    adapter.close()
    assert attempts == ["unload", "unload"]
    assert adapter._translator is None


def test_asr_failed_native_unload_retains_owner_for_retry(tmp_path):
    attempts = []

    class Native(NativeModel):
        def unload_model(self):
            attempts.append("unload")
            if len(attempts) == 1:
                raise RuntimeError("unload failed")

    def factory(identifier, **kwargs):
        wrapper = NativeModel()
        wrapper.model = Native()
        return wrapper

    source = model_source(tmp_path, {"model.bin": b"model", "tokenizer.json": b"{}"})
    asr = AsrModelManager(
        selected_id="small",
        model_paths={"small": source},
        device="cpu",
        model_factory=factory,
    )
    asr.prepare()
    with pytest.raises(AsrUnavailable, match="cleanup"):
        asr.close()
    assert asr.unavailable
    assert asr._model is not None
    asr.close()
    assert attempts == ["unload", "unload"]
    assert asr._model is None


def test_installed_whisper_consumes_verified_mapping_with_library_defaults(
    tmp_path, monkeypatch
):
    import ctranslate2
    import tokenizers

    tokenizer = tokenizers.Tokenizer(
        tokenizers.models.WordLevel({"<unk>": 0}, unk_token="<unk>")
    )
    source = model_source(
        tmp_path,
        {"model.bin": b"approved", "tokenizer.json": tokenizer.to_str().encode()},
    )
    (tmp_path / "preprocessor_config.json").write_text('{"feature_size": 999}')
    observed = []

    class NativeWhisper(NativeModel):
        def __init__(self, identifier, *, files, **kwargs):
            assert not Path(identifier).is_dir()
            assert files["model.bin"].read() == b"approved"
            assert "tokenizer.json" not in files
            assert "preprocessor_config.json" not in files
            observed.append(identifier)

    monkeypatch.setattr(ctranslate2.models, "Whisper", NativeWhisper)
    asr = AsrModelManager(
        selected_id="small", model_paths={"small": source}, device="cpu"
    )
    asr.prepare()
    assert asr._model.feature_extractor.mel_filters.shape[0] == 80
    assert asr._model.hf_tokenizer.get_vocab() == {"<unk>": 0}
    asr.close()
    assert all(not Path(path).exists() for path in observed)


def _sentencepiece_conformance(tmp_path):
    from io import BytesIO

    from sentencepiece import SentencePieceTrainer

    encoded = BytesIO()
    SentencePieceTrainer.train(
        sentence_iterator=iter(["hello world", "hello translator"]),
        model_writer=encoded,
        vocab_size=16,
        hard_vocab_limit=False,
        minloglevel=2,
    )
    source = model_source(
        tmp_path, {"model.bin": b"model", "sentencepiece.bpe.model": encoded.getvalue()}
    )
    translator = NllbTranslator.load(
        source, device="cpu", translator_factory=lambda *_args, **_kwargs: NativeModel()
    )
    assert translator.count_tokens("hello world") > 0
    translator.close()


def test_installed_sentencepiece_loads_verified_model_proto(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            (
                "import sys; from pathlib import Path; "
                "sys.path.insert(0, sys.argv[1]); "
                "from test_model_lease import _sentencepiece_conformance; "
                "_sentencepiece_conformance(Path(sys.argv[2])); "
                "print('sentencepiece-model-proto:PASS')"
            ),
            str(Path(__file__).parent),
            str(tmp_path),
        ],
        check=False,
        capture_output=True,
        timeout=20,
    )
    assert completed.returncode == 0
    assert completed.stdout == b"sentencepiece-model-proto:PASS\n"
    assert completed.stderr == b""

"""Bounded integrity tests; no installed models or GPU are touched."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

from translator_sidecar.local.model_manifest import (
    ManifestError,
    ManifestPolicy,
    ModelEntry,
    ModelFile,
    ModelManifest,
    ModelSource,
)

_SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "translator_product_freeze.py"
)
_SPEC = importlib.util.spec_from_file_location("translator_product_freeze", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
freeze = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(freeze)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, ModelManifest, Path]:
    monkeypatch.setattr(freeze, "_source_tree", lambda _repository: "a" * 40)
    entries = {}
    profiles = {
        "ru_male": "piper-ru-male",
        "ru_female": "piper-ru-female",
        "en_male": "piper-en-male",
        "en_female": "piper-en-female",
    }
    for model_id, role in [
        ("turbo", "asr"),
        ("small", "asr"),
        ("nllb", "mt"),
        *((voice, "tts") for voice in profiles.values()),
    ]:
        directory = tmp_path / "models" / model_id
        directory.mkdir(parents=True)
        data = model_id.encode()
        (directory / "model.bin").write_bytes(data)
        entries[model_id] = ModelEntry(
            id=model_id,
            role=role,
            source=ModelSource("test/source", "a" * 40, "MIT"),
            languages=("ru", "en"),
            cache_path=directory,
            acquisition="download",
            files=(ModelFile("model.bin", len(data), _sha(data)),),
        )
    manifest = ModelManifest(
        schema_version=1,
        policy=ManifestPolicy(
            100, 0, "personal_noncommercial", False, False, tmp_path, ()
        ),
        models=entries,
        planned_download_bytes=0,
        cache_root=tmp_path,
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    corpus = tmp_path / "corpus.json"
    corpus.write_text('{"samples":[]}', encoding="utf-8")
    observation = {
        "schema_version": 1,
        "requested_models": {"asr": "turbo", "mt": "nllb", "tts": profiles},
        "effective_models": {"asr": "turbo", "mt": "nllb", "tts": profiles},
        "runtime": {
            "asr_device": "cuda",
            "mt_device": "cuda",
            "tts_device": "cpu",
            "driver": "fixture-driver",
            "build_profile": "release",
            "fallback_branch": "none",
        },
        "corpora": [
            {
                "id": "holdout-v1",
                "source_id": "licensed-source-1",
                "path": str(corpus),
                "sha256": _sha(corpus.read_bytes()),
            }
        ],
    }
    observation_path = tmp_path / "observation.json"
    observation_path.write_text(json.dumps(observation), encoding="utf-8")
    return manifest_path, observation_path, manifest, corpus


def _build(
    tmp_path: Path, fixture: tuple[Path, Path, ModelManifest, Path], **kwargs: object
) -> dict:
    manifest_path, observation_path, manifest, _ = fixture
    return freeze.build_receipt(
        repository=tmp_path,
        manifest_path=manifest_path,
        observation_path=observation_path,
        manifest=manifest,
        **kwargs,
    )


def _edit_observation(path: Path, edit) -> None:
    document = json.loads(path.read_text(encoding="utf-8"))
    edit(document)
    path.write_text(json.dumps(document), encoding="utf-8")


def test_exact_receipt_binds_models_runtime_and_corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    receipt = _build(tmp_path, fixture)
    assert receipt["status"] == "exact"
    assert receipt["candidate_tree"] == "a" * 40
    assert receipt["model_manifest_sha256"] == _sha(fixture[0].read_bytes())
    assert receipt["observation_sha256"] == _sha(fixture[1].read_bytes())
    assert len(receipt["model_files"]) == 6
    assert receipt["corpora"] == [
        {
            "id": "holdout-v1",
            "source_id": "licensed-source-1",
            "manifest_sha256": _sha(fixture[3].read_bytes()),
        }
    ]
    assert all("/" not in row["model_id"] for row in receipt["model_files"])


def test_silent_fallback_fails_and_explicit_fallback_is_separate_cell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    _edit_observation(
        fixture[1], lambda row: row["effective_models"].__setitem__("asr", "small")
    )
    with pytest.raises(ValueError, match="requested/effective"):
        _build(tmp_path, fixture)
    _edit_observation(
        fixture[1],
        lambda row: row["runtime"].__setitem__("fallback_branch", "asr_cpu_small"),
    )
    with pytest.raises(ValueError, match="requested/effective"):
        _build(tmp_path, fixture)
    receipt = _build(tmp_path, fixture, allow_fallback=True)
    assert receipt["status"] == "fallback_separate_cell"
    assert len(receipt["model_files"]) == 7


def test_missing_or_corrupt_pinned_model_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    model_path = tmp_path / "models" / "turbo" / "model.bin"
    model_path.unlink()
    with pytest.raises(ManifestError, match="missing"):
        _build(tmp_path, fixture)
    model_path.write_bytes(b"tamro")
    with pytest.raises(ManifestError, match="checksum"):
        _build(tmp_path, fixture)


def test_corpus_hash_and_unpinned_model_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    fixture[3].write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="corpus source manifest hash"):
        _build(tmp_path, fixture)
    fixture[3].write_text('{"samples":[]}', encoding="utf-8")
    _edit_observation(
        fixture[1], lambda row: row["requested_models"].__setitem__("mt", "unlisted")
    )
    with pytest.raises(ValueError, match="requested/effective"):
        _build(tmp_path, fixture)
    _edit_observation(
        fixture[1], lambda row: row["effective_models"].__setitem__("mt", "unlisted")
    )
    with pytest.raises(ValueError, match="model is missing"):
        _build(tmp_path, fixture)


def test_source_tree_rejects_uncommitted_changes(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "config",
            "user.email",
            "fixture@example.invalid",
        ],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.name", "Fixture"], check=True
    )
    (repository / "tracked").write_text("v1", encoding="utf-8")
    subprocess.run(["git", "-C", str(repository), "add", "tracked"], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "commit", "-qm", "fixture"], check=True
    )
    assert len(freeze._source_tree(repository)) == 40
    (repository / "tracked").write_text("v2", encoding="utf-8")
    with pytest.raises(ValueError, match="not clean"):
        freeze._source_tree(repository)

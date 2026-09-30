"""Fail-closed provenance receipt for a committed product-evaluation candidate.

The observation is supplied by the evaluator, not inferred from runtime logs.
This tool checks its declared identities and installed model bytes; it does not
certify that the live provider actually selected those models.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from translator_sidecar.local.model_manifest import ModelManifest, load_manifest

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PROFILE_ROLES = ("ru_male", "ru_female", "en_male", "en_female")
_MODEL_ROLES = ("asr", "mt", "tts")
_MAX_METADATA_BYTES = 10 * 1024 * 1024


def _sha256(path: Path, *, max_bytes: int | None = None) -> str:
    if path.is_symlink():
        raise ValueError("metadata symlink is not allowed")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            size += len(chunk)
            if max_bytes is not None and size > max_bytes:
                raise ValueError("metadata file exceeds size limit")
            digest.update(chunk)
    return digest.hexdigest()


def _json_file(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise ValueError("metadata symlink is not allowed")
    if path.stat().st_size > _MAX_METADATA_BYTES:
        raise ValueError("metadata file exceeds size limit")
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise TypeError("observation must be a JSON object")
    return document


def _source_tree(repository: Path) -> str:
    def git(*arguments: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    if Path(git("rev-parse", "--show-toplevel")).resolve() != repository.resolve():
        raise ValueError("repository must be the Git root")
    if git("status", "--porcelain", "--untracked-files=all"):
        raise ValueError("candidate source tree is not clean")
    tree = git("rev-parse", "HEAD^{tree}")
    if not re.fullmatch(r"[0-9a-f]{40}", tree):
        raise ValueError("candidate source tree identity is invalid")
    return tree


def _model_ids(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != set(_MODEL_ROLES):
        raise ValueError("ASR, MT and TTS model identities are required")
    if not all(isinstance(value[role], str) and value[role] for role in ("asr", "mt")):
        raise ValueError("ASR and MT model identities are required")
    voices = value["tts"]
    if (
        not isinstance(voices, dict)
        or set(voices) != set(_PROFILE_ROLES)
        or not all(
            isinstance(model_id, str) and model_id for model_id in voices.values()
        )
    ):
        raise ValueError("all four TTS profile model identities are required")
    return value


def _runtime(value: object) -> dict[str, str]:
    fields = {
        "asr_device",
        "mt_device",
        "tts_device",
        "driver",
        "build_profile",
        "fallback_branch",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(
            "runtime device, driver, profile and fallback fields are required"
        )
    if not all(
        isinstance(item, str) and item.strip() and item != "unknown"
        for item in value.values()
    ):
        raise ValueError("runtime identity cannot be empty or unknown")
    if any(
        value[key] not in {"cpu", "cuda"}
        for key in ("asr_device", "mt_device", "tts_device")
    ):
        raise ValueError("runtime device is invalid")
    return value


def _corpora(value: object) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise ValueError("at least one corpus source is required")
    result = []
    identities: set[str] = set()
    for row in value:
        if not isinstance(row, dict) or set(row) != {
            "id",
            "source_id",
            "path",
            "sha256",
        }:
            raise ValueError("corpus source identity is incomplete")
        corpus_id, source_id, expected = row["id"], row["source_id"], row["sha256"]
        if (
            not isinstance(corpus_id, str)
            or not corpus_id
            or corpus_id in identities
            or not isinstance(source_id, str)
            or not source_id
            or not isinstance(expected, str)
            or not _SHA256.fullmatch(expected)
        ):
            raise ValueError("corpus source identity is invalid or duplicated")
        path = Path(row["path"])
        if (
            not path.is_absolute()
            or path.is_symlink()
            or _sha256(path, max_bytes=_MAX_METADATA_BYTES) != expected
        ):
            raise ValueError("corpus source manifest hash or path changed")
        identities.add(corpus_id)
        result.append(
            {"id": corpus_id, "source_id": source_id, "manifest_sha256": expected}
        )
    return result


def build_receipt(
    *,
    repository: Path,
    manifest_path: Path,
    observation_path: Path,
    allow_fallback: bool = False,
    manifest: ModelManifest | None = None,
) -> dict[str, Any]:
    """Validate a committed candidate and every selected pinned model file."""
    tree = _source_tree(repository)
    manifest_sha256 = _sha256(manifest_path, max_bytes=_MAX_METADATA_BYTES)
    model_manifest = manifest or load_manifest(manifest_path)
    observation = _json_file(observation_path)
    if observation.get("schema_version") != 1:
        raise ValueError("unsupported observation schema")
    if set(observation) != {
        "schema_version",
        "requested_models",
        "effective_models",
        "runtime",
        "corpora",
    }:
        raise ValueError("observation fields are incomplete or unexpected")
    requested = _model_ids(observation.get("requested_models"))
    effective = _model_ids(observation.get("effective_models"))
    runtime = _runtime(observation.get("runtime"))
    different = requested != effective
    if different and (not allow_fallback or runtime["fallback_branch"] == "none"):
        raise ValueError("requested/effective model mismatch without admitted fallback")
    if not different and runtime["fallback_branch"] != "none" and not allow_fallback:
        raise ValueError("fallback branch requires explicit admission")
    corpora = _corpora(observation.get("corpora"))

    selected = {requested["asr"], requested["mt"], effective["asr"], effective["mt"]}
    selected.update(requested["tts"].values())
    selected.update(effective["tts"].values())
    for role in ("asr", "mt"):
        for model_id in (requested[role], effective[role]):
            model = model_manifest.models.get(model_id)
            if (
                model is None
                or model.role != role
                or not {"ru", "en"}.issubset(model.languages)
            ):
                raise ValueError(
                    f"{role} model is missing, wrong role or language: {model_id}"
                )
    for models in (requested["tts"], effective["tts"]):
        for profile, model_id in models.items():
            model = model_manifest.models.get(model_id)
            if (
                model is None
                or model.role != "tts"
                or profile[:2] not in model.languages
            ):
                raise ValueError(
                    f"TTS model is missing, wrong role or language: {model_id}"
                )
    files = []
    for model_id in sorted(selected):
        model = model_manifest.models.get(model_id)
        if model is None or not model.files:
            raise ValueError(f"model is missing from pinned manifest: {model_id}")
        for entry in model.files:
            model_manifest.resolve_runtime_file(model_id, entry.path)
            files.append(
                {
                    "model_id": model_id,
                    "path": entry.path,
                    "size_bytes": entry.size_bytes,
                    "sha256": entry.sha256,
                }
            )

    return {
        "schema_version": 1,
        "status": "fallback_separate_cell"
        if runtime["fallback_branch"] != "none" or different
        else "exact",
        "candidate_tree": tree,
        "model_manifest_sha256": manifest_sha256,
        "observation_sha256": _sha256(observation_path, max_bytes=_MAX_METADATA_BYTES),
        "requested_models": requested,
        "effective_models": effective,
        "runtime": runtime,
        "model_files": files,
        "corpora": corpora,
        "limitation": "effective identities are evaluator-declared, not independently observed runtime telemetry",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--allow-fallback", action="store_true")
    args = parser.parse_args()
    receipt = build_receipt(
        repository=args.repository,
        manifest_path=args.manifest,
        observation_path=args.observation,
        allow_fallback=args.allow_fallback,
    )
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

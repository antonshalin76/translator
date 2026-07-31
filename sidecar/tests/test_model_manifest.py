from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import BinaryIO, Iterable

import pytest

from translator_sidecar.local.model_manifest import (
    DownloadLedger,
    FilesystemOps,
    InstallDurabilityError,
    ManifestError,
    ModelDownloader,
    load_manifest,
)


MIB = 1024 * 1024
GIB = 1024 * MIB


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _manifest(
    cache_path: Path,
    *,
    size_bytes: int = 7,
    sha256: str | None = None,
    license_id: str = "CC-BY-NC-4.0",
    revision: str = "a" * 40,
) -> dict[str, object]:
    payload = b"payload"
    return {
        "schema_version": 1,
        "policy": {
            "download_budget_bytes": 2 * GIB,
            "post_download_free_floor_bytes": 20 * GIB,
            "usage_mode": "personal_noncommercial",
            "redistribution": False,
            "certified_or_safety_critical": False,
            "staging_path": str(cache_path / ".staging"),
            "redirect_hosts": [
                "cdn-lfs.huggingface.co",
                "cdn-lfs-us-1.hf.co",
                "cdn-lfs-eu-1.hf.co",
                "cas-bridge.xethub.hf.co",
            ],
        },
        "models": [
            {
                "id": "selected-mt",
                "role": "mt",
                "source": {
                    "repository": "owner/model",
                    "revision": revision,
                    "license": license_id,
                },
                "languages": ["ru", "en"],
                "cache_path": str(cache_path),
                "acquisition": "download",
                "files": [
                    {
                        "path": "model.bin",
                        "size_bytes": size_bytes,
                        "sha256": sha256 or _sha256(payload),
                        "source_url": (
                            "https://huggingface.co/owner/model/resolve/"
                            f"{revision}/model.bin"
                        ),
                        "source_path": "model.bin",
                    }
                ],
            }
        ],
    }


def _write_manifest(tmp_path: Path, document: dict[str, object]) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _irina_manifest(tmp_path: Path) -> dict[str, object]:
    revision = "0d907f158acc877ddeebcbf827659ee13bea8bcd"
    cache_path = tmp_path / "cache" / "piper"
    document = _manifest(cache_path, license_id="MIT", revision=revision)
    model = document["models"][0]  # type: ignore[index]
    model.update(  # type: ignore[union-attr]
        {
            "id": "piper-ru-irina-medium",
            "role": "tts",
            "source": {
                "repository": "rhasspy/piper-voices",
                "revision": revision,
                "license": "MIT",
                "dataset_license": "Unknown",
                "license_waiver": "PIPER_RU_IRINA_PERSONAL_LOCAL_V1",
            },
            "languages": ["ru"],
            "files": [
                {
                    "path": "ru_RU-irina-medium.onnx",
                    "size_bytes": 63_201_294,
                    "sha256": (
                        "8ff38212d23da300bbe3705c645e6e5b"
                        "9475f0bfde01558eb17813e22acaaaaa"
                    ),
                    "source_url": (
                        "https://huggingface.co/rhasspy/piper-voices/resolve/"
                        f"{revision}/ru/ru_RU/irina/medium/"
                        "ru_RU-irina-medium.onnx"
                    ),
                    "source_path": (
                        "ru/ru_RU/irina/medium/ru_RU-irina-medium.onnx"
                    ),
                },
                {
                    "path": "ru_RU-irina-medium.onnx.json",
                    "size_bytes": 4_765,
                    "sha256": (
                        "c2ec28bb38e2b59e93b959b3e40348c"
                        "1afebbd272f30fed5d41205d08e98a9d7"
                    ),
                    "source_url": (
                        "https://huggingface.co/rhasspy/piper-voices/resolve/"
                        f"{revision}/ru/ru_RU/irina/medium/"
                        "ru_RU-irina-medium.onnx.json"
                    ),
                    "source_path": (
                        "ru/ru_RU/irina/medium/"
                        "ru_RU-irina-medium.onnx.json"
                    ),
                },
            ],
        }
    )
    return document


def _two_file_manifest(tmp_path: Path) -> dict[str, object]:
    document = _manifest(tmp_path / "cache")
    revision = "a" * 40
    document["models"][0]["files"].append(  # type: ignore[index]
        {
            "path": "config.json",
            "size_bytes": 2,
            "sha256": _sha256(b"{}"),
            "source_url": (
                "https://huggingface.co/owner/model/resolve/"
                f"{revision}/config.json"
            ),
            "source_path": "config.json",
        }
    )
    return document


def _hf_reuse_manifest(
    tmp_path: Path,
) -> tuple[dict[str, object], Path, Path]:
    revision = "a" * 40
    repository_root = tmp_path / "models--owner--model"
    snapshot = repository_root / "snapshots" / revision
    blobs = repository_root / "blobs"
    snapshot.mkdir(parents=True)
    blobs.mkdir()
    payload = b"trusted"
    blob = blobs / _sha256(payload)
    blob.write_bytes(payload)
    target = snapshot / "model.bin"
    target.symlink_to(Path("../../blobs") / blob.name)
    document = _manifest(
        snapshot,
        size_bytes=len(payload),
        sha256=_sha256(payload),
        revision=revision,
    )
    model = document["models"][0]  # type: ignore[index]
    model["acquisition"] = "reuse"  # type: ignore[index]
    model_file = model["files"][0]  # type: ignore[index]
    model_file.pop("source_url")  # type: ignore[union-attr]
    model_file.pop("source_path")  # type: ignore[union-attr]
    return document, target, blob


class RecordingFilesystemOps(FilesystemOps):
    def __init__(
        self,
        *,
        free_bytes: int | list[int] = 100 * GIB,
        staging_device: int = 7,
        final_device: int = 7,
        fail_at: str | None = None,
    ) -> None:
        self.calls: list[tuple[str, Path]] = []
        self.free_bytes = free_bytes
        self.staging_device = staging_device
        self.final_device = final_device
        self.fail_at = fail_at
        self.open_fds: set[int] = set()
        self.fd_paths: dict[int, Path] = {}
        self.model_committed = False

    def ensure_directory(self, path: Path) -> None:
        self.calls.append(("ensure_directory", path))
        super().ensure_directory(path)

    def available_bytes(self, path: Path) -> int:
        self.calls.append(("available_bytes", path))
        if self.fail_at == "available_bytes":
            raise OSError("probe failed")
        if isinstance(self.free_bytes, list):
            return self.free_bytes.pop(0)
        return self.free_bytes

    def device_id(self, path: Path) -> int:
        self.calls.append(("device_id", path))
        if self.fail_at == "device_id":
            raise OSError("device probe failed")
        if path.name == ".staging":
            return self.staging_device
        return self.final_device

    def open_exclusive(self, path: Path) -> BinaryIO:
        self.calls.append(("open_exclusive", path))
        if self.fail_at == "open_exclusive":
            raise OSError("exclusive open failed")
        return super().open_exclusive(path)

    def fsync_file(self, file: BinaryIO) -> None:
        self.calls.append(("fsync_file", Path(f"fd-{file.fileno()}")))
        if self.fail_at == "fsync_file":
            raise OSError("fsync failed")
        super().fsync_file(file)

    def atomic_replace(self, source: Path, target: Path) -> None:
        self.calls.append(("atomic_replace", target))
        if self.fail_at == "atomic_replace":
            raise OSError("rename failed")
        super().atomic_replace(source, target)

    def fsync_directory(self, path: Path) -> None:
        self.calls.append(("fsync_directory", path))
        if self.fail_at == "ledger_fsync_directory":
            raise OSError("directory fsync failed")
        super().fsync_directory(path)

    def quarantine(self, target: Path, quarantine_path: Path) -> None:
        self.calls.append(("quarantine", quarantine_path))
        super().quarantine(target, quarantine_path)

    def unlink(self, path: Path) -> None:
        self.calls.append(("unlink", path))
        super().unlink(path)

    def open_directory(self, path: Path) -> int:
        descriptor = super().open_directory(path)
        self.calls.append(("open_directory", path))
        self.open_fds.add(descriptor)
        self.fd_paths[descriptor] = path
        return descriptor

    def close_directory(self, descriptor: int) -> None:
        super().close_directory(descriptor)
        self.calls.append(("close_directory", self.fd_paths[descriptor]))
        self.open_fds.discard(descriptor)

    def available_bytes_fd(self, descriptor: int) -> int:
        path = self.fd_paths[descriptor]
        self.calls.append(("available_bytes", path))
        if self.fail_at == "available_bytes":
            raise OSError("probe failed")
        if isinstance(self.free_bytes, list):
            return self.free_bytes.pop(0)
        return self.free_bytes

    def device_id_fd(self, descriptor: int) -> int:
        path = self.fd_paths[descriptor]
        self.calls.append(("device_id", path))
        if self.fail_at == "device_id":
            raise OSError("device probe failed")
        if path.name == ".staging":
            return self.staging_device
        return self.final_device

    def open_exclusive_at(self, descriptor: int, name: str) -> BinaryIO:
        path = self.fd_paths[descriptor] / name
        self.calls.append(("open_exclusive", path))
        if self.fail_at == "open_exclusive":
            raise OSError("exclusive open failed")
        return super().open_exclusive_at(descriptor, name)

    def open_readonly_at(self, descriptor: int, name: str) -> int:
        self.calls.append(
            ("open_readonly", self.fd_paths[descriptor] / name)
        )
        return super().open_readonly_at(descriptor, name)

    def replace_at(
        self, descriptor: int, source_name: str, target_name: str
    ) -> None:
        self.calls.append(
            ("atomic_replace", self.fd_paths[descriptor] / target_name)
        )
        if self.fail_at == "atomic_replace":
            raise OSError("rename failed")
        super().replace_at(descriptor, source_name, target_name)

    def commit_noreplace(
        self,
        staging_fd: int,
        part_name: str,
        target_fd: int,
        target_name: str,
    ) -> None:
        self.calls.append(
            ("commit_noreplace", self.fd_paths[target_fd] / target_name)
        )
        if self.fail_at == "atomic_replace":
            raise OSError("rename failed")
        super().commit_noreplace(
            staging_fd, part_name, target_fd, target_name
        )
        self.model_committed = True

    def fsync_directory_fd(self, descriptor: int) -> None:
        self.calls.append(("fsync_directory", self.fd_paths[descriptor]))
        if self.fail_at == "ledger_fsync_directory" or (
            self.fail_at == "fsync_directory" and self.model_committed
        ):
            raise OSError("directory fsync failed")
        super().fsync_directory_fd(descriptor)

    def unlink_at(self, descriptor: int, name: str) -> None:
        self.calls.append(("unlink", self.fd_paths[descriptor] / name))
        super().unlink_at(descriptor, name)

    def quarantine_at(
        self,
        target_fd: int,
        target_name: str,
        staging_fd: int,
        quarantine_name: str,
    ) -> None:
        self.calls.append(
            (
                "quarantine",
                self.fd_paths[staging_fd]
                / ".quarantine"
                / quarantine_name,
            )
        )
        super().quarantine_at(
            target_fd,
            target_name,
            staging_fd,
            quarantine_name,
        )


class CompetingTargetFilesystemOps(RecordingFilesystemOps):
    def atomic_replace(self, source: Path, target: Path) -> None:
        target.write_bytes(b"competitor")
        super().atomic_replace(source, target)

    def commit_noreplace(
        self,
        staging_fd: int,
        part_name: str,
        target_fd: int,
        target_name: str,
    ) -> None:
        descriptor = os.open(
            target_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=target_fd,
        )
        try:
            os.write(descriptor, b"competitor")
        finally:
            os.close(descriptor)
        super().commit_noreplace(
            staging_fd, part_name, target_fd, target_name
        )


class ParentSwapFilesystemOps(RecordingFilesystemOps):
    def __init__(self, target_path: Path, moved: Path, outside: Path) -> None:
        super().__init__()
        self.target_path = target_path
        self.moved = moved
        self.outside = outside
        self.swapped = False

    def _swap(self) -> None:
        if self.swapped:
            return
        self.target_path.rename(self.moved)
        self.target_path.symlink_to(self.outside, target_is_directory=True)
        self.swapped = True

    def commit_noreplace(
        self,
        staging_fd: int,
        part_name: str,
        target_fd: int,
        target_name: str,
    ) -> None:
        self._swap()
        super().commit_noreplace(
            staging_fd, part_name, target_fd, target_name
        )


class LedgerParentSwapFilesystemOps(RecordingFilesystemOps):
    def __init__(self, staging: Path, moved: Path, outside: Path) -> None:
        super().__init__()
        self.staging = staging
        self.moved = moved
        self.outside = outside
        self.swapped = False

    def _swap(self) -> None:
        if self.swapped:
            return
        self.staging.rename(self.moved)
        self.staging.symlink_to(self.outside, target_is_directory=True)
        self.swapped = True

    def atomic_replace(self, source: Path, target: Path) -> None:
        self._swap()
        super().atomic_replace(source, target)

    def replace_at(
        self, descriptor: int, source_name: str, target_name: str
    ) -> None:
        self._swap()
        super().replace_at(descriptor, source_name, target_name)


class LedgerInodeSwapFilesystemOps(RecordingFilesystemOps):
    def __init__(self) -> None:
        super().__init__()
        self.swapped = False
        self.last_read_fd: int | None = None

    def open_readonly_at(self, descriptor: int, name: str) -> int:
        if name == ".download-ledger.json" and not self.swapped:
            replacement = os.open(
                ".download-ledger.swap",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=descriptor,
            )
            try:
                os.write(
                    replacement,
                    b'{"schema_version":1,"transferred_bytes":0}',
                )
            finally:
                os.close(replacement)
            os.replace(
                ".download-ledger.swap",
                name,
                src_dir_fd=descriptor,
                dst_dir_fd=descriptor,
            )
            self.swapped = True
        opened = super().open_readonly_at(descriptor, name)
        self.last_read_fd = opened
        return opened


class SwappingRuntimeFileOps:
    def __init__(self, target: Path, replacement: Path) -> None:
        self.target = target
        self.replacement = replacement
        self.swapped = False
        self.last_fd: int | None = None

    def open_stable(
        self, path: Path, *, allowed_symlink_root: Path | None
    ) -> tuple[int, Path]:
        if not self.swapped:
            self.target.unlink()
            self.target.symlink_to(self.replacement)
            self.swapped = True
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
        self.last_fd = descriptor
        return descriptor, path.resolve(strict=True)


class TrackingRuntimeFileOps:
    def __init__(self) -> None:
        self.last_fd: int | None = None

    def open_stable(
        self, path: Path, *, allowed_symlink_root: Path | None
    ) -> tuple[int, Path]:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
        self.last_fd = descriptor
        return descriptor, path.resolve(strict=True)


def _assert_fd_closed(descriptor: int | None) -> None:
    assert descriptor is not None
    with pytest.raises(OSError):
        os.fstat(descriptor)


def test_repository_manifest_matches_approved_task6_inventory() -> None:
    manifest_path = Path(__file__).resolve().parents[2] / "models" / "manifest.json"
    manifest = load_manifest(manifest_path)

    assert set(manifest.models) == {
        "faster-whisper-small",
        "faster-whisper-large-v3",
        "piper-ru-dmitri-medium",
        "piper-en-ryan-medium",
        "nllb-200-distilled-600m-ct2-int8",
        "piper-ru-irina-medium",
        "piper-en-hfc-female-medium",
    }
    approved_metadata = {
        "faster-whisper-small": (
            "asr",
            "reuse",
            "Systran/faster-whisper-small",
            "536b0662742c02347bc0e980a01041f333bce120",
            ("ru", "en"),
            "/home/anton/.cache/huggingface/hub/"
            "models--Systran--faster-whisper-small/snapshots/"
            "536b0662742c02347bc0e980a01041f333bce120",
        ),
        "faster-whisper-large-v3": (
            "asr",
            "reuse",
            "Systran/faster-whisper-large-v3",
            "edaa852ec7e145841d8ffdb056a99866b5f0a478",
            ("ru", "en"),
            "/home/anton/Source/uncle-freud-bot/.data/faster-whisper/"
            "models--Systran--faster-whisper-large-v3/snapshots/"
            "edaa852ec7e145841d8ffdb056a99866b5f0a478",
        ),
        "piper-ru-dmitri-medium": (
            "tts",
            "reuse",
            "rhasspy/piper-voices",
            "0d907f158acc877ddeebcbf827659ee13bea8bcd",
            ("ru",),
            "/home/anton/Source/uncle-freud-bot/.data/piper-voices",
        ),
        "piper-en-ryan-medium": (
            "tts",
            "reuse",
            "rhasspy/piper-voices",
            "0d907f158acc877ddeebcbf827659ee13bea8bcd",
            ("en",),
            "/home/anton/Source/uncle-freud-bot/.data/piper-voices",
        ),
        "nllb-200-distilled-600m-ct2-int8": (
            "mt",
            "download",
            "mijuanlo/nllb-200-distilled-600M-ct2-int8",
            "16bc5ff0482f9f1c0d35bdef950721ce58640789",
            ("ru", "en"),
            "/home/anton/Source/translator/models/cache/"
            "nllb-200-distilled-600M-ct2-int8",
        ),
        "piper-ru-irina-medium": (
            "tts",
            "download",
            "rhasspy/piper-voices",
            "0d907f158acc877ddeebcbf827659ee13bea8bcd",
            ("ru",),
            "/home/anton/Source/translator/models/cache/piper",
        ),
        "piper-en-hfc-female-medium": (
            "tts",
            "download",
            "rhasspy/piper-voices",
            "0d907f158acc877ddeebcbf827659ee13bea8bcd",
            ("en",),
            "/home/anton/Source/translator/models/cache/piper",
        ),
    }
    observed_metadata = {
        model.id: (
            model.role,
            model.acquisition,
            model.source.repository,
            model.source.revision,
            tuple(model.languages),
            str(model.cache_path),
        )
        for model in manifest.models.values()
    }
    assert observed_metadata == approved_metadata
    approved_licenses = {
        "faster-whisper-small": ("MIT", None, None),
        "faster-whisper-large-v3": ("MIT", None, None),
        "piper-ru-dmitri-medium": ("MIT", "CC0", None),
        "piper-en-ryan-medium": ("MIT", "CC-BY-NC-SA-4.0", None),
        "nllb-200-distilled-600m-ct2-int8": (
            "CC-BY-NC-4.0",
            None,
            None,
        ),
        "piper-ru-irina-medium": (
            "MIT",
            "Unknown",
            "PIPER_RU_IRINA_PERSONAL_LOCAL_V1",
        ),
        "piper-en-hfc-female-medium": (
            "MIT",
            "CC-BY-NC-SA-4.0",
            None,
        ),
    }
    assert {
        model.id: (
            model.source.license,
            model.source.dataset_license,
            model.source.license_waiver,
        )
        for model in manifest.models.values()
    } == approved_licenses
    approved_downloads = {
        (
            "nllb-200-distilled-600m-ct2-int8",
            "mijuanlo/nllb-200-distilled-600M-ct2-int8",
            "16bc5ff0482f9f1c0d35bdef950721ce58640789",
            "/home/anton/Source/translator/models/cache/"
            "nllb-200-distilled-600M-ct2-int8",
        ): {
            "config.json": (
                1_065,
                "bf8ade7c3f1683e5f13001bab18b04a1ccd1a6801208efd227ed13b2ff6f15e7",
            ),
            "model.bin": (
                622_596_105,
                "398726640cc2a02cc6a35277fa3cf2159ce8a1a66b48aa1b6c8837a47e3dd00c",
            ),
            "sentencepiece.bpe.model": (
                4_852_054,
                "14bb8dfb35c0ffdea7bc01e56cea38b9e3d5efcdcb9c251d6b40538e1aab555a",
            ),
            "shared_vocabulary.json": (
                5_921_176,
                "af53bfd0e6f726209e7325e45b87ab3b14e5856f7d42d7b9be91de3287c45267",
            ),
        },
        (
            "piper-ru-irina-medium",
            "rhasspy/piper-voices",
            "0d907f158acc877ddeebcbf827659ee13bea8bcd",
            "/home/anton/Source/translator/models/cache/piper",
        ): {
            "ru_RU-irina-medium.onnx": (
                63_201_294,
                "8ff38212d23da300bbe3705c645e6e5b9475f0bfde01558eb17813e22acaaaaa",
            ),
            "ru_RU-irina-medium.onnx.json": (
                4_765,
                "c2ec28bb38e2b59e93b959b3e40348c1afebbd272f30fed5d41205d08e98a9d7",
            ),
        },
        (
            "piper-en-hfc-female-medium",
            "rhasspy/piper-voices",
            "0d907f158acc877ddeebcbf827659ee13bea8bcd",
            "/home/anton/Source/translator/models/cache/piper",
        ): {
            "en_US-hfc_female-medium.onnx": (
                63_201_294,
                "914c473788fc1fa8b63ace1cdcdb44588f4ae523d3ab37df1536616835a140b7",
            ),
            "en_US-hfc_female-medium.onnx.json": (
                5_033,
                "03f1fa0622b80463283592d97aca9f6e89aec345a5c56b7257723e0093c58b6c",
            ),
        },
    }
    observed_downloads: dict[
        tuple[str, str, str, str], dict[str, tuple[int, str]]
    ] = {}
    for model in manifest.models.values():
        if model.acquisition != "download":
            continue
        key = (
            model.id,
            model.source.repository,
            model.source.revision,
            str(model.cache_path),
        )
        observed_downloads[key] = {
            file.path: (file.size_bytes, file.sha256) for file in model.files
        }

    assert observed_downloads == approved_downloads
    approved_source_urls = {
        (
            "nllb-200-distilled-600m-ct2-int8",
            "config.json",
        ): (
            "https://huggingface.co/mijuanlo/"
            "nllb-200-distilled-600M-ct2-int8/resolve/"
            "16bc5ff0482f9f1c0d35bdef950721ce58640789/config.json"
        ),
        (
            "nllb-200-distilled-600m-ct2-int8",
            "model.bin",
        ): (
            "https://huggingface.co/mijuanlo/"
            "nllb-200-distilled-600M-ct2-int8/resolve/"
            "16bc5ff0482f9f1c0d35bdef950721ce58640789/model.bin"
        ),
        (
            "nllb-200-distilled-600m-ct2-int8",
            "sentencepiece.bpe.model",
        ): (
            "https://huggingface.co/mijuanlo/"
            "nllb-200-distilled-600M-ct2-int8/resolve/"
            "16bc5ff0482f9f1c0d35bdef950721ce58640789/"
            "sentencepiece.bpe.model"
        ),
        (
            "nllb-200-distilled-600m-ct2-int8",
            "shared_vocabulary.json",
        ): (
            "https://huggingface.co/mijuanlo/"
            "nllb-200-distilled-600M-ct2-int8/resolve/"
            "16bc5ff0482f9f1c0d35bdef950721ce58640789/"
            "shared_vocabulary.json"
        ),
        (
            "piper-ru-irina-medium",
            "ru_RU-irina-medium.onnx",
        ): (
            "https://huggingface.co/rhasspy/piper-voices/resolve/"
            "0d907f158acc877ddeebcbf827659ee13bea8bcd/"
            "ru/ru_RU/irina/medium/ru_RU-irina-medium.onnx"
        ),
        (
            "piper-ru-irina-medium",
            "ru_RU-irina-medium.onnx.json",
        ): (
            "https://huggingface.co/rhasspy/piper-voices/resolve/"
            "0d907f158acc877ddeebcbf827659ee13bea8bcd/"
            "ru/ru_RU/irina/medium/ru_RU-irina-medium.onnx.json"
        ),
        (
            "piper-en-hfc-female-medium",
            "en_US-hfc_female-medium.onnx",
        ): (
            "https://huggingface.co/rhasspy/piper-voices/resolve/"
            "0d907f158acc877ddeebcbf827659ee13bea8bcd/"
            "en/en_US/hfc_female/medium/en_US-hfc_female-medium.onnx"
        ),
        (
            "piper-en-hfc-female-medium",
            "en_US-hfc_female-medium.onnx.json",
        ): (
            "https://huggingface.co/rhasspy/piper-voices/resolve/"
            "0d907f158acc877ddeebcbf827659ee13bea8bcd/"
            "en/en_US/hfc_female/medium/en_US-hfc_female-medium.onnx.json"
        ),
    }
    observed_source_urls = {
        (model.id, file.path): file.source_url
        for model in manifest.models.values()
        if model.acquisition == "download"
        for file in model.files
    }
    assert observed_source_urls == approved_source_urls
    approved_source_paths = {
        (
            "nllb-200-distilled-600m-ct2-int8",
            "config.json",
        ): "config.json",
        (
            "nllb-200-distilled-600m-ct2-int8",
            "model.bin",
        ): "model.bin",
        (
            "nllb-200-distilled-600m-ct2-int8",
            "sentencepiece.bpe.model",
        ): "sentencepiece.bpe.model",
        (
            "nllb-200-distilled-600m-ct2-int8",
            "shared_vocabulary.json",
        ): "shared_vocabulary.json",
        (
            "piper-ru-irina-medium",
            "ru_RU-irina-medium.onnx",
        ): "ru/ru_RU/irina/medium/ru_RU-irina-medium.onnx",
        (
            "piper-ru-irina-medium",
            "ru_RU-irina-medium.onnx.json",
        ): "ru/ru_RU/irina/medium/ru_RU-irina-medium.onnx.json",
        (
            "piper-en-hfc-female-medium",
            "en_US-hfc_female-medium.onnx",
        ): "en/en_US/hfc_female/medium/en_US-hfc_female-medium.onnx",
        (
            "piper-en-hfc-female-medium",
            "en_US-hfc_female-medium.onnx.json",
        ): "en/en_US/hfc_female/medium/en_US-hfc_female-medium.onnx.json",
    }
    assert {
        (model.id, file.path): file.source_path
        for model in manifest.models.values()
        if model.acquisition == "download"
        for file in model.files
    } == approved_source_paths
    approved_reused_files = {
        ("faster-whisper-small", "config.json"): (
            2_370,
            "b55496ac7940a7ae47d2c01eab40edfd8701feec1229d9cce3b40014383fb828",
        ),
        ("faster-whisper-small", "model.bin"): (
            483_546_902,
            "3e305921506d8872816023e4c273e75d2419fb89b24da97b4fe7bce14170d671",
        ),
        ("faster-whisper-small", "tokenizer.json"): (
            2_203_239,
            "fb7b63191e9bb045082c79fd742a3106a12c99513ab30df4a0d47fa6cb6fd0ab",
        ),
        ("faster-whisper-small", "vocabulary.txt"): (
            459_861,
            "34ce3fe1c5041027b3f8d42912270993f986dbc4bb34cf27f951e34a1e453913",
        ),
        ("faster-whisper-large-v3", "config.json"): (
            2_394,
            "a9306624f5ec14270a014b647e5c316b6e03a662c369758d1b90697a7b0655b9",
        ),
        ("faster-whisper-large-v3", "model.bin"): (
            3_087_284_237,
            "69f74147e3334731bc3a76048724833325d2ec74642fb52620eda87352e3d4f1",
        ),
        ("faster-whisper-large-v3", "preprocessor_config.json"): (
            340,
            "7ccc62c6f2765af1f3b46c00c9b5894426835a05021c8b9c01eecb6dfb542711",
        ),
        ("faster-whisper-large-v3", "tokenizer.json"): (
            2_480_617,
            "6d8cbd7cd0d8d5815e478dac67b85a26bbe77c1f5e0c6d76d1ce2abc0e5f21ca",
        ),
        ("faster-whisper-large-v3", "vocabulary.json"): (
            1_068_114,
            "c69260f2ab26d659b7c398f9a2b2b48ed0df16c3b47d7326782fd9cba71690c1",
        ),
        ("piper-ru-dmitri-medium", "ru_RU-dmitri-medium.onnx"): (
            63_201_294,
            "f073356ebc4bd0f80c5af58df2953a5988bd5bdab1eb38635ce960b071fbefcb",
        ),
        ("piper-ru-dmitri-medium", "ru_RU-dmitri-medium.onnx.json"): (
            4_824,
            "667ef3117bc642c2892dff7690d8bdc8ca4228aeaa783b2dc1416df632855e0d",
        ),
        ("piper-en-ryan-medium", "en_US-ryan-medium.onnx"): (
            63_201_294,
            "abf4c274862564ed647ba0d2c47f8ee7c9b717d27bdad9219100eb310db4047a",
        ),
        ("piper-en-ryan-medium", "en_US-ryan-medium.onnx.json"): (
            4_883,
            "44034c056cb15681b2ad494307c7f3f2e4499d1253c700c711fa0a4607ffe78d",
        ),
    }
    observed_reused_files = {
        (model.id, file.path): (file.size_bytes, file.sha256)
        for model in manifest.models.values()
        if model.acquisition == "reuse"
        for file in model.files
    }
    assert observed_reused_files == approved_reused_files
    assert manifest.planned_download_bytes == 759_782_786
    assert manifest.policy.download_budget_bytes == 2 * GIB
    assert manifest.policy.post_download_free_floor_bytes == 20 * GIB
    assert manifest.policy.usage_mode == "personal_noncommercial"
    assert manifest.policy.redistribution is False
    assert manifest.policy.certified_or_safety_critical is False
    assert str(manifest.policy.staging_path) == (
        "/home/anton/Source/translator/models/cache/.staging"
    )
    assert set(manifest.policy.redirect_hosts) == {
        "cdn-lfs.huggingface.co",
        "cdn-lfs-us-1.hf.co",
        "cdn-lfs-eu-1.hf.co",
        "cas-bridge.xethub.hf.co",
        "us.aws.cdn.hf.co",
    }


def test_repository_reused_assets_resolve_through_pinned_integrity_policy() -> None:
    manifest_path = Path(__file__).resolve().parents[2] / "models" / "manifest.json"
    manifest = load_manifest(manifest_path)

    resolved = {
        (model.id, file.path): manifest.resolve_runtime_file(
            model.id, file.path
        )
        for model in manifest.models.values()
        if model.acquisition == "reuse"
        for file in model.files
    }

    assert len(resolved) == 13
    assert all(path.exists() for path in resolved.values())


def test_hf_snapshot_symlink_must_resolve_inside_pinned_blob_root(
    tmp_path: Path,
) -> None:
    document, _, blob = _hf_reuse_manifest(tmp_path)
    manifest = load_manifest(_write_manifest(tmp_path, document))
    filesystem = TrackingRuntimeFileOps()

    resolved = manifest.resolve_runtime_file(
        "selected-mt", "model.bin", filesystem=filesystem
    )
    assert resolved.samefile(blob)
    _assert_fd_closed(filesystem.last_fd)


def test_hf_snapshot_symlink_rejects_same_content_outside_blob_root(
    tmp_path: Path,
) -> None:
    document, target, _ = _hf_reuse_manifest(tmp_path)
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"trusted")
    target.unlink()
    target.symlink_to(outside)
    manifest = load_manifest(_write_manifest(tmp_path, document))
    filesystem = TrackingRuntimeFileOps()

    with pytest.raises(ManifestError, match="runtime|blob"):
        manifest.resolve_runtime_file(
            "selected-mt", "model.bin", filesystem=filesystem
        )
    _assert_fd_closed(filesystem.last_fd)


def test_hf_snapshot_link_swap_before_stable_open_fails_closed(
    tmp_path: Path,
) -> None:
    document, target, _ = _hf_reuse_manifest(tmp_path)
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"trusted")
    manifest = load_manifest(_write_manifest(tmp_path, document))
    filesystem = SwappingRuntimeFileOps(target, outside)

    with pytest.raises(ManifestError, match="runtime|blob"):
        manifest.resolve_runtime_file(
            "selected-mt", "model.bin", filesystem=filesystem
        )
    assert filesystem.swapped
    _assert_fd_closed(filesystem.last_fd)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("license", ""),
        ("revision", "main"),
    ],
)
def test_manifest_rejects_unknown_provenance(
    tmp_path: Path, field: str, value: str
) -> None:
    document = _manifest(tmp_path / "cache")
    source = document["models"][0]["source"]  # type: ignore[index]
    source[field] = value  # type: ignore[index]

    with pytest.raises(ManifestError):
        load_manifest(_write_manifest(tmp_path, document))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("size_bytes", 0),
        ("sha256", ""),
        ("sha256", "not-a-sha256"),
    ],
)
def test_manifest_rejects_unknown_file_evidence(
    tmp_path: Path, field: str, value: int | str
) -> None:
    document = _manifest(tmp_path / "cache")
    model_file = document["models"][0]["files"][0]  # type: ignore[index]
    model_file[field] = value  # type: ignore[index]

    with pytest.raises(ManifestError):
        load_manifest(_write_manifest(tmp_path, document))


def test_manifest_requires_absolute_cache_path(tmp_path: Path) -> None:
    document = _manifest(Path("models/cache/model"))

    with pytest.raises(ManifestError, match="absolute"):
        load_manifest(_write_manifest(tmp_path, document))


def test_manifest_rejects_path_escape_and_unallowlisted_file(tmp_path: Path) -> None:
    document = _manifest(tmp_path / "cache")
    document["models"][0]["files"][0]["path"] = "../other.bin"  # type: ignore[index]

    with pytest.raises(ManifestError):
        load_manifest(_write_manifest(tmp_path, document))


def test_manifest_rejects_nested_local_target_path(tmp_path: Path) -> None:
    document = _manifest(tmp_path / "cache")
    document["models"][0]["files"][0]["path"] = "nested/model.bin"  # type: ignore[index]

    with pytest.raises(ManifestError, match="path"):
        load_manifest(_write_manifest(tmp_path, document))


@pytest.mark.parametrize("model_id", ["../escape", "nested/model", "a..b/other"])
def test_manifest_rejects_model_id_that_can_escape_staging(
    tmp_path: Path, model_id: str
) -> None:
    document = _manifest(tmp_path / "cache")
    document["models"][0]["id"] = model_id  # type: ignore[index]

    with pytest.raises(ManifestError, match="model id"):
        load_manifest(_write_manifest(tmp_path, document))


def test_source_url_must_match_exact_declared_relative_source_path(
    tmp_path: Path,
) -> None:
    document = _manifest(tmp_path / "cache")
    model_file = document["models"][0]["files"][0]  # type: ignore[index]
    model_file["source_path"] = "approved/model.bin"  # type: ignore[index]

    with pytest.raises(ManifestError, match="source"):
        load_manifest(_write_manifest(tmp_path, document))


def test_manifest_enforces_exact_two_gib_download_budget(tmp_path: Path) -> None:
    document = _manifest(tmp_path / "cache", size_bytes=2 * GIB + 1)

    with pytest.raises(ManifestError, match="budget"):
        load_manifest(_write_manifest(tmp_path, document))


@pytest.mark.parametrize(
    "dataset_license",
    ["", "unknown", "UNKNOWN", " Unknown ", "unlicensed", "Proprietary"],
)
def test_unapproved_dataset_license_variants_fail_closed(
    tmp_path: Path, dataset_license: str
) -> None:
    document = _manifest(tmp_path / "cache", license_id="MIT")
    source = document["models"][0]["source"]  # type: ignore[index]
    source["dataset_license"] = dataset_license  # type: ignore[index]

    with pytest.raises(ManifestError, match="license|waiver"):
        load_manifest(_write_manifest(tmp_path, document))


@pytest.mark.parametrize(
    "license_id",
    ["", "unknown", "UNKNOWN", " Unknown ", "unlicensed", "Proprietary"],
)
def test_unapproved_primary_license_variants_fail_closed(
    tmp_path: Path, license_id: str
) -> None:
    document = _manifest(tmp_path / "cache", license_id=license_id)

    with pytest.raises(ManifestError, match="license"):
        load_manifest(_write_manifest(tmp_path, document))


@pytest.mark.parametrize(
    ("license_id", "dataset_license"),
    [
        ("MIT", None),
        ("MIT", "CC0"),
        ("MIT", "CC-BY-NC-SA-4.0"),
        ("CC-BY-NC-4.0", None),
    ],
)
def test_exact_approved_license_identifiers_are_accepted(
    tmp_path: Path,
    license_id: str,
    dataset_license: str | None,
) -> None:
    document = _manifest(tmp_path / "cache", license_id=license_id)
    if dataset_license is not None:
        source = document["models"][0]["source"]  # type: ignore[index]
        source["dataset_license"] = dataset_license  # type: ignore[index]

    load_manifest(_write_manifest(tmp_path, document))


def test_manifest_rejects_usage_incompatible_license(tmp_path: Path) -> None:
    document = _manifest(tmp_path / "cache")
    document["policy"]["usage_mode"] = "commercial_redistribution"  # type: ignore[index]

    with pytest.raises(ManifestError, match="usage"):
        load_manifest(_write_manifest(tmp_path, document))


def test_unknown_dataset_license_requires_exact_personal_waiver(
    tmp_path: Path,
) -> None:
    document = _manifest(tmp_path / "cache", license_id="MIT")
    source = document["models"][0]["source"]  # type: ignore[index]
    source["dataset_license"] = "Unknown"  # type: ignore[index]

    with pytest.raises(ManifestError, match="waiver"):
        load_manifest(_write_manifest(tmp_path, document))


def test_exact_irina_personal_waiver_is_accepted(tmp_path: Path) -> None:
    document = _irina_manifest(tmp_path)
    manifest = load_manifest(_write_manifest(tmp_path, document))
    assert manifest.models["piper-ru-irina-medium"].source.license_waiver == (
        "PIPER_RU_IRINA_PERSONAL_LOCAL_V1"
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "model_id",
        "role",
        "acquisition",
        "languages",
        "cache_path",
        "repository",
        "revision",
        "license",
        "dataset_license",
        "waiver",
        "usage_mode",
        "redistribution",
        "safety_critical",
        "onnx_path",
        "onnx_size",
        "onnx_sha256",
        "onnx_source_url",
        "onnx_source_path",
        "config_path",
        "config_size",
        "config_sha256",
        "config_source_url",
        "config_source_path",
    ],
)
def test_irina_waiver_is_bound_to_every_approved_field(
    tmp_path: Path, mutation: str
) -> None:
    document = deepcopy(_irina_manifest(tmp_path))
    policy = document["policy"]  # type: ignore[index]
    model = document["models"][0]  # type: ignore[index]
    source = model["source"]  # type: ignore[index]
    onnx = model["files"][0]  # type: ignore[index]
    config = model["files"][1]  # type: ignore[index]
    mutations = {
        "model_id": lambda: model.__setitem__("id", "other"),
        "role": lambda: model.__setitem__("role", "mt"),
        "acquisition": lambda: model.__setitem__("acquisition", "reuse"),
        "languages": lambda: model.__setitem__("languages", ["en"]),
        "cache_path": lambda: model.__setitem__(
            "cache_path", str(tmp_path / "cache" / "other")
        ),
        "repository": lambda: source.__setitem__("repository", "other/repo"),
        "revision": lambda: source.__setitem__("revision", "b" * 40),
        "license": lambda: source.__setitem__("license", "CC0"),
        "dataset_license": lambda: source.__setitem__(
            "dataset_license", "CC0"
        ),
        "waiver": lambda: source.__setitem__("license_waiver", "OTHER"),
        "usage_mode": lambda: policy.__setitem__("usage_mode", "commercial"),
        "redistribution": lambda: policy.__setitem__("redistribution", True),
        "safety_critical": lambda: policy.__setitem__(
            "certified_or_safety_critical", True
        ),
        "onnx_path": lambda: onnx.__setitem__("path", "other.onnx"),
        "onnx_size": lambda: onnx.__setitem__("size_bytes", 1),
        "onnx_sha256": lambda: onnx.__setitem__("sha256", "a" * 64),
        "onnx_source_url": lambda: onnx.__setitem__(
            "source_url", "https://huggingface.co/other"
        ),
        "onnx_source_path": lambda: onnx.__setitem__(
            "source_path", "other/ru_RU-irina-medium.onnx"
        ),
        "config_path": lambda: config.__setitem__("path", "other.json"),
        "config_size": lambda: config.__setitem__("size_bytes", 1),
        "config_sha256": lambda: config.__setitem__("sha256", "a" * 64),
        "config_source_url": lambda: config.__setitem__(
            "source_url", "https://huggingface.co/other"
        ),
        "config_source_path": lambda: config.__setitem__(
            "source_path", "other/ru_RU-irina-medium.onnx.json"
        ),
    }
    mutations[mutation]()

    with pytest.raises(ManifestError):
        load_manifest(_write_manifest(tmp_path, document))


def test_download_ledger_counts_partial_files_and_retry_bytes(tmp_path: Path) -> None:
    staging = tmp_path / ".staging"
    staging.mkdir()
    (staging / "model.bin.part").write_bytes(b"x" * 4)
    ledger = DownloadLedger(
        budget_bytes=10,
        planned_bytes=7,
        staging_dir=staging,
    )

    assert ledger.transferred_bytes == 4
    ledger.record_received(6)
    with pytest.raises(ManifestError, match="budget"):
        ledger.record_received(1)
    assert ledger.transferred_bytes == 10


def test_download_ledger_accepts_exact_aggregate_plan_and_rejects_overflow(
    tmp_path: Path,
) -> None:
    exact = DownloadLedger(
        budget_bytes=10,
        planned_bytes=10,
        staging_dir=tmp_path,
    )
    assert exact.planned_bytes == 10

    with pytest.raises(ManifestError, match="budget"):
        DownloadLedger(
            budget_bytes=10,
            planned_bytes=11,
            staging_dir=tmp_path,
        )


def test_download_ledger_persists_retry_bytes_across_restart(
    tmp_path: Path,
) -> None:
    first = DownloadLedger(
        budget_bytes=10,
        planned_bytes=7,
        staging_dir=tmp_path,
    )
    first.record_received(6)

    restarted = DownloadLedger(
        budget_bytes=10,
        planned_bytes=7,
        staging_dir=tmp_path,
    )

    assert restarted.transferred_bytes == 6
    with pytest.raises(ManifestError, match="budget"):
        restarted.record_received(5)
    assert restarted.transferred_bytes == 6


def test_download_ledger_survives_separate_process_restart(
    tmp_path: Path,
) -> None:
    writer = (
        "from pathlib import Path;"
        "from translator_sidecar.local.model_manifest import DownloadLedger;"
        f"p=Path({str(tmp_path)!r});"
        "x=DownloadLedger(budget_bytes=10,planned_bytes=7,staging_dir=p);"
        "x.record_received(6)"
    )
    reader = (
        "from pathlib import Path;"
        "from translator_sidecar.local.model_manifest import DownloadLedger;"
        f"p=Path({str(tmp_path)!r});"
        "x=DownloadLedger(budget_bytes=10,planned_bytes=7,staging_dir=p);"
        "print(x.transferred_bytes)"
    )

    subprocess.run([sys.executable, "-c", writer], check=True)
    result = subprocess.run(
        [sys.executable, "-c", reader],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "6"


def test_persisted_ledger_does_not_double_count_its_existing_part(
    tmp_path: Path,
) -> None:
    first = DownloadLedger(
        budget_bytes=10,
        planned_bytes=7,
        staging_dir=tmp_path,
    )
    first.record_received(3)
    (tmp_path / "model.bin.part").write_bytes(b"pay")

    restarted = DownloadLedger(
        budget_bytes=10,
        planned_bytes=7,
        staging_dir=tmp_path,
    )

    assert restarted.transferred_bytes == 3


def test_fresh_ledger_counts_existing_part_once_and_persists_it(
    tmp_path: Path,
) -> None:
    (tmp_path / "model.bin.part").write_bytes(b"pay")

    first = DownloadLedger(
        budget_bytes=10,
        planned_bytes=7,
        staging_dir=tmp_path,
    )
    (tmp_path / "model.bin.part").unlink()
    restarted = DownloadLedger(
        budget_bytes=10,
        planned_bytes=7,
        staging_dir=tmp_path,
    )

    assert first.transferred_bytes == 3
    assert restarted.transferred_bytes == 3


def test_download_ledger_update_uses_durable_atomic_operations(
    tmp_path: Path,
) -> None:
    filesystem = RecordingFilesystemOps()
    ledger = DownloadLedger(
        budget_bytes=10,
        planned_bytes=7,
        staging_dir=tmp_path,
        filesystem=filesystem,
    )

    ledger.record_received(3)

    operations = [name for name, _ in filesystem.calls]
    assert operations.index("open_exclusive") < operations.index("fsync_file")
    assert operations.index("fsync_file") < operations.index("atomic_replace")
    assert operations.index("atomic_replace") < operations.index(
        "fsync_directory"
    )
    assert not list(tmp_path.glob(".download-ledger.*.tmp"))


@pytest.mark.parametrize(
    "fail_at",
    ["open_exclusive", "fsync_file", "atomic_replace"],
)
def test_download_ledger_precommit_failure_preserves_previous_state(
    tmp_path: Path, fail_at: str
) -> None:
    filesystem = RecordingFilesystemOps(fail_at=fail_at)
    ledger = DownloadLedger(
        budget_bytes=10,
        planned_bytes=7,
        staging_dir=tmp_path,
        filesystem=filesystem,
    )

    with pytest.raises(ManifestError, match="ledger"):
        ledger.record_received(3)

    assert not list(tmp_path.glob(".download-ledger.*.tmp"))
    assert not (tmp_path / ".download-ledger.json").exists()
    assert ledger.transferred_bytes == 0


def test_download_ledger_directory_fsync_failure_is_indeterminate_and_recovers(
    tmp_path: Path,
) -> None:
    filesystem = RecordingFilesystemOps(fail_at="ledger_fsync_directory")
    ledger = DownloadLedger(
        budget_bytes=10,
        planned_bytes=7,
        staging_dir=tmp_path,
        filesystem=filesystem,
    )

    with pytest.raises(ManifestError) as error:
        ledger.record_received(3)
    assert error.value.__class__.__name__ == "LedgerDurabilityError"
    assert ledger.state == "durability_indeterminate"
    with pytest.raises(ManifestError, match="ledger"):
        ledger.record_received(1)

    recovered = DownloadLedger(
        budget_bytes=10,
        planned_bytes=7,
        staging_dir=tmp_path,
    )
    assert recovered.transferred_bytes == 3


def test_download_ledger_parent_swap_uses_pinned_directory_and_fails_closed(
    tmp_path: Path,
) -> None:
    staging = tmp_path / "staging"
    moved = tmp_path / "staging-pinned"
    outside = tmp_path / "outside"
    outside.mkdir()
    filesystem = LedgerParentSwapFilesystemOps(staging, moved, outside)
    ledger = DownloadLedger(
        budget_bytes=10,
        planned_bytes=7,
        staging_dir=staging,
        filesystem=filesystem,
    )

    with pytest.raises(ManifestError) as error:
        ledger.record_received(3)

    assert error.value.__class__.__name__ == "LedgerDurabilityError"
    assert filesystem.swapped
    assert ledger.state == "durability_indeterminate"
    with pytest.raises(ManifestError, match="ledger"):
        ledger.record_received(1)
    assert not (outside / ".download-ledger.json").exists()
    state = json.loads(
        (moved / ".download-ledger.json").read_text(encoding="utf-8")
    )
    assert state["transferred_bytes"] == 3
    assert filesystem.open_fds == set()


def test_download_ledger_rejects_inode_swap_between_stat_and_open(
    tmp_path: Path,
) -> None:
    first = DownloadLedger(
        budget_bytes=10,
        planned_bytes=7,
        staging_dir=tmp_path,
    )
    first.record_received(3)
    filesystem = LedgerInodeSwapFilesystemOps()

    with pytest.raises(ManifestError, match="ledger"):
        DownloadLedger(
            budget_bytes=10,
            planned_bytes=7,
            staging_dir=tmp_path,
            filesystem=filesystem,
        )

    assert filesystem.swapped
    _assert_fd_closed(filesystem.last_read_fd)
    assert filesystem.open_fds == set()


@pytest.mark.parametrize("state_kind", ["malformed", "symlink", "directory"])
def test_download_ledger_state_is_regular_valid_json_or_fails_closed(
    tmp_path: Path, state_kind: str
) -> None:
    ledger_path = tmp_path / ".download-ledger.json"
    if state_kind == "malformed":
        ledger_path.write_text("{broken", encoding="utf-8")
    elif state_kind == "symlink":
        outside = tmp_path / "outside.json"
        outside.write_text(
            '{"schema_version":1,"transferred_bytes":0}',
            encoding="utf-8",
        )
        ledger_path.symlink_to(outside)
    else:
        ledger_path.mkdir()

    with pytest.raises(ManifestError, match="ledger"):
        DownloadLedger(
            budget_bytes=10,
            planned_bytes=7,
            staging_dir=tmp_path,
        )


def test_preflight_requires_twenty_gib_after_remaining_download(
    tmp_path: Path,
) -> None:
    ledger = DownloadLedger(
        budget_bytes=2 * GIB,
        planned_bytes=100 * MIB,
        staging_dir=tmp_path,
    )

    ledger.check_free_space(
        free_bytes=20 * GIB + 100 * MIB,
        remaining_download_bytes=100 * MIB,
        floor_bytes=20 * GIB,
    )


def test_preflight_rejects_one_byte_below_free_space_floor(tmp_path: Path) -> None:
    ledger = DownloadLedger(
        budget_bytes=2 * GIB,
        planned_bytes=100 * MIB,
        staging_dir=tmp_path,
    )

    with pytest.raises(ManifestError, match="free space"):
        ledger.check_free_space(
            free_bytes=20 * GIB + 100 * MIB - 1,
            remaining_download_bytes=100 * MIB,
            floor_bytes=20 * GIB,
        )


@pytest.mark.parametrize(
    "invalid_initial_url",
    [
        "http://huggingface.co/owner/model/resolve/" + "a" * 40 + "/model.bin",
        "https://user@huggingface.co/owner/model/resolve/"
        + "a" * 40
        + "/model.bin",
        "https://huggingface.co/other/model/resolve/" + "a" * 40 + "/model.bin",
        "https://huggingface.co/owner/model/resolve/" + "b" * 40 + "/model.bin",
        "https://huggingface.co/owner/model/resolve/" + "a" * 40 + "/other.bin",
    ],
)
def test_initial_download_url_must_match_exact_pinned_source(
    tmp_path: Path, invalid_initial_url: str
) -> None:
    manifest = load_manifest(
        _write_manifest(tmp_path, _manifest(tmp_path / "cache"))
    )
    with pytest.raises(ManifestError, match="source"):
        manifest.validate_download_chain(
            "selected-mt",
            "model.bin",
            [invalid_initial_url],
        )


def test_redirect_chain_fails_closed_outside_pinned_hugging_face_hosts(
    tmp_path: Path,
) -> None:
    document = _manifest(tmp_path / "cache")
    manifest = load_manifest(_write_manifest(tmp_path, document))
    source_url = document["models"][0]["files"][0]["source_url"]  # type: ignore[index]

    manifest.validate_download_chain(
        "selected-mt",
        "model.bin",
        [
            source_url,
            "https://cas-bridge.xethub.hf.co/"
            "xet-bridge-us/owner/model/pinned-content",
        ],
    )
    with pytest.raises(ManifestError, match="redirect"):
        manifest.validate_download_chain(
            "selected-mt",
            "model.bin",
            [source_url, "https://example.invalid/model.bin"],
        )
    with pytest.raises(ManifestError, match="redirect"):
        manifest.validate_download_chain(
            "selected-mt",
            "model.bin",
            [source_url, "https://user@cas-bridge.xethub.hf.co/content"],
        )
    with pytest.raises(ManifestError, match="redirect"):
        manifest.validate_download_chain(
            "selected-mt",
            "model.bin",
            [source_url, "http://cas-bridge.xethub.hf.co/content"],
        )


def test_existing_part_file_blocks_nonexclusive_install(tmp_path: Path) -> None:
    payload = b"payload"
    document = _manifest(tmp_path / "cache", sha256=_sha256(payload))
    manifest = load_manifest(_write_manifest(tmp_path, document))
    staging = tmp_path / "cache" / ".staging"
    staging.mkdir(parents=True)
    (staging / "model.bin.part").write_bytes(b"stale")

    with pytest.raises(ManifestError, match="staging"):
        ModelDownloader(manifest).install_bytes(
            "selected-mt",
            "model.bin",
            [payload],
        )


def test_downloader_probes_free_space_and_same_device_before_transfer(
    tmp_path: Path,
) -> None:
    document = _manifest(tmp_path / "cache")
    manifest = load_manifest(_write_manifest(tmp_path, document))
    fs = RecordingFilesystemOps()

    ModelDownloader(manifest, filesystem=fs).install_bytes(
        "selected-mt", "model.bin", [b"payload"]
    )

    operation_names = [name for name, _ in fs.calls]
    assert operation_names.index("available_bytes") < operation_names.index(
        "open_exclusive"
    )
    assert operation_names.count("available_bytes") == 1
    assert operation_names.count("device_id") >= 2


def test_downloader_repeats_preflight_for_every_file_transfer(
    tmp_path: Path,
) -> None:
    document = _two_file_manifest(tmp_path)
    manifest = load_manifest(_write_manifest(tmp_path, document))
    fs = RecordingFilesystemOps()
    downloader = ModelDownloader(manifest, filesystem=fs)

    downloader.install_bytes("selected-mt", "model.bin", [b"payload"])
    downloader.install_bytes("selected-mt", "config.json", [b"{}"])

    available_calls = [
        path for name, path in fs.calls if name == "available_bytes"
    ]
    assert available_calls == [
        tmp_path / "cache" / ".staging",
        tmp_path / "cache" / ".staging",
    ]


def test_first_preflight_reserves_all_remaining_manifest_bytes(
    tmp_path: Path,
) -> None:
    manifest = load_manifest(
        _write_manifest(tmp_path, _two_file_manifest(tmp_path))
    )
    fs = RecordingFilesystemOps(free_bytes=20 * GIB + 8)

    with pytest.raises(ManifestError, match="free space"):
        ModelDownloader(manifest, filesystem=fs).install_bytes(
            "selected-mt", "model.bin", [b"payload"]
        )

    assert "open_exclusive" not in [name for name, _ in fs.calls]


def test_second_preflight_accepts_exact_still_missing_bytes(
    tmp_path: Path,
) -> None:
    manifest = load_manifest(
        _write_manifest(tmp_path, _two_file_manifest(tmp_path))
    )
    fs = RecordingFilesystemOps(
        free_bytes=[20 * GIB + 9, 20 * GIB + 2]
    )
    downloader = ModelDownloader(manifest, filesystem=fs)

    downloader.install_bytes("selected-mt", "model.bin", [b"payload"])
    downloader.install_bytes("selected-mt", "config.json", [b"{}"])

    assert len(
        [
            path
            for name, path in fs.calls
            if name == "open_exclusive" and path.name.endswith(".part")
        ]
    ) == 2


def test_second_preflight_rejects_one_byte_below_still_missing_bytes(
    tmp_path: Path,
) -> None:
    manifest = load_manifest(
        _write_manifest(tmp_path, _two_file_manifest(tmp_path))
    )
    fs = RecordingFilesystemOps(
        free_bytes=[20 * GIB + 9, 20 * GIB + 1]
    )
    downloader = ModelDownloader(manifest, filesystem=fs)

    downloader.install_bytes("selected-mt", "model.bin", [b"payload"])
    with pytest.raises(ManifestError, match="free space"):
        downloader.install_bytes("selected-mt", "config.json", [b"{}"])

    assert len(
        [
            path
            for name, path in fs.calls
            if name == "open_exclusive" and path.name.endswith(".part")
        ]
    ) == 1


@pytest.mark.parametrize(
    ("filesystem", "message"),
    [
        (
            RecordingFilesystemOps(staging_device=1, final_device=2),
            "filesystem",
        ),
        (RecordingFilesystemOps(fail_at="available_bytes"), "free space"),
        (RecordingFilesystemOps(fail_at="device_id"), "filesystem"),
        (RecordingFilesystemOps(free_bytes=20 * GIB + 6), "free space"),
    ],
)
def test_downloader_fails_closed_before_transfer_when_probe_is_unsafe(
    tmp_path: Path,
    filesystem: RecordingFilesystemOps,
    message: str,
) -> None:
    manifest = load_manifest(
        _write_manifest(tmp_path, _manifest(tmp_path / "cache"))
    )

    with pytest.raises(ManifestError, match=message):
        ModelDownloader(manifest, filesystem=filesystem).install_bytes(
            "selected-mt", "model.bin", [b"payload"]
        )
    assert "open_exclusive" not in [name for name, _ in filesystem.calls]


def test_downloader_uses_atomic_noreplace_against_competing_target(
    tmp_path: Path,
) -> None:
    manifest = load_manifest(
        _write_manifest(tmp_path, _manifest(tmp_path / "cache"))
    )
    filesystem = CompetingTargetFilesystemOps()

    with pytest.raises(ManifestError, match="target|commit"):
        ModelDownloader(manifest, filesystem=filesystem).install_bytes(
            "selected-mt", "model.bin", [b"payload"]
        )

    assert (tmp_path / "cache" / "model.bin").read_bytes() == b"competitor"
    assert "commit_noreplace" in [name for name, _ in filesystem.calls]
    assert filesystem.open_fds == set()


def test_downloader_fails_closed_after_parent_path_swap_without_escape(
    tmp_path: Path,
) -> None:
    target_path = tmp_path / "cache"
    moved = tmp_path / "cache-pinned"
    outside = tmp_path / "outside"
    outside.mkdir()
    manifest = load_manifest(
        _write_manifest(tmp_path, _manifest(target_path))
    )
    filesystem = ParentSwapFilesystemOps(target_path, moved, outside)

    with pytest.raises(
        ManifestError, match="filesystem|durability|durable|identity"
    ):
        ModelDownloader(manifest, filesystem=filesystem).install_bytes(
            "selected-mt", "model.bin", [b"payload"]
        )

    assert filesystem.swapped
    assert "commit_noreplace" in [name for name, _ in filesystem.calls]
    assert not (outside / "model.bin").exists()
    assert filesystem.open_fds == set()


@pytest.mark.parametrize("payload", [b"short", b"payload-too-long"])
def test_declared_size_mismatch_removes_staging_and_final(
    tmp_path: Path, payload: bytes
) -> None:
    document = _manifest(tmp_path / "cache")
    manifest = load_manifest(_write_manifest(tmp_path, document))

    with pytest.raises(ManifestError, match="size"):
        ModelDownloader(manifest).install_bytes(
            "selected-mt", "model.bin", [payload]
        )

    assert not (tmp_path / "cache" / "model.bin").exists()
    assert not (tmp_path / "cache" / ".staging" / "model.bin.part").exists()


def test_checksum_failure_removes_part_and_never_exposes_final(
    tmp_path: Path,
) -> None:
    document = _manifest(
        tmp_path / "cache",
        sha256=_sha256(b"expected"),
    )
    manifest = load_manifest(_write_manifest(tmp_path, document))
    downloader = ModelDownloader(manifest)

    with pytest.raises(ManifestError, match="checksum"):
        downloader.install_bytes("selected-mt", "model.bin", [b"damaged"])

    target = tmp_path / "cache" / "model.bin"
    part = tmp_path / "cache" / ".staging" / "model.bin.part"
    assert not target.exists()
    assert not part.exists()


@pytest.mark.parametrize("failure", [RuntimeError("disconnect"), OSError("socket")])
def test_transport_iterator_failure_cleans_partial_and_is_typed(
    tmp_path: Path, failure: Exception
) -> None:
    manifest = load_manifest(
        _write_manifest(tmp_path, _manifest(tmp_path / "cache"))
    )

    def failing_chunks() -> Iterable[bytes]:
        yield b"pay"
        raise failure

    with pytest.raises(ManifestError, match="transport"):
        ModelDownloader(manifest).install_bytes(
            "selected-mt", "model.bin", failing_chunks()
        )

    assert not (tmp_path / "cache" / "model.bin").exists()
    assert not (tmp_path / "cache" / ".staging" / "model.bin.part").exists()
    restarted = DownloadLedger(
        budget_bytes=2 * GIB,
        planned_bytes=7,
        staging_dir=tmp_path / "cache" / ".staging",
    )
    assert restarted.transferred_bytes == 3


@pytest.mark.parametrize("fail_at", ["fsync_file", "atomic_replace"])
def test_install_failure_before_commit_cleans_part_and_final(
    tmp_path: Path, fail_at: str
) -> None:
    manifest = load_manifest(
        _write_manifest(tmp_path, _manifest(tmp_path / "cache"))
    )
    fs = RecordingFilesystemOps(fail_at=fail_at)

    with pytest.raises(ManifestError):
        ModelDownloader(manifest, filesystem=fs).install_bytes(
            "selected-mt", "model.bin", [b"payload"]
        )

    assert not (tmp_path / "cache" / "model.bin").exists()
    assert not (tmp_path / "cache" / ".staging" / "model.bin.part").exists()


def test_directory_fsync_failure_quarantines_indeterminate_target(
    tmp_path: Path,
) -> None:
    manifest = load_manifest(
        _write_manifest(tmp_path, _manifest(tmp_path / "cache"))
    )
    fs = RecordingFilesystemOps(fail_at="fsync_directory")

    with pytest.raises(InstallDurabilityError) as error:
        ModelDownloader(manifest, filesystem=fs).install_bytes(
            "selected-mt", "model.bin", [b"payload"]
        )

    quarantine_path = error.value.quarantine_path
    operation_names = [name for name, _ in fs.calls]
    assert operation_names.index("atomic_replace") < operation_names.index(
        "quarantine"
    )
    assert error.value.state == "durability_indeterminate"
    assert quarantine_path.parent == (
        tmp_path / "cache" / ".staging" / ".quarantine"
    )
    assert quarantine_path.is_file()
    assert not quarantine_path.is_symlink()
    assert quarantine_path.read_bytes() == b"payload"
    assert not (tmp_path / "cache" / "model.bin").exists()
    assert not (tmp_path / "cache" / ".staging" / "model.bin.part").exists()


def test_verified_file_is_atomically_installed_and_runtime_is_offline(
    tmp_path: Path,
) -> None:
    payload = b"payload"
    document = _manifest(tmp_path / "cache", sha256=_sha256(payload))
    manifest = load_manifest(_write_manifest(tmp_path, document))
    fs = RecordingFilesystemOps()
    downloader = ModelDownloader(manifest, filesystem=fs)

    installed = downloader.install_bytes(
        "selected-mt",
        "model.bin",
        [payload[:3], payload[3:]],
    )

    assert installed == (tmp_path / "cache" / "model.bin").resolve()
    assert installed.read_bytes() == payload
    assert manifest.resolve_runtime_file("selected-mt", "model.bin") == installed
    assert {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }.items() <= manifest.runtime_environment().items()
    operation_names = [name for name, _ in fs.calls]
    assert "open_exclusive" in operation_names
    assert operation_names.index("fsync_file") < operation_names.index(
        "atomic_replace"
    )
    assert operation_names.index("atomic_replace") < operation_names.index(
        "fsync_directory"
    )


@pytest.mark.parametrize("unsafe_kind", ["missing", "directory", "symlink"])
def test_runtime_resolution_rejects_non_regular_or_missing_asset(
    tmp_path: Path, unsafe_kind: str
) -> None:
    payload = b"payload"
    manifest = load_manifest(
        _write_manifest(
            tmp_path,
            _manifest(tmp_path / "cache", sha256=_sha256(payload)),
        )
    )
    target = tmp_path / "cache" / "model.bin"
    target.parent.mkdir(parents=True)
    if unsafe_kind == "directory":
        target.mkdir()
    elif unsafe_kind == "symlink":
        outside = tmp_path / "outside.bin"
        outside.write_bytes(payload)
        target.symlink_to(outside)

    with pytest.raises(ManifestError, match="runtime"):
        manifest.resolve_runtime_file("selected-mt", "model.bin")


@pytest.mark.parametrize(
    ("payload", "message"),
    [(b"short", "size"), (b"payloae", "checksum")],
)
def test_runtime_resolution_rejects_regular_file_size_or_hash_mismatch(
    tmp_path: Path, payload: bytes, message: str
) -> None:
    manifest = load_manifest(
        _write_manifest(tmp_path, _manifest(tmp_path / "cache"))
    )
    target = tmp_path / "cache" / "model.bin"
    target.parent.mkdir(parents=True)
    target.write_bytes(payload)

    with pytest.raises(ManifestError, match=message):
        manifest.resolve_runtime_file("selected-mt", "model.bin")


def test_quarantine_fsyncs_quarantine_target_and_staging_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging = tmp_path / "staging"
    target = tmp_path / "target"
    staging.mkdir()
    target.mkdir()
    (target / "model.bin").write_bytes(b"payload")
    fs = FilesystemOps()
    staging_fd = fs.open_directory(staging)
    target_fd = fs.open_directory(target)
    original_fsync = os.fsync
    original_mkdir = os.mkdir
    original_link = os.link
    original_unlink = os.unlink
    events: list[tuple[str, int | None]] = []

    def recording_fsync(descriptor: int) -> None:
        events.append(("fsync", descriptor))
        original_fsync(descriptor)

    def recording_mkdir(
        path: str, mode: int = 0o777, *, dir_fd: int | None = None
    ) -> None:
        events.append(("mkdir", dir_fd))
        original_mkdir(path, mode=mode, dir_fd=dir_fd)

    def recording_link(
        src: str,
        dst: str,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> None:
        events.append(("link", dst_dir_fd))
        original_link(
            src,
            dst,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
            follow_symlinks=follow_symlinks,
        )

    def recording_unlink(
        path: str, *, dir_fd: int | None = None
    ) -> None:
        events.append(("unlink", dir_fd))
        original_unlink(path, dir_fd=dir_fd)

    monkeypatch.setattr(os, "fsync", recording_fsync)
    monkeypatch.setattr(os, "mkdir", recording_mkdir)
    monkeypatch.setattr(os, "link", recording_link)
    monkeypatch.setattr(os, "unlink", recording_unlink)
    try:
        fs.quarantine_at(
            target_fd,
            "model.bin",
            staging_fd,
            "model.bin.quarantine",
        )
        mkdir_index = events.index(("mkdir", staging_fd))
        unlink_index = events.index(("unlink", target_fd))
        link_index = next(
            index for index, (event, _) in enumerate(events) if event == "link"
        )
        quarantine_fd = events[link_index][1]
        target_sync = events.index(("fsync", target_fd))
        staging_sync = events.index(("fsync", staging_fd))
        quarantine_sync = events.index(("fsync", quarantine_fd))
        assert unlink_index < target_sync
        assert mkdir_index < staging_sync
        assert link_index < quarantine_sync
        assert len({fd for event, fd in events if event == "fsync"}) >= 3
        assert not (target / "model.bin").exists()
        assert (
            staging / ".quarantine" / "model.bin.quarantine"
        ).read_bytes() == b"payload"
    finally:
        fs.close_directory(target_fd)
        fs.close_directory(staging_fd)


def test_secure_commit_never_replaces_competing_target(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    target = tmp_path / "target"
    staging.mkdir()
    target.mkdir()
    fs = FilesystemOps()
    staging_fd = fs.open_directory(staging)
    target_fd = fs.open_directory(target)
    try:
        with fs.open_exclusive_at(staging_fd, "model.bin.part") as output:
            output.write(b"model")
            fs.fsync_file(output)
        (target / "model.bin").write_bytes(b"competitor")

        with pytest.raises(FileExistsError):
            fs.commit_noreplace(
                staging_fd,
                "model.bin.part",
                target_fd,
                "model.bin",
            )

        assert (target / "model.bin").read_bytes() == b"competitor"
        assert (staging / "model.bin.part").read_bytes() == b"model"
    finally:
        fs.close_directory(target_fd)
        fs.close_directory(staging_fd)


def test_secure_commit_uses_pinned_directory_after_parent_path_swap(
    tmp_path: Path,
) -> None:
    staging = tmp_path / "staging"
    target = tmp_path / "target"
    moved_target = tmp_path / "target-pinned"
    outside = tmp_path / "outside"
    staging.mkdir()
    target.mkdir()
    outside.mkdir()
    fs = FilesystemOps()
    staging_fd = fs.open_directory(staging)
    target_fd = fs.open_directory(target)
    try:
        with fs.open_exclusive_at(staging_fd, "model.bin.part") as output:
            output.write(b"model")
            fs.fsync_file(output)
        target.rename(moved_target)
        target.symlink_to(outside, target_is_directory=True)

        fs.commit_noreplace(
            staging_fd,
            "model.bin.part",
            target_fd,
            "model.bin",
        )

        assert (moved_target / "model.bin").read_bytes() == b"model"
        assert not (outside / "model.bin").exists()
        assert not fs.directory_matches(target, target_fd)
    finally:
        fs.close_directory(target_fd)
        fs.close_directory(staging_fd)


def test_secure_directory_open_rejects_symlink_component(
    tmp_path: Path,
) -> None:
    real = tmp_path / "real"
    nested = real / "nested"
    nested.mkdir(parents=True)
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)

    with pytest.raises(OSError):
        FilesystemOps().open_directory(link / "nested")

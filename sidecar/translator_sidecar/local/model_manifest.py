"""Fail-closed local model manifest and atomic installer."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import BinaryIO, Iterable
from urllib.parse import quote, urlparse
from uuid import uuid4


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_REPOSITORY_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_MODEL_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_HUGGINGFACE_HOST = "huggingface.co"
_QUARANTINE_DIR = ".quarantine"
_LANGUAGES = {"ru", "en"}
_ROLES = {"asr", "mt", "tts"}
_ACQUISITIONS = {"reuse", "download"}
_APPROVED_LICENSES = {
    "MIT",
    "CC0",
    "CC-BY-NC-4.0",
    "CC-BY-NC-SA-4.0",
}
_UNKNOWN_LICENSES = {"", "unknown", "unlicensed", "none", "n/a"}
_IRINA_WAIVER = "PIPER_RU_IRINA_PERSONAL_LOCAL_V1"
_IRINA_REVISION = "0d907f158acc877ddeebcbf827659ee13bea8bcd"
_IRINA_FILES = {
    "ru_RU-irina-medium.onnx": (
        63_201_294,
        "8ff38212d23da300bbe3705c645e6e5b9475f0bfde01558eb17813e22acaaaaa",
        f"https://{_HUGGINGFACE_HOST}/rhasspy/piper-voices/resolve/"
        f"{_IRINA_REVISION}/ru/ru_RU/irina/medium/"
        "ru_RU-irina-medium.onnx",
        "ru/ru_RU/irina/medium/ru_RU-irina-medium.onnx",
    ),
    "ru_RU-irina-medium.onnx.json": (
        4_765,
        "c2ec28bb38e2b59e93b959b3e40348c1afebbd272f30fed5d41205d08e98a9d7",
        f"https://{_HUGGINGFACE_HOST}/rhasspy/piper-voices/resolve/"
        f"{_IRINA_REVISION}/ru/ru_RU/irina/medium/"
        "ru_RU-irina-medium.onnx.json",
        "ru/ru_RU/irina/medium/ru_RU-irina-medium.onnx.json",
    ),
}


class ManifestError(RuntimeError):
    """The manifest or a model installation violated a safety invariant."""


class InstallDurabilityError(ManifestError):
    """A renamed target could not be proven durable and was quarantined."""

    state = "durability_indeterminate"

    def __init__(self, quarantine_path: Path) -> None:
        self.quarantine_path = quarantine_path
        super().__init__("model install durability is indeterminate")


class LedgerDurabilityError(ManifestError):
    """A ledger rename succeeded but directory durability is unknown."""

    state = "durability_indeterminate"

    def __init__(self) -> None:
        super().__init__("download ledger durability is indeterminate")


class _TransportReadError(ManifestError):
    def __init__(self) -> None:
        super().__init__("download transport failed")


@dataclass(frozen=True, slots=True)
class ManifestPolicy:
    download_budget_bytes: int
    post_download_free_floor_bytes: int
    usage_mode: str
    redistribution: bool
    certified_or_safety_critical: bool
    staging_path: Path
    redirect_hosts: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ModelSource:
    repository: str
    revision: str
    license: str
    dataset_license: str | None = None
    license_waiver: str | None = None


@dataclass(frozen=True, slots=True)
class ModelFile:
    path: str
    size_bytes: int
    sha256: str
    source_url: str | None = None
    source_path: str | None = None


@dataclass(frozen=True, slots=True)
class ModelEntry:
    id: str
    role: str
    source: ModelSource
    languages: tuple[str, ...]
    cache_path: Path
    acquisition: str
    files: tuple[ModelFile, ...]


class RuntimeFileOps:
    """Stable runtime-file opening boundary."""

    def open_stable(
        self, path: Path, *, allowed_symlink_root: Path | None
    ) -> tuple[int, Path]:
        initial = path.lstat()
        if stat.S_ISLNK(initial.st_mode):
            if allowed_symlink_root is None:
                raise OSError("runtime symlink is not approved")
            resolved = path.resolve(strict=True)
            _require_within(resolved, allowed_symlink_root)
            descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
            final = path.lstat()
            if (initial.st_dev, initial.st_ino) != (final.st_dev, final.st_ino):
                os.close(descriptor)
                raise OSError("runtime symlink changed during open")
        else:
            descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
            resolved = path
        descriptor_stat = os.fstat(descriptor)
        resolved_stat = resolved.stat()
        if not stat.S_ISREG(descriptor_stat.st_mode) or (
            descriptor_stat.st_dev,
            descriptor_stat.st_ino,
        ) != (resolved_stat.st_dev, resolved_stat.st_ino):
            os.close(descriptor)
            raise OSError("runtime file identity changed during open")
        return descriptor, resolved


@dataclass(frozen=True, slots=True)
class ModelManifest:
    schema_version: int
    policy: ManifestPolicy
    models: dict[str, ModelEntry]
    planned_download_bytes: int

    def model_file(self, model_id: str, file_path: str) -> tuple[ModelEntry, ModelFile]:
        try:
            model = self.models[model_id]
        except KeyError as error:
            raise ManifestError("unknown model id") from error
        for model_file in model.files:
            if model_file.path == file_path:
                return model, model_file
        raise ManifestError("file is not allowlisted for model")

    def validate_download_chain(
        self, model_id: str, file_path: str, urls: list[str]
    ) -> None:
        model, model_file = self.model_file(model_id, file_path)
        if model.acquisition != "download" or model_file.source_url is None:
            raise ManifestError("file has no approved download source")
        if not urls or urls[0] != model_file.source_url:
            raise ManifestError("initial source URL does not match pinned source")

        for index, url in enumerate(urls):
            parsed = urlparse(url)
            if (
                parsed.scheme != "https"
                or parsed.username is not None
                or parsed.password is not None
                or parsed.hostname is None
            ):
                label = "source" if index == 0 else "redirect"
                raise ManifestError(f"{label} URL is not allowed")
            if index == 0:
                if parsed.hostname != _HUGGINGFACE_HOST:
                    raise ManifestError("source URL host is not allowed")
            elif parsed.hostname == _HUGGINGFACE_HOST:
                expected_cache_path = (
                    f"/api/resolve-cache/models/{model.source.repository}/"
                    f"{model.source.revision}/"
                    f"{quote(model_file.source_path or '', safe='')}"
                )
                if parsed.path != expected_cache_path:
                    raise ManifestError("redirect host cache path is not pinned")
            elif parsed.hostname not in self.policy.redirect_hosts:
                raise ManifestError("redirect host is not allowlisted")

    def resolve_runtime_file(
        self,
        model_id: str,
        file_path: str,
        *,
        filesystem: RuntimeFileOps | None = None,
    ) -> Path:
        model, model_file = self.model_file(model_id, file_path)
        target = _target_path(model, model_file)
        try:
            metadata = target.lstat()
        except OSError as error:
            raise ManifestError("runtime model file is missing") from error
        allowed_root: Path | None = None
        if stat.S_ISLNK(metadata.st_mode):
            allowed_root = _hugging_face_blob_root(model)
            if allowed_root is None:
                raise ManifestError("runtime symlink is outside a pinned HF snapshot")
        elif not stat.S_ISREG(metadata.st_mode):
            raise ManifestError("runtime model path is not a regular file")
        runtime_filesystem = filesystem or RuntimeFileOps()
        descriptor: int | None = None
        try:
            descriptor, resolved = runtime_filesystem.open_stable(
                target, allowed_symlink_root=allowed_root
            )
            if allowed_root is not None:
                _require_within(resolved, allowed_root)
            descriptor_stat = os.fstat(descriptor)
            if descriptor_stat.st_size != model_file.size_bytes:
                raise ManifestError("runtime model file size mismatch")
            if _sha256_fd(descriptor) != model_file.sha256:
                raise ManifestError("runtime model file checksum mismatch")
            return resolved
        except ManifestError:
            raise
        except OSError as error:
            raise ManifestError("runtime model file or blob is unsafe") from error
        finally:
            if descriptor is not None:
                os.close(descriptor)

    @staticmethod
    def runtime_environment() -> dict[str, str]:
        return {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
        }


class DownloadLedger:
    def __init__(
        self,
        *,
        budget_bytes: int,
        planned_bytes: int,
        staging_dir: Path,
        filesystem: FilesystemOps | None = None,
    ) -> None:
        if budget_bytes <= 0 or planned_bytes < 0 or planned_bytes > budget_bytes:
            raise ManifestError("planned model download exceeds budget")
        self.budget_bytes = budget_bytes
        self.planned_bytes = planned_bytes
        self.staging_dir = staging_dir
        self.filesystem = filesystem or FilesystemOps()
        self.state = "ready"
        self.filesystem.ensure_directory(staging_dir)
        directory_fd = self.filesystem.open_directory(staging_dir)
        try:
            persisted = self._load_persisted(directory_fd)
            if persisted is None:
                self.transferred_bytes = self._existing_partial_bytes(directory_fd)
                if self.transferred_bytes:
                    self._persist(self.transferred_bytes, directory_fd=directory_fd)
            else:
                self.transferred_bytes = persisted
        finally:
            self.filesystem.close_directory(directory_fd)
        if self.transferred_bytes > self.budget_bytes:
            raise ManifestError("partial model bytes exceed download budget")

    @property
    def _ledger_path(self) -> Path:
        return self.staging_dir / ".download-ledger.json"

    def _load_persisted(self, directory_fd: int) -> int | None:
        try:
            metadata = os.stat(
                self._ledger_path.name,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return None
        except OSError as error:
            raise ManifestError("download ledger state is inaccessible") from error
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise ManifestError("download ledger state is not a regular file")
        descriptor: int | None = None
        try:
            descriptor = self.filesystem.open_readonly_at(
                directory_fd, self._ledger_path.name
            )
            opened = os.fstat(descriptor)
            if (metadata.st_dev, metadata.st_ino) != (
                opened.st_dev,
                opened.st_ino,
            ):
                raise ManifestError("download ledger inode changed during open")
            payload = b""
            while block := os.read(descriptor, 4096):
                payload += block
                if len(payload) > 4096:
                    raise ManifestError("download ledger state is oversized")
            document = json.loads(payload)
            if (
                not isinstance(document, dict)
                or set(document) != {"schema_version", "transferred_bytes"}
                or document["schema_version"] != 1
                or isinstance(document["transferred_bytes"], bool)
                or not isinstance(document["transferred_bytes"], int)
                or document["transferred_bytes"] < 0
            ):
                raise ManifestError("download ledger state is invalid")
            return document["transferred_bytes"]
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ManifestError("download ledger state is invalid") from error
        finally:
            if descriptor is not None:
                os.close(descriptor)

    def _existing_partial_bytes(self, directory_fd: int) -> int:
        total = 0
        for entry in os.scandir(directory_fd):
            if not entry.name.endswith(".part"):
                continue
            metadata = entry.stat(follow_symlinks=False)
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                raise ManifestError("staging part is not a regular file")
            total += metadata.st_size
        return total

    def record_received(self, byte_count: int) -> None:
        self.check_can_receive(byte_count)
        candidate = self.transferred_bytes + byte_count
        self._persist(candidate)
        self.transferred_bytes = candidate

    def check_can_receive(self, byte_count: int) -> None:
        if self.state != "ready":
            raise ManifestError("download ledger is not usable")
        if byte_count < 0:
            raise ManifestError("received byte count must be non-negative")
        candidate = self.transferred_bytes + byte_count
        if candidate > self.budget_bytes:
            raise ManifestError("transferred model bytes exceed download budget")

    def _persist(
        self, transferred_bytes: int, *, directory_fd: int | None = None
    ) -> None:
        payload = json.dumps(
            {
                "schema_version": 1,
                "transferred_bytes": transferred_bytes,
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        temp_name = f".download-ledger.{uuid4().hex}.tmp"
        owns_directory_fd = directory_fd is None
        if directory_fd is None:
            directory_fd = self.filesystem.open_directory(self.staging_dir)
        renamed = False
        try:
            output = self.filesystem.open_exclusive_at(directory_fd, temp_name)
            with output:
                output.write(payload)
                self.filesystem.fsync_file(output)
            self.filesystem.replace_at(directory_fd, temp_name, self._ledger_path.name)
            renamed = True
            self.filesystem.fsync_directory_fd(directory_fd)
            if not self.filesystem.directory_matches(self.staging_dir, directory_fd):
                raise OSError("download ledger parent identity changed")
        except OSError as error:
            if not renamed:
                self.filesystem.unlink_at(directory_fd, temp_name)
                raise ManifestError("download ledger update failed") from error
            self.state = LedgerDurabilityError.state
            raise LedgerDurabilityError() from error
        finally:
            if owns_directory_fd:
                self.filesystem.close_directory(directory_fd)

    @staticmethod
    def check_free_space(
        *,
        free_bytes: int,
        remaining_download_bytes: int,
        floor_bytes: int,
    ) -> None:
        if (
            free_bytes < 0
            or remaining_download_bytes < 0
            or floor_bytes < 0
            or free_bytes - remaining_download_bytes < floor_bytes
        ):
            raise ManifestError("insufficient post-download free space")


class FilesystemOps:
    """Small injectable boundary for durable installation operations."""

    def ensure_directory(self, path: Path) -> None:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise OSError("directory path is unsafe")

    def open_directory(self, path: Path) -> int:
        if not path.is_absolute():
            raise OSError("directory path must be absolute")
        descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            for component in path.parts[1:]:
                next_descriptor = os.open(
                    component,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                    dir_fd=descriptor,
                )
                os.close(descriptor)
                descriptor = next_descriptor
            return descriptor
        except BaseException:
            os.close(descriptor)
            raise

    def close_directory(self, descriptor: int) -> None:
        os.close(descriptor)

    def available_bytes_fd(self, descriptor: int) -> int:
        filesystem = os.fstatvfs(descriptor)
        return filesystem.f_bavail * filesystem.f_frsize

    def device_id_fd(self, descriptor: int) -> int:
        return os.fstat(descriptor).st_dev

    def directory_matches(self, path: Path, descriptor: int) -> bool:
        try:
            current = self.open_directory(path)
        except OSError:
            return False
        try:
            expected_stat = os.fstat(descriptor)
            current_stat = os.fstat(current)
            return (expected_stat.st_dev, expected_stat.st_ino) == (
                current_stat.st_dev,
                current_stat.st_ino,
            )
        finally:
            self.close_directory(current)

    def exists_at(self, descriptor: int, name: str) -> bool:
        try:
            os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            return True
        except FileNotFoundError:
            return False

    def open_exclusive_at(self, descriptor: int, name: str) -> BinaryIO:
        file_descriptor = os.open(
            name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=descriptor,
        )
        return os.fdopen(file_descriptor, "wb")

    def open_readonly_at(self, descriptor: int, name: str) -> int:
        return os.open(
            name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=descriptor,
        )

    def replace_at(self, descriptor: int, source_name: str, target_name: str) -> None:
        os.replace(
            source_name,
            target_name,
            src_dir_fd=descriptor,
            dst_dir_fd=descriptor,
        )

    def commit_noreplace(
        self,
        staging_fd: int,
        part_name: str,
        target_fd: int,
        target_name: str,
    ) -> None:
        os.link(
            part_name,
            target_name,
            src_dir_fd=staging_fd,
            dst_dir_fd=target_fd,
            follow_symlinks=False,
        )
        os.unlink(part_name, dir_fd=staging_fd)

    def fsync_directory_fd(self, descriptor: int) -> None:
        os.fsync(descriptor)

    def unlink_at(self, descriptor: int, name: str) -> None:
        try:
            os.unlink(name, dir_fd=descriptor)
        except FileNotFoundError:
            return

    def quarantine_at(
        self,
        target_fd: int,
        target_name: str,
        staging_fd: int,
        quarantine_name: str,
    ) -> None:
        try:
            os.mkdir(_QUARANTINE_DIR, mode=0o700, dir_fd=staging_fd)
        except FileExistsError:
            pass
        quarantine_fd = os.open(
            _QUARANTINE_DIR,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=staging_fd,
        )
        try:
            os.link(
                target_name,
                quarantine_name,
                src_dir_fd=target_fd,
                dst_dir_fd=quarantine_fd,
                follow_symlinks=False,
            )
            os.unlink(target_name, dir_fd=target_fd)
            os.fsync(quarantine_fd)
            os.fsync(target_fd)
            os.fsync(staging_fd)
        finally:
            os.close(quarantine_fd)

    def available_bytes(self, path: Path) -> int:
        filesystem = os.statvfs(path)
        return filesystem.f_bavail * filesystem.f_frsize

    def device_id(self, path: Path) -> int:
        return path.stat().st_dev

    def open_exclusive(self, path: Path) -> BinaryIO:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
        )
        return os.fdopen(descriptor, "wb")

    def fsync_file(self, file: BinaryIO) -> None:
        file.flush()
        os.fsync(file.fileno())

    def atomic_replace(self, source: Path, target: Path) -> None:
        os.replace(source, target)

    def fsync_directory(self, path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def quarantine(self, target: Path, quarantine_path: Path) -> None:
        self.ensure_directory(quarantine_path.parent)
        os.replace(target, quarantine_path)

    def unlink(self, path: Path) -> None:
        try:
            path.unlink()
        except FileNotFoundError:
            return


class ModelDownloader:
    def __init__(
        self,
        manifest: ModelManifest,
        *,
        filesystem: FilesystemOps | None = None,
    ) -> None:
        self.manifest = manifest
        self.filesystem = filesystem or FilesystemOps()
        self.ledger = DownloadLedger(
            budget_bytes=manifest.policy.download_budget_bytes,
            planned_bytes=manifest.planned_download_bytes,
            staging_dir=manifest.policy.staging_path,
            filesystem=self.filesystem,
        )
        self._completed: set[tuple[str, str]] = set()

    def install_bytes(
        self, model_id: str, file_path: str, chunks: Iterable[bytes]
    ) -> Path:
        model, model_file = self.manifest.model_file(model_id, file_path)
        if model.acquisition != "download":
            raise ManifestError("reused model files cannot be downloaded")
        self.ledger.check_can_receive(model_file.size_bytes)
        target = _target_path(model, model_file)
        staging = self.manifest.policy.staging_path
        staging_fd: int | None = None
        target_fd: int | None = None
        try:
            self.filesystem.ensure_directory(staging)
            self.filesystem.ensure_directory(target.parent)
            staging_fd = self.filesystem.open_directory(staging)
            target_fd = self.filesystem.open_directory(target.parent)
            staging_device = self.filesystem.device_id_fd(staging_fd)
            target_device = self.filesystem.device_id_fd(target_fd)
        except OSError as error:
            if target_fd is not None:
                self.filesystem.close_directory(target_fd)
            if staging_fd is not None:
                self.filesystem.close_directory(staging_fd)
            raise ManifestError("filesystem identity probe failed") from error
        if staging_device != target_device:
            self.filesystem.close_directory(target_fd)
            self.filesystem.close_directory(staging_fd)
            raise ManifestError("staging and final paths use different filesystems")

        try:
            try:
                free_bytes = self.filesystem.available_bytes_fd(staging_fd)
            except OSError as error:
                raise ManifestError("free space probe failed") from error
            DownloadLedger.check_free_space(
                free_bytes=free_bytes,
                remaining_download_bytes=self._remaining_download_bytes(),
                floor_bytes=self.manifest.policy.post_download_free_floor_bytes,
            )

            part_name = f"{model_file.path.replace('/', '--')}.part"
            if self.filesystem.exists_at(target_fd, model_file.path):
                raise ManifestError("runtime target already exists")
            try:
                output = self.filesystem.open_exclusive_at(staging_fd, part_name)
            except OSError as error:
                raise ManifestError(
                    "staging file could not be created exclusively"
                ) from error

            digest = hashlib.sha256()
            received = 0
            committed = False
            try:
                with output:
                    iterator = iter(chunks)
                    while True:
                        try:
                            chunk = next(iterator)
                        except StopIteration:
                            break
                        except ManifestError:
                            raise
                        except Exception as error:
                            raise _TransportReadError() from error
                        if not isinstance(chunk, bytes):
                            raise ManifestError("download chunk must be bytes")
                        self.ledger.record_received(len(chunk))
                        received += len(chunk)
                        if received > model_file.size_bytes:
                            raise ManifestError(
                                "downloaded file size exceeds declaration"
                            )
                        output.write(chunk)
                        digest.update(chunk)
                    if received != model_file.size_bytes:
                        raise ManifestError("downloaded file size mismatch")
                    if digest.hexdigest() != model_file.sha256:
                        raise ManifestError("downloaded file checksum mismatch")
                    self.filesystem.fsync_file(output)
                self.filesystem.commit_noreplace(
                    staging_fd,
                    part_name,
                    target_fd,
                    model_file.path,
                )
                committed = True
                self.filesystem.fsync_directory_fd(target_fd)
                self.filesystem.fsync_directory_fd(staging_fd)
                if not self.filesystem.directory_matches(
                    target.parent, target_fd
                ) or not self.filesystem.directory_matches(staging, staging_fd):
                    raise OSError("pinned directory identity changed")
            except Exception as error:
                if committed:
                    quarantine_name = (
                        f"{model.id}--{model_file.path.replace('/', '--')}"
                        f"--{uuid4().hex}"
                    )
                    quarantine = staging / _QUARANTINE_DIR / quarantine_name
                    try:
                        self.filesystem.quarantine_at(
                            target_fd,
                            model_file.path,
                            staging_fd,
                            quarantine_name,
                        )
                    except OSError:
                        quarantine = target
                    raise InstallDurabilityError(quarantine) from error
                self.filesystem.unlink_at(staging_fd, part_name)
                if isinstance(error, ManifestError):
                    raise
                if isinstance(error, FileExistsError):
                    raise ManifestError(
                        "target commit lost an atomic no-replace race"
                    ) from error
                if isinstance(error, OSError):
                    raise ManifestError("durable model installation failed") from error
                raise ManifestError("download transport failed") from error

            self._completed.add((model.id, model_file.path))
            return target.resolve(strict=True)
        finally:
            self.filesystem.close_directory(target_fd)
            self.filesystem.close_directory(staging_fd)

    def _remaining_download_bytes(self) -> int:
        total = 0
        for model in self.manifest.models.values():
            if model.acquisition != "download":
                continue
            for model_file in model.files:
                identity = (model.id, model_file.path)
                if identity in self._completed:
                    continue
                target = _target_path(model, model_file)
                if target.exists() or target.is_symlink():
                    self.manifest.resolve_runtime_file(model.id, model_file.path)
                    self._completed.add(identity)
                    continue
                total += model_file.size_bytes
        return total


def load_manifest(path: Path) -> ModelManifest:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ManifestError("model manifest could not be read") from error
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise ManifestError("unsupported model manifest schema")

    policy = _parse_policy(document.get("policy"))
    raw_models = document.get("models")
    if not isinstance(raw_models, list) or not raw_models:
        raise ManifestError("manifest must contain models")

    models: dict[str, ModelEntry] = {}
    for raw_model in raw_models:
        model = _parse_model(raw_model, policy)
        if model.id in models:
            raise ManifestError("duplicate model id")
        models[model.id] = model
    planned = sum(
        model_file.size_bytes
        for model in models.values()
        if model.acquisition == "download"
        for model_file in model.files
    )
    if planned > policy.download_budget_bytes:
        raise ManifestError("planned model download exceeds budget")
    return ModelManifest(
        schema_version=1,
        policy=policy,
        models=models,
        planned_download_bytes=planned,
    )


def _parse_policy(raw: object) -> ManifestPolicy:
    if not isinstance(raw, dict):
        raise ManifestError("manifest policy is missing")
    try:
        budget = _positive_int(raw["download_budget_bytes"])
        floor = _positive_int(raw["post_download_free_floor_bytes"])
        usage_mode = raw["usage_mode"]
        redistribution = raw["redistribution"]
        safety = raw["certified_or_safety_critical"]
        staging = Path(raw["staging_path"])
        redirect_hosts = raw["redirect_hosts"]
    except (KeyError, TypeError) as error:
        raise ManifestError("manifest policy is incomplete") from error
    if (
        usage_mode != "personal_noncommercial"
        or redistribution is not False
        or safety is not False
    ):
        raise ManifestError("manifest usage policy is not approved")
    if not staging.is_absolute():
        raise ManifestError("staging path must be absolute")
    if not isinstance(redirect_hosts, list) or not redirect_hosts:
        raise ManifestError("redirect host allowlist is missing")
    normalized_hosts: list[str] = []
    for host in redirect_hosts:
        if (
            not isinstance(host, str)
            or not host
            or urlparse(f"https://{host}").hostname != host
            or "/" in host
            or "@" in host
        ):
            raise ManifestError("redirect host is invalid")
        normalized_hosts.append(host)
    if len(set(normalized_hosts)) != len(normalized_hosts):
        raise ManifestError("redirect host allowlist contains duplicates")
    return ManifestPolicy(
        download_budget_bytes=budget,
        post_download_free_floor_bytes=floor,
        usage_mode=usage_mode,
        redistribution=redistribution,
        certified_or_safety_critical=safety,
        staging_path=staging,
        redirect_hosts=tuple(normalized_hosts),
    )


def _parse_model(raw: object, policy: ManifestPolicy) -> ModelEntry:
    if not isinstance(raw, dict):
        raise ManifestError("model entry must be an object")
    try:
        model_id = raw["id"]
        role = raw["role"]
        acquisition = raw["acquisition"]
        languages = raw["languages"]
        cache_path = Path(raw["cache_path"])
        source = _parse_source(raw["source"])
        raw_files = raw["files"]
    except (KeyError, TypeError) as error:
        raise ManifestError("model entry is incomplete") from error
    if not isinstance(model_id, str) or not _MODEL_ID_RE.fullmatch(model_id):
        raise ManifestError("model id is invalid")
    if role not in _ROLES or acquisition not in _ACQUISITIONS:
        raise ManifestError("model role or acquisition is invalid")
    if (
        not isinstance(languages, list)
        or not languages
        or any(language not in _LANGUAGES for language in languages)
        or len(set(languages)) != len(languages)
    ):
        raise ManifestError("model languages are invalid")
    if not cache_path.is_absolute():
        raise ManifestError("model cache path must be absolute")
    if not isinstance(raw_files, list) or not raw_files:
        raise ManifestError("model file allowlist is empty")
    files = tuple(_parse_file(raw_file, source, acquisition) for raw_file in raw_files)
    if len({model_file.path for model_file in files}) != len(files):
        raise ManifestError("model file allowlist contains duplicates")
    model = ModelEntry(
        id=model_id,
        role=role,
        source=source,
        languages=tuple(languages),
        cache_path=cache_path,
        acquisition=acquisition,
        files=files,
    )
    _validate_license_waiver(model, policy)
    return model


def _parse_source(raw: object) -> ModelSource:
    if not isinstance(raw, dict):
        raise ManifestError("model source is missing")
    try:
        repository = raw["repository"]
        revision = raw["revision"]
        license_id = raw["license"]
    except KeyError as error:
        raise ManifestError("model source provenance is incomplete") from error
    if not isinstance(repository, str) or not _REPOSITORY_RE.fullmatch(repository):
        raise ManifestError("model repository is invalid")
    if not isinstance(revision, str) or not _REVISION_RE.fullmatch(revision):
        raise ManifestError("model revision must be a pinned commit")
    if not isinstance(license_id, str) or license_id not in _APPROVED_LICENSES:
        raise ManifestError("model license is unknown")
    dataset_license = raw.get("dataset_license")
    waiver = raw.get("license_waiver")
    if dataset_license is not None and not isinstance(dataset_license, str):
        raise ManifestError("dataset license is invalid")
    if waiver is not None and not isinstance(waiver, str):
        raise ManifestError("license waiver is invalid")
    if isinstance(dataset_license, str):
        normalized = dataset_license.strip().lower()
        if normalized in _UNKNOWN_LICENSES:
            dataset_license = "Unknown"
        elif dataset_license not in _APPROVED_LICENSES:
            raise ManifestError("dataset license is not approved")
    if dataset_license == "Unknown" and waiver is None:
        raise ManifestError("unknown dataset license requires a waiver")
    if waiver is not None and dataset_license != "Unknown":
        raise ManifestError("license waiver is not applicable")
    return ModelSource(
        repository=repository,
        revision=revision,
        license=license_id,
        dataset_license=dataset_license,
        license_waiver=waiver,
    )


def _parse_file(raw: object, source: ModelSource, acquisition: str) -> ModelFile:
    if not isinstance(raw, dict):
        raise ManifestError("model file entry must be an object")
    file_path, size_bytes, sha256 = _parse_file_identity(raw)
    source_url, source_path = _parse_file_download_source(raw, source, acquisition)
    return ModelFile(
        path=file_path,
        size_bytes=size_bytes,
        sha256=sha256,
        source_url=source_url,
        source_path=source_path,
    )


def _parse_file_identity(raw: dict[str, object]) -> tuple[str, int, str]:
    try:
        file_path = raw["path"]
        size_bytes = _positive_int(raw["size_bytes"])
        sha256 = raw["sha256"]
    except (KeyError, TypeError) as error:
        raise ManifestError("model file evidence is incomplete") from error
    if not isinstance(file_path, str) or not file_path:
        raise ManifestError("model file path is invalid")
    pure_path = PurePosixPath(file_path)
    if (
        pure_path.is_absolute()
        or len(pure_path.parts) != 1
        or ".." in pure_path.parts
        or "." in pure_path.parts
    ):
        raise ManifestError("model file path escapes its allowlist")
    if not isinstance(sha256, str) or not _SHA256_RE.fullmatch(sha256):
        raise ManifestError("model file checksum is invalid")
    return file_path, size_bytes, sha256


def _parse_file_download_source(
    raw: dict[str, object],
    source: ModelSource,
    acquisition: str,
) -> tuple[str | None, str | None]:
    source_url = raw.get("source_url")
    source_path = raw.get("source_path")
    if acquisition == "download":
        if not isinstance(source_url, str) or not isinstance(source_path, str):
            raise ManifestError("download source URL is missing")
        pure_source_path = PurePosixPath(source_path)
        if (
            pure_source_path.is_absolute()
            or ".." in pure_source_path.parts
            or "." in pure_source_path.parts
        ):
            raise ManifestError("download source path is invalid")
        parsed = urlparse(source_url)
        expected_path = f"/{source.repository}/resolve/{source.revision}/{source_path}"
        if (
            parsed.scheme != "https"
            or parsed.hostname != _HUGGINGFACE_HOST
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path != expected_path
            or parsed.query
            or parsed.fragment
        ):
            raise ManifestError("download source URL is not pinned")
    elif source_url is not None or source_path is not None:
        raise ManifestError("reused model file cannot define a download URL")
    return source_url, source_path


def _validate_license_waiver(model: ModelEntry, policy: ManifestPolicy) -> None:
    if model.source.license_waiver is None:
        return
    observed_files = {
        model_file.path: (
            model_file.size_bytes,
            model_file.sha256,
            model_file.source_url,
            model_file.source_path,
        )
        for model_file in model.files
    }
    valid = (
        policy.usage_mode == "personal_noncommercial"
        and policy.redistribution is False
        and policy.certified_or_safety_critical is False
        and model.id == "piper-ru-irina-medium"
        and model.role == "tts"
        and model.acquisition == "download"
        and model.languages == ("ru",)
        and model.cache_path.parts[-2:] == ("cache", "piper")
        and model.source.repository == "rhasspy/piper-voices"
        and model.source.revision == _IRINA_REVISION
        and model.source.license == "MIT"
        and model.source.dataset_license == "Unknown"
        and model.source.license_waiver == _IRINA_WAIVER
        and observed_files == _IRINA_FILES
    )
    if not valid:
        raise ManifestError("license waiver does not match approved Irina assets")


def _target_path(model: ModelEntry, model_file: ModelFile) -> Path:
    target = model.cache_path.joinpath(*PurePosixPath(model_file.path).parts)
    try:
        target.relative_to(model.cache_path)
    except ValueError as error:
        raise ManifestError("model target escapes cache path") from error
    return target


def _hugging_face_blob_root(model: ModelEntry) -> Path | None:
    cache_path = model.cache_path
    if (
        model.acquisition != "reuse"
        or cache_path.name != model.source.revision
        or cache_path.parent.name != "snapshots"
    ):
        return None
    blob_root = cache_path.parent.parent / "blobs"
    try:
        metadata = blob_root.lstat()
    except OSError:
        return None
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        return None
    return blob_root.resolve(strict=True)


def _require_within(path: Path, root: Path) -> None:
    try:
        path.relative_to(root)
    except ValueError as error:
        raise OSError("runtime file is outside approved blob root") from error


def _positive_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ManifestError("model byte value must be a positive integer")
    return value


def _sha256_fd(descriptor: int) -> str:
    digest = hashlib.sha256()
    os.lseek(descriptor, 0, os.SEEK_SET)
    while block := os.read(descriptor, 1024 * 1024):
        digest.update(block)
    return digest.hexdigest()

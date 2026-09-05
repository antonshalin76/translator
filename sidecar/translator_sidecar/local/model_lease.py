"""Verified immutable model snapshots owned for the native runtime lifetime."""

from __future__ import annotations

import fcntl
import hashlib
import os
from contextlib import ExitStack
from dataclasses import dataclass
from typing import BinaryIO

from .model_manifest import ManifestError, ModelManifest, RuntimeFileOps

_SEALS = (
    fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL
)


@dataclass(frozen=True, slots=True)
class VerifiedModelSource:
    manifest: ModelManifest
    model_id: str

    def acquire(self) -> VerifiedModelLease:
        return VerifiedModelLease(self.manifest, self.model_id)


class VerifiedModelLease:
    """Copy admitted files, then hash sealed snapshots before handing them to loaders.

    The component owner serializes use and close, after its native calls drain.
    ExitStack owns both snapshots and independently opened consumer streams.
    """

    def __init__(
        self,
        manifest: ModelManifest,
        model_id: str,
        *,
        filesystem: RuntimeFileOps | None = None,
    ) -> None:
        self._resources = ExitStack()
        self._snapshots: dict[str, int] = {}
        try:
            model = manifest.models[model_id]
            if not model.files:
                raise ManifestError("model has no runtime files")
            for entry in model.files:
                with manifest.open_runtime_file(
                    model_id, entry.path, filesystem=filesystem
                ) as (source, _path):
                    descriptor = os.memfd_create(
                        "translator-model", os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING
                    )
                    self._resources.callback(os.close, descriptor)
                    snapshot = self._resources.enter_context(
                        os.fdopen(descriptor, "w+b", closefd=False)
                    )
                    size = 0
                    while chunk := os.read(source, 1024 * 1024):
                        size += len(chunk)
                        if size > entry.size_bytes:
                            raise ManifestError("runtime model file size mismatch")
                        snapshot.write(chunk)
                    if size != entry.size_bytes:
                        raise ManifestError("runtime model file size mismatch")
                    snapshot.flush()
                    fcntl.fcntl(snapshot.fileno(), fcntl.F_ADD_SEALS, _SEALS)
                    snapshot.seek(0)
                    if (
                        os.fstat(snapshot.fileno()).st_size != entry.size_bytes
                        or hashlib.file_digest(snapshot, "sha256").hexdigest()
                        != entry.sha256
                    ):
                        raise ManifestError("runtime model snapshot integrity mismatch")
                    self._snapshots[entry.path] = snapshot.fileno()
        except BaseException:
            self.close()
            raise

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._snapshots)

    @property
    def identifier(self) -> str:
        if not self._snapshots:
            raise ManifestError("model lease is closed")
        return self.path(next(iter(self._snapshots)))

    def path(self, name: str) -> str:
        try:
            descriptor = self._snapshots[name]
        except KeyError:
            raise ManifestError(
                "model lease is closed or file is not declared"
            ) from None
        return f"/proc/self/fd/{descriptor}"

    def files(self) -> dict[str, BinaryIO]:
        if not self._snapshots:
            raise ManifestError("model lease is closed")
        # Opening procfs paths creates independent offsets; dup() would share them.
        return {
            name: self._resources.enter_context(open(self.path(name), "rb"))
            for name in self.names
        }

    def read_bytes(self, name: str) -> bytes:
        with open(self.path(name), "rb") as reader:
            return reader.read()

    def close(self) -> None:
        self._snapshots.clear()
        self._resources.close()

    def __enter__(self) -> VerifiedModelLease:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

"""Private, descriptor-bound WAV custody for verified product PCM evidence."""

from __future__ import annotations

import hashlib
import io
import os
import re
import stat
import wave
from pathlib import Path
from typing import Self

FRAME_BYTES = 960
MAX_PCM_BYTES = 1500 * FRAME_BYTES
MAX_WAV_BYTES = MAX_PCM_BYTES + 44
_BASENAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,123}\.wav\Z")
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC


def _private_directory(path: Path, forbidden_roots: tuple[Path, ...]) -> int:
    if (
        not path.is_absolute()
        or ".." in path.parts
        or any(path.is_relative_to(root.resolve()) for root in forbidden_roots)
    ):
        raise ValueError("PCM artifacts require a private directory outside Git")
    descriptor = os.open(path.anchor, _DIRECTORY_FLAGS)
    try:
        for component in path.parts[1:]:
            try:
                os.stat(".git", dir_fd=descriptor, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise ValueError("PCM artifacts cannot be stored in Git")
            child = os.open(component, _DIRECTORY_FLAGS, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        try:
            os.stat(".git", dir_fd=descriptor, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise ValueError("PCM artifacts cannot be stored in Git")
        identity = os.fstat(descriptor)
        if identity.st_uid != os.getuid() or stat.S_IMODE(identity.st_mode) != 0o700:
            raise ValueError("PCM artifact directory is not private and owned")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


class PcmArtifactStore:
    def __init__(self, directory: Path, *, forbidden_roots: tuple[Path, ...] = ()):
        self.directory = directory
        self.forbidden_roots = (
            Path(__file__).resolve().parents[1],
            *forbidden_roots,
        )
        self._descriptor = _private_directory(directory, self.forbidden_roots)
        self._identity = os.fstat(self._descriptor)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def close(self) -> None:
        if self._descriptor is not None:
            os.close(self._descriptor)
            self._descriptor = None

    def _verify_directory(self) -> None:
        if self._descriptor is None:
            raise ValueError("PCM artifact store is closed")
        descriptor = _private_directory(self.directory, self.forbidden_roots)
        try:
            current = os.fstat(descriptor)
            if (current.st_dev, current.st_ino) != (
                self._identity.st_dev,
                self._identity.st_ino,
            ):
                raise ValueError("PCM artifact directory identity changed")
        finally:
            os.close(descriptor)

    @staticmethod
    def _filename(filename: object) -> str:
        if not isinstance(filename, str) or _BASENAME.fullmatch(filename) is None:
            raise ValueError("PCM artifact filename is invalid")
        return filename

    def write_pcm(self, pcm: bytes, expected_hash: str, filename: str) -> dict:
        filename = self._filename(filename)
        if (
            not isinstance(pcm, bytes)
            or not 0 < len(pcm) <= MAX_PCM_BYTES
            or len(pcm) % FRAME_BYTES
            or not isinstance(expected_hash, str)
            or _HASH.fullmatch(expected_hash) is None
            or hashlib.sha256(pcm).hexdigest() != expected_hash
        ):
            raise ValueError("PCM bytes or identity differ")
        self._verify_directory()
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(24_000)
            audio.writeframes(pcm)
        wav_bytes = buffer.getvalue()
        descriptor = None
        created = False
        try:
            descriptor = os.open(
                filename,
                os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                0o600,
                dir_fd=self._descriptor,
            )
            created = True
            file_identity = os.fstat(descriptor)
            stream = os.fdopen(descriptor, "w+b")
            descriptor = None
            with stream:
                stream.write(wav_bytes)
                stream.flush()
                os.fsync(stream.fileno())
                stream.seek(0)
                if stream.read(len(wav_bytes) + 1) != wav_bytes:
                    raise ValueError("PCM artifact readback differs")
                self._verify_directory()
            os.fsync(self._descriptor)
            self._verify_directory()
        except BaseException:
            if created:
                try:
                    current = os.stat(
                        filename, dir_fd=self._descriptor, follow_symlinks=False
                    )
                except FileNotFoundError:
                    pass
                else:
                    if (current.st_dev, current.st_ino) == (
                        file_identity.st_dev,
                        file_identity.st_ino,
                    ):
                        os.unlink(filename, dir_fd=self._descriptor)
            raise
        finally:
            if descriptor is not None:
                os.close(descriptor)
        return {
            "filename": filename,
            "wav_sha256": hashlib.sha256(wav_bytes).hexdigest(),
            "pcm_sha256": expected_hash,
            "wav_bytes": len(wav_bytes),
            "pcm_bytes": len(pcm),
            "sample_rate_hz": 24_000,
            "channels": 1,
            "sample_format": "s16le",
            "sample_count": len(pcm) // 2,
            "frame_count": len(pcm) // FRAME_BYTES,
            "directory_device": self._identity.st_dev,
            "directory_inode": self._identity.st_ino,
            "file_device": file_identity.st_dev,
            "file_inode": file_identity.st_ino,
        }

    def read_verified(self, artifact: dict) -> bytes:
        self._verify_directory()
        if not isinstance(artifact, dict):
            raise TypeError("PCM artifact metadata is invalid")
        filename = self._filename(artifact.get("filename"))
        counts = (
            "pcm_bytes",
            "wav_bytes",
            "sample_count",
            "frame_count",
            "sample_rate_hz",
            "channels",
            "directory_device",
            "directory_inode",
            "file_device",
            "file_inode",
        )
        if (
            any(type(artifact.get(key)) is not int for key in counts)
            or not 0 < artifact["pcm_bytes"] <= MAX_PCM_BYTES
            or artifact["pcm_bytes"] % FRAME_BYTES
            or artifact["wav_bytes"] != artifact["pcm_bytes"] + 44
            or artifact["sample_count"] != artifact["pcm_bytes"] // 2
            or artifact["frame_count"] != artifact["pcm_bytes"] // FRAME_BYTES
            or artifact.get("sample_rate_hz") != 24_000
            or artifact.get("channels") != 1
            or artifact.get("sample_format") != "s16le"
            or artifact.get("directory_device") != self._identity.st_dev
            or artifact.get("directory_inode") != self._identity.st_ino
            or any(
                not isinstance(artifact.get(key), str)
                or _HASH.fullmatch(artifact[key]) is None
                for key in ("wav_sha256", "pcm_sha256")
            )
        ):
            raise ValueError("PCM artifact metadata differs")
        descriptor = os.open(
            filename,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK,
            dir_fd=self._descriptor,
        )
        try:
            source = os.fdopen(descriptor, "rb")
        except BaseException:
            os.close(descriptor)
            raise
        with source:
            identity = os.fstat(source.fileno())
            if (
                not stat.S_ISREG(identity.st_mode)
                or identity.st_uid != os.getuid()
                or stat.S_IMODE(identity.st_mode) != 0o600
                or identity.st_size != artifact["wav_bytes"]
                or identity.st_nlink != 1
                or (identity.st_dev, identity.st_ino)
                != (artifact["file_device"], artifact["file_inode"])
            ):
                raise ValueError("PCM artifact file custody differs")
            wav_bytes = source.read(MAX_WAV_BYTES + 1)
        self._verify_directory()
        if (
            len(wav_bytes) != artifact["wav_bytes"]
            or hashlib.sha256(wav_bytes).hexdigest() != artifact["wav_sha256"]
        ):
            raise ValueError("PCM artifact WAV identity differs")
        with wave.open(io.BytesIO(wav_bytes), "rb") as audio:
            if (
                audio.getframerate() != 24_000
                or audio.getnchannels() != 1
                or audio.getsampwidth() != 2
                or audio.getcomptype() != "NONE"
                or audio.getnframes() != artifact["sample_count"]
            ):
                raise ValueError("PCM artifact WAV format differs")
            pcm = audio.readframes(artifact["sample_count"] + 1)
        if (
            len(pcm) != artifact["pcm_bytes"]
            or hashlib.sha256(pcm).hexdigest() != artifact["pcm_sha256"]
        ):
            raise ValueError("PCM artifact payload identity differs")
        return wav_bytes

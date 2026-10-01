"""Filesystem custody tests for captured product PCM evidence."""

from __future__ import annotations

import hashlib
import importlib
import io
import os
import stat
import wave
from pathlib import Path

import pytest

PCM = b"\x01\x02" * 960
PCM_SHA = hashlib.sha256(PCM).hexdigest()


@pytest.fixture
def pcm_module():
    return importlib.import_module("translator_product_pcm")


@pytest.fixture
def private(tmp_path: Path) -> Path:
    directory = tmp_path / "private"
    directory.mkdir(mode=0o700)
    return directory


def test_pcm_artifact_exact_private_durable_and_readable(
    pcm_module, private, monkeypatch
):
    syncs = []
    original_sync = os.fsync

    def sync(fd):
        syncs.append(stat.S_ISDIR(os.fstat(fd).st_mode))
        original_sync(fd)

    monkeypatch.setattr(os, "fsync", sync)
    with pcm_module.PcmArtifactStore(private) as store:
        artifact = store.write_pcm(PCM, PCM_SHA, "case.wav")
        wav_bytes = store.read_verified(artifact)
    path = private / artifact["filename"]
    assert artifact["filename"] == "case.wav"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert wav_bytes == path.read_bytes()
    assert artifact["wav_sha256"] == hashlib.sha256(wav_bytes).hexdigest()
    assert artifact["pcm_sha256"] == PCM_SHA
    assert artifact["sample_rate_hz"] == 24_000
    assert artifact["channels"] == 1
    assert artifact["sample_format"] == "s16le"
    assert artifact["sample_count"] == 960
    assert artifact["frame_count"] == 2
    assert artifact["pcm_bytes"] == len(PCM)
    assert artifact["wav_bytes"] == len(wav_bytes)
    assert artifact["directory_device"] == private.stat().st_dev
    assert artifact["directory_inode"] == private.stat().st_ino
    assert syncs == [False, True]
    with wave.open(io.BytesIO(wav_bytes), "rb") as audio:
        assert (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) == (
            24_000,
            1,
            2,
        )
        assert audio.getnframes() == 960
        assert audio.readframes(961) == PCM


@pytest.mark.parametrize("kind", ["relative", "public", "repo", "symlink"])
def test_pcm_store_rejects_nonprivate_or_git_or_linked_directory(
    pcm_module, private, tmp_path, kind
):
    if kind == "relative":
        directory = Path("relative")
    elif kind == "public":
        private.chmod(0o755)
        directory = private
    elif kind == "repo":
        (tmp_path / ".git").mkdir()
        directory = private
    else:
        directory = tmp_path / "link"
        directory.symlink_to(private, target_is_directory=True)
    with pytest.raises((ValueError, OSError)):
        pcm_module.PcmArtifactStore(directory)
    assert list(private.iterdir()) == []


def test_pcm_store_rejects_explicit_forbidden_root(pcm_module, private, tmp_path):
    with pytest.raises(ValueError):
        pcm_module.PcmArtifactStore(private, forbidden_roots=(tmp_path,))


@pytest.mark.parametrize(
    "filename", ["../escape.wav", "/escape.wav", "nested/a.wav", "not-a-wav"]
)
def test_pcm_store_rejects_escaped_or_invalid_basename(pcm_module, private, filename):
    with pcm_module.PcmArtifactStore(private) as store, pytest.raises(ValueError):
        store.write_pcm(PCM, PCM_SHA, filename)
    assert list(private.iterdir()) == []


@pytest.mark.parametrize("collision", ["file", "symlink"])
def test_pcm_store_never_overwrites_collision(pcm_module, private, tmp_path, collision):
    target = tmp_path / "existing"
    target.write_bytes(b"preserved")
    path = private / "case.wav"
    if collision == "file":
        path.write_bytes(b"preserved")
    else:
        path.symlink_to(target)
    with (
        pcm_module.PcmArtifactStore(private) as store,
        pytest.raises(FileExistsError),
    ):
        store.write_pcm(PCM, PCM_SHA, "case.wav")
    assert path.read_bytes() == b"preserved"
    assert target.read_bytes() == b"preserved"


@pytest.mark.parametrize(
    "pcm",
    [bytearray(PCM), b"", PCM[:-1], PCM[:-2]],
    ids=["mutable", "empty", "odd_bytes", "partial_frame"],
)
def test_pcm_store_rejects_mutable_or_misaligned_pcm(pcm_module, private, pcm):
    with pcm_module.PcmArtifactStore(private) as store, pytest.raises(ValueError):
        store.write_pcm(pcm, hashlib.sha256(pcm).hexdigest(), "case.wav")
    assert list(private.iterdir()) == []


def test_pcm_store_rejects_changed_expected_pcm_hash(pcm_module, private):
    with pcm_module.PcmArtifactStore(private) as store, pytest.raises(ValueError):
        store.write_pcm(PCM, "0" * 64, "case.wav")
    assert list(private.iterdir()) == []


@pytest.mark.parametrize("stage", ["file_sync", "directory_sync", "readback"])
def test_pcm_store_removes_owned_partial_artifact_on_failure(
    pcm_module, private, monkeypatch, stage
):
    original_sync, original_fdopen = os.fsync, os.fdopen

    def sync(fd):
        is_directory = stat.S_ISDIR(os.fstat(fd).st_mode)
        if is_directory == (stage == "directory_sync") and stage != "readback":
            raise OSError("synthetic durability failure")
        original_sync(fd)

    class CorruptReadback:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def read(self, count):
            actual = self.stream.read(count)
            return actual[:-1] + bytes([actual[-1] ^ 1])

    if stage == "readback":
        monkeypatch.setattr(
            os,
            "fdopen",
            lambda *args, **kwargs: CorruptReadback(original_fdopen(*args, **kwargs)),
        )
    else:
        monkeypatch.setattr(os, "fsync", sync)
    with (
        pcm_module.PcmArtifactStore(private) as store,
        pytest.raises((ValueError, OSError)),
    ):
        store.write_pcm(PCM, PCM_SHA, "case.wav")
    assert list(private.iterdir()) == []


@pytest.mark.parametrize("replacement", ["directory", "symlink"])
def test_pcm_store_directory_replacement_during_write_fails_closed(
    pcm_module, private, tmp_path, monkeypatch, replacement
):
    original_sync = os.fsync
    old = tmp_path / "original-directory"
    other = tmp_path / "other"
    other.mkdir(mode=0o700)
    swapped = False

    def sync(fd):
        nonlocal swapped
        if not stat.S_ISDIR(os.fstat(fd).st_mode) and not swapped:
            swapped = True
            private.rename(old)
            if replacement == "directory":
                private.mkdir(mode=0o700)
            else:
                private.symlink_to(other, target_is_directory=True)
        original_sync(fd)

    with pcm_module.PcmArtifactStore(private) as store:
        monkeypatch.setattr(os, "fsync", sync)
        with pytest.raises((ValueError, OSError)):
            store.write_pcm(PCM, PCM_SHA, "case.wav")
    assert list(old.iterdir()) == []
    assert list(private.iterdir()) == []
    assert list(other.iterdir()) == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("filename", "../escape.wav"),
        ("wav_sha256", "0" * 64),
        ("pcm_sha256", "0" * 64),
        ("sample_count", 961),
        ("frame_count", 3),
        ("sample_rate_hz", 16_000),
        ("channels", 2),
        ("sample_format", "float32"),
        ("directory_device", -1),
        ("directory_inode", -1),
        ("file_device", -1),
        ("file_inode", -1),
        ("wav_bytes", 2_000_000),
    ],
)
def test_pcm_reader_rejects_unbound_metadata(pcm_module, private, field, value):
    with pcm_module.PcmArtifactStore(private) as store:
        artifact = store.write_pcm(PCM, PCM_SHA, "case.wav")
        with pytest.raises((ValueError, OSError)):
            store.read_verified({**artifact, field: value})


@pytest.mark.parametrize(
    "mutation",
    ["bytes", "symlink", "oversize", "directory", "fifo", "public", "replacement"],
)
def test_pcm_reader_rechecks_file_and_directory_at_send_time(
    pcm_module, private, tmp_path, mutation
):
    with pcm_module.PcmArtifactStore(private) as store:
        artifact = store.write_pcm(PCM, PCM_SHA, "case.wav")
        path = private / "case.wav"
        if mutation == "bytes":
            path.write_bytes(path.read_bytes()[:-1] + b"\xff")
        elif mutation == "symlink":
            target = tmp_path / "elsewhere.wav"
            path.rename(target)
            path.symlink_to(target)
        elif mutation == "oversize":
            with path.open("ab") as stream:
                stream.truncate(2_000_000)
        elif mutation == "fifo":
            path.rename(tmp_path / "original.wav")
            os.mkfifo(path, mode=0o600)
        elif mutation == "public":
            path.chmod(0o644)
        elif mutation == "replacement":
            content = path.read_bytes()
            path.rename(tmp_path / "original.wav")
            path.write_bytes(content)
            path.chmod(0o600)
        else:
            private.rename(tmp_path / "original")
            private.mkdir(mode=0o700)
        with pytest.raises((ValueError, OSError)):
            store.read_verified(artifact)

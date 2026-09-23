"""CUDA runtime compatibility bootstrap for local CTranslate2 adapters."""

from __future__ import annotations

import ctypes
import os
import stat
import threading
from collections import deque
from collections.abc import Iterable
from pathlib import Path

_DEFAULT_LIBRARY_DIRS = (Path("/usr/local/lib/ollama/cuda_v12"),)
_PRELOAD_NAMES = (
    "libcudart.so.12",
    "libcublasLt.so.12",
    "libcublas.so.12",
    "libcudnn.so.9",
)
_LOADED_LIBRARIES: dict[str, tuple[ctypes.CDLL, int, tuple[int, int]]] = {}
_CONFIGURE_LOCK = threading.Lock()
_ROOT_UID = 0
_MAX_LIBRARY_TREE_ENTRIES = 4096
_MAX_SYMLINKS = 40
_UNSAFE_MODE = stat.S_IWGRP | stat.S_IWOTH | stat.S_ISUID | stat.S_ISGID


def configure_cuda_runtime(
    *,
    extra_library_dirs: Iterable[Path] = (),
    preload: bool = True,
) -> tuple[Path, ...]:
    """Expose local CUDA 12/cuDNN 9 libraries to CTranslate2 if present."""

    with _CONFIGURE_LOCK:
        previous_loader_path = os.environ.get("LD_LIBRARY_PATH")
        opened_libraries: list[tuple[str, int, tuple[int, int]]] = []
        try:
            library_dirs = _admitted_library_dirs(
                explicit=(
                    *_env_library_dirs("TRANSLATOR_CUDA_LIBRARY_PATH"),
                    *extra_library_dirs,
                ),
                defaults=_DEFAULT_LIBRARY_DIRS,
            )
            if preload:
                opened_libraries = _open_preload_libraries(library_dirs)
            _replace_ld_library_path(library_dirs)
            if not library_dirs:
                return ()
            if preload:
                _preload_libraries(opened_libraries)
            return library_dirs
        except BaseException:
            if previous_loader_path is None:
                os.environ.pop("LD_LIBRARY_PATH", None)
            else:
                os.environ["LD_LIBRARY_PATH"] = previous_loader_path
            raise
        finally:
            _close_uncommitted_libraries(opened_libraries)


def _env_library_dirs(name: str) -> tuple[Path, ...]:
    value = os.environ.get(name, "")
    if not value:
        return ()
    items = value.split(os.pathsep)
    if any(not item for item in items):
        _unsafe()
    return tuple(Path(item) for item in items)


def _admitted_library_dirs(
    *, explicit: Iterable[Path], defaults: Iterable[Path]
) -> tuple[Path, ...]:
    seen: set[Path] = set()
    result: list[Path] = []
    for path in explicit:
        resolved = _admit_library_directory(path)
        if resolved not in seen:
            seen.add(resolved)
            result.append(resolved)
    for path in defaults:
        try:
            os.lstat(path)
        except FileNotFoundError:
            continue
        except OSError:
            _unsafe()
        resolved = _admit_library_directory(path)
        if resolved not in seen:
            seen.add(resolved)
            result.append(resolved)
    return tuple(result)


def _unsafe() -> None:
    raise RuntimeError("unsafe_cuda_runtime") from None


def _trusted_owner(uid: int) -> bool:
    return uid in {_ROOT_UID, os.geteuid()}


def _safe_directory(metadata: os.stat_result, *, traversal: bool) -> None:
    if not stat.S_ISDIR(metadata.st_mode) or not _trusted_owner(metadata.st_uid):
        _unsafe()
    writable = bool(metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH))
    root_sticky = (
        traversal
        and metadata.st_uid == _ROOT_UID
        and bool(metadata.st_mode & stat.S_ISVTX)
    )
    if (writable and not root_sticky) or metadata.st_mode & (
        stat.S_ISUID | stat.S_ISGID
    ):
        _unsafe()


def _safe_file(metadata: os.stat_result) -> None:
    if (
        not stat.S_ISREG(metadata.st_mode)
        or not _trusted_owner(metadata.st_uid)
        or metadata.st_mode & _UNSAFE_MODE
    ):
        _unsafe()


def _secure_resolve(path: Path, *, expected: str) -> Path:
    if path.anchor != os.sep or ".." in path.parts:
        _unsafe()
    pending = deque(path.parts)
    resolved = Path(os.sep)
    symlinks = 0
    while pending:
        component = pending.popleft()
        if component == os.sep:
            resolved = Path(os.sep)
            try:
                root_metadata = os.lstat(resolved)
            except OSError:
                _unsafe()
            _safe_directory(root_metadata, traversal=True)
            continue
        if component in {"", "."}:
            continue
        if component == "..":
            resolved = resolved.parent
            continue
        candidate = resolved / component
        try:
            metadata = os.lstat(candidate)
        except OSError:
            _unsafe()
        if stat.S_ISLNK(metadata.st_mode):
            if not _trusted_owner(metadata.st_uid):
                _unsafe()
            symlinks += 1
            if symlinks > _MAX_SYMLINKS:
                _unsafe()
            try:
                target = Path(os.readlink(candidate))
            except OSError:
                _unsafe()
            if target.anchor not in {"", os.sep}:
                _unsafe()
            pending.extendleft(reversed(target.parts))
            continue
        if pending:
            _safe_directory(metadata, traversal=True)
        resolved = candidate

    try:
        metadata = os.lstat(resolved)
        canonical = path.resolve(strict=True)
    except (OSError, RuntimeError):
        _unsafe()
    if canonical != resolved:
        _unsafe()
    if expected == "file":
        _safe_file(metadata)
    elif expected == "directory":
        _safe_directory(metadata, traversal=False)
    else:
        _unsafe()
    return resolved


def _directory_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_nlink,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _validate_library_tree(root: Path) -> None:
    pending = [root]
    visited: set[tuple[int, int]] = set()
    entries = 0
    while pending:
        directory = pending.pop()
        try:
            before = os.stat(directory, follow_symlinks=False)
        except OSError:
            _unsafe()
        _safe_directory(before, traversal=False)
        identity = (before.st_dev, before.st_ino)
        if identity in visited:
            continue
        visited.add(identity)
        try:
            with os.scandir(directory) as children:
                for entry in children:
                    entries += 1
                    if entries > _MAX_LIBRARY_TREE_ENTRIES:
                        _unsafe()
                    child = Path(entry.path)
                    try:
                        lexical = os.lstat(child)
                    except OSError:
                        _unsafe()
                    if stat.S_ISLNK(lexical.st_mode):
                        if not _trusted_owner(lexical.st_uid):
                            _unsafe()
                        try:
                            target = child.resolve(strict=True)
                        except (OSError, RuntimeError):
                            _unsafe()
                        try:
                            target_metadata = os.stat(target, follow_symlinks=False)
                        except OSError:
                            _unsafe()
                        if stat.S_ISDIR(target_metadata.st_mode):
                            target = _secure_resolve(child, expected="directory")
                            pending.append(target)
                        else:
                            _secure_resolve(child, expected="file")
                    elif stat.S_ISDIR(lexical.st_mode):
                        _safe_directory(lexical, traversal=False)
                        pending.append(child)
                    elif stat.S_ISREG(lexical.st_mode):
                        _safe_file(lexical)
                    else:
                        _unsafe()
        except OSError:
            _unsafe()
        try:
            after = os.stat(directory, follow_symlinks=False)
        except OSError:
            _unsafe()
        if _directory_identity(before) != _directory_identity(after):
            _unsafe()


def _admit_library_directory(path: Path) -> Path:
    if not path.is_absolute():
        _unsafe()
    resolved = _secure_resolve(path, expected="directory")
    _validate_library_tree(resolved)
    return resolved


def _replace_ld_library_path(library_dirs: tuple[Path, ...]) -> None:
    if library_dirs:
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(map(str, library_dirs))
    else:
        os.environ.pop("LD_LIBRARY_PATH", None)


def _file_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _open_preload_libraries(
    library_dirs: tuple[Path, ...],
) -> list[tuple[str, int, tuple[int, int]]]:
    opened: list[tuple[str, int, tuple[int, int]]] = []
    try:
        for name in _PRELOAD_NAMES:
            for directory in library_dirs:
                candidate = directory / name
                try:
                    os.lstat(candidate)
                except FileNotFoundError:
                    continue
                except OSError:
                    _unsafe()
                resolved = _secure_resolve(candidate, expected="file")
                try:
                    descriptor = os.open(
                        resolved,
                        os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                    )
                except OSError:
                    _unsafe()
                try:
                    held = os.fstat(descriptor)
                    visible = os.stat(resolved, follow_symlinks=False)
                    _safe_file(held)
                    if _file_identity(held) != _file_identity(visible):
                        _unsafe()
                except BaseException:
                    os.close(descriptor)
                    raise
                opened.append((name, descriptor, (held.st_dev, held.st_ino)))
                break
    except BaseException:
        for _, descriptor, _ in opened:
            os.close(descriptor)
        raise
    return opened


def _preload_libraries(
    opened_libraries: list[tuple[str, int, tuple[int, int]]],
) -> None:
    for name, descriptor, identity in opened_libraries:
        loaded = _LOADED_LIBRARIES.get(name)
        if loaded is not None:
            if loaded[2] != identity:
                _unsafe()
            continue
        try:
            library = ctypes.CDLL(
                f"/proc/self/fd/{descriptor}", mode=ctypes.RTLD_GLOBAL
            )
        except BaseException as error:
            if isinstance(error, Exception):
                raise RuntimeError("cuda_runtime_unavailable") from None
            raise
        held = os.fstat(descriptor)
        if (held.st_dev, held.st_ino) != identity:
            _unsafe()
        _LOADED_LIBRARIES[name] = (library, descriptor, identity)


def _close_uncommitted_libraries(
    opened_libraries: list[tuple[str, int, tuple[int, int]]],
) -> None:
    try:
        for name, descriptor, identity in opened_libraries:
            loaded = _LOADED_LIBRARIES.get(name)
            if loaded is not None and loaded[1:] == (descriptor, identity):
                continue
            try:
                os.close(descriptor)
            except OSError:
                pass
    finally:
        opened_libraries.clear()

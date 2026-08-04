"""CUDA runtime compatibility bootstrap for local CTranslate2 adapters."""

from __future__ import annotations

import ctypes
import os
from collections.abc import Iterable
from pathlib import Path


_DEFAULT_LIBRARY_DIRS = (
    Path("/usr/local/lib/ollama/cuda_v12"),
    Path(
        "/home/anton/Source/uncle-freud-bot/.venv/lib/python3.12/"
        "site-packages/nvidia/cudnn/lib"
    ),
)
_PRELOAD_NAMES = (
    "libcudart.so.12",
    "libcublas.so.12",
    "libcublasLt.so.12",
    "libcudnn.so.9",
)
_LOADED_LIBRARIES: list[ctypes.CDLL] = []


def configure_cuda_runtime(
    *,
    extra_library_dirs: Iterable[Path] = (),
    preload: bool = True,
) -> tuple[Path, ...]:
    """Expose local CUDA 12/cuDNN 9 libraries to CTranslate2 if present."""

    library_dirs = _existing_unique_dirs(
        (
            *_env_library_dirs("TRANSLATOR_CUDA_LIBRARY_PATH"),
            *extra_library_dirs,
            *_DEFAULT_LIBRARY_DIRS,
        )
    )
    if not library_dirs:
        return ()
    _prepend_ld_library_path(library_dirs)
    if preload:
        _preload_libraries(library_dirs)
    return library_dirs


def _env_library_dirs(name: str) -> tuple[Path, ...]:
    value = os.environ.get(name, "")
    if not value:
        return ()
    return tuple(Path(item) for item in value.split(os.pathsep) if item)


def _existing_unique_dirs(paths: Iterable[Path]) -> tuple[Path, ...]:
    seen: set[Path] = set()
    result = []
    for path in paths:
        expanded = path.expanduser()
        if not expanded.is_dir():
            continue
        resolved = expanded.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        result.append(resolved)
    return tuple(result)


def _prepend_ld_library_path(library_dirs: tuple[Path, ...]) -> None:
    seen: set[str] = set()
    paths = []
    for path in (
        *(str(path) for path in library_dirs),
        *os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep),
    ):
        if not path or path in seen:
            continue
        seen.add(path)
        paths.append(path)
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(paths)


def _preload_libraries(library_dirs: tuple[Path, ...]) -> None:
    for name in _PRELOAD_NAMES:
        for directory in library_dirs:
            library_path = directory / name
            if not library_path.is_file():
                continue
            _LOADED_LIBRARIES.append(
                ctypes.CDLL(str(library_path), mode=ctypes.RTLD_GLOBAL)
            )
            break

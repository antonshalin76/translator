from __future__ import annotations

import os

from translator_sidecar.local import cuda_runtime


def test_configure_cuda_runtime_prepends_unique_existing_dirs(
    tmp_path, monkeypatch
) -> None:
    cuda_dir = tmp_path / "cuda12"
    cudnn_dir = tmp_path / "cudnn9"
    missing_dir = tmp_path / "missing"
    cuda_dir.mkdir()
    cudnn_dir.mkdir()
    monkeypatch.setattr(cuda_runtime, "_DEFAULT_LIBRARY_DIRS", ())
    monkeypatch.setenv(
        "TRANSLATOR_CUDA_LIBRARY_PATH",
        os.pathsep.join((str(cuda_dir), str(missing_dir))),
    )
    monkeypatch.setenv("LD_LIBRARY_PATH", os.pathsep.join((str(cudnn_dir), "/usr/lib")))

    configured = cuda_runtime.configure_cuda_runtime(
        extra_library_dirs=(cudnn_dir, cuda_dir),
        preload=False,
    )

    assert configured == (cuda_dir.resolve(), cudnn_dir.resolve())
    assert os.environ["LD_LIBRARY_PATH"].split(os.pathsep) == [
        str(cuda_dir.resolve()),
        str(cudnn_dir.resolve()),
        "/usr/lib",
    ]

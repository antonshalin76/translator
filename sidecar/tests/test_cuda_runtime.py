from __future__ import annotations

import os
import threading
from pathlib import Path

import pytest

from translator_sidecar.local import cuda_runtime


def _isolate_cuda_environment(monkeypatch) -> None:
    monkeypatch.setattr(cuda_runtime, "_DEFAULT_LIBRARY_DIRS", ())
    monkeypatch.setattr(cuda_runtime, "_PRELOAD_NAMES", ("libcudart.so.12",))
    monkeypatch.setattr(cuda_runtime, "_LOADED_LIBRARIES", {})
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)


def _allow_namespaced_test_root(monkeypatch) -> None:
    root_uid = Path("/").stat().st_uid
    if root_uid != 0:
        monkeypatch.setattr(cuda_runtime, "_ROOT_UID", root_uid, raising=False)


def _close_retained_library_descriptors() -> None:
    for _library, descriptor, _identity in cuda_runtime._LOADED_LIBRARIES.values():
        try:
            os.close(descriptor)
        except OSError:
            pass
    cuda_runtime._LOADED_LIBRARIES.clear()


def test_configure_cuda_runtime_prepends_unique_existing_dirs(
    tmp_path, monkeypatch
) -> None:
    cuda_dir = tmp_path / "cuda12"
    cudnn_dir = tmp_path / "cudnn9"
    cuda_dir.mkdir()
    cudnn_dir.mkdir()
    cuda_dir.chmod(0o700)
    cudnn_dir.chmod(0o700)
    _allow_namespaced_test_root(monkeypatch)
    monkeypatch.setattr(cuda_runtime, "_DEFAULT_LIBRARY_DIRS", ())
    monkeypatch.setenv("TRANSLATOR_CUDA_LIBRARY_PATH", str(cuda_dir))
    monkeypatch.setenv("LD_LIBRARY_PATH", str(tmp_path / "attacker-controlled"))

    configured = cuda_runtime.configure_cuda_runtime(
        extra_library_dirs=(cudnn_dir, cuda_dir),
        preload=False,
    )

    assert configured == (cuda_dir.resolve(), cudnn_dir.resolve())
    assert os.environ["LD_LIBRARY_PATH"].split(os.pathsep) == [
        str(cuda_dir.resolve()),
        str(cudnn_dir.resolve()),
    ]


def test_operator_cuda_path_must_be_absolute_and_existing(
    tmp_path: Path, monkeypatch
) -> None:
    configured_dir = tmp_path / "cudnn"
    configured_dir.mkdir()
    _allow_namespaced_test_root(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(
        "TRANSLATOR_CUDA_LIBRARY_PATH",
        os.pathsep.join(("cudnn", str(configured_dir))),
    )
    monkeypatch.setattr(cuda_runtime, "_DEFAULT_LIBRARY_DIRS", ())

    with pytest.raises(RuntimeError) as error:
        cuda_runtime.configure_cuda_runtime(preload=False)

    assert str(error.value) == "unsafe_cuda_runtime"


def test_public_cuda_and_service_sources_contain_no_private_checkout_paths() -> None:
    repository = Path(__file__).resolve().parents[2]
    public_runtime_files = (
        repository / "sidecar/translator_sidecar/local/cuda_runtime.py",
        repository / "crates/translator-daemon/src/process_sidecar.rs",
        repository / "systemd/translator.service",
    )
    private_markers = (
        "/" + "home" + "/",
        "/" + "Source" + "/",
        "uncle" + "-freud",
    )

    for path in public_runtime_files:
        content = path.read_text(encoding="utf-8")
        assert all(marker not in content for marker in private_markers), path


def test_cuda_runtime_rejects_world_writable_operator_directory_without_side_effects(
    tmp_path: Path, monkeypatch
) -> None:
    _isolate_cuda_environment(monkeypatch)
    _allow_namespaced_test_root(monkeypatch)
    cuda_dir = tmp_path / "hostile-cuda-directory-marker"
    cuda_dir.mkdir(mode=0o700)
    cuda_dir.chmod(0o777)
    monkeypatch.setenv("TRANSLATOR_CUDA_LIBRARY_PATH", str(cuda_dir))
    cdll_calls: list[str] = []
    monkeypatch.setattr(
        cuda_runtime.ctypes,
        "CDLL",
        lambda path, **_kwargs: cdll_calls.append(str(path)),
    )

    with pytest.raises(RuntimeError) as error:
        cuda_runtime.configure_cuda_runtime()

    assert str(error.value) == "unsafe_cuda_runtime"
    assert cuda_dir.name not in str(error.value)
    assert "LD_LIBRARY_PATH" not in os.environ
    assert cdll_calls == []


def test_cuda_runtime_rejects_writable_library_before_cdll(
    tmp_path: Path, monkeypatch
) -> None:
    _isolate_cuda_environment(monkeypatch)
    _allow_namespaced_test_root(monkeypatch)
    cuda_dir = tmp_path / "cuda"
    cuda_dir.mkdir(mode=0o700)
    library = cuda_dir / "libcudart.so.12"
    library.write_bytes(b"synthetic library")
    library.chmod(0o666)
    monkeypatch.setenv("TRANSLATOR_CUDA_LIBRARY_PATH", str(cuda_dir))
    cdll_calls: list[str] = []
    monkeypatch.setattr(
        cuda_runtime.ctypes,
        "CDLL",
        lambda path, **_kwargs: cdll_calls.append(str(path)),
    )

    with pytest.raises(RuntimeError) as error:
        cuda_runtime.configure_cuda_runtime()

    assert str(error.value) == "unsafe_cuda_runtime"
    assert cdll_calls == []


def test_cuda_runtime_rejects_unsafe_soname_symlink_target_before_cdll(
    tmp_path: Path, monkeypatch
) -> None:
    _isolate_cuda_environment(monkeypatch)
    _allow_namespaced_test_root(monkeypatch)
    cuda_dir = tmp_path / "cuda"
    cuda_dir.mkdir(mode=0o700)
    unsafe_parent = tmp_path / "unsafe-target-parent-marker"
    unsafe_parent.mkdir(mode=0o700)
    unsafe_parent.chmod(0o777)
    target = unsafe_parent / "libcudart.so.12.0"
    target.touch(mode=0o600)
    (cuda_dir / "libcudart.so.12").symlink_to(target)
    monkeypatch.setenv("TRANSLATOR_CUDA_LIBRARY_PATH", str(cuda_dir))
    cdll_calls: list[str] = []
    monkeypatch.setattr(
        cuda_runtime.ctypes,
        "CDLL",
        lambda path, **_kwargs: cdll_calls.append(str(path)),
    )

    with pytest.raises(RuntimeError) as error:
        cuda_runtime.configure_cuda_runtime()

    assert str(error.value) == "unsafe_cuda_runtime"
    assert unsafe_parent.name not in str(error.value)
    assert cdll_calls == []


def test_cuda_runtime_rejects_unsafe_intermediate_symlink_parent_before_cdll(
    tmp_path: Path, monkeypatch
) -> None:
    _isolate_cuda_environment(monkeypatch)
    _allow_namespaced_test_root(monkeypatch)
    cuda_dir = tmp_path / "cuda"
    unsafe_parent = tmp_path / "replaceable-link-parent"
    safe_target_parent = tmp_path / "safe-target-parent"
    cuda_dir.mkdir(mode=0o700)
    unsafe_parent.mkdir(mode=0o700)
    unsafe_parent.chmod(0o777)
    safe_target_parent.mkdir(mode=0o700)
    target = safe_target_parent / "libcudart.so.12.0"
    target.write_bytes(b"trusted-library")
    target.chmod(0o600)
    intermediate = unsafe_parent / "replaceable-link"
    intermediate.symlink_to(target)
    (cuda_dir / "libcudart.so.12").symlink_to(intermediate)
    monkeypatch.setenv("TRANSLATOR_CUDA_LIBRARY_PATH", str(cuda_dir))
    cdll_calls: list[str] = []
    monkeypatch.setattr(
        cuda_runtime.ctypes,
        "CDLL",
        lambda path, **_kwargs: cdll_calls.append(str(path)),
    )

    with pytest.raises(RuntimeError) as error:
        cuda_runtime.configure_cuda_runtime()

    assert str(error.value) == "unsafe_cuda_runtime"
    assert unsafe_parent.name not in str(error.value)
    assert cdll_calls == []


def test_cuda_runtime_loads_safe_soname_from_held_file_descriptor(
    tmp_path: Path, monkeypatch
) -> None:
    _isolate_cuda_environment(monkeypatch)
    _allow_namespaced_test_root(monkeypatch)
    cuda_dir = tmp_path / "cuda"
    cuda_dir.mkdir(mode=0o700)
    target = cuda_dir / "libcudart.so.12.0"
    target.write_bytes(b"trusted-library")
    target.chmod(0o600)
    soname = cuda_dir / "libcudart.so.12"
    soname.symlink_to(target.name)
    replacement = cuda_dir / "replacement.so"
    replacement.write_bytes(b"replacement")
    replacement.chmod(0o600)
    monkeypatch.setenv("TRANSLATOR_CUDA_LIBRARY_PATH", str(cuda_dir))
    loaded: list[tuple[str, int, bytes]] = []

    def fake_cdll(path: str, **_kwargs):
        fd = int(Path(path).name)
        soname.unlink()
        soname.symlink_to(replacement.name)
        with open(path, "rb", closefd=True) as held_file:
            loaded.append((path, os.fstat(fd).st_ino, held_file.read()))
        return object()

    monkeypatch.setattr(cuda_runtime.ctypes, "CDLL", fake_cdll)

    configured = cuda_runtime.configure_cuda_runtime()

    assert configured == (cuda_dir.resolve(),)
    assert len(loaded) == 1
    assert loaded[0][0].startswith("/proc/self/fd/")
    assert loaded[0][1] == target.stat().st_ino
    assert loaded[0][2] == b"trusted-library"
    _close_retained_library_descriptors()


def test_cuda_runtime_redacts_cdll_failure_and_closes_held_descriptor(
    tmp_path: Path, monkeypatch
) -> None:
    _isolate_cuda_environment(monkeypatch)
    _allow_namespaced_test_root(monkeypatch)
    cuda_dir = tmp_path / "hostile-path-marker"
    cuda_dir.mkdir(mode=0o700)
    library = cuda_dir / "libcudart.so.12"
    library.write_bytes(b"synthetic library")
    library.chmod(0o600)
    monkeypatch.setenv("TRANSLATOR_CUDA_LIBRARY_PATH", str(cuda_dir))
    attempted_descriptors: list[int] = []

    def fail_cdll(path: str, **_kwargs):
        attempted_descriptors.append(int(Path(path).name))
        raise OSError(f"failed to load {cuda_dir}")

    monkeypatch.setattr(cuda_runtime.ctypes, "CDLL", fail_cdll)

    with pytest.raises(RuntimeError) as error:
        cuda_runtime.configure_cuda_runtime()

    assert str(error.value) == "cuda_runtime_unavailable"
    assert cuda_dir.name not in str(error.value)
    assert error.value.__suppress_context__
    assert len(attempted_descriptors) == 1
    with pytest.raises(OSError):
        os.fstat(attempted_descriptors[0])
    assert cuda_runtime._LOADED_LIBRARIES == {}


def test_cuda_runtime_discards_ambient_loader_path(tmp_path: Path, monkeypatch) -> None:
    _isolate_cuda_environment(monkeypatch)
    _allow_namespaced_test_root(monkeypatch)
    cuda_dir = tmp_path / "cuda"
    cuda_dir.mkdir(mode=0o700)
    monkeypatch.setenv("TRANSLATOR_CUDA_LIBRARY_PATH", str(cuda_dir))
    monkeypatch.setenv("LD_LIBRARY_PATH", str(tmp_path / "attacker-controlled"))

    configured = cuda_runtime.configure_cuda_runtime(preload=False)

    assert configured == (cuda_dir.resolve(),)
    assert os.environ["LD_LIBRARY_PATH"] == str(cuda_dir.resolve())


def test_cuda_runtime_preloads_cublas_dependencies_before_dependents(
    tmp_path: Path, monkeypatch
) -> None:
    preload_names = tuple(
        name
        for name in cuda_runtime._PRELOAD_NAMES
        if name in {"libcudart.so.12", "libcublasLt.so.12", "libcublas.so.12"}
    )
    _isolate_cuda_environment(monkeypatch)
    _allow_namespaced_test_root(monkeypatch)
    monkeypatch.setattr(cuda_runtime, "_PRELOAD_NAMES", preload_names)
    cuda_dir = tmp_path / "cuda"
    cuda_dir.mkdir(mode=0o700)
    inode_names: dict[int, str] = {}
    for name in preload_names:
        library = cuda_dir / name
        library.write_bytes(name.encode())
        library.chmod(0o600)
        inode_names[library.stat().st_ino] = name
    monkeypatch.setenv("TRANSLATOR_CUDA_LIBRARY_PATH", str(cuda_dir))
    loaded_names: list[str] = []

    def dependency_checked_cdll(path: str, **_kwargs):
        name = inode_names[os.stat(path).st_ino]
        if name == "libcublas.so.12" and "libcublasLt.so.12" not in loaded_names:
            raise OSError("libcublasLt.so.12 is not loaded")
        loaded_names.append(name)
        return object()

    monkeypatch.setattr(cuda_runtime.ctypes, "CDLL", dependency_checked_cdll)
    try:
        configured = cuda_runtime.configure_cuda_runtime()
    finally:
        _close_retained_library_descriptors()

    assert configured == (cuda_dir.resolve(),)
    assert loaded_names == [
        "libcudart.so.12",
        "libcublasLt.so.12",
        "libcublas.so.12",
    ]


def test_cuda_runtime_serializes_parallel_first_configuration(
    tmp_path: Path, monkeypatch
) -> None:
    _isolate_cuda_environment(monkeypatch)
    _allow_namespaced_test_root(monkeypatch)
    cuda_dir = tmp_path / "cuda"
    cuda_dir.mkdir(mode=0o700)
    library = cuda_dir / "libcudart.so.12"
    library.write_bytes(b"trusted-library")
    library.chmod(0o600)

    callers_ready = threading.Barrier(3)
    first_cdll_entered = threading.Event()
    second_cdll_entered = threading.Event()
    release_first_cdll = threading.Event()
    state_lock = threading.Lock()
    cdll_calls = 0
    active_cdll_calls = 0
    max_active_cdll_calls = 0
    results: list[tuple[Path, ...]] = []
    errors: list[BaseException] = []
    loaded_snapshot: dict[str, tuple[object, int, tuple[int, int]]] = {}

    def controlled_cdll(_path: str, **_kwargs):
        nonlocal cdll_calls, active_cdll_calls, max_active_cdll_calls
        with state_lock:
            call_index = cdll_calls
            cdll_calls += 1
            active_cdll_calls += 1
            max_active_cdll_calls = max(max_active_cdll_calls, active_cdll_calls)
        try:
            if call_index == 0:
                first_cdll_entered.set()
                release_first_cdll.wait(timeout=5)
            else:
                second_cdll_entered.set()
            return object()
        finally:
            with state_lock:
                active_cdll_calls -= 1

    def configure() -> None:
        callers_ready.wait(timeout=5)
        try:
            results.append(
                cuda_runtime.configure_cuda_runtime(extra_library_dirs=(cuda_dir,))
            )
        except BaseException as error:
            errors.append(error)

    monkeypatch.setattr(cuda_runtime.ctypes, "CDLL", controlled_cdll)
    workers = [threading.Thread(target=configure) for _ in range(2)]
    try:
        for worker in workers:
            worker.start()
        callers_ready.wait(timeout=5)
        assert first_cdll_entered.wait(timeout=5)
        second_entered_before_release = second_cdll_entered.wait(timeout=0.5)
    finally:
        release_first_cdll.set()
        for worker in workers:
            worker.join(timeout=5)
        loaded_snapshot = dict(cuda_runtime._LOADED_LIBRARIES)
        _close_retained_library_descriptors()

    assert all(not worker.is_alive() for worker in workers)
    assert not second_entered_before_release
    assert errors == []
    assert results == [(cuda_dir.resolve(),), (cuda_dir.resolve(),)]
    assert cdll_calls == 1
    assert max_active_cdll_calls == 1
    assert len(loaded_snapshot) == 1


def test_cuda_runtime_rejects_parallel_conflicting_library_identity(
    tmp_path: Path, monkeypatch
) -> None:
    _isolate_cuda_environment(monkeypatch)
    _allow_namespaced_test_root(monkeypatch)
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir(mode=0o700)
    second_dir.mkdir(mode=0o700)
    for directory, content in (
        (first_dir, b"first-library"),
        (second_dir, b"second-library"),
    ):
        library = directory / "libcudart.so.12"
        library.write_bytes(content)
        library.chmod(0o600)
    first_library = first_dir / "libcudart.so.12"

    first_cdll_entered = threading.Event()
    second_cdll_entered = threading.Event()
    second_configure_started = threading.Event()
    release_first_cdll = threading.Event()
    state_lock = threading.Lock()
    cdll_calls = 0
    results: list[tuple[Path, ...]] = []
    errors: list[BaseException] = []
    loaded_snapshot: dict[str, tuple[object, int, tuple[int, int]]] = {}

    def controlled_cdll(_path: str, **_kwargs):
        nonlocal cdll_calls
        with state_lock:
            call_index = cdll_calls
            cdll_calls += 1
        if call_index == 0:
            first_cdll_entered.set()
            release_first_cdll.wait(timeout=5)
        else:
            second_cdll_entered.set()
        return object()

    def configure(directory: Path, *, started: threading.Event | None = None) -> None:
        if started is not None:
            started.set()
        try:
            results.append(
                cuda_runtime.configure_cuda_runtime(extra_library_dirs=(directory,))
            )
        except BaseException as error:
            errors.append(error)

    monkeypatch.setattr(cuda_runtime.ctypes, "CDLL", controlled_cdll)
    first_worker = threading.Thread(target=configure, args=(first_dir,))
    second_worker = threading.Thread(
        target=configure,
        args=(second_dir,),
        kwargs={"started": second_configure_started},
    )
    try:
        first_worker.start()
        assert first_cdll_entered.wait(timeout=5)
        second_worker.start()
        assert second_configure_started.wait(timeout=5)
        second_entered_before_release = second_cdll_entered.wait(timeout=0.5)
    finally:
        release_first_cdll.set()
        first_worker.join(timeout=5)
        second_worker.join(timeout=5)
        loaded_snapshot = dict(cuda_runtime._LOADED_LIBRARIES)
        _close_retained_library_descriptors()

    assert not first_worker.is_alive()
    assert not second_worker.is_alive()
    assert not second_entered_before_release
    assert results == [(first_dir.resolve(),)]
    assert [str(error) for error in errors] == ["unsafe_cuda_runtime"]
    assert cdll_calls == 1
    loaded = loaded_snapshot["libcudart.so.12"]
    assert loaded[2] == (first_library.stat().st_dev, first_library.stat().st_ino)


def test_cuda_runtime_restores_loader_path_after_preload_failure(
    tmp_path: Path, monkeypatch
) -> None:
    _isolate_cuda_environment(monkeypatch)
    _allow_namespaced_test_root(monkeypatch)
    cuda_dir = tmp_path / "cuda"
    cuda_dir.mkdir(mode=0o700)
    library = cuda_dir / "libcudart.so.12"
    library.write_bytes(b"trusted-library")
    library.chmod(0o600)
    previous = str(tmp_path / "previous-loader-path")
    monkeypatch.setenv("LD_LIBRARY_PATH", previous)
    monkeypatch.setattr(
        cuda_runtime.ctypes,
        "CDLL",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("load failed")),
    )

    with pytest.raises(RuntimeError) as error:
        cuda_runtime.configure_cuda_runtime(extra_library_dirs=(cuda_dir,))

    assert str(error.value) == "cuda_runtime_unavailable"
    assert os.environ["LD_LIBRARY_PATH"] == previous


def test_cuda_runtime_enforces_exact_library_tree_entry_limit(
    tmp_path: Path, monkeypatch
) -> None:
    _isolate_cuda_environment(monkeypatch)
    _allow_namespaced_test_root(monkeypatch)
    cuda_dir = tmp_path / "cuda"
    cuda_dir.mkdir(mode=0o700)
    for index in range(cuda_runtime._MAX_LIBRARY_TREE_ENTRIES):
        (cuda_dir / f"library-{index:04d}.so").touch(mode=0o600)

    assert cuda_runtime.configure_cuda_runtime(
        extra_library_dirs=(cuda_dir,), preload=False
    ) == (cuda_dir.resolve(),)

    (cuda_dir / "one-entry-too-many.so").touch(mode=0o600)
    with pytest.raises(RuntimeError) as error:
        cuda_runtime.configure_cuda_runtime(
            extra_library_dirs=(cuda_dir,), preload=False
        )

    assert str(error.value) == "unsafe_cuda_runtime"


@pytest.mark.parametrize(
    ("symlink_count", "accepted"),
    ((cuda_runtime._MAX_SYMLINKS, True), (cuda_runtime._MAX_SYMLINKS + 1, False)),
)
def test_cuda_runtime_enforces_exact_symlink_limit(
    tmp_path: Path, monkeypatch, symlink_count: int, accepted: bool
) -> None:
    _isolate_cuda_environment(monkeypatch)
    _allow_namespaced_test_root(monkeypatch)
    cuda_dir = tmp_path / f"cuda-{symlink_count}"
    cuda_dir.mkdir(mode=0o700)
    target = cuda_dir / "libcudart-target.so"
    target.write_bytes(b"trusted-library")
    target.chmod(0o600)
    previous = target
    for index in range(symlink_count - 1):
        link = cuda_dir / f"link-{index:02d}.so"
        link.symlink_to(previous.name)
        previous = link
    (cuda_dir / "libcudart.so.12").symlink_to(previous.name)
    monkeypatch.setattr(cuda_runtime.ctypes, "CDLL", lambda *_args, **_kwargs: object())

    try:
        if accepted:
            assert cuda_runtime.configure_cuda_runtime(
                extra_library_dirs=(cuda_dir,)
            ) == (cuda_dir.resolve(),)
        else:
            with pytest.raises(RuntimeError) as error:
                cuda_runtime.configure_cuda_runtime(extra_library_dirs=(cuda_dir,))
            assert str(error.value) == "unsafe_cuda_runtime"
    finally:
        _close_retained_library_descriptors()


def test_cuda_runtime_allows_root_sticky_traversal_but_rejects_sticky_leaf(
    tmp_path: Path, monkeypatch
) -> None:
    _isolate_cuda_environment(monkeypatch)
    sticky_leaf = tmp_path / "sticky-cuda"
    sticky_leaf.mkdir(mode=0o700)
    sticky_leaf.chmod(0o1777)
    metadata = os.lstat(sticky_leaf)
    with monkeypatch.context() as policy_patch:
        policy_patch.setattr(cuda_runtime, "_ROOT_UID", os.geteuid())
        cuda_runtime._safe_directory(metadata, traversal=True)
        with pytest.raises(RuntimeError) as policy_error:
            cuda_runtime._safe_directory(metadata, traversal=False)

    assert str(policy_error.value) == "unsafe_cuda_runtime"

    _allow_namespaced_test_root(monkeypatch)
    safe_leaf = tmp_path / "cuda"
    safe_leaf.mkdir(mode=0o700)

    assert cuda_runtime.configure_cuda_runtime(
        extra_library_dirs=(safe_leaf,), preload=False
    ) == (safe_leaf.resolve(),)

    with pytest.raises(RuntimeError) as error:
        cuda_runtime.configure_cuda_runtime(
            extra_library_dirs=(sticky_leaf,), preload=False
        )

    assert str(error.value) == "unsafe_cuda_runtime"


def test_cuda_runtime_keeps_only_committed_preloads_after_later_failure(
    tmp_path: Path, monkeypatch
) -> None:
    _isolate_cuda_environment(monkeypatch)
    _allow_namespaced_test_root(monkeypatch)
    names = ("libcudart.so.12", "libcublasLt.so.12")
    monkeypatch.setattr(cuda_runtime, "_PRELOAD_NAMES", names)
    cuda_dir = tmp_path / "cuda"
    cuda_dir.mkdir(mode=0o700)
    inode_names: dict[int, str] = {}
    for name in names:
        library = cuda_dir / name
        library.write_bytes(name.encode())
        library.chmod(0o600)
        inode_names[library.stat().st_ino] = name
    attempted_descriptors: list[int] = []

    def fail_second_cdll(path: str, **_kwargs):
        descriptor = int(Path(path).name)
        attempted_descriptors.append(descriptor)
        if inode_names[os.fstat(descriptor).st_ino] == names[1]:
            raise OSError("second preload failed")
        return object()

    monkeypatch.setattr(cuda_runtime.ctypes, "CDLL", fail_second_cdll)
    try:
        with pytest.raises(RuntimeError) as error:
            cuda_runtime.configure_cuda_runtime(extra_library_dirs=(cuda_dir,))

        assert str(error.value) == "cuda_runtime_unavailable"
        assert list(cuda_runtime._LOADED_LIBRARIES) == [names[0]]
        assert cuda_runtime._LOADED_LIBRARIES[names[0]][1] == attempted_descriptors[0]
        os.fstat(attempted_descriptors[0])
        with pytest.raises(OSError):
            os.fstat(attempted_descriptors[1])
    finally:
        _close_retained_library_descriptors()


def test_cuda_runtime_closes_descriptors_when_preload_never_takes_ownership(
    tmp_path: Path, monkeypatch
) -> None:
    _isolate_cuda_environment(monkeypatch)
    _allow_namespaced_test_root(monkeypatch)
    cuda_dir = tmp_path / "cuda"
    cuda_dir.mkdir(mode=0o700)
    library = cuda_dir / "libcudart.so.12"
    library.write_bytes(b"trusted-library")
    library.chmod(0o600)
    previous = str(tmp_path / "previous-loader-path")
    monkeypatch.setenv("LD_LIBRARY_PATH", previous)
    opened_descriptors: list[int] = []
    original_open = cuda_runtime._open_preload_libraries

    def capture_opened(library_dirs: tuple[Path, ...]):
        opened = original_open(library_dirs)
        opened_descriptors.extend(descriptor for _, descriptor, _ in opened)
        return opened

    monkeypatch.setattr(cuda_runtime, "_open_preload_libraries", capture_opened)
    monkeypatch.setattr(
        cuda_runtime,
        "_preload_libraries",
        lambda _opened: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    try:
        with pytest.raises(KeyboardInterrupt):
            cuda_runtime.configure_cuda_runtime(extra_library_dirs=(cuda_dir,))
        descriptor_is_closed = []
        for descriptor in opened_descriptors:
            try:
                os.fstat(descriptor)
            except OSError:
                descriptor_is_closed.append(True)
            else:
                descriptor_is_closed.append(False)
    finally:
        for descriptor in opened_descriptors:
            try:
                os.close(descriptor)
            except OSError:
                pass

    assert descriptor_is_closed == [True]
    assert os.environ["LD_LIBRARY_PATH"] == previous


def test_cuda_runtime_does_not_drop_queue_ownership_before_preload(
    tmp_path: Path, monkeypatch
) -> None:
    _isolate_cuda_environment(monkeypatch)
    _allow_namespaced_test_root(monkeypatch)
    cuda_dir = tmp_path / "cuda"
    cuda_dir.mkdir(mode=0o700)
    library = cuda_dir / "libcudart.so.12"
    library.write_bytes(b"trusted-library")
    library.chmod(0o600)
    opened_descriptors: list[int] = []
    original_open = cuda_runtime._open_preload_libraries

    class InterruptAfterPop(list):
        def pop(self, index=-1):
            super().pop(index)
            raise KeyboardInterrupt

    def interruptible_open(library_dirs: tuple[Path, ...]):
        opened = original_open(library_dirs)
        opened_descriptors.extend(descriptor for _, descriptor, _ in opened)
        return InterruptAfterPop(opened)

    monkeypatch.setattr(cuda_runtime, "_open_preload_libraries", interruptible_open)
    monkeypatch.setattr(cuda_runtime.ctypes, "CDLL", lambda *_args, **_kwargs: object())

    try:
        configured = cuda_runtime.configure_cuda_runtime(extra_library_dirs=(cuda_dir,))
        descriptor_is_open = []
        for descriptor in opened_descriptors:
            try:
                os.fstat(descriptor)
            except OSError:
                descriptor_is_open.append(False)
            else:
                descriptor_is_open.append(True)
    finally:
        _close_retained_library_descriptors()

    assert configured == (cuda_dir.resolve(),)
    assert descriptor_is_open == [True]

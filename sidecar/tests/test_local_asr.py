from __future__ import annotations

import builtins
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
from threading import Event, Lock
import traceback
import weakref

import numpy as np
import pytest

from translator_sidecar.local.asr import (
    AsrModelManager,
    AsrUnavailable,
    AsrUnsupported,
)
from translator_sidecar.provider_contract import Language, TranslationMode


class Segment:
    def __init__(self, text: str) -> None:
        self.text = text


class AdmissionProbeLock:
    def __init__(self) -> None:
        self._lock = Lock()
        self._counter_lock = Lock()
        self.attempts = 0
        self.second_attempt = Event()

    def __enter__(self) -> AdmissionProbeLock:
        with self._counter_lock:
            self.attempts += 1
            if self.attempts == 2:
                self.second_attempt.set()
        assert self._lock.acquire(timeout=2)
        return self

    def __exit__(
        self,
        _exc_type: object,
        _exc_value: object,
        _traceback: object,
    ) -> None:
        self._lock.release()


class FakeNativeModel:
    def __init__(
        self,
        *,
        events: list[str],
        name: str,
        failure: Exception | None = None,
    ) -> None:
        self.events = events
        self.name = name
        self.failure = failure

    def unload_model(self) -> None:
        self.events.append(f"unload:{self.name}")
        if self.failure is not None:
            raise self.failure


class FakeWhisper:
    def __init__(
        self,
        *,
        text: str = " final text ",
        failures: list[Exception] | None = None,
        entered: Event | None = None,
        release: Event | None = None,
        active: list[int] | None = None,
        active_lock: Lock | None = None,
        events: list[str] | None = None,
    ) -> None:
        self.text = text
        self.failures = failures or []
        self.entered = entered
        self.release = release
        self.active = active
        self.active_lock = active_lock
        self.events = events
        self.calls: list[tuple[np.ndarray, dict[str, object]]] = []

    def transcribe(self, audio: np.ndarray, **kwargs: object) -> tuple[object, object]:
        self.calls.append((audio, kwargs))

        def segments() -> object:
            if self.active is not None and self.active_lock is not None:
                with self.active_lock:
                    self.active[0] += 1
                    self.active[1] = max(self.active[1], self.active[0])
            if self.events is not None:
                self.events.append("infer:enter")
            if self.entered is not None:
                self.entered.set()
            try:
                if self.release is not None:
                    assert self.release.wait(timeout=2)
                if self.failures:
                    raise self.failures.pop(0)
                yield Segment(self.text)
            finally:
                if self.active is not None and self.active_lock is not None:
                    with self.active_lock:
                        self.active[0] -= 1

        return segments(), object()


def pcm() -> bytes:
    return np.array([0, 16384, -16384, 32767], dtype=np.int16).tobytes()


@pytest.mark.parametrize(
    ("mode", "beam_size"),
    [
        (TranslationMode.QUALITY_FIRST, 5),
        (TranslationMode.BALANCED, 3),
        (TranslationMode.STREAMING_FIRST, 1),
    ],
)
def test_asr_converts_s16le_and_uses_mode_beam(
    tmp_path: Path, mode: TranslationMode, beam_size: int
) -> None:
    model = FakeWhisper()
    manager = AsrModelManager(
        selected_id="small",
        model_paths={"small": tmp_path},
        device="cpu",
        model_factory=lambda _path, **_kwargs: model,
    )

    assert manager.transcribe(pcm(), language=Language.RU, mode=mode) == ("final text")

    audio, kwargs = model.calls[0]
    assert audio.dtype == np.float32
    assert audio.tolist() == pytest.approx([0, 0.5, -0.5, 32767 / 32768])
    assert kwargs == {
        "language": "ru",
        "beam_size": beam_size,
        "vad_filter": False,
        "condition_on_previous_text": False,
    }


@pytest.mark.parametrize(
    ("device", "compute_type"),
    [("cuda", "float16"), ("cpu", "int8")],
)
def test_asr_loads_absolute_local_path_without_download(
    tmp_path: Path, device: str, compute_type: str
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    model = FakeWhisper()

    def factory(path: str, **kwargs: object) -> FakeWhisper:
        calls.append((path, kwargs))
        return model

    manager = AsrModelManager(
        selected_id="small",
        model_paths={"small": tmp_path},
        device=device,
        model_factory=factory,
    )
    manager.transcribe(pcm(), language=Language.EN, mode=TranslationMode.BALANCED)

    assert calls == [
        (
            str(tmp_path),
            {
                "device": device,
                "compute_type": compute_type,
                "local_files_only": True,
                "num_workers": 1,
            },
        )
    ]
    assert manager.resident_model_id == "small"
    assert manager.residency_generation == 1
    if device == "cpu":
        assert manager.degraded
    assert manager.actual_device == device


def test_asr_prepare_establishes_residency_without_running_inference(
    tmp_path: Path,
) -> None:
    model = FakeWhisper()
    manager = AsrModelManager(
        selected_id="small",
        model_paths={"small": tmp_path},
        device="cpu",
        model_factory=lambda _path, **_kwargs: model,
    )

    manager.prepare()

    assert manager.resident_model_id == "small"
    assert manager.residency_generation == 1
    assert model.calls == []


def test_asr_explicit_release_proves_native_model_is_gone(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    model_refs: list[weakref.ReferenceType[FakeWhisper]] = []

    def factory(_path: str, **_kwargs: object) -> FakeWhisper:
        model = FakeWhisper(events=events)
        model.model = FakeNativeModel(events=events, name="small")
        model_refs.append(weakref.ref(model))
        return model

    manager = AsrModelManager(
        selected_id="small",
        model_paths={"small": tmp_path},
        device="cpu",
        model_factory=factory,
        release_cuda=lambda: events.append("release"),
    )
    manager.transcribe(
        pcm(),
        language=Language.EN,
        mode=TranslationMode.BALANCED,
    )

    assert manager.release()
    assert manager.resident_model_id is None
    assert model_refs[0]() is None
    assert events == ["infer:enter", "unload:small", "release"]


def test_asr_enables_offline_controls_before_model_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in (
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "HF_DATASETS_OFFLINE",
    ):
        monkeypatch.delenv(name, raising=False)

    def factory(_path: str, **_kwargs: object) -> FakeWhisper:
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
        assert os.environ["HF_DATASETS_OFFLINE"] == "1"
        return FakeWhisper()

    manager = AsrModelManager(
        selected_id="small",
        model_paths={"small": tmp_path},
        device="cpu",
        model_factory=factory,
    )
    manager.transcribe(pcm(), language=Language.EN, mode=TranslationMode.BALANCED)


@pytest.mark.parametrize(
    "path_factory",
    [
        lambda tmp_path: Path("relative-model"),
        lambda tmp_path: tmp_path / "missing",
        lambda tmp_path: tmp_path / "model.bin",
    ],
)
def test_asr_rejects_non_local_model_path_before_load(
    tmp_path: Path,
    path_factory: object,
) -> None:
    selected_path = path_factory(tmp_path)  # type: ignore[operator]
    if selected_path.name == "model.bin":
        selected_path.write_bytes(b"not a model directory")
    manager = AsrModelManager(
        selected_id="small",
        model_paths={"small": selected_path},
        device="cpu",
        model_factory=lambda _path, **_kwargs: pytest.fail("must not load"),
    )

    with pytest.raises(AsrUnavailable, match="unavailable"):
        manager.transcribe(pcm(), language=Language.EN, mode=TranslationMode.BALANCED)


def test_asr_falls_back_to_small_cpu_when_cuda_is_unavailable(
    tmp_path: Path,
) -> None:
    small_path = tmp_path / "small"
    small_path.mkdir()
    calls: list[tuple[str, dict[str, object]]] = []

    def factory(path: str, **kwargs: object) -> FakeWhisper:
        calls.append((path, kwargs))
        return FakeWhisper(text="cpu result")

    manager = AsrModelManager(
        selected_id="large-v3",
        model_paths={
            "large-v3": tmp_path / "large-v3",
            "small": small_path,
        },
        device="cuda",
        model_factory=factory,
        cuda_available=lambda: False,
    )

    assert (
        manager.transcribe(pcm(), language=Language.RU, mode=TranslationMode.BALANCED)
        == "cpu result"
    )
    assert calls == [
        (
            str(small_path),
            {
                "device": "cpu",
                "compute_type": "int8",
                "local_files_only": True,
                "num_workers": 1,
            },
        )
    ]
    assert manager.actual_device == "cpu"
    assert manager.resident_model_id == "small"
    assert manager.degraded


def test_cpu_rejects_large_candidate_before_model_load(tmp_path: Path) -> None:
    calls = 0

    def factory(_path: str, **_kwargs: object) -> FakeWhisper:
        nonlocal calls
        calls += 1
        return FakeWhisper()

    manager = AsrModelManager(
        selected_id="large-v3",
        model_paths={"large-v3": tmp_path, "small": tmp_path},
        device="cpu",
        model_factory=factory,
    )

    with pytest.raises(AsrUnsupported, match="large"):
        manager.transcribe(
            pcm(), language=Language.RU, mode=TranslationMode.QUALITY_FIRST
        )
    assert calls == 0


def test_cuda_oom_on_large_unloads_then_retries_small_once(
    tmp_path: Path,
) -> None:
    (tmp_path / "large-v3").mkdir()
    (tmp_path / "small").mkdir()
    events: list[str] = []
    model_refs: list[weakref.ReferenceType[FakeWhisper]] = []
    loads = {"large-v3": 0, "small": 0}
    large_entered = Event()
    release_large = Event()
    admission_lock = AdmissionProbeLock()

    def factory(path: str, **_kwargs: object) -> FakeWhisper:
        model_id = Path(path).name
        loads[model_id] += 1
        if model_id == "small":
            assert model_refs[0]() is None
        events.append(f"load:{model_id}")
        model = FakeWhisper(
            text="small result",
            failures=(
                [RuntimeError("CUDA out of memory: private large marker")]
                if model_id == "large-v3"
                else None
            ),
            entered=large_entered if model_id == "large-v3" else None,
            release=release_large if model_id == "large-v3" else None,
            events=events,
        )
        model.model = FakeNativeModel(events=events, name=model_id)
        model_refs.append(weakref.ref(model))
        return model

    manager = AsrModelManager(
        selected_id="large-v3",
        model_paths={
            "large-v3": tmp_path / "large-v3",
            "small": tmp_path / "small",
        },
        device="cuda",
        model_factory=factory,
        release_cuda=lambda: events.append("release"),
        admission_lock=admission_lock,
    )

    def run(language: Language) -> str:
        return manager.transcribe(
            pcm(), language=language, mode=TranslationMode.QUALITY_FIRST
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(run, Language.RU)
        assert large_entered.wait(timeout=2)
        second = pool.submit(run, Language.EN)
        assert admission_lock.second_attempt.wait(timeout=2)
        assert not second.done()
        assert events == ["load:large-v3", "infer:enter"]
        assert loads == {"large-v3": 1, "small": 0}
        release_large.set()
        futures = [first, second]
        assert [future.result(timeout=3) for future in futures] == [
            "small result",
            "small result",
        ]
    assert events == [
        "load:large-v3",
        "infer:enter",
        "unload:large-v3",
        "release",
        "load:small",
        "infer:enter",
        "infer:enter",
    ]
    assert loads == {"large-v3": 1, "small": 1}
    assert manager.resident_model_id == "small"
    assert manager.residency_generation == 2
    assert manager.degraded
    assert model_refs[0]() is None
    assert model_refs[1]() is not None


def test_cuda_oom_while_loading_large_falls_back_to_small(
    tmp_path: Path,
) -> None:
    large_path = tmp_path / "large-v3"
    small_path = tmp_path / "small"
    large_path.mkdir()
    small_path.mkdir()
    events: list[str] = []

    def factory(path: str, **_kwargs: object) -> FakeWhisper:
        model_id = Path(path).name
        events.append(f"load:{model_id}")
        if model_id == "large-v3":
            raise RuntimeError("CUDA out of memory: private load marker")
        model = FakeWhisper(text="small result")
        model.model = FakeNativeModel(events=events, name=model_id)
        return model

    manager = AsrModelManager(
        selected_id="large-v3",
        model_paths={"large-v3": large_path, "small": small_path},
        device="cuda",
        model_factory=factory,
        release_cuda=lambda: events.append("release"),
    )

    assert (
        manager.transcribe(
            pcm(), language=Language.RU, mode=TranslationMode.QUALITY_FIRST
        )
        == "small result"
    )
    assert events == ["load:large-v3", "release", "load:small"]
    assert manager.resident_model_id == "small"
    assert manager.residency_generation == 2
    assert manager.degraded


def test_cuda_oom_while_loading_small_enters_terminal_unavailable(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    loads = 0

    def factory(_path: str, **_kwargs: object) -> FakeWhisper:
        nonlocal loads
        loads += 1
        raise RuntimeError("CUDA out of memory: private load marker")

    manager = AsrModelManager(
        selected_id="small",
        model_paths={"small": tmp_path},
        device="cuda",
        model_factory=factory,
        release_cuda=lambda: events.append("release"),
    )

    with pytest.raises(AsrUnavailable, match="unavailable") as raised:
        manager.transcribe(pcm(), language=Language.EN, mode=TranslationMode.BALANCED)
    with pytest.raises(AsrUnavailable, match="unavailable"):
        manager.transcribe(pcm(), language=Language.EN, mode=TranslationMode.BALANCED)
    assert loads == 1
    assert events == ["release"]
    assert manager.resident_model_id is None
    assert manager.residency_generation == 1
    assert manager.unavailable
    rendered = "".join(
        traceback.format_exception(
            type(raised.value), raised.value, raised.value.__traceback__
        )
    )
    assert "private load marker" not in rendered


def test_cuda_fallback_fails_closed_if_old_wrapper_is_still_alive(
    tmp_path: Path,
) -> None:
    large_path = tmp_path / "large-v3"
    small_path = tmp_path / "small"
    large_path.mkdir()
    small_path.mkdir()
    events: list[str] = []
    retained_large = FakeWhisper(failures=[RuntimeError("CUDA out of memory")])
    retained_large.model = FakeNativeModel(events=events, name="large-v3")

    def factory(path: str, **_kwargs: object) -> FakeWhisper:
        model_id = Path(path).name
        events.append(f"load:{model_id}")
        if model_id == "large-v3":
            return retained_large
        return FakeWhisper(text="must not run")

    manager = AsrModelManager(
        selected_id="large-v3",
        model_paths={"large-v3": large_path, "small": small_path},
        device="cuda",
        model_factory=factory,
        release_cuda=lambda: events.append("release"),
    )

    with pytest.raises(AsrUnavailable, match="unavailable"):
        manager.transcribe(pcm(), language=Language.RU, mode=TranslationMode.BALANCED)
    assert events == ["load:large-v3", "unload:large-v3", "release"]
    assert manager.resident_model_id is None
    assert manager.unavailable


def test_cuda_oom_on_small_enters_unavailable_without_retry(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    loads = 0
    model_ref: weakref.ReferenceType[FakeWhisper] | None = None

    def factory(_path: str, **_kwargs: object) -> FakeWhisper:
        nonlocal loads, model_ref
        loads += 1
        model = FakeWhisper(
            failures=[RuntimeError("CUDA out of memory: private small marker")]
        )
        model.model = FakeNativeModel(events=events, name="small")
        model_ref = weakref.ref(model)
        return model

    manager = AsrModelManager(
        selected_id="small",
        model_paths={"small": tmp_path},
        device="cuda",
        model_factory=factory,
        release_cuda=lambda: events.append("release"),
    )

    with pytest.raises(AsrUnavailable, match="unavailable") as raised:
        manager.transcribe(
            pcm(), language=Language.EN, mode=TranslationMode.STREAMING_FIRST
        )
    with pytest.raises(AsrUnavailable, match="unavailable"):
        manager.transcribe(
            pcm(), language=Language.EN, mode=TranslationMode.STREAMING_FIRST
        )
    assert loads == 1
    assert events == ["unload:small", "release"]
    assert model_ref is not None
    assert model_ref() is None
    assert manager.resident_model_id is None
    assert manager.residency_generation == 2
    assert manager.unavailable
    rendered = "".join(
        traceback.format_exception(
            type(raised.value), raised.value, raised.value.__traceback__
        )
    )
    assert "private small marker" not in rendered


def test_cpu_oom_on_small_enters_unavailable_without_retry(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    loads = 0
    model = FakeWhisper(failures=[RuntimeError("out of memory")])
    model.model = FakeNativeModel(events=events, name="small")

    def factory(_path: str, **_kwargs: object) -> FakeWhisper:
        nonlocal loads
        loads += 1
        return model

    manager = AsrModelManager(
        selected_id="small",
        model_paths={"small": tmp_path},
        device="cpu",
        model_factory=factory,
    )

    with pytest.raises(AsrUnavailable, match="unavailable"):
        manager.transcribe(pcm(), language=Language.EN, mode=TranslationMode.BALANCED)
    with pytest.raises(AsrUnavailable, match="unavailable"):
        manager.transcribe(pcm(), language=Language.EN, mode=TranslationMode.BALANCED)
    assert loads == 1
    assert len(model.calls) == 1
    assert events == ["unload:small"]
    assert manager.resident_model_id is None
    assert manager.unavailable


@pytest.mark.parametrize("failure_at", ["unload", "release"])
def test_asr_cleanup_failure_is_sanitized_and_fail_closed(
    tmp_path: Path, failure_at: str
) -> None:
    marker = f"private {failure_at} cleanup marker"
    events: list[str] = []
    model = FakeWhisper(failures=[RuntimeError("CUDA out of memory")])
    model.model = FakeNativeModel(
        events=events,
        name="small",
        failure=RuntimeError(marker) if failure_at == "unload" else None,
    )

    def release_cuda() -> None:
        events.append("release")
        if failure_at == "release":
            raise RuntimeError(marker)

    manager = AsrModelManager(
        selected_id="small",
        model_paths={"small": tmp_path},
        device="cuda",
        model_factory=lambda _path, **_kwargs: model,
        release_cuda=release_cuda,
    )

    with pytest.raises(AsrUnavailable, match="unavailable") as raised:
        manager.transcribe(pcm(), language=Language.EN, mode=TranslationMode.BALANCED)
    assert manager.resident_model_id is None
    assert manager.unavailable
    assert events[0] == "unload:small"
    cleanup_events = list(events)
    with pytest.raises(AsrUnavailable, match="unavailable"):
        manager.transcribe(pcm(), language=Language.EN, mode=TranslationMode.BALANCED)
    assert events == cleanup_events
    rendered = "".join(
        traceback.format_exception(
            type(raised.value), raised.value, raised.value.__traceback__
        )
    )
    assert marker not in rendered


def test_asr_serializes_native_inference_across_directions(
    tmp_path: Path,
) -> None:
    entered = Event()
    release = Event()
    admission_lock = AdmissionProbeLock()
    active = [0, 0]
    active_lock = Lock()
    events: list[str] = []
    model = FakeWhisper(
        entered=entered,
        release=release,
        active=active,
        active_lock=active_lock,
        events=events,
    )
    manager = AsrModelManager(
        selected_id="small",
        model_paths={"small": tmp_path},
        device="cpu",
        model_factory=lambda _path, **_kwargs: model,
        admission_lock=admission_lock,
    )

    def run(language: Language) -> str:
        return manager.transcribe(
            pcm(), language=language, mode=TranslationMode.BALANCED
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(run, Language.RU)
        assert entered.wait(timeout=2)
        second = pool.submit(run, Language.EN)
        assert admission_lock.second_attempt.wait(timeout=2)
        assert events == ["infer:enter"]
        assert not second.done()
        release.set()
        assert [future.result(timeout=3) for future in (first, second)] == [
            "final text",
            "final text",
        ]
    assert active[1] == 1
    assert events == ["infer:enter", "infer:enter"]


def test_asr_cold_start_loads_exactly_one_resident_model(
    tmp_path: Path,
) -> None:
    load_entered = Event()
    release_load = Event()
    admission_lock = AdmissionProbeLock()
    loads: list[FakeWhisper] = []

    def factory(_path: str, **_kwargs: object) -> FakeWhisper:
        model = FakeWhisper()
        loads.append(model)
        load_entered.set()
        assert release_load.wait(timeout=2)
        return model

    manager = AsrModelManager(
        selected_id="small",
        model_paths={"small": tmp_path},
        device="cpu",
        model_factory=factory,
        admission_lock=admission_lock,
    )

    def run(language: Language) -> str:
        return manager.transcribe(
            pcm(), language=language, mode=TranslationMode.BALANCED
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(run, Language.RU)
        assert load_entered.wait(timeout=2)
        second = pool.submit(run, Language.EN)
        assert admission_lock.second_attempt.wait(timeout=2)
        assert len(loads) == 1
        assert not second.done()
        release_load.set()
        assert first.result(timeout=3) == "final text"
        assert second.result(timeout=3) == "final text"
    assert len(loads) == 1
    assert manager.resident_model_id == "small"
    assert manager.residency_generation == 1


def test_asr_consumes_all_lazy_segments(tmp_path: Path) -> None:
    class MultiSegmentWhisper:
        def transcribe(
            self, _audio: np.ndarray, **_kwargs: object
        ) -> tuple[object, object]:
            def segments() -> object:
                yield Segment(" hello")
                yield Segment(" world ")

            return segments(), object()

    manager = AsrModelManager(
        selected_id="small",
        model_paths={"small": tmp_path},
        device="cpu",
        model_factory=lambda _path, **_kwargs: MultiSegmentWhisper(),
    )

    assert (
        manager.transcribe(pcm(), language=Language.EN, mode=TranslationMode.BALANCED)
        == "hello world"
    )


def test_asr_does_not_return_partial_text_after_lazy_failure(
    tmp_path: Path,
) -> None:
    marker = "private failure after first segment"

    class PartialThenFailureWhisper:
        def transcribe(
            self, _audio: np.ndarray, **_kwargs: object
        ) -> tuple[object, object]:
            def segments() -> object:
                yield Segment("partial")
                raise RuntimeError(marker)

            return segments(), object()

    manager = AsrModelManager(
        selected_id="small",
        model_paths={"small": tmp_path},
        device="cpu",
        model_factory=lambda _path, **_kwargs: PartialThenFailureWhisper(),
    )

    with pytest.raises(AsrUnavailable, match="unavailable") as raised:
        manager.transcribe(pcm(), language=Language.EN, mode=TranslationMode.BALANCED)
    rendered = "".join(
        traceback.format_exception(
            type(raised.value), raised.value, raised.value.__traceback__
        )
    )
    assert marker not in rendered


def test_asr_sanitizes_missing_dependency_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_import = builtins.__import__

    def blocked_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "faster_whisper":
            raise ModuleNotFoundError("private dependency marker")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    manager = AsrModelManager(
        selected_id="small",
        model_paths={"small": tmp_path},
        device="cpu",
    )

    with pytest.raises(AsrUnavailable, match="unavailable") as raised:
        manager.transcribe(pcm(), language=Language.RU, mode=TranslationMode.BALANCED)
    rendered = "".join(
        traceback.format_exception(
            type(raised.value), raised.value, raised.value.__traceback__
        )
    )
    assert "private dependency marker" not in rendered


@pytest.mark.parametrize("failure_at", ["load", "inference"])
def test_asr_sanitizes_non_oom_native_failure_traceback(
    tmp_path: Path, failure_at: str
) -> None:
    marker = f"private {failure_at} marker"

    def factory(_path: str, **_kwargs: object) -> FakeWhisper:
        if failure_at == "load":
            raise RuntimeError(marker)
        return FakeWhisper(failures=[RuntimeError(marker)])

    manager = AsrModelManager(
        selected_id="small",
        model_paths={"small": tmp_path},
        device="cpu",
        model_factory=factory,
    )
    with pytest.raises(AsrUnavailable, match="unavailable") as raised:
        manager.transcribe(pcm(), language=Language.RU, mode=TranslationMode.BALANCED)
    rendered = "".join(
        traceback.format_exception(
            type(raised.value), raised.value, raised.value.__traceback__
        )
    )
    assert marker not in rendered


@pytest.mark.parametrize("payload", [b"", b"\x00"])
def test_asr_rejects_invalid_pcm_before_model_load(
    tmp_path: Path, payload: bytes
) -> None:
    calls = 0

    def factory(_path: str, **_kwargs: object) -> FakeWhisper:
        nonlocal calls
        calls += 1
        return FakeWhisper()

    manager = AsrModelManager(
        selected_id="small",
        model_paths={"small": tmp_path},
        device="cpu",
        model_factory=factory,
    )
    with pytest.raises(AsrUnavailable, match="PCM"):
        manager.transcribe(payload, language=Language.RU, mode=TranslationMode.BALANCED)
    assert calls == 0

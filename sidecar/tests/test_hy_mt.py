from __future__ import annotations

import os
import stat
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pytest

from translator_sidecar.local.hy_mt import HyMtTranslator, _server_command
from translator_sidecar.local.mt import LocalTranslationError
from translator_sidecar.provider_contract import Language, TranslationMode


def translated(text: str = "translated") -> dict:
    return {"choices": [{"message": {"content": text}, "finish_reason": "stop"}]}


class FakeLease:
    names = ("Hy-MT2-1.8B-Q4_K_M.gguf",)

    def __init__(self, path: Path) -> None:
        self.fd = os.open(path, os.O_RDONLY)
        self.closed = False

    def path(self, _name: str) -> str:
        return f"/proc/self/fd/{self.fd}"

    def descriptor(self, _name: str) -> int:
        return self.fd

    def close(self) -> None:
        if not self.closed:
            os.close(self.fd)
            self.closed = True


class FakeSource:
    def __init__(self, path: Path) -> None:
        self.lease = FakeLease(path)

    def acquire(self) -> FakeLease:
        return self.lease


class FakeProcess:
    def __init__(self) -> None:
        self.returncode: int | None = None
        self.terminated = False
        self.waited = False

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = 0

    def wait(self, timeout: float | None = None) -> int:
        self.waited = True
        return self.returncode or 0

    def kill(self) -> None:
        self.returncode = -9


def load_fake_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    responder,
) -> tuple[HyMtTranslator, FakeProcess, FakeLease]:
    import translator_sidecar.local.hy_mt as hy_mt

    model = tmp_path / "model.gguf"
    model.write_bytes(b"gguf")
    server = tmp_path / "llama-server"
    backend = tmp_path / "libggml-cuda.so"
    server.write_bytes(b"server")
    backend.write_bytes(b"backend")
    server.chmod(0o755)
    backend.chmod(0o644)
    source = FakeSource(model)
    process = FakeProcess()
    monkeypatch.setattr(hy_mt, "VerifiedModelSource", FakeSource)
    monkeypatch.setattr(hy_mt.subprocess, "Popen", lambda *_a, **_kw: process)
    monkeypatch.setattr(HyMtTranslator, "_request_json", responder)
    adapter = HyMtTranslator.load(
        source, server_path=server, backend_path=backend, startup_timeout_s=0.1
    )
    return adapter, process, source.lease


def test_hy_command_is_offline_private_and_resource_bounded() -> None:
    command = _server_command(
        Path("/usr/local/lib/ollama/llama-server"),
        "/proc/self/fd/12",
        Path("/private/translator.sock"),
    )
    assert command[:3] == [
        "/usr/local/lib/ollama/llama-server",
        "--model",
        "/proc/self/fd/12",
    ]
    assert command[command.index("--host") + 1] == "/private/translator.sock"
    assert command[command.index("--threads") + 1] == "2"
    assert command[command.index("--threads-batch") + 1] == "2"
    assert command[command.index("--parallel") + 1] == "1"
    assert command[command.index("--gpu-layers") + 1] == "99"
    assert {"--offline", "--no-ui", "--log-disable", "--jinja"}.issubset(command)


def test_hy_load_owns_child_fd_socket_and_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import translator_sidecar.local.hy_mt as hy_mt

    model = tmp_path / "model.gguf"
    model.write_bytes(b"gguf")
    server = tmp_path / "llama-server"
    backend = tmp_path / "libggml-cuda.so"
    server.write_bytes(b"server")
    backend.write_bytes(b"backend")
    server.chmod(0o755)
    backend.chmod(0o644)
    source = FakeSource(model)
    process = FakeProcess()
    calls: dict[str, Any] = {}

    def start(command, **kwargs):
        calls["command"] = command
        calls["kwargs"] = kwargs
        return process

    monkeypatch.setattr(hy_mt, "VerifiedModelSource", FakeSource)
    monkeypatch.setattr(hy_mt.subprocess, "Popen", start)
    monkeypatch.setattr(
        HyMtTranslator,
        "_request_json",
        lambda self, method, *_args, **_kwargs: (
            {"status": "ok"} if method == "GET" else translated()
        ),
    )

    translator = HyMtTranslator.load(
        source, server_path=server, backend_path=backend, startup_timeout_s=1
    )
    socket_path = Path(calls["command"][calls["command"].index("--host") + 1])
    assert stat.S_IMODE(socket_path.parent.stat().st_mode) == 0o700
    assert calls["kwargs"]["pass_fds"] == (source.lease.fd,)
    assert "proxy" not in " ".join(calls["kwargs"]["env"]).lower()
    assert calls["kwargs"]["env"]["GGML_BACKEND_PATH"] == str(backend)
    assert calls["kwargs"]["stdout"] is not None
    assert calls["kwargs"]["stderr"] is not None
    assert not translator.unavailable

    translator.close()
    translator.close()
    assert process.terminated and process.waited
    assert source.lease.closed
    assert not socket_path.parent.exists()


@pytest.mark.parametrize("visible", ["", "-1", "0"])
def test_hy_load_preserves_explicit_cuda_visibility(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, visible: str
) -> None:
    import translator_sidecar.local.hy_mt as hy_mt

    captured: dict[str, Any] = {}
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    monkeypatch.setattr(
        hy_mt.subprocess,
        "Popen",
        lambda *_a, **kw: captured.update(kw) or FakeProcess(),
    )
    monkeypatch.setattr(
        HyMtTranslator,
        "_request_json",
        lambda self, method, *_a, **_kw: (
            {"status": "ok"} if method == "GET" else translated()
        ),
    )
    monkeypatch.setattr(hy_mt, "VerifiedModelSource", FakeSource)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"gguf")
    server = tmp_path / "llama-server"
    backend = tmp_path / "libggml-cuda.so"
    server.write_bytes(b"server")
    backend.write_bytes(b"backend")
    server.chmod(0o755)
    backend.chmod(0o644)

    adapter = HyMtTranslator.load(
        FakeSource(model), server_path=server, backend_path=backend
    )
    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == visible
    adapter.close()


def test_hy_request_timeout_fails_closed_and_reaps_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import translator_sidecar.local.hy_mt as hy_mt

    model = tmp_path / "model.gguf"
    model.write_bytes(b"gguf")
    server = tmp_path / "llama-server"
    backend = tmp_path / "libggml-cuda.so"
    server.write_bytes(b"server")
    backend.write_bytes(b"backend")
    server.chmod(0o755)
    backend.chmod(0o644)
    source = FakeSource(model)
    process = FakeProcess()

    monkeypatch.setattr(hy_mt, "VerifiedModelSource", FakeSource)
    monkeypatch.setattr(hy_mt.subprocess, "Popen", lambda *_a, **_kw: process)

    post_count = 0

    def request(self, method, *_args, **_kwargs):
        nonlocal post_count
        if method == "GET":
            return {"status": "ok"}
        post_count += 1
        if post_count <= 2:
            return translated()
        raise TimeoutError("private source text")

    monkeypatch.setattr(HyMtTranslator, "_request_json", request)
    translator = HyMtTranslator.load(
        source, server_path=server, backend_path=backend, startup_timeout_s=1
    )
    with pytest.raises(
        LocalTranslationError, match="local MT inference failed"
    ) as error:
        translator.translate(
            "Привет.",
            source_language=Language.RU,
            target_language=Language.EN,
            mode=TranslationMode.QUALITY_FIRST,
        )
    assert "private source text" not in str(error.value)
    assert translator.unavailable
    assert process.terminated and process.waited
    assert source.lease.closed
    assert post_count == 3
    with pytest.raises(LocalTranslationError, match="unavailable"):
        translator.translate(
            "Again.",
            source_language=Language.EN,
            target_language=Language.RU,
            mode=TranslationMode.QUALITY_FIRST,
        )


def test_hy_startup_timeout_reaps_child_and_releases_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import translator_sidecar.local.hy_mt as hy_mt

    model = tmp_path / "model.gguf"
    model.write_bytes(b"gguf")
    server = tmp_path / "llama-server"
    backend = tmp_path / "libggml-cuda.so"
    server.write_bytes(b"server")
    backend.write_bytes(b"backend")
    server.chmod(0o755)
    backend.chmod(0o644)
    source = FakeSource(model)
    process = FakeProcess()
    monkeypatch.setattr(hy_mt, "VerifiedModelSource", FakeSource)
    monkeypatch.setattr(hy_mt.subprocess, "Popen", lambda *_a, **_kw: process)
    monkeypatch.setattr(
        HyMtTranslator,
        "_request_json",
        lambda self, *_a, **_kw: (_ for _ in ()).throw(TimeoutError()),
    )

    with pytest.raises(LocalTranslationError, match="could not be loaded"):
        HyMtTranslator.load(
            source, server_path=server, backend_path=backend, startup_timeout_s=0.01
        )
    assert process.terminated and process.waited
    assert source.lease.closed


@pytest.mark.parametrize(
    ("response", "backend_failed"),
    [
        ({}, True),
        ({"choices": []}, True),
        (
            {
                "choices": [
                    {"message": {"content": "translated"}, "finish_reason": "length"}
                ]
            },
            False,
        ),
        (
            {"choices": [{"message": {"content": "   "}, "finish_reason": "stop"}]},
            False,
        ),
    ],
)
def test_hy_rejects_malformed_or_incomplete_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    response: dict,
    backend_failed: bool,
) -> None:
    def respond(self, method, *_args, **_kwargs):
        if method == "GET":
            return {"status": "ok"}
        respond.posts += 1
        return response if respond.posts == 3 else translated()

    respond.posts = 0

    translator, process, lease = load_fake_adapter(tmp_path, monkeypatch, respond)
    with pytest.raises(LocalTranslationError):
        translator.translate(
            "Hello.",
            source_language=Language.EN,
            target_language=Language.RU,
            mode=TranslationMode.QUALITY_FIRST,
        )
    assert translator.unavailable is backend_failed
    assert process.waited is backend_failed
    assert lease.closed is backend_failed
    if not backend_failed:
        assert (
            translator.translate(
                "Hello again.",
                source_language=Language.EN,
                target_language=Language.RU,
                mode=TranslationMode.QUALITY_FIRST,
            )
            == "translated"
        )
        translator.close()


def test_hy_rejects_bad_input_before_sending_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def respond(self, method, *_args, **_kwargs):
        calls.append(method)
        return {"status": "ok"} if method == "GET" else translated()

    translator, _, _ = load_fake_adapter(tmp_path, monkeypatch, respond)
    for text, source, target in (
        (" ", Language.RU, Language.EN),
        ("Hello", Language.EN, Language.EN),
        ("x" * 4097, Language.EN, Language.RU),
    ):
        with pytest.raises(LocalTranslationError):
            translator.translate(
                text,
                source_language=source,
                target_language=target,
                mode=TranslationMode.QUALITY_FIRST,
            )
    assert calls == ["GET", "POST", "POST"]
    translator.close()


def test_hy_output_budget_avoids_old_128_token_truncation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    budgets: list[int] = []

    def respond(self, method, _path, body=None, **_kwargs):
        if method == "GET":
            return {"status": "ok"}
        budgets.append(body["max_tokens"])
        return translated()

    adapter, _, _ = load_fake_adapter(tmp_path, monkeypatch, respond)
    assert budgets == [256, 256]
    adapter.close()


def test_hy_request_rejection_preserves_healthy_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import translator_sidecar.local.hy_mt as hy_mt

    posts = 0

    def respond(self, method, *_args, **_kwargs):
        nonlocal posts
        if method == "GET":
            return {"status": "ok"}
        posts += 1
        if posts == 3:
            raise hy_mt._RequestRejected("local MT input was rejected")
        return translated()

    adapter, process, lease = load_fake_adapter(tmp_path, monkeypatch, respond)
    with pytest.raises(LocalTranslationError, match="rejected"):
        adapter.translate(
            "Long request.",
            source_language=Language.EN,
            target_language=Language.RU,
            mode=TranslationMode.QUALITY_FIRST,
        )
    assert not adapter.unavailable
    assert (
        adapter.translate(
            "Next request.",
            source_language=Language.EN,
            target_language=Language.RU,
            mode=TranslationMode.QUALITY_FIRST,
        )
        == "translated"
    )
    adapter.close()
    assert process.waited and lease.closed


def test_hy_http_input_rejection_is_request_scoped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import translator_sidecar.local.hy_mt as hy_mt

    class RejectedResponse:
        status = 413

        def read(self, _limit: int) -> bytes:
            return b"private oversized request"

    class FakeConnection:
        def __init__(self, *_args):
            pass

        def request(self, *_args, **_kwargs) -> None:
            pass

        def getresponse(self) -> RejectedResponse:
            return RejectedResponse()

        def close(self) -> None:
            pass

    monkeypatch.setattr(hy_mt, "_UnixHTTPConnection", FakeConnection)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"gguf")
    adapter = HyMtTranslator(
        process=FakeProcess(),
        lease=FakeLease(model),
        socket_dir=TemporaryDirectory(),
    )
    try:
        with pytest.raises(LocalTranslationError, match="rejected"):
            adapter._request_json("POST", "/v1/chat/completions", {}, timeout=1)
        assert not adapter.unavailable
    finally:
        adapter.close()


def test_hy_dead_child_is_unavailable_before_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def respond(self, method, *_args, **_kwargs):
        calls.append(method)
        return {"status": "ok"} if method == "GET" else translated()

    translator, process, _ = load_fake_adapter(tmp_path, monkeypatch, respond)
    process.returncode = 1
    assert translator.unavailable
    with pytest.raises(LocalTranslationError, match="unavailable"):
        translator.translate(
            "Hello.",
            source_language=Language.EN,
            target_language=Language.RU,
            mode=TranslationMode.QUALITY_FIRST,
        )
    assert calls == ["GET", "POST", "POST"]
    translator.close()


def test_hy_health_retries_loading_response_before_bootstrap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    health_attempts = 0

    def respond(self, method, *_args, **_kwargs):
        nonlocal health_attempts
        if method == "GET":
            health_attempts += 1
            if health_attempts == 1:
                raise LocalTranslationError("local MT inference failed")
            return {"status": "ok"}
        return translated()

    translator, _, _ = load_fake_adapter(tmp_path, monkeypatch, respond)
    assert health_attempts == 2
    translator.close()


def test_hy_constructor_failure_cannot_spawn_child_and_releases_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import translator_sidecar.local.hy_mt as hy_mt

    model = tmp_path / "model.gguf"
    model.write_bytes(b"gguf")
    server = tmp_path / "llama-server"
    backend = tmp_path / "libggml-cuda.so"
    server.write_bytes(b"server")
    backend.write_bytes(b"backend")
    server.chmod(0o755)
    backend.chmod(0o644)
    source = FakeSource(model)
    spawned = []
    monkeypatch.setattr(hy_mt, "VerifiedModelSource", FakeSource)
    monkeypatch.setattr(hy_mt.subprocess, "Popen", lambda *_a, **_kw: spawned.append(1))
    monkeypatch.setattr(
        HyMtTranslator,
        "__init__",
        lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("constructor failed")),
    )

    with pytest.raises(LocalTranslationError, match="could not be loaded"):
        HyMtTranslator.load(source, server_path=server, backend_path=backend)
    assert spawned == []
    assert source.lease.closed


def test_hy_bootstrap_requires_both_translation_directions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import translator_sidecar.local.hy_mt as hy_mt

    model = tmp_path / "model.gguf"
    model.write_bytes(b"gguf")
    server = tmp_path / "llama-server"
    backend = tmp_path / "libggml-cuda.so"
    server.write_bytes(b"server")
    backend.write_bytes(b"backend")
    server.chmod(0o755)
    backend.chmod(0o644)
    source = FakeSource(model)
    process = FakeProcess()
    prompts: list[str] = []

    def respond(self, method, _path, body=None, **_kwargs):
        if method == "GET":
            return {"status": "ok"}
        prompts.append(str(body))
        return translated() if len(prompts) == 1 else {"choices": []}

    monkeypatch.setattr(hy_mt, "VerifiedModelSource", FakeSource)
    monkeypatch.setattr(hy_mt.subprocess, "Popen", lambda *_a, **_kw: process)
    monkeypatch.setattr(HyMtTranslator, "_request_json", respond)

    with pytest.raises(LocalTranslationError, match="could not be loaded"):
        HyMtTranslator.load(
            source, server_path=server, backend_path=backend, startup_timeout_s=1
        )
    assert len(prompts) == 2
    assert "English" in prompts[0]
    assert "Russian" in prompts[1]
    assert process.waited and source.lease.closed

"""Offline Hy-MT2 adapter with a private, owned llama-server process."""

from __future__ import annotations

import http.client
import json
import os
import socket
import stat
import subprocess
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from translator_sidecar.provider_contract import Language, ModelState, TranslationMode

from .model_lease import VerifiedModelLease, VerifiedModelSource
from .mt import (
    LocalTranslationCleanupPending,
    LocalTranslationError,
    LocalTranslationRequestError,
)

_MODEL_FILE = "Hy-MT2-1.8B-Q4_K_M.gguf"
_SERVER = Path("/usr/local/lib/ollama/llama-server")
_CUDA_BACKEND = Path("/usr/local/lib/ollama/cuda_v12/libggml-cuda.so")
_MAX_SOURCE_CHARS = 4_096
_MAX_RESPONSE_BYTES = 65_536
_INFERENCE_TIMEOUT_S = 10.0
_SMOKE_CASES = (
    ("Проверка перевода.", Language.RU, Language.EN),
    ("Translation check.", Language.EN, Language.RU),
)


class _UnixHTTPConnection(http.client.HTTPConnection):
    def __init__(self, socket_path: Path, timeout: float) -> None:
        super().__init__("localhost", timeout=timeout)
        self._socket_path = socket_path

    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(str(self._socket_path))


class _RequestRejected(LocalTranslationRequestError):
    pass


def _server_command(server: Path, model_path: str, socket_path: Path) -> list[str]:
    return [
        str(server),
        "--model",
        model_path,
        "--host",
        str(socket_path),
        "--threads",
        "2",
        "--threads-batch",
        "2",
        "--parallel",
        "1",
        "--ctx-size",
        "2048",
        "--gpu-layers",
        "99",
        "--jinja",
        "--offline",
        "--no-ui",
        "--log-disable",
    ]


def _trusted_local_binary(path: Path, *, executable: bool) -> bool:
    try:
        entry = path.lstat()
    except OSError:
        return False
    return (
        stat.S_ISREG(entry.st_mode)
        and not entry.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
        and (not executable or bool(entry.st_mode & stat.S_IXUSR))
    )


class HyMtTranslator:
    def __init__(
        self,
        *,
        process: subprocess.Popen[bytes] | None,
        lease: VerifiedModelLease,
        socket_dir: TemporaryDirectory[str],
    ) -> None:
        self._process = process
        self._lease: VerifiedModelLease | None = lease
        self._socket_dir: TemporaryDirectory[str] | None = socket_dir
        self._socket_path = Path(socket_dir.name) / "translator.sock"
        self._closed = False
        self.actual_device = "cuda"

    @property
    def unavailable(self) -> bool:
        return self._closed or self._process is None or self._process.poll() is not None

    @property
    def model_state(self) -> ModelState:
        return ModelState.FAILED if self.unavailable else ModelState.READY

    @classmethod
    def load(
        cls,
        source: VerifiedModelSource,
        *,
        server_path: Path = _SERVER,
        backend_path: Path = _CUDA_BACKEND,
        startup_timeout_s: float = 30.0,
    ) -> HyMtTranslator:
        if not isinstance(source, VerifiedModelSource):
            raise LocalTranslationError("local MT model path is unavailable")
        if not _trusted_local_binary(server_path, executable=True) or not (
            _trusted_local_binary(backend_path, executable=False)
        ):
            raise LocalTranslationError("local MT runtime could not be loaded")
        lease = None
        socket_dir = None
        adapter = None
        try:
            lease = source.acquire()
            model_path = lease.path(_MODEL_FILE)
            descriptor = lease.descriptor(_MODEL_FILE)
            socket_dir = TemporaryDirectory(prefix="translator-hy-mt-")
            socket_path = Path(socket_dir.name) / "translator.sock"
            adapter = cls(process=None, lease=lease, socket_dir=socket_dir)
            environment = {
                "PATH": "/usr/bin:/bin",
                "LANG": "C.UTF-8",
                "GGML_BACKEND_PATH": str(backend_path),
                "HF_HUB_OFFLINE": "1",
            }
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                environment["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]
            adapter._process = subprocess.Popen(
                _server_command(server_path, model_path, socket_path),
                pass_fds=(descriptor,),
                close_fds=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=environment,
            )
            adapter._wait_ready(startup_timeout_s)
            for text, source_language, target_language in _SMOKE_CASES:
                adapter.translate(
                    text,
                    source_language=source_language,
                    target_language=target_language,
                    mode=TranslationMode.QUALITY_FIRST,
                )
            return adapter
        except BaseException as error:
            if adapter is not None:
                try:
                    adapter.close()
                except Exception:
                    raise LocalTranslationCleanupPending(adapter) from None
            else:
                if socket_dir is not None:
                    socket_dir.cleanup()
                if lease is not None:
                    lease.close()
            if not isinstance(error, Exception):
                raise
            raise LocalTranslationError(
                "local MT runtime could not be loaded"
            ) from None

    def _wait_ready(self, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.unavailable:
                break
            try:
                result = self._request_json("GET", "/health", timeout=0.25)
                if result.get("status") == "ok":
                    return
            except (
                OSError,
                TimeoutError,
                ValueError,
                http.client.HTTPException,
                LocalTranslationError,
            ):
                pass
            time.sleep(0.05)
        raise LocalTranslationError("local MT runtime could not be loaded")

    def _request_json(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        *,
        timeout: float,
    ) -> dict[str, Any]:
        connection = _UnixHTTPConnection(self._socket_path, timeout)
        try:
            payload = (
                json.dumps(body, ensure_ascii=False).encode("utf-8") if body else None
            )
            connection.request(
                method,
                path,
                body=payload,
                headers={"Content-Type": "application/json"} if payload else {},
            )
            response = connection.getresponse()
            content = response.read(_MAX_RESPONSE_BYTES + 1)
            if response.status in {400, 413, 422}:
                raise _RequestRejected("local MT input was rejected")
            if response.status != 200 or len(content) > _MAX_RESPONSE_BYTES:
                raise LocalTranslationError("local MT inference failed")
            parsed = json.loads(content)
            if not isinstance(parsed, dict):
                raise LocalTranslationError("local MT inference failed")
            return parsed
        finally:
            connection.close()

    def translate(
        self,
        text: str,
        *,
        source_language: Language,
        target_language: Language,
        mode: TranslationMode,
    ) -> str:
        del mode
        if self.unavailable:
            raise LocalTranslationError("local MT is unavailable")
        normalized = text.strip()
        if not normalized:
            raise LocalTranslationRequestError("source text is empty")
        if source_language is target_language:
            raise LocalTranslationRequestError("language pair is not supported")
        if len(normalized) > _MAX_SOURCE_CHARS:
            raise LocalTranslationRequestError("local MT input exceeds work limit")
        target_name = "English" if target_language is Language.EN else "Russian"
        prompt = (
            f"Translate the following text into {target_name}. Note that you "
            "should only output the translated result without any additional "
            f"explanation:\n{normalized}"
        )
        body = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "top_p": 0.6,
            "top_k": 20,
            "repeat_penalty": 1.05,
            "max_tokens": 256,
            "stream": False,
        }
        try:
            response = self._request_json(
                "POST", "/v1/chat/completions", body, timeout=_INFERENCE_TIMEOUT_S
            )
            choice = response["choices"][0]
            translated = choice["message"]["content"].strip()
            finish_reason = choice["finish_reason"]
        except _RequestRejected:
            raise
        except Exception:
            try:
                self.close()
            except Exception:
                pass
            raise LocalTranslationError("local MT inference failed") from None
        if finish_reason == "length":
            raise LocalTranslationRequestError("local MT output token limit reached")
        if finish_reason != "stop" or not translated:
            raise LocalTranslationRequestError("local MT output is incomplete")
        return translated

    def close(self) -> None:
        self._closed = True
        process = self._process
        if process is not None:
            try:
                if process.poll() is None:
                    process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            except Exception:
                raise LocalTranslationError("local MT cleanup is incomplete") from None
            self._process = None
        if self._socket_dir is not None:
            self._socket_dir.cleanup()
            self._socket_dir = None
        if self._lease is not None:
            self._lease.close()
            self._lease = None

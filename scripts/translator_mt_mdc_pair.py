"""Compare one local MT backend against frozen Turbo MDC transcripts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import time
from pathlib import Path
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

import psutil
from translator_sidecar.local.model_lease import VerifiedModelSource
from translator_sidecar.local.model_manifest import load_manifest
from translator_sidecar.local.mt import NllbTranslator
from translator_sidecar.provider_contract import Language, TranslationMode

ROOT = Path(__file__).resolve().parents[1]
SCREEN_SHA256 = "2067c0ef209f9f2d895f09ba0b3ac8cb20bcc1615e6852fa344d8239cc2839a2"
SERVER = "http://127.0.0.1:11578/v1/chat/completions"
HY_GGUF_SHA256 = "dc5f44fcf1fa496ee7ad725982c0c8c553a4de00259b53af84c4b89fb0c06699"
HY_SERVER = Path("/usr/local/lib/ollama/llama-server")


class RejectRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        raise ValueError("local MT server redirect refused")


OPENER = build_opener(ProxyHandler({}), RejectRedirect())


def sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def validate_server(pid: int, started: float) -> None:
    process = psutil.Process(pid)
    if process.create_time() != started or not process.is_running():
        raise RuntimeError("pinned local MT server is unavailable")
    if not any(
        connection.status == psutil.CONN_LISTEN
        and connection.laddr.ip == "127.0.0.1"
        and connection.laddr.port == 11578
        for connection in process.net_connections(kind="tcp")
    ):
        raise RuntimeError("pinned local MT listener is unavailable")


def validate_server_command(command: list[str], model: Path, layers: int) -> None:
    if layers < 0:
        raise ValueError("Hy GPU layers must be nonnegative")
    expected = [
        str(HY_SERVER),
        "--model",
        str(model),
        "--host",
        "127.0.0.1",
        "--port",
        "11578",
        "--threads",
        "2",
        "--threads-batch",
        "2",
        "--parallel",
        "1",
        "--ctx-size",
        "2048",
        "--gpu-layers",
        str(layers),
        "--jinja",
        "--log-disable",
    ]
    if command != expected:
        raise ValueError("local MT server identity does not match pinned command")


def hy_translate(text: str, target: Language, pid: int, started: float) -> str:
    validate_server(pid, started)
    target_name = "English" if target is Language.EN else "Russian"
    prompt = (
        f"Translate the following text into {target_name}. Note that you "
        "should only output the translated result without any additional "
        f"explanation:\n{text}"
    )
    body = json.dumps(
        {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "top_p": 0.6,
            "top_k": 20,
            "repeat_penalty": 1.05,
            "max_tokens": 128,
            "stream": False,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = Request(SERVER, data=body, headers={"Content-Type": "application/json"})
    with OPENER.open(request, timeout=60) as response:
        choice = json.load(response)["choices"][0]
    output = choice["message"]["content"].strip()
    if not output or choice.get("finish_reason") != "stop":
        raise RuntimeError("Hy-MT2 returned an incomplete translation")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--screen-sha256", default=SCREEN_SHA256)
    parser.add_argument("--case-count", type=int, default=12)
    parser.add_argument("--turbo", type=Path, required=True)
    parser.add_argument("--backend", choices=("nllb", "hy_mt2"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hy-model", type=Path)
    parser.add_argument("--hy-server-pid", type=int)
    parser.add_argument("--hy-gpu-layers", type=int, default=0)
    args = parser.parse_args()

    if args.hy_gpu_layers < 0:
        raise ValueError("Hy GPU layers must be nonnegative")
    if args.backend == "nllb" and args.hy_gpu_layers != 0:
        raise ValueError("Hy GPU layers are not valid for NLLB")
    if (
        not args.output.is_absolute()
        or args.output.resolve().is_relative_to(ROOT)
        or any(parent.is_symlink() for parent in args.output.parents)
        or not args.output.parent.is_dir()
        or stat.S_IMODE(args.output.parent.stat().st_mode) != 0o700
        or args.output.exists()
    ):
        raise ValueError("output must be a new file in a private directory outside Git")
    screen_bytes = args.screen.read_bytes()
    if hashlib.sha256(screen_bytes).hexdigest() != args.screen_sha256:
        raise ValueError("selected screen identity changed")
    screen = json.loads(screen_bytes)
    turbo_bytes = args.turbo.read_bytes()
    if hashlib.sha256(turbo_bytes).hexdigest() != screen["turbo_report_sha256"]:
        raise ValueError("Turbo report identity changed")
    report = json.loads(turbo_bytes)
    selected = {(case["origin_id"], case["condition"]) for case in screen["cases"]}
    cases = [
        case
        for case in report["results"]
        if (case["origin_id"], case["condition"]) in selected
    ]
    if (
        args.case_count < 1
        or len(cases) != len(selected)
        or len(selected) != args.case_count
    ):
        raise ValueError("selected cases are missing or duplicated")
    if any(case["status"] != "completed" or not case["transcript"] for case in cases):
        raise ValueError("selected Turbo transcript is unavailable")

    pid: int | None = None
    server_started: float | None = None
    command: list[str] | None = None
    server_affinity: list[int] | None = None
    if args.backend == "hy_mt2":
        model = args.hy_model
        pid = args.hy_server_pid
        if (
            model is None
            or pid is None
            or not model.is_absolute()
            or model.is_symlink()
        ):
            raise ValueError("explicit Hy model and server PID are required")
        if sha256(model) != HY_GGUF_SHA256:
            raise ValueError("Hy GGUF identity changed")
        process = psutil.Process(pid)
        command = process.cmdline()
        if (
            process.uids().real != os.getuid()
            or Path(process.exe()).resolve() != HY_SERVER.resolve()
        ):
            raise ValueError("local MT server identity does not match pinned model")
        validate_server_command(command, model, args.hy_gpu_layers)
        server_affinity = process.cpu_affinity()
        server_started = process.create_time()
        validate_server(pid, server_started)
    elif args.hy_model is not None or args.hy_server_pid is not None:
        raise ValueError("Hy server options are not valid for NLLB")

    manifest_path = ROOT / "models/manifest.json"
    translator = None
    if args.backend == "nllb":
        translator = NllbTranslator.load(
            VerifiedModelSource(
                load_manifest(manifest_path), "nllb-200-distilled-600m-ct2-int8"
            ),
            device="cpu",
        )
    rows = []
    try:
        for source, warmup in (
            (Language.RU, "Проверьте звук."),
            (Language.EN, "Check the audio."),
        ):
            target = Language.EN if source is Language.RU else Language.RU
            if translator is None:
                hy_translate(warmup, target, pid, server_started)
            else:
                translator.translate(
                    warmup,
                    source_language=source,
                    target_language=target,
                    mode=TranslationMode.QUALITY_FIRST,
                )
        for case in cases:
            source = Language.RU if case["language"] == "ru_ru" else Language.EN
            target = Language.EN if source is Language.RU else Language.RU
            started = time.monotonic_ns()
            if translator is None:
                output = hy_translate(case["transcript"], target, pid, server_started)
                finish_reason = "stop"
            else:
                output = translator.translate(
                    case["transcript"],
                    source_language=source,
                    target_language=target,
                    mode=TranslationMode.QUALITY_FIRST,
                )
                finish_reason = None
            rows.append(
                {
                    "origin_id": case["origin_id"],
                    "condition": case["condition"],
                    "source_language": source.value,
                    "source": case["transcript"],
                    "output": output,
                    "latency_ms": round((time.monotonic_ns() - started) / 1e6, 2),
                    "finish_reason": finish_reason,
                }
            )
    finally:
        if translator is not None:
            translator.close()

    evidence = {
        "scope": "selected text-only diagnostic; EN source audio not human verified",
        "backend": args.backend,
        "screen_sha256": args.screen_sha256,
        "turbo_report_sha256": screen["turbo_report_sha256"],
        "manifest_sha256": sha256(manifest_path),
        "source_head": report["source_head"],
        "source_head_role": "Turbo ASR report provenance, not this runtime checkout",
        "runner_sha256": sha256(Path(__file__)),
        "runtime_code_sha256": {
            name: sha256(ROOT / name)
            for name in (
                "sidecar/translator_sidecar/local/mt.py",
                "sidecar/translator_sidecar/local/model_lease.py",
                "sidecar/translator_sidecar/local/model_manifest.py",
                "sidecar/translator_sidecar/provider_contract.py",
            )
        },
        "runner_cpu_affinity": sorted(os.sched_getaffinity(0)),
        "hy_gguf_sha256": HY_GGUF_SHA256 if server_started is not None else None,
        "hy_server_pid": pid if server_started is not None else None,
        "hy_server_binary_sha256": sha256(HY_SERVER)
        if server_started is not None
        else None,
        "hy_server_command": command,
        "hy_server_cpu_affinity": server_affinity,
        "requested_hy_gpu_layers": args.hy_gpu_layers
        if server_started is not None
        else None,
        "decoding": "QUALITY_FIRST"
        if translator is not None
        else "temperature=0 top_p=0.6 top_k=20 repeat_penalty=1.05 max_tokens=128",
        "cases": rows,
    }
    old_umask = os.umask(0o077)
    try:
        with args.output.open("x", encoding="utf-8") as file:
            json.dump(evidence, file, ensure_ascii=False, indent=2)
            file.write("\n")
    finally:
        os.umask(old_umask)
    print(json.dumps({"backend": args.backend, "count": len(rows)}))


if __name__ == "__main__":
    main()

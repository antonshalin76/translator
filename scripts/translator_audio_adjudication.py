"""Private, diagnostic audio-first check of frozen critical speech cases."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import stat
import time
import unicodedata
import urllib.request
from collections.abc import Callable
from pathlib import Path

from translator_mdc_asr_run import sha256, validate_manifest

REPO = Path(__file__).resolve().parents[1]
OLLAMA = "http://127.0.0.1:11434"
MODEL_BLOBS = Path("/usr/share/ollama/.ollama/models/blobs")
MAX_CASES = 12
MAX_TOKENS = 48
REQUEST_TIMEOUT_S = 120

Post = Callable[[str, dict | None], dict]


class RejectRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        raise ValueError("local Ollama redirect refused")


def private_path(path: Path, *, existing: bool) -> None:
    if not path.is_absolute() or path.resolve().is_relative_to(REPO):
        raise ValueError("private evidence path must be absolute and outside Git")
    if any(parent.is_symlink() for parent in (path, *path.parents)):
        raise ValueError("private evidence path contains a symlink")
    if not path.parent.is_dir() or stat.S_IMODE(path.parent.stat().st_mode) & 0o077:
        raise ValueError("private evidence directory must be mode 0700")
    if existing:
        if not path.is_file() or stat.S_IMODE(path.stat().st_mode) & 0o077:
            raise ValueError("private evidence file must be owner-only")
    elif path.exists():
        raise FileExistsError(path)


def normalize(value: str) -> str:
    return " ".join(
        "".join(
            " " if unicodedata.category(char).startswith("P") else char
            for char in value.casefold().replace("ё", "е")
        ).split()
    )


def post_json(path: str, body: dict | None) -> dict:
    if path not in ("/api/tags", "/api/show", "/v1/chat/completions"):
        raise ValueError("unsupported local Ollama path")
    request = urllib.request.Request(
        OLLAMA + path,
        data=None if body is None else json.dumps(body).encode("utf-8"),
        headers={} if body is None else {"Content-Type": "application/json"},
        method="GET" if body is None else "POST",
    )
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}), RejectRedirect()
    )
    with opener.open(request, timeout=REQUEST_TIMEOUT_S) as response:
        return json.load(response)


def validate_model(model: str, digest: str, blobs: Path, post: Post) -> None:
    tags = post("/api/tags", None)["models"]
    matches = [item for item in tags if item["name"] == model]
    if (
        len(matches) != 1
        or matches[0]["digest"] != digest
        or matches[0]["size"] < 1_000_000_000
    ):
        raise ValueError("pinned local model identity unavailable")
    shown = post("/api/show", {"model": model})
    if shown["details"]["format"] != "gguf" or "audio" not in shown["capabilities"]:
        raise ValueError("pinned model does not expose local GGUF audio")
    from_lines = [
        line[5:].strip()
        for line in shown["modelfile"].splitlines()
        if line.startswith("FROM ")
    ]
    if len(from_lines) != 1:
        raise ValueError("local model blob identity unavailable")
    blob = Path(from_lines[0])
    if (
        not blob.is_absolute()
        or blob.is_symlink()
        or blob.parent.resolve() != blobs.resolve()
        or not blob.name.startswith("sha256-")
        or not blob.is_file()
        or not shown.get("tensors")
    ):
        raise ValueError("model is not backed by the pinned local blob store")


def validate_inputs(
    manifest: Path, screen: Path, spec: Path, expected_spec_hash: str, output: Path
) -> list[tuple[dict, dict, dict]]:
    private_path(spec, existing=True)
    private_path(output, existing=False)
    if sha256(spec) != expected_spec_hash:
        raise ValueError("frozen case spec changed")
    plan = json.loads(spec.read_text(encoding="utf-8"))
    if plan["schema"] != 1 or plan["oracle_source"] != "written_reference_hypothesis":
        raise ValueError("unsupported diagnostic case spec")
    if (
        sha256(manifest) != plan["manifest_sha256"]
        or sha256(screen) != plan["screen_sha256"]
    ):
        raise ValueError("frozen corpus or selection changed")
    samples = validate_manifest(manifest, plan["manifest_sha256"])
    by_key = {(item["origin_id"], item["condition"]): item for item in samples}
    by_file = {item["audio_file"]: item for item in samples}
    if len(by_key) != len(samples):
        raise ValueError("duplicate manifest case")
    selected = json.loads(screen.read_text(encoding="utf-8"))["cases"]
    expected = [(item["origin_id"], item["condition"]) for item in selected]
    cases = plan["cases"]
    actual = [(item["origin_id"], item["condition"]) for item in cases]
    if (
        not actual
        or len(actual) > MAX_CASES
        or len(set(actual)) != len(actual)
        or set(actual) != set(expected)
    ):
        raise ValueError("missing, extra, or duplicate critical case")
    resolved = []
    for case in cases:
        target = by_key[case["origin_id"], case["condition"]]
        contrast = by_file[case["contrast_audio_file"]]
        if (
            contrast["language"] != target["language"]
            or contrast["origin_id"] == target["origin_id"]
        ):
            raise ValueError("contrast must be a different same-language utterance")
        question = case["question"]
        aliases = case["expected_aliases"]
        if (
            not isinstance(question, str)
            or not question.strip()
            or not isinstance(aliases, list)
            or not aliases
            or any(
                not isinstance(alias, str) or not normalize(alias) for alias in aliases
            )
            or any(normalize(alias) in normalize(question) for alias in aliases)
        ):
            raise ValueError("invalid or answer-bearing diagnostic question")
        for sample in (target, contrast):
            audio = manifest.parent / sample["audio_file"]
            if audio.is_symlink() or sha256(audio) != sample["sha256"]:
                raise ValueError("frozen audio changed")
        resolved.append((case, target, contrast))
    return resolved


def prompt(question: str) -> str:
    return (
        "Use only the attached speech, not outside knowledge. If no audio is attached, "
        "reply exactly NO_AUDIO. If the fact is not explicitly spoken, reply exactly "
        "NOT_STATED. If uncertain, reply exactly UNSURE. Otherwise answer briefly "
        "in the language of the question. Question: " + question
    )


def ask(
    model: str, question: str, audio: bytes | None, post: Post
) -> tuple[str, int | None]:
    content = [{"type": "text", "text": prompt(question)}]
    if audio is not None:
        content.append(
            {
                "type": "input_audio",
                "input_audio": {
                    "data": base64.b64encode(audio).decode("ascii"),
                    "format": "wav",
                },
            }
        )
    response = post(
        "/v1/chat/completions",
        {
            "model": model,
            "temperature": 0,
            "max_tokens": MAX_TOKENS,
            "reasoning_effort": "none",
            "messages": [{"role": "user", "content": content}],
        },
    )
    answer = response["choices"][0]["message"]["content"]
    count = response.get("usage", {}).get("prompt_tokens")
    return answer.strip(), count if isinstance(count, int) and count > 0 else None


def verified_audio(manifest: Path, sample: dict) -> bytes:
    audio = (manifest.parent / sample["audio_file"]).read_bytes()
    if hashlib.sha256(audio).hexdigest() != sample["sha256"]:
        raise ValueError("frozen audio changed after preflight")
    return audio


def run(
    manifest: Path,
    screen: Path,
    spec: Path,
    spec_hash: str,
    model: str,
    model_digest: str,
    output: Path,
    blobs: Path = MODEL_BLOBS,
    post: Post = post_json,
) -> list[dict]:
    cases = validate_inputs(manifest, screen, spec, spec_hash, output)
    validate_model(model, model_digest, blobs, post)
    results = []
    descriptor = os.open(
        output, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        for case, target, contrast in cases:
            result = {
                "origin_id": case["origin_id"],
                "condition": case["condition"],
                "target_audio_sha256": target["sha256"],
                "contrast_audio_sha256": contrast["sha256"],
                "spec_sha256": spec_hash,
                "manifest_sha256": sha256(manifest),
                "screen_sha256": sha256(screen),
                "model": model,
                "model_digest": model_digest,
                "decision": "UNRESOLVED",
                "attempt_status": "COMPLETED",
            }
            started = time.monotonic_ns()
            try:
                control, control_tokens = ask(model, case["question"], None, post)
                result["no_audio"] = {
                    "answer": control,
                    "prompt_tokens": control_tokens,
                }
                if control not in ("NO_AUDIO", "NOT_STATED") or control_tokens is None:
                    result["reason"] = "no_audio_control_failed"
                else:
                    other, other_tokens = ask(
                        model,
                        case["question"],
                        verified_audio(manifest, contrast),
                        post,
                    )
                    result["contrast"] = {
                        "answer": other,
                        "prompt_tokens": other_tokens,
                    }
                    if (
                        other != "NOT_STATED"
                        or other_tokens is None
                        or other_tokens <= control_tokens
                    ):
                        result["reason"] = "contrast_control_failed"
                    else:
                        answer, target_tokens = ask(
                            model,
                            case["question"],
                            verified_audio(manifest, target),
                            post,
                        )
                        result["target"] = {
                            "answer": answer,
                            "prompt_tokens": target_tokens,
                        }
                        if (
                            target_tokens is None
                            or target_tokens <= control_tokens
                            or answer in ("NO_AUDIO", "NOT_STATED", "UNSURE")
                        ):
                            result["reason"] = "target_audio_not_established"
                        elif normalize(answer) in {
                            normalize(alias) for alias in case["expected_aliases"]
                        }:
                            result["decision"] = "AGREES_WITH_WRITTEN_REFERENCE"
                        else:
                            result["reason"] = "answer_does_not_match_expected_alias"
            except Exception as error:  # noqa: BLE001 - retain a failed attempt without a retry
                result.update(attempt_status="ERROR", error_type=type(error).__name__)
            result["elapsed_ms"] = (time.monotonic_ns() - started) / 1e6
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            results.append(result)
    return results


def finish(rows: list[dict]) -> int:
    errors = sum(row["attempt_status"] == "ERROR" for row in rows)
    print(
        f"attempts={len(rows)} diagnostic_agreements="
        f"{sum(row['decision'] == 'AGREES_WITH_WRITTEN_REFERENCE' for row in rows)} "
        f"unresolved={sum(row['decision'] == 'UNRESOLVED' for row in rows)} "
        f"errors={errors}"
    )
    return int(errors > 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--spec-sha256", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-digest", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = run(
        args.manifest,
        args.screen,
        args.spec,
        args.spec_sha256,
        args.model,
        args.model_digest,
        args.output,
    )
    raise SystemExit(finish(rows))

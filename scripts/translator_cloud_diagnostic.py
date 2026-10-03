"""Bounded cloud diagnosis of frozen Translator speech and translation receipts.

This is not a product runtime, an audio ground-truth oracle, or a release gate.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import re
import time
import uuid
import wave
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from translator_audio_adjudication import private_path
from translator_mdc_asr_run import sha256, validate_manifest
from translator_mdc_asr_score import errors

ARMS = (
    "original_main_small_nllb_piper",
    "small_nllb_same_code",
    "nllb",
    "hy",
)
MAX_CASES = 24
MAX_WAV_BYTES = 5_000_000
OPENAI_MODEL = "gpt-transcribe"
GOOGLE_MODEL = "gemini-3.5-transcribe"
CLAUDE_MODEL = "claude-sonnet-5-5"
ERROR_CATEGORIES = ("negation", "number", "name", "omission", "other")
EXPECTED_MODELS = {
    "original_main_small_nllb_piper": (
        "small",
        "nllb-200-distilled-600m-ct2-int8",
        "piper-medium",
    ),
    "small_nllb_same_code": (
        "faster-whisper-small",
        "nllb-200-distilled-600m-ct2-int8",
        "piper-medium",
    ),
    "nllb": (
        "faster-whisper-large-v3-turbo",
        "nllb-200-distilled-600m-ct2-int8",
        "piper-medium",
    ),
    "hy": ("faster-whisper-large-v3-turbo", "hy-mt2-1.8b-gguf-q4-k-m", "piper-medium"),
}

Transcriber = Callable[[bytes, str], dict]
Judge = Callable[[str, dict], dict]


class RemoteCleanupUncertain(Exception):
    def __init__(self, remote_identity: str):
        super().__init__("remote upload cleanup unverified")
        self.remote_identity = remote_identity


def verified_model_identity(row: dict, arm: str) -> bool:
    expected = EXPECTED_MODELS[arm]
    for boundary in ("effective_models_open", "effective_models_after"):
        observed = row.get(boundary, {})
        if arm == ARMS[0]:
            if observed.get("provider_state") != "ready":
                return False
            observed = observed.get("models", {})
        for component, model_id, device in zip(
            ("asr", "mt", "tts"), expected, ("cuda", "cuda", "cpu")
        ):
            model = observed.get(component, {})
            if (model.get("id"), model.get("state"), model.get("device")) != (
                model_id,
                "ready",
                device,
            ):
                return False
    return True


def receipt_rows(path: Path, expected_hash: str) -> list[dict]:
    if sha256(path) != expected_hash:
        raise ValueError("frozen receipt hash changed")
    return [
        row
        for line in path.read_text(encoding="utf-8").splitlines()
        if (row := json.loads(line)).get("type") == "attempt"
    ]


def load_cases(
    manifest: Path,
    screen: Path,
    original: Path,
    candidate: Path,
    expected_hashes: tuple[str, str, str, str],
) -> list[dict]:
    """Bind the complete 24-case, four-arm comparison before any network call."""
    manifest_hash, screen_hash, original_hash, candidate_hash = expected_hashes
    samples = validate_manifest(manifest, manifest_hash)
    if sha256(screen) != screen_hash:
        raise ValueError("frozen screen hash changed")
    selected = json.loads(screen.read_text(encoding="utf-8"))["cases"]
    keys = [(row["origin_id"], row["condition"]) for row in selected]
    if len(keys) != MAX_CASES or len(set(keys)) != MAX_CASES:
        raise ValueError("screen must contain 24 distinct cases")
    by_key = {(row["origin_id"], row["condition"]): row for row in samples}
    if len(by_key) != len(samples) or not set(keys) <= set(by_key):
        raise ValueError("screen does not match corpus")
    reports = receipt_rows(original, original_hash) + receipt_rows(
        candidate, candidate_hash
    )
    indexed: dict[tuple[str, str, str], dict] = {}
    for row in reports:
        key = (row["origin_id"], row["condition"], row["backend"])
        if key in indexed:
            raise ValueError("duplicate receipt attempt")
        indexed[key] = row
    expected = {(origin, condition, arm) for origin, condition in keys for arm in ARMS}
    if set(indexed) != expected:
        raise ValueError("receipt arms or cases do not match screen")
    cases = []
    for origin, condition in keys:
        sample = by_key[origin, condition]
        arms = {arm: indexed[origin, condition, arm] for arm in ARMS}
        baseline = arms[ARMS[0]]
        for arm, row in arms.items():
            if not verified_model_identity(row, arm):
                raise ValueError(f"model identity mismatch for {origin}/{arm}")
            if (
                row.get("status") != "completed"
                or row.get("reference") != sample["reference"]
                or row.get("language") != sample["language"]
                or row.get("speaker_id") != sample["speaker_id"]
                or row.get("wav_sha256") != sample["sha256"]
                or row.get("critical_labels") != sample.get("critical_labels", [])
                or row.get("mode") != baseline["mode"]
                or row.get("mode") != "quality_first"
                or row.get("voice_gender") != baseline["voice_gender"]
                or row.get("voice_gender") != "female"
                or not isinstance(row.get("asr_text"), str)
                or not isinstance(row.get("mt_text"), str)
            ):
                raise ValueError(f"receipt mismatch for {origin}/{condition}/{arm}")
        audio = manifest.parent / sample["audio_file"]
        if audio.is_symlink() or audio.stat().st_size > MAX_WAV_BYTES:
            raise ValueError("selected WAV is unsafe or too large")
        cases.append({"sample": sample, "audio": audio, "arms": arms})
    return cases


def blind_mapping(index: int) -> dict[str, str]:
    """Each arm occupies each prompt position six times over 24 cases."""
    arms = list(ARMS)
    random.Random(20261001).shuffle(arms)
    return {
        label: arms[(slot + index) % len(arms)] for slot, label in enumerate("ABCD")
    }


def verified_audio(case: dict) -> bytes:
    path = case["audio"]
    if path.is_symlink():
        raise ValueError("audio path became a symlink")
    audio = path.read_bytes()
    if (
        len(audio) > MAX_WAV_BYTES
        or hashlib.sha256(audio).hexdigest() != case["sample"]["sha256"]
    ):
        raise ValueError("audio changed before send")
    return audio


def validate_judgment(value: dict) -> dict:
    if not isinstance(value, dict) or set(value) != set("ABCD"):
        raise ValueError("judge labels missing or duplicated")
    for label in "ABCD":
        if not isinstance(value[label], dict) or set(value[label]) != {
            "reference",
            "own_asr",
        }:
            raise ValueError("judge basis missing")
        for basis in ("reference", "own_asr"):
            verdict = value[label][basis]
            if (
                not isinstance(verdict, dict)
                or verdict.get("verdict") not in {"PASS", "FAIL", "UNCERTAIN"}
                or not isinstance(verdict.get("critical_errors"), list)
                or any(
                    item not in ERROR_CATEGORIES for item in verdict["critical_errors"]
                )
            ):
                raise ValueError("judge verdict malformed")
    return value


def run_cases(
    cases: list[dict],
    output: Path,
    openai: Transcriber,
    google: Transcriber,
    judge: Judge,
    *,
    limit: int = MAX_CASES,
    control_status: dict[str, str] | None = None,
    provenance: dict | None = None,
) -> list[dict]:
    private_path(output, existing=False)
    if not 1 <= limit <= MAX_CASES or len(cases) != MAX_CASES:
        raise ValueError("unbounded or incomplete case set")
    descriptor = os.open(
        output, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600
    )
    results = []
    remote_cleanup_blocked = False
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        for index, case in enumerate(cases[:limit]):
            sample = case["sample"]
            mapping = blind_mapping(index)
            row = {
                "type": "diagnostic_case",
                "origin_id": sample["origin_id"],
                "condition": sample["condition"],
                "language": sample["language"],
                "speaker_id": sample["speaker_id"],
                "critical_labels": sample.get("critical_labels", []),
                "wav_sha256": sample["sha256"],
                "reference": sample["reference"],
                "reference_provenance": "MDC_written_not_acoustically_verified",
                "control_status": control_status or {},
                "provenance": provenance or {},
                "judge_model": CLAUDE_MODEL,
                "judge_rubric": "critical-facts-v1",
                "mapping": mapping,
                "arms": {
                    arm: {
                        "asr_text": case["arms"][arm]["asr_text"],
                        "mt_text": case["arms"][arm]["mt_text"],
                    }
                    for arm in ARMS
                },
                "status": "INCOMPLETE",
            }
            if remote_cleanup_blocked:
                row["status"] = "BLOCKED_REMOTE_CLEANUP"
                for provider in ("openai", "google", "claude"):
                    row[provider] = {"status": "NOT_RUN"}
            else:
                try:
                    audio = verified_audio(case)
                except (OSError, ValueError):
                    row["status"] = "AUDIO_CHANGED"
                    for provider in ("openai", "google", "claude"):
                        row[provider] = {"status": "NOT_RUN"}
                else:
                    for name, transcribe in (("openai", openai), ("google", google)):
                        started = time.monotonic_ns()
                        try:
                            result = transcribe(audio, sample["language"])
                            if (
                                not isinstance(result.get("text"), str)
                                or not result["text"].strip()
                            ):
                                raise ValueError("empty or malformed transcript")
                            row[name] = {"status": "COMPLETED", **result}
                        except RemoteCleanupUncertain as error:
                            row[name] = {
                                "status": "ERROR",
                                "error_type": type(error).__name__,
                                "remote_identity": error.remote_identity,
                            }
                            remote_cleanup_blocked = True
                        except Exception as error:  # noqa: BLE001 - preserve each failed attempt
                            row[name] = {
                                "status": "ERROR",
                                "error_type": type(error).__name__,
                            }
                        row[name]["elapsed_ms"] = (time.monotonic_ns() - started) / 1e6
                    variants = {
                        label: {
                            "translation": case["arms"][arm]["mt_text"],
                            "own_asr": case["arms"][arm]["asr_text"],
                        }
                        for label, arm in mapping.items()
                    }
                    started = time.monotonic_ns()
                    if remote_cleanup_blocked:
                        row["claude"] = {"status": "NOT_RUN"}
                    else:
                        try:
                            verdicts = validate_judgment(
                                judge(sample["reference"], variants)
                            )
                            row["claude"] = {
                                "status": "COMPLETED",
                                "verdicts": verdicts,
                            }
                        except Exception as error:  # noqa: BLE001 - no retry or raw error logging
                            row["claude"] = {
                                "status": "ERROR",
                                "error_type": type(error).__name__,
                            }
                        row["claude"]["elapsed_ms"] = (
                            time.monotonic_ns() - started
                        ) / 1e6
                    if all(
                        row[name]["status"] == "COMPLETED"
                        for name in ("openai", "google", "claude")
                    ):
                        row["status"] = "COMPLETED"
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            results.append(row)
    return results


def read_keys(path: Path) -> dict[str, str]:
    """Read only three explicit values; never execute or export the dotenv file."""
    wanted = {"OPENAI_API_KEY", "GOOGLE_GENERATIVE_AI_API_KEY", "ANTHROPIC_API_KEY"}
    found = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line or line.lstrip().startswith("#"):
            continue
        name, value = line.split("=", 1)
        if name not in wanted:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if not value or name in found:
            raise ValueError("missing or duplicate provider credential")
        found[name] = value
    if set(found) != wanted:
        raise ValueError("required provider credentials unavailable")
    return found


class Providers:
    """Direct, no-proxy HTTPS adapters; never expose response errors or keys."""

    def __init__(self, keys: dict[str, str]):
        self.keys = keys
        self.client = httpx.Client(
            trust_env=False,
            follow_redirects=False,
            timeout=httpx.Timeout(120.0, connect=10.0),
        )

    def close(self) -> None:
        self.client.close()

    def post(self, url: str, headers: dict[str, str], **kwargs) -> dict:
        response = self.client.post(url, headers=headers, **kwargs)
        if response.status_code != 200:
            raise ValueError(f"provider_http_{response.status_code}")
        return response.json()

    def openai(self, audio: bytes, language: str) -> dict:
        response = self.post(
            "https://api.openai.com/v1/audio/transcriptions",
            {"Authorization": f"Bearer {self.keys['OPENAI_API_KEY']}"},
            data={
                "model": OPENAI_MODEL,
                "language": language[:2],
                "response_format": "json",
            },
            files={"file": ("sample.wav", audio, "audio/wav")},
        )
        text = response.get("text")
        if not isinstance(text, str):
            raise TypeError("openai_response_invalid")
        return {"model": OPENAI_MODEL, "text": text, "usage": response.get("usage")}

    def cleanup_google_file(self, display_name: str, known_name: str | None) -> None:
        headers = {"x-goog-api-key": self.keys["GOOGLE_GENERATIVE_AI_API_KEY"]}
        names = [known_name] if known_name else []
        if not names:
            token = ""
            for _ in range(20):
                try:
                    listed = self.client.get(
                        "https://generativelanguage.googleapis.com/v1beta/files",
                        headers=headers,
                        params={
                            "pageSize": 100,
                            **({"pageToken": token} if token else {}),
                        },
                    )
                    if listed.status_code != 200:
                        raise RemoteCleanupUncertain(display_name)
                    page = listed.json()
                    names.extend(
                        item["name"]
                        for item in page.get("files", [])
                        if item.get("displayName") == display_name
                    )
                    token = page.get("nextPageToken", "")
                except (httpx.HTTPError, ValueError, KeyError, TypeError) as error:
                    raise RemoteCleanupUncertain(display_name) from error
                if not token:
                    break
            else:
                raise RemoteCleanupUncertain(display_name)
        if not names:
            raise RemoteCleanupUncertain(display_name)
        for name in names:
            if not re.fullmatch(r"files/[A-Za-z0-9_-]+", name):
                raise RemoteCleanupUncertain(display_name)
            try:
                deleted = self.client.delete(
                    f"https://generativelanguage.googleapis.com/v1beta/{name}",
                    headers=headers,
                )
            except httpx.HTTPError as error:
                raise RemoteCleanupUncertain(name) from error
            if deleted.status_code not in (200, 204):
                raise RemoteCleanupUncertain(name)

    def google(self, audio: bytes, language: str) -> dict:
        headers = {"x-goog-api-key": self.keys["GOOGLE_GENERATIVE_AI_API_KEY"]}
        display_name = "translator-diagnostic-" + uuid.uuid4().hex + ".wav"
        start = self.client.post(
            "https://generativelanguage.googleapis.com/upload/v1beta/files",
            headers=headers
            | {
                "X-Goog-Upload-Protocol": "resumable",
                "X-Goog-Upload-Command": "start",
                "X-Goog-Upload-Header-Content-Length": str(len(audio)),
                "X-Goog-Upload-Header-Content-Type": "audio/wav",
                "Content-Type": "application/json",
            },
            json={"file": {"display_name": display_name}},
        )
        if start.status_code != 200:
            raise ValueError(f"google_upload_start_{start.status_code}")
        upload_url = start.headers.get("x-goog-upload-url", "")
        parsed = urlsplit(upload_url)
        if (
            parsed.scheme != "https"
            or parsed.hostname != "generativelanguage.googleapis.com"
        ):
            raise ValueError("google_upload_url_untrusted")
        name = None
        try:
            uploaded = self.client.post(
                upload_url,
                headers=headers
                | {
                    "X-Goog-Upload-Offset": "0",
                    "X-Goog-Upload-Command": "upload, finalize",
                    "Content-Type": "audio/wav",
                },
                content=audio,
            )
            if uploaded.status_code != 200:
                raise ValueError(f"google_upload_finalize_{uploaded.status_code}")
            file = uploaded.json()["file"]
            name = file["name"]
            if not re.fullmatch(r"files/[A-Za-z0-9_-]+", name):
                name = None
                raise ValueError("google_file_name_untrusted")
            response = self.post(
                "https://generativelanguage.googleapis.com/v1beta/interactions",
                headers,
                json={
                    "model": GOOGLE_MODEL,
                    "input": [
                        {"type": "audio", "uri": file["uri"], "mime_type": "audio/wav"}
                    ],
                    "generation_config": {
                        "transcription_config": {
                            "language_codes": [
                                "ru-RU" if language == "ru_ru" else "en-US"
                            ],
                            "mode": {"type": "verbatim"},
                        }
                    },
                    "store": False,
                },
            )
        finally:
            self.cleanup_google_file(display_name, name)
        if response.get("status") != "completed":
            raise ValueError("google_response_incomplete")
        steps = response.get("steps", [])
        if not isinstance(steps, list) or any(
            step.get("type") != "model_output" for step in steps
        ):
            raise ValueError("google_response_invalid")
        text = "".join(
            part["text"]
            for step in steps
            for part in step["content"]
            if part.get("type") == "text"
        )
        return {
            "model": GOOGLE_MODEL,
            "text": text.strip(),
            "usage": response.get("usage"),
        }

    def claude(self, source: str, variants: dict) -> dict:
        verdict_schema = {
            "type": "object",
            "properties": {
                "verdict": {"type": "string", "enum": ["PASS", "FAIL", "UNCERTAIN"]},
                "critical_errors": {
                    "type": "array",
                    "items": {"type": "string", "enum": list(ERROR_CATEGORIES)},
                },
            },
            "required": ["verdict", "critical_errors"],
            "additionalProperties": False,
        }
        schema = {
            "type": "object",
            "properties": {
                label: {
                    "type": "object",
                    "properties": {
                        basis: verdict_schema for basis in ("reference", "own_asr")
                    },
                    "required": ["reference", "own_asr"],
                    "additionalProperties": False,
                }
                for label in "ABCD"
            },
            "required": list("ABCD"),
            "additionalProperties": False,
        }
        task = (
            "Evaluate four anonymous translations. For each variant, compare the translation "
            "separately against (1) the written source reference and (2) that variant's own "
            "ASR source text. PASS means all meaningful facts are retained; FAIL means a "
            "material changed/lost fact; UNCERTAIN means source or target ambiguity. "
            "Pay special attention to negation, numbers, names, omissions and actor/action changes. "
            "Mark critical_errors using the allowed categories. Do not infer audio truth from "
            "the written reference. Do not favor style or fluency over meaning."
        )
        response = self.post(
            "https://api.anthropic.com/v1/messages",
            {
                "x-api-key": self.keys["ANTHROPIC_API_KEY"],
                "anthropic-version": "2023-06-01",
            },
            json={
                "model": CLAUDE_MODEL,
                "max_tokens": 1600,
                "system": task,
                "messages": [
                    {
                        "role": "user",
                        "content": json.dumps(
                            {"written_reference": source, "variants": variants},
                            ensure_ascii=False,
                        ),
                    }
                ],
                "output_config": {"format": {"type": "json_schema", "schema": schema}},
            },
        )
        if response.get("stop_reason") != "end_turn":
            raise ValueError("claude_response_incomplete")
        parts = response.get("content", [])
        text_parts = [part for part in parts if part.get("type") == "text"]
        if len(text_parts) != 1 or any(
            part.get("type") not in ("text", "thinking", "redacted_thinking")
            for part in parts
        ):
            raise ValueError("claude_response_invalid")
        return validate_judgment(json.loads(text_parts[0]["text"]))


def silence_wav() -> bytes:
    stream = io.BytesIO()
    with wave.open(stream, "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(16000)
        file.writeframes(b"\0" * 32000)
    return stream.getvalue()


def controls(providers: Providers) -> dict:
    """One speechless audio control per ASR provider, before scored cases."""
    audio = silence_wav()
    results = {}
    for name in ("openai", "google"):
        try:
            response = getattr(providers, name)(audio, "en_us")
            text = response["text"].strip()
            results[name] = {
                "status": "PASS" if text in ("", "NO_SPEECH") else "FAIL",
                "model": response["model"],
                "text": text,
            }
        except RemoteCleanupUncertain as error:
            results[name] = {
                "status": "ERROR",
                "error_type": type(error).__name__,
                "remote_identity": error.remote_identity,
            }
            break
        except Exception as error:  # noqa: BLE001 - diagnostics retain failure type
            results[name] = {"status": "ERROR", "error_type": type(error).__name__}
    return results


def write_control_failure(
    output: Path, controls_result: dict, hashes: tuple[str, ...]
) -> None:
    """Retain safe cleanup identity even if scored requests never begin."""
    receipt = {
        "type": "control_failure",
        "controls": controls_result,
        "input_sha256": dict(
            zip(("manifest", "screen", "original", "candidate"), hashes)
        ),
        "runner_sha256": sha256(Path(__file__)),
    }
    descriptor = os.open(
        output, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(json.dumps(receipt, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def summarize(rows: list[dict]) -> dict:
    providers = {
        name: {
            status: sum(row[name]["status"] == status for row in rows)
            for status in ("COMPLETED", "ERROR", "NOT_RUN")
        }
        for name in ("openai", "google", "claude")
    }
    verdicts = {
        arm: {
            basis: {status: 0 for status in ("PASS", "FAIL", "UNCERTAIN", "NOT_RUN")}
            for basis in ("reference", "own_asr")
        }
        for arm in ARMS
    }
    asr_totals: dict[str, dict[str, dict[str, int]]] = {}
    for row in rows:
        if row["claude"]["status"] == "COMPLETED":
            for label, arm in row["mapping"].items():
                for basis in ("reference", "own_asr"):
                    decision = row["claude"]["verdicts"][label][basis]["verdict"]
                    verdicts[arm][basis][decision] += 1
        else:
            for arm in ARMS:
                for basis in ("reference", "own_asr"):
                    verdicts[arm][basis]["NOT_RUN"] += 1
        language = row["language"]
        for name in (*ARMS, "openai", "google"):
            transcript = (
                row["arms"][name]["asr_text"] if name in ARMS else row[name].get("text")
            )
            if transcript is None or (
                name not in ARMS and row[name]["status"] != "COMPLETED"
            ):
                continue
            mistakes, words = errors(row["reference"], transcript)
            total = asr_totals.setdefault(language, {}).setdefault(
                name, {"cases": 0, "word_errors": 0, "reference_words": 0}
            )
            total["cases"] += 1
            total["word_errors"] += mistakes
            total["reference_words"] += words
    for per_language in asr_totals.values():
        for total in per_language.values():
            total["wer_against_written_reference"] = (
                total["word_errors"] / total["reference_words"]
                if total["reference_words"]
                else None
            )
    return {
        "requested_cases": len(rows),
        "complete_24": len(rows) == MAX_CASES
        and all(row["status"] == "COMPLETED" for row in rows),
        "case_status": {
            status: sum(row["status"] == status for row in rows)
            for status in sorted({row["status"] for row in rows})
        },
        "providers": providers,
        "verdicts": verdicts,
        "asr_against_written_reference": asr_totals,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in (
        "manifest",
        "screen",
        "original",
        "candidate",
        "secrets_file",
        "output",
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    for name in ("manifest", "screen", "original", "candidate"):
        parser.add_argument("--" + name + "-sha256", required=True)
    parser.add_argument("--limit", type=int, default=MAX_CASES)
    args = parser.parse_args()
    paths = (args.manifest, args.screen, args.original, args.candidate)
    hashes = tuple(
        getattr(args, name + "_sha256")
        for name in ("manifest", "screen", "original", "candidate")
    )
    private_path(args.output, existing=False)
    cases = load_cases(*paths, hashes)
    keys = read_keys(args.secrets_file)
    providers = Providers(keys)
    try:
        control_results = controls(providers)
        if any(item["status"] != "PASS" for item in control_results.values()):
            write_control_failure(args.output, control_results, hashes)
            print(
                json.dumps(
                    {
                        "controls": {
                            name: item["status"]
                            for name, item in control_results.items()
                        },
                        "cases": 0,
                        "output_sha256": sha256(args.output),
                    }
                )
            )
            return 1
        rows = run_cases(
            cases,
            args.output,
            providers.openai,
            providers.google,
            providers.claude,
            limit=args.limit,
            control_status={
                name: item["status"] for name, item in control_results.items()
            },
            provenance={
                "input_sha256": dict(
                    zip(("manifest", "screen", "original", "candidate"), hashes)
                ),
                "runner_sha256": sha256(Path(__file__)),
                "provider_models": {
                    "openai": OPENAI_MODEL,
                    "google": GOOGLE_MODEL,
                    "claude": CLAUDE_MODEL,
                },
            },
        )
    finally:
        providers.close()
    summary = summarize(rows)
    summary["controls"] = {
        name: item["status"] for name, item in control_results.items()
    }
    summary["output_sha256"] = sha256(args.output)
    print(json.dumps(summary))
    return int(not summary["complete_24"])


if __name__ == "__main__":
    raise SystemExit(main())

"""Independent ASR proxy for the saved Piper/Supertonic direct-synthesis pilot.

This does not measure naturalness or the 24-case full-chain TTS output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import wave
from collections.abc import Callable
from pathlib import Path

from translator_audio_adjudication import private_path
from translator_cloud_diagnostic import (
    MAX_WAV_BYTES,
    OPENAI_MODEL,
    Providers,
    read_keys,
    silence_wav,
)
from translator_mdc_asr_run import sha256
from translator_mdc_asr_score import errors

Transcriber = Callable[[bytes, str], dict]
EXPECTED_KEYS = {
    (case, language, gender)
    for case in ("negation_amount", "time_correction")
    for language in ("ru", "en")
    for gender in ("female", "male")
}


def checked_wav(report_path: Path, filename: str) -> Path:
    if not isinstance(filename, str) or not filename or Path(filename).name != filename:
        raise ValueError("TTS WAV must be a basename")
    if not filename.lower().endswith(".wav"):
        raise ValueError("TTS pilot source must be a WAV")
    wav = report_path.parent / filename
    if any(part.is_symlink() for part in (wav, *wav.parents)):
        raise ValueError("TTS WAV symlink component")
    if not wav.is_file() or wav.resolve() != report_path.parent.resolve() / filename:
        raise ValueError("TTS WAV escapes report directory")
    try:
        with wave.open(str(wav), "rb") as source:
            if source.getnframes() < 1 or source.getnchannels() < 1:
                raise ValueError("empty TTS WAV")
    except (OSError, EOFError, wave.Error) as error:
        raise ValueError("invalid TTS WAV") from error
    return wav


def load_cases(
    piper_report: Path,
    supertonic_report: Path,
    piper_hash: str,
    supertonic_hash: str,
) -> list[dict]:
    cases = []
    paired = {}
    seen_by_backend = {"piper": set(), "supertonic": set()}
    for backend, path, expected_hash in (
        ("piper", piper_report, piper_hash),
        ("supertonic", supertonic_report, supertonic_hash),
    ):
        if sha256(path) != expected_hash:
            raise ValueError("TTS report hash changed")
        report = json.loads(path.read_text(encoding="utf-8"))
        rows = report["results"]
        if report["backend"] != backend or len(rows) != 8:
            raise ValueError("TTS pilot report incomplete")
        for row in rows:
            key = (row["case"], row["lang"], row["gender"])
            if key not in EXPECTED_KEYS or key in seen_by_backend[backend]:
                raise ValueError("duplicate or unexpected TTS pilot case")
            seen_by_backend[backend].add(key)
            if backend == "piper":
                paired[key] = row["text"]
            elif paired.get(key) != row["text"]:
                raise ValueError("unpaired TTS pilot text")
            wav = checked_wav(path, row["wav"])
            if wav.stat().st_size > MAX_WAV_BYTES or sha256(wav) != row["wav_sha256"]:
                raise ValueError("TTS pilot WAV changed")
            cases.append({"backend": backend, "row": row, "wav": wav})
    if (
        seen_by_backend["piper"] != EXPECTED_KEYS
        or seen_by_backend["supertonic"] != EXPECTED_KEYS
    ):
        raise ValueError("TTS pilot pair count changed")
    return cases


def run_cases(cases: list[dict], output: Path, transcribe: Transcriber) -> list[dict]:
    private_path(output, existing=False)
    if len(cases) != 16:
        raise ValueError("TTS pilot must contain 16 outputs")
    descriptor = os.open(
        output, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600
    )
    results = []
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        for case in cases:
            source = case["row"]
            record = {
                "type": "direct_tts_pilot_diagnostic",
                "backend": case["backend"],
                "case": source["case"],
                "lang": source["lang"],
                "gender": source["gender"],
                "target_text": source["text"],
                "wav_sha256": source["wav_sha256"],
                "asr_model": OPENAI_MODEL,
                "status": "ERROR",
            }
            started = time.monotonic_ns()
            try:
                audio = case["wav"].read_bytes()
                if hashlib.sha256(audio).hexdigest() != source["wav_sha256"]:
                    raise ValueError("TTS WAV changed before send")
                result = transcribe(audio, source["lang"])
                if (
                    not isinstance(result.get("text"), str)
                    or not result["text"].strip()
                ):
                    raise ValueError("empty TTS transcription")
                record["transcript"] = result["text"]
                record["status"] = "COMPLETED"
                mistakes, words = errors(source["text"], result["text"])
                record["word_errors"] = mistakes
                record["reference_words"] = words
            except Exception as error:  # noqa: BLE001 - preserve the failed attempt
                record["error_type"] = type(error).__name__
            record["elapsed_ms"] = (time.monotonic_ns() - started) / 1e6
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            results.append(record)
    return results


def summarize(rows: list[dict]) -> dict:
    by_arm = {}
    for backend in ("piper", "supertonic"):
        for language in ("ru", "en"):
            selected = [
                row
                for row in rows
                if row["backend"] == backend and row["lang"] == language
            ]
            complete = [row for row in selected if row["status"] == "COMPLETED"]
            mistakes = sum(row["word_errors"] for row in complete)
            words = sum(row["reference_words"] for row in complete)
            by_arm[f"{backend}/{language}"] = {
                "attempted": len(selected),
                "completed": len(complete),
                "word_errors": mistakes,
                "reference_words": words,
                "asr_proxy_wer": mistakes / words if words else None,
            }
    return {
        "attempted": len(rows),
        "complete_16": len(rows) == 16
        and all(row["status"] == "COMPLETED" for row in rows),
        "by_backend_language": by_arm,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("piper_report", "supertonic_report", "secrets_file", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--piper-sha256", required=True)
    parser.add_argument("--supertonic-sha256", required=True)
    args = parser.parse_args()
    cases = load_cases(
        args.piper_report,
        args.supertonic_report,
        args.piper_sha256,
        args.supertonic_sha256,
    )
    private_path(args.output, existing=False)
    providers = Providers(read_keys(args.secrets_file))
    try:
        silence = providers.openai(silence_wav(), "en_us")["text"]
        if silence.strip():
            print(json.dumps({"control": "FAIL", "attempted": 0}))
            return 1
        rows = run_cases(cases, args.output, providers.openai)
    finally:
        providers.close()
    result = summarize(rows)
    result.update(
        {
            "control": "PASS",
            "model": OPENAI_MODEL,
            "piper_report_sha256": args.piper_sha256,
            "supertonic_report_sha256": args.supertonic_sha256,
            "runner_sha256": sha256(Path(__file__)),
            "output_sha256": sha256(args.output),
        }
    )
    print(json.dumps(result))
    return int(not result["complete_16"])


if __name__ == "__main__":
    raise SystemExit(main())

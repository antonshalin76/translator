"""Measure product Piper frames from pinned, saved MT outputs (no playback)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import stat
import time
from pathlib import Path

from translator_sidecar.local.model_lease import VerifiedModelSource
from translator_sidecar.local.model_manifest import load_manifest
from translator_sidecar.local.tts import PiperTts, PiperVoiceRegistry
from translator_sidecar.provider_contract import (
    Language,
    TranslationMode,
    VoiceEngine,
    VoiceGender,
    VoiceProfile,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_SHA256 = "36398ee5dc5c4c2fadcf906b54edc7519590dccfb778d838677f46af2133f5d8"
SCREEN_SHA256 = "005db22422ad98c9c70ecd3d01f578be571ea4bebca81398d5fff7cda86214ae"
NLLB_SHA256 = "cd9ac14fc633db2e8f346b4da23e9c3c1203960f3ca141a9a1fd9179e9effa82"
HY_GPU_SHA256 = "da5fb41004288b66cc1bbe8fa2d7fa199c8e5a24c9b12836c82c5bcf5837f43d"
CASE_IDS = (
    "ru-71601",
    "ru-71597",
    "ru-71963",
    "ru-71599",
    "en-87466",
    "en-71488",
    "en-78643",
    "en-20265",
)
VOICES = {
    Language.RU: "piper-ru-irina-medium",
    Language.EN: "piper-en-hfc-female-medium",
}
FRAME_BYTES = 24_000 * 20 // 1000 * 2
MAX_FRAMES = 30_000 // 20


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_report(path: Path, expected_sha256: str) -> dict:
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError("saved MT report identity changed")
    return json.loads(content)


def paired_cases(nllb: dict, hy: dict) -> list[tuple[str, dict, dict]]:
    if (
        nllb.get("screen_sha256") != SCREEN_SHA256
        or hy.get("screen_sha256") != SCREEN_SHA256
        or nllb.get("manifest_sha256") != hy.get("manifest_sha256")
        or nllb.get("source_head") != hy.get("source_head")
    ):
        raise ValueError("saved MT reports do not share one frozen source")

    def index(report: dict) -> dict[str, dict]:
        rows = report.get("cases", [])
        indexed = {row["origin_id"]: row for row in rows}
        if len(indexed) != len(rows):
            raise ValueError("saved MT report contains duplicate cases")
        return indexed

    left, right = index(nllb), index(hy)
    if not set(CASE_IDS).issubset(left) or not set(CASE_IDS).issubset(right):
        raise ValueError("saved MT report is missing a selected case")
    pairs = []
    for case_id in CASE_IDS:
        a, b = left[case_id], right[case_id]
        if (
            a.get("condition") != "clean"
            or b.get("condition") != "clean"
            or a.get("source_language") != case_id[:2]
            or b.get("source_language") != case_id[:2]
            or not a.get("source")
            or a["source"] != b.get("source")
            or not a.get("output")
            or not b.get("output")
        ):
            raise ValueError("selected MT texts do not form a clean pair")
        pairs.append((case_id, a, b))
    return pairs


def run_order(pass_index: int) -> tuple[str, str]:
    if pass_index == 0:
        return "nllb", "hy_gpu"
    if pass_index == 1:
        return "hy_gpu", "nllb"
    raise ValueError("exactly two counterbalanced passes are defined")


def measure_pcm(tts: PiperTts, text: str, target: str) -> dict:
    language = Language(target)
    profile = VoiceProfile(
        language=language, gender=VoiceGender.FEMALE, engine=VoiceEngine.PIPER
    )
    started = time.monotonic_ns()
    frames = iter(
        tts.synthesize_frames(
            text,
            target_language=language,
            voice_profile=profile,
            mode=TranslationMode.QUALITY_FIRST,
            output_sample_rate_hz=24_000,
            output_channels=1,
            frame_duration_ms=20,
            continuation=False,
        )
    )
    digest = hashlib.sha256()
    frame_count = 0
    nonzero = False
    first_pcm_ms = None
    for frame in frames:
        if len(frame) != FRAME_BYTES:
            raise ValueError("product Piper emitted a malformed PCM frame")
        frame_count += 1
        if frame_count > MAX_FRAMES:
            raise ValueError("product Piper exceeded the 30-second output bound")
        if first_pcm_ms is None:
            first_pcm_ms = (time.monotonic_ns() - started) / 1e6
        digest.update(frame)
        nonzero |= any(frame)
    if not frame_count or not nonzero:
        raise ValueError("product Piper emitted no non-silent audio")
    return {
        "first_pcm_ms": round(first_pcm_ms, 2),
        "total_ms": round((time.monotonic_ns() - started) / 1e6, 2),
        "frame_count": frame_count,
        "duration_ms": frame_count * 20,
        "pcm_sha256": digest.hexdigest(),
    }


def write_private(path: Path, report: dict) -> None:
    prior_umask = os.umask(0o077)
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    finally:
        os.umask(prior_umask)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nllb", type=Path, required=True)
    parser.add_argument("--hy-gpu", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        not args.output.is_absolute()
        or args.output.resolve().is_relative_to(ROOT)
        or any(parent.is_symlink() for parent in args.output.parents)
        or not args.output.parent.is_dir()
        or stat.S_IMODE(args.output.parent.stat().st_mode) != 0o700
        or args.output.exists()
    ):
        raise ValueError("output must be a new file in a private directory outside Git")

    report = {
        "scope": "saved MT output into product Piper frames; no ASR, playback, or EN listening",
        "input_sha256": {"nllb": NLLB_SHA256, "hy_gpu": HY_GPU_SHA256},
        "screen_sha256": SCREEN_SHA256,
        "case_ids": CASE_IDS,
        "voice_gender": "female",
        "format": "signed 16-bit little-endian mono, 24000 Hz, 20 ms",
        "warmup": "one short utterance per target voice; excluded from timings",
        "runner_sha256": sha256(Path(__file__)),
        "runtime_code_sha256": {
            name: sha256(ROOT / name)
            for name in (
                "sidecar/translator_sidecar/local/tts.py",
                "sidecar/translator_sidecar/local/model_lease.py",
                "sidecar/translator_sidecar/local/model_manifest.py",
                "sidecar/translator_sidecar/provider_contract.py",
            )
        },
        "rows": [],
        "status": "FAILED",
    }
    tts = None
    try:
        nllb = read_report(args.nllb, NLLB_SHA256)
        hy = read_report(args.hy_gpu, HY_GPU_SHA256)
        pairs = paired_cases(nllb, hy)
        manifest_path = ROOT / "models/manifest.json"
        if sha256(manifest_path) != MANIFEST_SHA256:
            raise ValueError("product model manifest identity changed")
        if nllb["manifest_sha256"] != MANIFEST_SHA256:
            raise ValueError("saved MT manifest identity changed")
        manifest = load_manifest(manifest_path)
        report["manifest_sha256"] = MANIFEST_SHA256
        report["voices"] = {
            language.value: {
                "id": model_id,
                "file_sha256": {
                    entry.path: entry.sha256
                    for entry in manifest.models[model_id].files
                },
            }
            for language, model_id in VOICES.items()
        }
        registry = PiperVoiceRegistry(
            {
                (language, VoiceGender.FEMALE): VerifiedModelSource(manifest, model_id)
                for language, model_id in VOICES.items()
            }
        )
        tts = PiperTts(registry)
        registry.prepare()
        measure_pcm(tts, "Привет.", "ru")
        measure_pcm(tts, "Hello.", "en")
        for pass_index in range(2):
            for case_id, left, right in pairs:
                target = "en" if case_id.startswith("ru-") else "ru"
                rows = {"nllb": left, "hy_gpu": right}
                for backend in run_order(pass_index):
                    output = rows[backend]["output"]
                    measurement = measure_pcm(tts, output, target)
                    report["rows"].append(
                        {
                            "pass": pass_index + 1,
                            "origin_id": case_id,
                            "backend": backend,
                            "target_language": target,
                            "text_sha256": hashlib.sha256(
                                output.encode("utf-8")
                            ).hexdigest(),
                            **measurement,
                        }
                    )
        report["status"] = "COMPLETE"
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        if tts is not None:
            tts.close()
        usage = resource.getrusage(resource.RUSAGE_SELF)
        report["peak_rss_kib"] = usage.ru_maxrss
        report["swaps"] = usage.ru_nswap
        write_private(args.output, report)
    print(json.dumps({"status": report["status"], "rows": len(report["rows"])}))


if __name__ == "__main__":
    main()

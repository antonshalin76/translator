from __future__ import annotations

import argparse
import json
import resource
import time
import wave
from pathlib import Path

from translator_sidecar.local.asr import AsrModelManager
from translator_sidecar.local.model_lease import VerifiedModelSource
from translator_sidecar.local.model_manifest import load_manifest
from translator_sidecar.provider_contract import Language, TranslationMode


def read_pcm(path: Path) -> bytes:
    with wave.open(str(path), "rb") as source:
        if (
            source.getframerate() != 16_000
            or source.getnchannels() != 1
            or source.getsampwidth() != 2
        ):
            raise ValueError("native ASR fixture must be 16 kHz mono s16le")
        return source.readframes(source.getnframes())


def _device_checkpoint(
    manager: AsrModelManager,
    *,
    expected_device: str,
    stage: str,
) -> dict[str, str]:
    actual_device = manager.actual_device
    if actual_device != expected_device:
        raise RuntimeError(
            f"native ASR device mismatch {stage}: "
            f"expected {expected_device}, got {actual_device}"
        )
    resident_model_id = manager.resident_model_id
    if resident_model_id != "small":
        raise RuntimeError(
            f"native ASR resident mismatch {stage}: "
            f"expected small, got {resident_model_id}"
        )
    return {"device": actual_device, "resident_model_id": resident_model_id}


def transcribe_with_device_oracle(
    manager: AsrModelManager,
    *,
    ru_pcm: bytes,
    en_pcm: bytes,
    expected_device: str,
) -> dict[str, object]:
    started = time.perf_counter()
    manager.prepare()
    load_ms = round((time.perf_counter() - started) * 1000, 3)
    checkpoints = {
        "after_prepare": _device_checkpoint(
            manager,
            expected_device=expected_device,
            stage="after_prepare",
        )
    }

    started = time.perf_counter()
    ru = manager.transcribe(
        ru_pcm,
        language=Language.RU,
        mode=TranslationMode.BALANCED,
    )
    checkpoints["after_ru"] = _device_checkpoint(
        manager,
        expected_device=expected_device,
        stage="after_ru",
    )
    en = manager.transcribe(
        en_pcm,
        language=Language.EN,
        mode=TranslationMode.BALANCED,
    )
    checkpoints["after_en"] = _device_checkpoint(
        manager,
        expected_device=expected_device,
        stage="after_en",
    )
    return {
        "load_ms": load_ms,
        "actual_device": manager.actual_device,
        "resident_model_id": manager.resident_model_id,
        "device_checkpoints": checkpoints,
        "ru": ru,
        "en": en,
        "inference_ms": round((time.perf_counter() - started) * 1000, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--model-id", required=True)
    args = parser.parse_args()

    manager = AsrModelManager(
        selected_id="small",
        model_paths={
            "small": VerifiedModelSource(load_manifest(args.manifest), args.model_id)
        },
        device=args.device,
    )
    results: dict[str, object] = {}
    try:
        results.update(
            transcribe_with_device_oracle(
                manager,
                ru_pcm=read_pcm(args.corpus_root / "audio/ru-short-02/clean.wav"),
                en_pcm=read_pcm(args.corpus_root / "audio/en-short-03/clean.wav"),
                expected_device=args.device,
            )
        )
    finally:
        started = time.perf_counter()
        manager.close()
        results["close_ms"] = round((time.perf_counter() - started) * 1000, 3)
        results["peak_rss_mib"] = round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 3
        )
    print(json.dumps(results, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

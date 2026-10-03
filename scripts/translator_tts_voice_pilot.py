"""Offline, non-product Piper/Supertonic voice diagnostic.

Run each backend with its own pinned environment. Reports and WAV files belong
in an ignored evaluation directory, never in the production model cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
import wave
from pathlib import Path

import numpy as np

CASES = (
    (
        "ru",
        "negation_amount",
        "Не отправляйте деньги Ивану Петрову. Сумма — 15 420 рублей.",
    ),
    ("ru", "time_correction", "Поезд номер 47 прибудет в 18:05, не в 18:50."),
    (
        "en",
        "negation_amount",
        "Do not transfer $1,540 to Ryan. The reference number is 7049.",
    ),
    ("en", "time_correction", "Flight 42 departs at 6:15 p.m., not 6:50 p.m."),
)

PIPER_MODELS = {
    ("ru", "female"): ("piper", "ru_RU-irina-medium.onnx"),
    ("ru", "male"): ("piper-voices/ru/ru_RU/dmitri/medium", "ru_RU-dmitri-medium.onnx"),
    ("en", "female"): ("piper", "en_US-hfc_female-medium.onnx"),
    ("en", "male"): ("piper-voices/en/en_US/ryan/medium", "en_US-ryan-medium.onnx"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("piper", "supertonic"), required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--female-piper-cache", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.backend == "piper":
        from piper import PiperVoice

        voices = {}
        identities = {}
        load_start = time.perf_counter()
        for key, (directory, filename) in PIPER_MODELS.items():
            if key[1] == "female":
                if args.female_piper_cache is None:
                    parser.error("--female-piper-cache is required for Piper")
                model = args.female_piper_cache / filename
            else:
                model = args.model_root / directory / filename
            config = model.with_name(filename + ".json")
            if not model.is_file() or not config.is_file():
                raise FileNotFoundError(model)
            identities[f"{key[0]}_{key[1]}"] = {
                "model_sha256": sha256(model),
                "config_sha256": sha256(config),
            }
            voices[key] = PiperVoice.load(model, config_path=config, use_cuda=False)
        load_ms = (time.perf_counter() - load_start) * 1000

        def synthesize(
            text: str, lang: str, gender: str
        ) -> tuple[np.ndarray, int, float]:
            chunks = iter(voices[(lang, gender)].synthesize(text))
            start = time.perf_counter()
            first = next(chunks)
            first_pcm_ms = (time.perf_counter() - start) * 1000
            arrays = [first.audio_float_array]
            arrays.extend(chunk.audio_float_array for chunk in chunks)
            return np.concatenate(arrays), first.sample_rate, first_pcm_ms

    else:
        from supertonic import TTS

        model_dir = args.model_root / "supertonic-3"
        identities = {
            name: sha256(model_dir / name)
            for name in (
                "onnx/duration_predictor.onnx",
                "onnx/text_encoder.onnx",
                "onnx/vector_estimator.onnx",
                "onnx/vocoder.onnx",
                "voice_styles/M1.json",
                "voice_styles/F1.json",
            )
        }
        load_start = time.perf_counter()
        engine = TTS(model_dir=model_dir, auto_download=False, intra_op_num_threads=2)
        styles = {
            "female": engine.get_voice_style("F1"),
            "male": engine.get_voice_style("M1"),
        }
        load_ms = (time.perf_counter() - load_start) * 1000

        def synthesize(
            text: str, lang: str, gender: str
        ) -> tuple[np.ndarray, int, float]:
            start = time.perf_counter()
            samples, _ = engine.synthesize(text, voice_style=styles[gender], lang=lang)
            first_pcm_ms = (time.perf_counter() - start) * 1000
            return np.asarray(samples).reshape(-1), engine.sample_rate, first_pcm_ms

    # Warm each language/style path outside the request timing boundary.
    for lang in ("ru", "en"):
        for gender in ("female", "male"):
            synthesize("Привет." if lang == "ru" else "Hello.", lang, gender)

    results = []
    for lang, case_id, text in CASES:
        for gender in ("female", "male"):
            start = time.perf_counter()
            samples, sample_rate, first_pcm_ms = synthesize(text, lang, gender)
            elapsed_ms = (time.perf_counter() - start) * 1000
            if sample_rate <= 0 or not np.isfinite(samples).all() or not len(samples):
                raise ValueError(f"invalid audio: {lang} {case_id} {gender}")
            audio_seconds = len(samples) / sample_rate
            filename = f"{args.backend}-{lang}-{gender}-{case_id}.wav"
            output = args.output_dir / filename
            save_wav(output, samples, sample_rate)
            results.append(
                {
                    "case": case_id,
                    "lang": lang,
                    "gender": gender,
                    "text": text,
                    "wav": filename,
                    "wav_sha256": sha256(output),
                    "sample_rate_hz": sample_rate,
                    "audio_seconds": round(audio_seconds, 3),
                    "first_pcm_ms": round(first_pcm_ms, 2),
                    "elapsed_ms": round(elapsed_ms, 2),
                    "rtf": round(elapsed_ms / (audio_seconds * 1000), 3),
                    "peak": round(float(np.max(np.abs(samples))), 5),
                    "rms": round(float(np.sqrt(np.mean(np.square(samples)))), 5),
                    "clipped_fraction": round(
                        float(np.mean(np.abs(samples) >= 0.999)), 6
                    ),
                }
            )

    report = {
        "backend": args.backend,
        "model_identities": identities,
        "load_ms": round(load_ms, 2),
        "warmup": "one short utterance per language/gender, excluded from request timing",
        "timing": "direct in-process synthesis; first_pcm_ms is full-utterance completion for Supertonic",
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "results": results,
    }
    report_path = args.output_dir / f"{args.backend}-report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "report": str(report_path),
                "cases": len(results),
                "load_ms": report["load_ms"],
            }
        )
    )


if __name__ == "__main__":
    main()

"""Use a frozen, CPU-only ASR model as a TTS intelligibility diagnostic.

This is not a listening test or an independent product quality score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import jiwer
from faster_whisper import WhisperModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--only-wav", action="append")
    parser.add_argument("--output-name", default="asr-proxy-report.json")
    args = parser.parse_args()

    source = [
        json.loads((args.audio_dir / f"{name}-report.json").read_text())
        for name in ("piper", "supertonic")
    ]
    digest = hashlib.sha256()
    with (args.model_dir / "model.bin").open("rb") as model_file:
        for block in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(block)
    model_hash = digest.hexdigest()
    model = WhisperModel(
        str(args.model_dir), device="cpu", compute_type="int8", cpu_threads=2
    )
    normalize = jiwer.Compose(
        [
            jiwer.ToLowerCase(),
            jiwer.RemovePunctuation(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
        ]
    )
    observations = []
    for report in source:
        for case in report["results"]:
            if args.only_wav and case["wav"] not in args.only_wav:
                continue
            start = time.perf_counter()
            segments, _ = model.transcribe(
                str(args.audio_dir / case["wav"]),
                language=case["lang"],
                beam_size=5,
                temperature=0,
                condition_on_previous_text=False,
                vad_filter=False,
            )
            transcript = " ".join(segment.text.strip() for segment in segments)
            reference = normalize(case["text"])
            hypothesis = normalize(transcript)
            observations.append(
                {
                    "backend": report["backend"],
                    "case": case["case"],
                    "lang": case["lang"],
                    "gender": case["gender"],
                    "reference": case["text"],
                    "transcript": transcript,
                    "normalized_reference": reference,
                    "normalized_transcript": hypothesis,
                    "wer": jiwer.wer(reference, hypothesis),
                    "asr_ms": round((time.perf_counter() - start) * 1000, 2),
                }
            )

    output = args.audio_dir / args.output_name
    output.write_text(
        json.dumps(
            {
                "asr_model": str(args.model_dir.name),
                "asr_model_sha256": model_hash,
                "device": "CPU int8, two threads",
                "scope": "diagnostic proxy only; automatic punctuation/number formatting may distort WER",
                "results": observations,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"report": str(output), "cases": len(observations)}))


if __name__ == "__main__":
    main()

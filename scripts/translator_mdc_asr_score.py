"""Score exact paired Turbo/Qwen MDC reports with speaker-cluster uncertainty."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import unicodedata
from collections import defaultdict
from pathlib import Path

import jiwer


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(value: str) -> str:
    return " ".join(
        "".join(
            " " if unicodedata.category(char).startswith("P") else char
            for char in value.casefold().replace("ё", "е")
        ).split()
    )


def errors(reference: str, transcript: str) -> tuple[int, int]:
    result = jiwer.process_words(normalize(reference), normalize(transcript))
    return (
        result.substitutions + result.deletions + result.insertions,
        result.hits + result.substitutions + result.deletions,
    )


def score(items: list[dict]) -> dict:
    if not items:
        return {"n": 0}
    mistakes, words = zip(
        *(errors(item["reference"], item["transcript"]) for item in items)
    )
    return {
        "n": len(items),
        "speaker_count": len({item["speaker_id"] for item in items}),
        "reference_words": sum(words),
        "word_errors": sum(mistakes),
        "wer": sum(mistakes) / sum(words),
        "cer": jiwer.cer(
            [normalize(item["reference"]) for item in items],
            [normalize(item["transcript"]) for item in items],
        ),
        "exact_sentences": sum(
            normalize(item["reference"]) == normalize(item["transcript"])
            for item in items
        ),
        "median_inference_ms": statistics.median(item["elapsed_ms"] for item in items),
    }


def paired_speaker_interval(
    baseline: list[dict], candidate: list[dict], repetitions: int = 3000
) -> dict:
    if len(baseline) != len(candidate):
        raise ValueError("unpaired result count")
    by_speaker: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for base, trial in zip(baseline, candidate):
        if (base["audio_file"], base["speaker_id"], base["reference"]) != (
            trial["audio_file"],
            trial["speaker_id"],
            trial["reference"],
        ):
            raise ValueError("unpaired sample or reference")
        base_errors, words = errors(base["reference"], base["transcript"])
        trial_errors, trial_words = errors(trial["reference"], trial["transcript"])
        if words != trial_words:
            raise ValueError("unpaired reference words")
        by_speaker[base["speaker_id"]].append((trial_errors - base_errors, words))
    speakers = sorted(by_speaker)
    observed = sum(diff for values in by_speaker.values() for diff, _ in values) / sum(
        words for values in by_speaker.values() for _, words in values
    )
    rng = random.Random(20260924)
    estimates = []
    for _ in range(repetitions):
        sampled = [
            entry
            for speaker in rng.choices(speakers, k=len(speakers))
            for entry in by_speaker[speaker]
        ]
        estimates.append(
            sum(diff for diff, _ in sampled) / sum(words for _, words in sampled)
        )
    estimates.sort()
    return {
        "qwen_minus_turbo_wer": observed,
        "speaker_cluster_bootstrap_95pct": [
            estimates[int(0.025 * repetitions)],
            estimates[int(0.975 * repetitions) - 1],
        ],
    }


def load_report(
    path: Path, manifest: dict, manifest_hash: str
) -> tuple[str, list[dict]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report["schema"] != 1 or report["manifest_sha256"] != manifest_hash:
        raise ValueError("report not bound to frozen corpus")
    expected = {item["audio_file"]: item for item in manifest["samples"]}
    results = report["results"]
    if len(results) != len(expected) or {item["audio_file"] for item in results} != set(
        expected
    ):
        raise ValueError("missing, extra, or duplicate attempts")
    by_file = {item["audio_file"]: item for item in results}
    ordered = []
    for sample in manifest["samples"]:
        item = by_file[sample["audio_file"]]
        for field in (
            "origin_id",
            "language",
            "split",
            "speaker_id",
            "reference",
            "condition",
            "critical_labels",
        ):
            if item[field] != sample[field]:
                raise ValueError(f"report {field} mismatch")
        if item["status"] != "completed" or not isinstance(item.get("transcript"), str):
            raise ValueError("failed or invalid attempt")
        ordered.append(item)
    return report["model_id"], ordered


def compare(
    manifest_path: Path, turbo_path: Path, qwen_path: Path, output: Path
) -> dict:
    if output.exists():
        raise FileExistsError(output)
    manifest_hash = sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    turbo_id, turbo = load_report(turbo_path, manifest, manifest_hash)
    qwen_id, qwen = load_report(qwen_path, manifest, manifest_hash)
    if (turbo_id, qwen_id) != ("turbo", "qwen17"):
        raise ValueError("expected Turbo/Qwen 1.7B pair")
    result = {
        "schema": 1,
        "scorer_sha256": sha256(Path(__file__)),
        "manifest_sha256": manifest_hash,
        "report_sha256": {"turbo": sha256(turbo_path), "qwen17": sha256(qwen_path)},
        "slices": {},
    }
    for locale in ("ru_ru", "en_us"):
        for condition in ("clean", "speech_shaped_noise_10db"):
            for label in ("all", "names", "numbers", "negation_candidate"):
                indices = [
                    index
                    for index, item in enumerate(turbo)
                    if item["language"] == locale
                    and item["condition"] == condition
                    and (label == "all" or label in item["critical_labels"])
                ]
                if not indices:
                    continue
                base = [turbo[index] for index in indices]
                trial = [qwen[index] for index in indices]
                result["slices"][f"{locale}:{condition}:{label}"] = {
                    "turbo": score(base),
                    "qwen17": score(trial),
                    "paired": paired_speaker_interval(base, trial),
                }
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("manifest", "turbo", "qwen17", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.manifest, args.turbo, args.qwen17, args.output)
    print(f"scored_slices={len(result['slices'])}")

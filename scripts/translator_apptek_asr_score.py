"""Score paired AppTek ASR outputs against the publisher's manual references."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import unicodedata
from pathlib import Path

import jiwer
from translator_apptek_corpus import sha256
from translator_mdc_asr_run import (
    QWEN17_DIRECTORY_SHA256,
    QWEN17_WEIGHT_SHA256,
    TURBO_DIRECTORY_SHA256,
    TURBO_SHA256,
)

PREVIOUS_RUNNER_SHA256 = (
    "ea9a814ccd5b3860d2f8e93e18c85c34fd350b6d5d9e42e5336c372f68def221"
)
EXPECTED_MODEL_IDENTITY = {
    "turbo": {
        "model_bin_sha256": TURBO_SHA256,
        "directory_sha256": TURBO_DIRECTORY_SHA256,
        "beam_size": 5,
        "vad_filter": False,
    },
    "qwen17": {
        "weight_sha256": list(QWEN17_WEIGHT_SHA256),
        "directory_sha256": QWEN17_DIRECTORY_SHA256,
        "max_new_tokens": 256,
    },
}


def normalize(text: str) -> str:
    spaced = "".join(
        " " if unicodedata.category(char).startswith("P") else char
        for char in text.casefold()
    )
    return " ".join(spaced.split())


def validate_rows(samples: list[dict], rows: list[dict]) -> dict[str, dict]:
    expected = {sample["origin_id"]: sample for sample in samples}
    if len(expected) != len(samples) or len(rows) != len(samples):
        raise ValueError("AppTek report count or manifest origins differ")
    actual = {row["origin_id"]: row for row in rows}
    if len(actual) != len(rows) or set(actual) != set(expected):
        raise ValueError("AppTek report has duplicate, extra or missing origins")
    for origin, sample in expected.items():
        row = actual[origin]
        if any(row.get(key) != value for key, value in sample.items()):
            raise ValueError(f"AppTek report is not bound to manifest: {origin}")
        if row.get("status") != "completed" or not isinstance(
            row.get("transcript"), str
        ):
            raise ValueError(f"AppTek report has failed or incomplete origin: {origin}")
        elapsed = row.get("elapsed_ms")
        if (
            type(elapsed) not in (int, float)
            or not math.isfinite(elapsed)
            or elapsed < 0
        ):
            raise ValueError(f"AppTek report has invalid elapsed time: {origin}")
    return actual


def subset_metrics(
    samples: list[dict], turbo: dict[str, dict], qwen: dict[str, dict]
) -> dict:
    if not samples:
        raise ValueError("empty AppTek score subset")
    references = [sample["reference"] for sample in samples]
    turbo_hypotheses = [turbo[sample["origin_id"]]["transcript"] for sample in samples]
    qwen_hypotheses = [qwen[sample["origin_id"]]["transcript"] for sample in samples]
    reference_normalized = [normalize(text) for text in references]
    turbo_normalized = [normalize(text) for text in turbo_hypotheses]
    qwen_normalized = [normalize(text) for text in qwen_hypotheses]
    turbo_wer = jiwer.wer(reference_normalized, turbo_normalized)
    qwen_wer = jiwer.wer(reference_normalized, qwen_normalized)
    return {
        "n": len(samples),
        "turbo_wer": turbo_wer,
        "qwen17_wer": qwen_wer,
        "qwen_minus_turbo_wer_pp": (qwen_wer - turbo_wer) * 100,
        "turbo_raw_wer": jiwer.wer(references, turbo_hypotheses),
        "qwen17_raw_wer": jiwer.wer(references, qwen_hypotheses),
        "turbo_median_saved_wav_to_text_ms": statistics.median(
            turbo[sample["origin_id"]]["elapsed_ms"] for sample in samples
        ),
        "qwen17_median_saved_wav_to_text_ms": statistics.median(
            qwen[sample["origin_id"]]["elapsed_ms"] for sample in samples
        ),
    }


def compare(samples: list[dict], turbo_rows: list[dict], qwen_rows: list[dict]) -> dict:
    turbo = validate_rows(samples, turbo_rows)
    qwen = validate_rows(samples, qwen_rows)
    cohorts = {
        cohort: subset_metrics(
            [item for item in samples if item["cohort"] == cohort], turbo, qwen
        )
        for cohort in ("general", "critical")
        if any(item["cohort"] == cohort for item in samples)
    }
    return {
        **cohorts,
        "per_accent": {
            accent: {
                cohort: subset_metrics(
                    [
                        item
                        for item in samples
                        if item["accent"] == accent and item["cohort"] == cohort
                    ],
                    turbo,
                    qwen,
                )
                for cohort in ("general", "critical")
                if any(
                    item["accent"] == accent and item["cohort"] == cohort
                    for item in samples
                )
            }
            for accent in sorted({item["accent"] for item in samples})
        },
        "critical_by_label": {
            label: subset_metrics(
                [item for item in samples if item.get("critical_label") == label],
                turbo,
                qwen,
            )
            for label in sorted(
                {
                    item["critical_label"]
                    for item in samples
                    if item.get("critical_label")
                }
            )
        },
    }


def score(manifest_path: Path, turbo_path: Path, qwen_path: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    manifest_hash = sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    turbo = json.loads(turbo_path.read_text(encoding="utf-8"))
    qwen = json.loads(qwen_path.read_text(encoding="utf-8"))
    if (
        manifest.get("purpose") != "apptek_publisher_reference_diagnostic"
        or len(manifest["samples"]) != 70
        or manifest.get("schema") != 1
        or turbo.get("schema") != 1
        or qwen.get("schema") != 1
        or turbo.get("purpose") != manifest["purpose"]
        or qwen.get("purpose") != manifest["purpose"]
        or turbo.get("model_id") != "turbo"
        or qwen.get("model_id") != "qwen17"
        or turbo.get("model_identity") != EXPECTED_MODEL_IDENTITY["turbo"]
        or qwen.get("model_identity") != EXPECTED_MODEL_IDENTITY["qwen17"]
        or turbo.get("manifest_sha256") != manifest_hash
        or qwen.get("manifest_sha256") != manifest_hash
        or turbo.get("source_head") != qwen.get("source_head")
        or turbo.get("harness_sha256") != qwen.get("harness_sha256")
        or turbo.get("harness_sha256", {}).get("builder")
        != sha256(Path(__file__).with_name("translator_apptek_corpus.py"))
        or turbo.get("harness_sha256", {}).get("model_loader")
        != sha256(Path(__file__).with_name("translator_mdc_asr_run.py"))
        or turbo.get("harness_sha256", {}).get("runner")
        not in {
            PREVIOUS_RUNNER_SHA256,
            sha256(Path(__file__).with_name("translator_apptek_asr_run.py")),
        }
    ):
        raise ValueError("AppTek paired report identity differs")
    result = {
        "schema": 1,
        "purpose": "apptek_publisher_reference_diagnostic",
        "reference_provenance": "publisher_manual_transcript_not_independently_audio_adjudicated",
        "dataset_revision": manifest["revision"],
        "manifest_sha256": manifest_hash,
        "report_sha256": {"turbo": sha256(turbo_path), "qwen17": sha256(qwen_path)},
        "source_head": turbo["source_head"],
        "scorer_sha256": sha256(Path(__file__)),
        "metrics": compare(manifest["samples"], turbo["results"], qwen["results"]),
    }
    with output.open("x", encoding="utf-8") as destination:
        json.dump(result, destination, ensure_ascii=False, indent=2)
        destination.write("\n")
    return result


def main() -> None:
    os.umask(0o077)
    parser = argparse.ArgumentParser()
    for name in ("manifest", "turbo", "qwen17", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    report = score(args.manifest, args.turbo, args.qwen17, args.output)
    print(
        f"general_n={report['metrics']['general']['n']} critical_n={report['metrics']['critical']['n']}"
    )


if __name__ == "__main__":
    main()

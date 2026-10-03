"""Run one pinned ASR model over frozen AppTek short-turn WAVs."""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import time
from pathlib import Path

from translator_apptek_corpus import (
    ACCENTS,
    CRITICAL,
    INVENTORY_SHA256,
    load_inventory,
    sha256,
)
from translator_mdc_asr_run import load_model


def validate_corpus(path: Path, expected_sha256: str) -> list[dict]:
    if sha256(path) != expected_sha256:
        raise ValueError("frozen AppTek manifest changed")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != 1
        or manifest.get("purpose") != "apptek_publisher_reference_diagnostic"
    ):
        raise ValueError("unsupported AppTek corpus contract")
    samples = manifest["samples"]
    if not samples or len({item["origin_id"] for item in samples}) != len(samples):
        raise ValueError("empty or duplicate AppTek origins")
    if len({item["audio_file"] for item in samples}) != len(samples):
        raise ValueError("duplicate AppTek clip")
    for sample in samples:
        relative = Path(sample["audio_file"])
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or sha256(path.parent / relative) != sample["sha256"]
        ):
            raise ValueError(f"frozen AppTek clip changed: {sample['origin_id']}")
        if (
            not sample["reference"].strip()
            or not 0 <= sample["start_sample"] < sample["end_sample"]
        ):
            raise ValueError(
                f"invalid AppTek reference or boundary: {sample['origin_id']}"
            )
    return samples


def verify_manifest_sources(samples: list[dict], inventory: dict) -> None:
    sources = inventory["sources"]
    if len(samples) != 70 or len(sources) != 28:
        raise ValueError("AppTek clip or source count differs")
    if {item["source_file"] for item in samples} != set(sources):
        raise ValueError("AppTek sources differ from pinned LFS inventory")
    for accent in ACCENTS:
        accent_samples = [item for item in samples if item["accent"] == accent]
        general = [item for item in accent_samples if item["cohort"] == "general"]
        critical = [item for item in accent_samples if item["cohort"] == "critical"]
        if (
            len(general) != 2
            or len(critical) != 3
            or len({item["source_file"] for item in general}) != 1
            or len({item["source_file"] for item in critical}) != 1
            or general[0]["source_file"] == critical[0]["source_file"]
            or len({item["speaker_id"] for item in general}) != 2
            or {item["critical_label"] for item in critical} != set(CRITICAL)
        ):
            raise ValueError(f"AppTek call structure differs: {accent}")
    for item in samples:
        source = item["source_file"]
        if (
            not source.startswith(f"diarization/{item['accent']}/audio/")
            or item["source_sha256"] != sources[source]["sha256"]
        ):
            raise ValueError(f"AppTek source digest differs: {item['origin_id']}")


def run(
    manifest_path: Path,
    manifest_sha256: str,
    model_id: str,
    model_dir: Path,
    repo: Path,
    output: Path,
    inventory_path: Path,
) -> dict:
    attempts = output.with_suffix(".attempts.jsonl")
    if output.exists() or attempts.exists():
        raise FileExistsError("refusing to overwrite AppTek ASR evidence")
    samples = validate_corpus(manifest_path, manifest_sha256)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    builder_hash = sha256(repo / "scripts" / "translator_apptek_corpus.py")
    runner_hash = sha256(Path(__file__))
    loader_hash = sha256(repo / "scripts" / "translator_mdc_asr_run.py")
    if (
        len(samples) != 70
        or len({item["accent"] for item in samples}) != 14
        or manifest.get("builder_sha256") != builder_hash
        or manifest.get("inventory_sha256") != INVENTORY_SHA256
    ):
        raise ValueError("AppTek frozen corpus identity differs")
    verify_manifest_sources(samples, load_inventory(inventory_path))
    source_head = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    load_started = time.monotonic_ns()
    model, transcribe, model_identity = load_model(model_id, model_dir, repo)
    load_ms = (time.monotonic_ns() - load_started) / 1e6
    results = []
    with attempts.open("x", encoding="utf-8") as stream:
        for index, sample in enumerate(samples):
            row = dict(sample)
            started = time.monotonic_ns()
            try:
                if (
                    sha256(manifest_path.parent / sample["audio_file"])
                    != sample["sha256"]
                ):
                    raise ValueError("clip bytes changed before inference")
                row.update(
                    status="completed",
                    transcript=transcribe(
                        manifest_path.parent / sample["audio_file"], "en_us"
                    ),
                )
            except Exception as error:  # noqa: BLE001 - preserve failed attempts
                row.update(status="failed", error_type=type(error).__name__)
            row["elapsed_ms"] = (time.monotonic_ns() - started) / 1e6
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            results.append(row)
            if (index + 1) % 10 == 0:
                print(f"{model_id} {index + 1}/{len(samples)}", flush=True)
    report = {
        "schema": 1,
        "purpose": "apptek_publisher_reference_diagnostic",
        "model_id": model_id,
        "model_identity": model_identity,
        "manifest_sha256": manifest_sha256,
        "source_head": source_head,
        "harness_sha256": {
            "builder": builder_hash,
            "runner": runner_hash,
            "model_loader": loader_hash,
        },
        "load_ms": load_ms,
        "peak_process_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        // 1024,
        "results": results,
    }
    if (
        any(
            (sha256(repo / "scripts" / name) != expected)
            for name, expected in (
                ("translator_apptek_corpus.py", builder_hash),
                ("translator_apptek_asr_run.py", runner_hash),
                ("translator_mdc_asr_run.py", loader_hash),
            )
        )
        or sha256(manifest_path) != manifest_sha256
    ):
        raise RuntimeError("AppTek harness or corpus changed during inference")
    with output.open("x", encoding="utf-8") as destination:
        json.dump(report, destination, ensure_ascii=False, indent=2)
        destination.write("\n")
    del model
    if any(item["status"] != "completed" for item in results):
        raise RuntimeError("AppTek ASR run contains failed attempts")
    return report


def main() -> None:
    os.umask(0o077)
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--model-id", choices=("turbo", "qwen17"), required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    args = parser.parse_args()
    report = run(
        args.manifest,
        args.expected_manifest_sha256,
        args.model_id,
        args.model_dir,
        args.repo,
        args.output,
        args.inventory,
    )
    print(
        f"completed={sum(item['status'] == 'completed' for item in report['results'])}"
    )


if __name__ == "__main__":
    main()

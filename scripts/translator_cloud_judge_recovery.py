"""Separate, audit-preserving retry of failed cloud text-judge attempts."""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path

from translator_audio_adjudication import private_path
from translator_cloud_diagnostic import CLAUDE_MODEL, Providers, read_keys, summarize
from translator_mdc_asr_run import sha256


def failed_rows(rows: list[dict]) -> list[dict]:
    if len(rows) != 24:
        raise ValueError("primary receipt is not a full 24-case run")
    keys = [(row["origin_id"], row["condition"]) for row in rows]
    if len(set(keys)) != 24:
        raise ValueError("duplicate primary case")
    if any(row["claude"]["status"] not in ("ERROR", "COMPLETED") for row in rows):
        raise ValueError("primary judge state unsupported")
    return [row for row in rows if row["claude"]["status"] == "ERROR"]


def run_failed(
    rows: list[dict], output: Path, providers: Providers, primary_hash: str
) -> list[dict]:
    private_path(output, existing=False)
    targets = failed_rows(rows)
    if not targets:
        raise ValueError("no failed judgments to assess")
    descriptor = os.open(
        output, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600
    )
    attempts = []
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        for row in targets:
            variants = {
                label: {
                    "translation": row["arms"][arm]["mt_text"],
                    "own_asr": row["arms"][arm]["asr_text"],
                }
                for label, arm in row["mapping"].items()
            }
            result = {
                "type": "supplemental_judge_attempt",
                "origin_id": row["origin_id"],
                "condition": row["condition"],
                "primary_receipt_sha256": primary_hash,
                "model": CLAUDE_MODEL,
                "status": "ERROR",
            }
            started = time.monotonic_ns()
            try:
                result["verdicts"] = providers.claude(row["reference"], variants)
                result["status"] = "COMPLETED"
            except Exception as error:  # noqa: BLE001 - keep failed attempt, no hidden retry
                result["error_type"] = type(error).__name__
            result["elapsed_ms"] = (time.monotonic_ns() - started) / 1e6
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            attempts.append(result)
    return attempts


def combine(primary: list[dict], supplemental: list[dict]) -> list[dict]:
    combined = copy.deepcopy(primary)
    indexed = {(row["origin_id"], row["condition"]): row for row in combined}
    if len(indexed) != len(combined):
        raise ValueError("duplicate primary case")
    seen = set()
    for attempt in supplemental:
        key = (attempt["origin_id"], attempt["condition"])
        if (
            key in seen
            or key not in indexed
            or indexed[key]["claude"]["status"] != "ERROR"
        ):
            raise ValueError("supplemental attempt not bound to failed primary case")
        seen.add(key)
        if attempt["status"] == "COMPLETED":
            indexed[key]["claude"] = {
                "status": "COMPLETED",
                "source": "supplemental",
                "verdicts": attempt["verdicts"],
            }
            if all(
                indexed[key][name]["status"] == "COMPLETED"
                for name in ("openai", "google", "claude")
            ):
                indexed[key]["status"] = "COMPLETED"
    return combined


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--primary-sha256", required=True)
    parser.add_argument("--secrets-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sha256(args.primary) != args.primary_sha256:
        raise ValueError("primary receipt hash changed")
    rows = [
        json.loads(line)
        for line in args.primary.read_text(encoding="utf-8").splitlines()
    ]
    private_path(args.output, existing=False)
    providers = Providers(read_keys(args.secrets_file))
    try:
        attempts = run_failed(rows, args.output, providers, args.primary_sha256)
    finally:
        providers.close()
    summary = summarize(combine(rows, attempts))
    print(
        json.dumps(
            {
                "supplemental_attempts": len(attempts),
                "supplemental_completed": sum(
                    row["status"] == "COMPLETED" for row in attempts
                ),
                "combined": summary,
                "primary_sha256": args.primary_sha256,
                "supplemental_sha256": sha256(args.output),
                "runner_sha256": sha256(Path(__file__)),
            }
        )
    )
    return int(not summary["complete_24"])


if __name__ == "__main__":
    raise SystemExit(main())

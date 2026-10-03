"""Run one pinned local ASR model against one frozen MDC corpus split."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

TURBO_SHA256 = "e76620f83d5f5b69efd3d87e3dc180c1bd21df9fbebacfd4335e5e1efcc018da"
TURBO_DIRECTORY_SHA256 = (
    "2bbf4f77213aae02d99ce993cc93880b05d4204bb40c49d796b02b5f6b6d06a1"
)
QWEN17_WEIGHT_SHA256 = (
    "a4cd1f1a04d90b757dc7f7dd26254e69a013b19e80efe590a83c6a3bde8608d6",
    "6e0b9d9e09e2e0238e7ef3cc8a484ab387e91b90f1900bedf88bc92d7929ccfc",
)
QWEN17_DIRECTORY_SHA256 = (
    "fed91fc61c395e5cf9e851742942c13e776df1fd1afcdc42a1e4a93a93cc7a8a"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def directory_sha256(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(directory.rglob("*")):
        if path.is_file() and ".cache" not in path.relative_to(directory).parts:
            digest.update(str(path.relative_to(directory)).encode())
            digest.update(b"\0")
            digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def validate_manifest(path: Path, expected_hash: str) -> list[dict]:
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_hash:
        raise ValueError("frozen manifest hash changed")
    manifest = json.loads(data)
    if manifest["schema"] != 1 or manifest["purpose"] not in (
        "asr_model_selection",
        "asr_independent_holdout",
    ):
        raise ValueError("unsupported corpus contract")
    samples = manifest["samples"]
    if not samples or len({item["audio_file"] for item in samples}) != len(samples):
        raise ValueError("empty or duplicate corpus audio")
    for sample in samples:
        relative = Path(sample["audio_file"])
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or sha256(path.parent / relative) != sample["sha256"]
        ):
            raise ValueError(f"frozen audio changed: {sample['origin_id']}")
    return samples


def load_model(model_id: str, model_dir: Path, repo: Path):
    if model_id == "turbo":
        if sha256(model_dir / "model.bin") != TURBO_SHA256:
            raise ValueError("Turbo model hash changed")
        if directory_sha256(model_dir) != TURBO_DIRECTORY_SHA256:
            raise ValueError("Turbo config or tokenizer hash changed")
        sys.path.insert(0, str(repo / "sidecar"))
        from translator_sidecar.local.cuda_runtime import configure_cuda_runtime

        configure_cuda_runtime()
        from faster_whisper import WhisperModel

        model = WhisperModel(
            str(model_dir),
            device="cuda",
            compute_type="float16",
            local_files_only=True,
            num_workers=1,
        )

        def transcribe(path: Path, language: str) -> str:
            segments, _ = model.transcribe(
                str(path),
                language="ru" if language == "ru_ru" else "en",
                beam_size=5,
                vad_filter=False,
                condition_on_previous_text=False,
            )
            return "".join(segment.text for segment in segments).strip()

        return (
            model,
            transcribe,
            {
                "model_bin_sha256": TURBO_SHA256,
                "directory_sha256": TURBO_DIRECTORY_SHA256,
                "beam_size": 5,
                "vad_filter": False,
            },
        )
    if model_id == "qwen17":
        weights = sorted(model_dir.glob("*.safetensors"))
        if tuple(sha256(path) for path in weights) != QWEN17_WEIGHT_SHA256:
            raise ValueError("Qwen 1.7B weight hashes changed")
        if directory_sha256(model_dir) != QWEN17_DIRECTORY_SHA256:
            raise ValueError("Qwen 1.7B model directory changed")
        import torch
        from qwen_asr import Qwen3ASRModel

        torch.cuda.set_per_process_memory_fraction(0.72, device=0)
        model = Qwen3ASRModel.from_pretrained(
            str(model_dir),
            dtype=torch.bfloat16,
            device_map="cuda:0",
            max_inference_batch_size=1,
            max_new_tokens=256,
        )
        torch.cuda.synchronize()

        def transcribe(path: Path, language: str) -> str:
            response = model.transcribe(
                audio=str(path),
                language="Russian" if language == "ru_ru" else "English",
            )
            torch.cuda.synchronize()
            return response[0].text.strip()

        return (
            model,
            transcribe,
            {
                "weight_sha256": QWEN17_WEIGHT_SHA256,
                "directory_sha256": QWEN17_DIRECTORY_SHA256,
                "max_new_tokens": 256,
            },
        )
    raise ValueError(f"unsupported model: {model_id}")


def run(
    manifest: Path,
    expected_manifest_hash: str,
    model_id: str,
    model_dir: Path,
    repo: Path,
    output: Path,
) -> dict:
    attempts = output.with_suffix(".attempts.jsonl")
    if output.exists() or attempts.exists():
        raise FileExistsError("refusing to overwrite ASR evidence")
    samples = validate_manifest(manifest, expected_manifest_hash)
    manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
    corpus_hash = sha256(repo / "scripts" / "translator_mdc_corpus.py")
    runner_hash = sha256(Path(__file__))
    if manifest_data.get("builder_sha256") != corpus_hash:
        raise ValueError("corpus builder bytes differ from frozen manifest")
    source_head = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    started = time.monotonic_ns()
    model, transcribe, identity = load_model(model_id, model_dir, repo)
    load_ms = (time.monotonic_ns() - started) / 1e6
    results = []
    peak_rss_mib = 0
    with attempts.open("x", encoding="utf-8") as stream:
        for index, sample in enumerate(samples):
            result = {
                key: sample[key]
                for key in (
                    "origin_id",
                    "language",
                    "split",
                    "speaker_id",
                    "reference",
                    "condition",
                    "critical_labels",
                    "audio_file",
                )
            }
            started = time.monotonic_ns()
            try:
                result.update(
                    status="completed",
                    transcript=transcribe(
                        manifest.parent / sample["audio_file"], sample["language"]
                    ),
                )
            except Exception as error:  # noqa: BLE001 - retain every failed attempt
                result.update(status="failed", error_type=type(error).__name__)
            result["elapsed_ms"] = (time.monotonic_ns() - started) / 1e6
            peak_rss_mib = max(
                peak_rss_mib, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
            )
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            stream.flush()
            results.append(result)
            if (index + 1) % 20 == 0:
                print(
                    f"{model_id}:{sample['split']} {index + 1}/{len(samples)}",
                    flush=True,
                )
    report = {
        "schema": 1,
        "model_id": model_id,
        "model_identity": identity,
        "manifest_sha256": expected_manifest_hash,
        "source_head": source_head,
        "harness_sha256": {"corpus": corpus_hash, "runner": runner_hash},
        "load_ms": load_ms,
        "peak_process_rss_mib_sampled": peak_rss_mib,
        "results": results,
    }
    if (
        sha256(Path(__file__)) != runner_hash
        or sha256(repo / "scripts" / "translator_mdc_corpus.py") != corpus_hash
    ):
        raise RuntimeError("evaluator changed during model run")
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    del model
    if any(item["status"] != "completed" for item in results):
        raise RuntimeError("ASR run contains failed attempts")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--model-id", choices=("turbo", "qwen17"), required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(
        args.manifest,
        args.expected_manifest_sha256,
        args.model_id,
        args.model_dir,
        args.repo,
        args.output,
    )
    print(
        f"{args.model_id}: completed={sum(item['status'] == 'completed' for item in report['results'])}"
    )

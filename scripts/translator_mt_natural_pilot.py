"""Run a small, text-only paired NLLB/Hy-MT2 diagnostic on frozen phrases."""

from __future__ import annotations

import hashlib
import json
import statistics
import time
from pathlib import Path
from urllib.request import Request, urlopen

from sacrebleu.metrics import CHRF
from translator_sidecar.local.model_lease import VerifiedModelSource
from translator_sidecar.local.model_manifest import load_manifest
from translator_sidecar.local.mt import NllbTranslator
from translator_sidecar.provider_contract import Language, TranslationMode

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "sidecar/tests/quality_corpus/mt-natural-pilot-20260924.json"
OUTPUT = ROOT / "docs/benchmarks/mt-natural-pilot-20260924.json"
MODEL_ID = "hf.co/tencent/Hy-MT2-1.8B-GGUF:Q4_K_M"
SERVER = "http://127.0.0.1:11578/v1/chat/completions"


def hy_translate(text: str, target: Language) -> str:
    target_name = "English" if target is Language.EN else "Russian"
    prompt = (
        f"Translate the following text into {target_name}. Note that you "
        "should only output the translated result without any additional "
        f"explanation:\n{text}"
    )
    body = json.dumps(
        {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "top_p": 0.6,
            "top_k": 20,
            "repeat_penalty": 1.05,
            "max_tokens": 128,
            "stream": False,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = Request(SERVER, data=body, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=45) as response:
        result = json.load(response)
    translated = result["choices"][0]["message"]["content"].strip()
    if not translated:
        raise RuntimeError("Hy-MT2 returned an empty translation")
    return translated


def main() -> None:
    corpus_bytes = CORPUS.read_bytes()
    corpus = json.loads(corpus_bytes)
    cases = corpus["cases"]
    manifest = load_manifest(ROOT / "models/manifest.json")
    nllb = NllbTranslator.load(
        VerifiedModelSource(manifest, "nllb-200-distilled-600m-ct2-int8"),
        device="cpu",
    )
    rows = []
    try:
        for source in (Language.RU, Language.EN):
            target = Language.EN if source is Language.RU else Language.RU
            warmup = "Проверьте звук." if source is Language.RU else "Check the audio."
            nllb.translate(
                warmup,
                source_language=source,
                target_language=target,
                mode=TranslationMode.QUALITY_FIRST,
            )
            hy_translate(warmup, target)
        for case in cases:
            source = Language(case["source_language"])
            target = Language.EN if source is Language.RU else Language.RU
            start = time.monotonic_ns()
            baseline = nllb.translate(
                case["source"],
                source_language=source,
                target_language=target,
                mode=TranslationMode.QUALITY_FIRST,
            )
            baseline_ms = (time.monotonic_ns() - start) / 1e6
            start = time.monotonic_ns()
            candidate = hy_translate(case["source"], target)
            candidate_ms = (time.monotonic_ns() - start) / 1e6
            rows.append(
                {
                    **case,
                    "nllb": {"output": baseline, "latency_ms": round(baseline_ms, 2)},
                    "hy_mt2": {
                        "output": candidate,
                        "latency_ms": round(candidate_ms, 2),
                    },
                }
            )
    finally:
        nllb.close()

    metric = CHRF(word_order=2)
    scores = {}
    for language in ("ru", "en"):
        selected = [row for row in rows if row["source_language"] == language]
        references = [row["reference"] for row in selected]
        scores[language] = {}
        for name in ("nllb", "hy_mt2"):
            outputs = [row[name]["output"] for row in selected]
            latencies = [row[name]["latency_ms"] for row in selected]
            scores[language][name] = {
                "chrf2": round(metric.corpus_score(outputs, [references]).score, 2),
                "median_latency_ms": round(statistics.median(latencies), 2),
            }
    report = {
        "scope": "text-only diagnostic; not a release holdout or full-chain E2E",
        "corpus_sha256": hashlib.sha256(corpus_bytes).hexdigest(),
        "nllb_manifest_sha256": hashlib.sha256(
            (ROOT / "models/manifest.json").read_bytes()
        ).hexdigest(),
        "candidate_model": MODEL_ID,
        "candidate_server": "llama-server with GGUF Jinja template, CPU, 2 threads",
        "decoding": "temperature=0, top_p=0.6, top_k=20, repeat_penalty=1.05",
        "scores": scores,
        "cases": rows,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(scores, ensure_ascii=False))


if __name__ == "__main__":
    main()

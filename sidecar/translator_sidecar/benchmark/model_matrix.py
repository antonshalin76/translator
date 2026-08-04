"""Approved model candidates for local quality and latency diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


CandidateRole = Literal["asr", "tts", "endpointing"]
_FASTER_WHISPER_URL = "https://github.com/SYSTRAN/faster-whisper"
_ALREADY_INTEGRATED = "already integrated"
_APACHE_2_LICENSE = "Apache-2.0"


@dataclass(frozen=True, slots=True)
class ModelCandidate:
    id: str
    role: CandidateRole
    runtime: str
    repository: str
    license: str
    languages: tuple[str, ...]
    purpose: str
    source_urls: tuple[str, ...]
    strengths: tuple[str, ...]
    risks: tuple[str, ...]
    priority: int | None = None

    def to_report(self) -> dict[str, object]:
        return {
            "id": self.id,
            "role": self.role,
            "runtime": self.runtime,
            "repository": self.repository,
            "license": self.license,
            "languages": list(self.languages),
            "purpose": self.purpose,
            "source_urls": list(self.source_urls),
            "strengths": list(self.strengths),
            "risks": list(self.risks),
            "priority": self.priority,
        }


_ASR_CANDIDATES = (
    ModelCandidate(
        id="faster-whisper-small",
        role="asr",
        runtime="local_provider",
        repository="Systran/faster-whisper-small",
        license="MIT",
        languages=("ru", "en"),
        purpose="current latency baseline",
        source_urls=(_FASTER_WHISPER_URL,),
        strengths=(_ALREADY_INTEGRATED, "fast baseline"),
        risks=("quality bottleneck on long/noisy phrases",),
        priority=10,
    ),
    ModelCandidate(
        id="faster-whisper-large-v3",
        role="asr",
        runtime="local_provider",
        repository="Systran/faster-whisper-large-v3",
        license="MIT",
        languages=("ru", "en"),
        purpose="current high-quality Whisper baseline",
        source_urls=(_FASTER_WHISPER_URL,),
        strengths=(_ALREADY_INTEGRATED, "strong multilingual fallback"),
        risks=("heavier than the live latency target allows by default",),
        priority=20,
    ),
    ModelCandidate(
        id="faster-whisper-large-v3-turbo-ct2",
        role="asr",
        runtime="faster_whisper_ct2",
        repository="deepdml/faster-whisper-large-v3-turbo-ct2",
        license="MIT",
        languages=("ru", "en"),
        purpose="drop-in quality/speed candidate for the existing adapter",
        source_urls=(
            "https://huggingface.co/openai/whisper-large-v3-turbo",
            _FASTER_WHISPER_URL,
        ),
        strengths=("lower risk than changing ASR architecture", "faster than large-v3"),
        risks=("must be pinned in the manifest before live use",),
        priority=30,
    ),
    ModelCandidate(
        id="gigaam-v3-e2e-rnnt",
        role="asr",
        runtime="gigaam",
        repository="ai-sage/GigaAM-v3",
        license="MIT",
        languages=("ru",),
        purpose="Russian quality candidate with punctuation and normalization",
        source_urls=("https://github.com/salute-developers/GigaAM",),
        strengths=("strong Russian community signal", "e2e punctuation/normalization"),
        risks=("streaming path and CPU preprocessing cost must be measured",),
        priority=40,
    ),
    ModelCandidate(
        id="gigaam-v3-e2e-ctc",
        role="asr",
        runtime="gigaam",
        repository="ai-sage/GigaAM-v3",
        license="MIT",
        languages=("ru",),
        purpose="Russian speed candidate with punctuation and normalization",
        source_urls=("https://github.com/salute-developers/GigaAM",),
        strengths=("faster decoding lane for Russian", "e2e punctuation/normalization"),
        risks=("quality on names/short words must be measured",),
        priority=50,
    ),
    ModelCandidate(
        id="qwen3-asr-0.6b-hf",
        role="asr",
        runtime="transformers",
        repository="Qwen/Qwen3-ASR-0.6B-hf",
        license=_APACHE_2_LICENSE,
        languages=("ru", "en"),
        purpose="multilingual low-latency Qwen3-ASR candidate",
        source_urls=("https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf",),
        strengths=("streaming/offline model family", "RU and EN in one model"),
        risks=("Transformers runtime and VRAM fit must be validated locally",),
        priority=60,
    ),
    ModelCandidate(
        id="qwen3-asr-1.7b-hf",
        role="asr",
        runtime="transformers",
        repository="Qwen/Qwen3-ASR-1.7B-hf",
        license=_APACHE_2_LICENSE,
        languages=("ru", "en"),
        purpose="multilingual quality reference for Qwen3-ASR",
        source_urls=("https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf",),
        strengths=("quality reference among open ASR candidates",),
        risks=("likely too heavy for the first live duplex default",),
        priority=70,
    ),
    ModelCandidate(
        id="parakeet-unified-en-0.6b",
        role="asr",
        runtime="nemo",
        repository="nvidia/parakeet-unified-en-0.6b",
        license="NVIDIA Open Model License",
        languages=("en",),
        purpose="English streaming ASR candidate",
        source_urls=("https://huggingface.co/nvidia/parakeet-unified-en-0.6b",),
        strengths=("streaming/offline English ASR", "low-latency design"),
        risks=("English-only source lane",),
        priority=80,
    ),
    ModelCandidate(
        id="parakeet-tdt-0.6b-v3",
        role="asr",
        runtime="nemo",
        repository="nvidia/parakeet-tdt-0.6b-v3",
        license="CC-BY-4.0",
        languages=("ru", "en"),
        purpose="multilingual Parakeet quality benchmark",
        source_urls=("https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3",),
        strengths=("multilingual NVIDIA ASR", "long-audio support"),
        risks=("community signal says Russian may trail GigaAM",),
        priority=90,
    ),
)

_TTS_CANDIDATES = (
    ModelCandidate(
        id="piper-medium",
        role="tts",
        runtime="local_provider",
        repository="rhasspy/piper-voices",
        license="MIT",
        languages=("ru", "en"),
        purpose="current live TTS baseline",
        source_urls=("https://rhasspy.github.io/piper-samples/",),
        strengths=(_ALREADY_INTEGRATED, "very fast fallback"),
        risks=("naturalness, prosody, and homographs are weak spots",),
        priority=10,
    ),
    ModelCandidate(
        id="kokoro-82m",
        role="tts",
        runtime="kokoro",
        repository="hexgrad/Kokoro-82M",
        license=_APACHE_2_LICENSE,
        languages=("en",),
        purpose="English TTS speed/quality candidate",
        source_urls=("https://huggingface.co/hexgrad/Kokoro-82M",),
        strengths=("small model", "strong English quality signal"),
        risks=("not a Russian TTS replacement",),
        priority=20,
    ),
    ModelCandidate(
        id="silero-v5_5-ru",
        role="tts",
        runtime="silero",
        repository="snakers4/silero-models",
        license="MIT",
        languages=("ru",),
        purpose="Russian TTS stress/homograph candidate",
        source_urls=("https://github.com/snakers4/silero-models",),
        strengths=("Russian stress and homograph controls",),
        risks=("customer-perceived naturalness must be measured against newer TTS",),
        priority=30,
    ),
    ModelCandidate(
        id="qwen3-tts-0.6b-customvoice",
        role="tts",
        runtime="qwen_tts",
        repository="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        license=_APACHE_2_LICENSE,
        languages=("ru", "en"),
        purpose="single-engine RU/EN quality TTS candidate",
        source_urls=("https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",),
        strengths=("streaming TTS family", "one engine for both directions"),
        risks=("Russian stress quality must be measured before live default",),
        priority=40,
    ),
    ModelCandidate(
        id="qwen3-tts-1.7b-customvoice",
        role="tts",
        runtime="qwen_tts",
        repository="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        license=_APACHE_2_LICENSE,
        languages=("ru", "en"),
        purpose="single-engine RU/EN quality reference",
        source_urls=("https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",),
        strengths=("higher quality reference for TTS comparisons",),
        risks=("heavy for live duplex on 12GB VRAM",),
        priority=50,
    ),
    ModelCandidate(
        id="xtts-v2",
        role="tts",
        runtime="coqui_xtts",
        repository="coqui/XTTS-v2",
        license="CPML",
        languages=("ru", "en"),
        purpose="offline quality comparator, not default",
        source_urls=("https://huggingface.co/coqui/XTTS-v2",),
        strengths=("multilingual voice-cloning reference",),
        risks=("license and latency make it unsuitable as default",),
        priority=60,
    ),
    ModelCandidate(
        id="f5-tts-russian",
        role="tts",
        runtime="f5_tts",
        repository="Misha24-10/F5-TTS_RUSSIAN",
        license="needs-review",
        languages=("ru",),
        purpose="community-mentioned Russian TTS comparator",
        source_urls=("https://huggingface.co/Misha24-10/F5-TTS_RUSSIAN",),
        strengths=("fresh Russian community signal",),
        risks=("license/provenance must be reviewed before redistribution",),
        priority=70,
    ),
)

_CANDIDATES = _ASR_CANDIDATES + _TTS_CANDIDATES
_BY_ID = {candidate.id: candidate for candidate in _CANDIDATES}

if len(_BY_ID) != len(_CANDIDATES):
    raise RuntimeError("duplicate benchmark model candidate id")


def _sorted(candidates: tuple[ModelCandidate, ...]) -> list[ModelCandidate]:
    return sorted(
        candidates,
        key=lambda candidate: (
            candidate.priority if candidate.priority is not None else 10_000,
            candidate.id,
        ),
    )


def asr_candidates() -> list[ModelCandidate]:
    return _sorted(_ASR_CANDIDATES)


def tts_candidates() -> list[ModelCandidate]:
    return _sorted(_TTS_CANDIDATES)


def asr_candidate_ids() -> list[str]:
    return [candidate.id for candidate in asr_candidates()]


def tts_candidate_ids() -> list[str]:
    return [candidate.id for candidate in tts_candidates()]


def default_asr_candidate_ids() -> list[str]:
    return asr_candidate_ids()


def default_executable_asr_candidate_ids() -> list[str]:
    return [
        candidate.id
        for candidate in asr_candidates()
        if candidate.runtime == "local_provider"
    ]


def default_tts_candidate_ids() -> list[str]:
    return [
        "piper-medium",
        "kokoro-82m",
        "silero-v5_5-ru",
        "qwen3-tts-0.6b-customvoice",
    ]


def candidate_by_id(
    model_id: str, *, role: CandidateRole | None = None
) -> ModelCandidate:
    candidate = _BY_ID[model_id]
    if role is not None and candidate.role != role:
        raise KeyError(model_id)
    return candidate


def candidate_report(
    model_ids: list[str],
    *,
    role: CandidateRole,
    include_unknown: bool = False,
) -> list[dict[str, object]]:
    report: list[dict[str, object]] = []
    for model_id in model_ids:
        try:
            candidate = candidate_by_id(model_id, role=role)
        except KeyError:
            if not include_unknown:
                raise
            report.append(
                {
                    "id": model_id,
                    "role": role,
                    "runtime": "unknown",
                    "repository": None,
                    "license": None,
                    "languages": [],
                    "purpose": "custom caller-supplied candidate",
                    "source_urls": [],
                    "strengths": [],
                    "risks": ["not present in the approved benchmark matrix"],
                    "priority": None,
                }
            )
        else:
            report.append(candidate.to_report())
    return report


def registry_report() -> dict[str, list[dict[str, object]]]:
    return {
        "asr": [candidate.to_report() for candidate in asr_candidates()],
        "tts": [candidate.to_report() for candidate in tts_candidates()],
    }

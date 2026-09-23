"""Freeze a speaker-disjoint MDC Spontaneous Speech 5.0 ASR comparison corpus."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import subprocess
import tarfile
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

ARCHIVES = {
    "ru": "aa0910f5fef7f24afccf5bca075995f0b7a970d96da72626c80507d629d05a6f",
    "en": "390b5a54c9afe0cc01da039ad206248f85682f247dd2b27d4cc0ab9a68e860b6",
}
CRITICAL_TEST = {
    "ru": {
        "names": {"71366", "71599", "72227", "72230", "74019"},
        "numbers": {"72019", "72020", "72182", "74007", "79465"},
    },
    "en": {
        "names": {"3", "20216", "67620", "70813", "71793"},
        "numbers": {"20253", "20265", "70798", "78643", "80604"},
    },
}
NEGATION = {
    "ru": re.compile(r"\b(?:не|нет|никогда|нельзя|никак|ни)\b", re.IGNORECASE),
    "en": re.compile(r"\b(?:not|no|never|nothing|don't|can't|won't)\b", re.IGNORECASE),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def rank(locale: str, split: str, audio_id: str) -> bytes:
    return hashlib.sha256(f"mdc-sps5-v1|{locale}|{split}|{audio_id}".encode()).digest()


def eligible_rows(rows: list[dict[str, str]], split: str) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if row["split"] == split
        and row["client_id"]
        and row["transcription"].strip()
        and not row["quality_tags"]
        and 3000 <= int(row["duration_ms"]) <= 20000
        and 3 <= len(row["transcription"].split()) <= 65
    ]


def select(rows: list[dict[str, str]], locale: str, split: str) -> list[dict[str, str]]:
    eligible = eligible_rows(rows, split)
    if split == "dev" and locale == "ru":
        return sorted(eligible, key=lambda row: rank(locale, split, row["audio_id"]))
    target, cap = (40, 2) if split == "dev" else (40, 8 if locale == "ru" else 3)
    by_id = {row["audio_id"]: row for row in eligible}
    forced = set().union(*CRITICAL_TEST[locale].values()) if split == "test" else set()
    if forced - by_id.keys():
        raise ValueError(
            f"missing critical {locale} test rows: {sorted(forced - by_id.keys())}"
        )
    selected = [by_id[audio_id] for audio_id in sorted(forced)]
    counts = Counter(row["client_id"] for row in selected)
    if any(count > cap for count in counts.values()):
        raise ValueError(f"critical {locale} rows exceed speaker cap")
    remaining = [row for row in eligible if row["audio_id"] not in forced]
    while len(selected) < target:
        candidates = [row for row in remaining if counts[row["client_id"]] < cap]
        if not candidates:
            raise ValueError(f"insufficient speaker-balanced {locale}/{split} rows")
        chosen = min(
            candidates,
            key=lambda row: (
                counts[row["client_id"]],
                rank(locale, split, row["audio_id"]),
            ),
        )
        selected.append(chosen)
        counts[chosen["client_id"]] += 1
        remaining.remove(chosen)
    return sorted(selected, key=lambda row: rank(locale, split, row["audio_id"]))


def select_noise(rows: list[dict[str, str]], locale: str) -> set[str]:
    chosen = []
    speakers = Counter()
    groups = (
        lambda row: row["audio_id"] in CRITICAL_TEST[locale]["names"],
        lambda row: row["audio_id"] in CRITICAL_TEST[locale]["numbers"],
        lambda row: bool(NEGATION[locale].search(row["transcription"])),
        lambda row: True,
    )
    for group in groups:
        for _ in range(2):
            available = [
                row for row in rows if row["audio_id"] not in chosen and group(row)
            ]
            if not available:
                raise ValueError(f"insufficient noise cases: {locale}")
            row = min(
                available,
                key=lambda item: (
                    speakers[item["client_id"]],
                    rank(locale, "noise", item["audio_id"]),
                ),
            )
            chosen.append(row["audio_id"])
            speakers[row["client_id"]] += 1
    return set(chosen)


def noise_at_10db(speech: np.ndarray, key: str) -> tuple[np.ndarray, float]:
    centered = speech.astype(np.float64) - float(np.mean(speech))
    speech_rms = float(np.sqrt(np.mean(centered**2)))
    if speech_rms < 1e-5:
        raise ValueError("silent speech cannot be noise-augmented")
    rng = np.random.default_rng(
        int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
    )
    amplitude = np.abs(np.fft.rfft(centered))
    phase = rng.uniform(-np.pi, np.pi, len(amplitude))
    noise = np.fft.irfft(amplitude * np.exp(1j * phase), n=len(speech))
    noise -= np.mean(noise)
    noise *= speech_rms / (10 ** (10 / 20) * np.sqrt(np.mean(noise**2)))
    mixed = speech.astype(np.float64) + noise
    peak = float(np.max(np.abs(mixed)))
    if peak > 0.99:
        mixed *= 0.99 / peak
    return mixed.astype(np.float32), 20 * np.log10(
        speech_rms / np.sqrt(np.mean(noise**2))
    )


def realized_snr_db(speech: np.ndarray, mixed: np.ndarray) -> float:
    clean = speech.astype(np.float64)
    recorded = mixed.astype(np.float64)
    gain = float(np.dot(clean, recorded) / np.dot(clean, clean))
    residual = recorded - gain * clean
    return float(
        20
        * np.log10(
            np.sqrt(np.mean((gain * clean) ** 2)) / np.sqrt(np.mean(residual**2))
        )
    )


def build(archives: Path, output: Path) -> dict[str, dict]:
    if output.exists():
        raise FileExistsError(output)
    tables: dict[str, list[dict[str, str]]] = {}
    for locale, expected in ARCHIVES.items():
        path = archives / f"sps5-{locale}.tar.gz"
        if sha256(path) != expected:
            raise ValueError(f"archive hash changed: {locale}")
        with tarfile.open(path, "r:gz") as archive:
            member = next(
                item
                for item in archive
                if item.name.endswith(f"/ss-corpus-{locale}.tsv")
            )
            tables[locale] = list(
                csv.DictReader(
                    io.TextIOWrapper(archive.extractfile(member), encoding="utf-8"),
                    delimiter="\t",
                )
            )
    chosen = {
        (locale, split): select(tables[locale], locale, split)
        for locale in ARCHIVES
        for split in ("dev", "test")
    }
    for locale in ARCHIVES:
        dev_speakers = {row["client_id"] for row in chosen[locale, "dev"]}
        test_speakers = {row["client_id"] for row in chosen[locale, "test"]}
        if dev_speakers & test_speakers:
            raise ValueError(f"speaker leakage across official splits: {locale}")
    output.mkdir(mode=0o700)
    reports = {}
    for split in ("dev", "test"):
        destination = output / split
        destination.mkdir(mode=0o700)
        clips = destination / "clips"
        clips.mkdir(mode=0o700)
        samples = []
        for locale in ARCHIVES:
            noise_ids = (
                select_noise(chosen[locale, split], locale)
                if split == "test"
                else set()
            )
            path = archives / f"sps5-{locale}.tar.gz"
            with tarfile.open(path, "r:gz") as archive:
                members = {
                    Path(item.name).name: item
                    for item in archive
                    if item.name.endswith(".mp3")
                }
                for row in chosen[locale, split]:
                    filename = row["audio_file"]
                    if Path(filename).name != filename or filename not in members:
                        raise ValueError(
                            f"invalid source audio file: {locale}/{row['audio_id']}"
                        )
                    source = archive.extractfile(members[filename]).read()
                    origin = f"{locale}-{row['audio_id']}"
                    clean = clips / f"{origin}-clean.wav"
                    conversion = subprocess.run(
                        [
                            "ffmpeg",
                            "-nostdin",
                            "-v",
                            "error",
                            "-i",
                            "pipe:0",
                            "-ac",
                            "1",
                            "-ar",
                            "16000",
                            "-c:a",
                            "pcm_s16le",
                            "-f",
                            "wav",
                            str(clean),
                        ],
                        input=source,
                        capture_output=True,
                        check=False,
                    )
                    if conversion.returncode:
                        raise RuntimeError(f"audio conversion failed: {origin}")
                    critical = [
                        kind
                        for kind, ids in CRITICAL_TEST[locale].items()
                        if split == "test" and row["audio_id"] in ids
                    ]
                    if NEGATION[locale].search(row["transcription"]):
                        critical.append("negation_candidate")
                    base = {
                        "origin_id": origin,
                        "language": "ru_ru" if locale == "ru" else "en_us",
                        "split": split,
                        "speaker_id": row["client_id"],
                        "reference": row["transcription"],
                        "critical_labels": critical,
                        "source_audio_sha256": hashlib.sha256(source).hexdigest(),
                    }
                    samples.append(
                        {
                            **base,
                            "condition": "clean",
                            "audio_file": str(clean.relative_to(destination)),
                            "sha256": sha256(clean),
                        }
                    )
                    if row["audio_id"] in noise_ids:
                        speech, rate = sf.read(clean, dtype="float32")
                        if rate != 16000 or speech.ndim != 1:
                            raise ValueError(
                                f"unexpected decoded audio shape: {origin}"
                            )
                        noisy, _ = noise_at_10db(speech, origin)
                        noise_path = clips / f"{origin}-noise10.wav"
                        sf.write(noise_path, noisy, 16000, subtype="PCM_16")
                        recorded, recorded_rate = sf.read(noise_path, dtype="float32")
                        if recorded_rate != 16000:
                            raise ValueError(f"unexpected noise audio rate: {origin}")
                        measured = realized_snr_db(speech, recorded)
                        if not 9.5 <= measured <= 10.5:
                            raise ValueError(
                                f"stored noise SNR outside tolerance: {origin}"
                            )
                        samples.append(
                            {
                                **base,
                                "condition": "speech_shaped_noise_10db",
                                "audio_file": str(noise_path.relative_to(destination)),
                                "sha256": sha256(noise_path),
                                "realized_snr_db": measured,
                            }
                        )
        manifest = {
            "schema": 1,
            "purpose": "asr_model_selection"
            if split == "dev"
            else "asr_independent_holdout",
            "dataset": "Mozilla Common Voice Spontaneous Speech 5.0",
            "source_archive_sha256": ARCHIVES,
            "builder_sha256": sha256(Path(__file__)),
            "selection": "official speaker-disjoint dev/test; 3-20s; 3-65 words; no quality tags; deterministic speaker balancing; transcript-tagged critical cases",
            "limitations": "Critical labels checked against written transcripts, not manually adjudicated against audio; no product-chain or release evidence.",
            "samples": samples,
        }
        manifest_path = destination / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        reports[split] = {
            "manifest_sha256": sha256(manifest_path),
            "samples": len(samples),
            "clean_speakers": {
                locale: len(
                    {
                        sample["speaker_id"]
                        for sample in samples
                        if sample["language"].startswith(locale)
                        and sample["condition"] == "clean"
                    }
                )
                for locale in ARCHIVES
            },
        }
    return reports


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archives", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.archives, args.output), indent=2))

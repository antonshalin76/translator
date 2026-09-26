"""Freeze an AppTek publisher-reference diagnostic before ASR inference."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import re
import tempfile
import wave
from pathlib import Path

DATASET_REVISION = "b98967d9946f7f59f58d08624a2a00fe98fe0219"
DATASET = "apptek-com/apptek_callcenter_dialogues"
INVENTORY_SHA256 = "869b00a1654823360aaf51c81e3d0c876712a6d9afd9c1fdc7e912b0cc30cedd"
METADATA_SHA256 = {
    "en-AU": "51ef07de90266f5da3cb902f8e63f25af661ead1f43812dac5502e1ee5c31ae6",
    "en-CA": "e96d13b4d66774b1c3df08335c7599e1405965ffaf79d08c6545ab6fa3bccdc6",
    "en-CN": "b616cb6d091d8d609079b99da406d69cc343bdbe0c23c1d211e5fa1976d5ea9f",
    "en-GB": "72cdb5ee475b4a2b41ec970b6a84fe110ab7b60969d2d10248b9d9d5100090cb",
    "en-GB_SCT": "2b79ca8945989ceab44d6e8b14db5ef5a0e62179393fc2c5635231c739f82c25",
    "en-GB_WLS": "88fb3d76cddbf6d785c051280c3cbb4cd6678e55a5e34452d862cb18b1936102",
    "en-IE": "83f52b0b530bfb416493a68da0a48692f86a0818721091d78c8c1680500dacc7",
    "en-IN": "989c5db3097426b8f64263487db4f9d37357ccc2d82b20d9d5dd82bcbfd51672",
    "en-MX": "7d3f9e7920991eebdc6ea9877daabe42647f9dba0dce8e387e0db7bbd45ffce9",
    "en-SG": "f1cc73e3cc5e8a39884b0296590089502eb4d5853bf11e0a6cf17cacbc965df1",
    "en-US_Aave": "72370e6f98a8a73d987f9f8e9e242e44c34ca76ee8baf3e6800242429c66c01a",
    "en-US_General": "2c37db51246dd8be8c3e1946afd8ee5ad15ef5857db4e1a39393abc711524d72",
    "en-US_Southern": "0430583c0ce7de77dcf489ea2ef6c8202800edf9d7e8a2b2bacc46cd7ce4d495",
    "en-ZA": "e8996164682fdcae3d9d05677d8ad6985ed1fbfa2ac8712c7e5cff6cca49dcd4",
}
ACCENTS = tuple(METADATA_SHA256)
EXCLUDED_DEVELOPMENT_FILE = "diarization/en-AU/audio/en_AU_Agriculture_1586330.wav"
NEGATION = re.compile(
    r"\b(?:no|not|never|can't|cannot|won't|don't|didn't|isn't|aren't|wasn't|shouldn't)\b",
    re.IGNORECASE,
)
NUMBER = re.compile(
    r"\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|twenty|thirty|forty|fifty|hundred|thousand|million|percent|dollars?|pounds?)\b",
    re.IGNORECASE,
)
NAME = re.compile(r"\b(?:Mr\.?|Mrs\.?|Ms\.?|Dr\.?|my name is)\b", re.IGNORECASE)
CRITICAL = {"negation": NEGATION, "number": NUMBER, "name_candidate": NAME}
SAMPLE_RATE = 16000
PAD_SAMPLES = SAMPLE_RATE // 4


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def rank(kind: str, accent: str, value: str) -> bytes:
    return hashlib.sha256(f"apptek-20260927|{kind}|{accent}|{value}".encode()).digest()


def source_file(row: dict, accent: str) -> str:
    relative = Path(row["file_name"])
    if (
        len(relative.parts) != 2
        or relative.parts[0] != "audio"
        or relative.suffix != ".wav"
    ):
        raise ValueError("unsafe source file path")
    return f"diarization/{accent}/{relative.as_posix()}"


def sample_position(seconds: float) -> int:
    return math.floor(seconds * SAMPLE_RATE + 0.5)


def eligible_turns(row: dict) -> list[dict]:
    duration = row["duration"]
    segments = row["segments"]
    if (
        not isinstance(duration, (int, float))
        or not math.isfinite(duration)
        or duration <= 0
    ):
        raise ValueError("invalid conversation duration")
    for segment in segments:
        start, end = segment["start"], segment["end"]
        if (
            not all(
                isinstance(value, (int, float)) and math.isfinite(value)
                for value in (start, end)
            )
            or not 0 <= start < end <= duration
        ):
            raise ValueError("invalid segment boundary")
    eligible = []
    for index, segment in enumerate(segments):
        start, end = segment["start"], segment["end"]
        reference = segment["text"].strip()
        if not (3 <= end - start <= 12 and 3 <= len(reference.split()) <= 45):
            continue
        padded_start = max(0, sample_position(start) - PAD_SAMPLES)
        padded_end = min(sample_position(duration), sample_position(end) + PAD_SAMPLES)
        if any(
            other_index != index
            and sample_position(other["start"]) < padded_end
            and sample_position(other["end"]) > padded_start
            for other_index, other in enumerate(segments)
        ):
            continue
        eligible.append(
            {
                "segment_index": index,
                "start_sample": padded_start,
                "end_sample": padded_end,
                "speaker_id": segment["speaker_id"],
                "role": segment["role"],
                "gender": segment["gender"],
                "reference": reference,
            }
        )
    return eligible


def critical_assignment(
    turns: list[dict], accent: str
) -> tuple[dict, dict, dict] | None:
    choices = [
        [turn for turn in turns if pattern.search(turn["reference"])]
        for pattern in CRITICAL.values()
    ]
    if any(not choice for choice in choices):
        return None
    assignments = (
        assignment
        for assignment in itertools.product(*choices)
        if len({turn["segment_index"] for turn in assignment}) == 3
    )
    return min(
        assignments,
        key=lambda assignment: tuple(
            rank(label, accent, str(turn["segment_index"]))
            for label, turn in zip(CRITICAL, assignment, strict=True)
        ),
        default=None,
    )


def selected_turn(
    row: dict, accent: str, cohort: str, label: str | None, turn: dict
) -> dict:
    source = source_file(row, accent)
    origin = f"{accent}-{cohort}-{sha256_bytes(f'{source}|{turn['segment_index']}'.encode())[:16]}"
    return {
        "origin_id": origin,
        "accent": accent,
        "cohort": cohort,
        "critical_label": label,
        "source_file": source,
        **turn,
    }


def select_accent(rows: list[dict], accent: str, excluded: set[str]) -> list[dict]:
    candidates = []
    for row in rows:
        source = source_file(row, accent)
        if source in excluded:
            continue
        turns = eligible_turns(row)
        candidates.append((row, source, turns))
    general = [
        item
        for item in candidates
        if len({turn["speaker_id"] for turn in item[2]}) >= 2
    ]
    if not general:
        raise ValueError(f"no general call for {accent}")
    general_row, general_source, general_turns = min(
        general, key=lambda item: rank("general", accent, item[1])
    )
    speakers = sorted(
        {turn["speaker_id"] for turn in general_turns},
        key=lambda speaker: rank("speaker", accent, f"{general_source}|{speaker}"),
    )[:2]
    selected = [
        selected_turn(
            general_row,
            accent,
            "general",
            None,
            min(
                (turn for turn in general_turns if turn["speaker_id"] == speaker),
                key=lambda turn: rank(
                    "turn", accent, f"{general_source}|{turn['segment_index']}"
                ),
            ),
        )
        for speaker in speakers
    ]
    critical = [
        (row, source, assignment)
        for row, source, turns in candidates
        if source != general_source
        if (assignment := critical_assignment(turns, accent)) is not None
    ]
    if not critical:
        raise ValueError(f"no separate critical call for {accent}")
    critical_row, _, assignment = min(
        critical, key=lambda item: rank("critical", accent, item[1])
    )
    selected.extend(
        selected_turn(critical_row, accent, "critical", label, turn)
        for label, turn in zip(CRITICAL, assignment, strict=True)
    )
    return selected


def select_corpus(
    rows_by_accent: dict[str, list[dict]], excluded: set[str]
) -> list[dict]:
    if set(rows_by_accent) != set(ACCENTS):
        raise ValueError("accent inventory differs")
    selected = [
        item
        for accent in ACCENTS
        for item in select_accent(rows_by_accent[accent], accent, excluded)
    ]
    if len(selected) != 70 or len({item["origin_id"] for item in selected}) != 70:
        raise ValueError("selected corpus count or identity differs")
    if len({item["source_file"] for item in selected}) != 28:
        raise ValueError("selected call count differs")
    return selected


def read_metadata(path: Path, expected_sha256: str, accent: str) -> list[dict]:
    if sha256(path) != expected_sha256:
        raise ValueError(f"metadata bytes changed: {accent}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not rows or any(row["accent"] != accent for row in rows):
        raise ValueError(f"metadata accent changed: {accent}")
    if len({row["file_name"] for row in rows}) != len(rows):
        raise ValueError(f"duplicate source file: {accent}")
    return rows


def load_inventory(path: Path) -> dict:
    if sha256(path) != INVENTORY_SHA256:
        raise ValueError("pinned LFS inventory bytes changed")
    packet = json.loads(path.read_text(encoding="utf-8"))
    if (
        packet.get("schema") != 1
        or packet.get("dataset") != DATASET
        or packet.get("revision") != DATASET_REVISION
    ):
        raise ValueError("pinned LFS inventory identity changed")
    return packet


def verify_source(path: Path, expected_sha256: str, expected_size: int) -> None:
    if path.stat().st_size != expected_size or sha256(path) != expected_sha256:
        raise ValueError(f"source audio differs from pinned LFS object: {path.name}")


def extract_clip(path: Path, start_sample: int, end_sample: int) -> bytes:
    with wave.open(str(path), "rb") as source:
        if (
            source.getnchannels() != 1
            or source.getsampwidth() != 2
            or source.getframerate() != SAMPLE_RATE
            or source.getcomptype() != "NONE"
        ):
            raise ValueError("source WAV format changed")
        if not 0 <= start_sample < end_sample <= source.getnframes():
            raise ValueError("clip interval lies outside source WAV")
        source.setpos(start_sample)
        return source.readframes(end_sample - start_sample)


def selection(metadata_dir: Path) -> dict:
    rows = {
        accent: read_metadata(metadata_dir / f"{accent}-metadata.jsonl", digest, accent)
        for accent, digest in METADATA_SHA256.items()
    }
    return {
        "schema": 1,
        "purpose": "apptek_publisher_reference_diagnostic",
        "dataset": DATASET,
        "revision": DATASET_REVISION,
        "metadata_sha256": METADATA_SHA256,
        "builder_sha256": sha256(Path(__file__)),
        "samples": select_corpus(rows, {EXCLUDED_DEVELOPMENT_FILE}),
    }


def freeze(
    metadata_dir: Path,
    selection_path: Path,
    inventory_path: Path,
    source_dir: Path,
    output: Path,
) -> dict:
    if output.exists():
        raise FileExistsError(output)
    selected = json.loads(selection_path.read_text(encoding="utf-8"))
    if selected != selection(metadata_dir):
        raise ValueError("selection differs from pinned metadata or builder")
    inventory = load_inventory(inventory_path)
    sources = inventory["sources"]
    selected_paths = {item["source_file"] for item in selected["samples"]}
    if set(sources) != selected_paths:
        raise ValueError("LFS inventory differs from selected sources")
    for source in sorted(selected_paths):
        relative = Path(source)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe selected source path")
        item = sources[source]
        verify_source(source_dir / relative, item["sha256"], item["size"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="apptek-freeze-", dir=output.parent
    ) as temporary:
        temporary_dir = Path(temporary)
        clips = temporary_dir / "clips"
        clips.mkdir()
        samples = []
        for item in selected["samples"]:
            source = item["source_file"]
            audio_file = f"clips/{item['origin_id']}.wav"
            pcm = extract_clip(
                source_dir / source, item["start_sample"], item["end_sample"]
            )
            path = temporary_dir / audio_file
            with wave.open(str(path), "wb") as destination:
                destination.setparams((1, 2, SAMPLE_RATE, 0, "NONE", "not compressed"))
                destination.writeframes(pcm)
            samples.append(
                {
                    **item,
                    "source_sha256": sources[source]["sha256"],
                    "audio_file": audio_file,
                    "sha256": sha256(path),
                }
            )
        manifest = {
            **{
                key: selected[key]
                for key in (
                    "schema",
                    "purpose",
                    "dataset",
                    "revision",
                    "metadata_sha256",
                    "builder_sha256",
                )
            },
            "selection_sha256": sha256(selection_path),
            "inventory_sha256": sha256(inventory_path),
            "samples": samples,
        }
        (temporary_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temporary_dir.rename(output)
    return manifest


def main() -> None:
    os.umask(0o077)
    parser = argparse.ArgumentParser()
    subcommands = parser.add_subparsers(dest="operation", required=True)
    select = subcommands.add_parser("select")
    select.add_argument("--metadata-dir", type=Path, required=True)
    select.add_argument("--output", type=Path, required=True)
    frozen = subcommands.add_parser("freeze")
    for name in ("metadata-dir", "selection", "inventory", "source-dir", "output"):
        frozen.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.operation == "select":
        with args.output.open("x", encoding="utf-8") as destination:
            json.dump(
                selection(args.metadata_dir), destination, ensure_ascii=False, indent=2
            )
            destination.write("\n")
        print(f"selection_sha256={sha256(args.output)}")
    else:
        manifest = freeze(
            args.metadata_dir,
            args.selection,
            args.inventory,
            args.source_dir,
            args.output,
        )
        print(
            f"manifest_sha256={sha256(args.output / 'manifest.json')} samples={len(manifest['samples'])}"
        )


if __name__ == "__main__":
    main()

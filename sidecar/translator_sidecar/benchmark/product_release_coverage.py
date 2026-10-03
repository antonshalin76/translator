"""Classify release-evaluation metadata coverage without scoring product quality."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

MODES = ("quality_first", "balanced", "streaming_first")
DIRECTIONS = ("microphone", "speaker")
ASSIGNMENTS = ("mic_ru_speaker_en", "mic_en_speaker_ru")
GENDERS = ("male", "female")
BUCKETS = (
    "short",
    "long",
    "negation_scope",
    "numbers_identifiers",
    "names_roles",
    "discourse_boundary",
)
CONDITIONS = (
    "noise_10db",
    "room_response",
    "gain_plus_6db",
    "gain_minus_6db",
    "telephony_8khz",
    "silence",
)
DROP_STATUSES = frozenset(
    {"timeout", "cancelled", "missing_terminal", "provider_error", "missing_audible"}
)
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_CELL_FIELDS = (
    "provider",
    "fallback",
    "mode",
    "audio_direction",
    "language_assignment",
    "target_gender",
)


def _text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _hash(value: object) -> bool:
    return isinstance(value, str) and _HASH.fullmatch(value) is not None


def _normalized(value: str) -> str:
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", value).casefold()))


def _list(value: object) -> list[Any]:
    return value if isinstance(value, list) else []


def _development_inventory(value: object) -> tuple[set[str], set[str], set[str]] | None:
    if not isinstance(value, Mapping) or value.get("schema") != 1:
        return None
    keys = ("seen_case_ids", "seen_speaker_ids", "seen_normalized_sources")
    entries = [_list(value.get(key)) for key in keys]
    if any(not rows or any(not _text(item) for item in rows) for rows in entries):
        return None
    return set(entries[0]), set(entries[1]), {_normalized(item) for item in entries[2]}


def _case_metadata(case: object) -> bool:
    if not isinstance(case, Mapping):
        return False
    return (
        all(
            _text(case.get(key))
            for key in (
                "id",
                "source",
                "reference",
                "speaker_id",
                "accent",
                "template_family",
                "license_id",
            )
        )
        and _normalized(case["accent"])
        not in {"", "unknown", "null", "none", "unspecified", "n a"}
        and isinstance(case.get("source_language"), str)
        and case.get("source_language") in {"ru", "en"}
        and case.get("speaker_gender") in GENDERS
        and case.get("primary_bucket") in BUCKETS
        and _hash(case.get("origin_audio_sha256"))
        and case.get("access_terms_verified") is True
    )


def _holdout(
    holdout: object, development: object
) -> tuple[dict[str, Any], dict[str, set[str]]]:
    reasons: set[str] = set()
    ids: dict[str, set[str]] = {"ru": set(), "en": set()}
    inventory = _development_inventory(development)
    if inventory is None:
        reasons.add("development_inventory_missing")
    if (
        not isinstance(holdout, Mapping)
        or holdout.get("schema") != 1
        or holdout.get("purpose") != "independent_release_holdout"
    ):
        return {"status": "NOT_DONE", "reasons": ["holdout_metadata"]}, ids
    cases = _list(holdout.get("cases"))
    fixtures = _list(holdout.get("fixtures"))
    if not cases:
        reasons.add("case_count")
    by_id: dict[str, Mapping[str, Any]] = {}
    seen_sources: set[str] = set()
    seen_audio: set[str] = set()
    speaker_identity: dict[str, tuple[str, str]] = {}
    buckets: dict[str, Counter[str]] = defaultdict(Counter)
    families: dict[str, Counter[str]] = defaultdict(Counter)
    speakers: dict[str, set[str]] = defaultdict(set)
    bucket_speakers: dict[tuple[str, str], set[str]] = defaultdict(set)
    genders: dict[str, set[str]] = defaultdict(set)
    accents: dict[str, set[str]] = defaultdict(set)
    controls: dict[tuple[str, str], set[str]] = defaultdict(set)
    for case in cases:
        if not _case_metadata(case):
            reasons.add("case_metadata")
            continue
        language = case["source_language"]
        case_id = case["id"]
        source = _normalized(case["source"])
        if not source or case_id in by_id:
            reasons.add("duplicate_case")
            continue
        if source in seen_sources:
            reasons.add("duplicate_source")
        if case["origin_audio_sha256"] in seen_audio:
            reasons.add("duplicate_audio")
        seen_sources.add(source)
        seen_audio.add(case["origin_audio_sha256"])
        by_id[case_id] = case
        ids[language].add(case_id)
        buckets[language][case["primary_bucket"]] += 1
        families[language][case["template_family"]] += 1
        speaker = case["speaker_id"]
        speakers[language].add(speaker)
        bucket_speakers[language, case["primary_bucket"]].add(speaker)
        genders[language].add(case["speaker_gender"])
        accent = _normalized(case["accent"])
        accents[language].add(accent)
        polarity = case.get("control_polarity")
        if case["primary_bucket"] in BUCKETS[2:5]:
            if polarity not in ("positive", "negative"):
                reasons.add("critical_controls")
            else:
                controls[language, case["primary_bucket"]].add(polarity)
        elif polarity is not None:
            reasons.add("critical_controls")
        identity = (case["speaker_gender"], accent)
        if speaker in speaker_identity and speaker_identity[speaker] != identity:
            reasons.add("speaker_metadata")
        speaker_identity[speaker] = identity
        if inventory is not None and (
            case_id in inventory[0] or speaker in inventory[1] or source in inventory[2]
        ):
            reasons.add("development_overlap")
    for language in ("ru", "en"):
        if len(ids[language]) != 120:
            reasons.add("case_count")
        if any(buckets[language][bucket] < 20 for bucket in BUCKETS):
            reasons.add("bucket_coverage")
        if any(count > 12 for count in families[language].values()):
            reasons.add("template_family")
        if any(
            controls[language, bucket] != {"positive", "negative"}
            for bucket in BUCKETS[2:5]
        ):
            reasons.add("critical_controls")
        if (
            len(speakers[language]) < 4
            or genders[language] != set(GENDERS)
            or not (accents[language] - {"default"})
            or any(
                bucket_speakers[language, bucket] != speakers[language]
                for bucket in BUCKETS
            )
        ):
            reasons.add("speaker_coverage")

    covered: dict[tuple[str, str], set[str]] = defaultdict(set)
    offsets: dict[str, set[int]] = defaultdict(set)
    fixture_keys: set[tuple[str, str, int | None]] = set()
    for fixture in fixtures:
        if not isinstance(fixture, Mapping):
            reasons.add("fixture_metadata")
            continue
        case_id = fixture.get("case_id")
        condition = fixture.get("condition")
        offset = fixture.get("offset_ms")
        source = by_id.get(case_id) if isinstance(case_id, str) else None
        valid_condition = isinstance(condition, str) and condition in {
            "clean",
            "packet_offset",
            *CONDITIONS,
        }
        if (
            source is None
            or not valid_condition
            or not _hash(fixture.get("audio_sha256"))
            or fixture.get("origin_audio_sha256") != source["origin_audio_sha256"]
            or (
                condition == "packet_offset"
                and (type(offset) is not int or not 0 <= offset < 20)
            )
            or (condition != "packet_offset" and offset is not None)
        ):
            reasons.add("fixture_metadata")
            continue
        key = (case_id, condition, offset)
        if key in fixture_keys:
            reasons.add("fixture_metadata")
        fixture_keys.add(key)
        if (
            condition == "clean"
            and fixture["audio_sha256"] != source["origin_audio_sha256"]
        ):
            reasons.add("fixture_metadata")
        if (
            condition in CONDITIONS
            and fixture["audio_sha256"] == source["origin_audio_sha256"]
        ):
            reasons.add("fixture_metadata")
        covered[source["source_language"], condition].add(case_id)
        if condition == "packet_offset":
            offsets[source["source_language"]].add(offset)
    for language in ("ru", "en"):
        if covered[language, "clean"] != ids[language]:
            reasons.add("fixture_coverage")
        if any(len(covered[language, condition]) < 20 for condition in CONDITIONS):
            reasons.add("fixture_coverage")
        if len(covered[language, "packet_offset"]) < 20 or offsets[language] != set(
            range(20)
        ):
            reasons.add("fixture_coverage")
    return {
        "status": "ELIGIBLE" if not reasons else "NOT_DONE",
        "reasons": sorted(reasons),
        "semantic_cases": {language: len(ids[language]) for language in ("ru", "en")},
    }, ids


def _source_language(direction: str, assignment: str) -> str:
    return (
        "ru"
        if (direction, assignment)
        in {("microphone", "mic_ru_speaker_en"), ("speaker", "mic_en_speaker_ru")}
        else "en"
    )


def _cell_key(values: tuple[str, ...]) -> str:
    return "|".join(values)


def _cell_report(rows: list[Mapping[str, Any]], case_ids: set[str]) -> dict[str, Any]:
    reasons: set[str] = set()
    first_measured = next(
        (index for index, row in enumerate(rows) if row.get("phase") != "warmup"),
        len(rows),
    )
    warmups = rows[:first_measured]
    measured = rows[first_measured:]
    warmup_ids = [row.get("case_id") for row in warmups]
    measured_ids = [row.get("case_id") for row in measured]
    if len(warmups) < 10 or not all(_text(case_id) for case_id in warmup_ids):
        reasons.add("warmups")
    if (
        len(measured) != 120
        or any(row.get("phase") != "measured" for row in measured)
        or not all(_text(case_id) for case_id in measured_ids)
        or len(set(measured_ids)) != 120
        or set(measured_ids) != case_ids
    ):
        reasons.add("measured_cases")
    drops = 0
    for row in rows:
        status = row.get("status")
        elapsed = row.get("elapsed_ms")
        vad_ms = row.get("overlap_vad_ms")
        if (
            not isinstance(elapsed, (int, float))
            or isinstance(elapsed, bool)
            or not 0 <= elapsed <= 10_000
            or type(vad_ms) is not int
            or vad_ms < 0
            or (
                status == "completed"
                and (
                    row.get("terminal_observed") is not True
                    or row.get("audible_observed") is not True
                )
            )
            or not isinstance(status, str)
            or status not in {"completed", *DROP_STATUSES}
        ):
            reasons.add("failed_attempt")
        if isinstance(status, str) and status in DROP_STATUSES:
            drops += 1
    overlap = sum(
        row.get("overlap_vad_ms", 0) >= 500
        for row in measured
        if type(row.get("overlap_vad_ms")) is int
    )
    if overlap < 30:
        reasons.add("overlap")
    return {
        "status": "COMPLETE" if not reasons else "NOT_DONE",
        "reasons": sorted(reasons),
        "warmups": len(warmups),
        "measured": len(measured),
        "overlap_attempts": overlap,
        "drop_attempts": drops,
        "quality_status": "UNMEASURED",
    }


def classify_release_coverage(
    holdout: object,
    development: object,
    attempts: object,
    *,
    required_paths: Sequence[tuple[str, str]],
) -> dict[str, Any]:
    """Classify declared metadata; a complete result is never a product PASS."""
    holdout_report, case_ids = _holdout(holdout, development)
    paths = tuple(required_paths) if isinstance(required_paths, Sequence) else ()
    path_ok = (
        bool(paths)
        and all(
            isinstance(path, tuple)
            and len(path) == 2
            and all(_text(item) for item in path)
            for path in paths
        )
        and len(set(paths)) == len(paths)
    )
    expected = [
        (provider, fallback, mode, direction, assignment, gender)
        for provider, fallback in paths
        if path_ok
        for mode in MODES
        for direction in DIRECTIONS
        for assignment in ASSIGNMENTS
        for gender in GENDERS
    ]
    rows_by_cell: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    unexpected = 0
    expected_set = set(expected)
    if not isinstance(attempts, list):
        attempts = []
        unexpected += 1
    for attempt in attempts:
        if not isinstance(attempt, Mapping):
            unexpected += 1
            continue
        cell = tuple(attempt.get(field) for field in _CELL_FIELDS)
        if all(isinstance(value, str) for value in cell) and cell in expected_set:
            rows_by_cell[cell].append(attempt)
        else:
            unexpected += 1
    cells = {
        _cell_key(cell): _cell_report(
            rows_by_cell[cell],
            case_ids[_source_language(cell[3], cell[4])],
        )
        for cell in expected
    }
    complete = (
        path_ok
        and unexpected == 0
        and holdout_report["status"] == "ELIGIBLE"
        and all(cell["status"] == "COMPLETE" for cell in cells.values())
    )
    return {
        "status": "COVERAGE_COMPLETE" if complete else "NOT_DONE",
        "scope": "metadata_only",
        "holdout": holdout_report,
        "cells": cells,
        "unexpected_attempts": unexpected,
        "required_paths_status": "DECLARED" if path_ok else "NOT_DONE",
    }

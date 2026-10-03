from __future__ import annotations

from copy import deepcopy

import pytest

from translator_sidecar.benchmark.product_release_coverage import (
    classify_release_coverage,
)

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


def evidence():
    cases = []
    fixtures = []
    for language in ("ru", "en"):
        for bucket in BUCKETS:
            for number in range(20):
                speaker = number % 4
                case_id = f"{language}-{bucket}-{number}"
                origin_hash = f"{len(cases) + 1:064x}"
                cases.append(
                    {
                        "id": case_id,
                        "source_language": language,
                        "source": f"{language} source {bucket} {number}",
                        "reference": f"translated {case_id}",
                        "speaker_id": f"{language}-speaker-{speaker}",
                        "speaker_gender": "male" if speaker % 2 else "female",
                        "accent": "regional" if speaker == 3 else "default",
                        "primary_bucket": bucket,
                        "control_polarity": ("positive" if number % 2 else "negative")
                        if bucket in BUCKETS[2:5]
                        else None,
                        "template_family": f"{bucket}-{number // 8}",
                        "origin_audio_sha256": origin_hash,
                        "license_id": "test-license",
                        "access_terms_verified": True,
                    }
                )
                fixtures.append(
                    {
                        "case_id": case_id,
                        "condition": "clean",
                        "audio_sha256": origin_hash,
                        "origin_audio_sha256": origin_hash,
                    }
                )
                if bucket == BUCKETS[0]:
                    for condition in CONDITIONS:
                        fixtures.append(
                            {
                                "case_id": case_id,
                                "condition": condition,
                                "audio_sha256": f"{len(fixtures) + 1000:064x}",
                                "origin_audio_sha256": origin_hash,
                            }
                        )
                    fixtures.append(
                        {
                            "case_id": case_id,
                            "condition": "packet_offset",
                            "offset_ms": number,
                            "audio_sha256": f"{len(fixtures) + 1000:064x}",
                            "origin_audio_sha256": origin_hash,
                        }
                    )
    holdout = {
        "schema": 1,
        "purpose": "independent_release_holdout",
        "cases": cases,
        "fixtures": fixtures,
    }
    development = {
        "schema": 1,
        "seen_case_ids": ["development-case"],
        "seen_speaker_ids": ["development-speaker"],
        "seen_normalized_sources": ["development source"],
    }
    attempts = []
    for mode in ("quality_first", "balanced", "streaming_first"):
        for direction in ("microphone", "speaker"):
            for assignment in ("mic_ru_speaker_en", "mic_en_speaker_ru"):
                language = (
                    "ru"
                    if (direction, assignment)
                    in (
                        ("microphone", "mic_ru_speaker_en"),
                        ("speaker", "mic_en_speaker_ru"),
                    )
                    else "en"
                )
                ids = [
                    case["id"] for case in cases if case["source_language"] == language
                ]
                for gender in ("male", "female"):
                    cell = {
                        "provider": "local",
                        "fallback": "none",
                        "mode": mode,
                        "audio_direction": direction,
                        "language_assignment": assignment,
                        "target_gender": gender,
                    }
                    attempts.extend(
                        {
                            **cell,
                            "phase": "warmup",
                            "case_id": f"warmup-{i}",
                            "status": "completed",
                            "terminal_observed": True,
                            "audible_observed": True,
                            "elapsed_ms": 200,
                            "overlap_vad_ms": 0,
                        }
                        for i in range(10)
                    )
                    attempts.extend(
                        {
                            **cell,
                            "phase": "measured",
                            "case_id": case_id,
                            "status": "completed",
                            "terminal_observed": True,
                            "audible_observed": True,
                            "elapsed_ms": 200,
                            "overlap_vad_ms": 500 if i < 30 else 0,
                        }
                        for i, case_id in enumerate(ids)
                    )
    return holdout, development, attempts


def classify(holdout, development, attempts):
    return classify_release_coverage(
        holdout, development, attempts, required_paths=(("local", "none"),)
    )


def test_complete_metadata_is_coverage_only() -> None:
    holdout, development, attempts = evidence()
    report = classify(holdout, development, attempts)
    assert report["status"] == "COVERAGE_COMPLETE"
    assert report["scope"] == "metadata_only"
    assert report["holdout"]["status"] == "ELIGIBLE"
    assert len(report["cells"]) == 24
    assert all(cell["status"] == "COMPLETE" for cell in report["cells"].values())
    assert "PASS" not in str(report)


@pytest.mark.parametrize(
    "accent",
    [
        None,
        "",
        " \t\n\u00a0\u2003",
        "unknown",
        " UnKnOwN ",
        "\u00a0UNKNOWN\u2003",
        "\uff35\uff4e\uff4b\uff4e\uff4f\uff57\uff4e",
        "null",
        " \uff2e\uff35\uff2c\uff2c ",
        "none",
        "\tNoNe\n",
        "unspecified",
        "\u00a0UNSPECIFIED\u2003",
        "n/a",
        " N / A ",
        "\uff2e\uff0f\uff21",
    ],
)
@pytest.mark.parametrize("scope", ["all", "one"])
def test_unknown_accent_cannot_supply_release_metadata(accent, scope) -> None:
    holdout, development, attempts = evidence()
    assert classify(holdout, development, attempts)["status"] == "COVERAGE_COMPLETE"
    for case in holdout["cases"] if scope == "all" else holdout["cases"][:1]:
        case["accent"] = accent
    report = classify(holdout, development, attempts)
    assert report["status"] == "NOT_DONE"
    assert report["holdout"]["status"] == "NOT_DONE"
    assert "case_metadata" in report["holdout"]["reasons"]
    if scope == "all":
        assert report["holdout"]["semantic_cases"] == {"ru": 0, "en": 0}


def test_regional_speakers_cannot_cover_a_missing_case_accent() -> None:
    holdout, development, attempts = evidence()
    del holdout["cases"][0]["accent"]
    report = classify(holdout, development, attempts)
    assert report["status"] == "NOT_DONE"
    assert "case_metadata" in report["holdout"]["reasons"]


@pytest.mark.parametrize(
    "accent",
    [
        "default",
        "DEFAULT",
        " DEFAULT ",
        "\u00a0DeFaUlT\u2003",
        "\uff24\uff25\uff26\uff21\uff35\uff2c\uff34",
    ],
)
def test_normalized_default_is_not_regional_accent_coverage(accent) -> None:
    holdout, development, attempts = evidence()
    assert classify(holdout, development, attempts)["status"] == "COVERAGE_COMPLETE"
    for case in holdout["cases"]:
        case["accent"] = accent
    report = classify(holdout, development, attempts)
    assert report["status"] == "NOT_DONE"
    assert "speaker_coverage" in report["holdout"]["reasons"]
    assert "case_metadata" not in report["holdout"]["reasons"]


def test_equivalent_accent_labels_preserve_regional_speaker_coverage() -> None:
    holdout, development, attempts = evidence()
    for index, case in enumerate(holdout["cases"]):
        if case["accent"] == "default":
            case["accent"] = ["default", " DEFAULT ", "\u00a0DeFaUlT\u2003"][index % 3]
        else:
            case["accent"] = ["northwestern", " NORTHWESTERN "][(index // 4) % 2]
    report = classify(holdout, development, attempts)
    assert report["status"] == "COVERAGE_COMPLETE"
    assert report["holdout"]["status"] == "ELIGIBLE"


def test_distinct_accent_labels_remain_inconsistent_for_one_speaker() -> None:
    holdout, development, attempts = evidence()
    holdout["cases"][0]["accent"] = "northwestern"
    report = classify(holdout, development, attempts)
    assert report["status"] == "NOT_DONE"
    assert "speaker_metadata" in report["holdout"]["reasons"]


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (lambda h, d: h["cases"].pop(), "case_count"),
        (
            lambda h, d: h["cases"][1].update(source=h["cases"][0]["source"].upper()),
            "duplicate_source",
        ),
        (
            lambda h, d: [
                case.update(template_family="shared") for case in h["cases"][:13]
            ],
            "template_family",
        ),
        (lambda h, d: h["cases"][0].update(reference=""), "case_metadata"),
        (
            lambda h, d: h["cases"][0].update(access_terms_verified=False),
            "case_metadata",
        ),
        (
            lambda h, d: [
                case.update(control_polarity="positive")
                for case in h["cases"]
                if case["primary_bucket"] == "negation_scope"
            ],
            "critical_controls",
        ),
        (
            lambda h, d: d["seen_speaker_ids"].append(h["cases"][0]["speaker_id"]),
            "development_overlap",
        ),
        (
            lambda h, d: [
                case.update(speaker_id="ru-speaker-0")
                for case in h["cases"]
                if case["source_language"] == "ru"
            ],
            "speaker_coverage",
        ),
        (lambda h, d: h["fixtures"].pop(), "fixture_coverage"),
        (lambda h, d: h["fixtures"][0].update(audio_sha256="bad"), "fixture_metadata"),
    ],
)
def test_ineligible_holdout_is_not_done(mutate, reason) -> None:
    holdout, development, attempts = evidence()
    mutate(holdout, development)
    report = classify(holdout, development, attempts)
    assert report["status"] == "NOT_DONE"
    assert reason in report["holdout"]["reasons"]


def test_acoustic_transform_cannot_reuse_origin_bytes() -> None:
    holdout, development, attempts = evidence()
    noise = next(
        item for item in holdout["fixtures"] if item["condition"] == "noise_10db"
    )
    noise["audio_sha256"] = noise["origin_audio_sha256"]
    report = classify(holdout, development, attempts)
    assert report["status"] == "NOT_DONE"
    assert "fixture_metadata" in report["holdout"]["reasons"]


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (lambda rows: rows.pop(), "measured_cases"),
        (lambda rows: rows[10].update(case_id=rows[11]["case_id"]), "measured_cases"),
        (lambda rows: rows[10].update(status="not_run"), "failed_attempt"),
        (lambda rows: rows[10].update(audible_observed=False), "failed_attempt"),
        (lambda rows: rows[10].update(elapsed_ms=10001), "failed_attempt"),
        (lambda rows: rows[0].update(phase="measured"), "warmups"),
        (lambda rows: [row.update(overlap_vad_ms=0) for row in rows[10:11]], "overlap"),
    ],
)
def test_incomplete_cell_keeps_not_done(mutate, reason) -> None:
    holdout, development, attempts = evidence()
    mutate(attempts)
    report = classify(holdout, development, attempts)
    assert report["status"] == "NOT_DONE"
    assert any(reason in cell["reasons"] for cell in report["cells"].values())


def test_missing_declared_path_and_unknown_attempt_are_not_done() -> None:
    holdout, development, attempts = evidence()
    report = classify_release_coverage(
        holdout,
        development,
        attempts,
        required_paths=(("local", "none"), ("local", "cpu_small")),
    )
    assert report["status"] == "NOT_DONE"
    assert len(report["cells"]) == 48
    assert sum(cell["status"] == "NOT_DONE" for cell in report["cells"].values()) == 24

    unknown = deepcopy(attempts[0])
    unknown["provider"] = "unlisted"
    report = classify(holdout, development, [*attempts, unknown])
    assert report["status"] == "NOT_DONE"
    assert report["unexpected_attempts"] == 1


def test_recorded_drop_keeps_coverage_complete_and_visible() -> None:
    holdout, development, attempts = evidence()
    attempts[10].update(
        status="timeout",
        terminal_observed=False,
        audible_observed=False,
        elapsed_ms=10000,
    )
    report = classify(holdout, development, attempts)
    assert report["status"] == "COVERAGE_COMPLETE"
    assert next(iter(report["cells"].values()))["drop_attempts"] == 1
    assert next(iter(report["cells"].values()))["quality_status"] == "UNMEASURED"


def test_more_than_ten_excluded_warmups_are_eligible_but_late_warmup_is_not() -> None:
    holdout, development, attempts = evidence()
    extra = deepcopy(attempts[0])
    extra["case_id"] = "warmup-extra"
    attempts.insert(10, extra)
    assert classify(holdout, development, attempts)["status"] == "COVERAGE_COMPLETE"
    attempts[11]["phase"] = "warmup"
    assert classify(holdout, development, attempts)["status"] == "NOT_DONE"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda h, rows: h["cases"][0].update(source_language=[]),
        lambda h, rows: h["cases"][0].update(primary_bucket=[]),
        lambda h, rows: h["fixtures"][0].update(condition=[]),
        lambda h, rows: rows[0].update(provider=[]),
        lambda h, rows: rows[0].update(case_id=[]),
    ],
)
def test_malformed_metadata_fails_closed_without_exception(mutate) -> None:
    holdout, development, attempts = evidence()
    mutate(holdout, attempts)
    assert classify(holdout, development, attempts)["status"] == "NOT_DONE"

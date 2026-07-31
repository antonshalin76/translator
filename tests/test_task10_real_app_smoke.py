from __future__ import annotations

import json
import re
import stat
import tempfile
import unittest
import importlib.machinery
import importlib.util
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TASK7_REPORT = ROOT / "docs/benchmarks/task7-live-human-round-trip.json"


def read(path: str) -> str:
    return (ROOT / path).read_text()


def read_json(path: str) -> dict[str, Any]:
    return json.loads(read(path))


def requires_local_artifacts(*paths: str):
    return unittest.skipUnless(
        all((ROOT / path).exists() for path in paths),
        "local planning/run evidence is not published",
    )


def walk_keys(value: Any) -> list[str]:
    if isinstance(value, dict):
        keys = list(value)
        for child in value.values():
            keys.extend(walk_keys(child))
        return keys
    if isinstance(value, list):
        keys: list[str] = []
        for child in value:
            keys.extend(walk_keys(child))
        return keys
    return []


def load_smoke_module() -> Any:
    script = ROOT / "scripts/translator-task10-real-app-smoke"
    loader = importlib.machinery.SourceFileLoader("task10_smoke", str(script))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


class Task10RealAppSmokeTests(unittest.TestCase):
    def test_smoke_script_is_explicit_route_only_and_local_provider_only(self) -> None:
        script_path = ROOT / "scripts/translator-task10-real-app-smoke"
        self.assertTrue(script_path.exists(), "Task 10 smoke script is missing")
        self.assertTrue(
            script_path.stat().st_mode & stat.S_IXUSR,
            "Task 10 smoke script must be executable",
        )
        script = script_path.read_text()

        for token in (
            "--manual-route-stream-id",
            "--start-service",
            "--stop-after",
            "--confirm-meet-stream-id",
            "--virtual-mic-selected-confirmed",
            "--remote-outgoing-translation-confirmed",
            "--repo-c4-scan",
            "/v1/routes/manual-override",
            "Translator_Virtual_Mic",
            "telegram-desktop",
            "firefox",
            "google-chrome",
        ):
            self.assertIn(token, script)

        self.assertNotIn("pactl move-sink-input", script)
        self.assertNotRegex(script, re.compile(r"\bopenai\b", re.IGNORECASE))
        self.assertNotIn("OPENAI_API_KEY", script)
        self.assertNotRegex(script, re.compile(r"debug[_-]text.*true", re.IGNORECASE))
        self.assertNotRegex(script, re.compile(r"debug[_-]capture.*true", re.IGNORECASE))

    @requires_local_artifacts("docs/benchmarks/task10-validation-report.json")
    def test_task10_reports_exist_and_record_live_acceptance_completion(self) -> None:
        validation = read_json("docs/benchmarks/task10-validation-report.json")

        self.assertEqual(validation["schema_version"], "translator.task10-validation.v1")
        self.assertTrue(validation["completed"])
        self.assertTrue(validation["mvp_a_gate_satisfied"])
        self.assertEqual(
            validation["live_second_endpoint"]["status"],
            "passed_by_live_user_confirmation",
        )
        self.assertEqual(
            validation["live_second_endpoint"]["required_evidence"],
            "Telegram or Meet call with real second endpoint and simultaneous incoming/outgoing speech",
        )

        smoke_reports = validation["smoke_reports"]
        self.assertEqual(
            smoke_reports["telegram_desktop"],
            "docs/benchmarks/task10-telegram-smoke-report.json",
        )
        self.assertEqual(
            smoke_reports["google_meet_browser"],
            "docs/benchmarks/task10-meet-smoke-report.json",
        )
        self.assertEqual(
            validation["latency_ledger"],
            "docs/benchmarks/task10-latency-ledger.json",
        )
        self.assertEqual(
            validation["privacy_marker_scan"],
            "docs/benchmarks/task10-privacy-marker-scan.json",
        )
        self.assertEqual(validation["repo_c4_scan"]["result"], "passed")

    @requires_local_artifacts(
        "docs/benchmarks/task10-validation-report.json",
        "docs/benchmarks/task10-telegram-smoke-report.json",
        "docs/benchmarks/task10-meet-smoke-report.json",
    )
    def test_virtual_microphone_presence_is_not_live_call_app_proof(self) -> None:
        validation = read_json("docs/benchmarks/task10-validation-report.json")
        telegram = read_json("docs/benchmarks/task10-telegram-smoke-report.json")
        meet = read_json("docs/benchmarks/task10-meet-smoke-report.json")

        item = next(
            entry
            for entry in validation["acceptance"]
            if entry["item"].startswith("Translator_Virtual_Mic is selected")
        )
        self.assertEqual(item["status"], "passed_by_live_user_confirmation")
        self.assertTrue(telegram["virtual_microphone"]["source_present"])
        self.assertTrue(meet["virtual_microphone"]["source_present"])
        self.assertFalse(telegram["virtual_microphone"]["call_app_selection_confirmed"])
        self.assertFalse(telegram["virtual_microphone"]["remote_outgoing_translation_confirmed"])

    @requires_local_artifacts(
        "docs/benchmarks/task7-live-human-round-trip.json",
        "docs/benchmarks/task10-validation-report.json",
        "docs/benchmarks/task10-latency-ledger.json",
    )
    def test_task10_carries_task7_latency_debt_verbatim(self) -> None:
        task7 = json.loads(TASK7_REPORT.read_text())
        validation = read_json("docs/benchmarks/task10-validation-report.json")
        ledger = read_json("docs/benchmarks/task10-latency-ledger.json")

        for payload in (validation["task7_debt_carried"], ledger["task7_debt_carried"]):
            self.assertEqual(
                payload["canonical_evidence"],
                "docs/benchmarks/task7-live-human-round-trip.json",
            )
            self.assertEqual(
                payload["local_provider_latency_classification"],
                task7["classification"],
            )
            self.assertEqual(
                payload["task7_complete"],
                task7["acceptance"]["task7_complete"],
            )
            self.assertFalse(payload["task7_complete"])
            self.assertEqual(
                payload["requires_mvp_b_provider_comparison"],
                task7["acceptance"]["requires_mvp_b_provider_comparison"],
            )
            self.assertEqual(
                payload["blocked_acceptance_item"],
                task7["acceptance"]["blocked_acceptance_item"],
            )

        self.assertEqual(ledger["local_provider_latency_classification"], "fails_usable_limit")
        self.assertEqual(
            ledger["physical_mic_onset_to_returned_ru_first_audible_ms"],
            task7["latency_ms"]["physical_mic_onset_to_returned_ru_first_audible"],
        )

    @requires_local_artifacts(
        "docs/benchmarks/task10-telegram-smoke-report.json",
        "docs/benchmarks/task10-meet-smoke-report.json",
        "docs/benchmarks/task10-latency-ledger.json",
        "docs/benchmarks/task10-privacy-marker-scan.json",
        "docs/benchmarks/task10-validation-report.json",
    )
    def test_real_app_smoke_reports_record_blockers_without_spoken_content(self) -> None:
        forbidden_keys = {
            "pcm",
            "pcm_bytes",
            "raw_pcm",
            "transcript",
            "translation_text",
            "source_text",
            "target_text",
            "spoken_content",
            "spoken_phrase",
        }
        for path in (
            "docs/benchmarks/task10-telegram-smoke-report.json",
            "docs/benchmarks/task10-meet-smoke-report.json",
            "docs/benchmarks/task10-latency-ledger.json",
            "docs/benchmarks/task10-privacy-marker-scan.json",
            "docs/benchmarks/task10-validation-report.json",
        ):
            payload = read_json(path)
            self.assertTrue(payload["schema_version"].startswith("translator.task10-"))
            self.assertFalse(forbidden_keys.intersection(walk_keys(payload)), path)
            rendered = json.dumps(payload, ensure_ascii=False)
            self.assertNotIn("OPENAI_API_KEY", rendered)
            self.assertNotRegex(rendered, re.compile(r"\bopenai session\b", re.IGNORECASE))

        scan = read_json("docs/benchmarks/task10-privacy-marker-scan.json")
        self.assertTrue(scan["report_payload_key_scan"]["passed"])
        self.assertEqual(scan["log_scan"]["status"], "passed")
        self.assertTrue(scan["passed"])

        validation = read_json("docs/benchmarks/task10-validation-report.json")
        log_item = next(
            entry
            for entry in validation["acceptance"]
            if entry["item"] == "Synthetic and real-app logs contain no spoken content"
        )
        self.assertEqual(log_item["status"], "passed")

    def test_browser_stream_is_not_meet_candidate_without_confirmation(self) -> None:
        smoke = load_smoke_module()
        browser_call = {
            "application.name": "Firefox",
            "application.process.binary": "firefox",
            "media.role": "communication",
            "media.name": "WebRTC Voice",
        }
        meet_call = {
            "application.name": "Google Chrome",
            "application.process.binary": "google-chrome",
            "media.role": "communication",
            "media.name": "Meet Audio",
        }

        self.assertIsNone(smoke.classify_task10_stream(41, browser_call, None))
        self.assertEqual(
            smoke.classify_task10_stream(41, browser_call, 41),
            "google_meet_browser",
        )
        self.assertEqual(
            smoke.classify_task10_stream(42, meet_call, None),
            "google_meet_browser",
        )

    @requires_local_artifacts("docs/benchmarks/task7-live-human-round-trip.json")
    def test_separate_telegram_and_meet_runs_can_satisfy_future_live_gate(self) -> None:
        smoke = load_smoke_module()
        task7 = json.loads(TASK7_REPORT.read_text())
        audio_state = {
            "task10_route_candidates": [
                {
                    "stream_id": 42,
                    "app_key": "google_meet_browser",
                    "call_like": True,
                    "meet_confirmed": True,
                    "current_sink_name": "alsa_output.first",
                    "process_binary": "google-chrome",
                    "media_role": "communication",
                    "application_name": "Google Meet via Firefox/Chromium/Chrome",
                    "on_translator_remote_in": False,
                }
            ],
            "virtual_microphone": {
                "expected_description": "Translator_Virtual_Mic",
                "source_present": True,
            },
        }
        environment: dict[str, Any] = {}
        telegram_route = {
            "attempted": True,
            "status": "passed",
            "stream_id": 41,
            "candidate_app_key": "telegram_desktop",
        }
        meet_route = {
            "attempted": True,
            "status": "passed",
            "stream_id": 42,
            "candidate_app_key": "google_meet_browser",
        }
        recording = {"provided": True, "status": "present", "bytes": 1, "sha256": "0" * 64}
        telegram = smoke.app_report(
            "telegram_desktop",
            environment,
            {
                **audio_state,
                "task10_route_candidates": [
                    {
                        **audio_state["task10_route_candidates"][0],
                        "stream_id": 41,
                        "app_key": "telegram_desktop",
                        "application_name": "Telegram Desktop",
                        "process_binary": "telegram-desktop",
                    }
                ],
            },
            telegram_route,
            "telegram_desktop",
            recording,
            True,
            True,
            True,
        )
        meet = smoke.app_report(
            "google_meet_browser",
            environment,
            audio_state,
            meet_route,
            "google_meet_browser",
            recording,
            True,
            True,
            True,
        )

        with tempfile.TemporaryDirectory() as temp:
            output_dir = Path(temp)
            smoke.write_json(smoke.report_path(output_dir, "telegram_desktop"), telegram)
            merged = smoke.merge_existing_app_reports(
                output_dir,
                {
                    "telegram_desktop": smoke.app_report(
                        "telegram_desktop",
                        environment,
                        audio_state,
                        {"attempted": False, "status": "not_attempted"},
                        "google_meet_browser",
                        recording,
                        True,
                        True,
                        True,
                    ),
                    "google_meet_browser": meet,
                    "latency_ledger": smoke.latency_ledger(task7),
                },
            )

        self.assertTrue(merged["telegram_desktop"]["completed"])
        self.assertTrue(merged["google_meet_browser"]["completed"])
        validation = smoke.validation_report(
            task7,
            environment,
            merged,
            {"passed": True, "log_scan": {"status": "passed"}},
            {"result": "passed"},
        )
        self.assertTrue(validation["completed"])
        self.assertTrue(validation["mvp_a_gate_satisfied"])

    def test_live_app_report_requires_candidate_for_that_app(self) -> None:
        smoke = load_smoke_module()
        recording = {"provided": True, "status": "present", "bytes": 1, "sha256": "0" * 64}
        report = smoke.app_report(
            "google_meet_browser",
            {},
            {
                "task10_route_candidates": [
                    {
                        "stream_id": 41,
                        "app_key": "telegram_desktop",
                        "call_like": True,
                        "meet_confirmed": False,
                    }
                ],
                "virtual_microphone": {
                    "expected_description": "Translator_Virtual_Mic",
                    "source_present": True,
                },
            },
            {"attempted": True, "status": "passed"},
            "google_meet_browser",
            recording,
            True,
            True,
            True,
        )

        self.assertFalse(report["completed"])
        self.assertEqual(report["route_discovery"]["candidate_count"], 0)
        self.assertIn("no_active_task10_route_candidate", report["blockers"])

    def test_live_app_report_requires_manual_route_for_same_candidate(self) -> None:
        smoke = load_smoke_module()
        recording = {"provided": True, "status": "present", "bytes": 1, "sha256": "0" * 64}
        report = smoke.app_report(
            "google_meet_browser",
            {},
            {
                "task10_route_candidates": [
                    {
                        "stream_id": 42,
                        "app_key": "google_meet_browser",
                        "call_like": True,
                        "meet_confirmed": True,
                    }
                ],
                "virtual_microphone": {
                    "expected_description": "Translator_Virtual_Mic",
                    "source_present": True,
                },
            },
            {
                "attempted": True,
                "status": "passed",
                "stream_id": 41,
                "candidate_app_key": "telegram_desktop",
            },
            "google_meet_browser",
            recording,
            True,
            True,
            True,
        )

        self.assertFalse(report["completed"])
        self.assertEqual(report["route_discovery"]["candidate_count"], 1)
        self.assertIn("incoming_manual_route_not_bound_to_app_candidate", report["blockers"])

    def test_merge_rejects_stale_completed_report_without_bound_route(self) -> None:
        smoke = load_smoke_module()
        stale_completed_report = {
            "schema_version": "translator.task10-real-app-smoke.v1",
            "app_key": "telegram_desktop",
            "completed": True,
            "route_discovery": {
                "candidates": [
                    {
                        "stream_id": 41,
                        "app_key": "telegram_desktop",
                        "call_like": True,
                    }
                ]
            },
            "incoming_route_selection": {
                "attempted": True,
                "status": "passed",
            },
            "virtual_microphone": {
                "call_app_selection_confirmed": True,
                "remote_outgoing_translation_confirmed": True,
            },
            "live_second_endpoint": {
                "status": "passed",
                "recording": {"provided": True, "status": "present"},
                "overlapping_speech_confirmed": True,
            },
        }
        current_blocked_report = {
            **stale_completed_report,
            "completed": False,
            "incoming_route_selection": {"attempted": False, "status": "not_attempted"},
            "live_second_endpoint": {"status": "blocked"},
        }

        with tempfile.TemporaryDirectory() as temp:
            output_dir = Path(temp)
            smoke.write_json(smoke.report_path(output_dir, "telegram_desktop"), stale_completed_report)
            merged = smoke.merge_existing_app_reports(
                output_dir,
                {
                    "telegram_desktop": current_blocked_report,
                    "google_meet_browser": current_blocked_report,
                    "latency_ledger": {},
                },
            )

        self.assertFalse(merged["telegram_desktop"]["completed"])

    @requires_local_artifacts(
        "docs/planning/translator-live-duplex-task-prompts.md",
        "docs/planning/translator-live-duplex-tasks.md",
    )
    def test_task10_planning_stays_open_until_real_second_endpoint_passes(self) -> None:
        prompts = read("docs/planning/translator-live-duplex-task-prompts.md")
        tasks = read("docs/planning/translator-live-duplex-tasks.md")

        prompt_section = prompts.split("## Task 10 Prompt", 1)[1].split("## Task 11 Prompt", 1)[0]
        task_section = tasks.split("## Task 10.", 1)[1].split("## Task 11.", 1)[0]

        self.assertNotIn("- [x] Completed", prompt_section)
        self.assertNotIn("- [x] Completed", task_section)


if __name__ == "__main__":
    unittest.main()

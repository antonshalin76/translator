from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import re
import stat
import unittest
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text()


def read_json(path: str) -> dict[str, Any]:
    return json.loads(read(path))


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


def load_module() -> Any:
    script = ROOT / "scripts/translator-simulated-app-stream-smoke"
    loader = importlib.machinery.SourceFileLoader("simulated_app_stream_smoke", str(script))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


class SimulatedAppStreamSmokeTests(unittest.TestCase):
    def test_script_is_executable_and_does_not_claim_live_task10(self) -> None:
        script_path = ROOT / "scripts/translator-simulated-app-stream-smoke"
        self.assertTrue(script_path.exists(), "simulated stream smoke script is missing")
        self.assertTrue(script_path.stat().st_mode & stat.S_IXUSR)
        script = script_path.read_text()

        for token in (
            "telegram_desktop",
            "google_meet_browser",
            "zoom_desktop",
            "simulated_app_streams",
            "/v1/routes/manual-override",
            "does_not_satisfy_task10_live_second_endpoint",
        ):
            self.assertIn(token, script)

        self.assertNotRegex(script, re.compile(r"\bopenai\b", re.IGNORECASE))
        self.assertNotIn("OPENAI_API_KEY", script)
        self.assertNotIn("pgrep -a", script)

    def test_report_shape_keeps_synthetic_evidence_separate(self) -> None:
        report = read_json("docs/benchmarks/task10-simulated-app-streams-report.json")

        self.assertEqual(report["schema_version"], "translator.task10-simulated-app-streams.v1")
        self.assertTrue(report["simulated_only"])
        self.assertTrue(report["does_not_satisfy_task10_live_second_endpoint"])
        self.assertFalse(report["task10_completed"])
        self.assertEqual(
            report["canonical_live_report"],
            "docs/benchmarks/task10-validation-report.json",
        )
        self.assertEqual(
            sorted(case["app_key"] for case in report["cases"]),
            ["google_meet_browser", "telegram_desktop", "zoom_desktop"],
        )

        zoom = next(case for case in report["cases"] if case["app_key"] == "zoom_desktop")
        self.assertEqual(zoom["task_scope"], "task12_diagnostic")
        self.assertFalse(zoom["counts_toward_task10_mvp_a"])
        self.assertIn(zoom["route_attempt"]["status"], {"blocked", "failed", "passed"})

    def test_browser_move_without_call_like_candidate_is_not_task10_route_pass(self) -> None:
        report = read_json("docs/benchmarks/task10-simulated-app-streams-report.json")
        meet = next(case for case in report["cases"] if case["app_key"] == "google_meet_browser")

        if meet["task10_candidate"] and meet["task10_candidate"]["call_like"] is False:
            self.assertEqual(
                meet["diagnostic_result"],
                "browser_stream_move_passed_without_call_like_candidate",
            )
            self.assertFalse(report["task10_synthetic_routes_passed"])

        smoke = load_module()
        cases = {case.app_key: case for case in smoke.CASES}
        result = smoke.classify_case_result(
            cases["google_meet_browser"],
            {"status": "passed"},
            {"on_translator_remote_in": True},
            {"call_like": False},
        )
        self.assertEqual(result, "browser_stream_move_passed_without_call_like_candidate")

    def test_report_carries_task7_debt_and_no_sensitive_payload(self) -> None:
        report = read_json("docs/benchmarks/task10-simulated-app-streams-report.json")
        debt = report["task7_debt_carried"]

        self.assertFalse(debt["task7_complete"])
        self.assertEqual(debt["local_provider_latency_classification"], "fails_usable_limit")
        self.assertTrue(debt["requires_mvp_b_provider_comparison"])

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
            "command_line",
            "cmdline",
            "url",
            "meeting_url",
            "pwd",
        }
        self.assertFalse(forbidden_keys.intersection(walk_keys(report)))
        rendered = json.dumps(report, ensure_ascii=False)
        self.assertNotIn("OPENAI_API_KEY", rendered)
        self.assertNotRegex(rendered, re.compile(r"zoommtg://", re.IGNORECASE))
        self.assertNotRegex(rendered, re.compile(r"\bpwd=", re.IGNORECASE))

    def test_planning_notes_do_not_mark_task10_complete(self) -> None:
        prompts = read("docs/planning/translator-live-duplex-task-prompts.md")
        tasks = read("docs/planning/translator-live-duplex-tasks.md")

        prompt_section = prompts.split("## Task 10 Prompt", 1)[1].split("## Task 11 Prompt", 1)[0]
        task_section = tasks.split("## Task 10.", 1)[1].split("## Task 11.", 1)[0]

        self.assertIn("task10-simulated-app-streams-report.json", prompt_section)
        self.assertIn("task10-simulated-app-streams-report.json", task_section)
        self.assertNotIn("- [x] Completed", prompt_section)
        self.assertNotIn("- [x] Completed", task_section)

    def test_case_definitions_keep_meet_browser_and_zoom_diagnostic_distinct(self) -> None:
        smoke = load_module()
        cases = {case.app_key: case for case in smoke.CASES}

        self.assertEqual(cases["google_meet_browser"].stream_kind, "browser_web_audio")
        self.assertEqual(cases["zoom_desktop"].task_scope, "task12_diagnostic")
        self.assertFalse(cases["zoom_desktop"].counts_toward_task10_mvp_a)
        self.assertEqual(cases["telegram_desktop"].stream_kind, "pulse_property_stream")


if __name__ == "__main__":
    unittest.main()

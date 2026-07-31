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


def load_preflight_module() -> Any:
    script = ROOT / "scripts/translator-task11-openai-preflight"
    loader = importlib.machinery.SourceFileLoader("task11_openai_preflight", str(script))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


class Task11OpenAIAdapterPreflightTests(unittest.TestCase):
    def test_preflight_script_is_executable_and_offline_only(self) -> None:
        script_path = ROOT / "scripts/translator-task11-openai-preflight"
        self.assertTrue(script_path.exists(), "Task 11 preflight script is missing")
        self.assertTrue(script_path.stat().st_mode & stat.S_IXUSR)
        script = script_path.read_text()

        for token in (
            "translator.task11-openai-adapter-preflight.v1",
            "cloud_opt_in_required",
            "missing_credentials_safe_error",
            "audio_leaves_machine_visible_before_session",
            "does_not_satisfy_task11_live_comparison",
            "task7_debt_carried",
            "task10_debt_carried",
            "openai_sidecar_runtime",
        ):
            self.assertIn(token, script)

        self.assertNotIn("curl https://", script.lower())
        self.assertNotIn("websocket.WebSocket", script)
        self.assertIn("sample_rate_hz=16_000", script)
        self.assertIn("frame_duration_ms=20", script)
        self.assertNotIn("frame_duration_ms=100", script)
        self.assertNotRegex(script, re.compile(r"sk-[A-Za-z0-9_-]+"))

    def test_synthetic_speech_smoke_script_is_executable_and_privacy_bounded(self) -> None:
        script_path = ROOT / "scripts/translator-task11-openai-synthetic-smoke"
        self.assertTrue(script_path.exists(), "Task 11 synthetic smoke script is missing")
        self.assertTrue(script_path.stat().st_mode & stat.S_IXUSR)
        script = script_path.read_text()

        for token in (
            "PiperTts",
            "websocket.WebSocket",
            "stored_audio",
            "stored_transcript_or_translation",
            "debug_text_enabled",
            "plaintext_credential_persisted",
        ):
            self.assertIn(token, script)

        self.assertNotRegex(script, re.compile(r"sk-[A-Za-z0-9_-]+"))
        self.assertNotIn("print(os.environ", script)

    def test_report_shape_carries_debts_and_keeps_task11_open(self) -> None:
        report = read_json("docs/benchmarks/task11-openai-adapter-preflight.json")

        self.assertEqual(
            report["schema_version"],
            "translator.task11-openai-adapter-preflight.v1",
        )
        self.assertTrue(report["preflight_contract_passed"])
        self.assertFalse(report["task11_completed"])
        self.assertTrue(report["does_not_satisfy_task11_live_comparison"])
        self.assertIn(
            report["live_openai_comparison"]["status"],
            {
                "pending",
                "pending_synthetic_speech_comparison",
                "passed_synthetic_speech_comparison",
            },
        )
        self.assertIn(
            report["real_app_openai_smoke"]["status"],
            {
                "pending_after_mvp_a_gate",
                "pending_after_websocket_smoke",
                "pending_after_synthetic_speech_comparison",
            },
        )
        self.assertEqual(
            report["openai_docs_basis"]["transport"],
            "websocket_realtime_translations_24khz_pcm16",
        )
        self.assertFalse(report["openai_docs_basis"]["source_transcription_configured"])
        self.assertTrue(report["openai_docs_basis"]["source_transcript_events_auto"])
        runtime = report["openai_sidecar_runtime"]
        self.assertEqual(
            runtime["status"],
            "wired_with_deterministic_fake_websocket_contract",
        )
        self.assertTrue(runtime["daemon_openai_launch_allowed"])
        self.assertTrue(runtime["grpc_provider_dispatches_openai_sessions"])
        self.assertTrue(runtime["runtime_websocket_payloads_exclude_credential"])
        self.assertEqual(runtime["daemon_provider_pcm_format"], "16000hz_mono_s16le_20ms")
        self.assertEqual(runtime["openai_wire_pcm_format"], "24000hz_mono_s16le")

        task7 = report["task7_debt_carried"]
        self.assertFalse(task7["task7_complete"])
        self.assertTrue(task7["requires_mvp_b_provider_comparison"])
        self.assertEqual(task7["local_provider_latency_classification"], "fails_usable_limit")

        task10 = report["task10_debt_carried"]
        self.assertTrue(task10["mvp_a_gate_satisfied"])
        self.assertEqual(
            task10["live_second_endpoint_status"],
            "passed_by_live_user_confirmation",
        )

    def test_report_records_safe_preflight_cases_without_secret_or_spoken_payload(self) -> None:
        report = read_json("docs/benchmarks/task11-openai-adapter-preflight.json")
        cases = {case["case"]: case for case in report["preflight_cases"]}

        self.assertEqual(cases["cloud_opt_in_required"]["safe_error_code"], "cloud_not_enabled")
        self.assertFalse(cases["cloud_opt_in_required"]["network_session_started"])
        self.assertEqual(
            cases["missing_credentials_safe_error"]["safe_error_code"],
            "provider_auth_failed",
        )
        self.assertFalse(cases["missing_credentials_safe_error"]["network_session_started"])
        self.assertTrue(
            cases["audio_leaves_machine_visible_before_session"]["audio_leaves_machine"]
        )
        self.assertFalse(
            cases["audio_leaves_machine_visible_before_session"]["network_session_started"]
        )
        if "credential_model_probe" in report:
            self.assertEqual(report["credential_model_probe"]["model"], "gpt-realtime-translate")
            self.assertFalse(report["credential_model_probe"]["plaintext_credential_persisted"])
        if "realtime_websocket_smoke" in report:
            self.assertFalse(report["realtime_websocket_smoke"]["stored_audio"])
            self.assertFalse(
                report["realtime_websocket_smoke"]["stored_transcript_or_translation"]
            )
            if report["realtime_websocket_smoke"]["status"] == "passed":
                expected_status = (
                    "pending_after_synthetic_speech_comparison"
                    if report.get("synthetic_speech_comparison", {}).get("status")
                    == "passed"
                    else "pending_after_websocket_smoke"
                )
                self.assertEqual(
                    report["real_app_openai_smoke"]["status"],
                    expected_status,
                )

        forbidden_keys = {
            "pcm",
            "pcm_bytes",
            "raw_pcm",
            "audio",
            "transcript",
            "translation",
            "translation_text",
            "source_text",
            "target_text",
            "input_text",
            "output_text",
            "spoken_content",
            "spoken_phrase",
            "api_key",
            "authorization",
        }
        self.assertFalse(forbidden_keys.intersection(walk_keys(report)))
        rendered = json.dumps(report, ensure_ascii=False)
        self.assertNotIn("OPENAI_API_KEY", rendered)
        self.assertNotRegex(rendered, re.compile(r"sk-[A-Za-z0-9_-]+"))

    def test_synthetic_speech_comparison_report_is_privacy_safe(self) -> None:
        report = read_json("docs/benchmarks/task11-openai-adapter-preflight.json")
        comparison = report["synthetic_speech_comparison"]

        self.assertIn(
            comparison["status"],
            {
                "pending_after_websocket_smoke",
                "passed",
                "partial",
                "failed",
                "blocked_synthetic_speech_unavailable",
                "blocked_missing_credential",
            },
        )
        self.assertFalse(comparison["stored_audio"])
        self.assertFalse(comparison["stored_transcript_or_translation"])
        self.assertFalse(comparison["debug_text_enabled"])
        self.assertFalse(comparison["plaintext_credential_persisted"])
        self.assertEqual(comparison["provider_id"], "openai")
        self.assertEqual(comparison["model"], "gpt-realtime-translate")

        if comparison["status"] == "passed":
            self.assertTrue(comparison["simultaneous"])
            self.assertEqual(comparison["sample_count_per_direction"], 1)
            self.assertEqual(set(comparison["directions"]), {"ru_to_en", "en_to_ru"})
            self.assertEqual(
                comparison["directions"]["ru_to_en"]["source_language"],
                "ru",
            )
            self.assertEqual(
                comparison["directions"]["ru_to_en"]["target_language"],
                "en",
            )
            self.assertEqual(
                comparison["directions"]["en_to_ru"]["source_language"],
                "en",
            )
            self.assertEqual(
                comparison["directions"]["en_to_ru"]["target_language"],
                "ru",
            )
            for direction in comparison["directions"].values():
                self.assertGreater(direction["sent_frame_count"], 0)
                self.assertGreater(direction["sent_duration_ms"], 0)
                self.assertGreater(direction["output_audio_delta_count"], 0)
                self.assertGreater(direction["first_output_audio_delta_ms"], 0)

    def test_preflight_module_builds_report_from_current_task_debt(self) -> None:
        module = load_preflight_module()
        report = module.build_report()
        task7 = read_json("docs/benchmarks/task7-live-human-round-trip.json")
        task10 = read_json("docs/benchmarks/task10-validation-report.json")

        self.assertEqual(
            report["task7_debt_carried"]["local_provider_latency_classification"],
            task7["classification"],
        )
        self.assertEqual(
            report["task10_debt_carried"]["mvp_a_gate_satisfied"],
            task10["mvp_a_gate_satisfied"],
        )
        self.assertFalse(report["task11_completed"])
        self.assertIn("synthetic_speech_comparison", report)

    def test_planning_notes_record_preflight_without_marking_task11_complete(self) -> None:
        prompts = read("docs/planning/translator-live-duplex-task-prompts.md")
        tasks = read("docs/planning/translator-live-duplex-tasks.md")

        prompt_section = prompts.split("## Task 11 Prompt", 1)[1].split("## Task 12 Prompt", 1)[0]
        task_section = tasks.split("## Task 11.", 1)[1].split("## Task 12.", 1)[0]

        self.assertIn("task11-openai-adapter-preflight.json", prompt_section)
        self.assertIn("task11-openai-adapter-preflight.json", task_section)
        self.assertNotIn("- [x] Completed", prompt_section)
        self.assertNotIn("- [x] Completed", task_section)


if __name__ == "__main__":
    unittest.main()

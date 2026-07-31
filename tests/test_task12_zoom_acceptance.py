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
FORBIDDEN_PRIVATE_REPORT_KEYS = {
    "pcm",
    "pcm_bytes",
    "raw_pcm",
    "transcript",
    "translation_text",
    "source_text",
    "target_text",
    "spoken_content",
    "spoken_phrase",
    "meeting_link",
    "meeting_url",
    "join_url",
    "meeting_id",
    "meeting_password",
    "url",
    "password",
    "pwd",
    "authorization",
    "api_key",
    "token",
    "secret",
    "credential",
    "credentials",
}


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


def load_zoom_module() -> Any:
    script = ROOT / "scripts/translator-task12-zoom-diagnostic"
    loader = importlib.machinery.SourceFileLoader("task12_zoom_diagnostic", str(script))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


class Task12ZoomAcceptanceTests(unittest.TestCase):
    def test_zoom_diagnostic_script_is_executable_and_scope_safe(self) -> None:
        script_path = ROOT / "scripts/translator-task12-zoom-diagnostic"
        self.assertTrue(script_path.exists(), "Task 12 Zoom diagnostic script is missing")
        self.assertTrue(script_path.stat().st_mode & stat.S_IXUSR)
        script = script_path.read_text()

        for token in (
            "translator.task12-zoom-diagnostic.v1",
            "zoom_desktop",
            "Translator_Virtual_Mic",
            "does_not_satisfy_task12_acceptance",
            "task7_debt_carried",
            "task10_debt_carried",
            "task11_debt_carried",
            "zoom_setup_notes",
        ):
            self.assertIn(token, script)

        self.assertNotIn("zoommtg://", script.lower())
        self.assertNotRegex(script, re.compile(r"\bpwd=", re.IGNORECASE))
        self.assertNotIn("OPENAI_API_KEY", script)

    def test_zoom_diagnostic_report_records_completed_acceptance_and_carries_debts(self) -> None:
        report = read_json("docs/benchmarks/task12-zoom-diagnostic-report.json")

        self.assertEqual(report["schema_version"], "translator.task12-zoom-diagnostic.v1")
        self.assertTrue(report["task12_completed"])
        self.assertFalse(report["does_not_satisfy_task12_acceptance"])
        self.assertEqual(report["blockers"], [])
        self.assertEqual(report["zoom_desktop"]["app_key"], "zoom_desktop")
        route_selection = report["acceptance"]["zoom_route_selection"]
        self.assertEqual(route_selection["status"], "passed")
        if route_selection["reason"] != "superseded_by_live_duplex_confirmation":
            self.assertTrue(
                any(
                    candidate["on_translator_remote_in"]
                    or candidate["pipewire_linked_to_translator_remote_in"]
                    for candidate in report["zoom_desktop"]["route_discovery"]["candidates"]
                )
            )
        self.assertIn(
            route_selection["reason"],
            {
                "zoom_sink_input_bound_to_translator_remote_in",
                "zoom_pipewire_links_bound_to_translator_remote_in",
                "superseded_by_live_duplex_confirmation",
            },
        )
        self.assertEqual(report["acceptance"]["zoom_outgoing_translation"]["status"], "passed")
        self.assertEqual(report["acceptance"]["zoom_incoming_translation"]["status"], "passed")

        self.assertFalse(report["task7_debt_carried"]["task7_complete"])
        self.assertEqual(
            report["task7_debt_carried"]["local_provider_latency_classification"],
            "fails_usable_limit",
        )
        self.assertTrue(report["task10_debt_carried"]["mvp_a_gate_satisfied"])
        self.assertFalse(report["task11_debt_carried"]["task11_completed"])
        self.assertEqual(
            report["task11_debt_carried"]["live_openai_comparison_status"],
            "passed_synthetic_speech_comparison",
        )
        self.assertEqual(
            report["task11_debt_carried"]["openai_sidecar_runtime_status"],
            "wired_with_deterministic_fake_websocket_contract",
        )

    def test_zoom_diagnostic_report_has_setup_notes_without_private_payload(self) -> None:
        report = read_json("docs/benchmarks/task12-zoom-diagnostic-report.json")

        notes = report["zoom_setup_notes"]
        self.assertIn("microphone", notes)
        self.assertIn("speaker", notes)
        self.assertEqual(notes["microphone"], "Select Translator_Virtual_Mic in Zoom audio settings")
        self.assertEqual(notes["speaker"], "Keep Zoom playback on the normal physical sink; translator routes only the selected sink-input")

        self.assertFalse(FORBIDDEN_PRIVATE_REPORT_KEYS.intersection(walk_keys(report)))
        rendered = json.dumps(report, ensure_ascii=False)
        self.assertNotIn("OPENAI_API_KEY", rendered)
        self.assertNotRegex(rendered, re.compile(r"zoommtg://", re.IGNORECASE))
        self.assertNotRegex(rendered, re.compile(r"\bpwd=", re.IGNORECASE))

    def test_zoom_live_translation_report_records_full_duplex_acceptance(self) -> None:
        report = read_json("docs/benchmarks/task12-zoom-live-translation-check.json")

        self.assertEqual(
            report["schema_version"],
            "translator.task12-zoom-live-translation-check.v1",
        )
        self.assertTrue(report["task12_completed"])
        self.assertFalse(report["does_not_satisfy_task12_acceptance"])
        self.assertEqual(report["acceptance"]["zoom_route_selection"]["status"], "passed")
        self.assertEqual(report["acceptance"]["zoom_incoming_translation"]["status"], "passed")
        self.assertEqual(
            report["acceptance"]["zoom_outgoing_translation"]["status"],
            "passed",
        )

        live = report["live_zoom_translation"]
        self.assertEqual(live["provider_id"], "local")
        self.assertFalse(live["audio_leaves_machine"])
        self.assertFalse(live["debug_text_enabled"])
        self.assertFalse(live["debug_capture_enabled"])
        self.assertEqual(live["route_method"], "pipe_wire_links")
        self.assertEqual(live["direction"]["source_language"], "ru")
        self.assertEqual(live["direction"]["target_language"], "en")
        self.assertTrue(live["user_confirmation"]["incoming_translation_audible"])
        self.assertTrue(live["user_confirmation"]["duplex_translation_worked"])
        self.assertFalse(live["user_confirmation"]["content_recorded"])

        self.assertTrue(report["runtime_adjustments"]["headphone_sink_override_used"])
        self.assertTrue(
            report["runtime_adjustments"]["headphone_sink_override_unset_after_cleanup"]
        )
        self.assertEqual(
            report["runtime_adjustments"]["incoming_playback_volume_before_fix"],
            "0%",
        )
        self.assertEqual(
            report["runtime_adjustments"]["incoming_playback_volume_after_fix"],
            "100%",
        )
        self.assertEqual(
            report["cleanup"]["translator_service_active_state_after_cleanup"],
            "inactive",
        )

        self.assertEqual(report["blockers"], [])
        self.assertFalse(report["task7_debt_carried"]["task7_complete"])
        self.assertTrue(report["task10_debt_carried"]["mvp_a_gate_satisfied"])
        self.assertFalse(report["task11_debt_carried"]["task11_completed"])
        self.assertEqual(
            report["task11_debt_carried"]["live_openai_comparison_status"],
            "passed_synthetic_speech_comparison",
        )
        self.assertEqual(
            report["task11_debt_carried"]["openai_sidecar_runtime_status"],
            "wired_with_deterministic_fake_websocket_contract",
        )

        self.assertFalse(FORBIDDEN_PRIVATE_REPORT_KEYS.intersection(walk_keys(report)))
        rendered = json.dumps(report, ensure_ascii=False)
        self.assertNotIn("OPENAI_API_KEY", rendered)
        self.assertNotRegex(rendered, re.compile(r"zoommtg://", re.IGNORECASE))
        self.assertNotRegex(rendered, re.compile(r"\bpwd=", re.IGNORECASE))

    def test_zoom_module_builds_report_from_current_debt_reports(self) -> None:
        module = load_zoom_module()
        report = module.build_report()
        task7 = read_json("docs/benchmarks/task7-live-human-round-trip.json")
        task10 = read_json("docs/benchmarks/task10-validation-report.json")
        task11 = read_json("docs/benchmarks/task11-openai-adapter-preflight.json")

        self.assertEqual(
            report["task7_debt_carried"]["local_provider_latency_classification"],
            task7["classification"],
        )
        self.assertEqual(
            report["task10_debt_carried"]["mvp_a_gate_satisfied"],
            task10["mvp_a_gate_satisfied"],
        )
        self.assertEqual(
            report["task11_debt_carried"]["task11_completed"],
            task11["task11_completed"],
        )
        self.assertTrue(report["task12_completed"])

    def test_zoom_diagnostic_call_like_heuristic_matches_routing_contract(self) -> None:
        module = load_zoom_module()

        self.assertTrue(module.is_call_like({"media.role": "communication"}))
        self.assertTrue(module.is_call_like({"media.name": "WebRTC Voice"}))
        self.assertTrue(module.is_call_like({"stream.description": "Meet Audio"}))
        self.assertTrue(module.is_call_like({"media.role": "music", "media.name": "Meeting Lobby"}))
        self.assertFalse(module.is_call_like({"media.role": "music", "media.name": "Zoom Notification"}))

    def test_zoom_diagnostic_detects_pipewire_link_routing(self) -> None:
        module = load_zoom_module()
        links = """
translator_remote_in:playback_FL
  |<- ZOOM VoiceEngine:output_FL
translator_remote_in:playback_FR
  |<- ZOOM VoiceEngine:output_FR
ZOOM VoiceEngine:output_FL
  |-> translator_remote_in:playback_FL
ZOOM VoiceEngine:output_FR
  |-> translator_remote_in:playback_FR
"""

        self.assertTrue(module.pipewire_link_route_present("ZOOM VoiceEngine", links))
        self.assertFalse(module.pipewire_link_route_present("ZOOM VoiceEngine:bad", links))
        self.assertFalse(module.pipewire_link_route_present("Other App", links))

    def test_zoom_diagnostic_uses_pipewire_node_name_for_link_routing(self) -> None:
        module = load_zoom_module()
        links = """
ZOOM VoiceEngine:output_FL
  |-> translator_remote_in:playback_FL
ZOOM VoiceEngine:output_FR
  |-> translator_remote_in:playback_FR
"""

        def fake_command_json(args, timeout=10.0):
            del timeout
            if args[-1] == "sink-inputs":
                return (
                    [
                        {
                            "index": 77,
                            "sink": 42,
                            "properties": {
                                "application.name": "Zoom Workplace",
                                "application.process.binary": "zoom",
                                "node.name": "ZOOM VoiceEngine",
                                "media.role": "communication",
                            },
                        }
                    ],
                    None,
                )
            if args[-1] == "sinks":
                return ([{"index": 42, "name": "alsa_output"}], None)
            if args[-1] == "sources":
                return (
                    [
                        {
                            "name": "translator_virtual_mic",
                            "description": "Translator_Virtual_Mic",
                        }
                    ],
                    None,
                )
            return ([], None)

        def fake_run_command(args, timeout=10.0):
            del timeout
            if args[:2] == ["pw-link", "-l"]:
                return {"ok": True, "stdout": links, "safe_error": None}
            return {"ok": True, "stdout": "", "safe_error": None}

        original_command_json = module.command_json
        original_run_command = module.run_command
        try:
            module.command_json = fake_command_json
            module.run_command = fake_run_command
            state = module.collect_audio_state()
        finally:
            module.command_json = original_command_json
            module.run_command = original_run_command

        candidate = state["zoom_route_candidates"][0]
        self.assertEqual(candidate["application_name"], "Zoom Workplace")
        self.assertEqual(candidate["node_name"], "ZOOM VoiceEngine")
        self.assertTrue(candidate["pipewire_linked_to_translator_remote_in"])

    def test_planning_notes_record_completed_zoom_duplex_without_closing_debts(self) -> None:
        prompts = read("docs/planning/translator-live-duplex-task-prompts.md")
        tasks = read("docs/planning/translator-live-duplex-tasks.md")

        prompt_section = prompts.split("## Task 12 Prompt", 1)[1]
        task_section = tasks.split("## Task 12.", 1)[1]

        self.assertIn("task12-zoom-diagnostic-report.json", prompt_section)
        self.assertIn("task12-zoom-diagnostic-report.json", task_section)
        self.assertIn("task12-zoom-live-translation-check.json", prompt_section)
        self.assertIn("task12-zoom-live-translation-check.json", task_section)
        self.assertIn("Full Task 12 Zoom duplex acceptance is user-confirmed", prompt_section)
        self.assertIn("Full Task 12 Zoom duplex acceptance is user-confirmed", task_section)
        self.assertIn("Task 7 latency and Task 11 OpenAI comparison debts", prompt_section)
        self.assertIn("Task 7 latency and Task 11 OpenAI comparison debts", task_section)
        self.assertNotIn("- [x] Completed", prompt_section)
        self.assertNotIn("- [x] Completed", task_section)


if __name__ == "__main__":
    unittest.main()

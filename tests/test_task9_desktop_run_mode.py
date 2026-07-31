from __future__ import annotations

import json
import os
import re
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text()


class Task9DesktopRunModeTests(unittest.TestCase):
    def test_systemd_unit_is_user_session_owner_with_bounded_restart(self) -> None:
        unit = read("systemd/translator.service")

        self.assertIn("Type=simple", unit)
        self.assertIn("ExecStart=/usr/bin/env translator-daemon", unit)
        self.assertIn("EnvironmentFile=-%h/Source/translator/.env", unit)
        self.assertIn("Restart=on-failure", unit)
        self.assertRegex(unit, r"(?m)^RestartSec=[1-9][0-9]*s?$")
        self.assertIn("RuntimeDirectory=translator", unit)
        self.assertIn("RuntimeDirectoryMode=0700", unit)
        self.assertIn("RuntimeDirectoryPreserve=restart", unit)
        self.assertIn("KillMode=control-group", unit)
        self.assertIn("UMask=0077", unit)
        self.assertIn("NoNewPrivileges=true", unit)
        self.assertIn("ProtectSystem=strict", unit)
        self.assertRegex(unit, r"(?m)^ReadWritePaths=.*%t/translator")
        self.assertIn("WantedBy=default.target", unit)
        self.assertNotRegex(unit, r"(?m)^(User|Group|WantedBy)=root$")
        self.assertNotIn("sudo", unit)

    def test_desktop_lifecycle_script_installs_unit_and_tauri_autostart_only(self) -> None:
        script_path = ROOT / "scripts/translator-desktop"
        self.assertTrue(script_path.exists(), "Task 9 desktop lifecycle script is missing")
        mode = script_path.stat().st_mode
        self.assertTrue(mode & stat.S_IXUSR, "desktop lifecycle script must be executable")
        script = script_path.read_text()

        for action in (
            "install",
            "up",
            "start",
            "stop",
            "down",
            "restart",
            "status",
            "logs",
            "disable",
            "uninstall",
        ):
            self.assertRegex(script, rf"(?m)^\s*{action}\)")

        self.assertIn("systemctl --user daemon-reload", script)
        self.assertIn("systemctl --user enable translator.service", script)
        self.assertIn("systemctl --user start translator.service", script)
        self.assertIn("systemctl --user stop translator.service", script)
        self.assertIn("translator-daemon --audio-graph-cleanup", script)
        self.assertIn("journalctl --user-unit translator.service", script)
        self.assertIn('command_target="${user_bin_dir}/translator"', script)
        self.assertIn("install_command", script)
        self.assertIn("remove_command_if_owned", script)
        self.assertIn("install_ui_binary_if_available", script)
        self.assertIn('ui_binary_source="${project_root}/target/release/translator-ui"', script)
        self.assertIn('ui_binary_target="${user_bin_dir}/translator-ui"', script)
        self.assertIn("install_daemon_binary_if_available", script)
        self.assertIn('daemon_binary_source="${project_root}/target/release/translator-daemon"', script)
        self.assertIn('daemon_binary_target="${user_bin_dir}/translator-daemon"', script)
        self.assertIn("refusing to overwrite existing translator command", script)
        self.assertIn("start_unit_installing_if_needed", script)
        self.assertIn("print_service_summary", script)
        self.assertIn("systemctl --user disable translator.service", script)
        self.assertNotIn("sudo", script)
        self.assertNotIn("systemctl start translator.service", script)
        self.assertNotIn("systemctl enable translator.service", script)

        self.assertIn("systemd/user", script)
        self.assertIn("autostart", script)
        self.assertIn("XDG_CONFIG_HOME", script)
        self.assertIn(".local/bin", script)
        self.assertIn("translator-ui.desktop", script)
        self.assertIn("Exec=translator-ui", script)
        self.assertNotIn("Exec=translator-daemon", script)

    def test_up_starts_daemon_and_current_desktop_ui(self) -> None:
        script_path = ROOT / "scripts/translator-desktop"

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fake_bin = tmp_path / "bin"
            fake_bin.mkdir()
            log_path = tmp_path / "calls.log"
            runtime_dir = tmp_path / "runtime"
            (runtime_dir / "translator").mkdir(parents=True)
            (runtime_dir / "translator" / "control.token").write_text("a" * 64)

            (fake_bin / "systemctl").write_text(
                "#!/usr/bin/env bash\n"
                "printf 'systemctl %s\\n' \"$*\" >>\"${TRANSLATOR_TEST_LOG}\"\n"
                "if [ \"$1\" = \"--user\" ] && [ \"${2:-}\" = \"list-unit-files\" ]; then exit 1; fi\n"
                "if [ \"$1\" = \"--user\" ] && [ \"${2:-}\" = \"show\" ]; then\n"
                "  printf '%s\\n' 'LoadState=loaded' 'ActiveState=active' 'SubState=running' 'MainPID=123' 'Result=success'\n"
                "fi\n"
            )
            (fake_bin / "pgrep").write_text(
                "#!/usr/bin/env bash\n"
                "printf 'pgrep %s\\n' \"$*\" >>\"${TRANSLATOR_TEST_LOG}\"\n"
                "exit 1\n"
            )
            (fake_bin / "setsid").write_text(
                "#!/usr/bin/env bash\n"
                "printf 'setsid %s\\n' \"$*\" >>\"${TRANSLATOR_TEST_LOG}\"\n"
                "exit 0\n"
            )
            (fake_bin / "curl").write_text(
                "#!/usr/bin/env bash\n"
                "printf 'curl %s\\n' \"$*\" >>\"${TRANSLATOR_TEST_LOG}\"\n"
                "exit 0\n"
            )
            (fake_bin / "translator-ui").write_text(
                "#!/usr/bin/env bash\n"
                "printf '%s\\n' translator-ui >>\"${TRANSLATOR_TEST_LOG}\"\n"
            )
            for helper in fake_bin.iterdir():
                helper.chmod(0o700)

            env = os.environ.copy()
            env.update(
                {
                    "HOME": str(tmp_path / "home"),
                    "XDG_CONFIG_HOME": str(tmp_path / "config"),
                    "XDG_RUNTIME_DIR": str(runtime_dir),
                    "PATH": f"{fake_bin}:{env['PATH']}",
                    "TRANSLATOR_TEST_LOG": str(log_path),
                    "WAYLAND_DISPLAY": "wayland-test",
                }
            )

            result = subprocess.run(
                [str(script_path), "up"],
                cwd="/tmp",
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            calls = log_path.read_text()
            self.assertIn("systemctl --user start translator.service", calls)
            self.assertIn("pgrep -u", calls)
            self.assertIn("translator-ui", calls)
            self.assertIn("setsid -f", calls)
            self.assertIn("curl -sS -m 0.5 -o /dev/null http://127.0.0.1:47681/v1/status", calls)
            self.assertLess(calls.index("curl "), calls.index("setsid -f"))

    def test_task9_smoke_script_checks_user_service_without_cloud_or_live_provider(self) -> None:
        smoke_path = ROOT / "scripts/translator-task9-smoke"
        self.assertTrue(smoke_path.exists(), "Task 9 smoke script is missing")
        mode = smoke_path.stat().st_mode
        self.assertTrue(mode & stat.S_IXUSR, "Task 9 smoke script must be executable")
        smoke = smoke_path.read_text()

        self.assertIn("systemctl --user start translator.service", smoke)
        self.assertIn("systemctl --user kill --signal=KILL translator.service", smoke)
        self.assertIn("systemctl --user stop translator.service", smoke)
        self.assertIn("systemctl --user show translator.service", smoke)
        self.assertRegex(smoke, r"translator-daemon\s+--audio-graph-cleanup")
        self.assertIn("pactl list short sinks", smoke)
        self.assertIn("pactl list short sources", smoke)
        self.assertNotIn("openai", smoke.lower())
        self.assertNotIn("curl https://", smoke.lower())

    def test_tauri_backend_reports_daemon_state_without_managing_systemd_unit(self) -> None:
        tauri = read("apps/translator-ui/src-tauri/src/main.rs")

        self.assertIn("control_token_path", tauri)
        self.assertIn("XDG_RUNTIME_DIR", tauri)
        self.assertNotIn('Command::new("translator-daemon"', tauri)
        self.assertNotIn('Command::new("systemctl"', tauri)
        self.assertNotRegex(tauri, r"\bsystemctl\s+--user\s+(?:start|stop|restart|enable|disable)")

    def test_task9_validation_report_carries_task7_latency_debt(self) -> None:
        report_path = ROOT / "docs/benchmarks/task9-validation-report.json"
        self.assertTrue(report_path.exists(), "Task 9 validation report is missing")
        report = json.loads(report_path.read_text())
        task7 = json.loads((ROOT / "docs/benchmarks/task7-live-human-round-trip.json").read_text())
        task7_debt = report["task7_debt_carried"]

        self.assertEqual(report["schema_version"], "translator.task9-validation.v1")
        self.assertEqual(
            task7_debt["canonical_evidence"],
            "docs/benchmarks/task7-live-human-round-trip.json",
        )
        self.assertEqual(task7_debt["local_provider_latency_classification"], task7["classification"])
        self.assertEqual(task7_debt["task7_complete"], task7["acceptance"]["task7_complete"])
        self.assertFalse(task7_debt["task7_complete"])
        self.assertTrue(task7_debt["requires_mvp_b_provider_comparison"])
        self.assertEqual(
            task7_debt["blocked_acceptance_item"],
            task7["acceptance"]["blocked_acceptance_item"],
        )

    def test_task9_files_do_not_enable_cloud_or_persist_debug_text(self) -> None:
        checked_paths = [
            ROOT / "systemd/translator.service",
            ROOT / "scripts/translator-desktop",
            ROOT / "scripts/translator-task9-smoke",
        ]
        combined = "\n".join(path.read_text() for path in checked_paths if path.exists())

        self.assertNotIn("OPENAI_API_KEY", combined)
        self.assertNotRegex(combined, re.compile(r"debug_text.*true", re.IGNORECASE))
        self.assertNotRegex(combined, re.compile(r"debug-capture.*true", re.IGNORECASE))
        self.assertFalse(
            any(
                token in combined
                for token in ("transcript", "translation_text", "raw_pcm", "pcm_bytes")
            )
        )

    def test_daemon_main_uses_runtime_route_journal_for_crash_recovery(self) -> None:
        daemon = read("crates/translator-daemon/src/main.rs")

        self.assertIn("default_route_journal_path", daemon)
        self.assertIn("PulseRoutingWatcher::new_with_route_journal", daemon)

    def test_planning_documents_record_terminal_lifecycle_command(self) -> None:
        design = read("docs/planning/translator-live-duplex-design.md")
        prompts = read("docs/planning/translator-live-duplex-task-prompts.md")
        tasks = read("docs/planning/translator-live-duplex-tasks.md")
        combined = "\n".join((design, prompts, tasks))

        self.assertIn("scripts/translator-desktop up", combined)
        self.assertIn("scripts/translator-desktop down", combined)
        self.assertIn("scripts/translator-desktop restart", combined)
        self.assertIn("scripts/translator-desktop logs", combined)
        self.assertIn("translator up", combined)
        self.assertIn("~/.local/bin/translator", combined)
        self.assertIn("audio graph cleanup", combined.lower())


if __name__ == "__main__":
    unittest.main()

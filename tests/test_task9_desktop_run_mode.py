from __future__ import annotations

import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text()


def requires_local_artifacts(*paths: str):
    return unittest.skipUnless(
        all((ROOT / path).exists() for path in paths),
        "missing_external_prerequisite:private_human_evidence",
    )


class Task9DesktopRunModeTests(unittest.TestCase):
    def _desktop_security_test_environment(
        self, tmp_path: Path
    ) -> tuple[dict[str, str], Path, Path, Path]:
        fake_bin = tmp_path / "bin"
        fake_bin.mkdir()
        log_path = tmp_path / "calls.log"
        config_root = tmp_path / "config"
        config_root.mkdir(mode=0o700)
        environment_dir = config_root / "translator"
        environment_file = environment_dir / "environment"

        (fake_bin / "systemctl").write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            'printf \'systemctl %s\\n\' "$*" >>"${TRANSLATOR_TEST_LOG}"\n'
            'if [ "$1" = "--user" ] && [ "${2:-}" = "list-unit-files" ]; then\n'
            "  printf '%s\\n' 'translator.service enabled'\n"
            "  exit 0\n"
            "fi\n"
            'if [ "$1" = "--user" ] && [ "${2:-}" = "start" ]; then\n'
            '  test ! -L "${TRANSLATOR_TEST_ENV_DIR}"\n'
            '  test -d "${TRANSLATOR_TEST_ENV_DIR}"\n'
            '  test "$(/usr/bin/stat -c %a -- "${TRANSLATOR_TEST_ENV_DIR}")" = 700\n'
            '  test "$(/usr/bin/stat -c %u -- "${TRANSLATOR_TEST_ENV_DIR}")" = "${TRANSLATOR_TEST_UID}"\n'
            '  test ! -L "${TRANSLATOR_TEST_ENV_FILE}"\n'
            '  test -f "${TRANSLATOR_TEST_ENV_FILE}"\n'
            '  test "$(/usr/bin/stat -c %a -- "${TRANSLATOR_TEST_ENV_FILE}")" = 600\n'
            '  test "$(/usr/bin/stat -c %u -- "${TRANSLATOR_TEST_ENV_FILE}")" = "${TRANSLATOR_TEST_UID}"\n'
            '  test "$(/usr/bin/stat -c %h -- "${TRANSLATOR_TEST_ENV_FILE}")" = 1\n'
            "fi\n"
        )
        (fake_bin / "curl").write_text("#!/usr/bin/env bash\nexit 0\n")
        (fake_bin / "pgrep").write_text("#!/usr/bin/env bash\nexit 0\n")
        (fake_bin / "bun").write_text("#!/usr/bin/env bash\nexit 1\n")
        for helper in fake_bin.iterdir():
            helper.chmod(0o700)

        env = {
            "HOME": str(tmp_path / "home"),
            "XDG_CONFIG_HOME": str(config_root),
            "PATH": f"{fake_bin}:/usr/bin:/bin",
            "TRANSLATOR_TEST_ENV_DIR": str(environment_dir),
            "TRANSLATOR_TEST_ENV_FILE": str(environment_file),
            "TRANSLATOR_TEST_LOG": str(log_path),
            "TRANSLATOR_TEST_UID": str(os.getuid()),
        }
        return env, environment_dir, environment_file, log_path

    def _run_desktop_action(
        self, env: dict[str, str], action: str
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(ROOT / "scripts/translator-desktop"), action],
            cwd="/tmp",
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

    def _run_desktop_security_case(
        self, tmp_path: Path, action: str = "start"
    ) -> tuple[subprocess.CompletedProcess[str], Path, Path, Path]:
        env, environment_dir, environment_file, log_path = (
            self._desktop_security_test_environment(tmp_path)
        )
        result = self._run_desktop_action(env, action)
        return result, environment_dir, environment_file, log_path

    def _assert_start_rejected_before_systemd(
        self, env: dict[str, str], log_path: Path
    ) -> subprocess.CompletedProcess[str]:
        result = self._run_desktop_action(env, "start")
        self.assertNotEqual(result.returncode, 0)
        calls = log_path.read_text() if log_path.exists() else ""
        self.assertNotIn("systemctl --user start translator.service", calls)
        return result

    def _install_test_daemon(self, tmp_path: Path, source: str = "/bin/true") -> Path:
        daemon = tmp_path / "home" / ".local" / "bin" / "translator-daemon"
        daemon.parent.mkdir(parents=True)
        shutil.copy2(source, daemon)
        daemon.chmod(0o700)
        return daemon

    def _run_daemon_wrapper(
        self, env: dict[str, str], config_root: Path, daemon: Path
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                str(ROOT / "scripts/translator-desktop"),
                "_run-daemon",
                str(config_root),
                str(daemon),
            ],
            cwd="/tmp",
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_start_creates_private_service_environment_before_systemd(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result, environment_dir, environment_file, log_path = (
                self._run_desktop_security_case(Path(tmp))
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(environment_dir.is_symlink())
            self.assertTrue(environment_dir.is_dir())
            self.assertEqual(stat.S_IMODE(environment_dir.stat().st_mode), 0o700)
            self.assertEqual(environment_dir.stat().st_uid, os.getuid())
            self.assertFalse(environment_file.is_symlink())
            self.assertTrue(environment_file.is_file())
            self.assertEqual(stat.S_IMODE(environment_file.stat().st_mode), 0o600)
            self.assertEqual(environment_file.stat().st_uid, os.getuid())
            self.assertEqual(environment_file.stat().st_nlink, 1)
            self.assertIn(
                "systemctl --user start translator.service", log_path.read_text()
            )

    def test_start_accepts_safe_environment_without_disclosing_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env, environment_dir, environment_file, log_path = (
                self._desktop_security_test_environment(Path(tmp))
            )
            environment_dir.mkdir(mode=0o700)
            marker = "private-environment-value-7f8a"
            environment_file.write_text(f"TRANSLATOR_HEADPHONE_SINK={marker}\n")
            environment_file.chmod(0o600)

            result = self._run_desktop_action(env, "start")

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                environment_file.read_text(), f"TRANSLATOR_HEADPHONE_SINK={marker}\n"
            )
            observable_output = result.stdout + result.stderr + log_path.read_text()
            self.assertNotIn(marker, observable_output)

    def test_start_rejects_invalid_environment_content_before_systemd(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env, environment_dir, environment_file, log_path = (
                self._desktop_security_test_environment(Path(tmp))
            )
            environment_dir.mkdir(mode=0o700)
            marker = "private-loader-value-before-systemd-4d2c"
            environment_file.write_text(f"LD_PRELOAD={marker}\n")
            environment_file.chmod(0o600)

            result = self._assert_start_rejected_before_systemd(env, log_path)

            self.assertEqual(result.returncode, 78)
            self.assertIn("unsupported service environment key", result.stderr)
            self.assertNotIn(marker, result.stdout + result.stderr)

    def test_install_creates_private_service_environment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result, environment_dir, environment_file, log_path = (
                self._run_desktop_security_case(Path(tmp), action="install")
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(environment_dir.is_dir())
            self.assertTrue(environment_file.is_file())
            self.assertEqual(stat.S_IMODE(environment_dir.stat().st_mode), 0o700)
            self.assertEqual(stat.S_IMODE(environment_file.stat().st_mode), 0o600)
            self.assertFalse(environment_file.is_symlink())
            self.assertEqual(environment_file.stat().st_nlink, 1)
            self.assertIn("systemctl --user daemon-reload", log_path.read_text())

    def test_start_rejects_symlink_service_environment_before_systemd(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            env, environment_dir, environment_file, log_path = (
                self._desktop_security_test_environment(tmp_path)
            )
            environment_dir.mkdir(mode=0o700)
            target = tmp_path / "must-not-change"
            target.write_text("private-target-value\n")
            target.chmod(0o600)
            environment_file.symlink_to(target)

            result = self._assert_start_rejected_before_systemd(env, log_path)

            self.assertEqual(target.read_text(), "private-target-value\n")
            self.assertNotIn("private-target-value", result.stderr)

    def test_start_rejects_permissive_service_environment_before_systemd(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env, environment_dir, environment_file, log_path = (
                self._desktop_security_test_environment(Path(tmp))
            )
            environment_dir.mkdir(mode=0o700)
            environment_file.touch(mode=0o644)

            self._assert_start_rejected_before_systemd(env, log_path)

    def test_start_rejects_non_regular_service_environment_before_systemd(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env, environment_dir, environment_file, log_path = (
                self._desktop_security_test_environment(Path(tmp))
            )
            environment_dir.mkdir(mode=0o700)
            os.mkfifo(environment_file, mode=0o600)

            self._assert_start_rejected_before_systemd(env, log_path)

    def test_start_rejects_hard_linked_service_environment_before_systemd(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            env, environment_dir, environment_file, log_path = (
                self._desktop_security_test_environment(tmp_path)
            )
            environment_dir.mkdir(mode=0o700)
            source = tmp_path / "linked-environment"
            source.touch(mode=0o600)
            os.link(source, environment_file)

            self._assert_start_rejected_before_systemd(env, log_path)

    def test_start_rejects_permissive_service_environment_directory_before_systemd(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env, environment_dir, environment_file, log_path = (
                self._desktop_security_test_environment(Path(tmp))
            )
            environment_dir.mkdir(mode=0o755)
            environment_file.touch(mode=0o600)

            self._assert_start_rejected_before_systemd(env, log_path)

    def test_start_rejects_attacker_writable_configuration_root_before_systemd(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env, environment_dir, environment_file, log_path = (
                self._desktop_security_test_environment(Path(tmp))
            )
            environment_dir.mkdir(mode=0o700)
            environment_file.touch(mode=0o600)
            environment_dir.parent.chmod(0o777)

            self._assert_start_rejected_before_systemd(env, log_path)

    def test_start_rejects_symlink_service_environment_directory_before_systemd(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            env, environment_dir, _, log_path = self._desktop_security_test_environment(
                tmp_path
            )
            target = tmp_path / "environment-target"
            target.mkdir(mode=0o700)
            environment_dir.symlink_to(target, target_is_directory=True)

            self._assert_start_rejected_before_systemd(env, log_path)

    def test_daemon_wrapper_treats_command_syntax_as_literal_data(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            env, environment_dir, environment_file, _ = (
                self._desktop_security_test_environment(tmp_path)
            )
            environment_dir.mkdir(mode=0o700)
            marker = tmp_path / "command-syntax-executed"
            environment_file.write_text(
                f"TRANSLATOR_HEADPHONE_SINK=$(touch {marker})\n"
            )
            environment_file.chmod(0o600)
            daemon = self._install_test_daemon(tmp_path, source="/usr/bin/env")
            env["LD_LIBRARY_PATH"] = "synthetic-loader-path"
            env["PYTHONPATH"] = "synthetic-python-path"

            result = self._run_daemon_wrapper(env, environment_dir.parent, daemon)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(marker.exists())
            self.assertIn(f"TRANSLATOR_HEADPHONE_SINK=$(touch {marker})", result.stdout)
            self.assertNotIn("LD_LIBRARY_PATH=", result.stdout)
            self.assertNotIn("PYTHONPATH=", result.stdout)

    def test_daemon_wrapper_rejects_loader_environment_without_disclosure(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            env, environment_dir, environment_file, _ = (
                self._desktop_security_test_environment(tmp_path)
            )
            environment_dir.mkdir(mode=0o700)
            marker = "private-loader-value-5e6f"
            environment_file.write_text(f"LD_PRELOAD={marker}\n")
            environment_file.chmod(0o600)
            daemon = self._install_test_daemon(tmp_path)

            result = self._run_daemon_wrapper(env, environment_dir.parent, daemon)

            self.assertEqual(result.returncode, 78)
            self.assertIn("unsupported service environment key", result.stderr)
            self.assertNotIn(marker, result.stdout + result.stderr)

    def test_daemon_wrapper_rejects_unsafe_file_identities(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            env, environment_dir, environment_file, _ = (
                self._desktop_security_test_environment(tmp_path)
            )
            environment_dir.mkdir(mode=0o700)
            daemon = self._install_test_daemon(tmp_path)
            target = tmp_path / "environment-target"
            target.touch(mode=0o600)
            environment_file.symlink_to(target)

            symlink_result = self._run_daemon_wrapper(
                env, environment_dir.parent, daemon
            )

            self.assertEqual(symlink_result.returncode, 78)
            environment_file.unlink()
            environment_file.touch(mode=0o644)

            mode_result = self._run_daemon_wrapper(env, environment_dir.parent, daemon)

            self.assertEqual(mode_result.returncode, 78)
            environment_file.unlink()
            source = tmp_path / "linked-environment"
            source.touch(mode=0o600)
            os.link(source, environment_file)

            link_result = self._run_daemon_wrapper(env, environment_dir.parent, daemon)

            self.assertEqual(link_result.returncode, 78)

    def test_daemon_wrapper_rejects_ambiguous_environment_syntax(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            env, environment_dir, environment_file, _ = (
                self._desktop_security_test_environment(tmp_path)
            )
            environment_dir.mkdir(mode=0o700)
            daemon = self._install_test_daemon(tmp_path)
            invalid_documents = (
                b"TRANSLATOR_HEADPHONE_SINK=one\nTRANSLATOR_HEADPHONE_SINK=two\n",
                b"export TRANSLATOR_HEADPHONE_SINK=value\n",
                b"TRANSLATOR_HEADPHONE_SINK=\xff\n",
                b"#\n" * 256,
            )

            for document in invalid_documents:
                environment_file.write_bytes(document)
                environment_file.chmod(0o600)
                result = self._run_daemon_wrapper(env, environment_dir.parent, daemon)
                self.assertEqual(result.returncode, 78)

    def test_systemd_unit_is_user_session_owner_with_bounded_restart(self) -> None:
        unit = read("systemd/translator.service")

        self.assertIn("Type=simple", unit)
        self.assertIn(
            "ExecStart=%h/.local/bin/translator _run-daemon "
            "%E %h/.local/bin/translator-daemon",
            unit,
        )
        self.assertNotIn("EnvironmentFile=", unit)
        self.assertNotIn("ExecStartPre=", unit)
        unset_environment = {
            name
            for line in unit.splitlines()
            if line.startswith("UnsetEnvironment=")
            for name in line.removeprefix("UnsetEnvironment=").split()
        }
        documented_loader_controls = {
            "GLIBC_TUNABLES",
            "LD_ASSUME_KERNEL",
            "LD_AUDIT",
            "LD_BIND_NOT",
            "LD_BIND_NOW",
            "LD_DEBUG",
            "LD_DEBUG_OUTPUT",
            "LD_DYNAMIC_WEAK",
            "LD_HWCAP_MASK",
            "LD_LIBRARY_PATH",
            "LD_ORIGIN_PATH",
            "LD_POINTER_GUARD",
            "LD_PREFER_MAP_32BIT_EXEC",
            "LD_PRELOAD",
            "LD_PROFILE",
            "LD_PROFILE_OUTPUT",
            "LD_SHOW_AUXV",
            "LD_TRACE_LOADED_OBJECTS",
            "LD_TRACE_PRELINKING",
            "LD_USE_LOAD_BIAS",
            "LD_VERBOSE",
            "LD_WARN",
        }
        self.assertLessEqual(documented_loader_controls, unset_environment)
        self.assertLessEqual(
            {
                "GCONV_PATH",
                "LOCPATH",
                "BASH_ENV",
                "ENV",
                "SHELLOPTS",
                "PYTHONHOME",
                "PYTHONPATH",
            },
            unset_environment,
        )
        self.assertNotIn("%h/Source", unit)
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

    def test_desktop_lifecycle_script_installs_unit_and_tauri_autostart_only(
        self,
    ) -> None:
        script_path = ROOT / "scripts/translator-desktop"
        self.assertTrue(
            script_path.exists(), "Task 9 desktop lifecycle script is missing"
        )
        mode = script_path.stat().st_mode
        self.assertTrue(
            mode & stat.S_IXUSR, "desktop lifecycle script must be executable"
        )
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
        self.assertIn(
            'ui_binary_source="${project_root}/target/release/translator-ui"', script
        )
        self.assertIn('ui_binary_target="${user_bin_dir}/translator-ui"', script)
        self.assertIn("install_daemon_binary_if_available", script)
        self.assertIn(
            'daemon_binary_source="${project_root}/target/release/translator-daemon"',
            script,
        )
        self.assertIn(
            'daemon_binary_target="${user_bin_dir}/translator-daemon"', script
        )
        self.assertIn("refusing to overwrite existing translator command", script)
        self.assertIn("start_unit_installing_if_needed", script)
        self.assertIn("_run-daemon", script)
        self.assertIn("print_service_summary", script)
        self.assertIn("systemctl --user disable translator.service", script)
        self.assertNotIn("sudo", script)
        self.assertNotIn("systemctl start translator.service", script)
        self.assertNotIn("systemctl enable translator.service", script)

        self.assertIn("systemd/user", script)
        self.assertIn("autostart", script)
        self.assertIn("systemd-path user-configuration", script)
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
                'printf \'systemctl %s\\n\' "$*" >>"${TRANSLATOR_TEST_LOG}"\n'
                'if [ "$1" = "--user" ] && [ "${2:-}" = "list-unit-files" ]; then exit 1; fi\n'
                'if [ "$1" = "--user" ] && [ "${2:-}" = "show" ]; then\n'
                "  printf '%s\\n' 'LoadState=loaded' 'ActiveState=active' 'SubState=running' 'MainPID=123' 'Result=success'\n"
                "fi\n"
            )
            (fake_bin / "pgrep").write_text(
                "#!/usr/bin/env bash\n"
                'printf \'pgrep %s\\n\' "$*" >>"${TRANSLATOR_TEST_LOG}"\n'
                "exit 1\n"
            )
            (fake_bin / "setsid").write_text(
                "#!/usr/bin/env bash\n"
                'printf \'setsid %s\\n\' "$*" >>"${TRANSLATOR_TEST_LOG}"\n'
                "exit 0\n"
            )
            (fake_bin / "curl").write_text(
                "#!/usr/bin/env bash\n"
                'printf \'curl %s\\n\' "$*" >>"${TRANSLATOR_TEST_LOG}"\n'
                "exit 0\n"
            )
            (fake_bin / "bun").write_text(
                "#!/usr/bin/env bash\n"
                'printf \'bun %s\\n\' "$*" >>"${TRANSLATOR_TEST_LOG}"\n'
                "exit 1\n"
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
            self.assertIn("bun run tauri build --no-bundle", calls)
            self.assertIn(
                "curl -sS -m 0.5 -o /dev/null http://127.0.0.1:47681/v1/status", calls
            )
            self.assertLess(calls.index("curl "), calls.index("setsid -f"))

    def test_task9_smoke_script_checks_user_service_without_cloud_or_live_provider(
        self,
    ) -> None:
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

    def test_tauri_backend_reports_daemon_state_without_managing_systemd_unit(
        self,
    ) -> None:
        tauri = read("apps/translator-ui/src-tauri/src/main.rs")

        self.assertIn("control_token_path", tauri)
        self.assertIn("XDG_RUNTIME_DIR", tauri)
        self.assertNotIn('Command::new("translator-daemon"', tauri)
        self.assertNotIn('Command::new("systemctl"', tauri)
        self.assertNotRegex(
            tauri, r"\bsystemctl\s+--user\s+(?:start|stop|restart|enable|disable)"
        )

    @requires_local_artifacts(
        "docs/benchmarks/task9-validation-report.json",
        "docs/benchmarks/task7-live-human-round-trip.json",
    )
    def test_task9_validation_report_carries_task7_latency_debt(self) -> None:
        report_path = ROOT / "docs/benchmarks/task9-validation-report.json"
        self.assertTrue(report_path.exists(), "Task 9 validation report is missing")
        report = json.loads(report_path.read_text())
        task7 = json.loads(
            (ROOT / "docs/benchmarks/task7-live-human-round-trip.json").read_text()
        )
        task7_debt = report["task7_debt_carried"]

        self.assertEqual(report["schema_version"], "translator.task9-validation.v1")
        self.assertEqual(
            task7_debt["canonical_evidence"],
            "docs/benchmarks/task7-live-human-round-trip.json",
        )
        self.assertEqual(
            task7_debt["local_provider_latency_classification"], task7["classification"]
        )
        self.assertEqual(
            task7_debt["task7_complete"], task7["acceptance"]["task7_complete"]
        )
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
        combined = "\n".join(
            path.read_text() for path in checked_paths if path.exists()
        )

        self.assertNotIn("OPENAI_API_KEY=", combined)
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

    @requires_local_artifacts(
        "docs/planning/translator-live-duplex-design.md",
        "docs/planning/translator-live-duplex-task-prompts.md",
        "docs/planning/translator-live-duplex-tasks.md",
    )
    def test_planning_documents_record_terminal_lifecycle_command(self) -> None:
        design = read("docs/planning/translator-live-duplex-design.md")
        prompts = read("docs/planning/translator-live-duplex-task-prompts.md")
        tasks = read("docs/planning/translator-live-duplex-tasks.md")
        combined = f"{design}\n{prompts}\n{tasks}"

        self.assertIn("scripts/translator-desktop up", combined)
        self.assertIn("scripts/translator-desktop down", combined)
        self.assertIn("scripts/translator-desktop restart", combined)
        self.assertIn("scripts/translator-desktop logs", combined)
        self.assertIn("translator up", combined)
        self.assertIn("~/.local/bin/translator", combined)
        self.assertIn("audio graph cleanup", combined.lower())


if __name__ == "__main__":
    unittest.main()

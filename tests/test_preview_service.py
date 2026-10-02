from __future__ import annotations

import configparser
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class PreviewResourceLimitTests(unittest.TestCase):
    def test_host_uid_validation_keeps_initial_user_namespace(self) -> None:
        unit = configparser.ConfigParser(strict=False, interpolation=None)
        unit.read(ROOT / "systemd/translator-preview.service")
        for key in (
            "PrivateUsers",
            "PrivateTmp",
            "ProtectSystem",
            "ProtectHome",
            "ReadWritePaths",
        ):
            with self.subTest(key=key):
                self.assertFalse(unit.has_option("Service", key))
        self.assertEqual(unit.get("Service", "NoNewPrivileges"), "true")
        self.assertEqual(unit.get("Service", "UMask"), "0077")

    def test_installed_service_bounds_models_and_reaps_the_whole_group(self) -> None:
        unit = configparser.ConfigParser(strict=False, interpolation=None)
        unit.read(ROOT / "systemd/translator-preview.service")
        for key, expected in {
            "MemoryHigh": "6G",
            "MemoryMax": "8G",
            "MemorySwapMax": "256M",
            "CPUQuota": "200%",
            "TasksMax": "256",
            "KillMode": "control-group",
            "TimeoutStopSec": "15",
        }.items():
            with self.subTest(key=key):
                self.assertEqual(unit.get("Service", key, fallback=None), expected)


class PreviewServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.bundle = self.root / "relocated"
        self.script = self.bundle / "scripts/translator-desktop"
        self.script.parent.mkdir(parents=True)
        shutil.copy2(ROOT / "scripts/translator-desktop", self.script)
        self.home = self.root / "home"
        self.config = self.root / "config"
        self.runtime = self.root / "runtime"
        for directory in (self.home, self.config, self.runtime):
            directory.mkdir(mode=0o700)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.calls = self.root / "calls"
        self.capture = self.root / "capture"
        self.unit = self.config / "systemd/user/translator-preview.service"
        self.command = self.home / ".local/bin/translator-preview"
        self.env = {
            "HOME": str(self.home),
            "XDG_CONFIG_HOME": str(self.config),
            "XDG_RUNTIME_DIR": str(self.runtime),
            "PATH": f"{self.bin}:/usr/bin:/bin",
            "PREVIEW_CALLS": str(self.calls),
            "PREVIEW_CAPTURE": str(self.capture),
        }
        self.sentinels = (
            self.config / "systemd/user/translator.service",
            self.config / "translator/environment",
        )
        for sentinel in self.sentinels:
            sentinel.parent.mkdir(parents=True, exist_ok=True)
            sentinel.write_text("canonical sentinel\n")
        preview_config = self.config / "translator-preview"
        preview_config.mkdir(mode=0o700)
        (preview_config / "environment").write_text("")
        (preview_config / "environment").chmod(0o600)
        helper = self.root / "daemon.c"
        helper.write_text(
            "#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n"
            "int main(int argc, char **argv) {\n"
            'const char *p=getenv("PREVIEW_CAPTURE");\n'
            'FILE *f=p?fopen(p,"w"):stdout; if(!f)return 1;\n'
            'for(int i=1;i<argc;i++)fprintf(f,"argv=%s\\n",argv[i]);\n'
            'const char *keys[]={"XDG_RUNTIME_DIR","XDG_STATE_HOME",'
            '"PULSE_SERVER","TRANSLATOR_DAEMON_URL","TRANSLATOR_PYTHON",'
            '"TRANSLATOR_SIDECAR_ROOT","TRANSLATOR_ASR_MODEL_ID",'
            '"TRANSLATOR_MT_MODEL_ID","TRANSLATOR_MODEL_CACHE_ROOT",'
            '"TRANSLATOR_CUDA_LIBRARY_PATH",'
            '"LD_TEST_MARKER","PYTHONPATH",NULL};\n'
            "for(int i=0;keys[i];i++){const char *v=getenv(keys[i]);\n"
            'if(v)fprintf(f,"%s=%s\\n",keys[i],v);}\n'
            "if(f!=stdout)fclose(f);return 0;}\n"
        )
        self.daemon = self.bundle / "target/release/translator-daemon"
        self.daemon.parent.mkdir(parents=True)
        subprocess.run(["cc", str(helper), "-o", str(self.daemon)], check=True)
        self.daemon.chmod(0o700)
        shutil.copy2(self.daemon, self.daemon.with_name("translator-ui"))
        python = self.bundle / "sidecar/.venv/bin/python"
        python.parent.mkdir(parents=True)
        python.symlink_to("/usr/bin/python3.12")
        (python.parent.parent / "pyvenv.cfg").write_text(
            "home = /usr/bin\ninclude-system-site-packages = false\nversion = 3.12.3\n"
        )
        phonemizer = python.parent.parent / "lib/python3.12/site-packages/piper"
        phonemizer.mkdir(parents=True)
        (phonemizer / "__init__.py").write_text("")
        self.phonemizer = phonemizer / "phonemize_espeak.py"
        self.phonemizer.write_text(
            "class EspeakPhonemizer:\n"
            "    def phonemize(self, voice, text):\n"
            "        return [['p']]\n"
        )
        models = self.bundle / "models/manifest.json"
        models.parent.mkdir()
        models.write_text("{}\n")
        (self.bundle / "model-cache").mkdir()
        unit_source = self.bundle / "systemd/translator-preview.service"
        unit_source.parent.mkdir()
        source = ROOT / "systemd/translator-preview.service"
        unit_source.write_text(source.read_text() if source.exists() else "[Unit]\n")
        self._helper(
            "systemctl",
            'printf "%s\\n" "$*" >> "$PREVIEW_CALLS"\n'
            'case "$*" in\n'
            '  *"is-active --quiet translator.service"*) '
            'test "${PREVIEW_PRODUCTION_ACTIVE:-0}" = 1; exit $?;;\n'
            '  *"is-active --quiet translator-preview.service"*) exit 1;;\n'
            '  *list-unit-files*) printf "translator-preview.service disabled\\n";;\n'
            '  *daemon-reload*) test "${PREVIEW_FAIL:-}" != reload || exit 1;;\n'
            "esac\nexit 0\n",
        )
        self._helper("curl", 'test "${PREVIEW_PRODUCTION_API:-0}" = 1\n')
        self._helper("pactl", "exit 0\n")
        self._helper("pgrep", "exit 1\n")
        self._helper(
            "ln",
            '/usr/bin/ln "$@"\ntest "${PREVIEW_FAIL:-}" != link\n',
        )
        self._helper(
            "install",
            '/usr/bin/install "$@"\n'
            'case "$*" in *translator-preview.service*) '
            'test "${PREVIEW_FAIL:-}" != unit;; esac\n',
        )

    def _helper(self, name: str, content: str) -> None:
        path = self.bin / name
        path.write_text("#!/usr/bin/bash\nset -eu\n" + content)
        path.chmod(0o700)

    def run_action(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(self.script), "--preview", *arguments],
            cwd="/tmp",
            env=self.env,
            text=True,
            capture_output=True,
            timeout=20,
            check=False,
        )

    def test_install_is_separate_and_does_not_enable_or_start(self) -> None:
        canonical = self.home / ".local/bin/translator"
        canonical.parent.mkdir(parents=True)
        canonical.write_text("canonical sentinel\n")
        result = self.run_action("install")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(canonical.read_text(), "canonical sentinel\n")
        self.assertEqual(self.command.resolve(), self.script)
        self.assertTrue(self.unit.is_file())
        self.assertFalse((self.config / "autostart/translator-ui.desktop").exists())
        self.assertNotIn("enable", self.calls.read_text())
        self.assertNotIn("start", self.calls.read_text())
        for sentinel in self.sentinels:
            self.assertEqual(sentinel.read_text(), "canonical sentinel\n")

    def test_relocated_wrapper_executes_frozen_profile_with_private_paths(self) -> None:
        self.env.update(LD_TEST_MARKER="discard", PYTHONPATH="/discard")
        result = self.run_action("_run-daemon", str(self.config), str(self.daemon))
        self.assertEqual(result.returncode, 0, result.stderr)
        capture = self.capture.read_text()
        self.assertIn("argv=--listen\nargv=127.0.0.1:47682\n", capture)
        self.assertIn(f"XDG_RUNTIME_DIR={self.runtime}/translator-preview\n", capture)
        self.assertIn(
            f"XDG_STATE_HOME={self.home}/.local/state/translator-preview\n", capture
        )
        self.assertIn(f"PULSE_SERVER=unix:{self.runtime}/pulse/native\n", capture)
        self.assertIn(f"TRANSLATOR_SIDECAR_ROOT={self.bundle}/sidecar\n", capture)
        self.assertIn(
            f"TRANSLATOR_PYTHON={self.bundle}/sidecar/.venv/bin/python\n", capture
        )
        self.assertIn(
            "TRANSLATOR_ASR_MODEL_ID=faster-whisper-large-v3-turbo\n", capture
        )
        self.assertIn("TRANSLATOR_MT_MODEL_ID=hy-mt2-1.8b-gguf-q4-k-m\n", capture)
        self.assertIn(
            f"TRANSLATOR_MODEL_CACHE_ROOT={self.bundle}/model-cache\n", capture
        )
        self.assertNotIn("LD_TEST_MARKER", capture)
        self.assertNotIn("PYTHONPATH", capture)
        self.assertNotIn(str(ROOT), capture)

    def test_explicit_mt_selection_passes_existing_guard(self) -> None:
        directory = self.config / "translator-preview"
        directory.mkdir(mode=0o700, exist_ok=True)
        environment = directory / "environment"
        environment.write_text(
            "TRANSLATOR_MT_MODEL_ID=nllb-200-distilled-600m-ct2-int8\n"
        )
        environment.chmod(0o600)
        result = self.run_action("_run-daemon", str(self.config), str(self.daemon))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "TRANSLATOR_MT_MODEL_ID=nllb-200-distilled-600m-ct2-int8",
            self.capture.read_text(),
        )

    def test_bundled_cuda_default_and_explicit_override(self) -> None:
        for component in ("cudnn", "cuda_nvrtc"):
            (self.bundle / "cuda/nvidia" / component / "lib").mkdir(parents=True)
        result = self.run_action("_run-daemon", str(self.config), str(self.daemon))
        self.assertEqual(result.returncode, 0, result.stderr)
        expected = ":".join(
            str(self.bundle / "cuda/nvidia" / component / "lib")
            for component in ("cudnn", "cuda_nvrtc")
        )
        self.assertIn(
            f"TRANSLATOR_CUDA_LIBRARY_PATH={expected}\n", self.capture.read_text()
        )
        (self.config / "translator-preview/environment").write_text(
            "TRANSLATOR_CUDA_LIBRARY_PATH=/private/cuda\n"
        )
        result = self.run_action("_run-daemon", str(self.config), str(self.daemon))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "TRANSLATOR_CUDA_LIBRARY_PATH=/private/cuda\n", self.capture.read_text()
        )

    def test_phonemizer_failure_blocks_install_before_owned_writes(self) -> None:
        self.phonemizer.write_text("import os\nos._exit(1)\n")
        result = self.run_action("install")
        self.assertEqual(result.returncode, 78, result.stderr)
        self.assertFalse(self.command.exists())
        self.assertFalse(self.unit.exists())
        self.assertNotIn(
            "daemon-reload", self.calls.read_text() if self.calls.exists() else ""
        )
        for sentinel in self.sentinels:
            self.assertEqual(sentinel.read_text(), "canonical sentinel\n")

    def test_start_refuses_active_production_without_stopping_it(self) -> None:
        self.env["PREVIEW_PRODUCTION_ACTIVE"] = "1"
        result = self.run_action("start")
        self.assertEqual(result.returncode, 73, result.stderr)
        calls = self.calls.read_text()
        self.assertNotIn("stop", calls)
        self.assertNotIn("start", calls)

    def test_stop_targets_preview_unit_and_exact_payload_cleanup(self) -> None:
        result = self.run_action("stop")
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.calls.read_text()
        self.assertIn("stop translator-preview.service", calls)
        self.assertNotIn("stop translator.service", calls)
        capture = self.capture.read_text()
        self.assertIn("argv=--audio-graph-cleanup", capture)
        self.assertIn(f"XDG_RUNTIME_DIR={self.runtime}/translator-preview", capture)

    def test_start_refuses_live_canonical_api(self) -> None:
        self.env["PREVIEW_PRODUCTION_API"] = "1"
        result = self.run_action("start")
        self.assertEqual(result.returncode, 73, result.stderr)
        self.assertNotIn("stop", self.calls.read_text())
        self.assertNotIn("start", self.calls.read_text())

    def test_unknown_environment_key_rejects_before_exec(self) -> None:
        directory = self.config / "translator-preview"
        directory.mkdir(mode=0o700, exist_ok=True)
        environment = directory / "environment"
        environment.write_text("UNSUPPORTED_PREVIEW_KEY=private-marker\n")
        environment.chmod(0o600)
        result = self.run_action("_run-daemon", str(self.config), str(self.daemon))
        self.assertEqual(result.returncode, 78, result.stderr)
        self.assertFalse(self.capture.exists())
        self.assertNotIn("private-marker", result.stdout + result.stderr)

    def test_unsafe_payload_rejects_before_exec(self) -> None:
        self.daemon.chmod(0o777)
        result = self.run_action("_run-daemon", str(self.config), str(self.daemon))
        self.assertEqual(result.returncode, 78, result.stderr)
        self.assertFalse(self.capture.exists())

    def test_unsafe_payload_rejects_before_installation(self) -> None:
        self.daemon.chmod(0o777)
        result = self.run_action("install")
        self.assertEqual(result.returncode, 78, result.stderr)
        self.assertFalse(self.command.exists())
        self.assertFalse(self.unit.exists())

    def test_start_requires_explicit_install_without_mutation(self) -> None:
        result = self.run_action("start")
        self.assertEqual(result.returncode, 78, result.stderr)
        self.assertFalse(self.command.exists())
        self.assertFalse(self.unit.exists())

    def test_actual_unit_entry_refuses_live_production(self) -> None:
        self.env["PREVIEW_PRODUCTION_ACTIVE"] = "1"
        result = self.run_action("_run-bundle-daemon", str(self.config))
        self.assertEqual(result.returncode, 73, result.stderr)
        self.assertFalse(self.capture.exists())

    def test_install_rejects_missing_binary_before_owned_writes(self) -> None:
        self.daemon.unlink()
        result = self.run_action("install")
        self.assertEqual(result.returncode, 78, result.stderr)
        self.assertFalse(self.command.exists())
        self.assertFalse(self.unit.exists())

    def test_existing_preview_target_is_preserved(self) -> None:
        self.command.parent.mkdir(parents=True)
        self.command.write_text("foreign preview sentinel\n")
        result = self.run_action("install")
        self.assertEqual(result.returncode, 73, result.stderr)
        self.assertEqual(self.command.read_text(), "foreign preview sentinel\n")
        self.assertFalse(self.unit.exists())

    def test_install_faults_roll_back_only_new_activation(self) -> None:
        for fault in ("link", "unit", "reload"):
            with self.subTest(fault=fault):
                self.env["PREVIEW_FAIL"] = fault
                result = self.run_action("install")
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertFalse(self.command.is_symlink())
                self.assertFalse(self.unit.exists())
                self.assertTrue(self.daemon.is_file())
                self.assertTrue((self.bundle / "model-cache").is_dir())
                for sentinel in self.sentinels:
                    self.assertEqual(sentinel.read_text(), "canonical sentinel\n")


if __name__ == "__main__":
    unittest.main()

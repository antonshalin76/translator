from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class PortableBundleTests(unittest.TestCase):
    def _bundle(self, root: Path) -> Path:
        script = root / "scripts" / "translator-desktop"
        script.parent.mkdir(parents=True)
        shutil.copy2(ROOT / "scripts/translator-desktop", script)

        sidecar = root / "sidecar"
        runtime = sidecar / ".venv/bin/python"
        runtime.parent.mkdir(parents=True)
        shutil.copy2(sys.executable, runtime)
        package = sidecar / "translator_sidecar"
        package.mkdir()
        (package / "__main__.py").write_text("print('bundle-sidecar:' + __file__)\n")
        return script

    def _launcher_environment(self, root: Path) -> tuple[dict[str, str], Path, Path]:
        home = root / "home"
        daemon = home / ".local/bin/translator-daemon"
        daemon.parent.mkdir(parents=True)
        shutil.copy2("/usr/bin/env", daemon)

        config_root = root / "config"
        config_root.mkdir(mode=0o700)
        environment_dir = config_root / "translator"
        environment_dir.mkdir(mode=0o700)
        environment_file = environment_dir / "environment"
        environment_file.write_text("")
        environment_file.chmod(0o600)
        env = {
            "HOME": str(home),
            "PATH": "/usr/bin:/bin",
        }
        return env, config_root, daemon

    def _run_wrapper(
        self, script: Path, env: dict[str, str], config_root: Path, daemon: Path
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(script), "_run-daemon", str(config_root), str(daemon)],
            cwd="/tmp",
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_moved_bundle_uses_its_own_runtime_without_checkout_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            staged = root / "staged"
            self._bundle(staged)
            installed = root / "installed"
            staged.rename(installed)
            script = installed / "scripts/translator-desktop"
            env, config_root, daemon = self._launcher_environment(root)

            result = self._run_wrapper(script, env, config_root, daemon)

            self.assertEqual(result.returncode, 0, result.stderr)
            values = dict(line.split("=", 1) for line in result.stdout.splitlines())
            sidecar = installed / "sidecar"
            self.assertEqual(values.get("TRANSLATOR_SIDECAR_ROOT"), str(sidecar))
            self.assertEqual(
                values.get("TRANSLATOR_PYTHON"), str(sidecar / ".venv/bin/python")
            )
            for path in (str(ROOT), str(staged), str(Path.home()) + "/"):
                self.assertNotIn(path, values["TRANSLATOR_SIDECAR_ROOT"])
                self.assertNotIn(path, values["TRANSLATOR_PYTHON"])
                self.assertNotIn(path, script.read_text())

            sidecar_result = subprocess.run(
                [values["TRANSLATOR_PYTHON"], "-m", "translator_sidecar"],
                cwd=values["TRANSLATOR_SIDECAR_ROOT"],
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(sidecar_result.returncode, 0, sidecar_result.stderr)
            self.assertIn(str(sidecar), sidecar_result.stdout)

    def test_private_service_environment_overrides_bundle_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = self._bundle(root / "bundle")
            env, config_root, daemon = self._launcher_environment(root)
            (config_root / "translator/environment").write_text(
                "TRANSLATOR_SIDECAR_ROOT=/private/sidecar\n"
                "TRANSLATOR_PYTHON=/private/python\n"
            )

            result = self._run_wrapper(script, env, config_root, daemon)

            self.assertEqual(result.returncode, 0, result.stderr)
            values = dict(line.split("=", 1) for line in result.stdout.splitlines())
            self.assertEqual(values["TRANSLATOR_SIDECAR_ROOT"], "/private/sidecar")
            self.assertEqual(values["TRANSLATOR_PYTHON"], "/private/python")

            (config_root / "translator/environment").write_text(
                "TRANSLATOR_SIDECAR_ROOT=/private/sidecar\n"
            )
            result = self._run_wrapper(script, env, config_root, daemon)
            self.assertEqual(result.returncode, 0, result.stderr)
            values = dict(line.split("=", 1) for line in result.stdout.splitlines())
            self.assertEqual(values["TRANSLATOR_SIDECAR_ROOT"], "/private/sidecar")
            self.assertEqual(
                values["TRANSLATOR_PYTHON"], "/private/sidecar/.venv/bin/python"
            )

    def test_inherited_runtime_override_remains_supported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = self._bundle(root / "bundle")
            env, config_root, daemon = self._launcher_environment(root)
            env["TRANSLATOR_SIDECAR_ROOT"] = "/inherited/sidecar"
            env["TRANSLATOR_PYTHON"] = "/inherited/python"

            result = self._run_wrapper(script, env, config_root, daemon)

            self.assertEqual(result.returncode, 0, result.stderr)
            values = dict(line.split("=", 1) for line in result.stdout.splitlines())
            self.assertEqual(values["TRANSLATOR_SIDECAR_ROOT"], "/inherited/sidecar")
            self.assertEqual(values["TRANSLATOR_PYTHON"], "/inherited/python")


if __name__ == "__main__":
    unittest.main()

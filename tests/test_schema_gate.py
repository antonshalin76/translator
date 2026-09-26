from __future__ import annotations

import os
import shutil
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _write_executable(path: Path, body: str) -> None:
    path.write_text("#!/usr/bin/env bash\nset -eu\n" + body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


class SchemaGateTests(unittest.TestCase):
    def test_schema_gate_ignores_false_clean_cmp_and_non_gnu_diff_shadows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary = Path(temporary_directory)
            candidate = temporary / "candidate"
            script = candidate / "scripts" / "translator-schema-check"
            script.parent.mkdir(parents=True)
            shutil.copy2(ROOT / "scripts/translator-schema-check", script)

            proto_target = candidate / "proto/translator/provider/v1/provider.proto"
            proto_target.parent.mkdir(parents=True)
            shutil.copy2(
                ROOT / "proto/translator/provider/v1/provider.proto",
                proto_target,
            )

            generated_target = (
                candidate
                / "sidecar/translator_sidecar/generated/translator/provider/v1"
            )
            generated_target.parent.mkdir(parents=True)
            shutil.copytree(
                ROOT / "sidecar/translator_sidecar/generated/translator/provider/v1",
                generated_target,
                dirs_exist_ok=True,
            )

            python_wrapper = candidate / "sidecar/.venv/bin/python"
            python_wrapper.parent.mkdir(parents=True)
            _write_executable(
                python_wrapper,
                f'exec "{ROOT / "sidecar/.venv/bin/python"}" "$@"\n',
            )

            fake_bin = temporary / "bin"
            fake_bin.mkdir()
            fake_diff_marker = temporary / "shadow-diff-was-called"
            fake_cmp_marker = temporary / "shadow-cmp-was-called"
            _write_executable(
                fake_bin / "diff",
                ': >"$TRANSLATOR_FAKE_DIFF_MARKER"\nexit 99\n',
            )
            _write_executable(
                fake_bin / "cmp",
                ': >"$TRANSLATOR_FAKE_CMP_MARKER"\nexit 0\n',
            )
            env = os.environ.copy()
            env.update(
                {
                    "PATH": f"{fake_bin}:{env['PATH']}",
                    "TRANSLATOR_FAKE_DIFF_MARKER": str(fake_diff_marker),
                    "TRANSLATOR_FAKE_CMP_MARKER": str(fake_cmp_marker),
                }
            )

            clean = subprocess.run(
                [str(script)],
                cwd="/tmp",
                env=env,
                text=True,
                capture_output=True,
                timeout=15,
                check=False,
            )
            self.assertEqual(clean.returncode, 0, clean.stdout + clean.stderr)
            self.assertFalse(fake_diff_marker.exists())
            self.assertFalse(fake_cmp_marker.exists())

            committed = generated_target / "provider_pb2.py"
            committed.write_text(
                committed.read_text(encoding="utf-8") + "\n# intentional drift\n",
                encoding="utf-8",
            )
            drift = subprocess.run(
                [str(script)],
                cwd="/tmp",
                env=env,
                text=True,
                capture_output=True,
                timeout=15,
                check=False,
            )

            self.assertEqual(drift.returncode, 1, drift.stdout + drift.stderr)
            self.assertIn("generated protobuf bindings are stale", drift.stderr)
            self.assertIn("committed/provider_pb2.py", drift.stdout)
            self.assertFalse(fake_diff_marker.exists())
            self.assertFalse(fake_cmp_marker.exists())


if __name__ == "__main__":
    unittest.main()

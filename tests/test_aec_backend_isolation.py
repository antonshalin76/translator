"""Executable isolation challenge for the AEC stage runner."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECK = ROOT / "scripts" / "translator-aec-backend-check"


class AecBackendIsolationTests(unittest.TestCase):
    def test_probe_runs_in_private_namespace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            marker = root / "host-only"
            marker.write_text("private")
            listener = socket.socket(socket.AF_UNIX)
            try:
                socket_path = root / "host-audio.sock"
                listener.bind(str(socket_path))
                listener.listen(1)
                listener.settimeout(0.1)
                nonce = os.urandom(16).hex()
                probe = root / "probe.py"
                probe.write_text(
                    "import json, os, socket\n"
                    f"nonce = {nonce!r}\n"
                    f"marker = {str(marker)!r}\n"
                    f"socket_path = {str(socket_path)!r}\n"
                    "s = socket.socket(socket.AF_UNIX)\n"
                    "facts = {'nonce': nonce,\n"
                    " 'host_file_visible': os.path.exists(marker),\n"
                    " 'host_audio_visible': os.path.exists('/dev/snd'),\n"
                    " 'host_runtime_visible': os.path.exists('/run/user'),\n"
                    " 'host_socket_connected': s.connect_ex(socket_path) == 0,\n"
                    " 'net_namespace': os.readlink('/proc/self/ns/net')}\n"
                    "print(json.dumps(facts))\n"
                )
                result = subprocess.run(
                    [str(CHECK), "--isolated", "--probe", str(probe)],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    timeout=15,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                facts = json.loads(result.stdout)
                self.assertEqual(facts["nonce"], nonce)
                for field in (
                    "host_file_visible",
                    "host_audio_visible",
                    "host_runtime_visible",
                    "host_socket_connected",
                ):
                    self.assertFalse(facts[field], field)
                self.assertNotEqual(
                    facts["net_namespace"], os.readlink("/proc/self/ns/net")
                )
                with self.assertRaises(socket.timeout):
                    listener.accept()
            finally:
                listener.close()

    def test_inherited_descriptor_and_default_socket_are_rejected(self) -> None:
        with open(os.devnull, "rb") as inherited:
            result = subprocess.run(
                [str(CHECK), "--isolated", "--preflight-only"],
                cwd=ROOT,
                env={**os.environ, "PIPEWIRE_REMOTE": "pipewire-0"},
                pass_fds=(inherited.fileno(),),
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unsafe inherited", result.stderr.lower())


if __name__ == "__main__":
    unittest.main()

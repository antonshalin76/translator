"""Isolation contract for the native AEC backend gate."""

from __future__ import annotations

import json
import random
import re
import runpy
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECK = ROOT / "scripts" / "translator-aec-backend-check"


def _session_contract() -> dict[str, object]:
    outer = runpy.run_path(str(CHECK), run_name="aec_runner_contract")
    session = {"__name__": "aec_runner_contract"}
    exec(outer["SESSION"], session)
    return session


class AecBackendCheckTests(unittest.TestCase):
    def test_preflight_reports_private_namespace(self) -> None:
        result = subprocess.run(
            [str(CHECK), "--isolated", "--preflight-only"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        facts = json.loads(result.stdout)
        self.assertTrue(facts["isolated"])
        self.assertFalse(facts["host_audio_access"])
        self.assertFalse(facts["host_runtime_access"])
        self.assertFalse(facts["network_access"])

    def test_first_buffer_failure_survives_long_untrusted_stderr(self) -> None:
        namespace = _session_contract()
        noise = b"START_WAIT /private/raw-pcm\n" * 80
        payload = noise + (
            b"AEC_BUFFER_PRECONDITION site=311 reasons=5,5,0 "
            b"state=4 started=1 callback=7 position=480\n"
            b"AEC_BUFFER_META raw=0,0,2,1,0,0 "
            b"ref=0,0,2,1,0,0\n"
            b"AEC_BUFFER_PRECONDITION site=999 reasons=1,1,1 "
            b"state=9 started=1 callback=8 position=960\n"
            b"AEC_BUFFER_META raw=1920,4,0,1,0,77 "
            b"ref=1920,4,0,1,0,77\n"
        )
        with tempfile.TemporaryFile() as diagnostic_file:
            diagnostic_file.write(payload)
            diagnostic_file.flush()
            report = namespace["read_native_diagnostics"](
                diagnostic_file, label="backend", session="0123456789abcdef"
            )
        self.assertEqual(report["status"], "OBSERVED")
        self.assertEqual(report["session"], "0123456789abcdef")
        self.assertEqual(report["buffer_reason"], [5, 5, 0])
        self.assertEqual(report["site"], 311)
        self.assertEqual(report["callback"], 7)
        self.assertEqual(report["position"], 480)
        self.assertEqual(report["state"], 4)
        self.assertEqual(report["raw_header"], [1, 0, 0])
        self.assertEqual(report["reference_header"], [1, 0, 0])
        self.assertEqual(report["raw_chunk"], [0, 0, 2])
        self.assertEqual(report["reference_chunk"], [0, 0, 2])
        self.assertNotIn("/private", json.dumps(report))

    def test_first_gap_snapshot_contains_only_numeric_provenance(self) -> None:
        namespace = _session_contract()
        with tempfile.TemporaryFile() as diagnostic_file:
            diagnostic_file.write(
                b"START_WAIT /private/pcm\n"
                * 80
                + b"AEC_GAP kind=2 expected_pos=0 observed_pos=0 "
                b"expected_clock=0 observed_clock=0 expected_xrun=2 observed_xrun=3 "
                b"expected_seq=9,9 "
                b"observed_seq=9,10\n"
            )
            diagnostic_file.flush()
            report = namespace["read_native_diagnostics"](
                diagnostic_file, label="backend", session="0123456789abcdef"
            )
        self.assertEqual(report["status"], "OBSERVED")
        self.assertEqual(report["gap_kind"], 2)
        self.assertEqual(report["expected_xrun"], 2)
        self.assertEqual(report["observed_xrun"], 3)
        self.assertEqual(report["expected_sequences"], [9, 9])
        self.assertEqual(report["observed_sequences"], [9, 10])
        self.assertNotIn("/private", json.dumps(report))

    def test_fixture_first_failure_is_numeric_and_private(self) -> None:
        namespace = _session_contract()
        with tempfile.TemporaryFile() as diagnostic_file:
            diagnostic_file.write(
                b"START_WAIT /private/pcm\n"
                b"AEC_FIXTURE callbacks=2256 state=3 duration=480 clock=15 "
                b"first=23390000000 last=23404444978 xrun=0 activated=1 "
                b"valid=1 failed=1 reason=3 expected_pos=23404444498 "
                b"observed_pos=23404444978 expected_clock=15 observed_clock=15 "
                b"raw_ready=1 ref_ready=1 raw_buffer_reason=0 "
                b"ref_buffer_reason=0\n"
            )
            diagnostic_file.flush()
            report = namespace["read_fixture_diagnostics"](
                diagnostic_file, session="0123456789abcdef"
            )
        self.assertEqual(report["status"], "OBSERVED")
        self.assertEqual(report["reason"], 3)
        self.assertEqual(report["expected_pos"], 23404444498)
        self.assertEqual(report["observed_pos"], 23404444978)
        self.assertEqual(report["expected_clock"], 15)
        self.assertEqual(report["observed_clock"], 15)
        self.assertNotIn("/private", json.dumps(report))

    def test_truncated_first_buffer_diagnostic_is_unobserved(self) -> None:
        namespace = _session_contract()
        with tempfile.TemporaryFile() as diagnostic_file:
            diagnostic_file.write(
                b"AEC_BUFFER_PRECONDITION site=311 reasons=5,5,0 "
                b"state=4 started=1 callback=7 position=480\n"
                b"AEC_BUFFER_META raw=0,0,2,1,0,0 ref=0,0,2"
            )
            diagnostic_file.flush()
            report = namespace["read_native_diagnostics"](
                diagnostic_file, label="backend", session="0123456789abcdef"
            )
        self.assertEqual(report["status"], "UNOBSERVED")

    def test_oversized_native_diagnostic_is_unobserved(self) -> None:
        namespace = _session_contract()
        with tempfile.TemporaryFile() as diagnostic_file:
            diagnostic_file.write(b"untrusted /private/pcm\n" * 4000)
            diagnostic_file.write(
                b"AEC_BUFFER_PRECONDITION site=311 reasons=5,5,0 "
                b"state=4 started=1 callback=7 position=480\n"
                b"AEC_BUFFER_META raw=0,0,2,1,0,0 ref=0,0,2,1,0,0\n"
            )
            diagnostic_file.flush()
            report = namespace["read_native_diagnostics"](
                diagnostic_file, label="backend", session="0123456789abcdef"
            )
        self.assertEqual(report["status"], "UNOBSERVED")

    def test_private_meta_gap_reports_structured_failure(self) -> None:
        result = subprocess.run(
            [
                str(CHECK),
                "--isolated",
                "--trial-mode",
                "meta-gap",
                "--trial-frames",
                "100",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("FATAL reason=4 processed=0", result.stderr)
        markers = [
            line.removeprefix("AEC_NATIVE_DIAGNOSTICS=")
            for line in result.stderr.splitlines()
            if line.startswith("AEC_NATIVE_DIAGNOSTICS=")
        ]
        self.assertEqual(len(markers), 1, result.stderr)
        report = json.loads(markers[0])
        self.assertEqual(report["status"], "OBSERVED")
        self.assertEqual(report["fatal_reason"], 4)
        self.assertEqual(report["buffer_reason"], [7, 7, 0])
        self.assertIsNotNone(re.fullmatch(r"[0-9a-f]{16}", report["session"]))
        self.assertGreater(report["callback"], 0)
        self.assertGreater(report["site"], 0)
        self.assertNotIn("/private", markers[0])
        self.assertNotIn("pcm", markers[0].lower())
        self.assertIn("cleanup_reaped=True", result.stderr)

    def test_cancel_lifecycle_accepts_ordered_armed_before_frames(self) -> None:
        result = subprocess.run(
            [
                str(CHECK),
                "--isolated",
                "--lifecycle-only",
                "--lifecycle-case",
                "cancel",
                "--lifecycle-cycles",
                "1",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        self.assertEqual(result.returncode, 4, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["completed_cycles"], 1)
        event = report["events"][0]
        self.assertEqual(event["case"], "cancel")
        self.assertIsNone(event["failure"])
        self.assertTrue(event["scope_absent"])
        self.assertEqual(event["fd_delta"], 0)
        self.assertEqual(event["thread_delta"], 0)

    def test_seq_skew_verdict_uses_structured_native_failure(self) -> None:
        runner = runpy.run_path(str(CHECK), run_name="aec_runner_contract")
        diagnostic = {
            "status": "OBSERVED",
            "source": "backend",
            "gap_kind": 2,
            "expected_sequences": [81, 81],
            "observed_sequences": [81, 82],
            "fatal_reason": 3,
            "fatal_processed": 0,
        }
        stderr = (
            "AEC_NATIVE_DIAGNOSTICS=" + json.dumps(diagnostic) + "\n"
            "AEC isolated session NOT_DONE: AEC helper startup FATAL reason=3 "
            "processed=0; cleanup_reaped=True\n"
        )
        verdict = runner["seq_skew_rejected_and_reaped"]
        self.assertTrue(verdict(2, stderr))
        self.assertFalse(verdict(0, stderr))
        self.assertFalse(
            verdict(2, stderr.replace("cleanup_reaped=True", "cleanup_reaped=False"))
        )
        self.assertFalse(verdict(2, stderr.replace("[81, 82]", "[81, 81]")))

    def test_diagnostic_lag_finds_known_delays_without_pcm_output(self) -> None:
        namespace = _session_contract()
        random_source = random.Random(19)
        raw = [random_source.uniform(-0.2, 0.2) for _ in range(48000)]
        for lag in (0, 371, 480, 1200, -200):
            with self.subTest(lag=lag):
                if lag >= 0:
                    clean = [0.0] * lag + raw[: len(raw) - lag]
                else:
                    clean = raw[-lag:] + [0.0] * -lag
                diagnostic = namespace["diagnose_window_lag"](raw, clean)
                self.assertEqual(diagnostic["lag_samples"], lag)
                self.assertGreaterEqual(diagnostic["correlation"], 0.99)
                self.assertNotIn("pcm", diagnostic)


if __name__ == "__main__":
    unittest.main()

"""Fail-closed parser contract for private AEC callback histories."""

from __future__ import annotations

import json
import runpy
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECK = ROOT / "scripts" / "translator-aec-backend-check"


def parser():
    outer = runpy.run_path(str(CHECK), run_name="aec_history_contract")
    namespace = {"__name__": "aec_history_contract"}
    exec(outer["SESSION"], namespace)
    return namespace


def receipt(namespace, label="backend", session=12, total=34, count=16):
    fields = namespace["CALLBACK_TRACE_FIELDS"]
    header = (
        f"AEC_CALLBACK_HISTORY session={session} component={label} "
        f"clock=CLOCK_MONOTONIC time_ns=123 total={total} "
        f"count={count} frozen=1\n"
    )
    lines = []
    for index in range(count):
        row = {field: 0 for field in fields}
        row.update(
            {
                "index": index,
                "callback": total - count + index + 1,
                "enter": 100000 + index * 100,
                "exit": 100010 + index * 100,
                "entry": index * 480,
                "admission": index * 480,
                "seq_position": index * 480,
                "publication": index * 480,
                "next_source": index * 480,
                "next": (index + 1) * 480,
                "cpu": 5,
                "dsp_begin": 100002 + index * 100,
                "dsp_end": 100008 + index * 100,
                "duration": 480,
                "clock": 3,
                "rate_num": 1,
                "rate_denom": 48000,
                "stage": 4,
                "valid": 7,
            }
        )
        lines.append(
            f"AEC_CALLBACK_TRACE session={session} component={label} "
            + " ".join(f"{field}={row[field]}" for field in fields)
            + "\n"
        )
    return (header + "".join(lines)).encode("ascii")


class CallbackHistoryParserTests(unittest.TestCase):
    def test_stage_preserves_bounded_full_history_on_failure(self):
        stage = runpy.run_path(
            str(ROOT / "scripts/translator-aec-backend-stage"),
            run_name="aec_history_stage_contract",
        )
        history = {
            "status": "OBSERVED",
            "source": "backend",
            "rows": [
                [(1 << 64) - 1] * len(parser()["CALLBACK_TRACE_FIELDS"])
                for _ in range(16)
            ],
        }
        marker = "AEC_CALLBACK_DIAGNOSTICS_BACKEND=" + json.dumps(
            history, separators=(",", ":")
        )
        self.assertGreater(len(marker), 4096)
        self.assertLess(len(marker), 16384)
        accepted, detail = stage["classify"](
            "speech-wrong-reference",
            2,
            "",
            marker
            + "\nAEC isolated session NOT_DONE: graph gap; cleanup_reaped=True\n",
            {},
        )
        self.assertFalse(accepted)
        self.assertEqual(detail["diagnostics"]["callback_backend"], history)

    def test_real_c_wire_output_is_accepted(self):
        namespace = parser()
        flags = subprocess.run(
            ["pkg-config", "--cflags", "libpipewire-0.3"],
            check=True,
            capture_output=True,
            text=True,
        )
        with tempfile.TemporaryDirectory(prefix="aec-history-wire-") as temporary:
            binary = Path(temporary) / "wire"
            source = (
                ROOT / "crates/translator-aec-backend/tests/callback_history_wire.c"
            )
            subprocess.run(
                [
                    "cc",
                    "-std=c11",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    *shlex.split(flags.stdout),
                    str(source),
                    "-o",
                    str(binary),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            output = subprocess.run([str(binary)], check=True, capture_output=True)
        observed = namespace["read_callback_history_bytes"](
            output.stderr, "backend", "000000000000000c"
        )
        self.assertEqual(observed["status"], "OBSERVED")
        self.assertEqual(observed["total"], 18)
        self.assertEqual(len(observed["rows"]), 16)
        self.assertTrue(observed["frozen"])
        fields = observed["fields"]
        terminal = dict(zip(fields, observed["rows"][-1], strict=True))
        self.assertEqual(terminal["stage"], 4)
        self.assertEqual(terminal["reason"], 4)
        self.assertEqual(terminal["publication"], 18 * 480)
        self.assertEqual(terminal["next_source"], 19 * 480)
        self.assertEqual(terminal["publication_clock"], 3)
        self.assertEqual(terminal["next_source_clock"], 23)
        self.assertEqual(terminal["publication_xrun"], 0)
        self.assertEqual(terminal["next_source_xrun"], 19)
        self.assertEqual(terminal["publication_duration"], 480)
        self.assertEqual(terminal["next_source_duration"], 481)
        self.assertEqual(terminal["publication_rate_num"], 1)
        self.assertEqual(terminal["publication_rate_denom"], 48000)
        self.assertEqual(terminal["next_source_rate_num"], 2)
        self.assertEqual(terminal["next_source_rate_denom"], 96000)

    def test_complete_wrapped_histories_are_session_bound(self):
        namespace = parser()
        for label in ("backend", "fixture", "witness"):
            with self.subTest(label=label):
                observed = namespace["read_callback_history_bytes"](
                    receipt(namespace, label), label, "000000000000000c"
                )
                self.assertEqual(observed["status"], "OBSERVED")
                self.assertEqual(observed["total"], 34)
                self.assertEqual(len(observed["rows"]), 16)
                self.assertEqual(observed["rows"][0][1], 19)
                self.assertEqual(observed["rows"][-1][1], 34)

    def test_short_circuit_position_is_explicitly_missing(self):
        namespace = parser()
        payload = receipt(namespace).replace(
            b"admission=0 ", b"admission=18446744073709551615 ", 1
        )
        observed = namespace["read_callback_history_bytes"](
            payload, "backend", "000000000000000c"
        )
        self.assertEqual(observed["status"], "OBSERVED")
        self.assertIsNone(
            observed["rows"][0][namespace["CALLBACK_TRACE_FIELDS"].index("admission")]
        )

    def test_corrupt_or_cross_session_histories_are_unobserved(self):
        namespace = parser()
        good = receipt(namespace)
        damaged = (
            good[:-1],
            good + good,
            good.replace(b"session=12", b"session=13", 1),
            good.replace(b"component=backend", b"component=witness", 1),
            good.replace(b"count=16", b"count=15", 1),
            good.replace(b"index=1 ", b"index=0 ", 1),
            good.replace(b"callback=20 ", b"callback=21 ", 1),
            good.replace(b"enter=100000", b"enter=-1", 1),
            good.replace(b"cpu=5", b"cpu=0", 1),
            good.replace(b"cpu=5", b"cpu=99", 1),
            good.replace(b"dsp_begin=100002", b"dsp_begin=0", 1),
            good.replace(b"time_ns=123", b"time_ns=0", 1),
            good.replace(b"stage=4", b"stage=9", 1),
            good.replace(b"exit=100010", b"exit=99999", 1),
            good.replace(b"clock=3", b"clock=\xff", 1),
            good + b"x" * namespace["DIAGNOSTIC_LIMIT"],
        )
        for value in damaged:
            with self.subTest(value=value[:100]):
                observed = namespace["read_callback_history_bytes"](
                    value, "backend", "000000000000000c"
                )
                self.assertEqual(observed["status"], "UNOBSERVED")


if __name__ == "__main__":
    unittest.main()

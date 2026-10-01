"""Isolation contract for the native AEC backend gate."""

from __future__ import annotations

import ast
import json
import os
import random
import re
import runpy
import select
import signal
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
CHECK = ROOT / "scripts" / "translator-aec-backend-check"


def _session_contract() -> dict[str, object]:
    outer = runpy.run_path(str(CHECK), run_name="aec_runner_contract")
    session = {"__name__": "aec_runner_contract"}
    exec(outer["SESSION"], session)
    return session


class AecBackendCheckTests(unittest.TestCase):
    def test_pause_scope_rejects_other_invocations_before_graph(self) -> None:
        runner = runpy.run_path(str(CHECK), run_name="aec_runner_contract")
        validate = runner["validate_pause_request"]
        self.assertIsNone(validate("wrong-reference", 4500, None, None, 0, 30))
        for case in (
            ("far-only", 4500, None, None, 0, 30),
            ("wrong-reference", 100, None, None, 0, 30),
            ("wrong-reference", 4500, "helper-crash", None, 0, 30),
            ("wrong-reference", 4500, None, "passthrough", 0, 30),
            ("wrong-reference", 4500, None, None, 30, 30),
            (None, 4500, None, None, 0, 30),
            ("wrong-reference", 4500, None, None, 0, 31),
        ):
            with (
                self.subTest(case=case),
                self.assertRaisesRegex(
                    runner["UnsafeInvocation"], "AEC pause control requires exact"
                ),
            ):
                validate(*case)
        result = subprocess.run(
            [str(CHECK), "--isolated", "--single-check", "--pause-helper-ms", "30"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("AEC pause control requires exact", result.stderr)
        self.assertNotIn("AEC scope NOT_DONE", result.stderr)
        for other_mode in ("--preflight-only", "--stream"):
            with self.subTest(other_mode=other_mode):
                denied = subprocess.run(
                    [
                        str(CHECK),
                        "--isolated",
                        other_mode,
                        "--trial-mode",
                        "wrong-reference",
                        "--trial-frames",
                        "4500",
                        "--pause-helper-ms",
                        "30",
                    ],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    timeout=8,
                    check=False,
                )
                self.assertEqual(denied.returncode, 2)
                self.assertIn("AEC pause control requires exact", denied.stderr)
                self.assertNotIn("AEC scope NOT_DONE", denied.stderr)
        valid_without_owner = subprocess.run(
            [
                str(CHECK),
                "--isolated",
                "--trial-mode",
                "wrong-reference",
                "--trial-frames",
                "4500",
                "--pause-helper-ms",
                "30",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
        self.assertEqual(valid_without_owner.returncode, 2)
        self.assertIn(
            "AEC pause control requires owner session", valid_without_owner.stderr
        )
        self.assertNotIn("AEC scope NOT_DONE", valid_without_owner.stderr)

    def test_prepared_pause_trigger_does_not_wait_for_supervisor(self) -> None:
        namespace = _session_contract()
        for mode in ("impact", "control"):
            with self.subTest(mode=mode):
                backend = subprocess.Popen(["/usr/bin/sleep", "5"])
                pidfd = os.pidfd_open(backend.pid)
                control = None
                try:
                    control = namespace["prepare_pause_control"](
                        pidfd, backend.pid, "0123456789abcdef", 30, mode
                    )
                    if mode == "impact":
                        time.sleep(1.2)
                    with (
                        mock.patch(
                            "select.select",
                            side_effect=AssertionError("trigger waited for supervisor"),
                        ),
                        mock.patch.object(
                            control.supervisor,
                            "communicate",
                            side_effect=AssertionError("trigger reaped supervisor"),
                        ),
                    ):
                        control.trigger(100)
                    receipt = control.finish()
                    self.assertEqual(receipt["control_mode"], mode)
                    self.assertTrue(receipt["supervisor_reaped"])
                    self.assertTrue(receipt["backend_resumed"])
                    self.assertEqual(receipt["stop_signals"], int(mode == "impact"))
                    self.assertEqual(receipt["continue_signals"], int(mode == "impact"))
                    self.assertIsNone(backend.poll())
                    status = Path(f"/proc/{backend.pid}/status").read_text()
                    states = re.findall(
                        r"^State:[ \t]*([A-Za-z])(?:[ \t]|$)", status, re.MULTILINE
                    )
                    self.assertEqual(len(states), 1)
                    self.assertNotIn(states[0], ("T", "t"))
                finally:
                    if control is not None:
                        control.close()
                    signal.pidfd_send_signal(pidfd, signal.SIGCONT)
                    backend.terminate()
                    backend.wait(timeout=2)
                    os.close(pidfd)

    def test_supervisor_pidfd_open_failure_reaps_spawned_child(self) -> None:
        namespace = _session_contract()
        backend = subprocess.Popen(["/usr/bin/sleep", "5"])
        pidfd = os.pidfd_open(backend.pid)
        spawned = []
        real_popen = subprocess.Popen

        def tracked_popen(*args, **kwargs):
            child = real_popen(*args, **kwargs)
            spawned.append(child)
            return child

        try:
            with (
                mock.patch.object(subprocess, "Popen", side_effect=tracked_popen),
                mock.patch.object(os, "pidfd_open", side_effect=OSError("injected")),
            ):
                with self.assertRaisesRegex(OSError, "injected"):
                    namespace["prepare_pause_control"](
                        pidfd, backend.pid, "0123456789abcdef", 30, "impact"
                    )
            self.assertEqual(len(spawned), 1)
            self.assertIsNotNone(spawned[0].poll())
            self.assertIsNone(backend.poll())
        finally:
            signal.pidfd_send_signal(pidfd, signal.SIGCONT)
            backend.terminate()
            backend.wait(timeout=2)
            os.close(pidfd)

    def test_supervisor_cleanup_error_still_reaps_child(self) -> None:
        namespace = _session_contract()
        backend = subprocess.Popen(["/usr/bin/sleep", "5"])
        pidfd = os.pidfd_open(backend.pid)
        control = namespace["prepare_pause_control"](
            pidfd, backend.pid, "0123456789abcdef", 30, "impact"
        )
        real_send = signal.pidfd_send_signal
        raised = False

        def fail_first_continue(target, sig, *args, **kwargs):
            nonlocal raised
            if sig == signal.SIGCONT and not raised:
                raised = True
                raise OSError("injected CONT failure")
            return real_send(target, sig, *args, **kwargs)

        try:
            with mock.patch.object(
                signal, "pidfd_send_signal", side_effect=fail_first_continue
            ):
                with self.assertRaisesRegex(RuntimeError, "cleanup"):
                    control.close()
            self.assertIsNotNone(control.supervisor.poll())
            self.assertTrue(
                all(
                    stream.closed
                    for stream in (
                        control.supervisor.stdin,
                        control.supervisor.stdout,
                        control.supervisor.stderr,
                    )
                )
            )
            self.assertIsNone(backend.poll())
        finally:
            real_send(pidfd, signal.SIGCONT)
            backend.terminate()
            backend.wait(timeout=2)
            os.close(pidfd)

    def test_frame_consumer_only_triggers_prepared_supervisor(self) -> None:
        source = runpy.run_path(str(CHECK), run_name="pause_ast_test")["SESSION"]
        tree = ast.parse(source)
        main = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
        calls = [node for node in ast.walk(main) if isinstance(node, ast.Call)]
        trigger = [
            node
            for node in calls
            if isinstance(node.func, ast.Attribute)
            and ast.unparse(node.func) == "pause_control.trigger"
        ]
        finish = [
            node
            for node in calls
            if isinstance(node.func, ast.Attribute)
            and ast.unparse(node.func) == "pause_control.finish"
        ]
        prepare = [
            node
            for node in calls
            if isinstance(node.func, ast.Name)
            and node.func.id == "prepare_pause_control"
        ]
        start = [
            node
            for node in calls
            if isinstance(node.func, ast.Attribute)
            and ast.unparse(node.func) == "ipc.sendall"
            and len(node.args) == 1
            and ast.unparse(node.args[0]) == "START"
        ]
        self.assertEqual(tuple(map(len, (trigger, prepare, start))), (1, 1, 1))
        self.assertLess(prepare[0].lineno, start[0].lineno)
        self.assertLess(start[0].lineno, trigger[0].lineno)
        consumer_loops = [
            node
            for node in ast.walk(main)
            if isinstance(node, ast.While)
            and node.lineno < trigger[0].lineno < node.end_lineno
        ]
        self.assertEqual(len(consumer_loops), 1)
        self.assertTrue(finish)
        self.assertTrue(
            all(
                not (
                    consumer_loops[0].lineno
                    < node.lineno
                    < consumer_loops[0].end_lineno
                )
                for node in finish
            )
        )
        self.assertFalse(
            any(
                isinstance(node.func, ast.Name)
                and node.func.id == "pause_owned_backend"
                for node in calls
            )
        )

    def test_pause_rejects_foreign_pidfd_before_signal(self) -> None:
        namespace = _session_contract()
        backend = subprocess.Popen(["/usr/bin/sleep", "5"])
        sibling = subprocess.Popen(["/usr/bin/sleep", "5"])
        pidfd = os.pidfd_open(sibling.pid)
        try:
            with self.assertRaisesRegex(RuntimeError, "pidfd identity"):
                namespace["pause_owned_backend"](
                    pidfd, backend.pid, "0123456789abcdef", 100, 30
                )
            self.assertIsNone(backend.poll())
            self.assertIsNone(sibling.poll())
            self.assertNotIn(
                "State:\tT", Path(f"/proc/{sibling.pid}/status").read_text()
            )
        finally:
            signal.pidfd_send_signal(pidfd, signal.SIGCONT)
            for process in (backend, sibling):
                process.terminate()
                process.wait(timeout=2)
            os.close(pidfd)

    def test_pause_rejects_before_hundred_verified_frames(self) -> None:
        namespace = _session_contract()
        backend = subprocess.Popen(["/usr/bin/sleep", "5"])
        pidfd = os.pidfd_open(backend.pid)
        try:
            with self.assertRaisesRegex(RuntimeError, "witness-confirmed"):
                namespace["pause_owned_backend"](
                    pidfd, backend.pid, "0123456789abcdef", 99, 30
                )
            self.assertIsNone(backend.poll())
        finally:
            backend.terminate()
            backend.wait(timeout=2)
            os.close(pidfd)

    def test_pause_owner_loss_after_stop_resumes_owned_child_only(self) -> None:
        owner_code = """
import os, runpy, subprocess, sys
outer = runpy.run_path(sys.argv[1], run_name="pause_owner_test")
session = {"__name__": "pause_owner_test"}
exec(outer["SESSION"], session)
backend = subprocess.Popen(["/usr/bin/sleep", "5"], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
pidfd = os.pidfd_open(backend.pid)
os.write(1, ("P" + str(backend.pid) + "\\n").encode())
session["pause_owned_backend"](
    pidfd, backend.pid, "0123456789abcdef", 100, 30,
    on_stopped=lambda: os.write(1, b"S"),
)
"""
        sibling = subprocess.Popen(["/usr/bin/sleep", "5"])
        owner = subprocess.Popen(
            ["/usr/bin/python3", "-I", "-c", owner_code, str(CHECK)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        owner_pidfd = os.pidfd_open(owner.pid)
        backend_pidfd = None
        try:
            self.assertTrue(select.select([owner.stdout], [], [], 2)[0])
            line = owner.stdout.readline()
            self.assertRegex(line, rb"^P[0-9]+\n$")
            backend_pid = int(line[1:])
            backend_pidfd = os.pidfd_open(backend_pid)
            self.assertTrue(select.select([owner.stdout], [], [], 2)[0])
            self.assertEqual(os.read(owner.stdout.fileno(), 1), b"S")
            self.assertIn("State:\tT", Path(f"/proc/{backend_pid}/status").read_text())
            signal.pidfd_send_signal(owner_pidfd, signal.SIGKILL)
            owner.wait(timeout=2)
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                try:
                    status = Path(f"/proc/{backend_pid}/status").read_text()
                except FileNotFoundError:
                    status = "TERMINAL"
                if "State:\tT" not in status:
                    break
                time.sleep(0.01)
            self.assertNotIn("State:\tT", status)
            self.assertIsNone(sibling.poll())
        finally:
            if backend_pidfd is not None:
                try:
                    signal.pidfd_send_signal(backend_pidfd, signal.SIGCONT)
                    signal.pidfd_send_signal(backend_pidfd, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                os.close(backend_pidfd)
            if owner.poll() is None:
                signal.pidfd_send_signal(owner_pidfd, signal.SIGKILL)
                owner.wait(timeout=2)
            os.close(owner_pidfd)
            if owner.stdout is not None:
                owner.stdout.close()
            if owner.stderr is not None:
                owner.stderr.close()
            sibling.terminate()
            sibling.wait(timeout=2)

    def test_pause_supervisor_resumes_exact_child_after_real_stop(self) -> None:
        namespace = _session_contract()
        backend = subprocess.Popen(["/usr/bin/sleep", "5"])
        pidfd = os.pidfd_open(backend.pid)
        try:
            receipt = namespace["pause_owned_backend"](
                pidfd, backend.pid, "0123456789abcdef", 100, 30
            )
            self.assertEqual(receipt["session"], "0123456789abcdef")
            self.assertEqual(receipt["after_witness_frames"], 100)
            self.assertEqual(receipt["requested_ms"], 30)
            self.assertTrue(receipt["supervisor_ready"])
            self.assertTrue(receipt["stop_observed"])
            self.assertTrue(receipt["resume_observed"])
            self.assertEqual(receipt["stop_signals"], 1)
            self.assertEqual(receipt["continue_signals"], 1)
            self.assertGreaterEqual(receipt["actual_ms"], 20)
            self.assertLess(receipt["stop_observed_ns"], receipt["resume_observed_ns"])
            self.assertIsNone(backend.poll())
        finally:
            signal.pidfd_send_signal(pidfd, signal.SIGCONT)
            backend.terminate()
            backend.wait(timeout=2)
            os.close(pidfd)

    def test_pause_supervisor_loss_after_stop_is_rescued(self) -> None:
        namespace = _session_contract()
        backend = subprocess.Popen(["/usr/bin/sleep", "5"])
        pidfd = os.pidfd_open(backend.pid)
        try:
            with self.assertRaises(RuntimeError):
                namespace["pause_owned_backend"](
                    pidfd,
                    backend.pid,
                    "0123456789abcdef",
                    100,
                    30,
                    test_kill_supervisor_after_stop=True,
                )
            self.assertIsNone(backend.poll())
            state = Path(f"/proc/{backend.pid}/status").read_text()
            self.assertNotIn("State:\tT", state)
        finally:
            signal.pidfd_send_signal(pidfd, signal.SIGCONT)
            backend.terminate()
            backend.wait(timeout=2)
            os.close(pidfd)

    def test_component_state_is_numeric_and_first_failure_only(self) -> None:
        namespace = _session_contract()
        snapshot = (
            b"AEC_FRAME_STATE session=81985529216486895 failed=1 reason=3 "
            b"callback=2257 expected_pos=23404444498 position=23404444978 "
            b"duration=480 clock=15 xrun=0 seq=2256 valid=7 "
            b"queue_head=2256 queue_tail=2255\n"
        )
        with tempfile.TemporaryFile() as diagnostic_file:
            diagnostic_file.write(b"START_WAIT /private/pcm\n" + snapshot)
            diagnostic_file.flush()
            report = namespace["read_component_state"](
                diagnostic_file, label="backend", session="0123456789abcdef"
            )
        self.assertEqual(report["status"], "OBSERVED")
        self.assertEqual(report["expected_position"], 23404444498)
        self.assertEqual(report["position"], 23404444978)
        self.assertEqual(report["valid_mask"], 7)
        self.assertNotIn("/private", json.dumps(report))

        healthy = snapshot.replace(b"failed=1 reason=3", b"failed=0 reason=0")
        with tempfile.TemporaryFile() as diagnostic_file:
            diagnostic_file.write(healthy)
            diagnostic_file.flush()
            for label in ("backend", "fixture", "witness"):
                state = namespace["read_component_state"](
                    diagnostic_file, label=label, session="0123456789abcdef"
                )
                self.assertEqual(state["status"], "NOT_FAILED")
                self.assertEqual(state["source"], label)

        cross_time = snapshot.replace(b"queue_head=2256", b"queue_head=2254")
        with tempfile.TemporaryFile() as diagnostic_file:
            diagnostic_file.write(cross_time)
            diagnostic_file.flush()
            state = namespace["read_component_state"](
                diagnostic_file, label="backend", session="0123456789abcdef"
            )
        self.assertEqual(state["status"], "OBSERVED")
        self.assertEqual((state["queue_head"], state["queue_tail"]), (2254, 2255))

        for payload in (
            snapshot + snapshot,
            snapshot[:-1],
            snapshot.replace(b"session=81985529216486895", b"session=9"),
            snapshot.replace(b"reason=3", b"reason=0"),
            snapshot.replace(b"valid=7", b"valid=8"),
            b"START_WAIT /private/pcm\n" * 4000 + snapshot,
        ):
            with self.subTest(payload=payload[-40:]):
                with tempfile.TemporaryFile() as diagnostic_file:
                    diagnostic_file.write(payload)
                    diagnostic_file.flush()
                    invalid = namespace["read_component_state"](
                        diagnostic_file, label="backend", session="0123456789abcdef"
                    )
                self.assertEqual(invalid["status"], "UNOBSERVED")

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

    def test_first_failure_time_binds_to_failed_snapshot_and_session(self) -> None:
        namespace = _session_contract()
        session = "0123456789abcdef"
        frame = (
            b"AEC_FRAME_STATE session=81985529216486895 failed=1 reason=3 "
            b"callback=102 expected_pos=48000 position=48960 duration=480 "
            b"clock=15 xrun=0 seq=101 valid=7 queue_head=1 queue_tail=2\n"
        )
        gap = (
            b"AEC_GAP kind=1 expected_pos=48000 observed_pos=48960 "
            b"expected_clock=15 observed_clock=15 expected_xrun=0 observed_xrun=0 "
            b"expected_seq=100,100 observed_seq=100,101\n"
        )
        first = (
            b"AEC_FIRST_FAILURE session=81985529216486895 clock=CLOCK_MONOTONIC "
            b"time_ns=1234 before_ns=100 after_ns=110\n"
        )
        with tempfile.TemporaryFile() as diagnostic_file:
            diagnostic_file.write(gap + frame + first)
            diagnostic_file.flush()
            report = namespace["read_native_diagnostics"](
                diagnostic_file, "backend", session
            )
        self.assertEqual(report["failure_before_ns"], 100)
        self.assertEqual(report["failure_after_ns"], 110)
        self.assertEqual(report["time_namespace"], 1234)
        self.assertEqual(report["clock"], "CLOCK_MONOTONIC")
        for payload in (
            gap + first,
            gap + frame + first + first,
            gap + frame + first.replace(b"81985529216486895", b"1"),
            gap + frame.replace(b"failed=1 reason=3", b"failed=0 reason=0") + first,
            gap + frame.replace(b"reason=3", b"reason=4") + first,
            gap + frame.replace(b"position=48960", b"position=48000") + first,
        ):
            with (
                self.subTest(payload=payload[-70:]),
                tempfile.TemporaryFile() as diagnostic_file,
            ):
                diagnostic_file.write(payload)
                diagnostic_file.flush()
                report = namespace["read_native_diagnostics"](
                    diagnostic_file, "backend", session
                )
                self.assertNotIn("failure_before_ns", report)

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

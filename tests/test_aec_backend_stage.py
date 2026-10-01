"""Fail-closed receipt classification for the isolated native AEC matrix."""

from __future__ import annotations

import contextlib
import functools
import io
import json
import runpy
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

STAGE = Path(__file__).resolve().parents[1] / "scripts/translator-aec-backend-stage"
CHECK = STAGE.with_name("translator-aec-backend-check")


def stage() -> dict[str, object]:
    return runpy.run_path(str(STAGE), run_name="aec_stage_contract")


FROZEN = {
    "binary:translator-aec-backend": "a" * 64,
    "binary:translator-aec-fixture": "b" * 64,
    "binary:translator-aec-witness": "c" * 64,
    "plugin": "d" * 64,
}


def trial(case: str, **changes: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "status": "NONPHYSICAL_TRIAL_PASS",
        "aec_proof": False,
        "cleanup_reaped": True,
        "fixture_mode": case,
        "frames": 4500,
        "raw_peak": 0.1,
        "clean_peak": 0.1,
        "changed_samples": 1,
        "cleanup_ms": 10,
        "identity": {
            "backend": FROZEN["binary:translator-aec-backend"],
            "fixture": FROZEN["binary:translator-aec-fixture"],
            "witness": FROZEN["binary:translator-aec-witness"],
            "aec_plugin": FROZEN["plugin"],
        },
        "plugin_mapped": True,
        "component_states": {
            role: {
                "status": "NOT_FAILED",
                "session": "0123456789abcdef",
                "source": role,
                "reason": 0,
                "callback": 4500,
                "expected_position": 2_169_600,
                "position": 2_169_120,
                "duration": 480,
                "clock": 1,
                "xrun": 0,
                "sequence": 4500,
                "valid_mask": 7,
                "queue_head": 4500,
                "queue_tail": 4500,
            }
            for role in ("backend", "fixture", "witness")
        },
        "fixture_delay_ms": 0,
        "speech": None,
        "metric": {"median_erle_db": 17.0},
    }
    payload.update(changes)
    return payload


class AecBackendStageTests(unittest.TestCase):
    def test_first_failure_temporal_order_is_conservative(self) -> None:
        classify = __import__("runpy").run_path(
            str(CHECK), run_name="aec_pause_classification"
        )["classify_failure_order"]
        session = "0123456789abcdef"
        pause = {
            "session": session,
            "time_namespace": 1234,
            "control_mode": "impact",
            "supervisor_reaped": True,
            "backend_resumed": True,
            "stop_signals": 1,
            "continue_signals": 1,
            "stop_send_before_ns": 100,
            "stop_send_after_ns": 110,
            "stop_observed_ns": 120,
            "continue_send_before_ns": 200,
            "continue_send_after_ns": 210,
            "resume_observed_ns": 230,
        }
        native = {
            "status": "OBSERVED",
            "session": session,
            "time_namespace": 1234,
            "clock": "CLOCK_MONOTONIC",
            "failure_before_ns": 50,
            "failure_after_ns": 90,
        }
        cases = (
            ((50, 90), "BEFORE_INTERVENTION"),
            ((211, 215), "AFTER_INTERVENTION"),
            ((205, 220), "OVERLAP"),
            ((90, 115), "OVERLAP"),
            ((90, 100), "OVERLAP"),
            ((210, 215), "OVERLAP"),
            ((215, 211), "UNVERIFIED"),
        )
        for (before, after), expected in cases:
            with self.subTest(before=before, after=after):
                native["failure_before_ns"] = before
                native["failure_after_ns"] = after
                self.assertEqual(classify(pause, native, session, True), expected)
        native["failure_before_ns"] = 211
        native["failure_after_ns"] = 215
        for changed_pause, changed_native, cleaned in (
            ({**pause, "session": "0000000000000001"}, native, True),
            (pause, {**native, "time_namespace": 1235}, True),
            (pause, {**native, "failure_before_ns": None}, True),
            ({**pause, "continue_send_after_ns": 190}, native, True),
            (pause, native, False),
            (pause, {**native, "clock": "CLOCK_REALTIME"}, True),
            ({**pause, "control_mode": "control"}, native, True),
            ({**pause, "supervisor_reaped": False}, native, True),
            ({**pause, "backend_resumed": False}, native, True),
        ):
            self.assertEqual(
                classify(changed_pause, changed_native, session, cleaned), "UNVERIFIED"
            )

    def test_public_pause_receipt_keeps_temporal_order_diagnostic_only(self) -> None:
        classify = __import__("runpy").run_path(
            str(CHECK), run_name="aec_pause_classification"
        )["classify_pause_outcome"]
        session = "0123456789abcdef"
        pause = {
            "session": session,
            "requested_ms": 30,
            "after_witness_frames": 100,
            "control_mode": "impact",
            "time_namespace": 1234,
            "supervisor_ready": True,
            "supervisor_reaped": True,
            "backend_resumed": True,
            "stop_observed": True,
            "resume_observed": True,
            "stop_signals": 1,
            "continue_signals": 1,
            "stop_send_before_ns": 1_000_000_000,
            "stop_send_after_ns": 1_001_000_000,
            "stop_observed_ns": 1_002_000_000,
            "continue_send_before_ns": 1_031_000_000,
            "continue_send_after_ns": 1_032_000_000,
            "resume_observed_ns": 1_033_000_000,
            "actual_ms": 31.0,
        }
        native = {
            "status": "OBSERVED",
            "session": session,
            "source": "backend",
            "gap_kind": 1,
            "clock": "CLOCK_MONOTONIC",
            "time_namespace": 1234,
            "failure_before_ns": 1_032_100_000,
            "failure_after_ns": 1_032_200_000,
        }
        failure = "".join(
            (
                "AEC_PAUSE_RECEIPT=" + json.dumps(pause) + "\n",
                "AEC_NATIVE_DIAGNOSTICS=" + json.dumps(native) + "\n",
                "AEC isolated session NOT_DONE: AEC helper FATAL reason=3 processed=100; cleanup_reaped=True cleanup_ms=5\n",
            )
        )
        result = classify(2, "", failure, session)
        self.assertEqual(result["temporal_order"], "AFTER_INTERVENTION")
        self.assertEqual(result["classification"], "OTHER_FAILURE")
        self.assertFalse(result["aec_proof"])
        duplicate = failure + "AEC_PAUSE_RECEIPT=" + json.dumps(pause) + "\n"
        self.assertEqual(
            classify(2, "", duplicate, session)["temporal_order"], "UNVERIFIED"
        )

    def test_pause_outcome_requires_verified_impact_and_first_gap(self) -> None:
        classify_pause = __import__("runpy").run_path(
            str(CHECK), run_name="aec_pause_classification"
        )["classify_pause_outcome"]
        session = "0123456789abcdef"
        pause = {
            "session": session,
            "requested_ms": 30,
            "after_witness_frames": 100,
            "supervisor_ready": True,
            "stop_observed": True,
            "resume_observed": True,
            "stop_observed_ns": 1_000_000_000,
            "resume_observed_ns": 1_031_000_000,
            "actual_ms": 31.0,
            "stop_signals": 1,
            "continue_signals": 1,
            "supervisor_reaped": True,
            "backend_resumed": True,
        }
        marker = "AEC_PAUSE_RECEIPT=" + json.dumps(pause) + "\n"
        clean = json.dumps(
            trial("wrong-reference", metric={"median_erle_db": 3.0}, test_pause=pause)
        )
        self.assertEqual(
            classify_pause(0, clean, marker, session)["classification"], "NO_REPRO"
        )
        components = {
            role: {
                "status": "OBSERVED",
                "session": session,
                "source": role,
                "reason": 3,
                "callback": 101,
                "expected_position": 48_000,
                "position": 48_480,
            }
            for role in ("backend", "fixture", "witness")
        }
        native = {
            "status": "OBSERVED",
            "session": session,
            "source": "backend",
            "gap_kind": 1,
            "expected_position": 48_000,
            "observed_position": 48_480,
            "expected_sequences": [100, 100],
            "observed_sequences": [100, 101],
        }
        fixture = {
            "status": "OBSERVED",
            "session": session,
            "source": "fixture",
            "expected_pos": 48_000,
            "observed_pos": 48_480,
        }
        failure = "".join(
            (
                marker,
                "AEC_COMPONENT_DIAGNOSTICS=" + json.dumps(components) + "\n",
                "AEC_NATIVE_DIAGNOSTICS=" + json.dumps(native) + "\n",
                "AEC_FIXTURE_DIAGNOSTICS=" + json.dumps(fixture) + "\n",
                "AEC isolated session NOT_DONE: AEC helper FATAL reason=3 processed=100; cleanup_reaped=True cleanup_ms=5\n",
            )
        )
        self.assertNotIn("callback", native)
        result = classify_pause(2, "", failure, session)
        self.assertEqual(result["classification"], "OTHER_FAILURE")
        self.assertEqual(result["temporal_order"], "UNVERIFIED")
        self.assertEqual(result["diagnostics"]["backend"], native)
        self.assertFalse(
            stage()["classify"]("wrong-reference", 2, "", failure, FROZEN)[0]
        )
        for bad in (
            failure.replace('callback": 101', 'callback": 99'),
            failure.replace("cleanup_reaped=True", "cleanup_reaped=False"),
            failure.replace(session, "0000000000000001"),
            failure.replace("AEC helper FATAL reason=3", "AEC IPC timeout"),
            failure.replace('gap_kind": 1', 'gap_kind": 0'),
            failure.replace("AEC_PAUSE_RECEIPT=", "AEC_PAUSE_MISSING="),
            failure + "AEC_PAUSE_RECEIPT=" + json.dumps(pause) + "\n",
        ):
            with self.subTest(bad=bad[-70:]):
                self.assertEqual(
                    classify_pause(2, "", bad, session)["classification"],
                    "OTHER_FAILURE",
                )
        self.assertEqual(
            classify_pause(2, "", marker + "IPC timeout\n", session)["classification"],
            "OTHER_FAILURE",
        )
        self.assertEqual(
            classify_pause(
                2, "", marker + "AEC_BUFFER_PRECONDITION site=311\n", session
            )["classification"],
            "OTHER_FAILURE",
        )

    def test_unexpected_native_failure_retains_bounded_first_failure(self) -> None:
        classify = stage()["classify"]
        session = "0123456789abcdef"
        components = {
            role: {
                "status": "OBSERVED",
                "session": session,
                "source": role,
                "reason": 5,
                "callback": 2256,
                "expected_position": 1_082_880,
                "position": 1_083_360,
            }
            for role in ("backend", "fixture", "witness")
        }
        native = {
            "status": "OBSERVED",
            "session": session,
            "source": "backend",
            "gap_kind": 1,
            "expected_position": 1_082_880,
            "observed_position": 1_083_360,
        }
        fixture = {
            "status": "OBSERVED",
            "session": session,
            "source": "fixture",
            "expected_pos": 1_082_880,
            "observed_pos": 1_083_360,
        }
        stderr = "\n".join(
            (
                "AEC_COMPONENT_DIAGNOSTICS=" + json.dumps(components),
                "AEC_NATIVE_DIAGNOSTICS=" + json.dumps(native),
                "AEC_FIXTURE_DIAGNOSTICS=" + json.dumps(fixture),
                "AEC isolated session NOT_DONE: AEC FRAME gap or graph change; cleanup_reaped=True cleanup_ms=4",
            )
        )
        passed, detail = classify("wrong-reference", 2, "", stderr, FROZEN)
        self.assertFalse(passed)
        self.assertEqual(detail["reason"], "nonzero exit")
        self.assertEqual(detail["diagnostics"]["components"], components)
        self.assertEqual(detail["diagnostics"]["backend"], native)
        self.assertEqual(detail["diagnostics"]["fixture"], fixture)
        self.assertIn("AEC FRAME gap", detail["failure"])

    def test_held_outer_request_retains_custody_through_late_outcome(self) -> None:
        namespace = stage()
        supervise = namespace["supervise_outer_scope_acquisition"]
        outer = Path("/sys/fs/cgroup/missing-outer.scope")
        inner = Path("/sys/fs/cgroup/missing-inner.scope")

        class ScopeRejected(Exception):
            pass

        class ScopeCreationUnknown(Exception):
            pass

        class StopObservation(Exception):
            pass

        for outcome in ("rejected", "late_success", "unknown"):
            with self.subTest(outcome=outcome):
                released = threading.Event()
                clock = [0.0]
                child = mock.Mock(pid=1234, returncode=None)
                child.poll.side_effect = functools.partial(
                    lambda target: target.returncode,
                    child,
                )
                child.wait.side_effect = functools.partial(
                    lambda target, **_kwargs: setattr(target, "returncode", -9),
                    child,
                )

                def acquire(
                    *_args: object, release: threading.Event, result: str
                ) -> Path:
                    release.wait()
                    if result == "rejected":
                        raise ScopeRejected("manager rejected")
                    if result == "unknown":
                        raise ScopeCreationUnknown("manager outcome unknown")
                    return outer

                def tick(
                    _seconds: float,
                    *,
                    current_clock: list[float],
                    release: threading.Event,
                    result: str,
                ) -> None:
                    current_clock[0] += 0.05
                    if current_clock[0] >= 9:
                        release.set()
                    if result == "unknown" and current_clock[0] >= 10:
                        raise StopObservation
                    threading.Event().wait(0.0001)

                bound_acquire = functools.partial(
                    acquire, release=released, result=outcome
                )
                bound_tick = functools.partial(
                    tick,
                    current_clock=clock,
                    release=released,
                    result=outcome,
                )

                output = io.StringIO()
                gate_read, gate_write = namespace["os"].pipe()
                namespace["os"].close(gate_read)
                try:
                    with (
                        mock.patch.object(
                            namespace["time"],
                            "monotonic",
                            side_effect=functools.partial(
                                lambda current: current[0], clock
                            ),
                        ),
                        mock.patch.object(
                            namespace["time"], "sleep", side_effect=bound_tick
                        ),
                        mock.patch.object(
                            namespace["signal"], "pidfd_send_signal"
                        ) as kill_child,
                        mock.patch.dict(
                            supervise.__globals__,
                            {
                                "kill_exact_scope": lambda _scope, bound: (bound, True),
                            },
                        ),
                        contextlib.redirect_stdout(output),
                    ):
                        kill_child.side_effect = functools.partial(
                            lambda target, *_args: setattr(target, "returncode", -9),
                            child,
                        )
                        runner = {
                            "acquire_scope": bound_acquire,
                            "ScopeRejected": ScopeRejected,
                        }
                        args = (
                            runner,
                            child,
                            55,
                            outer,
                            inner,
                            "translator-aec-" + "a" * 32,
                            gate_write,
                            "held-case",
                        )
                        if outcome == "unknown":
                            with self.assertRaises(StopObservation):
                                supervise(*args)
                        else:
                            admitted, bound, reason = supervise(*args)
                            self.assertFalse(admitted)
                            self.assertIsNone(bound)
                            self.assertEqual(
                                reason,
                                "outer_scope_rejected"
                                if outcome == "rejected"
                                else "outer_scope_timeout",
                            )
                        kill_child.assert_called()
                        self.assertIn('"status": "CleanupPending"', output.getvalue())
                finally:
                    released.set()

    def test_outer_request_admits_only_bound_live_scope(self) -> None:
        namespace = stage()
        child = mock.Mock(pid=1234)
        child.poll.return_value = None
        with tempfile.TemporaryDirectory() as directory:
            outer = Path(directory)
            gate_read, gate_write = namespace["os"].pipe()
            namespace["os"].close(gate_read)
            try:
                admitted, bound, reason = namespace[
                    "supervise_outer_scope_acquisition"
                ](
                    {
                        "acquire_scope": lambda *_args: outer,
                        "ScopeRejected": RuntimeError,
                    },
                    child,
                    55,
                    outer,
                    outer / "inner",
                    "translator-aec-" + "a" * 32,
                    gate_write,
                    "fast-case",
                )
                self.assertTrue(admitted)
                self.assertIsNotNone(bound)
                self.assertIsNone(reason)
                child.poll.assert_called()
                namespace["os"].close(bound)
            finally:
                namespace["os"].close(gate_write)

    def test_native_case_selects_exact_prebuilt_binary_without_cargo_runtime(
        self,
    ) -> None:
        namespace = stage()
        artifact = {
            "reason": "compiler-artifact",
            "target": {"name": "aec_backend_session", "kind": ["test"]},
            "profile": {"test": True},
            "executable": "/tmp/target/debug/deps/aec_backend_session-abc",
        }
        output = (
            json.dumps({"reason": "build-script-executed"})
            + "\n"
            + json.dumps(artifact)
        )
        binary = namespace["native_test_binary_from_build_output"](output)
        self.assertEqual(str(binary), artifact["executable"])
        self.assertEqual(
            namespace["native_test_command"](
                binary, "isolated_native_stream_is_owned_and_cancelled"
            ),
            [
                artifact["executable"],
                "--exact",
                "isolated_native_stream_is_owned_and_cancelled",
                "--ignored",
                "--nocapture",
            ],
        )
        with self.assertRaises(ValueError):
            namespace["native_test_binary_from_build_output"](
                output + "\n" + json.dumps(artifact)
            )

    def test_stage_lifecycle_requires_exact_p_then_terminal_identity(self) -> None:
        parse = stage()["parse_native_lifecycle"]
        unit = "translator-aec-" + "a" * 32
        session = "0123456789abcdef"
        p = {
            "event": "P",
            "unit": unit,
            "pid": 1234,
            "session": session,
            "manager_owner": ":1.42",
        }
        a = dict(p, event="A", job="/org/freedesktop/systemd1/job/17")
        self.assertEqual(
            parse((json.dumps(p) + "\n" + json.dumps(a) + "\n").encode(), unit), "A"
        )
        self.assertEqual(parse((json.dumps(p) + "\n").encode(), unit), "PENDING")
        self.assertEqual(
            parse(
                (
                    json.dumps(
                        {
                            key: value
                            for key, value in p.items()
                            if key != "manager_owner"
                        }
                        | {"event": "N"}
                    )
                    + "\n"
                ).encode(),
                unit,
            ),
            "N",
        )
        for events in (
            [a],
            [p, p],
            [p, dict(a, unit="translator-aec-" + "b" * 32)],
            [p, dict(a, pid=1235)],
        ):
            with self.assertRaises(ValueError):
                parse(
                    "".join(json.dumps(event) + "\n" for event in events).encode(), unit
                )

    def test_stage_binds_launcher_from_real_newline_p_marker(self) -> None:
        bind = stage()["bind_stage_launcher"]
        unit = "translator-aec-" + "a" * 32
        outer = Path("/sys/fs/cgroup/user.slice/test.scope")
        event = {
            "event": "P",
            "unit": unit,
            "pid": 1234,
            "session": "0123456789abcdef",
            "manager_owner": ":1.42",
        }
        with (
            mock.patch("os.pidfd_open", return_value=77),
            mock.patch.object(
                Path, "read_text", return_value="0::/user.slice/test.scope"
            ),
        ):
            self.assertEqual(
                bind((json.dumps(event) + "\n").encode(), unit, outer),
                (1234, 77),
            )

    def test_native_settlement_keeps_ambiguous_and_direct_failures_pending(
        self,
    ) -> None:
        settled = stage()["native_case_settled"]
        self.assertTrue(settled("A", False, False, True, True))
        self.assertTrue(settled("N", False, False, True, True))
        self.assertTrue(settled("N", False, True, True, True))
        self.assertTrue(settled("NO_REQUEST", False, False, True, True))
        self.assertTrue(settled("NO_REQUEST", False, True, True, True))
        self.assertTrue(settled("NO_REQUEST", True, True, True, True))
        self.assertFalse(settled("PENDING", False, False, True, True))
        self.assertTrue(settled("NO_REQUEST", True, False, True, True))
        accepted = stage()["native_case_accepted"]
        self.assertTrue(accepted("A", False, True))
        self.assertTrue(accepted("NO_REQUEST", True, True))
        self.assertFalse(accepted("N", False, True))
        self.assertFalse(accepted("NO_REQUEST", True, False))
        self.assertFalse(accepted("A", True, True))
        self.assertFalse(settled("A", False, True, False, True))
        self.assertFalse(settled("A", False, True, True, False))

    def test_gio_runtime_is_in_frozen_artifact_set(self) -> None:
        namespace = stage()
        frozen = namespace["frozen_inputs"]()
        self.assertGreater(sum(key.startswith("gi:") for key in frozen), 0)
        self.assertEqual(sum(key.startswith("typelib:") for key in frozen), 3)
        self.assertEqual(sum(key.startswith("gio-lib:") for key in frozen), 3)

    def test_fault_requires_confirmed_injection_and_matching_failure(self) -> None:
        classify = stage()["classify"]
        session = "0123456789abcdef"
        fault = {
            "session": session,
            "fault": "raw-unlink",
            "injection_at_frame": 10,
            "injection_confirmed": True,
            "post_injection_failure": True,
            "cleanup_reaped": True,
        }
        diagnostic = {"session": session, "fatal_reason": 10, "fatal_processed": 10}
        stderr = (
            "AEC_FAULT_RECEIPT=" + json.dumps(fault) + "\n"
            "AEC_NATIVE_DIAGNOSTICS=" + json.dumps(diagnostic) + "\n"
            "AEC isolated session NOT_DONE: AEC helper FATAL reason=10 "
            "processed=10; cleanup_reaped=True\n"
        )
        self.assertTrue(classify("fault:raw-unlink", 2, "", stderr, FROZEN)[0])
        self.assertFalse(classify("fault:raw-unlink", 0, "", stderr, FROZEN)[0])
        self.assertFalse(classify("fault:output-unlink", 2, "", stderr, FROZEN)[0])
        for field, value in (
            ("injection_at_frame", None),
            ("injection_confirmed", False),
            ("post_injection_failure", False),
            ("cleanup_reaped", False),
        ):
            modified = dict(fault, **{field: value})
            rejected = stderr.replace(json.dumps(fault), json.dumps(modified))
            self.assertFalse(classify("fault:raw-unlink", 2, "", rejected, FROZEN)[0])
        self.assertFalse(
            classify(
                "fault:raw-unlink",
                2,
                "",
                stderr.replace('"fatal_reason": 10', '"fatal_reason": 4'),
                FROZEN,
            )[0]
        )

    def test_trials_bind_mode_frames_metric_and_identity(self) -> None:
        classify = stage()["classify"]
        good = trial("far-only")
        self.assertTrue(classify("far-only", 0, json.dumps(good), "", FROZEN)[0])
        for changes in (
            {"fixture_mode": "speech-far"},
            {"frames": 100},
            {"metric": None},
            {"metric": {"median_erle_db": 14.99}},
            {"identity": {}},
            {"cleanup_reaped": False},
            {"aec_proof": True},
        ):
            self.assertFalse(
                classify(
                    "far-only",
                    0,
                    json.dumps(trial("far-only", **changes)),
                    "",
                    FROZEN,
                )[0],
                changes,
            )
        self.assertFalse(classify("speech-far", 0, json.dumps(good), "", FROZEN)[0])
        self.assertTrue(
            classify(
                "wrong-reference",
                0,
                json.dumps(trial("wrong-reference", metric={"median_erle_db": 3.0})),
                "",
                FROZEN,
            )[0]
        )
        self.assertFalse(
            classify(
                "wrong-reference",
                0,
                json.dumps(trial("wrong-reference")),
                "",
                FROZEN,
            )[0]
        )

    def test_success_requires_three_bound_component_states(self) -> None:
        classify = stage()["classify"]
        good = trial("far-only")
        self.assertTrue(classify("far-only", 0, json.dumps(good), "", FROZEN)[0])
        crossed = {
            **good["component_states"],
            "backend": {
                **good["component_states"]["backend"],
                "queue_head": 1,
                "queue_tail": 2,
            },
        }
        self.assertTrue(
            classify(
                "far-only",
                0,
                json.dumps(trial("far-only", component_states=crossed)),
                "",
                FROZEN,
            )[0]
        )
        for states in (
            {},
            {
                key: value
                for key, value in good["component_states"].items()
                if key != "witness"
            },
            {
                **good["component_states"],
                "backend": {
                    **good["component_states"]["backend"],
                    "status": "UNOBSERVED",
                },
            },
            {
                **good["component_states"],
                "fixture": {
                    **good["component_states"]["fixture"],
                    "session": "fedcba9876543210",
                },
            },
            {
                **good["component_states"],
                "witness": {**good["component_states"]["witness"], "source": "backend"},
            },
        ):
            self.assertFalse(
                classify(
                    "far-only",
                    0,
                    json.dumps(trial("far-only", component_states=states)),
                    "",
                    FROZEN,
                )[0]
            )

    def test_near_and_startup_have_distinct_contracts(self) -> None:
        classify = stage()["classify"]
        near = trial(
            "near-only",
            metric={
                "active_seconds": 25,
                "minimum_waveform_correlation": 0.95,
                "median_level_change_db": -0.4,
                "minimum_level_change_db_diagnostic": -0.5,
                "maximum_level_change_db_diagnostic": -0.3,
            },
        )
        self.assertTrue(classify("near-only", 0, json.dumps(near), "", FROZEN)[0])
        self.assertFalse(
            classify(
                "near-only",
                0,
                json.dumps(dict(near, metric={"active_seconds": 1})),
                "",
                FROZEN,
            )[0]
        )
        self.assertFalse(
            classify(
                "near-only",
                0,
                json.dumps(
                    dict(
                        near,
                        metric={
                            **near["metric"],
                            "minimum_level_change_db_diagnostic": -20.0,
                        },
                    )
                ),
                "",
                FROZEN,
            )[0]
        )
        startup = trial(
            "prevalid-once",
            frames=100,
            fixture_delay_ms=250,
            metric=None,
            prevalid={"quarantined": 2, "empty": 1, "sentinel": 1},
        )
        self.assertTrue(
            classify(
                "startup-delay-250",
                0,
                json.dumps(startup),
                "",
                FROZEN,
            )[0]
        )
        self.assertFalse(
            classify(
                "startup-delay-750",
                0,
                json.dumps(startup),
                "",
                FROZEN,
            )[0]
        )
        self.assertFalse(
            classify(
                "startup-delay-250",
                0,
                json.dumps(
                    dict(
                        startup, prevalid={"quarantined": 0, "empty": 0, "sentinel": 0}
                    )
                ),
                "",
                FROZEN,
            )[0]
        )

    def test_lifecycle_and_gap_require_causal_receipt(self) -> None:
        classify = stage()["classify"]
        events = [
            {
                "cycle": index,
                "case": (
                    "cancel"
                    if index % 5 == 0
                    else "seq-skew"
                    if index % 5 == 1
                    else "normal"
                ),
                "exit": 0 if index % 5 >= 2 else 2,
                "failure": None,
                "scope_absent": True,
                "owned_process_count": 0,
                "owned_graph_count": 0,
                "fd_delta": 0,
                "thread_delta": 0,
            }
            for index in range(100)
        ]
        lifecycle = {
            "status": "NONPHYSICAL_LIFECYCLE_PASS",
            "aec_proof": False,
            "completed_cycles": 100,
            "events": events,
        }
        self.assertTrue(classify("lifecycle", 0, json.dumps(lifecycle), "", FROZEN)[0])
        self.assertFalse(
            classify(
                "lifecycle",
                0,
                json.dumps(dict(lifecycle, events=[])),
                "",
                FROZEN,
            )[0]
        )
        self.assertFalse(
            classify(
                "lifecycle",
                0,
                json.dumps(
                    dict(
                        lifecycle, events=[dict(events[0], case="normal"), *events[1:]]
                    )
                ),
                "",
                FROZEN,
            )[0]
        )
        gap = {
            "fatal_reason": 4,
            "fatal_processed": 0,
            "status": "OBSERVED",
            "source": "backend",
            "buffer_reason": [7, 7, 0],
            "raw_chunk": [1920, 4, 0],
            "reference_chunk": [1920, 4, 0],
            "raw_header": [1, 18, 0],
            "reference_header": [1, 18, 0],
            "started": 1,
            "callback": 1,
        }
        stderr = (
            "AEC_NATIVE_DIAGNOSTICS=" + json.dumps(gap) + "\n"
            "AEC isolated session NOT_DONE: cleanup_reaped=True\n"
        )
        self.assertTrue(classify("meta-gap", 2, "", stderr, FROZEN)[0])
        for changed in (
            dict(gap, buffer_reason=[5, 5, 0]),
            dict(gap, raw_header=[1, 0, 0]),
            {key: value for key, value in gap.items() if key != "raw_header"},
        ):
            other_stderr = stderr.replace(json.dumps(gap), json.dumps(changed))
            self.assertFalse(classify("meta-gap", 2, "", other_stderr, FROZEN)[0])
        self.assertFalse(
            classify(
                "meta-gap",
                2,
                "",
                stderr.replace('"fatal_processed": 0', '"fatal_processed": 1'),
                FROZEN,
            )[0]
        )

    def test_owner_receipt_requires_exact_test_count(self) -> None:
        owner = stage()["owner_receipt"]
        names = (
            "cancel_is_shared_and_reaps_owned_process_group",
            "concurrent_cancel_calls_join_one_cleanup",
            "dropped_caller_still_drives_cleanup",
            "no_progress_is_bounded_and_cannot_become_success",
            "exited_leader_does_not_release_descendant_group",
            "isolated_runner_rejects_stream_with_foreign_session",
            "exact_stream_requires_clean_eof_exit_and_cleanup_before_result",
            "late_fatal_extra_frame_partial_eof_and_nonzero_exit_never_return_result",
            "owner_shutdown_joins_the_same_cleanup_task",
            "failed_first_cleanup_keeps_live_descendant_owned_until_retry",
            "session::scope_tests::replaced_scope_cannot_receive_kill",
            "session::scope_tests::unavailable_kill_control_retains_cleanup_obligation",
            "session::scope_tests::failed_cgroup_write_keeps_owner_pending_until_scope_is_gone",
            "session::scope_tests::panicked_cleanup_task_never_reports_idle",
            "owned_group_is_a_private_session",
            "same_session_member_in_other_group_prevents_early_reap",
            "session::scope_tests::late_scope_is_bound_before_retry_kill",
            "session::scope_tests::absent_scope_does_not_clear_late_creation_obligation",
            "session::scope_tests::collector_panic_keeps_process_custody",
            "session::scope_tests::caller_runtime_drop_does_not_drop_process_custody",
            "session::scope_tests::concurrent_shutdown_calls_share_entry_deadline",
            "session::scope_tests::shutdown_during_held_startup_still_cancels_within_entry_budget",
            "session::scope_tests::trusted_runner_command_has_no_caller_control",
            "session::scope_tests::explicit_manager_rejection_releases_absent_scope_custody",
            "session::scope_tests::admitted_then_vanished_scope_releases_custody",
            "session::scope_tests::invalid_or_conflicting_manager_marker_cannot_settle_creation",
            "session::scope_tests::pre_request_exit_without_marker_releases_owner_after_reap",
            "session::scope_tests::sent_request_then_eof_without_terminal_remains_unknown",
            "session::scope_tests::observed_scope_is_killed_while_manager_outcome_is_unknown",
        )
        lines = "".join(f"test {name} ... ok\n" for name in names)
        summary = (
            "test result: ok. 29 passed; 0 failed; 5 ignored; "
            "0 measured; 0 filtered out\n"
        )
        self.assertTrue(owner("owner-contract", 0, lines + summary))
        self.assertFalse(owner("owner-contract", 0, summary))
        self.assertFalse(
            owner(
                "owner-contract",
                0,
                lines.replace(
                    "cancel_is_shared_and_reaps_owned_process_group", "foreign_test"
                )
                + summary,
            )
        )

    def test_frozen_artifact_set_includes_runtime_and_harness(self) -> None:
        paths = stage()["frozen_inputs"]()
        for key in (
            "binary:translator-aec-backend",
            "plugin",
            "runtime:pipewire",
            "source:scripts/translator-aec-backend-check",
            "source:scripts/translator-aec-backend-stage",
            "source:crates/translator-daemon/src/aec_backend_session.rs",
            "config:SESSION",
            "speech:far",
            "speech:near",
            "dlopen:libspa-audioconvert.so",
            "dlopen:libspa-support.so",
            "dlopen:libpipewire-module-adapter.so",
        ):
            self.assertIn(key, paths)
        self.assertTrue(any(name.startswith("owner-test:") for name in paths))
        self.assertTrue(any(name.startswith("linked:") for name in paths))
        self.assertTrue(all(len(digest) == 64 for digest in paths.values()))


if __name__ == "__main__":
    unittest.main()

"""Fail-closed receipt classification for the isolated native AEC matrix."""

from __future__ import annotations

import json
import runpy
import unittest
from pathlib import Path

STAGE = Path(__file__).resolve().parents[1] / "scripts/translator-aec-backend-stage"


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
        "fixture_delay_ms": 0,
        "speech": None,
        "metric": {"median_erle_db": 17.0},
    }
    payload.update(changes)
    return payload


class AecBackendStageTests(unittest.TestCase):
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

"""Negative controls for the packaged-sidecar overlap evidence oracle."""

from __future__ import annotations

import ast
import copy
import runpy
import unittest
from pathlib import Path
from types import SimpleNamespace

CHECK = (
    Path(__file__).resolve().parents[1] / "scripts/translator_duplex_overlap_check.py"
)
ORACLE = runpy.run_path(str(CHECK), run_name="duplex_overlap_contract")
evaluate_attempt = ORACLE["evaluate_attempt"]
evaluate_overlap = ORACLE["evaluate_overlap"]
NS_PER_MS = 1_000_000
CLOCK_BASE_NS = 10_000 * NS_PER_MS


def attempt(language: str, frames: int, first_pcm_ms: int) -> dict:
    direction = (
        "AUDIO_DIRECTION_MICROPHONE" if language == "ru" else "AUDIO_DIRECTION_SPEAKER"
    )
    identity = {
        "session_id": f"{language}-session",
        "direction_id": direction,
        "stream_id": f"{language}-stream",
        "utterance_id": f"{language}-utterance",
    }
    events = []

    def event(kind: str, at_ms: int, **body) -> None:
        events.append(
            {
                "kind": kind,
                "received_ns": CLOCK_BASE_NS + at_ms * NS_PER_MS,
                "body": {
                    **identity,
                    "event_sequence": len(events),
                    **body,
                },
            }
        )

    event("session_opened", -20)
    event(
        "health",
        -10,
        provider_id="PROVIDER_ID_LOCAL",
        state="PROVIDER_STATE_READY",
        models=[
            {
                "kind": kind,
                "id": model_id,
                "state": "MODEL_STATE_READY",
                "device": device,
            }
            for kind, model_id, device in (
                (
                    "MODEL_KIND_ASR",
                    "faster-whisper-large-v3-turbo",
                    "COMPUTE_DEVICE_CUDA",
                ),
                (
                    "MODEL_KIND_MT",
                    "hy-mt2-1.8b-gguf-q4-k-m",
                    "COMPUTE_DEVICE_CUDA",
                ),
                ("MODEL_KIND_TTS", "piper-medium", "COMPUTE_DEVICE_CPU"),
            )
        ],
    )
    event("transcript_delta", first_pcm_ms - 300, text="Source speech.", is_final=True)
    event(
        "translation_delta",
        first_pcm_ms - 100,
        text="Translated speech.",
        is_final=True,
    )
    for sequence in range(2):
        event(
            "audio_delta",
            first_pcm_ms + sequence * 20,
            sequence=sequence,
            sample_rate_hz=24_000,
            channels=1,
            sample_format="SAMPLE_FORMAT_S16LE",
            frame_duration_ms=20,
            byte_count=960,
            pcm_sha256="a" * 64,
            nonzero=True,
        )
    event("latency", first_pcm_ms + 40, provider_total_ms=first_pcm_ms + 40)
    event(
        "utterance_final",
        first_pcm_ms + 50,
        final_audio_sequence=1,
        outcome="UTTERANCE_OUTCOME_COMPLETED",
    )
    event("session_closed", first_pcm_ms + 60, reason="SESSION_CLOSE_REASON_USER_STOP")
    return {
        "language": language,
        **identity,
        "sent_frames": [
            {
                "sequence": sequence,
                "sent_ns": CLOCK_BASE_NS + sequence * 100 * NS_PER_MS,
                "nonzero": True,
                "end_of_utterance": sequence == frames - 1,
            }
            for sequence in range(frames)
        ],
        "events": events,
    }


def matching_events(trace: dict, kind: str) -> list[dict]:
    return [event for event in trace["events"] if event["kind"] == kind]


class DuplexOverlapOracleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.short = attempt("ru", 3, 600)
        self.long = attempt("en", 12, 1500)

    def check(self) -> dict:
        return evaluate_overlap(self.short, self.long, short_language="ru")

    def test_probe_builder_matches_the_actual_server_contract_before_bootstrap(
        self,
    ) -> None:
        server = CHECK.parents[1] / "sidecar/translator_sidecar/grpc_server.py"
        schema = next(
            node.value
            for node in ast.parse(server.read_text()).body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "_PROBE_REQUEST_VERSION"
                for target in node.targets
            )
        )
        request = ORACLE["probe_request"](
            SimpleNamespace(ProviderProbeRequest=lambda **body: SimpleNamespace(**body))
        )
        self.assertEqual(request.schema_version, ast.literal_eval(schema))

    def test_O8_rejects_final_inference_before_observed_end_of_utterance(self) -> None:
        trace = copy.deepcopy(self.short)
        matching_events(trace, "transcript_delta")[0]["received_ns"] = (
            trace["sent_frames"][-1]["sent_ns"] - 1
        )
        with self.assertRaises(ValueError):
            evaluate_attempt(trace)

    def test_O1_accepts_observed_voiced_overlap_in_both_directions(self) -> None:
        self.assertEqual(evaluate_attempt(self.short)["status"], "completed")
        self.assertEqual(self.check()["status"], "pass")
        self.assertEqual(
            evaluate_overlap(
                attempt("ru", 12, 1500), attempt("en", 3, 600), short_language="en"
            )["status"],
            "pass",
        )

    def test_O2_rejects_sequential_send_intervals(self) -> None:
        for frame in self.long["sent_frames"]:
            frame["sent_ns"] += 1000 * NS_PER_MS
        for event in self.long["events"]:
            event["received_ns"] += 1000 * NS_PER_MS
        with self.assertRaisesRegex(ValueError, "overlap|send intervals"):
            self.check()

    def test_O3_rejects_future_silence_padding(self) -> None:
        for frame in self.long["sent_frames"]:
            if frame["sent_ns"] > CLOCK_BASE_NS + 600 * NS_PER_MS:
                frame["nonzero"] = False
        with self.assertRaisesRegex(ValueError, "voiced|nonzero|speech"):
            self.check()

    def test_O4_rejects_short_pcm_after_last_opposite_speech_frame(self) -> None:
        self.short = attempt("ru", 3, 1200)
        with self.assertRaisesRegex(ValueError, "voiced|nonzero|speech"):
            self.check()

    def test_O5_rejects_crossed_session_direction_stream_or_utterance(self) -> None:
        for field in ("session_id", "direction_id", "stream_id", "utterance_id"):
            with self.subTest(field=field):
                trace = copy.deepcopy(self.short)
                matching_events(trace, "audio_delta")[0]["body"][field] = self.long[
                    field
                ]
                with self.assertRaisesRegex(ValueError, "identity"):
                    evaluate_attempt(trace)

    def test_O6_rejects_small_asr_fallback(self) -> None:
        matching_events(self.short, "health")[0]["body"]["models"][0]["id"] = (
            "faster-whisper-small"
        )
        with self.assertRaisesRegex(ValueError, "health|model|fallback"):
            self.check()

    def test_O6_rejects_degraded_or_wrong_device_health(self) -> None:
        for field, value in (
            ("state", "MODEL_STATE_FAILED"),
            ("device", "COMPUTE_DEVICE_CPU"),
        ):
            with self.subTest(field=field):
                trace = copy.deepcopy(self.short)
                matching_events(trace, "health")[0]["body"]["models"][0][field] = value
                with self.assertRaisesRegex(ValueError, "health|model|device"):
                    evaluate_attempt(trace)

    def test_O7_rejects_duplicate_or_unordered_audio_sequences(self) -> None:
        for sequence in (0, 3):
            with self.subTest(sequence=sequence):
                trace = copy.deepcopy(self.short)
                matching_events(trace, "audio_delta")[1]["body"]["sequence"] = sequence
                with self.assertRaisesRegex(ValueError, "sequence|order"):
                    evaluate_attempt(trace)

    def test_O7_rejects_event_sequence_or_receive_time_regression(self) -> None:
        for field, value in (
            ("event_sequence", 0),
            ("received_ns", CLOCK_BASE_NS - 30 * NS_PER_MS),
        ):
            with self.subTest(field=field):
                trace = copy.deepcopy(self.short)
                event = matching_events(trace, "audio_delta")[0]
                if field == "event_sequence":
                    event["body"][field] = value
                else:
                    event[field] = value
                with self.assertRaisesRegex(ValueError, "sequence|order|clock"):
                    evaluate_attempt(trace)

    def test_O7_rejects_missing_duplicate_or_noncompleted_terminal(self) -> None:
        for fault in ("missing", "duplicate", "cancelled"):
            with self.subTest(fault=fault):
                trace = copy.deepcopy(self.short)
                terminal = matching_events(trace, "utterance_final")[0]
                if fault == "missing":
                    trace["events"].remove(terminal)
                elif fault == "duplicate":
                    trace["events"].insert(-1, copy.deepcopy(terminal))
                else:
                    terminal["body"]["outcome"] = "UTTERANCE_OUTCOME_CANCELLED"
                with self.assertRaises(ValueError):
                    evaluate_attempt(trace)

    def test_O7_rejects_empty_text_silent_or_malformed_pcm(self) -> None:
        for kind, field, value in (
            ("transcript_delta", "text", " "),
            ("translation_delta", "text", ""),
            ("audio_delta", "nonzero", False),
            ("audio_delta", "byte_count", 2),
        ):
            with self.subTest(kind=kind, field=field):
                trace = copy.deepcopy(self.short)
                for event in matching_events(trace, kind):
                    event["body"][field] = value
                with self.assertRaisesRegex(ValueError, "text|silent|PCM|audio"):
                    evaluate_attempt(trace)

    def test_O7_rejects_final_audio_sequence_or_terminal_order_mismatch(self) -> None:
        for fault in ("last_audio", "terminal_order"):
            with self.subTest(fault=fault):
                trace = copy.deepcopy(self.short)
                if fault == "last_audio":
                    matching_events(trace, "utterance_final")[0]["body"][
                        "final_audio_sequence"
                    ] = 0
                else:
                    trace["events"][-3], trace["events"][-2] = (
                        trace["events"][-2],
                        trace["events"][-3],
                    )
                    for index, event in enumerate(trace["events"]):
                        event["body"]["event_sequence"] = index
                        event["received_ns"] = CLOCK_BASE_NS + index * 100 * NS_PER_MS
                with self.assertRaisesRegex(ValueError, "terminal|sequence|order"):
                    evaluate_attempt(trace)

    def test_O8_rejects_unpaced_burst_submission(self) -> None:
        for frame in self.long["sent_frames"]:
            frame["sent_ns"] = CLOCK_BASE_NS + frame["sequence"] * NS_PER_MS
        with self.assertRaisesRegex(ValueError, "pacing|100|interval"):
            self.check()

    def test_O8_rejects_input_sequence_or_end_of_utterance_faults(self) -> None:
        for fault in ("sequence", "missing_end", "early_end"):
            with self.subTest(fault=fault):
                trace = copy.deepcopy(self.short)
                if fault == "sequence":
                    trace["sent_frames"][1]["sequence"] = 0
                elif fault == "missing_end":
                    trace["sent_frames"][-1]["end_of_utterance"] = False
                else:
                    trace["sent_frames"][0]["end_of_utterance"] = True
                with self.assertRaisesRegex(ValueError, "sequence|end|EOU"):
                    evaluate_attempt(trace)


if __name__ == "__main__":
    unittest.main()

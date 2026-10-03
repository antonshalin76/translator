"""Check Rust-produced synthetic NDJSON through the unchanged Task 7 consumers."""

from __future__ import annotations

import json
import sys

from translator_sidecar.benchmark.task7 import BenchmarkDirection
from translator_sidecar.benchmark.task7_e2e import (
    BridgeEvent,
    BridgeEventStream,
    Task7E2EError,
)


def main() -> None:
    count = 0
    for line in sys.stdin:
        raw = json.loads(line)
        event = BridgeEvent.parse(line)
        assert event.raw == raw
        assert event.event == raw["event"]
        assert event.monotonic_ns == raw["monotonic_ns"]
        assert event.utterance_id == raw.get("utterance_id")
        assert event.sequence == raw.get("sequence")
        assert event.queue_lag_ms == raw.get("queue_lag_ms")
        assert event.error_code == raw.get("code")
        assert event.retryable == raw.get("retryable")
        assert event.terminal_outcome == raw.get("outcome")
        assert event.restart_attempt == raw.get("attempt")
        assert event.provider_latency_ms == raw.get(
            "provider_total_ms", raw.get("tts_first_audio_ms")
        )
        direction = {
            "microphone": BenchmarkDirection.RU_TO_EN,
            "speaker": BenchmarkDirection.EN_TO_RU,
            None: None,
        }[raw.get("direction")]
        assert event.direction == direction
        stream = BridgeEventStream([line])
        try:
            if event.event == "failure":
                try:
                    stream.next_global({"ready"}, timeout_s=1)
                except Task7E2EError as error:
                    assert str(error) == (
                        f"bridge failed: {raw['stage']}:{raw['code']}"
                    )
                else:
                    raise AssertionError("failure did not interrupt stream wait")
            elif event.event == "generation_restart":
                for leg in BenchmarkDirection:
                    assert (
                        stream.next_for(
                            leg,
                            {"speech_started"},
                            timeout_s=1,
                            after_restart_generation=0,
                        )
                        == event
                    )
            elif direction is None:
                assert stream.next_global({event.event}, timeout_s=1) == event
            else:
                assert stream.next_for(direction, {event.event}, timeout_s=1) == event
        finally:
            stream._reader.join(timeout=1)
            assert not stream._reader.is_alive()
        count += 1
    assert count > 0
    print(count)


if __name__ == "__main__":
    main()

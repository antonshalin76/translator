"""Prove voiced overlap through one privately launched packaged sidecar.

Config is JSON with an ``environment`` mapping. Fixtures is JSON with a
``fixtures`` mapping containing ru_short, en_long, ru_long, and en_short;
each entry supplies wav, sha256, provenance, and reference. This measures
authenticated transport and inference, not physical acoustic product quality.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import inspect
import io
import json
import os
import signal
import sys
import tempfile
import time
import traceback
import wave
from itertools import pairwise
from pathlib import Path
from uuid import uuid4

FRAME_BYTES = 3200
MODELS = {
    "MODEL_KIND_ASR": ("faster-whisper-large-v3-turbo", "COMPUTE_DEVICE_CUDA"),
    "MODEL_KIND_MT": ("hy-mt2-1.8b-gguf-q4-k-m", "COMPUTE_DEVICE_CUDA"),
    "MODEL_KIND_TTS": ("piper-medium", "COMPUTE_DEVICE_CPU"),
}
DIRECTIONS = {"ru": "AUDIO_DIRECTION_MICROPHONE", "en": "AUDIO_DIRECTION_SPEAKER"}
ENVIRONMENT_KEYS = {
    "TRANSLATOR_ASR_MODEL_ID",
    "TRANSLATOR_MT_MODEL_ID",
    "TRANSLATOR_MODEL_CACHE_ROOT",
    "TRANSLATOR_CUDA_LIBRARY_PATH",
    "CUDA_VISIBLE_DEVICES",
}
UTTERANCE_EVENTS = {
    "transcript_delta",
    "translation_delta",
    "audio_delta",
    "latency",
    "utterance_final",
}


def checked_health(body: dict) -> None:
    models = {model["kind"]: model for model in body.get("models", [])}
    if (
        body.get("provider_id") != "PROVIDER_ID_LOCAL"
        or body.get("state") != "PROVIDER_STATE_READY"
        or body.get("safe_error")
        or body.get("retry")
        or len(models) != len(body.get("models", []))
        or models.keys() != MODELS.keys()
    ):
        raise ValueError("provider health is not the pinned ready local chain")
    for kind, (model_id, device) in MODELS.items():
        model = models[kind]
        if (
            model.get("id") != model_id
            or model.get("device") != device
            or model.get("state") != "MODEL_STATE_READY"
            or model.get("safe_error_code")
        ):
            raise ValueError("model health differs: fallback, device, or readiness")


def evaluate_attempt(trace: dict) -> dict:
    """Check the observed transport transcript for exactly one completed utterance."""
    frames, events = trace["sent_frames"], trace["events"]
    if (
        not frames
        or not events
        or trace["direction_id"] != DIRECTIONS[trace["language"]]
    ):
        raise ValueError("missing input or direction identity")
    if [frame["sequence"] for frame in frames] != list(range(len(frames))):
        raise ValueError("input sequence differs")
    if [frame["end_of_utterance"] for frame in frames] != [
        *([False] * (len(frames) - 1)),
        True,
    ]:
        raise ValueError("end of utterance must occur once on the final input")
    times = [frame["sent_ns"] for frame in frames]
    if any(
        not 80_000_000 <= later - earlier <= 250_000_000
        for earlier, later in pairwise(times)
    ):
        raise ValueError("100 ms input pacing differs")
    if not any(frame["nonzero"] for frame in frames):
        raise ValueError("input has no nonzero speech")
    sequences, received = [], []
    for event in events:
        kind, body = event["kind"], event["body"]
        fields = ("session_id", "direction_id")
        if kind in UTTERANCE_EVENTS:
            fields += ("stream_id", "utterance_id")
            if event["received_ns"] < times[-1]:
                raise ValueError(
                    "utterance receive clock precedes observed end of input"
                )
        if any(body.get(field) != trace[field] for field in fields):
            raise ValueError(
                "event identity crossed a session, direction, or utterance"
            )
        sequences.append(int(body["event_sequence"]))
        received.append(event["received_ns"])
        if kind == "health":
            checked_health(body)
        elif kind not in UTTERANCE_EVENTS | {"session_opened", "session_closed"}:
            raise ValueError("unexpected provider event or error")
    if sequences != sorted(set(sequences)) or received != sorted(received):
        raise ValueError("event sequence or receive clock order differs")
    kinds = [event["kind"] for event in events]
    if (
        kinds[:2] != ["session_opened", "health"]
        or kinds.count("session_opened") != 1
        or kinds.count("session_closed") != 1
        or kinds[-1] != "session_closed"
    ):
        raise ValueError("session open/close terminal order differs")
    if events[-1]["body"].get("reason") != "SESSION_CLOSE_REASON_USER_STOP":
        raise ValueError("session close terminal reason differs")

    def by_kind(kind):
        return [event for event in events if event["kind"] == kind]

    final, audio, latency = (
        by_kind(kind) for kind in ("utterance_final", "audio_delta", "latency")
    )
    if (
        len(final) != 1
        or final[0]["body"]["outcome"] != "UTTERANCE_OUTCOME_COMPLETED"
        or not audio
        or len(audio) > 1500
        or len(latency) != 1
    ):
        raise ValueError("utterance terminal or audio count differs")
    texts = {}
    for kind in ("transcript_delta", "translation_delta"):
        text = [
            event["body"]["text"]
            for event in by_kind(kind)
            if event["body"].get("is_final")
        ]
        if len(text) != 1 or not text[0].strip():
            raise ValueError("missing or empty final text")
        texts[kind] = text[0]
    for event in audio:
        body = event["body"]
        if (
            body.get("sample_rate_hz") != 24_000
            or body.get("channels") != 1
            or body.get("sample_format") != "SAMPLE_FORMAT_S16LE"
            or body.get("frame_duration_ms") != 20
            or body.get("byte_count") != 960
        ):
            raise ValueError("malformed product PCM audio")
    nonzero = [event for event in audio if event["body"]["nonzero"]]
    if not nonzero:
        raise ValueError("silent output PCM audio")
    if nonzero[0]["received_ns"] < times[0]:
        raise ValueError("output receive clock precedes input submission")
    utterance = [event for event in events if event["kind"] in UTTERANCE_EVENTS]
    if (
        [int(event["body"]["sequence"]) for event in audio] != list(range(len(audio)))
        or int(final[0]["body"].get("final_audio_sequence", -1)) != len(audio) - 1
        or utterance[-2:] != [latency[0], final[0]]
    ):
        raise ValueError("audio sequence or utterance terminal order differs")
    return {
        "status": "completed",
        "language": trace["language"],
        "first_pcm_ns": nonzero[0]["received_ns"],
        "pcm_duration_ms": len(audio) * 20,
        "asr_text": texts["transcript_delta"],
        "mt_text": texts["translation_delta"],
    }


def evaluate_overlap(left: dict, right: dict, *, short_language: str) -> dict:
    """Require later observed opposite speech after short-side nonzero output."""
    traces = {trace["language"]: trace for trace in (left, right)}
    if traces.keys() != DIRECTIONS.keys() or any(
        left[field] == right[field]
        for field in ("session_id", "stream_id", "utterance_id")
    ):
        raise ValueError("overlap requires two distinct direction identities")
    results = {language: evaluate_attempt(trace) for language, trace in traces.items()}
    starts = [trace["sent_frames"][0]["sent_ns"] for trace in traces.values()]
    ends = [trace["sent_frames"][-1]["sent_ns"] for trace in traces.values()]
    if max(starts) >= min(ends):
        raise ValueError("actual send intervals do not overlap")
    opposite = traces["en" if short_language == "ru" else "ru"]
    if len(traces[short_language]["sent_frames"]) >= len(opposite["sent_frames"]):
        raise ValueError(
            "short speech must have fewer input frames than opposite speech"
        )
    first_pcm = results[short_language]["first_pcm_ns"]
    future = [
        frame
        for frame in opposite["sent_frames"]
        if frame["nonzero"] and frame["sent_ns"] > first_pcm
    ]
    if not future:
        raise ValueError("no later observed opposite nonzero speech frame")
    return {
        "status": "pass",
        "short_language": short_language,
        "send_overlap_ms": (min(ends) - max(starts)) / 1_000_000,
        "short_first_pcm_ns": first_pcm,
        "opposite_future_speech_frame": future[0],
        "directions": results,
    }


def digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def frozen_json(path: Path) -> tuple[dict, str]:
    data = path.read_bytes()
    return json.loads(data), hashlib.sha256(data).hexdigest()


def read_fixture(row: dict) -> bytes:
    path = Path(row["wav"])
    if (
        not path.is_absolute()
        or any(part.is_symlink() for part in (path, *path.parents))
        or not row.get("provenance")
        or not row.get("reference")
        or ".failed-short" in str(path) + row["provenance"]
    ):
        raise ValueError("fixture path or speech provenance is unsupported")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != row["sha256"]:
        raise ValueError("frozen WAV hash changed before submission")
    with wave.open(io.BytesIO(data), "rb") as source:
        if (
            source.getnchannels() != 1
            or source.getsampwidth() != 2
            or source.getframerate() != 16_000
            or source.getcomptype() != "NONE"
        ):
            raise ValueError("fixture must be mono 16 kHz s16le WAV")
        pcm = source.readframes(source.getnframes())
    if not FRAME_BYTES <= len(pcm) < 30 * 32_000 or not any(pcm):
        raise ValueError("fixture must contain speech and be shorter than 30 seconds")
    return pcm


def resource_limits() -> dict:
    membership = Path("/proc/self/cgroup").read_text().strip().splitlines()
    relative = next(
        line.split(":", 2)[2] for line in membership if line.startswith("0::")
    )
    root = Path("/sys/fs/cgroup")
    scope = (root / relative.lstrip("/")).resolve()
    if not scope.is_relative_to(root):
        raise ValueError("cannot resolve the owned cgroup")
    memory = (scope / "memory.max").read_text().strip()
    swap = (scope / "memory.swap.max").read_text().strip()
    quota, period = (scope / "cpu.max").read_text().split()
    if (
        "max" in (memory, swap, quota)
        or int(memory) > 8 * 1024**3
        or int(swap) > 256 * 1024**2
        or int(quota) > 2 * int(period)
    ):
        raise ValueError(
            "run requires a cgroup capped at 8 GiB RAM, 256 MiB swap, 2 CPUs"
        )
    return {
        "scope": str(scope),
        "memory_max": int(memory),
        "swap_max": int(swap),
        "cpu_quota": int(quota),
        "cpu_period": int(period),
    }


def probe_request(pb):
    return pb.ProviderProbeRequest(
        schema_version="translator.provider.probe_request.v1"
    )


def open_request(pb, identity: dict):
    language = identity["language"]
    source, target = (
        (pb.LANGUAGE_RU, pb.LANGUAGE_EN)
        if language == "ru"
        else (pb.LANGUAGE_EN, pb.LANGUAGE_RU)
    )
    return pb.ProviderRequest(
        open_session=pb.OpenProviderSession(
            schema_version="translator.provider.open_session.v1",
            session_id=identity["session_id"],
            direction_id=pb.AudioDirection.Value(identity["direction_id"]),
            provider_id=pb.PROVIDER_ID_LOCAL,
            source_language=source,
            target_language=target,
            mode=pb.TRANSLATION_MODE_QUALITY_FIRST,
            debug_text_enabled=True,
            requested_input_format=pb.PcmFormat(
                sample_rate_hz=16_000,
                channels=1,
                sample_format=pb.SAMPLE_FORMAT_S16LE,
                frame_duration_ms=100,
            ),
            requested_output_format=pb.PcmFormat(
                sample_rate_hz=24_000,
                channels=1,
                sample_format=pb.SAMPLE_FORMAT_S16LE,
                frame_duration_ms=20,
            ),
            voice_profile=pb.VoiceProfile(
                language=target,
                gender=pb.VOICE_GENDER_MALE,
                engine=pb.VOICE_ENGINE_PIPER,
            ),
        )
    )


async def run(arguments) -> dict:
    os.umask(0o077)
    if not arguments.output.is_absolute() or arguments.output.exists():
        raise ValueError("new absolute private output directory required")
    arguments.output.mkdir(mode=0o700, parents=True)
    report = {
        "status": "failed",
        "boundary": "packaged sidecar transport and inference",
        "physical_acoustic_quality": "not_evaluated",
        "release": False,
        "attempts": [],
    }
    process = channel = health_probe = None
    temporary = tempfile.TemporaryDirectory(prefix="translator-overlap-", delete=False)
    runtime = Path(temporary.name)
    socket_path = runtime / "sidecar.sock"
    journal = (arguments.output / "attempts.jsonl").open("x", encoding="utf-8")
    log = (arguments.output / "sidecar.log").open("xb")
    try:
        config, config_hash = frozen_json(arguments.config)
        fixtures, fixtures_hash = frozen_json(arguments.fixtures)
        environment = config["environment"]
        if (
            set(environment) - ENVIRONMENT_KEYS
            or any(not isinstance(value, str) for value in environment.values())
            or environment.get("TRANSLATOR_ASR_MODEL_ID") != MODELS["MODEL_KIND_ASR"][0]
            or environment.get("TRANSLATOR_MT_MODEL_ID") != MODELS["MODEL_KIND_MT"][0]
        ):
            raise ValueError("config must select the pinned offline Turbo-Hy chain")
        rows = fixtures["fixtures"]
        if set(rows) != {"ru_short", "en_long", "ru_long", "en_short"}:
            raise ValueError("four frozen short/long RU/EN fixtures required")
        for language in ("ru", "en"):
            if len(read_fixture(rows[f"{language}_short"])) >= len(
                read_fixture(rows[f"{language}_long"])
            ):
                raise ValueError("long speech must exceed short speech duration")
        sys.path.insert(0, str(arguments.sidecar_root))
        import grpc
        import psutil
        from google.protobuf.json_format import MessageToDict

        from translator_sidecar.generated.translator.provider.v1 import (
            provider_pb2 as pb,
        )
        from translator_sidecar.generated.translator.provider.v1 import (
            provider_pb2_grpc,
        )
        from translator_sidecar.local.cuda_runtime import _DEFAULT_LIBRARY_DIRS
        from translator_sidecar.local.hy_mt import HyMtTranslator

        pinned_files = [
            Path(__file__),
            arguments.config,
            arguments.fixtures,
            arguments.python.resolve(),
            arguments.sidecar_root / "pyproject.toml",
            arguments.sidecar_root / "uv.lock",
            arguments.sidecar_root.parent / "models/manifest.json",
            *sorted((arguments.sidecar_root / "translator_sidecar").rglob("*.py")),
        ]
        venv = arguments.python.parent.parent / "pyvenv.cfg"
        if venv.is_file():
            pinned_files.append(venv)
        for name in ("server_path", "backend_path"):
            pinned_files.append(
                inspect.signature(HyMtTranslator.load).parameters[name].default
            )
        library_dirs = [
            *_DEFAULT_LIBRARY_DIRS,
            *map(
                Path,
                environment.get("TRANSLATOR_CUDA_LIBRARY_PATH", "").split(os.pathsep),
            ),
        ]
        for directory in library_dirs:
            if directory.is_absolute() and directory.is_dir():
                pinned_files.extend(sorted(directory.glob("*.so*")))
        pins = {str(path): digest(path) for path in pinned_files}
        report["identity"] = {
            "config_sha256": config_hash,
            "fixtures_sha256": fixtures_hash,
            "python": str(arguments.python),
            "sidecar_root": str(arguments.sidecar_root),
            "files": pins,
            "fixtures": rows,
            "resource_limits": resource_limits(),
        }
        generation, token = str(uuid4()), os.urandom(32).hex()
        child_environment = {
            "PATH": os.defpath,
            "LANG": "C.UTF-8",
            "HF_HUB_OFFLINE": "1",
            "OMP_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2",
            "TMPDIR": str(runtime),
            **environment,
            "TRANSLATOR_SIDECAR_SOCKET": str(socket_path),
            "TRANSLATOR_SIDECAR_TOKEN": token,
            "TRANSLATOR_SIDECAR_GENERATION": generation,
        }
        process = await asyncio.create_subprocess_exec(
            str(arguments.python),
            "-m",
            "translator_sidecar",
            cwd=arguments.sidecar_root,
            env=child_environment,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
        report["sidecar_pid"] = process.pid
        channel = grpc.aio.insecure_channel(
            f"unix://{socket_path}", options=(("grpc.enable_retries", 0),)
        )
        stub = provider_pb2_grpc.ProviderTransportStub(channel)
        metadata = (("authorization", f"Bearer {token}"),)

        def record(event, trace):
            kind = event.WhichOneof("event")
            body = MessageToDict(
                getattr(event, kind),
                preserving_proto_field_name=True,
                always_print_fields_with_no_presence=True,
            )
            if kind == "audio_delta":
                pcm = event.audio_delta.pcm
                body.pop("pcm", None)
                body.update(
                    byte_count=len(pcm),
                    pcm_sha256=hashlib.sha256(pcm).hexdigest(),
                    nonzero=any(pcm),
                )
            trace["events"].append(
                {"kind": kind, "received_ns": time.monotonic_ns(), "body": body}
            )
            return kind, body

        def close_request(trace):
            return pb.ProviderRequest(
                close_session=pb.CloseProviderSession(
                    schema_version="translator.provider.close_session.v1",
                    session_id=trace["session_id"],
                    reason=pb.CLOSE_REQUEST_REASON_USER_STOP,
                )
            )

        async def attempt(case_id):
            language = case_id[:2]
            trace = {
                "case_id": case_id,
                "language": language,
                "direction_id": DIRECTIONS[language],
                "session_id": str(uuid4()),
                "stream_id": str(uuid4()),
                "utterance_id": str(uuid4()),
                "sent_frames": [],
                "events": [],
            }
            call = stub.Stream(metadata=metadata, timeout=arguments.timeout)
            wav_path = arguments.output / f"{case_id}-{trace['session_id']}.wav"
            try:
                pcm = read_fixture(rows[case_id])
                request = open_request(pb, trace)
                await call.write(request)
                for expected in ("session_opened", "health"):
                    event = await call.read()
                    if event is grpc.aio.EOF or record(event, trace)[0] != expected:
                        raise ValueError(
                            "sidecar did not open the authenticated session"
                        )
                checked_health(trace["events"][-1]["body"])
                with wave.open(str(wav_path), "wb") as output:
                    output.setnchannels(1)
                    output.setsampwidth(2)
                    output.setframerate(24_000)

                    async def send():
                        chunks = range(0, len(pcm), FRAME_BYTES)
                        for sequence, offset in enumerate(chunks):
                            original = pcm[offset : offset + FRAME_BYTES]
                            final = offset + FRAME_BYTES >= len(pcm)
                            captured = time.monotonic_ns()
                            await call.write(
                                pb.ProviderRequest(
                                    input_frame=pb.ProviderInputFrame(
                                        schema_version="translator.provider.input.v1",
                                        session_id=trace["session_id"],
                                        direction_id=request.open_session.direction_id,
                                        stream_id=trace["stream_id"],
                                        utterance_id=trace["utterance_id"],
                                        sequence=sequence,
                                        capture_monotonic_ns=captured,
                                        sample_rate_hz=16_000,
                                        channels=1,
                                        sample_format=pb.SAMPLE_FORMAT_S16LE,
                                        frame_duration_ms=100,
                                        source_language=request.open_session.source_language,
                                        target_language=request.open_session.target_language,
                                        mode=request.open_session.mode,
                                        pcm=original.ljust(FRAME_BYTES, b"\0"),
                                        end_of_utterance=final,
                                    )
                                )
                            )
                            trace["sent_frames"].append(
                                {
                                    "sequence": sequence,
                                    "sent_ns": time.monotonic_ns(),
                                    "nonzero": any(original),
                                    "end_of_utterance": final,
                                }
                            )
                            if not final:
                                await asyncio.sleep(0.1)

                    async def receive():
                        while True:
                            event = await call.read()
                            if event is grpc.aio.EOF:
                                raise ValueError(
                                    "transport EOF before utterance terminal"
                                )
                            kind, _ = record(event, trace)
                            if kind == "audio_delta":
                                output.writeframesraw(event.audio_delta.pcm)
                            if kind == "error":
                                raise ValueError(
                                    "provider inference error; no inference retry"
                                )
                            if kind == "utterance_final":
                                return

                    async with asyncio.TaskGroup() as group:
                        group.create_task(send())
                        group.create_task(receive())
                    await call.write(close_request(trace))
                    await call.done_writing()
                    while (event := await call.read()) is not grpc.aio.EOF:
                        record(event, trace)
                    trace["transport_eof"] = True
                trace["output_wav"] = str(wav_path)
                trace["output_wav_sha256"] = digest(wav_path)
                trace["result"] = evaluate_attempt(trace)
                return trace
            except BaseException as error:
                trace["error_type"] = type(error).__name__
                trace["error"] = "".join(traceback.format_exception(error, limit=4))
                raise
            finally:
                call.cancel()
                journal.write(json.dumps(trace, ensure_ascii=False) + "\n")
                journal.flush()
                os.fsync(journal.fileno())
                report["attempts"].append(
                    {
                        "case_id": case_id,
                        "session_id": trace["session_id"],
                        "status": trace.get("result", {}).get("status", "failed"),
                    }
                )

        async def terminal_health():
            observed = []
            report["terminal_health"] = observed
            for language in ("ru", "en"):
                identity = {
                    "language": language,
                    "direction_id": DIRECTIONS[language],
                    "session_id": str(uuid4()),
                }
                call = stub.Stream(metadata=metadata, timeout=5)
                try:
                    await call.write(open_request(pb, identity))
                    opened, health = await call.read(), await call.read()
                    if (
                        opened is grpc.aio.EOF
                        or health is grpc.aio.EOF
                        or not health.HasField("health")
                    ):
                        raise ValueError("terminal provider health missing")
                    body = MessageToDict(
                        health.health, preserving_proto_field_name=True
                    )
                    observed.append(body)
                    await call.write(close_request(identity))
                    await call.done_writing()
                    closed = await call.read()
                    if (
                        closed is grpc.aio.EOF
                        or not closed.HasField("session_closed")
                        or await call.read() is not grpc.aio.EOF
                    ):
                        raise ValueError("terminal health stream did not close at EOF")
                finally:
                    call.cancel()
            for body in observed:
                checked_health(body)
            return observed

        health_probe = terminal_health
        task = asyncio.current_task()
        loop = asyncio.get_running_loop()
        for caught_signal in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(caught_signal, task.cancel)
        async with asyncio.timeout(arguments.timeout - 15):
            await channel.channel_ready()
            while True:
                if process.returncode is not None:
                    raise ValueError("packaged sidecar exited during bootstrap")
                probe = await stub.Probe(
                    probe_request(pb),
                    metadata=metadata,
                    timeout=1,
                )
                if probe.generation_id != generation:
                    raise ValueError("sidecar generation identity differs")
                if probe.provider_ready:
                    break
                await asyncio.sleep(0.1)
            report["serial"] = [evaluate_attempt(await attempt(case)) for case in rows]
            report["overlap"] = []
            for ru_case, en_case, short_language in (
                ("ru_short", "en_long", "ru"),
                ("ru_long", "en_short", "en"),
            ):
                async with asyncio.TaskGroup() as group:
                    ru = group.create_task(attempt(ru_case))
                    en = group.create_task(attempt(en_case))
                report["overlap"].append(
                    evaluate_overlap(
                        ru.result(), en.result(), short_language=short_language
                    )
                )
            report["terminal_health"] = await terminal_health()
            if {str(path): digest(path) for path in pinned_files} != pins:
                raise ValueError("frozen input or runtime identity changed during run")
            report["status"] = "pass"
    except (Exception, asyncio.CancelledError) as error:
        report["error_type"] = type(error).__name__
        report["error"] = "".join(traceback.format_exception(error, limit=4))
    finally:
        if health_probe is not None and "terminal_health" not in report:
            try:
                await asyncio.wait_for(health_probe(), timeout=2)
            except (Exception, asyncio.CancelledError) as error:
                report["terminal_health_error"] = type(error).__name__
        if channel is not None:
            await channel.close()
        cleanup = {
            "child_reaped": process is None,
            "group_empty": process is None,
            "forced_kill": False,
        }
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(process.wait(), timeout=10)
            except TimeoutError:
                cleanup["forced_kill"] = True
                os.killpg(process.pid, signal.SIGKILL)
                await asyncio.wait_for(process.wait(), timeout=2)
            cleanup["child_reaped"] = process.returncode is not None
            members = []
            for entry in psutil.process_iter(["pid", "status"]):
                try:
                    if (
                        entry.info["status"] != psutil.STATUS_ZOMBIE
                        and os.getpgid(entry.pid) == process.pid
                    ):
                        members.append(entry)
                except (ProcessLookupError, PermissionError, psutil.NoSuchProcess):
                    pass
            if members:
                cleanup["forced_kill"] = True
                os.killpg(process.pid, signal.SIGKILL)
                _, members = await asyncio.to_thread(
                    psutil.wait_procs, members, timeout=2
                )
            cleanup["group_empty"] = not members
            cleanup["remaining_pids"] = [entry.pid for entry in members]
            cleanup["sidecar_returncode"] = process.returncode
        if cleanup["group_empty"]:
            temporary.cleanup()
        else:
            report["retained_runtime_directory"] = str(runtime)
        cleanup["socket_removed"] = not socket_path.exists()
        if not all(
            cleanup[field]
            for field in ("child_reaped", "group_empty", "socket_removed")
        ):
            report["status"] = "failed"
        report["cleanup"] = cleanup
        journal.close()
        log.close()
        with (arguments.output / "report.json").open("x", encoding="utf-8") as output:
            json.dump(report, output, ensure_ascii=False, indent=2)
            output.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("python", "sidecar-root", "config", "fixtures", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=300)
    arguments = parser.parse_args()
    if not 30 <= arguments.timeout <= 300:
        parser.error("timeout must be between 30 and 300 seconds")
    report = asyncio.run(run(arguments))
    print(
        json.dumps(
            {
                "status": report["status"],
                "report": str(arguments.output / "report.json"),
            }
        )
    )
    raise SystemExit(0 if report["status"] == "pass" else 1)


if __name__ == "__main__":
    main()

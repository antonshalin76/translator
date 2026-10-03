"""Synthetic worker for the real benchmark process/publication boundary."""

import json
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path


def signal_mask():
    return next(
        line.split()[1]
        for line in Path("/proc/thread-self/status").read_text().splitlines()
        if line.startswith("SigBlk:")
    )


def supervisor_startup_probe(mode, control, output):
    from translator_sidecar.benchmark import process_run

    gated = mode in {"startup_hold", "startup_validation"}
    original_start = threading.Thread.start
    original_popen = subprocess.Popen
    original_run = process_run._run_isolated
    original_validate = process_run._validate_report
    workers, threads, owner_masks, startup_handler_matches = [], [], [], []
    before_mask = signal_mask()
    if mode == "policy_custom":

        def custom_handler(_signum, _frame):
            raise AssertionError("custom handler should not run")

        signal.signal(signal.SIGINT, custom_handler)
    elif mode == "policy_ignore":
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    elif mode == "policy_sig_dfl":
        signal.signal(signal.SIGINT, signal.SIG_DFL)
    prior_handler = signal.getsignal(signal.SIGINT)

    def wait_marker(name):
        deadline = time.monotonic() + 8
        while not (control / name).exists():
            if time.monotonic() >= deadline:
                raise AssertionError(f"fixture gate timed out: {name}")
            time.sleep(0.005)

    def start(thread, *args, **kwargs):
        result = original_start(thread, *args, **kwargs)
        if thread.name.startswith(("ThreadPoolExecutor", "asyncio")):
            threads.append(thread)
            startup_handler_matches.append(
                signal.getsignal(signal.SIGINT) is prior_handler
            )
            if gated:
                assert threading.current_thread() is threading.main_thread()
                assert thread.is_alive()
                (control / "startup_held").touch()
                tick = 0
                deadline = time.monotonic() + 8
                while not (control / "release_startup").exists():
                    tick += 1
                    pending = control / "tick.tmp"
                    pending.write_text(str(tick))
                    pending.replace(control / "tick")
                    if time.monotonic() >= deadline:
                        raise AssertionError("fixture startup gate timed out")
                    time.sleep(0.005)
        return result

    def popen(command, **kwargs):
        worker_mode = "probe_hold" if mode == "startup_hold" else "probe_ok"
        if mode == "policy_error":
            worker_mode = "probe_error"
        elif mode == "policy_podcast_sigint":
            assert command[-3] == "podcast"
            worker_mode = "probe_podcast_hold"
        worker = original_popen(
            [sys.executable, __file__, worker_mode, *command[-2:], str(control)],
            **kwargs,
        )
        workers.append(worker)
        stat = Path(f"/proc/{worker.pid}/stat").read_text()
        pending = control / "worker_pid.tmp"
        pending.write_text(
            json.dumps(
                {
                    "pid": worker.pid,
                    "start": stat.rsplit(")", 1)[1].split()[19],
                }
            )
        )
        pending.replace(control / "worker_pid")
        return worker

    def observed_run(*args):
        owner_masks.append(signal_mask())
        try:
            return original_run(*args)
        finally:
            (control / "supervisor_finished").touch()

    def validate(*args):
        result = original_validate(*args)
        if mode == "startup_validation":
            (control / "validation_held").touch()
            wait_marker("release_validation")
        return result

    def invoke():
        try:
            if mode == "policy_podcast_sigint":
                from translator_sidecar.benchmark import podcast_quality

                def convert(**kwargs):
                    directory = kwargs["work_dir"]
                    directory.mkdir(parents=True)
                    pcm = b"\0\0" * 16000
                    (directory / "source.s16le").write_bytes(pcm)
                    return pcm, {"fixture": "synthetic"}

                def no_parent_provider(*_args, **_kwargs):
                    raise AssertionError("parent provider construction")

                podcast_quality._load_audio_pcm = convert
                podcast_quality.build_local_provider = no_parent_provider
                podcast_quality.main(
                    [
                        "--ru-audio",
                        "synthetic-ru",
                        "--en-audio",
                        "synthetic-en",
                        "--work-dir",
                        str(control / "prepared"),
                        "--output",
                        str(output),
                        "--model-run-seconds",
                        "10",
                        "--terminate-grace-seconds",
                        "2",
                    ]
                )
                return "success"
            process_run.run_benchmark(
                "task6",
                {},
                output,
                limits=process_run.RunLimits(
                    model_run_seconds=10,
                    terminate_grace_seconds=0.25,
                ),
            )
        except BaseException as error:
            return type(error).__name__
        return "success"

    threading.Thread.start = start
    process_run.subprocess.Popen = popen
    process_run._run_isolated = observed_run
    process_run._validate_report = validate
    try:
        if mode == "policy_thread":
            results = []
            caller = threading.Thread(target=lambda: results.append(invoke()))
            caller.start()
            caller.join(timeout=8)
            assert not caller.is_alive()
            result = results[0]
        else:
            result = invoke()
        reaped_at_return = bool(workers) and all(
            item.returncode is not None for item in workers
        )
        joined_at_return = bool(threads) and all(
            not item.is_alive() for item in threads
        )
        native_absent_at_return = bool(threads) and all(
            not Path(f"/proc/self/task/{item.native_id}").exists() for item in threads
        )
        if reaped_at_return and joined_at_return:
            # CPython releases its join lock just before the kernel removes the
            # native task entry. Observe that final bookkeeping without repair.
            deadline = time.monotonic() + 1
            while any(
                Path(f"/proc/self/task/{item.native_id}").exists() for item in threads
            ):
                assert time.monotonic() < deadline
                time.sleep(0.005)
        snapshot = {
            "outcome": result,
            "reaped": reaped_at_return,
            "threads_retired": joined_at_return,
            "native_threads_retired_at_return": native_absent_at_return,
            "native_threads_retired": bool(threads)
            and all(
                not Path(f"/proc/self/task/{item.native_id}").exists()
                for item in threads
            ),
            "handler_restored": signal.getsignal(signal.SIGINT) is prior_handler,
            "startup_handler_matches": startup_handler_matches,
            "main_mask_before": before_mask,
            "main_mask_after": signal_mask(),
            "owner_masks": owner_masks,
            "worker_count": len(workers),
            "thread_count": len(threads),
        }
        # Freeze observations before independently repairing any owner; later
        # fixture signals must not interrupt that repair or alter the receipt.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        pending = control / "outcome.tmp"
        pending.write_text(json.dumps(snapshot))
        pending.replace(control / "outcome")
        wait_marker("allow_cleanup")
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        for name in ("release", "release_startup", "release_validation"):
            (control / name).touch()
        for worker in workers:
            if worker.poll() is None:
                worker.kill()
            worker.wait(timeout=5)
        for thread in threads:
            thread.join(timeout=5)
            deadline = time.monotonic() + 5
            while Path(f"/proc/self/task/{thread.native_id}").exists():
                assert time.monotonic() < deadline
                time.sleep(0.005)
            assert not thread.is_alive()
        threading.Thread.start = original_start
        signal.signal(signal.SIGINT, prior_handler)
    return 0


def report():
    return {
        "schema_version": "translator.task6-benchmark.v2",
        "generated_at_unix_ns": 1,
        "environment": {},
        "fixture": {},
        "asr_candidates": [],
        "voice_profiles": [],
        "quality": {"passes_thresholds": False},
        "duplex_candidates": [],
        "normal_runtime": {},
    }


def main():
    mode, request_path, staging_path, control_path = sys.argv[1:]
    control = Path(control_path)
    if mode.startswith("startup_") or mode.startswith("policy_"):
        return supervisor_startup_probe(mode, control, Path(staging_path))
    if mode.startswith("probe_"):
        (control / "child_mask").write_text(signal_mask())
        mode = {
            "probe_hold": "signal_hold",
            "probe_ok": "ok",
            "probe_error": "operation_error",
            "probe_podcast_hold": "podcast_hold",
        }[mode]
    from translator_sidecar.benchmark import process_run

    if mode == "podcast_hold":
        import asyncio

        from translator_sidecar.benchmark import podcast_quality

        signal.signal(signal.SIGTERM, lambda *_: (control / "term").touch())

        async def held_podcast(request, *, fatal_cleanup):
            (control / "entered").touch()
            try:
                while not (control / "release").exists():
                    await asyncio.sleep(0.01)
                return {
                    "schema_version": "translator.podcast-quality-debug.v1",
                    "generated_at": "fixture",
                    "inputs": request["inputs"],
                    "candidate_matrix": {},
                    "models": [],
                }
            finally:
                (control / "cleanup").touch()

        podcast_quality._run_owned = held_podcast
        process_run._worker_main("podcast", Path(request_path), Path(staging_path))
        return 0

    if mode == "supervisor_sigint":
        original_popen = subprocess.Popen
        workers = []

        def popen(command, **kwargs):
            worker = original_popen(
                [
                    sys.executable,
                    __file__,
                    "signal_hold",
                    str(command[-2]),
                    str(command[-1]),
                    str(control),
                ],
                **kwargs,
            )
            workers.append(worker)
            stat = Path(f"/proc/{worker.pid}/stat").read_text()
            record = {"pid": worker.pid, "start": stat.rsplit(")", 1)[1].split()[19]}
            pending = control / "worker_pid.tmp"
            pending.write_text(json.dumps(record))
            pending.replace(control / "worker_pid")
            return worker

        process_run.subprocess.Popen = popen
        try:
            process_run.run_benchmark(
                "task6",
                {},
                Path(staging_path),
                limits=process_run.RunLimits(
                    model_run_seconds=10,
                    terminate_grace_seconds=0.25,
                ),
            )
        except KeyboardInterrupt:
            assert len(workers) == 1
            assert workers[0].returncode is not None
            assert not Path(f"/proc/{workers[0].pid}").exists()
            (control / "interrupt_joined").touch()
            return 0
        finally:
            (control / "release").touch()
            for worker in workers:
                if worker.poll() is None:
                    worker.kill()
                worker.wait(timeout=5)
        return 1

    from translator_sidecar.benchmark import task6_live

    def body(*, fatal_cleanup):
        if mode == "ignore_term":
            signal.signal(signal.SIGTERM, lambda *_: (control / "term").touch())
        if mode == "signal_hold":
            signal.signal(signal.SIGINT, lambda *_: (control / "worker_sigint").touch())
            signal.signal(signal.SIGTERM, lambda *_: (control / "term").touch())
            if (control / "hold_before_entry").exists():
                (control / "before_entry").touch()
                while not (control / "release").exists():
                    time.sleep(0.01)
        (control / "entered").touch()
        try:
            if mode == "fatal":
                fatal_cleanup(RuntimeError("private-worker-error"))
            if mode in {"hold", "ignore_term", "signal_hold"}:
                while not (control / "release").exists():
                    time.sleep(0.01)
            if mode == "operation_error":
                raise RuntimeError("private-worker-error")
            payload = report()
            if mode == "wrong_schema":
                payload["schema_version"] = "wrong"
            elif mode == "missing_field":
                payload.pop("quality")
            elif mode == "extra_field":
                payload["unexpected"] = "private-worker-error"
            elif mode == "wrong_type":
                payload["voice_profiles"] = "private-worker-error"
            elif mode == "nan":
                payload["quality"]["value"] = float("nan")
            return payload
        finally:
            (control / "cleanup").touch()

    task6_live._run_owned = body
    try:
        process_run._worker_main("task6", Path(request_path), Path(staging_path))
    except BaseException:
        return 70
    if mode == "truncated":
        Path(staging_path).write_text('{"schema_version":')
    if mode == "wrong_root":
        Path(staging_path).write_text(json.dumps([]))
    return 7 if mode == "nonzero" else 0


if __name__ == "__main__":
    raise SystemExit(main())

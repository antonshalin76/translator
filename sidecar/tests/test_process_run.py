from __future__ import annotations

import asyncio
import gc
import importlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

_FIXTURE = Path(__file__).parent / "fixtures" / "benchmark_process_child.py"


def boundary():
    # Keep collection possible before the new boundary exists: missing API is
    # zero executed contracts in RED, not proof of a lifecycle defect.
    return importlib.import_module("translator_sidecar.benchmark.process_run")


class WorkerFixture:
    def __init__(self, tmp_path, monkeypatch, mode="ok"):
        self.module = boundary()
        from translator_sidecar.benchmark import task6_live

        monkeypatch.setattr(
            task6_live,
            "_run_owned",
            lambda **_kwargs: pytest.fail("parent model body"),
        )
        self.control = tmp_path / "control"
        self.control.mkdir()
        self.output = tmp_path / "report.json"
        self.output.write_bytes(b"previous report")
        self.workers = []
        self.commands = []
        self.owner_threads = []
        original_popen = subprocess.Popen

        def popen(command, **kwargs):
            self.commands.append((command, kwargs))
            self.owner_threads.append(threading.current_thread())
            assert kwargs["start_new_session"] is True
            assert kwargs["close_fds"] is True
            assert kwargs.get("shell", False) is False
            assert all(
                kwargs[name] == subprocess.DEVNULL
                for name in (
                    "stdin",
                    "stdout",
                    "stderr",
                )
            )
            assert command[:3] == [
                sys.executable,
                "-m",
                "translator_sidecar.benchmark.process_run",
            ]
            assert command[-3] == "task6"
            assert json.loads(Path(command[-2]).read_text()) == {}
            assert Path(command[-1]).parent.stat().st_mode & 0o777 == 0o700
            worker = original_popen(
                [
                    sys.executable,
                    str(_FIXTURE),
                    mode,
                    *map(str, command[-2:]),
                    str(self.control),
                ],
                **kwargs,
            )
            self.workers.append(worker)
            return worker

        monkeypatch.setattr(self.module.subprocess, "Popen", popen)

    def run(self, **limits):
        return self.module.run_benchmark(
            "task6",
            {},
            self.output,
            limits=self.module.RunLimits(
                model_run_seconds=limits.get("model_run_seconds", 10),
                terminate_grace_seconds=0.1,
            ),
        )

    async def run_async(self):
        return await self.module.run_benchmark_async(
            "task6",
            {},
            self.output,
            limits=self.module.RunLimits(
                model_run_seconds=10,
                terminate_grace_seconds=0.1,
            ),
        )

    def assert_reaped(self):
        assert len(self.workers) == 1
        worker = self.workers[0]
        assert worker.returncode is not None
        assert not Path(f"/proc/{worker.pid}").exists()
        assert all(
            thread is not threading.main_thread() for thread in self.owner_threads
        )
        command = self.commands[0][0]
        assert not Path(command[-1]).parent.exists()

    def cleanup(self):
        (self.control / "release").touch()
        for worker in self.workers:
            if worker.poll() is None:
                worker.kill()
            worker.wait(timeout=5)


async def wait_for(predicate):
    async with asyncio.timeout(5):
        while not predicate():
            await asyncio.sleep(0.005)


async def cleanup_async_fixture(fixture, task):
    fixture.cleanup()
    try:
        done, pending = await asyncio.wait([task], timeout=5)
        assert not pending
        await asyncio.gather(*done, return_exceptions=True)
    finally:
        # Constructor may have returned after the first cleanup snapshot.
        fixture.cleanup()


def test_real_worker_publishes_only_after_cleanup_and_exact_reap(tmp_path, monkeypatch):
    fixture = WorkerFixture(tmp_path, monkeypatch)
    original_replace = os.replace
    commits = []

    def replace(source, destination):
        worker = fixture.workers[0]
        assert worker.returncode == 0 and not Path(f"/proc/{worker.pid}").exists()
        assert (fixture.control / "cleanup").exists()
        assert Path(destination).read_bytes() == b"previous report"
        commits.append(destination)
        return original_replace(source, destination)

    monkeypatch.setattr(fixture.module.os, "replace", replace)
    try:
        result = fixture.run()
        assert result == json.loads(fixture.output.read_text())
        assert result["quality"]["passes_thresholds"] is False
        assert len(commits) == 1
        fixture.assert_reaped()
    finally:
        fixture.cleanup()


@pytest.mark.parametrize(
    "mode",
    [
        "operation_error",
        "fatal",
        "nonzero",
        "truncated",
        "wrong_root",
        "wrong_schema",
        "missing_field",
        "extra_field",
        "wrong_type",
        "nan",
    ],
)
def test_failed_worker_or_invalid_report_never_replaces_previous_file(
    tmp_path,
    monkeypatch,
    mode,
):
    fixture = WorkerFixture(tmp_path, monkeypatch, mode)
    try:
        with pytest.raises(fixture.module.BenchmarkRunError) as captured:
            fixture.run()
        assert "private-worker-error" not in "".join(
            traceback.format_exception(captured.value),
        )
        assert fixture.output.read_bytes() == b"previous report"
        assert (fixture.control / "cleanup").exists() is (mode != "fatal")
        fixture.assert_reaped()
    finally:
        fixture.cleanup()


def test_deadline_escalates_ignored_term_and_reaps_exact_worker(tmp_path, monkeypatch):
    fixture = WorkerFixture(tmp_path, monkeypatch, "ignore_term")
    try:
        with pytest.raises(fixture.module.BenchmarkRunError):
            fixture.run(model_run_seconds=3)
        assert (fixture.control / "entered").exists()
        assert (fixture.control / "term").exists()
        assert not (fixture.control / "cleanup").exists()
        assert fixture.workers[0].returncode == -signal.SIGKILL
        assert fixture.output.read_bytes() == b"previous report"
        fixture.assert_reaped()
    finally:
        fixture.cleanup()


def test_replace_failure_preserves_old_report_after_worker_reap(tmp_path, monkeypatch):
    fixture = WorkerFixture(tmp_path, monkeypatch)

    def fail_replace(*_args):
        assert fixture.workers[0].returncode == 0
        raise OSError("private-worker-error")

    monkeypatch.setattr(fixture.module.os, "replace", fail_replace)
    try:
        with pytest.raises(fixture.module.BenchmarkRunError) as captured:
            fixture.run()
        assert "private-worker-error" not in "".join(
            traceback.format_exception(captured.value)
        )
        assert fixture.output.read_bytes() == b"previous report"
        fixture.assert_reaped()
    finally:
        fixture.cleanup()


def test_deadline_is_rechecked_after_report_validation(tmp_path, monkeypatch):
    fixture = WorkerFixture(tmp_path, monkeypatch)
    original = fixture.module._validate_report

    def validate(*args):
        result = original(*args)
        time.sleep(3)
        return result

    monkeypatch.setattr(fixture.module, "_validate_report", validate)
    try:
        with pytest.raises(fixture.module.BenchmarkRunError):
            fixture.run(model_run_seconds=2)
        assert (fixture.control / "cleanup").exists()
        assert fixture.workers[0].returncode == 0
        assert fixture.output.read_bytes() == b"previous report"
        fixture.assert_reaped()
    finally:
        fixture.cleanup()


def test_unconfirmed_reap_keeps_supervisor_and_staging_owned(tmp_path, monkeypatch):
    module = boundary()
    output = tmp_path / "report.json"
    output.write_bytes(b"previous report")
    killed, release = threading.Event(), threading.Event()
    launches, calls = [], []

    class Process:
        returncode = None

        def __init__(self, command, **_kwargs):
            launches.append(command)

        def poll(self):
            return self.returncode

        def terminate(self):
            calls.append("terminate")

        def kill(self):
            calls.append("kill")
            killed.set()

        def wait(self, timeout=None):
            if timeout is not None:
                raise subprocess.TimeoutExpired("fixture", timeout)
            assert release.wait(timeout=5)
            self.returncode = -signal.SIGKILL
            return self.returncode

    monkeypatch.setattr(module.subprocess, "Popen", Process)
    with ThreadPoolExecutor(max_workers=1) as caller:
        future = caller.submit(
            module.run_benchmark,
            "task6",
            {},
            output,
            limits=module.RunLimits(
                model_run_seconds=0.02, terminate_grace_seconds=0.02
            ),
        )
        try:
            assert killed.wait(timeout=2)
            assert not future.done() and calls == ["terminate", "kill"]
            assert len(launches) == 1 and Path(launches[0][-1]).parent.exists()
            assert output.read_bytes() == b"previous report"
            release.set()
            with pytest.raises(module.BenchmarkRunError):
                future.result(timeout=2)
            assert not Path(launches[0][-1]).parent.exists()
        finally:
            release.set()


@pytest.mark.parametrize("after_commit", [False, True])
def test_cancellation_obeys_final_publication_checkpoint(
    tmp_path,
    monkeypatch,
    after_commit,
):
    fixture = WorkerFixture(tmp_path, monkeypatch)
    held, release = threading.Event(), threading.Event()
    if after_commit:
        original = fixture.module.os.replace

        def replace(*args):
            result = original(*args)
            held.set()
            assert release.wait(timeout=5)
            return result

        monkeypatch.setattr(fixture.module.os, "replace", replace)
    else:
        original = fixture.module._validate_report

        def validate(*args):
            result = original(*args)
            held.set()
            assert release.wait(timeout=5)
            return result

        monkeypatch.setattr(fixture.module, "_validate_report", validate)

    async def scenario():
        task = asyncio.create_task(fixture.run_async())
        try:
            await wait_for(held.is_set)
            for _ in range(3):
                task.cancel()
                await asyncio.sleep(0)
                assert not task.done()
            if after_commit:
                committed = fixture.output.read_bytes()
                assert json.loads(fixture.output.read_text())["quality"] == {
                    "passes_thresholds": False,
                }
            else:
                assert fixture.output.read_bytes() == b"previous report"
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            if after_commit:
                assert fixture.output.read_bytes() == committed
                assert json.loads(committed)["quality"] == {"passes_thresholds": False}
            else:
                assert fixture.output.read_bytes() == b"previous report"
            fixture.assert_reaped()
        finally:
            release.set()
            await cleanup_async_fixture(fixture, task)

    asyncio.run(scenario())


def test_repeated_async_cancel_joins_held_worker_before_return(tmp_path, monkeypatch):
    fixture = WorkerFixture(tmp_path, monkeypatch, "ignore_term")

    async def scenario():
        task = asyncio.create_task(fixture.run_async())
        try:
            await wait_for(lambda: (fixture.control / "entered").exists())
            worker = fixture.workers[0]
            for _ in range(3):
                task.cancel()
                await asyncio.sleep(0)
                assert not task.done()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert worker.returncode == -signal.SIGKILL
            assert fixture.output.read_bytes() == b"previous report"
            fixture.assert_reaped()
        finally:
            await cleanup_async_fixture(fixture, task)

    asyncio.run(scenario())


def test_exec_failure_preserves_output_and_closes_startup_descriptors(
    tmp_path, monkeypatch
):
    module = boundary()
    output = tmp_path / "report.json"
    output.write_bytes(b"previous report")
    original_popen = subprocess.Popen

    def fail_exec(command, **kwargs):
        return original_popen([str(tmp_path / "missing-executable")], **kwargs)

    monkeypatch.setattr(module.subprocess, "Popen", fail_exec)
    gc.collect()
    baseline = set(os.listdir("/proc/self/fd"))
    with pytest.raises(module.BenchmarkRunError):
        module.run_benchmark("task6", {}, output)
    assert set(os.listdir("/proc/self/fd")) == baseline
    assert output.read_bytes() == b"previous report"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["report.json"]


def test_async_cancel_during_constructor_retains_same_supervisor(tmp_path, monkeypatch):
    fixture = WorkerFixture(tmp_path, monkeypatch, "hold")
    constructing, release = threading.Event(), threading.Event()
    original = fixture.module.subprocess.Popen

    def delayed_popen(*args, **kwargs):
        constructing.set()
        assert release.wait(timeout=5)
        return original(*args, **kwargs)

    monkeypatch.setattr(fixture.module.subprocess, "Popen", delayed_popen)

    async def scenario():
        task = asyncio.create_task(fixture.run_async())
        try:
            await wait_for(constructing.is_set)
            for _ in range(3):
                task.cancel()
                await asyncio.sleep(0)
                assert not task.done() and not fixture.workers
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert fixture.output.read_bytes() == b"previous report"
            fixture.assert_reaped()
        finally:
            release.set()
            await cleanup_async_fixture(fixture, task)

    asyncio.run(scenario())


@pytest.mark.parametrize("field", ["model_run_seconds", "terminate_grace_seconds"])
@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan")])
def test_invalid_limits_reject_before_any_process_or_path_effect(
    tmp_path,
    monkeypatch,
    field,
    value,
):
    module = boundary()
    monkeypatch.setattr(
        module.subprocess, "Popen", lambda *_a, **_k: pytest.fail("spawn")
    )
    with pytest.raises(ValueError):
        limits = module.RunLimits(**{field: value})
        module.run_benchmark(
            "task6", {}, tmp_path / "absent" / "report.json", limits=limits
        )
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("kind", ["unknown", "task6", "podcast"])
def test_real_module_routing_rejects_missing_request_without_models(tmp_path, kind):
    boundary()
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "translator_sidecar.benchmark.process_run",
            kind,
            str(tmp_path / "missing.json"),
            str(tmp_path / "result.json"),
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=5,
        check=False,
    )
    assert completed.returncode != 0
    assert completed.stdout == b"" and completed.stderr == b""
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("pre_entry_cleanup", [False, True])
def test_sync_interrupt_isolates_worker_and_waits_for_reap(tmp_path, pre_entry_cleanup):
    boundary()
    control = tmp_path / "control"
    control.mkdir()
    if pre_entry_cleanup:
        (control / "hold_before_entry").touch()
    output = tmp_path / "report.json"
    output.write_bytes(b"previous report")
    supervisor = subprocess.Popen(
        [
            sys.executable,
            str(_FIXTURE),
            "supervisor_sigint",
            "unused",
            str(output),
            str(control),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    worker_pid, worker_pidfd = None, None

    def capture_worker():
        nonlocal worker_pid, worker_pidfd
        if worker_pidfd is not None or not (control / "worker_pid").exists():
            return
        record = json.loads((control / "worker_pid").read_text())
        descriptor = os.pidfd_open(record["pid"])
        try:
            stat = Path(f"/proc/{record['pid']}/stat").read_text()
            assert stat.rsplit(")", 1)[1].split()[19] == record["start"]
        except BaseException:
            os.close(descriptor)
            raise
        worker_pid, worker_pidfd = record["pid"], descriptor

    try:
        deadline = time.monotonic() + 5
        marker = "before_entry" if pre_entry_cleanup else "entered"
        while not (control / marker).exists() or worker_pidfd is None:
            capture_worker()
            assert supervisor.poll() is None and time.monotonic() < deadline
            time.sleep(0.005)
        if pre_entry_cleanup:
            assert not (control / "entered").exists()
            assert Path(f"/proc/{worker_pid}").exists()
            return
        assert os.getpgid(supervisor.pid) == supervisor.pid
        assert os.getpgid(worker_pid) == worker_pid != supervisor.pid
        os.killpg(supervisor.pid, signal.SIGINT)
        deadline = time.monotonic() + 2
        while not (control / "term").exists():
            assert supervisor.poll() is None and time.monotonic() < deadline
            time.sleep(0.005)
        for _ in range(2):
            os.kill(supervisor.pid, signal.SIGINT)
            time.sleep(0.01)
            assert supervisor.poll() is None
        assert supervisor.wait(timeout=5) == 0
        assert (control / "interrupt_joined").exists()
        assert not (control / "worker_sigint").exists()
        assert not Path(f"/proc/{worker_pid}").exists()
        assert output.read_bytes() == b"previous report"
    finally:
        (control / "release").touch()
        capture_deadline = time.monotonic() + 5
        while worker_pidfd is None:
            try:
                capture_worker()
            except (FileNotFoundError, ProcessLookupError, AssertionError):
                break
            if worker_pidfd is not None or supervisor.poll() is not None:
                break
            if time.monotonic() >= capture_deadline:
                break
            time.sleep(0.005)
        if worker_pidfd is not None:
            try:
                signal.pidfd_send_signal(worker_pidfd, signal.SIGKILL)
            except ProcessLookupError:
                pass
        try:
            supervisor.wait(timeout=5)
        except subprocess.TimeoutExpired:
            supervisor.kill()
            supervisor.wait(timeout=5)
        finally:
            if worker_pidfd is not None:
                os.close(worker_pidfd)
        if worker_pid is not None:
            assert not Path(f"/proc/{worker_pid}").exists()


class SignalStartupProbe:
    """Retain exact fixture process identities independently of the supervisor."""

    def __init__(self, tmp_path, mode):
        self.control = tmp_path / "signal-control"
        self.control.mkdir()
        self.output = tmp_path / "report.json"
        self.output.write_bytes(b"previous report")
        self.worker_pid = self.worker_pidfd = None
        self.process = subprocess.Popen(
            [
                sys.executable,
                str(_FIXTURE),
                mode,
                "unused",
                str(self.output),
                str(self.control),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

    def capture_worker(self):
        record_path = self.control / "worker_pid"
        if self.worker_pidfd is not None or not record_path.exists():
            return
        record = json.loads(record_path.read_text())
        try:
            descriptor = os.pidfd_open(record["pid"])
        except ProcessLookupError:
            # Validation can start after the real owner has already reaped it.
            # Without an identity-bound pidfd the parent never sends a signal.
            return
        try:
            stat = Path(f"/proc/{record['pid']}/stat").read_text()
            assert stat.rsplit(")", 1)[1].split()[19] == record["start"]
        except (FileNotFoundError, ProcessLookupError):
            os.close(descriptor)
            return
        except BaseException:
            os.close(descriptor)
            raise
        self.worker_pid, self.worker_pidfd = record["pid"], descriptor

    def wait(self, predicate):
        deadline = time.monotonic() + 6
        while not predicate():
            try:
                self.capture_worker()
            except (FileNotFoundError, ProcessLookupError):
                # A healthy child can already have been reaped by its actual
                # supervisor; never signal a PID without a verified pidfd.
                pass
            assert self.process.poll() is None, "fixture exited before observation"
            assert time.monotonic() < deadline, "fixture observation deadline"
            time.sleep(0.005)

    def finish(self):
        (self.control / "allow_cleanup").touch()
        assert self.process.wait(timeout=6) == 0
        if self.worker_pid is not None:
            assert not Path(f"/proc/{self.worker_pid}").exists()

    def cleanup(self):
        for name in (
            "release",
            "release_startup",
            "release_validation",
            "allow_cleanup",
        ):
            (self.control / name).touch()
        try:
            self.capture_worker()
        except (FileNotFoundError, ProcessLookupError):
            pass
        try:
            if self.worker_pidfd is not None:
                try:
                    signal.pidfd_send_signal(self.worker_pidfd, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            try:
                self.process.wait(timeout=6)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        finally:
            if self.worker_pidfd is not None:
                os.close(self.worker_pidfd)
            if self.worker_pid is not None:
                assert not Path(f"/proc/{self.worker_pid}").exists()


@pytest.mark.parametrize("mode", ["startup_hold", "startup_validation"])
def test_default_sigint_during_executor_startup_retains_owner_and_receipt(
    tmp_path, mode
):
    boundary()
    probe = SignalStartupProbe(tmp_path, mode)
    try:
        checkpoint = "entered" if mode == "startup_hold" else "validation_held"
        probe.wait(
            lambda: (
                (probe.control / "startup_held").exists()
                and (probe.control / checkpoint).exists()
            )
        )
        probe.capture_worker()
        sent = 0
        for _ in range(3):
            tick = int((probe.control / "tick").read_text())
            os.kill(probe.process.pid, signal.SIGINT)
            sent += 1
            probe.wait(
                lambda tick=tick: (
                    (probe.control / "outcome").exists()
                    or int((probe.control / "tick").read_text()) > tick
                )
            )
            if (probe.control / "outcome").exists():
                break
        if mode == "startup_validation":
            (probe.control / "release_validation").touch()
        probe.wait(lambda: (probe.control / "supervisor_finished").exists())
        # Capture before releasing startup: a deferred signal must already be
        # visible to the supervisor's final publication check.
        returned_before_startup_release = (probe.control / "outcome").exists()
        output_before_release = probe.output.read_bytes()
        (probe.control / "release_startup").touch()
        probe.wait(lambda: (probe.control / "outcome").exists())
        receipt = json.loads((probe.control / "outcome").read_text())
        child_mask = (probe.control / "child_mask").read_text()
        probe.finish()
    finally:
        probe.cleanup()
    assert not returned_before_startup_release
    assert sent == 3
    assert output_before_release == b"previous report"
    assert probe.output.read_bytes() == b"previous report"
    assert receipt["outcome"] == "KeyboardInterrupt"
    assert (
        receipt["reaped"]
        and receipt["threads_retired"]
        and receipt["native_threads_retired"]
    )
    assert receipt["handler_restored"]
    assert receipt["worker_count"] == receipt["thread_count"] == 1
    assert receipt["owner_masks"] == [receipt["main_mask_before"]]
    assert receipt["main_mask_after"] == child_mask == receipt["main_mask_before"]


@pytest.mark.parametrize(
    "mode",
    [
        "policy_default",
        "policy_error",
        "policy_custom",
        "policy_ignore",
        "policy_sig_dfl",
        "policy_thread",
    ],
)
def test_supervisor_preserves_existing_handler_and_signal_masks(tmp_path, mode):
    boundary()
    probe = SignalStartupProbe(tmp_path, mode)
    try:
        probe.wait(lambda: (probe.control / "outcome").exists())
        receipt = json.loads((probe.control / "outcome").read_text())
        child_mask = (probe.control / "child_mask").read_text()
        probe.finish()
    finally:
        probe.cleanup()
    assert receipt["outcome"] == (
        "BenchmarkRunError" if mode == "policy_error" else "success"
    )
    assert (
        receipt["reaped"]
        and receipt["threads_retired"]
        and receipt["native_threads_retired"]
    )
    assert receipt["handler_restored"]
    if mode in {"policy_custom", "policy_ignore", "policy_sig_dfl", "policy_thread"}:
        assert receipt["startup_handler_matches"] == [True]
    assert receipt["worker_count"] == receipt["thread_count"] == 1
    assert receipt["owner_masks"] == [receipt["main_mask_before"]]
    assert receipt["main_mask_after"] == child_mask == receipt["main_mask_before"]
    if mode == "policy_error":
        assert probe.output.read_bytes() == b"previous report"
    else:
        assert json.loads(probe.output.read_text())["quality"] == {
            "passes_thresholds": False
        }


def test_real_podcast_module_dispatch_publishes_unknown_candidate_without_models(
    tmp_path,
    monkeypatch,
):
    module = boundary()
    cache = tmp_path / "empty-model-cache"
    cache.mkdir()
    monkeypatch.setenv("TRANSLATOR_MODEL_CACHE_ROOT", str(cache))
    output = tmp_path / "report.json"
    output.write_bytes(b"previous report")
    pcm_paths = {}
    for direction in ("ru_to_en", "en_to_ru"):
        path = tmp_path / f"{direction}.s16le"
        path.write_bytes(b"\0\0" * 16000)
        pcm_paths[direction] = str(path)
    inputs = {"segment_ms": 1000, "max_segments": 1, "fixture": "private synthetic PCM"}
    request = {
        "pcm_paths": pcm_paths,
        "inputs": inputs,
        "asr_model_ids": ["unknown-d077-asr"],
        "tts_model_ids": ["piper-medium"],
        "mode": "streaming_first",
        "voice_gender": "female",
    }
    workers, commands, commits = [], [], []
    original_popen, original_replace = subprocess.Popen, os.replace

    def popen(command, **kwargs):
        commands.append(command)
        worker = original_popen(command, **kwargs)
        workers.append(worker)
        return worker

    def replace(source, destination):
        assert len(workers) == 1 and workers[0].returncode == 0
        assert not Path(f"/proc/{workers[0].pid}").exists()
        assert output.read_bytes() == b"previous report"
        commits.append(destination)
        return original_replace(source, destination)

    monkeypatch.setattr(module.subprocess, "Popen", popen)
    monkeypatch.setattr(module.os, "replace", replace)
    try:
        result = module.run_benchmark(
            "podcast",
            request,
            output,
            limits=module.RunLimits(model_run_seconds=10, terminate_grace_seconds=0.1),
        )
        assert commands[0][:4] == [
            sys.executable,
            "-m",
            "translator_sidecar.benchmark.process_run",
            "podcast",
        ]
        assert len(workers) == len(commits) == 1
        assert result == json.loads(output.read_text())
        assert result["schema_version"] == "translator.podcast-quality-debug.v1"
        assert result["inputs"] == inputs
        assert len(result["models"]) == 1
        model = result["models"][0]
        assert (
            model["status"] == "skipped"
            and model["skip_reason"] == "unknown_asr_candidate"
        )
        assert model["asr_model_id"] == "unknown-d077-asr"
        assert model["mode"] == request["mode"] and model["voice_gender"] == "female"
        assert not model["segments"] and model["summary"]["segment_count"] == 0
        assert not Path(commands[0][-1]).parent.exists()
        assert not list(cache.iterdir())
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.kill()
            worker.wait(timeout=5)


def test_actual_podcast_cli_repeated_sigint_reaps_before_return(tmp_path):
    boundary()
    probe = SignalStartupProbe(tmp_path, "policy_podcast_sigint")
    try:
        probe.wait(lambda: (probe.control / "entered").exists())
        probe.capture_worker()
        os.kill(probe.process.pid, signal.SIGINT)
        probe.wait(lambda: (probe.control / "term").exists())
        for _ in range(2):
            os.kill(probe.process.pid, signal.SIGINT)
            time.sleep(0.01)
        probe.wait(lambda: (probe.control / "outcome").exists())
        receipt = json.loads((probe.control / "outcome").read_text())
        child_mask = (probe.control / "child_mask").read_text()
        output_before_cleanup = probe.output.read_bytes()
        probe.finish()
    finally:
        probe.cleanup()
    assert receipt["outcome"] == "KeyboardInterrupt"
    assert (
        receipt["reaped"]
        and receipt["threads_retired"]
        and receipt["native_threads_retired"]
    )
    assert receipt["handler_restored"]
    assert receipt["worker_count"] == receipt["thread_count"] == 1
    assert receipt["owner_masks"] == [receipt["main_mask_before"]]
    assert receipt["main_mask_after"] == child_mask == receipt["main_mask_before"]
    assert output_before_cleanup == probe.output.read_bytes() == b"previous report"

"""Real process-boundary tests for one isolated native AEC case."""

from __future__ import annotations

import functools
import json
import os
import runpy
import select
import signal
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path

STAGE = Path(__file__).resolve().parents[1] / "scripts/translator-aec-backend-stage"


class ScopeRejected(Exception):
    pass


def readable(fd: int, timeout: float) -> bool:
    return bool(select.select([fd], [], [], timeout)[0])


def kill_if_live(fd: int | None) -> None:
    if fd is not None:
        if not readable(fd, 0):
            signal.pidfd_send_signal(fd, signal.SIGKILL)
        os.close(fd)


class NativeCustodianTests(unittest.TestCase):
    def test_stage_sigkill_after_request_cleans_late_owned_scope(self) -> None:
        stage = runpy.run_path(str(STAGE), run_name="aec_custodian_late")
        self.assertIn("run_guarded_native_test", stage)
        nonce, name = "e" * 32, stage["NATIVE_TESTS"][0]
        with tempfile.TemporaryDirectory(prefix="aec-guardian-late-") as directory:
            root = Path(directory)
            receipts = root / "receipts"
            receipts.mkdir(mode=0o700)
            unrelated = root / ("translator-aec-" + "f" * 32 + ".scope")
            unrelated.mkdir()
            (unrelated / "sentinel").write_text("untouched", encoding="ascii")
            announced_read, announced_write = os.pipe()
            release_read, release_write = os.pipe()
            created_read, created_write = os.pipe()
            stage_pid = os.fork()
            if stage_pid == 0:
                os.close(announced_read)
                os.close(release_write)
                os.close(created_read)

                def acquire(child_pid, unit, _process, _check_cancel):
                    owned = root / f"{unit}.scope"
                    os.write(
                        announced_write,
                        f"{os.getpid()} {child_pid} {unit}\n".encode("ascii"),
                    )
                    os.read(release_read, 1)
                    owned.mkdir()
                    kill_file = owned / "cgroup.kill"
                    kill_file.write_text("", encoding="ascii")
                    os.write(created_write, b"P")
                    for _ in range(400):
                        if kill_file.read_text(encoding="ascii") == "1":
                            kill_file.unlink()
                            owned.rmdir()
                            return owned
                        select.select([], [], [], 0.02)
                    raise ScopeRejected("owned scope was never killed")

                runner = {
                    "owned_scope_path": lambda unit: root / f"{unit}.scope",
                    "acquire_scope": acquire,
                    "ScopeRejected": ScopeRejected,
                }
                try:
                    stage["run_guarded_native_test"](
                        Path("/usr/bin/true"),
                        name,
                        runner,
                        nonce,
                        receipts,
                    )
                except BaseException:
                    os._exit(3)
                os._exit(4)

            os.close(announced_write)
            os.close(release_read)
            os.close(created_write)
            guardian_fd = child_fd = None
            try:
                self.assertTrue(
                    readable(announced_read, 5), "outer request not reached"
                )
                guardian_pid, child_pid, unit = (
                    os.read(announced_read, 160).decode().split()
                )
                guardian_fd = os.pidfd_open(int(guardian_pid))
                child_fd = os.pidfd_open(int(child_pid))
                os.kill(stage_pid, signal.SIGKILL)
                os.waitpid(stage_pid, 0)
                stage_pid = 0
                self.assertFalse(readable(guardian_fd, 0), "guardian died with stage")
                os.write(release_write, b"R")
                self.assertTrue(readable(created_read, 3), "late scope not created")
                self.assertEqual(os.read(created_read, 1), b"P")
                self.assertTrue(readable(guardian_fd, 8), "guardian did not settle")
                receipt_path = receipts / f"{nonce}.json"
                receipt = json.loads(receipt_path.read_text())
                self.assertEqual(receipt_path.stat().st_mode & 0o777, 0o600)
                self.assertEqual(receipt["status"], "NOT_DONE")
                self.assertFalse(receipt["pass"])
                self.assertTrue(receipt["scope_absent"])
                self.assertTrue(receipt["child_reaped"])
                self.assertFalse((root / f"{unit}.scope").exists())
                self.assertTrue(readable(child_fd, 0))
                self.assertEqual((unrelated / "sentinel").read_text(), "untouched")
            finally:
                for fd in (announced_read, release_write, created_read):
                    os.close(fd)
                if stage_pid:
                    os.kill(stage_pid, signal.SIGKILL)
                    os.waitpid(stage_pid, 0)
                kill_if_live(child_fd)
                kill_if_live(guardian_fd)

    def test_guardian_exception_after_spawn_reaps_native_child(self) -> None:
        stage = runpy.run_path(str(STAGE), run_name="aec_custodian_fault")
        nonce, name = "d" * 32, stage["NATIVE_TESTS"][0]
        with tempfile.TemporaryDirectory(prefix="aec-guardian-fault-") as directory:
            root = Path(directory)
            receipts = root / "receipts"
            receipts.mkdir(mode=0o700)
            announced_read, announced_write = os.pipe()

            def failing_run(_binary, _name, _runner, *, custody, **_kwargs):
                proc = subprocess.Popen(
                    ["/usr/bin/sleep", "30"],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    close_fds=True,
                )
                custody["proc"] = proc
                custody["pidfd"] = os.pidfd_open(proc.pid)
                os.write(announced_write, str(proc.pid).encode("ascii"))
                raise OSError("injected after native spawn")

            stage["run_guarded_native_test"].__globals__[
                "run_supervised_native_test"
            ] = failing_run
            try:
                passed, _detail = stage["run_guarded_native_test"](
                    Path("/usr/bin/true"),
                    name,
                    {
                        "owned_scope_path": lambda unit: root / f"{unit}.scope",
                        "ScopeRejected": ScopeRejected,
                    },
                    nonce,
                    receipts,
                )
                self.assertTrue(readable(announced_read, 0), "fault missed spawn")
                child_pid = int(os.read(announced_read, 80))
                self.assertFalse(passed)
                with self.assertRaises(ProcessLookupError):
                    os.kill(child_pid, 0)
                receipt = json.loads((receipts / f"{nonce}.json").read_text())
                self.assertEqual(receipt["status"], "NOT_DONE")
                self.assertTrue(receipt["child_reaped"])
                self.assertTrue(receipt["scope_absent"])
            finally:
                os.close(announced_read)
                os.close(announced_write)

    def test_post_gate_exception_settles_terminal_a_and_n(self) -> None:
        stage = runpy.run_path(str(STAGE), run_name="aec_custodian_postgate")
        name = stage["NATIVE_TESTS"][0]
        with tempfile.TemporaryDirectory(prefix="aec-postgate-") as directory:
            root = Path(directory)
            receipts = root / "receipts"
            receipts.mkdir(mode=0o700)
            for index, outcome in enumerate(("A", "N")):
                with self.subTest(outcome=outcome):
                    nonce = f"{index + 1:032x}"
                    announced_read, announced_write = os.pipe()

                    def failing_run(
                        _binary,
                        _name,
                        _runner,
                        *,
                        custody,
                        inner_unit,
                        result: str,
                        announce: int,
                        **_kwargs,
                    ):
                        proc = subprocess.Popen(
                            ["/usr/bin/sleep", "30"],
                            stdin=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            close_fds=True,
                        )
                        os.write(announce, str(proc.pid).encode("ascii"))
                        custody["proc"] = proc
                        custody["pidfd"] = os.pidfd_open(proc.pid)
                        custody["native_gate_opened"] = True
                        read_fd, write_fd = os.pipe()
                        os.set_blocking(read_fd, False)
                        custody["lifecycle_fd"] = read_fd
                        custody["lifecycle_payload"] = bytearray()
                        custody["lifecycle_eof"] = False
                        event = {
                            "unit": inner_unit,
                            "pid": proc.pid,
                            "session": "0123456789abcdef",
                            "manager_owner": ":1.42",
                        }
                        p = dict(event, event="P")
                        terminal = dict(event, event=result)
                        if result == "A":
                            terminal["job"] = "/org/freedesktop/systemd1/job/17"
                        else:
                            del terminal["manager_owner"]
                        os.write(
                            write_fd,
                            (
                                json.dumps(p) + "\n" + json.dumps(terminal) + "\n"
                            ).encode(),
                        )
                        os.close(write_fd)
                        raise OSError("injected after native gate")

                    stage["run_guarded_native_test"].__globals__[
                        "run_supervised_native_test"
                    ] = functools.partial(
                        failing_run,
                        result=outcome,
                        announce=announced_write,
                    )
                    passed, _detail = stage["run_guarded_native_test"](
                        Path("/usr/bin/true"),
                        name,
                        {
                            "owned_scope_path": lambda unit: root / f"{unit}.scope",
                            "ScopeRejected": ScopeRejected,
                        },
                        nonce,
                        receipts,
                    )
                    self.assertFalse(passed)
                    os.close(announced_write)
                    child_pid = int(os.read(announced_read, 80))
                    os.close(announced_read)
                    with self.assertRaises(ProcessLookupError):
                        os.kill(child_pid, 0)
                    receipt = json.loads((receipts / f"{nonce}.json").read_text())
                    self.assertEqual(receipt["status"], "NOT_DONE")
                    self.assertTrue(receipt["scope_absent"])
                    self.assertTrue(receipt["child_reaped"])

    def test_stage_sigkill_before_permit_starts_no_case(self) -> None:
        stage = runpy.run_path(str(STAGE), run_name="aec_custodian_prepermit")
        with tempfile.TemporaryDirectory(prefix="aec-prepermit-") as directory:
            root = Path(directory)
            receipts = root / "receipts"
            receipts.mkdir(mode=0o700)
            ready_read, ready_write = os.pipe()
            request_read, request_write = os.pipe()
            stage_pid = os.fork()
            if stage_pid == 0:
                os.close(ready_read)
                os.close(request_read)
                original_getsid = os.getsid

                def hold_before_permit(pid):
                    if pid:
                        os.write(ready_write, str(pid).encode("ascii"))
                        select.select([], [], [], 30)
                    return original_getsid(pid)

                stage["run_guarded_native_test"].__globals__[
                    "os"
                ].getsid = hold_before_permit
                runner = {
                    "owned_scope_path": lambda unit: root / f"{unit}.scope",
                    "acquire_scope": lambda *_args: os.write(request_write, b"P"),
                    "ScopeRejected": ScopeRejected,
                }
                try:
                    stage["run_guarded_native_test"](
                        Path("/usr/bin/true"),
                        stage["NATIVE_TESTS"][0],
                        runner,
                        "c" * 32,
                        receipts,
                    )
                except BaseException:
                    os._exit(3)
                os._exit(4)
            os.close(ready_write)
            os.close(request_write)
            guardian_fd = None
            try:
                self.assertTrue(readable(ready_read, 5), "guardian was not ready")
                guardian_fd = os.pidfd_open(int(os.read(ready_read, 80)))
                os.kill(stage_pid, signal.SIGKILL)
                os.waitpid(stage_pid, 0)
                stage_pid = 0
                self.assertTrue(readable(guardian_fd, 5), "guardian did not exit")
                self.assertEqual(os.read(request_read, 1), b"")
                self.assertEqual(list(receipts.glob("*.json")), [])
                self.assertEqual(list(root.glob("*.scope")), [])
            finally:
                os.close(ready_read)
                os.close(request_read)
                if stage_pid:
                    os.kill(stage_pid, signal.SIGKILL)
                    os.waitpid(stage_pid, 0)
                kill_if_live(guardian_fd)

    def test_stage_sigkill_after_admission_cleans_exact_scope(self) -> None:
        stage = runpy.run_path(str(STAGE), run_name="aec_custodian_admitted")
        nonce, name = "b" * 32, stage["NATIVE_TESTS"][0]
        with tempfile.TemporaryDirectory(prefix="aec-admitted-") as directory:
            root = Path(directory)
            receipts = root / "receipts"
            receipts.mkdir(mode=0o700)
            native_started = root / "native-started"
            binary = root / "native-case"
            binary.write_text(
                "#!/bin/sh\ntouch "
                + str(native_started)
                + "\nexec /usr/bin/sleep 30\n",
                encoding="ascii",
            )
            binary.chmod(0o700)
            unrelated = root / ("translator-aec-" + "f" * 32 + ".scope")
            unrelated.mkdir()
            (unrelated / "sentinel").write_text("untouched", encoding="ascii")
            announced_read, announced_write = os.pipe()
            stage_pid = os.fork()
            if stage_pid == 0:
                os.close(announced_read)

                def acquire(child_pid, unit, _process, _check_cancel):
                    owned = root / f"{unit}.scope"
                    owned.mkdir()
                    kill_file = owned / "cgroup.kill"
                    kill_file.write_text("", encoding="ascii")

                    def observe_kill():
                        for _ in range(400):
                            if kill_file.read_text(encoding="ascii") == "1":
                                kill_file.unlink()
                                owned.rmdir()
                                return
                            select.select([], [], [], 0.02)

                    threading.Thread(target=observe_kill, daemon=True).start()
                    os.write(
                        announced_write,
                        f"{os.getpid()} {child_pid} {unit}\n".encode("ascii"),
                    )
                    return owned

                runner = {
                    "owned_scope_path": lambda unit: root / f"{unit}.scope",
                    "acquire_scope": acquire,
                    "ScopeRejected": ScopeRejected,
                }
                try:
                    stage["run_guarded_native_test"](
                        binary,
                        name,
                        runner,
                        nonce,
                        receipts,
                    )
                except BaseException:
                    os._exit(3)
                os._exit(4)
            os.close(announced_write)
            guardian_fd = child_fd = None
            try:
                self.assertTrue(readable(announced_read, 5), "admission not reached")
                guardian_pid, child_pid, unit = (
                    os.read(announced_read, 160).decode().split()
                )
                guardian_fd = os.pidfd_open(int(guardian_pid))
                child_fd = os.pidfd_open(int(child_pid))
                for _ in range(200):
                    if native_started.exists():
                        break
                    select.select([], [], [], 0.02)
                self.assertTrue(native_started.exists(), "native gate was not opened")
                os.kill(stage_pid, signal.SIGKILL)
                os.waitpid(stage_pid, 0)
                stage_pid = 0
                self.assertTrue(readable(guardian_fd, 8), "guardian did not settle")
                self.assertTrue(readable(child_fd, 0), "native child survived")
                receipt = json.loads((receipts / f"{nonce}.json").read_text())
                self.assertEqual(receipt["status"], "NOT_DONE")
                self.assertTrue(receipt["scope_absent"])
                self.assertTrue(receipt["child_reaped"])
                self.assertFalse((root / f"{unit}.scope").exists())
                self.assertEqual((unrelated / "sentinel").read_text(), "untouched")
            finally:
                os.close(announced_read)
                if stage_pid:
                    os.kill(stage_pid, signal.SIGKILL)
                    os.waitpid(stage_pid, 0)
                kill_if_live(child_fd)
                kill_if_live(guardian_fd)


if __name__ == "__main__":
    unittest.main()

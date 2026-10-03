"""Adversarial executable checks for the AEC private-graph launcher."""

from __future__ import annotations

import builtins
import contextlib
import fcntl
import io
import json
import os
import runpy
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

# The launcher uses /usr/bin/python3; the manifest collector uses an isolated venv.
if "/usr/lib/python3/dist-packages" not in sys.path:
    sys.path.append("/usr/lib/python3/dist-packages")
from gi.repository import Gio, GLib

ROOT = Path(__file__).resolve().parents[1]
CHECK = ROOT / "scripts" / "translator-aec-backend-check"


def _run(
    *args: str, env: dict[str, str] | None = None, pass_fds=()
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(CHECK), "--isolated", *args],
        cwd=ROOT,
        env=env,
        pass_fds=pass_fds,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )


class AecBackendIsolationHardeningTests(unittest.TestCase):
    def test_probe_cannot_open_host_resources_or_reach_parent_listeners(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            marker = root / "host-only"
            marker.write_text("private")
            unix_listener = socket.socket(socket.AF_UNIX)
            tcp_listener = socket.socket(socket.AF_INET)
            try:
                unix_path = root / "host.sock"
                unix_listener.bind(str(unix_path))
                unix_listener.listen(1)
                unix_listener.settimeout(0.1)
                tcp_listener.bind(("127.0.0.1", 0))
                tcp_listener.listen(1)
                tcp_listener.settimeout(0.1)
                self.assertEqual(marker.read_text(), "private")
                self.assertTrue(unix_path.exists())
                self.assertEqual(tcp_listener.getsockname()[0], "127.0.0.1")
                nonce = os.urandom(16).hex()
                probe = root / "probe.py"
                probe.write_text(
                    "import json, os, socket\n"
                    f"nonce = {nonce!r}\n"
                    f"marker = {str(marker)!r}\n"
                    f"unix_path = {str(unix_path)!r}\n"
                    f"tcp_port = {tcp_listener.getsockname()[1]!r}\n"
                    "def opened(path):\n"
                    " try:\n"
                    "  fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)\n"
                    " except OSError:\n"
                    "  return False\n"
                    " else:\n"
                    "  os.close(fd)\n"
                    "  return True\n"
                    "u = socket.socket(socket.AF_UNIX)\n"
                    "t = socket.socket(socket.AF_INET)\n"
                    "t.settimeout(0.2)\n"
                    "print(json.dumps({'nonce': nonce,\n"
                    " 'marker_opened': opened(marker),\n"
                    " 'audio_opened': opened('/dev/snd/controlC0'),\n"
                    " 'production_socket_visible': os.path.exists('/run/user/1000/pipewire-0'),\n"
                    " 'unix_connected': u.connect_ex(unix_path) == 0,\n"
                    " 'tcp_connected': t.connect_ex(('127.0.0.1', tcp_port)) == 0,\n"
                    " 'net_namespace': os.readlink('/proc/self/ns/net')}))\n"
                )
                result = _run("--probe", str(probe))
                self.assertEqual(result.returncode, 0, result.stderr)
                facts = json.loads(result.stdout)
                self.assertEqual(facts.pop("nonce"), nonce)
                self.assertNotEqual(
                    facts.pop("net_namespace"), os.readlink("/proc/self/ns/net")
                )
                self.assertEqual(facts, dict.fromkeys(facts, False))
                with self.assertRaises(socket.timeout):
                    unix_listener.accept()
                with self.assertRaises(socket.timeout):
                    tcp_listener.accept()
            finally:
                unix_listener.close()
                tcp_listener.close()

    def test_preflight_has_no_host_devices_routes_or_default_runtime(self) -> None:
        result = _run("--preflight-only")
        self.assertEqual(result.returncode, 0, result.stderr)
        facts = json.loads(result.stdout)
        self.assertTrue(facts["isolated"])
        self.assertTrue(facts["net_namespace_changed"])
        self.assertTrue(facts["mount_namespace_changed"])
        self.assertFalse(facts["host_audio_access"])
        self.assertFalse(facts["host_runtime_access"])
        self.assertFalse(facts["network_access"])
        self.assertFalse(facts["default_route"])
        self.assertEqual(facts["non_loopback_interfaces"], [])

    def test_lifecycle_descriptor_above_nine_is_accepted_only_for_marker(self) -> None:
        read_fd, write_fd = os.pipe()
        lifecycle_fd = fcntl.fcntl(write_fd, fcntl.F_DUPFD, 10)
        os.close(write_fd)
        try:
            result = _run(
                "--preflight-only",
                env={**os.environ, "TRANSLATOR_AEC_LIFECYCLE_FD": str(lifecycle_fd)},
                pass_fds=(lifecycle_fd,),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
        finally:
            os.close(lifecycle_fd)
        try:
            self.assertEqual(os.read(read_fd, 2), b"PA")
        finally:
            os.close(read_fd)

    def test_broken_stage_lifecycle_channel_prevents_scope_submit(self) -> None:
        namespace = runpy.run_path(str(CHECK), run_name="aec_runner_contract")
        run_scoped = namespace["run_scoped"]
        globals_ = run_scoped.__globals__
        stage_read, stage_write = os.pipe()
        os.close(stage_read)
        unit = "translator-aec-" + "c" * 32
        process = mock.Mock(pid=12347, returncode=0)
        process.communicate.return_value = (b"", b"")
        submitted = False
        acquire_calls = 0
        request_calls = 0
        cleanup = mock.Mock(return_value=(True, 1))
        stderr = io.StringIO()

        def fake_acquire(_pid, _unit, _process, _check_cancel, on_created, on_request):
            nonlocal submitted, acquire_calls, request_calls
            acquire_calls += 1
            request_calls += 1
            on_request()
            submitted = True
            raise namespace["ScopeRejected"]("mock manager rejection")

        try:
            with (
                mock.patch.dict(
                    globals_,
                    {
                        "acquire_scope": fake_acquire,
                        "cleanup_scope": cleanup,
                        "scope_present": mock.Mock(return_value=False),
                    },
                ),
                mock.patch.object(Path, "is_socket", return_value=True),
                mock.patch("subprocess.Popen", return_value=process),
                mock.patch.dict(
                    os.environ,
                    {
                        "TRANSLATOR_AEC_SCOPE_UNIT": unit,
                        "TRANSLATOR_AEC_EXPECTED_SESSION": "0123456789abcdef",
                        "TRANSLATOR_AEC_STAGE_LIFECYCLE_FD": str(stage_write),
                    },
                ),
                contextlib.redirect_stderr(stderr),
            ):
                self.assertEqual(
                    run_scoped(["/usr/bin/true"], stream=False, quick=True), 2
                )
        finally:
            os.close(stage_write)
        self.assertEqual(acquire_calls, 1)
        self.assertEqual(request_calls, 1)
        self.assertEqual(cleanup.call_count, 1)
        self.assertIn("BrokenPipeError", stderr.getvalue())
        self.assertFalse(
            submitted, "scope request was sent after custody channel failed"
        )

    def test_stage_lifecycle_receives_p_before_manager_submit_and_n_after_rejection(
        self,
    ) -> None:
        namespace = runpy.run_path(str(CHECK), run_name="aec_runner_contract")
        run_scoped = namespace["run_scoped"]
        globals_ = run_scoped.__globals__
        stage_read, stage_write = os.pipe()
        os.set_blocking(stage_read, False)
        unit = "translator-aec-" + "d" * 32
        process = mock.Mock(pid=12348, returncode=0)
        cleanup = mock.Mock(return_value=(True, 1))
        observed_before_submit = b""

        def fake_acquire(_pid, _unit, _process, _check_cancel, on_created, on_request):
            nonlocal observed_before_submit
            on_request()
            try:
                observed_before_submit = os.read(stage_read, 4096)
            except BlockingIOError:
                pass
            raise namespace["ScopeRejected"]("mock manager rejection")

        try:
            with (
                mock.patch.dict(
                    globals_,
                    {
                        "acquire_scope": fake_acquire,
                        "cleanup_scope": cleanup,
                        "scope_present": mock.Mock(return_value=False),
                    },
                ),
                mock.patch.object(Path, "is_socket", return_value=True),
                mock.patch("subprocess.Popen", return_value=process),
                mock.patch.dict(
                    os.environ,
                    {
                        "TRANSLATOR_AEC_SCOPE_UNIT": unit,
                        "TRANSLATOR_AEC_EXPECTED_SESSION": "0123456789abcdef",
                        "TRANSLATOR_AEC_STAGE_LIFECYCLE_FD": str(stage_write),
                    },
                ),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(
                    run_scoped(["/usr/bin/true"], stream=False, quick=True), 2
                )
            try:
                tail = os.read(stage_read, 4096)
            except BlockingIOError:
                tail = b""
        finally:
            os.close(stage_write)
            os.close(stage_read)
        self.assertEqual(cleanup.call_count, 1)
        lines = (observed_before_submit + tail).splitlines()
        self.assertEqual(len(lines), 2)
        events = [json.loads(line) for line in lines]
        self.assertEqual([event["event"] for event in events], ["P", "N"])
        self.assertIn(b'"event": "P"', observed_before_submit)
        self.assertTrue(
            all(
                event["unit"] == unit
                and event["session"] == "0123456789abcdef"
                and event["pid"] == os.getpid()
                for event in events
            )
        )

    def test_inherited_fd_is_rejected_before_probe(self) -> None:
        with open(os.devnull, "rb") as inherited:
            result = _run("--preflight-only", pass_fds=(inherited.fileno(),))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("inherited descriptor", result.stderr.lower())

    def test_default_server_override_is_rejected_before_probe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            sentinel = socket.socket(socket.AF_UNIX)
            try:
                address = str(Path(temporary) / "fake-default.sock")
                sentinel.bind(address)
                sentinel.listen(1)
                sentinel.settimeout(0.1)
                result = _run(
                    "--preflight-only",
                    env={**os.environ, "PIPEWIRE_REMOTE": address},
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("default server", result.stderr.lower())
                with self.assertRaises(socket.timeout):
                    sentinel.accept()
            finally:
                sentinel.close()

    def test_failed_scope_admission_keeps_expected_scope_in_cleanup_custody(
        self,
    ) -> None:
        namespace = runpy.run_path(str(CHECK), run_name="aec_runner_contract")
        run_scoped = namespace["run_scoped"]
        globals_ = run_scoped.__globals__
        unit = "translator-aec-" + "a" * 32
        process = mock.Mock(pid=12345)
        cleanup = mock.Mock(return_value=(True, 1))
        stderr = io.StringIO()
        with (
            mock.patch.dict(
                globals_,
                {
                    "acquire_scope": mock.Mock(
                        side_effect=namespace["UnsafeInvocation"](
                            "membership unavailable"
                        )
                    ),
                    "cleanup_scope": cleanup,
                    "scope_present": mock.Mock(return_value=False),
                },
            ),
            mock.patch.object(Path, "is_socket", return_value=True),
            mock.patch("subprocess.Popen", return_value=process),
            mock.patch.dict(os.environ, {"TRANSLATOR_AEC_SCOPE_UNIT": unit}),
            contextlib.redirect_stderr(stderr),
        ):
            self.assertEqual(run_scoped(["/usr/bin/true"], stream=False, quick=True), 2)
        self.assertEqual(cleanup.call_count, 1)
        self.assertEqual(cleanup.call_args.args[1], namespace["owned_scope_path"](unit))
        self.assertIn("cleanup_reaped=False", stderr.getvalue())

    def test_await_scope_checks_cancel_after_negative_membership_read(self) -> None:
        namespace = runpy.run_path(str(CHECK), run_name="aec_runner_contract")
        await_scope = namespace["await_scope"]
        process = mock.Mock()
        process.poll.return_value = None
        read_scope = mock.Mock(return_value=None)
        checks = 0

        def check_cancel() -> None:
            nonlocal checks
            checks += 1
            if checks == 2:
                raise namespace["ScopeCancelled"]("cancel during admission")

        with (
            mock.patch.dict(await_scope.__globals__, {"read_scope": read_scope}),
            mock.patch("time.sleep") as sleep,
            self.assertRaises(namespace["ScopeCancelled"]),
        ):
            await_scope(
                process.pid, "translator-aec-" + "d" * 32, process, check_cancel
            )
        self.assertEqual(checks, 2)
        self.assertEqual(read_scope.call_count, 1)
        sleep.assert_not_called()

    def test_unknown_scope_creation_cannot_report_cleanup_reaped(self) -> None:
        namespace = runpy.run_path(str(CHECK), run_name="aec_runner_contract")
        run_scoped = namespace["run_scoped"]
        globals_ = run_scoped.__globals__
        unit = "translator-aec-" + "b" * 32
        process = mock.Mock(pid=12346)
        cleanup = mock.Mock(return_value=(True, 1))
        stderr = io.StringIO()
        cancelled = namespace["ScopeCancelled"]("cancel before scope admission")
        await_scope = mock.Mock(side_effect=cancelled)

        with (
            mock.patch.dict(
                globals_,
                {
                    "acquire_scope": await_scope,
                    "cleanup_scope": cleanup,
                    "scope_present": mock.Mock(return_value=False),
                },
            ),
            mock.patch.object(Path, "is_socket", return_value=True),
            mock.patch("subprocess.Popen", return_value=process),
            mock.patch.dict(os.environ, {"TRANSLATOR_AEC_SCOPE_UNIT": unit}),
            contextlib.redirect_stderr(stderr),
        ):
            result = run_scoped(["/usr/bin/true"], stream=False, quick=True)

        self.assertEqual(result, 2)
        self.assertIs(await_scope.call_args.args[2], process)
        self.assertEqual(cleanup.call_count, 1)
        self.assertEqual(cleanup.call_args.args[1], namespace["owned_scope_path"](unit))
        self.assertIn("ScopeCancelled", stderr.getvalue())
        self.assertIn("cleanup_reaped=False", stderr.getvalue())

    def test_sigterm_inside_popen_keeps_child_in_cleanup_custody(self) -> None:
        namespace = runpy.run_path(str(CHECK), run_name="aec_runner_contract")
        run_scoped = namespace["run_scoped"]
        globals_ = run_scoped.__globals__
        unit = "translator-aec-" + "c" * 32
        original_popen = subprocess.Popen
        process: subprocess.Popen[bytes] | None = None
        stderr = io.StringIO()
        await_scope = mock.Mock()
        cleanup = mock.Mock(wraps=namespace["cleanup_scope"])
        read_fd, write_fd = os.pipe()

        def popen(*_args: object, **_kwargs: object) -> subprocess.Popen[bytes]:
            nonlocal process
            process = original_popen(
                ["/usr/bin/sleep", "30"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            handler = signal.getsignal(signal.SIGTERM)
            self.assertTrue(callable(handler))
            handler(signal.SIGTERM, None)
            return process

        try:
            with (
                mock.patch.dict(
                    globals_,
                    {
                        "scope_present": mock.Mock(return_value=False),
                        "acquire_scope": await_scope,
                        "cleanup_scope": cleanup,
                    },
                ),
                mock.patch.object(Path, "is_socket", return_value=True),
                mock.patch("subprocess.Popen", side_effect=popen),
                mock.patch.dict(
                    os.environ,
                    {
                        "TRANSLATOR_AEC_SCOPE_UNIT": unit,
                        "TRANSLATOR_AEC_LIFECYCLE_FD": str(write_fd),
                    },
                ),
                contextlib.redirect_stderr(stderr),
            ):
                result = run_scoped(["/usr/bin/true"], stream=False, quick=True)
                self.assertIsNotNone(process)
                child_reaped_at_return = process.poll() is not None
        finally:
            if process is not None and process.poll() is None:
                process.kill()
                process.wait(timeout=2)
            if process is not None:
                for pipe in (process.stdout, process.stderr):
                    if pipe is not None:
                        pipe.close()
            with contextlib.suppress(OSError):
                os.close(write_fd)

        self.assertEqual(result, 2)
        self.assertEqual(os.read(read_fd, 1), b"N")
        os.close(read_fd)
        self.assertTrue(child_reaped_at_return, "launcher did not reap its child")
        self.assertEqual(cleanup.call_count, 1)
        self.assertEqual(cleanup.call_args.args[1], namespace["owned_scope_path"](unit))
        await_scope.assert_not_called()
        self.assertIn("cleanup_reaped=True", stderr.getvalue())

    def test_explicit_manager_rejection_reports_definitive_no_create(self) -> None:
        namespace = runpy.run_path(str(CHECK), run_name="aec_runner_contract")
        run_scoped = namespace["run_scoped"]
        unit = "translator-aec-" + "e" * 32
        read_fd, write_fd = os.pipe()
        os.set_blocking(read_fd, False)
        process = mock.Mock(pid=12347)

        class ExplicitRejection(namespace["UnsafeInvocation"]):
            pass

        try:
            with (
                mock.patch.dict(
                    run_scoped.__globals__,
                    {
                        "ScopeRejected": ExplicitRejection,
                        "acquire_scope": mock.Mock(
                            side_effect=ExplicitRejection("manager rejected")
                        ),
                        "cleanup_scope": mock.Mock(return_value=(True, 1)),
                        "scope_present": mock.Mock(return_value=False),
                    },
                ),
                mock.patch.object(Path, "is_socket", return_value=True),
                mock.patch("subprocess.Popen", return_value=process) as spawn,
                mock.patch.dict(
                    os.environ,
                    {
                        "TRANSLATOR_AEC_SCOPE_UNIT": unit,
                        "TRANSLATOR_AEC_LIFECYCLE_FD": str(write_fd),
                    },
                ),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(run_scoped(["/usr/bin/true"], False, True), 2)
            self.assertEqual(os.read(read_fd, 1), b"N")
            self.assertNotIn(write_fd, spawn.call_args.kwargs["pass_fds"])
        finally:
            os.close(read_fd)
            with contextlib.suppress(OSError):
                os.close(write_fd)

    def test_completed_job_buffered_before_start_reply_admits_exact_pid(self) -> None:
        namespace = runpy.run_path(str(CHECK), run_name="aec_runner_contract")
        acquire = namespace["acquire_scope"]
        unit = "translator-aec-" + "f" * 32
        job = "/org/freedesktop/systemd1/job/54321"
        connection = mock.Mock()
        callback = None
        events: list[str] = []

        def subscribe(*args: object) -> int:
            nonlocal callback
            events.append("signal_subscribe")
            callback = args[-1]
            return 1

        def call_sync(
            _owner: str,
            _path: str,
            _interface: str,
            method: str,
            parameters: object,
            *_rest: object,
        ) -> GLib.Variant:
            events.append(method)
            if method == "GetNameOwner":
                return GLib.Variant("(s)", (":1.321",))
            if method == "StartTransientUnit":
                self.assertIsNotNone(callback)
                name, mode, properties, auxiliary = parameters.unpack()
                self.assertEqual((name, mode, auxiliary), (unit + ".scope", "fail", []))
                values = dict(properties)
                self.assertEqual(values["PIDs"], [12348])
                self.assertEqual(values["Slice"], "app.slice")
                self.assertEqual(values["MemoryMax"], 1 << 30)
                self.assertEqual(values["MemorySwapMax"], 0)
                self.assertEqual(values["TasksMax"], 32)
                self.assertEqual(values["CollectMode"], "inactive-or-failed")
                callback(
                    connection,
                    ":1.321",
                    "/org/freedesktop/systemd1",
                    "org.freedesktop.systemd1.Manager",
                    "JobRemoved",
                    GLib.Variant("(uoss)", (54321, job, name, "done")),
                )
                return GLib.Variant("(o)", (job,))
            return GLib.Variant("()", ())

        connection.signal_subscribe.side_effect = subscribe
        connection.call_sync.side_effect = call_sync
        connection.is_closed.return_value = False
        created = mock.Mock()

        def verify_admission(*_args: object) -> Path:
            created.assert_called_once_with(job, ":1.321")
            return Path("/tmp/verified-scope")

        await_scope = mock.Mock(side_effect=verify_admission)
        with (
            mock.patch.object(
                Gio.DBusConnection, "new_for_address_sync", return_value=connection
            ),
            mock.patch.dict(acquire.__globals__, {"await_scope": await_scope}),
        ):
            self.assertEqual(
                acquire(12348, unit, mock.Mock(), mock.Mock(), created),
                Path("/tmp/verified-scope"),
            )
        self.assertLess(
            events.index("signal_subscribe"), events.index("StartTransientUnit")
        )
        self.assertLess(events.index("Subscribe"), events.index("StartTransientUnit"))
        self.assertEqual(await_scope.call_count, 1)
        connection.signal_unsubscribe.assert_called_once_with(1)
        connection.close_sync.assert_called_once()

    def test_transport_failure_is_not_definitive_manager_rejection(self) -> None:
        namespace = runpy.run_path(str(CHECK), run_name="aec_runner_contract")
        acquire = namespace["acquire_scope"]
        connection = mock.Mock()
        local_error = GLib.Error(
            "bus transport lost", "g-io-error-quark", int(Gio.IOErrorEnum.TIMED_OUT)
        )

        def call_sync(
            _owner: str,
            _path: str,
            _interface: str,
            method: str,
            _parameters: object,
            *_rest: object,
        ) -> GLib.Variant:
            if method == "GetNameOwner":
                return GLib.Variant("(s)", (":1.322",))
            if method == "StartTransientUnit":
                raise local_error
            return GLib.Variant("()", ())

        connection.call_sync.side_effect = call_sync
        connection.signal_subscribe.return_value = 1
        requests: list[str] = []
        with mock.patch.object(
            Gio.DBusConnection, "new_for_address_sync", return_value=connection
        ):
            with self.assertRaises(namespace["UnsafeInvocation"]) as raised:
                acquire(
                    12349,
                    "translator-aec-" + "1" * 32,
                    mock.Mock(),
                    mock.Mock(),
                    on_request=lambda _owner: requests.append("P"),
                )
        self.assertEqual(requests, ["P"])
        self.assertNotIsInstance(raised.exception, namespace["ScopeRejected"])
        connection.signal_unsubscribe.assert_called_once_with(1)

    def test_remote_rejection_requires_absent_unit_and_job(self) -> None:
        namespace = runpy.run_path(str(CHECK), run_name="aec_runner_contract")
        acquire = namespace["acquire_scope"]
        unit = "translator-aec-" + "2" * 32
        connection = mock.Mock()

        def call_sync(
            _owner: str,
            _path: str,
            _interface: str,
            method: str,
            _parameters: object,
            *_rest: object,
        ) -> GLib.Variant:
            if method == "GetNameOwner":
                return GLib.Variant("(s)", (":1.323",))
            if method == "StartTransientUnit":
                raise Gio.DBusError.new_for_dbus_error(
                    "org.freedesktop.systemd1.UnitExists", "rejected"
                )
            if method == "GetUnit":
                raise Gio.DBusError.new_for_dbus_error(
                    "org.freedesktop.systemd1.NoSuchUnit", "absent"
                )
            if method == "ListJobs":
                return GLib.Variant("(a(usssoo))", ([],))
            return GLib.Variant("()", ())

        connection.call_sync.side_effect = call_sync
        connection.signal_subscribe.return_value = 1
        with mock.patch.object(
            Gio.DBusConnection, "new_for_address_sync", return_value=connection
        ):
            with self.assertRaises(namespace["ScopeRejected"]):
                acquire(12350, unit, mock.Mock(), mock.Mock())
        connection.signal_unsubscribe.assert_called_once_with(1)

    def test_cancel_during_manager_call_waits_for_exact_job_result(self) -> None:
        namespace = runpy.run_path(str(CHECK), run_name="aec_runner_contract")
        acquire = namespace["acquire_scope"]
        unit = "translator-aec-" + "3" * 32
        job = "/org/freedesktop/systemd1/job/54322"
        connection = mock.Mock()
        callback = None
        cancelled = False

        def subscribe(*args: object) -> int:
            nonlocal callback
            callback = args[-1]
            return 1

        def call_sync(
            _owner: str,
            _path: str,
            _interface: str,
            method: str,
            _parameters: object,
            *_rest: object,
        ) -> GLib.Variant:
            nonlocal cancelled
            if method == "GetNameOwner":
                return GLib.Variant("(s)", (":1.324",))
            if method == "StartTransientUnit":
                cancelled = True
                callback(
                    connection,
                    ":1.324",
                    "/org/freedesktop/systemd1",
                    "org.freedesktop.systemd1.Manager",
                    "JobRemoved",
                    GLib.Variant("(uoss)", (54322, job, unit + ".scope", "done")),
                )
                return GLib.Variant("(o)", (job,))
            return GLib.Variant("()", ())

        def check_cancel() -> None:
            if cancelled:
                raise namespace["ScopeCancelled"]("cancel after manager result")

        connection.signal_subscribe.side_effect = subscribe
        connection.call_sync.side_effect = call_sync
        connection.is_closed.return_value = False
        await_scope = mock.Mock(return_value=Path("/tmp/verified-scope"))
        with (
            mock.patch.object(
                Gio.DBusConnection, "new_for_address_sync", return_value=connection
            ),
            mock.patch.dict(acquire.__globals__, {"await_scope": await_scope}),
        ):
            self.assertEqual(
                acquire(12351, unit, mock.Mock(), check_cancel),
                Path("/tmp/verified-scope"),
            )
        self.assertTrue(cancelled)
        await_scope.assert_called_once()
        connection.signal_unsubscribe.assert_called_once_with(1)

    def test_ambiguous_acquisition_never_emits_terminal_marker(self) -> None:
        namespace = runpy.run_path(str(CHECK), run_name="aec_runner_contract")
        run_scoped = namespace["run_scoped"]
        read_fd, write_fd = os.pipe()
        os.set_blocking(read_fd, False)
        process = mock.Mock(pid=12352)
        retained = mock.Mock()
        try:
            with (
                mock.patch.dict(
                    run_scoped.__globals__,
                    {
                        "acquire_scope": mock.Mock(
                            side_effect=namespace["ScopeCreationUnknown"](
                                "transport unknown"
                            )
                        ),
                        "cleanup_scope": mock.Mock(return_value=(True, 1)),
                        "retain_ambiguous_scope": retained,
                        "scope_present": mock.Mock(return_value=False),
                    },
                ),
                mock.patch.object(Path, "is_socket", return_value=True),
                mock.patch("subprocess.Popen", return_value=process),
                mock.patch.dict(
                    os.environ,
                    {
                        "TRANSLATOR_AEC_SCOPE_UNIT": "translator-aec-" + "4" * 32,
                        "TRANSLATOR_AEC_LIFECYCLE_FD": str(write_fd),
                    },
                ),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(run_scoped(["/usr/bin/true"], False, True), 2)
            self.assertEqual(os.read(read_fd, 1), b"")
            retained.assert_called_once()
        finally:
            os.close(read_fd)
            with contextlib.suppress(OSError):
                os.close(write_fd)

    def test_missing_gi_before_request_reaps_child_and_marks_no_create(self) -> None:
        namespace = runpy.run_path(str(CHECK), run_name="aec_runner_contract")
        run_scoped = namespace["run_scoped"]
        read_fd, write_fd = os.pipe()
        process = mock.Mock(pid=12353)
        real_import = builtins.__import__

        def missing_gi(name: str, *args: object, **kwargs: object) -> object:
            if name == "gi.repository":
                raise ImportError("GI unavailable")
            return real_import(name, *args, **kwargs)

        try:
            with (
                mock.patch.dict(
                    run_scoped.__globals__,
                    {
                        "cleanup_scope": mock.Mock(return_value=(True, 1)),
                        "scope_present": mock.Mock(return_value=False),
                    },
                ),
                mock.patch.object(Path, "is_socket", return_value=True),
                mock.patch("subprocess.Popen", return_value=process),
                mock.patch.dict(
                    os.environ,
                    {
                        "TRANSLATOR_AEC_SCOPE_UNIT": "translator-aec-" + "5" * 32,
                        "TRANSLATOR_AEC_LIFECYCLE_FD": str(write_fd),
                    },
                ),
                mock.patch("builtins.__import__", side_effect=missing_gi),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(run_scoped(["/usr/bin/true"], False, True), 2)
            self.assertEqual(os.read(read_fd, 2), b"N")
        finally:
            os.close(read_fd)
            with contextlib.suppress(OSError):
                os.close(write_fd)

    def test_connect_failure_before_request_is_definitive_no_create(self) -> None:
        namespace = runpy.run_path(str(CHECK), run_name="aec_runner_contract")
        acquire = namespace["acquire_scope"]
        error = GLib.Error(
            "connection refused",
            "g-io-error-quark",
            int(Gio.IOErrorEnum.CONNECTION_REFUSED),
        )
        with mock.patch.object(
            Gio.DBusConnection, "new_for_address_sync", side_effect=error
        ):
            with self.assertRaises(namespace["ScopeRejected"]):
                acquire(
                    12354,
                    "translator-aec-" + "6" * 32,
                    mock.Mock(),
                    mock.Mock(),
                )

    def test_p_marker_precedes_start_and_failed_write_prevents_request(self) -> None:
        namespace = runpy.run_path(str(CHECK), run_name="aec_runner_contract")
        acquire = namespace["acquire_scope"]
        unit = "translator-aec-" + "7" * 32
        connection = mock.Mock()
        events: list[str] = []

        def call_sync(
            _owner: str,
            _path: str,
            _interface: str,
            method: str,
            _parameters: object,
            *_rest: object,
        ) -> GLib.Variant:
            events.append(method)
            if method == "GetNameOwner":
                return GLib.Variant("(s)", (":1.325",))
            if method == "StartTransientUnit":
                self.assertEqual(events[-2:], ["P", "StartTransientUnit"])
                raise Gio.DBusError.new_for_dbus_error(
                    "org.freedesktop.systemd1.UnitExists", "rejected"
                )
            if method == "GetUnit":
                raise Gio.DBusError.new_for_dbus_error(
                    "org.freedesktop.systemd1.NoSuchUnit", "absent"
                )
            if method == "ListJobs":
                return GLib.Variant("(a(usssoo))", ([],))
            return GLib.Variant("()", ())

        connection.call_sync.side_effect = call_sync
        connection.signal_subscribe.return_value = 1
        with mock.patch.object(
            Gio.DBusConnection, "new_for_address_sync", return_value=connection
        ):
            with self.assertRaises(namespace["ScopeRejected"]):
                acquire(
                    12355,
                    unit,
                    mock.Mock(),
                    mock.Mock(),
                    on_request=lambda _owner: events.append("P"),
                )
        self.assertEqual(events.count("P"), 1)
        events.clear()
        connection.reset_mock()
        connection.call_sync.side_effect = call_sync
        connection.signal_subscribe.return_value = 1
        with mock.patch.object(
            Gio.DBusConnection, "new_for_address_sync", return_value=connection
        ):
            with self.assertRaises(namespace["ScopeRejected"]):
                acquire(
                    12355,
                    unit,
                    mock.Mock(),
                    mock.Mock(),
                    on_request=mock.Mock(side_effect=OSError("closed marker")),
                )
        self.assertNotIn("StartTransientUnit", events)

    def test_connect_is_cancelled_before_any_manager_request(self) -> None:
        namespace = runpy.run_path(str(CHECK), run_name="aec_runner_contract")
        acquire = namespace["acquire_scope"]
        cancellation_seen = False

        def blocked_connect(*args: object) -> object:
            nonlocal cancellation_seen
            cancellable = args[-1]
            self.assertIsNotNone(cancellable)
            deadline = time.monotonic() + 1
            while not cancellable.is_cancelled() and time.monotonic() < deadline:
                time.sleep(0.001)
            cancellation_seen = cancellable.is_cancelled()
            raise GLib.Error(
                "connect cancelled",
                "g-io-error-quark",
                int(Gio.IOErrorEnum.CANCELLED),
            )

        on_request = mock.Mock()
        with (
            mock.patch.object(
                Gio.DBusConnection,
                "new_for_address_sync",
                side_effect=blocked_connect,
            ),
            mock.patch.dict(
                acquire.__globals__,
                {"MANAGER_PRE_REQUEST_TIMEOUT_SECONDS": 0.02},
            ),
        ):
            with self.assertRaises(namespace["ScopeRejected"]):
                acquire(
                    12356,
                    "translator-aec-" + "8" * 32,
                    mock.Mock(),
                    mock.Mock(),
                    on_request=on_request,
                )
        self.assertTrue(cancellation_seen)
        on_request.assert_not_called()

    def test_get_name_owner_is_cancelled_before_p_marker(self) -> None:
        namespace = runpy.run_path(str(CHECK), run_name="aec_runner_contract")
        acquire = namespace["acquire_scope"]
        connection = mock.Mock()
        cancellation_seen = False

        def blocked_owner(
            _owner: str,
            _path: str,
            _interface: str,
            method: str,
            _parameters: object,
            *_rest: object,
        ) -> GLib.Variant:
            nonlocal cancellation_seen
            self.assertEqual(method, "GetNameOwner")
            cancellable = _rest[-1]
            self.assertIsNotNone(cancellable)
            deadline = time.monotonic() + 1
            while not cancellable.is_cancelled() and time.monotonic() < deadline:
                time.sleep(0.001)
            cancellation_seen = cancellable.is_cancelled()
            raise GLib.Error(
                "owner lookup cancelled",
                "g-io-error-quark",
                int(Gio.IOErrorEnum.CANCELLED),
            )

        connection.call_sync.side_effect = blocked_owner
        on_request = mock.Mock()
        with (
            mock.patch.object(
                Gio.DBusConnection,
                "new_for_address_sync",
                return_value=connection,
            ),
            mock.patch.dict(
                acquire.__globals__,
                {"MANAGER_PRE_REQUEST_TIMEOUT_SECONDS": 0.02},
            ),
        ):
            with self.assertRaises(namespace["ScopeRejected"]):
                acquire(
                    12357,
                    "translator-aec-" + "9" * 32,
                    mock.Mock(),
                    mock.Mock(),
                    on_request=on_request,
                )
        self.assertTrue(cancellation_seen)
        on_request.assert_not_called()
        connection.signal_subscribe.assert_not_called()


if __name__ == "__main__":
    unittest.main()

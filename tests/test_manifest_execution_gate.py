from __future__ import annotations

import ast
import asyncio
import contextlib
import ctypes
import functools
import importlib.machinery
import importlib.util
import inspect
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import warnings
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "translator-test-manifest"
ISOLATED_TEST_ENV = "TRANSLATOR_INTERNAL_ISOLATED_MANIFEST_TEST"
ISOLATED_TEST_RECEIPT_PREFIX = "TRANSLATOR_ISOLATED_MANIFEST_TEST="
ISOLATED_TEST_WORKER_ENV = "TRANSLATOR_INTERNAL_ISOLATED_MANIFEST_TEST_WORKER"
ISOLATED_TEST_WORKER_RECEIPT_PREFIX = "TRANSLATOR_ISOLATED_MANIFEST_TEST_WORKER="
ISOLATED_TEST_PROBE_PATH_ENV = "TRANSLATOR_INTERNAL_ISOLATED_TEST_PROBE_PATH"
ISOLATED_TEST_PARENT_PID_ENV = "TRANSLATOR_INTERNAL_ISOLATED_TEST_PARENT_PID"
ISOLATED_TEST_TIMEOUT_SECONDS = 660
ISOLATED_TEST_OUTER_TIMEOUT_SECONDS = 690
_PR_SET_PDEATHSIG = 1
_PR_GET_CHILD_SUBREAPER = 37


def load_module(name: str, path: Path) -> ModuleType:
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


MANIFEST_RUNNER = load_module("translator_test_manifest_runtime", RUNNER_PATH)


def _child_subreaper_state() -> int:
    state = ctypes.c_int()
    libc = ctypes.CDLL(None, use_errno=True)
    prctl = libc.prctl
    prctl.restype = ctypes.c_int
    if prctl(_PR_GET_CHILD_SUBREAPER, ctypes.byref(state), 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "PR_GET_CHILD_SUBREAPER failed")
    return state.value


def _arm_parent_death_signal(expected_parent: int) -> None:
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGTERM})
    libc = ctypes.CDLL(None, use_errno=True)
    prctl = libc.prctl
    prctl.restype = ctypes.c_int
    if prctl(_PR_SET_PDEATHSIG, int(signal.SIGTERM), 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "PR_SET_PDEATHSIG failed")
    if os.getppid() != expected_parent:
        raise RuntimeError("isolated manifest test parent changed during admission")


def _reject_isolated_unittest_outcome_markers(callback: Callable[..., object]) -> None:
    if bool(getattr(callback, "__unittest_skip__", False)) or bool(
        getattr(callback, "__unittest_expecting_failure__", False)
    ):
        raise TypeError(
            "isolated process tests cannot use unittest skip or expected-failure markers"
        )


def _validate_isolated_test_class(test_class: type[unittest.TestCase]) -> None:
    isolated_callbacks = [
        callback
        for callback in vars(test_class).values()
        if getattr(callback, "_translator_process_isolated", False)
    ]
    if isolated_callbacks:
        _reject_isolated_unittest_outcome_markers(test_class)
    for callback in isolated_callbacks:
        _reject_isolated_unittest_outcome_markers(callback)


def _isolated_process_descendants(process_id: int) -> set[int]:
    descendants = MANIFEST_RUNNER._direct_child_pids(process_id)
    pending = list(descendants)
    while pending:
        child_id = pending.pop()
        for descendant_id in MANIFEST_RUNNER._direct_child_pids(child_id):
            if descendant_id not in descendants:
                descendants.add(descendant_id)
                pending.append(descendant_id)
    return descendants


def _isolated_process_is_quiescent(process_id: int) -> bool:
    try:
        status = (Path("/proc") / str(process_id) / "status").read_text(
            encoding="ascii"
        )
    except FileNotFoundError:
        return True
    state = next(
        (line.split()[1] for line in status.splitlines() if line.startswith("State:")),
        None,
    )
    if state is None:
        return False
    return state in {"T", "t", "X", "x", "Z"}


def _terminate_isolated_process(process: subprocess.Popen[str]) -> None:
    tracked: set[int] = set()
    quiesced = False
    quiescence_error: BaseException | None = None
    try:
        try:
            os.kill(process.pid, signal.SIGSTOP)
        except ProcessLookupError:
            pass
        stop_deadline = time.monotonic() + 5
        while time.monotonic() < stop_deadline and not _isolated_process_is_quiescent(
            process.pid
        ):
            time.sleep(0.01)
        while time.monotonic() < stop_deadline:
            descendants = _isolated_process_descendants(process.pid)
            tracked.update(descendants)
            for process_id in descendants:
                try:
                    os.kill(process_id, signal.SIGSTOP)
                except ProcessLookupError:
                    pass
            owned_processes = {process.pid, *tracked}
            if all(
                _isolated_process_is_quiescent(process_id)
                for process_id in owned_processes
            ):
                descendants = _isolated_process_descendants(process.pid)
                if descendants <= tracked:
                    quiesced = True
                    break
                tracked.update(descendants)
            time.sleep(0.01)
    except BaseException as error:
        quiescence_error = error
    with contextlib.suppress(MANIFEST_RUNNER.ManifestError, OSError):
        tracked.update(_isolated_process_descendants(process.pid))
    for process_id in tracked:
        try:
            os.kill(process_id, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    if process.poll() is None:
        process.kill()
    process.communicate(timeout=5)
    deadline = time.monotonic() + 5
    while tracked and time.monotonic() < deadline:
        remaining: set[int] = set()
        for process_id in tracked:
            try:
                os.kill(process_id, 0)
            except ProcessLookupError:
                continue
            remaining.add(process_id)
        tracked = remaining
        if tracked:
            time.sleep(0.01)
    if tracked:
        raise AssertionError("isolated manifest test process tree was not removed")
    if quiescence_error is not None:
        raise quiescence_error
    if not quiesced:
        raise AssertionError("isolated manifest test process tree did not quiesce")


def _run_isolated_test_supervisor(
    test_name: str,
    environment: dict[str, str],
    *,
    timeout_seconds: float = ISOLATED_TEST_OUTER_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess[str]:
    environment = dict(environment)
    environment[ISOLATED_TEST_PARENT_PID_ENV] = str(os.getpid())
    argv = [sys.executable, "-I", str(Path(__file__).resolve()), test_name]
    process = subprocess.Popen(
        argv,
        cwd=ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="strict",
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        _terminate_isolated_process(process)
        return subprocess.CompletedProcess(
            argv,
            124,
            "",
            "isolated manifest test supervisor timed out\n",
        )
    except BaseException:
        if process.poll() is None:
            _terminate_isolated_process(process)
        raise
    return subprocess.CompletedProcess(argv, process.returncode, stdout, stderr)


def isolated_process_test(
    method: Callable[[unittest.TestCase], None],
) -> Callable[[unittest.TestCase], None]:
    _reject_isolated_unittest_outcome_markers(method)
    if (
        inspect.isgeneratorfunction(method)
        or inspect.isasyncgenfunction(method)
        or inspect.iscoroutinefunction(method)
    ):
        raise TypeError("isolated process test body must be a synchronous function")

    @functools.wraps(method)
    def wrapper(test_case: unittest.TestCase) -> None:
        test_name = f"{type(test_case).__name__}.{method.__name__}"
        if (
            __name__ == "__main__"
            and os.environ.get(ISOLATED_TEST_WORKER_ENV) == test_name
        ):
            result = method(test_case)
            if result is None:
                return
            if inspect.isasyncgen(result):
                asyncio.run(result.aclose())
            elif inspect.isgenerator(result) or inspect.iscoroutine(result):
                result.close()
            raise AssertionError("isolated process test body returned a non-None value")

        subreaper_before = _child_subreaper_state()
        sigchld_before = signal.getsignal(signal.SIGCHLD)
        children_before = MANIFEST_RUNNER._direct_child_pids(os.getpid())
        tasks_before = {
            task.name for task in (Path("/proc") / str(os.getpid()) / "task").iterdir()
        }
        environment = MANIFEST_RUNNER._validation_environment(source=dict(os.environ))
        environment[ISOLATED_TEST_ENV] = test_name
        try:
            completed = _run_isolated_test_supervisor(test_name, environment)
        finally:
            test_case.assertEqual(_child_subreaper_state(), subreaper_before)
            test_case.assertIs(signal.getsignal(signal.SIGCHLD), sigchld_before)
            test_case.assertEqual(
                MANIFEST_RUNNER._direct_child_pids(os.getpid()), children_before
            )
            test_case.assertEqual(
                {
                    task.name
                    for task in (Path("/proc") / str(os.getpid()) / "task").iterdir()
                },
                tasks_before,
            )
        expected_stdout = ISOLATED_TEST_RECEIPT_PREFIX + test_name + "\n"
        test_case.assertEqual(
            completed.returncode,
            0,
            completed.stdout + completed.stderr,
        )
        test_case.assertEqual(completed.stdout, expected_stdout)
        test_case.assertEqual(completed.stderr, "")

    wrapper._translator_process_isolated = True
    return wrapper


class ManifestExecutionGateTests(unittest.TestCase):
    def assert_processes_gone(self, process_ids: list[int]) -> None:
        deadline = time.monotonic() + 1
        alive = process_ids
        while alive and time.monotonic() < deadline:
            remaining: list[int] = []
            for process_id in alive:
                try:
                    os.kill(process_id, 0)
                except ProcessLookupError:
                    continue
                remaining.append(process_id)
            alive = remaining
            if alive:
                time.sleep(0.01)
        for process_id in alive:
            os.kill(process_id, signal.SIGKILL)
        self.assertEqual(alive, [])

    def execute_single_receipted_child(
        self, child_source: str
    ) -> tuple[Exception | None, str, str]:
        command = {
            "id": "sidecar-test",
            "argv": [sys.executable, "-c", child_source],
            "cwd": ".",
            "env": {},
            "timeout_seconds": 900,
        }
        snapshot = MANIFEST_RUNNER._RepositorySnapshot(
            index=b"",
            files=(),
            refs=b"",
            head="a" * 40,
            tree="b" * 40,
            refs_sha256="c" * 64,
        )
        stored = {"collections": {"pytest": {"nodes": []}}}
        stdout = io.StringIO()
        stderr = io.StringIO()
        error: Exception | None = None
        with (
            mock.patch.object(MANIFEST_RUNNER, "EXPECTED_COMMANDS", [command]),
            mock.patch.object(MANIFEST_RUNNER, "_validate_execution_plan"),
            mock.patch.object(
                MANIFEST_RUNNER,
                "_load_stored_manifest_snapshot",
                return_value=(stored, b"manifest", "d" * 64),
            ),
            mock.patch.object(
                MANIFEST_RUNNER,
                "_load_allowlist_snapshot",
                return_value=({}, b"allowlist"),
            ),
            mock.patch.object(MANIFEST_RUNNER, "_verify_plan_binding"),
            mock.patch.object(
                MANIFEST_RUNNER,
                "_capture_repository_snapshot",
                return_value=snapshot,
            ),
            mock.patch.object(MANIFEST_RUNNER, "_verify_repository_snapshot"),
            mock.patch.object(MANIFEST_RUNNER, "_verify_manifest_snapshot_unchanged"),
            mock.patch.object(MANIFEST_RUNNER, "_verify_allowlist_snapshot_unchanged"),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            try:
                MANIFEST_RUNNER._execute_plan()
            except MANIFEST_RUNNER.ManifestError as caught:
                error = caught
        return error, stdout.getvalue(), stderr.getvalue()

    def execute_synthetic_unittest_suite(
        self, suite: unittest.TestSuite
    ) -> MANIFEST_RUNNER.ManifestError | None:
        leaves = list(MANIFEST_RUNNER._flatten_unittest_suite(suite))
        nodes = []
        for leaf in leaves:
            method = getattr(type(leaf), leaf._testMethodName)
            skipped = bool(
                getattr(type(leaf), "__unittest_skip__", False)
                or getattr(method, "__unittest_skip__", False)
            )
            reason = getattr(type(leaf), "__unittest_skip_why__", "") or getattr(
                method, "__unittest_skip_why__", ""
            )
            nodes.append(
                {
                    "id": leaf.id(),
                    "status": "external_skip" if skipped else "deterministic",
                    "reason": str(reason) if skipped else "",
                }
            )
        stored = {"collections": {"unittest": {"nodes": nodes}}}
        error: MANIFEST_RUNNER.ManifestError | None = None
        with (
            mock.patch.object(
                MANIFEST_RUNNER,
                "_bound_manifest_snapshot",
                return_value=(stored, b"manifest", "a" * 64, b"allowlist"),
            ),
            mock.patch.object(
                MANIFEST_RUNNER, "_discover_unittest_suite", return_value=suite
            ),
            mock.patch.object(MANIFEST_RUNNER, "_verify_manifest_snapshot_unchanged"),
            mock.patch.object(MANIFEST_RUNNER, "_verify_allowlist_snapshot_unchanged"),
            mock.patch.object(
                MANIFEST_RUNNER, "_print_outcome_receipt"
            ) as print_receipt,
            contextlib.redirect_stderr(io.StringIO()),
        ):
            try:
                MANIFEST_RUNNER._run_unittest_outcomes()
            except MANIFEST_RUNNER.ManifestError as caught:
                error = caught
        if error is None:
            print_receipt.assert_called_once()
        else:
            print_receipt.assert_not_called()
        return error

    def execute_synthetic_pytest_suite(
        self, synthetic_root: Path, node_id: str
    ) -> MANIFEST_RUNNER.ManifestError | None:
        stored = {
            "collections": {
                "pytest": {
                    "nodes": [{"id": node_id, "status": "deterministic", "reason": ""}]
                }
            }
        }
        error: MANIFEST_RUNNER.ManifestError | None = None
        with (
            mock.patch.object(MANIFEST_RUNNER, "ROOT", synthetic_root),
            mock.patch.object(
                MANIFEST_RUNNER,
                "_bound_manifest_snapshot",
                return_value=(stored, b"manifest", "a" * 64, b"allowlist"),
            ),
            mock.patch.object(MANIFEST_RUNNER, "_verify_manifest_snapshot_unchanged"),
            mock.patch.object(MANIFEST_RUNNER, "_verify_allowlist_snapshot_unchanged"),
            mock.patch.object(
                MANIFEST_RUNNER,
                "_pytest_test_directory",
                return_value=synthetic_root / "sidecar",
            ),
            mock.patch.object(
                MANIFEST_RUNNER, "_print_outcome_receipt"
            ) as print_receipt,
            mock.patch.dict(
                os.environ, {"PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}, clear=False
            ),
        ):
            try:
                MANIFEST_RUNNER._run_pytest_outcomes()
            except MANIFEST_RUNNER.ManifestError as caught:
                error = caught
        if error is None:
            print_receipt.assert_called_once()
        else:
            print_receipt.assert_not_called()
        return error

    def run_isolated_pytest_collection(
        self, synthetic_root: Path
    ) -> subprocess.CompletedProcess[str]:
        scripts_root = synthetic_root / "scripts"
        scripts_root.mkdir(exist_ok=True)
        runner_path = scripts_root / "translator-test-manifest"
        runner_path.write_bytes(RUNNER_PATH.read_bytes())
        return subprocess.run(
            [
                str(ROOT / "sidecar" / ".venv" / "bin" / "python"),
                "-I",
                str(runner_path),
                "_collect-pytest",
            ],
            cwd=synthetic_root / "sidecar",
            env=MANIFEST_RUNNER._validation_environment(source=dict(os.environ)),
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )

    def test_validation_environment_uses_an_explicit_allowlist(self) -> None:
        hostile = {
            "BASH_ENV": "/tmp/startup",
            "BASH_FUNC_fake%%": "() { exit 0; }",
            "ENV": "/tmp/startup",
            "GITHUB_PERSONAL_ACCESS_TOKEN": "synthetic-secret",
            "OPENAI_API_KEY": "synthetic-secret",
            "PYTHONPATH": "/tmp/modules",
            "PYTHONINSPECT": "1",
            "PYTEST_ADDOPTS": "--ignore=sidecar/tests",
            "PYTEST_PLUGINS": "hostile_plugin",
            "SHELLOPTS": "xtrace",
            "PS4": "trace ",
            "TRANSLATOR_GITLEAKS_BIN": "/trusted/gitleaks",
        }

        sanitized = MANIFEST_RUNNER._validation_environment(source=hostile)

        self.assertEqual(
            sanitized,
            {
                "LANG": "C.UTF-8",
                "LC_ALL": "C.UTF-8",
                "NO_COLOR": "1",
                "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
                "TZ": "UTC",
                "TRANSLATOR_GITLEAKS_BIN": "/trusted/gitleaks",
            },
        )
        for name, value in hostile.items():
            if name == "TRANSLATOR_GITLEAKS_BIN":
                continue
            with (
                self.subTest(name=name),
                self.assertRaises(MANIFEST_RUNNER.ManifestError),
            ):
                MANIFEST_RUNNER._validation_environment(
                    source={}, overrides={name: value}
                )

    def test_successful_child_cannot_mutate_later_gate_or_runner(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary) / "candidate"
            scripts = repository / "scripts"
            scripts.mkdir(parents=True)
            owned = {
                scripts / "translator-test-manifest": "trusted runner\n",
                scripts / "translator-publication-check": "trusted launcher\n",
                scripts / "translator-publication-check.bash": "trusted policy\n",
            }
            for path, content in owned.items():
                path.write_text(content, encoding="utf-8")
            for args in (
                ("init", "-q"),
                ("config", "user.email", "snapshot-test@example.invalid"),
                ("config", "user.name", "snapshot-test"),
                ("add", "-A"),
                ("commit", "-qm", "trusted candidate"),
            ):
                subprocess.run(
                    [
                        "/usr/bin/git",
                        "-c",
                        "gc.auto=0",
                        "-c",
                        "maintenance.auto=0",
                        *args,
                    ],
                    cwd=repository,
                    env=MANIFEST_RUNNER._trusted_git_environment(),
                    check=True,
                )

            with mock.patch.object(MANIFEST_RUNNER, "ROOT", repository):
                for target, original in owned.items():
                    with self.subTest(target=target.name):
                        snapshot = MANIFEST_RUNNER._capture_repository_snapshot()
                        child = subprocess.run(
                            [
                                sys.executable,
                                "-c",
                                (
                                    "from pathlib import Path; "
                                    f"Path({str(target)!r}).write_text('exit 0\\n')"
                                ),
                            ],
                            check=False,
                        )
                        self.assertEqual(child.returncode, 0)
                        with self.assertRaises(MANIFEST_RUNNER.ManifestError):
                            MANIFEST_RUNNER._verify_repository_snapshot(snapshot)
                        target.write_text(original, encoding="utf-8")

    def test_parent_requires_exact_publication_candidate_receipt(self) -> None:
        snapshot = MANIFEST_RUNNER._RepositorySnapshot(
            index=b"",
            files=(),
            refs=b"",
            head="a" * 40,
            tree="b" * 40,
            refs_sha256="c" * 64,
        )
        valid = (
            "publication precommit receipt: v1 "
            f"head={snapshot.head} tree={snapshot.tree} "
            f"refs-sha256={snapshot.refs_sha256} release=false\n"
            "publication precommit candidate only; this is not release evidence\n"
        )

        MANIFEST_RUNNER._verify_publication_receipt(valid, "", snapshot)
        for label, stdout, stderr in (
            ("empty", "", ""),
            ("trimmed", valid.rstrip(), ""),
            ("wrong mode", valid.replace("release=false", "release=true"), ""),
            ("unexpected stderr", valid, "unexpected"),
        ):
            with (
                self.subTest(label=label),
                self.assertRaises(MANIFEST_RUNNER.ManifestError),
            ):
                MANIFEST_RUNNER._verify_publication_receipt(stdout, stderr, snapshot)

    def test_manifest_check_requires_canonical_bytes_and_exact_types(self) -> None:
        expected = {"schema_version": 1, "value": "trusted"}
        canonical = MANIFEST_RUNNER._serialized(expected)
        variants = {
            "duplicate key": canonical.replace("{\n", '{\n  "schema_version": 1,\n', 1),
            "wrong scalar type": canonical.replace(
                '"schema_version": 1', '"schema_version": true', 1
            ),
            "leading whitespace": " " + canonical,
            "key order": json.dumps(
                {"value": "trusted", "schema_version": 1},
                indent=2,
                sort_keys=False,
            )
            + "\n",
        }
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = Path(temporary) / "manifest.json"
            with mock.patch.object(MANIFEST_RUNNER, "MANIFEST_PATH", manifest_path):
                manifest_path.write_text(canonical, encoding="ascii")
                MANIFEST_RUNNER._check(expected)
                for label, payload in variants.items():
                    with self.subTest(label=label):
                        manifest_path.write_text(payload, encoding="ascii")
                        with self.assertRaises(MANIFEST_RUNNER.ManifestError):
                            MANIFEST_RUNNER._check(expected)

    def test_allowlist_requires_integer_schema_version(self) -> None:
        with self.assertRaises(MANIFEST_RUNNER.ManifestError):
            MANIFEST_RUNNER._validate_allowlist_document(
                {"schema_version": True, "skip": []}
            )

    def test_execution_plan_requires_a_bounded_integer_timeout(self) -> None:
        for invalid_timeout in (True, 0, -1, 3601, "900"):
            commands = [dict(command) for command in MANIFEST_RUNNER.EXPECTED_COMMANDS]
            commands[0]["timeout_seconds"] = invalid_timeout
            with (
                self.subTest(timeout=invalid_timeout),
                mock.patch.object(MANIFEST_RUNNER, "EXPECTED_COMMANDS", commands),
                self.assertRaises(MANIFEST_RUNNER.ManifestError),
            ):
                MANIFEST_RUNNER._validate_execution_plan()

    def test_portability_allows_only_the_pinned_host_executable(self) -> None:
        MANIFEST_RUNNER._assert_portable("/usr/bin/systemd-analyze")

        for forbidden in (
            "/usr/bin/another-tool",
            "/" + "home/operator/tool",
            "C:" + "\\Users\\operator\\tool.exe",
        ):
            with (
                self.subTest(forbidden=forbidden),
                self.assertRaises(MANIFEST_RUNNER.ManifestError),
            ):
                MANIFEST_RUNNER._assert_portable(forbidden)

    @isolated_process_test
    def test_parent_rejects_successful_child_without_exact_receipt(self) -> None:
        stored, _, digest = MANIFEST_RUNNER._load_stored_manifest_snapshot()
        environment = MANIFEST_RUNNER._validation_environment(source=dict(os.environ))

        for child_source in (
            "pass",
            f"print({MANIFEST_RUNNER.OUTCOME_RECEIPT_PREFIX!r} + '{{}}')",
        ):
            command = {
                "argv": [sys.executable, "-c", child_source],
                "cwd": ".",
                "timeout_seconds": 30,
            }
            with contextlib.redirect_stdout(io.StringIO()):
                completed, payloads = MANIFEST_RUNNER._run_receipted_gate(
                    command, environment
                )
            self.assertEqual(completed.returncode, 0)
            with self.assertRaises(MANIFEST_RUNNER.ManifestError):
                MANIFEST_RUNNER._verify_outcome_receipt_payloads(
                    "pytest", payloads, stored, digest
                )

    @isolated_process_test
    def test_parent_rejects_resource_warning_emitted_at_child_shutdown(self) -> None:
        stored, _, digest = MANIFEST_RUNNER._load_stored_manifest_snapshot()
        environment = MANIFEST_RUNNER._validation_environment(source=dict(os.environ))
        for framework in ("pytest", "unittest"):
            canonical = json.dumps(
                MANIFEST_RUNNER._expected_outcome_receipt(framework, stored, digest),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            )
            child_source = (
                "leaks = [open('/dev/null')]\n"
                f"print({MANIFEST_RUNNER.OUTCOME_RECEIPT_PREFIX!r} + {canonical!r})\n"
            )
            command = {
                "argv": [
                    sys.executable,
                    "-I",
                    "-W",
                    "error::ResourceWarning",
                    "-c",
                    child_source,
                ],
                "cwd": ".",
                "timeout_seconds": 30,
            }

            with (
                self.subTest(framework=framework),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(MANIFEST_RUNNER.ManifestError),
            ):
                MANIFEST_RUNNER._run_receipted_gate(command, environment)

    @isolated_process_test
    def test_parent_rejects_exception_from_atexit_callback(self) -> None:
        stored, _, digest = MANIFEST_RUNNER._load_stored_manifest_snapshot()
        canonical = json.dumps(
            MANIFEST_RUNNER._expected_outcome_receipt("pytest", stored, digest),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        child_source = (
            "import atexit\n"
            "def fail_at_shutdown():\n"
            "    raise RuntimeError('synthetic shutdown failure')\n"
            "atexit.register(fail_at_shutdown)\n"
            f"print({MANIFEST_RUNNER.OUTCOME_RECEIPT_PREFIX!r} + {canonical!r})\n"
        )
        command = {
            "argv": [sys.executable, "-I", "-c", child_source],
            "cwd": ".",
            "timeout_seconds": 30,
        }
        environment = MANIFEST_RUNNER._validation_environment(source=dict(os.environ))

        with self.assertRaisesRegex(
            MANIFEST_RUNNER.ManifestError, "runtime resource issue"
        ):
            MANIFEST_RUNNER._run_receipted_gate(command, environment)

    @isolated_process_test
    def test_parent_rejects_generic_pytest_warning_at_child_shutdown(self) -> None:
        stored, _, digest = MANIFEST_RUNNER._load_stored_manifest_snapshot()
        canonical = json.dumps(
            MANIFEST_RUNNER._expected_outcome_receipt("pytest", stored, digest),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        child_source = (
            "import atexit, pytest, warnings\n"
            "def warn_at_shutdown():\n"
            "    warnings.warn('synthetic pytest warning', pytest.PytestWarning)\n"
            "atexit.register(warn_at_shutdown)\n"
            f"print({MANIFEST_RUNNER.OUTCOME_RECEIPT_PREFIX!r} + {canonical!r})\n"
        )
        command = {
            "argv": [sys.executable, "-I", "-c", child_source],
            "cwd": ".",
            "timeout_seconds": 30,
        }
        environment = MANIFEST_RUNNER._validation_environment(source=dict(os.environ))

        with self.assertRaisesRegex(
            MANIFEST_RUNNER.ManifestError, "runtime resource issue"
        ):
            MANIFEST_RUNNER._run_receipted_gate(command, environment)

    @isolated_process_test
    def test_parent_rejects_warning_subclass_at_child_shutdown(self) -> None:
        stored, _, digest = MANIFEST_RUNNER._load_stored_manifest_snapshot()
        canonical = json.dumps(
            MANIFEST_RUNNER._expected_outcome_receipt("pytest", stored, digest),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        child_source = (
            "import atexit, warnings\n"
            "class LeakNotice(ResourceWarning):\n"
            "    pass\n"
            "def warn_at_shutdown():\n"
            "    warnings.simplefilter('always', LeakNotice)\n"
            "    warnings.warn('late resource defect', LeakNotice)\n"
            "atexit.register(warn_at_shutdown)\n"
            f"print({MANIFEST_RUNNER.OUTCOME_RECEIPT_PREFIX!r} + {canonical!r})\n"
        )
        command = {
            "argv": [sys.executable, "-I", "-c", child_source],
            "cwd": ".",
            "timeout_seconds": 30,
        }
        environment = MANIFEST_RUNNER._validation_environment(source=dict(os.environ))

        with self.assertRaisesRegex(MANIFEST_RUNNER.ManifestError, "unexpected output"):
            MANIFEST_RUNNER._run_receipted_gate(command, environment)

    @isolated_process_test
    def test_parent_rejects_unawaited_coroutine_at_shutdown(self) -> None:
        stored, _, digest = MANIFEST_RUNNER._load_stored_manifest_snapshot()
        canonical = json.dumps(
            MANIFEST_RUNNER._expected_outcome_receipt("pytest", stored, digest),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        child_source = (
            "async def never_awaited():\n"
            "    return None\n"
            "pending = never_awaited()\n"
            f"print({MANIFEST_RUNNER.OUTCOME_RECEIPT_PREFIX!r} + {canonical!r})\n"
        )
        command = {
            "argv": [
                sys.executable,
                "-I",
                "-W",
                "error::RuntimeWarning",
                "-c",
                child_source,
            ],
            "cwd": ".",
            "timeout_seconds": 30,
        }
        environment = MANIFEST_RUNNER._validation_environment(source=dict(os.environ))

        with self.assertRaisesRegex(
            MANIFEST_RUNNER.ManifestError, "runtime resource issue"
        ):
            MANIFEST_RUNNER._run_receipted_gate(command, environment)

    @isolated_process_test
    def test_parent_rejects_pending_asyncio_task_at_shutdown(self) -> None:
        stored, _, digest = MANIFEST_RUNNER._load_stored_manifest_snapshot()
        canonical = json.dumps(
            MANIFEST_RUNNER._expected_outcome_receipt("pytest", stored, digest),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        child_source = (
            "import asyncio\n"
            "async def remain_pending():\n"
            "    await asyncio.sleep(3600)\n"
            "loop = asyncio.new_event_loop()\n"
            "task = loop.create_task(remain_pending())\n"
            "loop.run_until_complete(asyncio.sleep(0))\n"
            "loop.close()\n"
            f"print({MANIFEST_RUNNER.OUTCOME_RECEIPT_PREFIX!r} + {canonical!r})\n"
        )
        command = {
            "argv": [sys.executable, "-I", "-c", child_source],
            "cwd": ".",
            "timeout_seconds": 30,
        }
        environment = MANIFEST_RUNNER._validation_environment(source=dict(os.environ))

        with self.assertRaisesRegex(
            MANIFEST_RUNNER.ManifestError, "runtime resource issue"
        ):
            MANIFEST_RUNNER._run_receipted_gate(command, environment)

    def test_parent_requires_canonical_receipt_types_and_unique_keys(self) -> None:
        stored, _, digest = MANIFEST_RUNNER._load_stored_manifest_snapshot()
        valid = MANIFEST_RUNNER._expected_outcome_receipt("pytest", stored, digest)
        canonical = json.dumps(
            valid, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        wrong_types = {
            **valid,
            "schema_version": True,
            "contract_valid": 1,
            "exit_code": False,
        }
        duplicate_key = canonical.replace(
            '"schema_version":1', '"schema_version":false,"schema_version":1', 1
        )

        for label, payload in {
            "wrong scalar types": json.dumps(
                wrong_types,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ),
            "duplicate key": duplicate_key,
            "noncanonical whitespace": json.dumps(valid, sort_keys=True),
        }.items():
            with (
                self.subTest(label=label),
                self.assertRaises(MANIFEST_RUNNER.ManifestError),
            ):
                MANIFEST_RUNNER._verify_outcome_receipt_payloads(
                    "pytest", [payload], stored, digest
                )

    @isolated_process_test
    def test_parent_does_not_trim_child_receipt_payload(self) -> None:
        stored, _, digest = MANIFEST_RUNNER._load_stored_manifest_snapshot()
        valid = MANIFEST_RUNNER._expected_outcome_receipt("pytest", stored, digest)
        canonical = json.dumps(
            valid, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        child_source = (
            f"print({MANIFEST_RUNNER.OUTCOME_RECEIPT_PREFIX!r} + "
            f"'  ' + {canonical!r} + '  ')"
        )
        command = {
            "argv": [sys.executable, "-c", child_source],
            "cwd": ".",
            "timeout_seconds": 30,
        }
        environment = MANIFEST_RUNNER._validation_environment(source=dict(os.environ))

        with contextlib.redirect_stdout(io.StringIO()):
            completed, payloads = MANIFEST_RUNNER._run_receipted_gate(
                command, environment
            )

        self.assertEqual(completed.returncode, 0)
        self.assertEqual(payloads, [f"  {canonical}  "])
        with self.assertRaises(MANIFEST_RUNNER.ManifestError):
            MANIFEST_RUNNER._verify_outcome_receipt_payloads(
                "pytest", payloads, stored, digest
            )

    @isolated_process_test
    def test_parent_suppresses_output_until_child_and_receipt_are_trusted(self) -> None:
        canary = "PRIVATE_CHILD_CANARY_MUST_NOT_REACH_PARENT"
        child_sources = {
            "failed child": f"print({canary!r}); raise SystemExit(7)",
            "forged receipt": (
                f"print({canary!r}); "
                f"print({MANIFEST_RUNNER.OUTCOME_RECEIPT_PREFIX!r} + '{{}}')"
            ),
        }

        for label, child_source in child_sources.items():
            error, stdout, stderr = self.execute_single_receipted_child(child_source)
            with self.subTest(label=label):
                self.assertIsInstance(error, MANIFEST_RUNNER.ManifestError)
                self.assertNotIn(canary, stdout)
                self.assertNotIn(canary, stderr)

    @isolated_process_test
    def test_parent_rejects_successful_child_with_extra_stream_output(self) -> None:
        canary = "PRIVATE_SUCCESS_CANARY_MUST_NOT_REACH_PARENT"
        stored = {"collections": {"pytest": {"nodes": []}}}
        receipt = MANIFEST_RUNNER._expected_outcome_receipt("pytest", stored, "d" * 64)
        payload = json.dumps(
            receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        child_source = (
            "import sys\n"
            f"print({canary!r})\n"
            f"print({canary!r}, file=sys.stderr)\n"
            f"print({MANIFEST_RUNNER.OUTCOME_RECEIPT_PREFIX!r} + {payload!r})\n"
        )

        error, stdout, stderr = self.execute_single_receipted_child(child_source)

        self.assertIsInstance(error, MANIFEST_RUNNER.ManifestError)
        self.assertNotIn(canary, stdout)
        self.assertNotIn(canary, stderr)
        self.assertNotIn(MANIFEST_RUNNER.OUTCOME_RECEIPT_PREFIX + payload, stdout)

    @isolated_process_test
    def test_receipted_gate_converts_timeout_to_safe_failure(self) -> None:
        command = {
            "argv": [sys.executable, "-c", "pass"],
            "cwd": ".",
            "timeout_seconds": 7,
        }
        environment = MANIFEST_RUNNER._validation_environment(source=dict(os.environ))
        expired = subprocess.TimeoutExpired(command["argv"], 7)

        with (
            mock.patch.object(
                MANIFEST_RUNNER, "_run_gate_process", side_effect=expired
            ) as run,
            self.assertRaisesRegex(MANIFEST_RUNNER.ManifestError, "timed out"),
        ):
            MANIFEST_RUNNER._run_receipted_gate(command, environment)

        self.assertEqual(run.call_args.args, (command, environment))
        self.assertTrue(run.call_args.kwargs["capture_output"])

    @isolated_process_test
    def test_receipted_gate_reaps_child_stuck_on_non_daemon_thread(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            pid_path = Path(temporary) / "children.pid"
            child_source = (
                "import os, pathlib, subprocess, threading\n"
                f"grandchild = subprocess.Popen([{sys.executable!r}, '-c', "
                "'import time; time.sleep(3600)'])\n"
                f"pathlib.Path({str(pid_path)!r}).write_text("
                "f'{os.getpid()} {grandchild.pid}')\n"
                "threading.Thread(target=threading.Event().wait).start()\n"
            )
            command = {
                "argv": [sys.executable, "-c", child_source],
                "cwd": ".",
                "timeout_seconds": 1,
            }
            environment = MANIFEST_RUNNER._validation_environment(
                source=dict(os.environ)
            )

            started = time.monotonic()
            with self.assertRaisesRegex(MANIFEST_RUNNER.ManifestError, "timed out"):
                MANIFEST_RUNNER._run_receipted_gate(command, environment)
            elapsed = time.monotonic() - started

            self.assertLess(elapsed, 3)
            child_pids = [
                int(value) for value in pid_path.read_text(encoding="utf-8").split()
            ]
            self.assert_processes_gone(child_pids)

    @isolated_process_test
    def test_receipted_gate_reaps_descendant_after_parent_exits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            pid_path = Path(temporary) / "children.pid"
            child_source = (
                "import os, pathlib, subprocess\n"
                f"grandchild = subprocess.Popen([{sys.executable!r}, '-c', "
                "'import time; time.sleep(3600)'])\n"
                f"pathlib.Path({str(pid_path)!r}).write_text("
                "f'{os.getpid()} {grandchild.pid}')\n"
            )
            command = {
                "argv": [sys.executable, "-c", child_source],
                "cwd": ".",
                "timeout_seconds": 1,
            }
            environment = MANIFEST_RUNNER._validation_environment(
                source=dict(os.environ)
            )

            with self.assertRaisesRegex(MANIFEST_RUNNER.ManifestError, "timed out"):
                MANIFEST_RUNNER._run_receipted_gate(command, environment)

            child_pids = [
                int(value) for value in pid_path.read_text(encoding="utf-8").split()
            ]
            self.assert_processes_gone(child_pids)

    @isolated_process_test
    def test_gate_cancellation_reaps_parent_and_grandchild(self) -> None:
        for label, signal_number, expected_error in (
            ("SIGINT", signal.SIGINT, KeyboardInterrupt),
            ("SIGTERM", signal.SIGTERM, MANIFEST_RUNNER.ManifestError),
        ):
            with self.subTest(signal=label), tempfile.TemporaryDirectory() as temporary:
                pid_path = Path(temporary) / "children.pid"
                child_source = (
                    "import os, pathlib, signal, subprocess, time\n"
                    f"grandchild = subprocess.Popen([{sys.executable!r}, '-c', "
                    "'import time; time.sleep(3600)'])\n"
                    f"pathlib.Path({str(pid_path)!r}).write_text("
                    "f'{os.getpid()} {grandchild.pid}')\n"
                    f"os.kill(os.getppid(), {int(signal_number)})\n"
                    "time.sleep(3600)\n"
                )
                command = {
                    "argv": [sys.executable, "-c", child_source],
                    "cwd": ".",
                    "timeout_seconds": 30,
                }
                environment = MANIFEST_RUNNER._validation_environment(
                    source=dict(os.environ)
                )
                previous_sigterm = signal.getsignal(signal.SIGTERM)

                with self.assertRaises(expected_error) as raised:
                    MANIFEST_RUNNER._run_gate_process(
                        command,
                        environment,
                        capture_output=True,
                        encoding="utf-8",
                        errors="strict",
                    )

                if signal_number == signal.SIGTERM:
                    self.assertEqual(
                        str(raised.exception), "validation gate was interrupted"
                    )
                self.assertIs(signal.getsignal(signal.SIGTERM), previous_sigterm)
                child_pids = [
                    int(value) for value in pid_path.read_text(encoding="utf-8").split()
                ]
                self.assert_processes_gone(child_pids)

    @isolated_process_test
    def test_gate_child_does_not_inherit_blocked_termination_signals(self) -> None:
        child_source = (
            "import json, signal\n"
            "blocked = signal.pthread_sigmask(signal.SIG_BLOCK, [])\n"
            "print(json.dumps(sorted(int(value) for value in blocked)))\n"
        )
        command = {
            "argv": [sys.executable, "-c", child_source],
            "cwd": ".",
            "timeout_seconds": 30,
        }
        environment = MANIFEST_RUNNER._validation_environment(source=dict(os.environ))

        completed = MANIFEST_RUNNER._run_gate_process(
            command,
            environment,
            capture_output=True,
            encoding="utf-8",
            errors="strict",
        )

        blocked = json.loads(completed.stdout)
        self.assertNotIn(int(signal.SIGINT), blocked)
        self.assertNotIn(int(signal.SIGTERM), blocked)

    @isolated_process_test
    def test_gate_live_reaps_adopted_descendant_needed_by_child(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            process_path = Path(temporary) / "processes.txt"
            completion_marker = Path(temporary) / "group-gone"
            descendant_source = "import time; time.sleep(3600)"
            leader_source = (
                "import os, pathlib, subprocess\n"
                f"descendant = subprocess.Popen([{sys.executable!r}, '-c', "
                f"{descendant_source!r}], stdin=subprocess.DEVNULL, "
                "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
                f"pathlib.Path({str(process_path)!r}).write_text("
                "f'{os.getpid()} {os.getpgrp()} {descendant.pid} "
                "{os.getpgid(descendant.pid)}')\n"
                "os._exit(0)\n"
            )
            gate_source = (
                "import os, pathlib, signal, subprocess, sys, time\n"
                f"leader = subprocess.Popen([{sys.executable!r}, '-c', "
                f"{leader_source!r}], start_new_session=True, "
                "stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, "
                "stderr=subprocess.DEVNULL)\n"
                "leader.wait()\n"
                f"values = [int(value) for value in pathlib.Path("
                f"{str(process_path)!r}).read_text().split()]\n"
                "leader_pid, leader_pgid, _descendant_pid, descendant_pgid = values\n"
                "if (leader_pid, leader_pgid, descendant_pgid) != "
                "(leader.pid, leader.pid, leader.pid):\n"
                "    raise SystemExit(22)\n"
                "os.killpg(leader.pid, signal.SIGKILL)\n"
                "deadline = time.monotonic() + 1\n"
                "while time.monotonic() < deadline:\n"
                "    try:\n"
                "        os.killpg(leader.pid, 0)\n"
                "    except ProcessLookupError:\n"
                f"        pathlib.Path({str(completion_marker)!r}).touch()\n"
                "        time.sleep(0.25)\n"
                "        raise SystemExit(0)\n"
                "    time.sleep(0.01)\n"
                "raise SystemExit(23)\n"
            )
            command = {
                "argv": [sys.executable, "-c", gate_source],
                "cwd": ".",
                "timeout_seconds": 5,
            }
            environment = MANIFEST_RUNNER._validation_environment(
                source=dict(os.environ)
            )
            started = time.monotonic()
            completed = MANIFEST_RUNNER._run_gate_process(
                command,
                environment,
                capture_output=False,
            )

            self.assertEqual(completed.returncode, 0)
            self.assertLess(time.monotonic() - started, 2)
            self.assertTrue(completion_marker.exists())
            process_ids = [
                int(value) for value in process_path.read_text(encoding="utf-8").split()
            ]
            self.assert_processes_gone([process_ids[0], process_ids[2]])

    @isolated_process_test
    def test_gate_live_reaper_does_not_signal_running_adopted_child(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            process_path = Path(temporary) / "processes.txt"
            child_started = Path(temporary) / "child-started"
            child_finished = Path(temporary) / "child-finished"
            gate_finished = Path(temporary) / "gate-finished"
            descendant_source = (
                "import pathlib, time\n"
                f"pathlib.Path({str(child_started)!r}).touch()\n"
                "time.sleep(0.3)\n"
                f"pathlib.Path({str(child_finished)!r}).touch()\n"
            )
            leader_source = (
                "import os, pathlib, subprocess\n"
                f"descendant = subprocess.Popen([{sys.executable!r}, '-c', "
                f"{descendant_source!r}], stdin=subprocess.DEVNULL, "
                "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
                f"pathlib.Path({str(process_path)!r}).write_text("
                "f'{os.getpid()} {os.getpgrp()} {descendant.pid} "
                "{os.getpgid(descendant.pid)}')\n"
                "os._exit(0)\n"
            )
            gate_source = (
                "import os, pathlib, subprocess, sys, time\n"
                f"leader = subprocess.Popen([{sys.executable!r}, '-c', "
                f"{leader_source!r}], start_new_session=True, "
                "stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, "
                "stderr=subprocess.DEVNULL)\n"
                "leader.wait()\n"
                f"values = [int(value) for value in pathlib.Path("
                f"{str(process_path)!r}).read_text().split()]\n"
                "leader_pid, leader_pgid, descendant_pid, descendant_pgid = values\n"
                "if (leader_pid, leader_pgid, descendant_pgid) != "
                "(leader.pid, leader.pid, leader.pid):\n"
                "    raise SystemExit(31)\n"
                "deadline = time.monotonic() + 2\n"
                f"while not pathlib.Path({str(child_finished)!r}).exists():\n"
                "    try:\n"
                "        os.kill(descendant_pid, 0)\n"
                "    except ProcessLookupError:\n"
                "        raise SystemExit(32)\n"
                "    if time.monotonic() >= deadline:\n"
                "        raise SystemExit(33)\n"
                "    time.sleep(0.01)\n"
                "while time.monotonic() < deadline:\n"
                "    try:\n"
                "        os.killpg(leader.pid, 0)\n"
                "    except ProcessLookupError:\n"
                f"        pathlib.Path({str(gate_finished)!r}).touch()\n"
                "        raise SystemExit(0)\n"
                "    time.sleep(0.01)\n"
                "raise SystemExit(34)\n"
            )
            command = {
                "argv": [sys.executable, "-c", gate_source],
                "cwd": ".",
                "timeout_seconds": 5,
            }
            environment = MANIFEST_RUNNER._validation_environment(
                source=dict(os.environ)
            )

            completed = MANIFEST_RUNNER._run_gate_process(
                command,
                environment,
                capture_output=False,
            )

            self.assertEqual(completed.returncode, 0)
            self.assertTrue(child_started.exists())
            self.assertTrue(child_finished.exists())
            self.assertTrue(gate_finished.exists())
            process_ids = [
                int(value) for value in process_path.read_text(encoding="utf-8").split()
            ]
            self.assert_processes_gone([process_ids[0], process_ids[2]])

    def test_adopted_reaper_excludes_gate_child(self) -> None:
        with (
            mock.patch.object(
                MANIFEST_RUNNER,
                "_direct_child_pids",
                return_value={202, 303},
            ),
            mock.patch.object(MANIFEST_RUNNER.os, "waitpid") as waitpid,
        ):
            MANIFEST_RUNNER._reap_adopted_gate_children(exclude={202})

        waitpid.assert_called_once_with(303, os.WNOHANG)

    def test_process_tree_gate_tests_are_exec_isolated(self) -> None:
        process_tree_entrypoints = {
            "_build_manifest",
            "_collect_bun",
            "_collect_pytest",
            "_collect_rust",
            "_collect_unittest",
            "_enable_child_subreaper",
            "_execute_plan",
            "_json_collector",
            "_run_capture",
            "_run_gate_process",
            "_run_receipted_gate",
            "_verify_no_rust_doctests",
            "main",
        }
        process_tree_test_helpers = {"execute_single_receipted_child"}
        source = ast.parse(Path(__file__).read_text(encoding="utf-8"))
        test_class = next(
            node
            for node in source.body
            if isinstance(node, ast.ClassDef) and node.name == type(self).__name__
        )
        missing_isolation: list[str] = []
        for method in test_class.body:
            if not isinstance(method, ast.FunctionDef) or not method.name.startswith(
                "test_"
            ):
                continue
            reaches_process_tree = any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and (
                    (
                        isinstance(node.func.value, ast.Name)
                        and node.func.value.id == "MANIFEST_RUNNER"
                        and node.func.attr in process_tree_entrypoints
                    )
                    or (
                        isinstance(node.func.value, ast.Name)
                        and node.func.value.id == "self"
                        and node.func.attr in process_tree_test_helpers
                    )
                )
                for node in ast.walk(method)
            )
            callback = inspect.getattr_static(type(self), method.name)
            if reaches_process_tree and not getattr(
                callback, "_translator_process_isolated", False
            ):
                missing_isolation.append(method.name)

        self.assertEqual(missing_isolation, [])

    @isolated_process_test
    def test_subreaper_enable_is_process_local_after_fork(self) -> None:
        MANIFEST_RUNNER._enable_child_subreaper()
        read_fd, write_fd = os.pipe()
        child_pid = os.fork()
        if child_pid == 0:
            os.close(read_fd)
            try:
                before = _child_subreaper_state()
                MANIFEST_RUNNER._enable_child_subreaper()
                after = _child_subreaper_state()
                os.write(write_fd, f"{before},{after}".encode("ascii"))
            finally:
                os.close(write_fd)
            os._exit(0)

        os.close(write_fd)
        try:
            payload = os.read(read_fd, 32).decode("ascii")
        finally:
            os.close(read_fd)
        waited_pid, wait_status = os.waitpid(child_pid, 0)

        self.assertEqual(waited_pid, child_pid)
        self.assertEqual(os.waitstatus_to_exitcode(wait_status), 0)
        self.assertEqual(payload, "0,1")

    @isolated_process_test
    def test_parent_death_admission_normalizes_signal_and_detects_parent_race(
        self,
    ) -> None:
        previous_handler = signal.signal(signal.SIGTERM, signal.SIG_IGN)
        previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM})
        try:
            _arm_parent_death_signal(os.getppid())

            self.assertIs(signal.getsignal(signal.SIGTERM), signal.SIG_DFL)
            current_mask = signal.pthread_sigmask(signal.SIG_BLOCK, set())
            self.assertNotIn(signal.SIGTERM, current_mask)
            with self.assertRaisesRegex(RuntimeError, "parent changed"):
                _arm_parent_death_signal(os.getppid() + 1)
        finally:
            signal.signal(signal.SIGTERM, previous_handler)
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)

    def isolated_probe_leaves_detached_descendant(self) -> None:
        process_path = Path(os.environ[ISOLATED_TEST_PROBE_PATH_ENV])
        descendant = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(3600)"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        process_path.write_text(str(descendant.pid), encoding="ascii")

    def isolated_probe_hangs_with_detached_descendant(self) -> None:
        self.isolated_probe_leaves_detached_descendant()
        time.sleep(3600)

    def test_isolated_supervisor_contains_residual_and_timeout_trees(self) -> None:
        subreaper_before = _child_subreaper_state()
        sigchld_before = signal.getsignal(signal.SIGCHLD)
        children_before = MANIFEST_RUNNER._direct_child_pids(os.getpid())
        probes = (
            ("isolated_probe_leaves_detached_descendant", 10, 1),
            ("isolated_probe_hangs_with_detached_descendant", 1, 124),
        )
        for method_name, timeout_seconds, expected_returncode in probes:
            with self.subTest(probe=method_name), tempfile.TemporaryDirectory() as temp:
                process_path = Path(temp) / "descendant.pid"
                test_name = f"{type(self).__name__}.{method_name}"
                environment = MANIFEST_RUNNER._validation_environment(
                    source=dict(os.environ)
                )
                environment.update(
                    {
                        ISOLATED_TEST_ENV: test_name,
                        ISOLATED_TEST_PROBE_PATH_ENV: str(process_path),
                    }
                )

                completed = _run_isolated_test_supervisor(
                    test_name,
                    environment,
                    timeout_seconds=timeout_seconds,
                )

                self.assertEqual(completed.returncode, expected_returncode)
                self.assertEqual(completed.stdout, "")
                self.assertTrue(process_path.is_file())
                descendant_pid = int(process_path.read_text(encoding="ascii"))
                self.assert_processes_gone([descendant_pid])

        self.assertEqual(_child_subreaper_state(), subreaper_before)
        self.assertIs(signal.getsignal(signal.SIGCHLD), sigchld_before)
        self.assertEqual(
            MANIFEST_RUNNER._direct_child_pids(os.getpid()), children_before
        )

    @isolated_process_test
    def test_timeout_cleanup_discovers_fork_after_stopped_root_snapshot(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            child_path = temporary_path / "child.pid"
            trigger_path = temporary_path / "fork-now"
            descendant_path = temporary_path / "descendant.pid"
            descendant_path.touch()

            def wait_for_published_pid(path: Path) -> int:
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    try:
                        payload = path.read_text(encoding="ascii")
                    except FileNotFoundError:
                        pass
                    else:
                        raw_pid = payload.removesuffix("\n")
                        if (
                            payload.endswith("\n")
                            and raw_pid.isascii()
                            and raw_pid.isdigit()
                            and int(raw_pid) > 0
                        ):
                            return int(raw_pid)
                    time.sleep(0.001)
                raise AssertionError(f"{path.name} did not publish a valid PID")

            descendant_source = (
                "import pathlib, subprocess, sys, time\n"
                f"trigger = pathlib.Path({str(trigger_path)!r})\n"
                "while not trigger.exists():\n"
                "    time.sleep(0.001)\n"
                "descendant = subprocess.Popen(\n"
                "    [sys.executable, '-c', 'import time; time.sleep(3600)'],\n"
                "    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,\n"
                "    stderr=subprocess.DEVNULL, start_new_session=True,\n"
                ")\n"
                f"pathlib.Path({str(descendant_path)!r}).write_text(\n"
                "    str(descendant.pid) + '\\n', encoding='ascii'\n"
                ")\n"
                "time.sleep(3600)\n"
            )
            root_source = (
                "import pathlib, subprocess, sys, time\n"
                "child = subprocess.Popen(\n"
                f"    [sys.executable, '-c', {descendant_source!r}],\n"
                "    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,\n"
                "    stderr=subprocess.DEVNULL, start_new_session=True,\n"
                ")\n"
                f"pathlib.Path({str(child_path)!r}).write_text(\n"
                "    str(child.pid) + '\\n', encoding='ascii'\n"
                ")\n"
                "time.sleep(3600)\n"
            )
            root_process = subprocess.Popen(
                [sys.executable, "-c", root_source],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
                start_new_session=True,
            )
            child_pid: int | None = None
            descendant_pid: int | None = None
            snapshot_before_fork: set[int] = set()
            inspect_descendants = _isolated_process_descendants
            try:
                child_pid = wait_for_published_pid(child_path)
                self.assertIsNone(root_process.poll())

                def fork_after_snapshot(process_id: int) -> set[int]:
                    nonlocal descendant_pid, snapshot_before_fork
                    snapshot = inspect_descendants(process_id)
                    if (
                        process_id == root_process.pid
                        and not trigger_path.exists()
                        and _isolated_process_is_quiescent(root_process.pid)
                    ):
                        snapshot_before_fork = set(snapshot)
                        trigger_path.touch()
                        descendant_pid = wait_for_published_pid(descendant_path)
                    return snapshot

                with mock.patch(
                    f"{__name__}._isolated_process_descendants",
                    side_effect=fork_after_snapshot,
                ):
                    _terminate_isolated_process(root_process)

                self.assertIsNotNone(descendant_pid)
                self.assertNotIn(descendant_pid, snapshot_before_fork)
                self.assert_processes_gone([child_pid, descendant_pid])
                child_pid = None
                descendant_pid = None
            finally:
                if root_process.poll() is None:
                    _terminate_isolated_process(root_process)
                for process_id in (child_pid, descendant_pid):
                    if process_id is None:
                        continue
                    try:
                        os.kill(process_id, signal.SIGKILL)
                    except ProcessLookupError:
                        pass

    def test_timeout_cleanup_kills_root_before_reporting_quiescence_timeout(
        self,
    ) -> None:
        process = mock.Mock()
        process.pid = 424_242
        process.poll.return_value = None
        process.communicate.return_value = ("", "")
        with (
            mock.patch.object(time, "monotonic", side_effect=(0, 6, 6, 6)),
            mock.patch.object(os, "kill") as kill,
            mock.patch.object(os, "killpg") as killpg,
            mock.patch(f"{__name__}._isolated_process_descendants", return_value=set()),
            self.assertRaisesRegex(AssertionError, "did not quiesce"),
        ):
            _terminate_isolated_process(process)

        kill.assert_called_once_with(process.pid, signal.SIGSTOP)
        killpg.assert_called_once_with(process.pid, signal.SIGKILL)
        process.kill.assert_called_once_with()
        process.communicate.assert_called_once_with(timeout=5)

    def test_timeout_cleanup_kills_root_before_reporting_inspection_failure(
        self,
    ) -> None:
        process = mock.Mock()
        process.pid = 434_343
        process.poll.return_value = None
        process.communicate.return_value = ("", "")
        with (
            mock.patch.object(os, "kill") as kill,
            mock.patch.object(os, "killpg") as killpg,
            mock.patch(
                f"{__name__}._isolated_process_descendants",
                side_effect=MANIFEST_RUNNER.ManifestError("inspection failed"),
            ),
            self.assertRaisesRegex(MANIFEST_RUNNER.ManifestError, "inspection failed"),
        ):
            _terminate_isolated_process(process)

        kill.assert_called_once_with(process.pid, signal.SIGSTOP)
        killpg.assert_called_once_with(process.pid, signal.SIGKILL)
        process.kill.assert_called_once_with()
        process.communicate.assert_called_once_with(timeout=5)

    @isolated_process_test
    def test_isolated_supervisor_contains_tree_when_outer_parent_is_terminated(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            process_path = Path(temporary) / "descendant.pid"
            command = [
                sys.executable,
                "-I",
                str(Path(__file__).resolve()),
                "--outer-signal-probe",
                str(process_path),
            ]
            harness = subprocess.Popen(
                command,
                cwd=ROOT,
                env=MANIFEST_RUNNER._validation_environment(source=dict(os.environ)),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="strict",
                start_new_session=True,
            )
            tracked_processes: set[int] = set()
            try:
                deadline = time.monotonic() + 10
                detached_pid: int | None = None
                descendants: set[int] = set()
                while time.monotonic() < deadline and harness.poll() is None:
                    descendants = _isolated_process_descendants(harness.pid)
                    tracked_processes.update(descendants)
                    if process_path.is_file():
                        try:
                            detached_pid = int(process_path.read_text(encoding="ascii"))
                        except ValueError:
                            detached_pid = None
                        if detached_pid is not None:
                            tracked_processes.add(detached_pid)
                            if detached_pid in descendants and len(descendants) >= 3:
                                break
                    time.sleep(0.01)

                self.assertIsNone(harness.poll())
                self.assertIsNotNone(detached_pid)
                self.assertIn(detached_pid, descendants)
                self.assertGreaterEqual(len(descendants), 3)

                os.kill(harness.pid, signal.SIGTERM)
                stdout, stderr = harness.communicate(timeout=10)

                self.assertEqual(harness.returncode, -signal.SIGTERM)
                self.assertEqual(stdout, "")
                self.assertEqual(stderr, "")
                self.assert_processes_gone(sorted(tracked_processes))
                tracked_processes.clear()
            finally:
                if harness.poll() is None:
                    _terminate_isolated_process(harness)
                else:
                    harness.communicate(timeout=5)
                for process_id in tracked_processes:
                    try:
                        os.kill(process_id, signal.SIGKILL)
                    except ProcessLookupError:
                        pass

    def test_isolated_proxy_runs_with_a_live_shared_host_thread(self) -> None:
        release = threading.Event()
        shared_host_thread = threading.Thread(target=release.wait)
        shared_host_thread.start()
        try:
            isolated_callback = inspect.getattr_static(
                type(self),
                "test_gate_child_does_not_inherit_blocked_termination_signals",
            )
            isolated_callback(self)
        finally:
            release.set()
            shared_host_thread.join(timeout=5)

        self.assertFalse(shared_host_thread.is_alive())

    def test_isolated_test_receipt_requires_one_clean_pass(self) -> None:
        class RuntimeSkipTests(unittest.TestCase):
            def test_runtime_skip(self) -> None:
                self.skipTest("synthetic runtime skip")

        class ExpectedFailureTests(unittest.TestCase):
            @unittest.expectedFailure
            def test_expected_failure(self) -> None:
                self.fail("synthetic expected failure")

        class UnexpectedSuccessTests(unittest.TestCase):
            @unittest.expectedFailure
            def test_unexpected_success(self) -> None:
                pass

        for test_case in (
            RuntimeSkipTests("test_runtime_skip"),
            ExpectedFailureTests("test_expected_failure"),
            UnexpectedSuccessTests("test_unexpected_success"),
        ):
            test_name = test_case.id()
            stdout = io.StringIO()
            stderr = io.StringIO()
            with (
                self.subTest(test=test_name),
                mock.patch.dict(os.environ, {ISOLATED_TEST_WORKER_ENV: test_name}),
                mock.patch.object(
                    unittest.defaultTestLoader,
                    "loadTestsFromName",
                    return_value=unittest.TestSuite([test_case]),
                ),
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
            ):
                exit_code = _isolated_test_worker_main(test_name)

            self.assertEqual(exit_code, 1)
            self.assertEqual(stdout.getvalue(), "")
            self.assertIn("did not produce one clean pass", stderr.getvalue())

        for label, suite in (
            ("empty", unittest.TestSuite()),
            (
                "multiple",
                unittest.TestSuite(
                    [
                        unittest.FunctionTestCase(lambda: None),
                        unittest.FunctionTestCase(lambda: None),
                    ]
                ),
            ),
        ):
            stdout = io.StringIO()
            stderr = io.StringIO()
            test_name = f"synthetic.{label}"
            with (
                self.subTest(selection=label),
                mock.patch.dict(os.environ, {ISOLATED_TEST_WORKER_ENV: test_name}),
                mock.patch.object(
                    unittest.defaultTestLoader,
                    "loadTestsFromName",
                    return_value=suite,
                ),
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
            ):
                exit_code = _isolated_test_worker_main(test_name)

            self.assertEqual(exit_code, 1)
            self.assertEqual(stdout.getvalue(), "")
            self.assertEqual(
                stderr.getvalue(), "isolated manifest test selection failed\n"
            )

    def test_isolated_process_decorator_rejects_hidden_body_modes(self) -> None:
        def generator_body(_test_case: unittest.TestCase):
            yield "assertion never executed"

        async def async_body(_test_case: unittest.TestCase) -> None:
            raise AssertionError("assertion never awaited")

        async def async_generator_body(_test_case: unittest.TestCase):
            yield "assertion never iterated"

        for body in (generator_body, async_body, async_generator_body):
            with (
                self.subTest(body=body.__name__),
                self.assertRaisesRegex(TypeError, "synchronous function"),
            ):
                isolated_process_test(body)

        def lazy_generator():
            yield "assertion never iterated"

        async def lazy_coroutine() -> None:
            raise AssertionError("assertion never awaited")

        async def lazy_async_generator():
            yield "assertion never iterated"

        def scalar_body(_test_case: unittest.TestCase) -> int:
            return 1

        def returned_generator(_test_case: unittest.TestCase):
            return lazy_generator()

        def returned_coroutine(_test_case: unittest.TestCase):
            return lazy_coroutine()

        def returned_async_generator(_test_case: unittest.TestCase):
            return lazy_async_generator()

        for body in (
            scalar_body,
            returned_generator,
            returned_coroutine,
            returned_async_generator,
        ):
            wrapped = isolated_process_test(body)
            synthetic_case = unittest.FunctionTestCase(lambda: None)
            test_name = f"{type(synthetic_case).__name__}.{body.__name__}"
            with (
                self.subTest(returned=body.__name__),
                mock.patch.dict(wrapped.__globals__, {"__name__": "__main__"}),
                mock.patch.dict(os.environ, {ISOLATED_TEST_WORKER_ENV: test_name}),
                self.assertRaisesRegex(AssertionError, "returned a non-None value"),
            ):
                wrapped(synthetic_case)

    def test_isolated_process_decorator_rejects_unittest_outcome_markers(
        self,
    ) -> None:
        def callback() -> Callable[[unittest.TestCase], None]:
            def test_body(_test_case: unittest.TestCase) -> None:
                pass

            return test_body

        for label, marker in (
            ("skip", unittest.skip("synthetic skip")),
            ("expected_failure", unittest.expectedFailure),
        ):
            with (
                self.subTest(order="inside", marker=label),
                self.assertRaisesRegex(TypeError, "cannot use unittest"),
            ):
                isolated_process_test(marker(callback()))

            isolated = isolated_process_test(callback())
            marked_isolated = marker(isolated)

            class MarkedMethodTests(unittest.TestCase):
                test_body = marked_isolated

            with (
                self.subTest(order="outside", marker=label),
                self.assertRaisesRegex(TypeError, "cannot use unittest"),
            ):
                _validate_isolated_test_class(MarkedMethodTests)

            class MarkedClassTests(unittest.TestCase):
                test_body = isolated_process_test(callback())

            marked_class = marker(MarkedClassTests)
            with (
                self.subTest(order="class", marker=label),
                self.assertRaisesRegex(TypeError, "cannot use unittest"),
            ):
                _validate_isolated_test_class(marked_class)

    @isolated_process_test
    def test_gate_rejects_preexisting_child_without_stealing_status(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            command_marker = Path(temporary) / "gate-ran"
            command = {
                "argv": [
                    sys.executable,
                    "-c",
                    f"import pathlib; pathlib.Path({str(command_marker)!r}).touch()",
                ],
                "cwd": ".",
                "timeout_seconds": 5,
            }
            environment = MANIFEST_RUNNER._validation_environment(
                source=dict(os.environ)
            )
            existing_child = subprocess.Popen(
                [sys.executable, "-c", "raise SystemExit(7)"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            try:
                with (
                    mock.patch.object(
                        MANIFEST_RUNNER, "_enable_child_subreaper"
                    ) as enable_subreaper,
                    self.assertRaisesRegex(
                        MANIFEST_RUNNER.ManifestError, "exclusive process ownership"
                    ),
                ):
                    MANIFEST_RUNNER._run_gate_process(
                        command,
                        environment,
                        capture_output=False,
                    )

                enable_subreaper.assert_not_called()
                waited_pid, wait_status = os.waitpid(existing_child.pid, 0)
                existing_child.returncode = os.waitstatus_to_exitcode(wait_status)
                self.assertEqual(waited_pid, existing_child.pid)
                self.assertEqual(existing_child.returncode, 7)
                self.assertFalse(command_marker.exists())
            finally:
                if existing_child.returncode is None:
                    existing_child.wait(timeout=5)

    @isolated_process_test
    def test_gate_rejects_multithreaded_process_owner_before_spawn(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            command_marker = Path(temporary) / "gate-ran"
            command = {
                "argv": [
                    sys.executable,
                    "-c",
                    f"import pathlib; pathlib.Path({str(command_marker)!r}).touch()",
                ],
                "cwd": ".",
                "timeout_seconds": 5,
            }
            environment = MANIFEST_RUNNER._validation_environment(
                source=dict(os.environ)
            )
            release = threading.Event()
            worker = threading.Thread(target=release.wait)
            worker.start()
            try:
                with (
                    mock.patch.object(
                        MANIFEST_RUNNER, "_enable_child_subreaper"
                    ) as enable_subreaper,
                    self.assertRaisesRegex(
                        MANIFEST_RUNNER.ManifestError, "exclusive process ownership"
                    ),
                ):
                    MANIFEST_RUNNER._run_gate_process(
                        command,
                        environment,
                        capture_output=False,
                    )
                enable_subreaper.assert_not_called()
                self.assertFalse(command_marker.exists())
            finally:
                release.set()
                worker.join(timeout=5)
            self.assertFalse(worker.is_alive())

    @isolated_process_test
    def test_gate_rejects_ignored_sigchld_before_spawn(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            command_marker = Path(temporary) / "gate-ran"
            command = {
                "argv": [
                    sys.executable,
                    "-c",
                    f"import pathlib; pathlib.Path({str(command_marker)!r}).touch()",
                ],
                "cwd": ".",
                "timeout_seconds": 5,
            }
            environment = MANIFEST_RUNNER._validation_environment(
                source=dict(os.environ)
            )

            with (
                mock.patch.object(
                    MANIFEST_RUNNER, "_enable_child_subreaper"
                ) as enable_subreaper,
                mock.patch.object(
                    MANIFEST_RUNNER.signal,
                    "getsignal",
                    return_value=signal.SIG_IGN,
                ),
                self.assertRaisesRegex(
                    MANIFEST_RUNNER.ManifestError, "exclusive process ownership"
                ),
            ):
                MANIFEST_RUNNER._run_gate_process(
                    command,
                    environment,
                    capture_output=False,
                )

            enable_subreaper.assert_not_called()
            self.assertFalse(command_marker.exists())

    @isolated_process_test
    def test_exclusive_owner_runs_sequential_gates_after_subreaper_enable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            environment = MANIFEST_RUNNER._validation_environment(
                source=dict(os.environ)
            )
            markers = [Path(temporary) / f"gate-{index}" for index in range(2)]

            for marker in markers:
                completed = MANIFEST_RUNNER._run_gate_process(
                    {
                        "argv": [
                            sys.executable,
                            "-c",
                            f"import pathlib; pathlib.Path({str(marker)!r}).touch()",
                        ],
                        "cwd": ".",
                        "timeout_seconds": 5,
                    },
                    environment,
                    capture_output=False,
                )
                self.assertEqual(completed.returncode, 0)

            self.assertTrue(all(marker.exists() for marker in markers))

    @isolated_process_test
    def test_successful_gate_rejects_and_reaps_background_process(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            pid_path = Path(temporary) / "children.pid"
            child_source = (
                "import os, pathlib, subprocess\n"
                f"grandchild = subprocess.Popen([{sys.executable!r}, '-c', "
                "'import time; time.sleep(3600)'], "
                "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
                f"pathlib.Path({str(pid_path)!r}).write_text("
                "f'{os.getpid()} {grandchild.pid}')\n"
            )
            command = {
                "argv": [sys.executable, "-c", child_source],
                "cwd": ".",
                "timeout_seconds": 30,
            }
            environment = MANIFEST_RUNNER._validation_environment(
                source=dict(os.environ)
            )

            with self.assertRaisesRegex(
                MANIFEST_RUNNER.ManifestError, "left background processes"
            ):
                MANIFEST_RUNNER._run_gate_process(
                    command,
                    environment,
                    capture_output=True,
                    encoding="utf-8",
                    errors="strict",
                )

            child_pids = [
                int(value) for value in pid_path.read_text(encoding="utf-8").split()
            ]
            self.assert_processes_gone(child_pids)

    @isolated_process_test
    def test_receipted_gate_rejects_and_reaps_detached_descendant(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            pid_path = Path(temporary) / "detached.pid"
            stored = {"collections": {"pytest": {"nodes": []}}}
            receipt = MANIFEST_RUNNER._expected_outcome_receipt(
                "pytest", stored, "d" * 64
            )
            payload = json.dumps(
                receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=True
            )
            child_source = (
                "import pathlib, subprocess\n"
                f"grandchild = subprocess.Popen([{sys.executable!r}, '-c', "
                "'import time; time.sleep(3600)'], start_new_session=True, "
                "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
                f"pathlib.Path({str(pid_path)!r}).write_text(str(grandchild.pid))\n"
                f"print({MANIFEST_RUNNER.OUTCOME_RECEIPT_PREFIX!r} + {payload!r})\n"
            )
            command = {
                "argv": [sys.executable, "-c", child_source],
                "cwd": ".",
                "timeout_seconds": 30,
            }
            environment = MANIFEST_RUNNER._validation_environment(
                source=dict(os.environ)
            )

            with self.assertRaisesRegex(
                MANIFEST_RUNNER.ManifestError, "left background processes"
            ):
                MANIFEST_RUNNER._run_receipted_gate(command, environment)

            self.assert_processes_gone([int(pid_path.read_text(encoding="utf-8"))])

    @isolated_process_test
    def test_signal_during_residual_group_check_reaps_background_process(self) -> None:
        for label, signal_number, expected_error in (
            ("SIGINT", signal.SIGINT, KeyboardInterrupt),
            ("SIGTERM", signal.SIGTERM, MANIFEST_RUNNER.ManifestError),
        ):
            with self.subTest(signal=label), tempfile.TemporaryDirectory() as temporary:
                pid_path = Path(temporary) / "children.pid"
                child_source = (
                    "import os, pathlib, subprocess\n"
                    f"grandchild = subprocess.Popen([{sys.executable!r}, '-c', "
                    "'import time; time.sleep(3600)'], "
                    "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
                    f"pathlib.Path({str(pid_path)!r}).write_text("
                    "f'{os.getpid()} {grandchild.pid}')\n"
                )
                command = {
                    "argv": [sys.executable, "-c", child_source],
                    "cwd": ".",
                    "timeout_seconds": 30,
                }
                environment = MANIFEST_RUNNER._validation_environment(
                    source=dict(os.environ)
                )
                original_probe = MANIFEST_RUNNER._gate_process_group_exists
                sent_signal = False

                def interrupt_first_probe(
                    process_group: int,
                    active_signal: signal.Signals = signal_number,
                    probe: Callable[[int], bool] = original_probe,
                ) -> bool:
                    nonlocal sent_signal
                    if not sent_signal:
                        sent_signal = True
                        os.kill(os.getpid(), active_signal)
                    return probe(process_group)

                with (
                    mock.patch.object(
                        MANIFEST_RUNNER,
                        "_gate_process_group_exists",
                        side_effect=interrupt_first_probe,
                    ),
                    self.assertRaises(expected_error),
                ):
                    MANIFEST_RUNNER._run_gate_process(
                        command,
                        environment,
                        capture_output=True,
                        encoding="utf-8",
                        errors="strict",
                    )

                self.assertTrue(sent_signal)
                child_pids = [
                    int(value) for value in pid_path.read_text(encoding="utf-8").split()
                ]
                self.assert_processes_gone(child_pids)

    @isolated_process_test
    def test_invalid_utf8_gate_reaps_reaped_leader_background_process(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            pid_path = Path(temporary) / "children.pid"
            child_source = (
                "import os, pathlib, subprocess, sys\n"
                f"grandchild = subprocess.Popen([{sys.executable!r}, '-c', "
                "'import time; time.sleep(3600)'], "
                "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
                f"pathlib.Path({str(pid_path)!r}).write_text("
                "f'{os.getpid()} {grandchild.pid}')\n"
                "sys.stdout.buffer.write(bytes([255]))\n"
                "sys.stdout.buffer.flush()\n"
            )
            command = {
                "argv": [sys.executable, "-c", child_source],
                "cwd": ".",
                "timeout_seconds": 30,
            }
            environment = MANIFEST_RUNNER._validation_environment(
                source=dict(os.environ)
            )

            with self.assertRaisesRegex(MANIFEST_RUNNER.ManifestError, "invalid UTF-8"):
                MANIFEST_RUNNER._run_receipted_gate(command, environment)

            child_pids = [
                int(value) for value in pid_path.read_text(encoding="utf-8").split()
            ]
            self.assert_processes_gone(child_pids)

    @isolated_process_test
    def test_collector_timeout_reaps_process_group(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            pid_path = Path(temporary) / "children.pid"
            child_source = (
                "import os, pathlib, subprocess, time\n"
                f"grandchild = subprocess.Popen([{sys.executable!r}, '-c', "
                "'import time; time.sleep(3600)'])\n"
                f"pathlib.Path({str(pid_path)!r}).write_text("
                "f'{os.getpid()} {grandchild.pid}')\n"
                "time.sleep(3600)\n"
            )

            with self.assertRaisesRegex(MANIFEST_RUNNER.ManifestError, "timed out"):
                MANIFEST_RUNNER._run_capture(
                    [sys.executable, "-c", child_source],
                    cwd=ROOT,
                    timeout=1,
                )

            child_pids = [
                int(value) for value in pid_path.read_text(encoding="utf-8").split()
            ]
            self.assert_processes_gone(child_pids)

    @isolated_process_test
    def test_hostile_pytest_environment_cannot_change_collection(self) -> None:
        baseline = MANIFEST_RUNNER._collect_pytest()
        hostile = {
            "PYTEST_ADDOPTS": "--ignore=tests/test_local_asr.py",
            "PYTEST_PLUGINS": "module_that_must_not_be_imported",
        }

        with mock.patch.dict(os.environ, hostile):
            observed = MANIFEST_RUNNER._collect_pytest()

        self.assertEqual(observed, baseline)

    @isolated_process_test
    def test_pytest_collection_warning_invalidates_collection(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="test_manifest_warning_", dir=ROOT / "sidecar" / "tests"
        ) as temporary:
            test_path = Path(temporary) / "test_uncollected_class.py"
            test_path.write_text(
                "class TestSilentlyUncollected:\n"
                "    def __init__(self):\n"
                "        pass\n"
                "    def test_hidden(self):\n"
                "        assert False\n",
                encoding="utf-8",
            )

            with self.assertRaises(MANIFEST_RUNNER.ManifestError):
                MANIFEST_RUNNER._collect_pytest()

    @isolated_process_test
    def test_pytest_collection_resource_warning_invalidates_collection(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="test_manifest_resource_", dir=ROOT / "sidecar" / "tests"
        ) as temporary:
            test_path = Path(temporary) / "test_import_resource_leak.py"
            test_path.write_text(
                "import warnings\n"
                "warnings.warn('collection resource', ResourceWarning)\n"
                "def test_body():\n"
                "    pass\n",
                encoding="utf-8",
            )

            with self.assertRaises(MANIFEST_RUNNER.ManifestError):
                MANIFEST_RUNNER._collect_pytest()

    @isolated_process_test
    def test_pytest_collection_unraisable_resource_blocks_receipt(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="test_manifest_unraisable_", dir=ROOT / "sidecar" / "tests"
        ) as temporary:
            test_path = Path(temporary) / "test_import_resource_leak.py"
            test_path.write_text(
                "import gc\n"
                "leaked = open('/dev/null')\n"
                "del leaked\n"
                "gc.collect()\n"
                "def test_body():\n"
                "    pass\n",
                encoding="utf-8",
            )

            with self.assertRaises(MANIFEST_RUNNER.ManifestError):
                MANIFEST_RUNNER._collect_pytest()

    @isolated_process_test
    def test_pytest_repository_control_plugin_blocks_collection(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="test_manifest_plugin_", dir=ROOT / "sidecar" / "tests"
        ) as temporary:
            (Path(temporary) / "conftest.py").write_text(
                "def pytest_pyfunc_call(pyfuncitem):\n"
                "    del pyfuncitem\n"
                "    return True\n",
                encoding="utf-8",
            )

            with self.assertRaises(MANIFEST_RUNNER.ManifestError):
                MANIFEST_RUNNER._collect_pytest()

    def test_pytest_callable_replacement_cannot_hide_test_body(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            synthetic_root = Path(temporary)
            sidecar = synthetic_root / "sidecar"
            sidecar.mkdir()
            marker = synthetic_root / "body-ran"
            (sidecar / "test_callable_replacement.py").write_text(
                "from pathlib import Path\n"
                "import pytest\n"
                "@pytest.fixture(autouse=True)\n"
                "def replace_body(request):\n"
                "    request.node._obj = lambda: None\n"
                "def test_original_body():\n"
                f"    Path({str(marker)!r}).write_text('ran')\n"
                "    assert False\n",
                encoding="utf-8",
            )

            error = self.execute_synthetic_pytest_suite(
                synthetic_root,
                "sidecar/test_callable_replacement.py::test_original_body",
            )

            self.assertIsInstance(error, MANIFEST_RUNNER.ManifestError)
            self.assertFalse(marker.exists())

    def test_pytest_preserves_request_function_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            synthetic_root = Path(temporary)
            sidecar = synthetic_root / "sidecar"
            sidecar.mkdir()
            marker = synthetic_root / "body-ran"
            (sidecar / "test_request_identity.py").write_text(
                "from pathlib import Path\n"
                "def test_original(request):\n"
                "    assert request.function is test_original\n"
                f"    Path({str(marker)!r}).write_text('ran')\n",
                encoding="utf-8",
            )

            error = self.execute_synthetic_pytest_suite(
                synthetic_root,
                "sidecar/test_request_identity.py::test_original",
            )

            self.assertIsNone(error)
            self.assertEqual(marker.read_text(encoding="utf-8"), "ran")

    @isolated_process_test
    def test_sidecar_pytest_rejects_unittest_compatibility_items(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            synthetic_root = Path(temporary)
            sidecar = synthetic_root / "sidecar"
            sidecar.mkdir()
            marker = synthetic_root / "body-ran"
            (sidecar / "test_unittest_generator.py").write_text(
                "from pathlib import Path\n"
                "import unittest\n"
                "class CompatibilityTests(unittest.TestCase):\n"
                "    def test_hidden(self):\n"
                f"        Path({str(marker)!r}).write_text('ran')\n"
                "        self.fail('hidden assertion')\n"
                "        yield None\n",
                encoding="utf-8",
            )

            error = self.execute_synthetic_pytest_suite(
                synthetic_root,
                "sidecar/test_unittest_generator.py::CompatibilityTests::test_hidden",
            )

            self.assertIsInstance(error, MANIFEST_RUNNER.ManifestError)
            self.assertFalse(marker.exists())

        cases = {
            "plain": (
                "import unittest\n"
                "class CompatibilityTests(unittest.TestCase):\n"
                "    def test_body(self):\n"
                "        pass\n"
            ),
            "subtest": (
                "import unittest\n"
                "class CompatibilityTests(unittest.TestCase):\n"
                "    def test_body(self):\n"
                "        with self.subTest(value=1):\n"
                "            self.assertEqual(1, 1)\n"
            ),
            "isolated_asyncio": (
                "import unittest\n"
                "class CompatibilityTests(unittest.IsolatedAsyncioTestCase):\n"
                "    async def test_body(self):\n"
                "        self.assertTrue(True)\n"
            ),
            "declared_skip": (
                "import unittest\n"
                "class CompatibilityTests(unittest.TestCase):\n"
                "    @unittest.skip('external')\n"
                "    def test_body(self):\n"
                "        pass\n"
            ),
        }
        for label, source in cases.items():
            with (
                self.subTest(mode=label),
                tempfile.TemporaryDirectory(
                    prefix=f"test_manifest_unittest_{label}_",
                    dir=ROOT / "sidecar" / "tests",
                ) as temporary,
            ):
                (Path(temporary) / "test_unittest_item.py").write_text(
                    source,
                    encoding="utf-8",
                )

                with self.assertRaises(MANIFEST_RUNNER.ManifestError):
                    MANIFEST_RUNNER._collect_pytest()

        unsupported_collection_cases = {
            "doctest": (
                "test_contract.txt",
                ">>> 1 + 1\n2\n",
                "",
            ),
            "pytest_asyncio": (
                "test_asyncio_mode.py",
                "import pytest\n"
                "@pytest.mark.asyncio\n"
                "async def test_async_body():\n"
                "    raise AssertionError('must not execute')\n",
                "[tool.pytest.ini_options]\naddopts = '-p pytest_asyncio.plugin'\n",
            ),
        }
        for label, (
            filename,
            source,
            configuration,
        ) in unsupported_collection_cases.items():
            with self.subTest(mode=label), tempfile.TemporaryDirectory() as temporary:
                synthetic_root = Path(temporary)
                tests_root = synthetic_root / "sidecar" / "tests"
                tests_root.mkdir(parents=True)
                (tests_root / filename).write_text(source, encoding="utf-8")
                if configuration:
                    (synthetic_root / "sidecar" / "pyproject.toml").write_text(
                        configuration,
                        encoding="utf-8",
                    )

                with mock.patch.dict(
                    os.environ,
                    {"PYTEST_PLUGINS": "pytest_asyncio.plugin"},
                    clear=False,
                ):
                    result = self.run_isolated_pytest_collection(synthetic_root)

                self.assertEqual(result.returncode, 1)
                self.assertEqual(result.stdout, "")
                self.assertEqual(
                    result.stderr,
                    "test manifest error: pytest collection failed\n",
                )

    def test_first_active_marker_validates_complete_one_shot_iterator(self) -> None:
        inactive = MANIFEST_RUNNER.SimpleNamespace(
            args=(False,), kwargs={"reason": "inactive"}
        )
        first_active = MANIFEST_RUNNER.SimpleNamespace(
            args=(True,), kwargs={"reason": "first active"}
        )
        second_active = MANIFEST_RUNNER.SimpleNamespace(
            args=(True,), kwargs={"reason": "second active"}
        )
        dynamic = MANIFEST_RUNNER.SimpleNamespace(
            args=("False",), kwargs={"reason": "dynamic"}
        )

        selected = MANIFEST_RUNNER._first_active_marker(
            iter((inactive, first_active, second_active))
        )

        self.assertIs(selected, first_active)
        for markers in ((first_active, dynamic), (dynamic, first_active)):
            with (
                self.subTest(markers=markers),
                self.assertRaises(MANIFEST_RUNNER.ManifestError),
            ):
                MANIFEST_RUNNER._first_active_marker(iter(markers))

    def test_pytest_skipif_requires_resolved_boolean_conditions(self) -> None:
        parent_tasks_before = {
            task.name for task in (Path("/proc") / str(os.getpid()) / "task").iterdir()
        }
        with tempfile.TemporaryDirectory() as temporary:
            synthetic_root = Path(temporary)
            tests_root = synthetic_root / "sidecar" / "tests"
            tests_root.mkdir(parents=True)
            test_path = tests_root / "test_boolean_skipif.py"
            test_path.write_text(
                "import pytest\n"
                "@pytest.mark.skipif(condition=False, reason='must stay active')\n"
                "def test_keyword_false():\n"
                "    pass\n"
                "@pytest.mark.skipif(False, reason='must stay active')\n"
                "def test_positional_false():\n"
                "    pass\n"
                "@pytest.mark.skip(reason='plain skip')\n"
                "@pytest.mark.skipif(True, reason='active condition')\n"
                "def test_active_skipif_precedes_plain_skip():\n"
                "    pass\n",
                encoding="utf-8",
            )

            result = self.run_isolated_pytest_collection(synthetic_root)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stderr, "")
            document = json.loads(
                result.stdout.removeprefix(MANIFEST_RUNNER.COLLECTOR_RECEIPT_PREFIX)
            )
            selected = document["rows"]
            self.assertEqual(len(selected), 3)
            rows_by_id = {row["id"].rsplit("::", 1)[-1]: row for row in selected}
            self.assertEqual(
                rows_by_id["test_keyword_false"]["status"], "deterministic"
            )
            self.assertEqual(
                rows_by_id["test_positional_false"]["status"], "deterministic"
            )
            self.assertEqual(
                rows_by_id["test_active_skipif_precedes_plain_skip"],
                {
                    "id": (
                        "sidecar/tests/test_boolean_skipif.py::"
                        "test_active_skipif_precedes_plain_skip"
                    ),
                    "file": "sidecar/tests/test_boolean_skipif.py",
                    "status": "external_skip",
                    "reason": "active condition",
                },
            )

        invalid_marker_stacks = {
            "single_dynamic": (
                "@pytest.mark.skipif('False', reason='dynamic expression')\n"
            ),
            "dynamic_outside_active": (
                "@pytest.mark.skipif('False', reason='dynamic expression')\n"
                "@pytest.mark.skipif(True, reason='active condition')\n"
            ),
            "active_outside_dynamic": (
                "@pytest.mark.skipif(True, reason='active condition')\n"
                "@pytest.mark.skipif('False', reason='dynamic expression')\n"
            ),
            "xfail_dynamic_outside_active": (
                "@pytest.mark.xfail('False', reason='dynamic expression')\n"
                "@pytest.mark.xfail(True, reason='active condition')\n"
            ),
            "xfail_active_outside_dynamic": (
                "@pytest.mark.xfail(True, reason='active condition')\n"
                "@pytest.mark.xfail('False', reason='dynamic expression')\n"
            ),
        }
        for label, decorators in invalid_marker_stacks.items():
            with (
                self.subTest(markers=label),
                tempfile.TemporaryDirectory() as temporary,
            ):
                synthetic_root = Path(temporary)
                tests_root = synthetic_root / "sidecar" / "tests"
                tests_root.mkdir(parents=True)
                (tests_root / "test_string_skipif.py").write_text(
                    "import pytest\n"
                    f"{decorators}"
                    "def test_string_condition():\n"
                    "    pass\n",
                    encoding="utf-8",
                )

                result = self.run_isolated_pytest_collection(synthetic_root)

                self.assertEqual(result.returncode, 1)
                self.assertEqual(result.stdout, "")
                self.assertEqual(
                    result.stderr,
                    "test manifest error: pytest collection failed\n",
                )

        parent_tasks_after = {
            task.name for task in (Path("/proc") / str(os.getpid()) / "task").iterdir()
        }
        self.assertEqual(parent_tasks_after, parent_tasks_before)

    def test_pytest_uses_controlled_collection_not_repository_ini_options(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            synthetic_root = Path(temporary)
            tests_root = synthetic_root / "sidecar" / "tests"
            tests_root.mkdir(parents=True)
            (synthetic_root / "sidecar" / "pyproject.toml").write_text(
                "[tool.pytest.ini_options]\n"
                "addopts = '--ignore=tests/test_hidden.py -p missing_plugin'\n"
                "python_files = ['never_collect_*.py']\n",
                encoding="utf-8",
            )
            for name in ("test_visible.py", "test_hidden.py"):
                (tests_root / name).write_text(
                    f"def test_{name.removeprefix('test_').removesuffix('.py')}():\n"
                    "    pass\n",
                    encoding="utf-8",
                )

            result = self.run_isolated_pytest_collection(synthetic_root)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stderr, "")
            self.assertTrue(
                result.stdout.startswith(MANIFEST_RUNNER.COLLECTOR_RECEIPT_PREFIX)
            )
            document = json.loads(
                result.stdout.removeprefix(MANIFEST_RUNNER.COLLECTOR_RECEIPT_PREFIX)
            )
            self.assertEqual(len(document["rows"]), 2)
            self.assertEqual(
                document["known_files"],
                [
                    "sidecar/tests/test_hidden.py",
                    "sidecar/tests/test_visible.py",
                ],
            )

    def test_pytest_rejects_a_test_source_without_collected_items(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            synthetic_root = Path(temporary)
            tests_root = synthetic_root / "sidecar" / "tests"
            tests_root.mkdir(parents=True)
            (tests_root / "test_visible.py").write_text(
                "def test_visible():\n    pass\n", encoding="utf-8"
            )
            (tests_root / "test_empty.py").write_text(
                "HELPER_ONLY = True\n", encoding="utf-8"
            )

            result = self.run_isolated_pytest_collection(synthetic_root)

            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(result.stdout, "")

    def test_sidecar_pytest_rejects_xunit_style_fixtures(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            synthetic_root = Path(temporary)
            tests_root = synthetic_root / "sidecar" / "tests"
            tests_root.mkdir(parents=True)
            for fixture_name in ("setup_module", "setUpModule"):
                (tests_root / "test_xunit.py").write_text(
                    f"def {fixture_name}():\n    pass\ndef test_body():\n    pass\n",
                    encoding="utf-8",
                )
                with self.subTest(fixture=fixture_name):
                    result = self.run_isolated_pytest_collection(synthetic_root)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertEqual(result.stdout, "")

    def test_pytest_xunit_detection_matches_dynamic_pytest_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            synthetic_root = Path(temporary)
            tests_root = synthetic_root / "sidecar" / "tests"
            tests_root.mkdir(parents=True)
            cases = {
                "descriptor": (
                    "class SetupDescriptor:\n"
                    "    def __get__(self, instance, owner):\n"
                    "        return lambda *args: None\n"
                    "class TestExample:\n"
                    "    setup_method = SetupDescriptor()\n"
                    "    def test_body(self):\n"
                    "        pass\n"
                ),
                "module_getattr": (
                    "def __getattr__(name):\n"
                    "    if name == 'setup_module':\n"
                    "        return lambda *args: None\n"
                    "    raise AttributeError(name)\n"
                    "def test_body():\n"
                    "    pass\n"
                ),
            }
            for name, source in cases.items():
                (tests_root / "test_xunit.py").write_text(source, encoding="utf-8")
                with self.subTest(case=name):
                    result = self.run_isolated_pytest_collection(synthetic_root)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertEqual(result.stdout, "")

    def test_pytest_fixture_named_like_xunit_callback_remains_supported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            synthetic_root = Path(temporary)
            tests_root = synthetic_root / "sidecar" / "tests"
            tests_root.mkdir(parents=True)
            (tests_root / "test_fixture.py").write_text(
                "import pytest\n"
                "@pytest.fixture\n"
                "def setup_module():\n"
                "    return 1\n"
                "def test_body(setup_module):\n"
                "    assert setup_module == 1\n",
                encoding="utf-8",
            )

            result = self.run_isolated_pytest_collection(synthetic_root)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stderr, "")

    def test_unittest_collection_rejects_module_load_tests_hooks(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="test_manifest_load_tests_", dir=ROOT / "tests"
        ) as temporary:
            test_root = Path(temporary)
            (test_root / "__init__.py").write_text("", encoding="utf-8")
            (test_root / "test_hidden.py").write_text(
                "import unittest\n"
                "class HiddenTests(unittest.TestCase):\n"
                "    def test_hidden(self):\n"
                "        raise AssertionError('must not be filtered')\n"
                "def load_tests(loader, tests, pattern):\n"
                "    return unittest.TestSuite()\n",
                encoding="utf-8",
            )

            try:
                with self.assertRaises(MANIFEST_RUNNER.ManifestError):
                    MANIFEST_RUNNER._unittest_collector_rows()
            finally:
                resolved_root = test_root.resolve()
                for module_name, module in tuple(sys.modules.items()):
                    module_file = getattr(module, "__file__", None)
                    if module_file is not None and Path(
                        module_file
                    ).resolve().is_relative_to(resolved_root):
                        sys.modules.pop(module_name, None)

    @isolated_process_test
    def test_collector_requires_one_isolated_canonical_receipt(self) -> None:
        document = {
            "known_files": ["tests/test_contract.py"],
            "rows": [
                {
                    "file": "tests/test_contract.py",
                    "id": "tests.test_contract.ContractTests.test_ok",
                    "reason": "",
                    "status": "deterministic",
                }
            ],
        }
        canonical = MANIFEST_RUNNER._collector_receipt(document)

        with mock.patch.object(
            MANIFEST_RUNNER, "_run_capture", return_value=canonical
        ) as run_capture:
            rows, known_files = MANIFEST_RUNNER._json_collector(
                "_collect-unittest", cwd=ROOT, timeout=1
            )

        self.assertEqual(rows, document["rows"])
        self.assertEqual(known_files, document["known_files"])
        self.assertIn("-I", run_capture.call_args.args[0])
        self.assertTrue(run_capture.call_args.kwargs["require_empty_stderr"])
        with self.assertRaises(MANIFEST_RUNNER.ManifestError):
            MANIFEST_RUNNER._run_capture(
                [
                    sys.executable,
                    "-I",
                    "-c",
                    "import sys; print('unexpected', file=sys.stderr)",
                ],
                cwd=ROOT,
                timeout=10,
                require_empty_stderr=True,
            )
        for payload in (
            "unexpected stdout\n" + canonical,
            canonical + canonical,
            canonical.replace(":", ": ", 1),
            MANIFEST_RUNNER._collector_receipt(
                {"known_files": [1], "rows": document["rows"]}
            ),
            MANIFEST_RUNNER._collector_receipt(
                {"known_files": document["known_files"], "rows": [{"id": "x"}]}
            ),
        ):
            with (
                self.subTest(payload=payload[:40]),
                mock.patch.object(
                    MANIFEST_RUNNER, "_run_capture", return_value=payload
                ),
                self.assertRaises(MANIFEST_RUNNER.ManifestError),
            ):
                MANIFEST_RUNNER._json_collector(
                    "_collect-unittest", cwd=ROOT, timeout=1
                )

    def test_pytest_unraisable_resource_warning_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix=".manifest-pytest-resource-", dir=ROOT / "tests"
        ) as temporary:
            test_root = Path(temporary)
            test_path = test_root / "test_resource_leak.py"
            test_path.write_text(
                "import gc\n"
                "def test_leaks_file():\n"
                "    open('/dev/null')\n"
                "    gc.collect()\n",
                encoding="utf-8",
            )
            recorder = MANIFEST_RUNNER._PytestOutcomeRecorder(
                base=test_root,
                blocked_warning_types=(
                    ResourceWarning,
                    pytest.PytestUnraisableExceptionWarning,
                ),
            )
            guard = MANIFEST_RUNNER._UnraisableIssueGuard()
            with (
                guard,
                contextlib.chdir(test_root),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                exit_code = pytest.main(
                    [
                        "-q",
                        "-p",
                        "no:cacheprovider",
                        "--rootdir",
                        str(test_root),
                        str(test_path),
                    ],
                    plugins=[recorder],
                )

            self.assertEqual(exit_code, pytest.ExitCode.OK)
            issues = [*recorder.runtime_issues, *guard.issues]
            self.assertTrue(issues)
            deterministic = [
                {"id": node_id, "status": "deterministic", "reason": ""}
                for node_id in recorder.collected
            ]
            with self.assertRaises(MANIFEST_RUNNER.ManifestError):
                MANIFEST_RUNNER._verify_execution_outcomes(
                    "pytest",
                    deterministic,
                    recorder.collected,
                    recorder.outcomes(),
                    issues=issues,
                )

    def test_pytest_returned_generator_warning_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix=".manifest-pytest-return-", dir=ROOT / "tests"
        ) as temporary:
            test_root = Path(temporary)
            test_path = test_root / "test_returned_generator.py"
            test_path.write_text(
                "def hidden_assertions():\n"
                "    assert False\n"
                "    yield None\n"
                "def test_hidden_assertions():\n"
                "    return hidden_assertions()\n",
                encoding="utf-8",
            )
            recorder = MANIFEST_RUNNER._PytestOutcomeRecorder(
                base=test_root,
                blocked_warning_types=MANIFEST_RUNNER._blocked_pytest_warning_types(
                    pytest
                ),
            )
            guard = MANIFEST_RUNNER._UnraisableIssueGuard()
            with (
                guard,
                contextlib.chdir(test_root),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                exit_code = pytest.main(
                    [
                        "-q",
                        "-p",
                        "no:cacheprovider",
                        "--rootdir",
                        str(test_root),
                        str(test_path),
                    ],
                    plugins=[recorder],
                )

            self.assertEqual(exit_code, pytest.ExitCode.OK)
            self.assertIn("blocked runtime warning", recorder.runtime_issues)
            with self.assertRaises(MANIFEST_RUNNER.ManifestError):
                MANIFEST_RUNNER._verify_execution_outcomes(
                    "pytest",
                    [
                        {
                            "id": node_id,
                            "status": "deterministic",
                            "reason": "",
                        }
                        for node_id in recorder.collected
                    ],
                    recorder.collected,
                    recorder.outcomes(),
                    issues=[*recorder.runtime_issues, *guard.issues],
                )

    def test_pytest_filterwarnings_marker_cannot_hide_return_warning(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix=".manifest-pytest-filter-", dir=ROOT / "tests"
        ) as temporary:
            test_root = Path(temporary)
            test_path = test_root / "test_filter_warning_bypass.py"
            test_path.write_text(
                "import pytest\n"
                "@pytest.mark.filterwarnings(\n"
                "    'ignore::pytest.PytestReturnNotNoneWarning'\n"
                ")\n"
                "def test_hidden_assertions():\n"
                "    return (value for value in ())\n",
                encoding="utf-8",
            )
            recorder = MANIFEST_RUNNER._PytestOutcomeRecorder(
                base=test_root,
                blocked_warning_types=MANIFEST_RUNNER._blocked_pytest_warning_types(
                    pytest
                ),
            )
            guard = MANIFEST_RUNNER._UnraisableIssueGuard()
            with (
                guard,
                contextlib.chdir(test_root),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                exit_code = pytest.main(
                    [
                        "-q",
                        "-p",
                        "no:cacheprovider",
                        "--rootdir",
                        str(test_root),
                        str(test_path),
                    ],
                    plugins=[recorder],
                )

            self.assertEqual(exit_code, pytest.ExitCode.OK)
            self.assertIn("pytest warning filter marker", recorder.collection_issues)
            with self.assertRaises(MANIFEST_RUNNER.ManifestError):
                MANIFEST_RUNNER._verify_execution_outcomes(
                    "pytest",
                    [
                        {
                            "id": node_id,
                            "status": "deterministic",
                            "reason": "",
                        }
                        for node_id in recorder.collected
                    ],
                    recorder.collected,
                    recorder.outcomes(),
                    issues=[
                        *recorder.collection_issues,
                        *recorder.runtime_issues,
                        *guard.issues,
                    ],
                )
            sys.modules.pop("test_filter_warning_bypass", None)

    def test_pytest_child_thread_resource_warning_blocks_child_receipt(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix=".manifest-pytest-thread-", dir=ROOT / "tests"
        ) as temporary:
            test_root = Path(temporary)
            test_path = test_root / "test_thread_resource.py"
            test_path.write_text(
                "import threading\n"
                "import warnings\n"
                "def leak():\n"
                "    warnings.warn('thread leak', ResourceWarning)\n"
                "def test_thread_resource_warning():\n"
                "    worker = threading.Thread(target=leak)\n"
                "    worker.start()\n"
                "    worker.join()\n",
                encoding="utf-8",
            )
            recorder = MANIFEST_RUNNER._PytestOutcomeRecorder(
                base=test_root,
                blocked_warning_types=MANIFEST_RUNNER._blocked_pytest_warning_types(
                    pytest
                ),
            )
            guard = MANIFEST_RUNNER._UnraisableIssueGuard()
            with (
                guard,
                contextlib.chdir(test_root),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                exit_code = pytest.main(
                    [
                        "-q",
                        "-p",
                        "no:cacheprovider",
                        "--rootdir",
                        str(test_root),
                        str(test_path),
                    ],
                    plugins=[recorder],
                )

            self.assertEqual(exit_code, pytest.ExitCode.OK)
            self.assertIn("blocked runtime warning", recorder.runtime_issues)
            sys.modules.pop("test_thread_resource", None)

    def test_unittest_unraisable_resource_warning_is_rejected(self) -> None:
        class LeakingResourceTests(unittest.TestCase):
            def test_leaks_file(self) -> None:
                open("/dev/null")

        suite = unittest.defaultTestLoader.loadTestsFromTestCase(LeakingResourceTests)
        collected = {
            MANIFEST_RUNNER._unittest_test_id(test)
            for test in MANIFEST_RUNNER._flatten_unittest_suite(suite)
        }
        guard = MANIFEST_RUNNER._UnraisableIssueGuard()
        with guard:
            result = unittest.TextTestRunner(
                stream=io.StringIO(),
                buffer=True,
                resultclass=MANIFEST_RUNNER._UnittestOutcomeResult,
            ).run(suite)

        self.assertTrue(result.wasSuccessful())
        self.assertTrue(guard.issues)
        with self.assertRaises(MANIFEST_RUNNER.ManifestError):
            MANIFEST_RUNNER._verify_execution_outcomes(
                "unittest",
                [
                    {"id": node_id, "status": "deterministic", "reason": ""}
                    for node_id in collected
                ],
                collected,
                result.outcomes(),
                issues=[
                    *MANIFEST_RUNNER._unittest_execution_issues(result, collected),
                    *guard.issues,
                ],
            )

    def test_unittest_child_thread_resource_warning_is_rejected(self) -> None:
        class ThreadResourceTests(unittest.TestCase):
            def test_thread_resource_warning(self) -> None:
                def leak() -> None:
                    warnings.warn(
                        "synthetic thread leak", ResourceWarning, stacklevel=2
                    )

                worker = threading.Thread(target=leak)
                worker.start()
                worker.join()

        suite = unittest.defaultTestLoader.loadTestsFromTestCase(ThreadResourceTests)
        guard = MANIFEST_RUNNER._UnraisableIssueGuard()
        with guard:
            result = unittest.TextTestRunner(
                stream=io.StringIO(),
                buffer=True,
                resultclass=MANIFEST_RUNNER._UnittestOutcomeResult,
            ).run(suite)

        self.assertTrue(result.wasSuccessful())
        self.assertIn("unhandled thread exception", guard.issues)

    def test_unittest_discovery_resource_warning_blocks_child_receipt(self) -> None:
        class CleanTests(unittest.TestCase):
            def test_clean(self) -> None:
                pass

        suite = unittest.defaultTestLoader.loadTestsFromTestCase(CleanTests)
        node_id = next(MANIFEST_RUNNER._flatten_unittest_suite(suite)).id()
        stored = {
            "collections": {
                "unittest": {
                    "nodes": [{"id": node_id, "status": "deterministic", "reason": ""}]
                }
            }
        }

        def discover_with_resource_warning() -> unittest.TestSuite:
            open("/dev/null")
            return suite

        with (
            mock.patch.object(
                MANIFEST_RUNNER,
                "_bound_manifest_snapshot",
                return_value=(stored, b"manifest", "a" * 64, b"allowlist"),
            ),
            mock.patch.object(
                MANIFEST_RUNNER,
                "_discover_unittest_suite",
                side_effect=discover_with_resource_warning,
            ),
            mock.patch.object(MANIFEST_RUNNER, "_verify_manifest_snapshot_unchanged"),
            mock.patch.object(MANIFEST_RUNNER, "_verify_allowlist_snapshot_unchanged"),
            mock.patch.object(MANIFEST_RUNNER, "_print_outcome_receipt"),
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(MANIFEST_RUNNER.ManifestError),
        ):
            MANIFEST_RUNNER._run_unittest_outcomes()

    @isolated_process_test
    def test_unittest_collection_live_thread_blocks_child_receipt(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="test_manifest_thread_", dir=ROOT / "tests"
        ) as temporary:
            test_root = Path(temporary)
            (test_root / "__init__.py").write_text("", encoding="utf-8")
            (test_root / "test_live_thread.py").write_text(
                "import threading\n"
                "threading.Thread(\n"
                "    target=threading.Event().wait, daemon=True\n"
                ").start()\n"
                "import unittest\n"
                "class LiveThreadTests(unittest.TestCase):\n"
                "    def test_body(self):\n"
                "        pass\n",
                encoding="utf-8",
            )

            with self.assertRaises(MANIFEST_RUNNER.ManifestError):
                MANIFEST_RUNNER._collect_unittest()

    def test_pytest_indirect_and_imported_runtime_skips_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix=".manifest-pytest-", dir=ROOT / "tests"
        ) as temporary:
            test_root = Path(temporary)
            helper_path = test_root / "runtime_skip_helper.py"
            test_path = test_root / "test_runtime_skip.py"
            helper_path.write_text(
                "import pytest\n"
                "def pytest_skip():\n"
                "    pytest.skip('runtime imported helper')\n",
                encoding="utf-8",
            )
            test_path.write_text(
                "import pytest\n"
                "from runtime_skip_helper import pytest_skip\n"
                "@pytest.fixture\n"
                "def teardown_skip():\n"
                "    yield\n"
                "    pytest.skip('runtime teardown skip')\n"
                "def test_getattr_skip():\n"
                "    getattr(pytest, 'skip')('runtime getattr alias')\n"
                "def test_imported_skip():\n"
                "    pytest_skip()\n"
                "def test_teardown_skip(teardown_skip):\n"
                "    pass\n",
                encoding="utf-8",
            )
            recorder = MANIFEST_RUNNER._PytestOutcomeRecorder(base=test_root)
            stdout = io.StringIO()
            stderr = io.StringIO()
            sys.path.insert(0, str(test_root))
            try:
                with (
                    contextlib.chdir(test_root),
                    contextlib.redirect_stdout(stdout),
                    contextlib.redirect_stderr(stderr),
                ):
                    exit_code = pytest.main(
                        [
                            "-q",
                            "-p",
                            "no:cacheprovider",
                            "--rootdir",
                            str(test_root),
                            str(test_path),
                        ],
                        plugins=[recorder],
                    )
            finally:
                sys.path.remove(str(test_root))
                sys.modules.pop("runtime_skip_helper", None)
                sys.modules.pop("test_runtime_skip", None)

            self.assertEqual(exit_code, pytest.ExitCode.OK)
            self.assertFalse(recorder.collection_issues)
            self.assertFalse(recorder.deselected)
            self.assertEqual(len(recorder.collected), 3)
            outcomes = recorder.outcomes()
            self.assertEqual(
                {outcome["status"] for outcome in outcomes.values()},
                {"invalid_phase"},
            )
            deterministic = [
                {"id": node_id, "status": "deterministic", "reason": ""}
                for node_id in recorder.collected
            ]
            with self.assertRaises(MANIFEST_RUNNER.ManifestError):
                MANIFEST_RUNNER._verify_execution_outcomes(
                    "pytest", deterministic, recorder.collected, outcomes
                )

    def test_unittest_indirect_and_imported_runtime_skips_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix=".manifest-unittest-", dir=ROOT / "tests"
        ) as temporary:
            helper_path = Path(temporary) / "runtime_unittest_skip_helper.py"
            helper_path.write_text(
                "import unittest\n"
                "def unittest_skip():\n"
                "    raise unittest.SkipTest('runtime imported helper')\n",
                encoding="utf-8",
            )
            helper = load_module("runtime_unittest_skip_helper", helper_path)

            class RuntimeSkipTests(unittest.TestCase):
                def test_getattr_skip(self) -> None:
                    skip_now = getattr(self, "skipTest")  # noqa: B009
                    skip_now("runtime getattr alias")

                def test_imported_skip(self) -> None:
                    helper.unittest_skip()

            suite = unittest.defaultTestLoader.loadTestsFromTestCase(RuntimeSkipTests)
            collected = {
                MANIFEST_RUNNER._unittest_test_id(test)
                for test in MANIFEST_RUNNER._flatten_unittest_suite(suite)
            }
            stream = io.StringIO()
            result = unittest.TextTestRunner(
                stream=stream,
                buffer=True,
                resultclass=MANIFEST_RUNNER._UnittestOutcomeResult,
            ).run(suite)

            self.assertTrue(result.wasSuccessful())
            self.assertEqual(len(collected), 2)
            outcomes = result.outcomes()
            self.assertEqual(
                {outcome["status"] for outcome in outcomes.values()}, {"skipped"}
            )
            deterministic = [
                {"id": node_id, "status": "deterministic", "reason": ""}
                for node_id in collected
            ]
            with self.assertRaises(MANIFEST_RUNNER.ManifestError):
                MANIFEST_RUNNER._verify_execution_outcomes(
                    "unittest", deterministic, collected, outcomes
                )

    def test_unittest_success_without_starting_test_is_rejected(self) -> None:
        class ForgedSuccessTests(unittest.TestCase):
            def run(
                self, result: unittest.TestResult | None = None
            ) -> unittest.TestResult:
                assert result is not None
                result.addSuccess(self)
                return result

            def test_body_must_run(self) -> None:
                self.fail("the forged run method must not execute this body")

        suite = unittest.defaultTestLoader.loadTestsFromTestCase(ForgedSuccessTests)
        collected = {
            MANIFEST_RUNNER._unittest_test_id(test)
            for test in MANIFEST_RUNNER._flatten_unittest_suite(suite)
        }
        result = unittest.TextTestRunner(
            stream=io.StringIO(),
            buffer=True,
            resultclass=MANIFEST_RUNNER._UnittestOutcomeResult,
        ).run(suite)

        self.assertTrue(result.wasSuccessful())
        self.assertEqual(result.started, set())
        self.assertEqual(set(result.events), collected)
        self.assertIn(
            "unittest start lifecycle mismatch",
            MANIFEST_RUNNER._unittest_execution_issues(result, collected),
        )
        deterministic = [
            {"id": node_id, "status": "deterministic", "reason": ""}
            for node_id in collected
        ]
        with self.assertRaises(MANIFEST_RUNNER.ManifestError):
            MANIFEST_RUNNER._verify_execution_outcomes(
                "unittest",
                deterministic,
                collected,
                result.outcomes(),
                issues=MANIFEST_RUNNER._unittest_execution_issues(result, collected),
            )

    def test_unittest_success_without_stopping_test_is_rejected(self) -> None:
        class MissingStopTests(unittest.TestCase):
            def run(
                self, result: unittest.TestResult | None = None
            ) -> unittest.TestResult:
                assert result is not None
                result.startTest(self)
                result.addSuccess(self)
                return result

            def test_body_must_run(self) -> None:
                self.fail("the forged run method must not execute this body")

        suite = unittest.defaultTestLoader.loadTestsFromTestCase(MissingStopTests)
        collected = {
            MANIFEST_RUNNER._unittest_test_id(test)
            for test in MANIFEST_RUNNER._flatten_unittest_suite(suite)
        }
        result = unittest.TextTestRunner(
            stream=io.StringIO(),
            buffer=True,
            resultclass=MANIFEST_RUNNER._UnittestOutcomeResult,
        ).run(suite)

        self.assertTrue(result.wasSuccessful())
        self.assertEqual(result.started, collected)
        self.assertEqual(set(result.events), collected)
        self.assertEqual(result.stopped, set())
        self.assertIn(
            "unittest stop lifecycle mismatch",
            MANIFEST_RUNNER._unittest_execution_issues(result, collected),
        )

    def test_unittest_external_journal_rejects_result_state_forgery(self) -> None:
        class ResultForgeryTests(unittest.TestCase):
            def test_forged_subtest(self) -> None:
                with self.subTest(case="must-fail"):
                    self.fail("synthetic failure")
                result = self._outcome.result
                result.failures.clear()
                result.errors.clear()
                result.events[self.id()] = [{"status": "passed", "reason": ""}]

        suite = unittest.defaultTestLoader.loadTestsFromTestCase(ResultForgeryTests)
        error = self.execute_synthetic_unittest_suite(suite)

        self.assertIsInstance(error, MANIFEST_RUNNER.ManifestError)

    def test_unittest_result_callback_replacement_is_rejected(self) -> None:
        class ResultCallbackReplacementTests(unittest.TestCase):
            def test_replaces_error_callback(self) -> None:
                result = self._outcome.result
                result.addError = lambda test, error: result.addSuccess(test)

        suite = unittest.defaultTestLoader.loadTestsFromTestCase(
            ResultCallbackReplacementTests
        )
        error = self.execute_synthetic_unittest_suite(suite)

        self.assertIsInstance(error, MANIFEST_RUNNER.ManifestError)

    def test_unittest_leaf_protocol_rejects_noop_and_invalid_async_methods(
        self,
    ) -> None:
        class NoopCallTests(unittest.TestCase):
            def _callTestMethod(self, method: object) -> None:
                del method

            def test_body_must_run(self) -> None:
                self.fail("the overridden call hook must not hide this body")

        class GeneratorTests(unittest.TestCase):
            def test_generator(self) -> object:
                yield "not executed"

        class PlainAsyncTests(unittest.TestCase):
            async def test_async(self) -> None:
                pass

        class FullForgedRunTests(unittest.TestCase):
            def run(
                self, result: unittest.TestResult | None = None
            ) -> unittest.TestResult:
                assert result is not None
                result.startTest(self)
                result.addSuccess(self)
                result.stopTest(self)
                return result

            def test_body_must_run(self) -> None:
                self.fail("the forged run method must not hide this body")

        class SupportedAsyncTests(unittest.IsolatedAsyncioTestCase):
            async def test_async(self) -> None:
                pass

        for case in (
            NoopCallTests,
            GeneratorTests,
            PlainAsyncTests,
            FullForgedRunTests,
        ):
            leaf = next(
                MANIFEST_RUNNER._flatten_unittest_suite(
                    unittest.defaultTestLoader.loadTestsFromTestCase(case)
                )
            )
            with (
                self.subTest(case=case.__name__),
                self.assertRaises(MANIFEST_RUNNER.ManifestError),
            ):
                MANIFEST_RUNNER._validate_unittest_leaf(leaf)

        supported = next(
            MANIFEST_RUNNER._flatten_unittest_suite(
                unittest.defaultTestLoader.loadTestsFromTestCase(SupportedAsyncTests)
            )
        )
        MANIFEST_RUNNER._validate_unittest_leaf(supported)

        class InstanceOverrideTests(unittest.TestCase):
            def test_body_must_run(self) -> None:
                self.fail("the instance protocol override must not hide this body")

        for protocol_name in MANIFEST_RUNNER._UNITTEST_PROTOCOL_NAMES:
            leaf = next(
                MANIFEST_RUNNER._flatten_unittest_suite(
                    unittest.defaultTestLoader.loadTestsFromTestCase(
                        InstanceOverrideTests
                    )
                )
            )
            setattr(leaf, protocol_name, lambda *args, **kwargs: None)
            with (
                self.subTest(instance_override=protocol_name),
                self.assertRaises(MANIFEST_RUNNER.ManifestError),
            ):
                MANIFEST_RUNNER._validate_unittest_leaf(leaf)

        for protocol_name, _descriptor in MANIFEST_RUNNER._UNITTEST_ASYNC_PROTOCOL:
            leaf = next(
                MANIFEST_RUNNER._flatten_unittest_suite(
                    unittest.defaultTestLoader.loadTestsFromTestCase(
                        SupportedAsyncTests
                    )
                )
            )
            setattr(leaf, protocol_name, lambda *args, **kwargs: None)
            with (
                self.subTest(async_instance_override=protocol_name),
                self.assertRaises(MANIFEST_RUNNER.ManifestError),
            ):
                MANIFEST_RUNNER._validate_unittest_leaf(leaf)

    def test_unittest_preserves_static_and_class_method_tests(self) -> None:
        events: list[str] = []

        class DescriptorTests(unittest.TestCase):
            @staticmethod
            def test_static() -> None:
                events.append("static")

            @classmethod
            def test_class(cls) -> None:
                events.append(f"class:{cls.__name__}")

        suite = unittest.defaultTestLoader.loadTestsFromTestCase(DescriptorTests)

        error = self.execute_synthetic_unittest_suite(suite)

        self.assertIsNone(error)
        self.assertEqual(events, ["class:DescriptorTests", "static"])

    def test_unittest_instrumentation_rejects_non_none_body_results(self) -> None:
        class ReturnedGeneratorTests(unittest.TestCase):
            def test_hidden_assertion(self) -> object:
                def assertions() -> object:
                    self.fail("generator assertions must not stay hidden")
                    yield None

                return assertions()

        class ReturnedCoroutineTests(unittest.TestCase):
            def test_hidden_assertion(self) -> object:
                async def assertions() -> None:
                    self.fail("coroutine assertions must not stay hidden")

                return assertions()

        class ReturnedValueTests(unittest.TestCase):
            def test_non_none(self) -> object:
                return object()

        class AsyncReturnedValueTests(unittest.IsolatedAsyncioTestCase):
            async def test_non_none(self) -> object:
                return object()

        for case in (
            ReturnedGeneratorTests,
            ReturnedCoroutineTests,
            ReturnedValueTests,
            AsyncReturnedValueTests,
        ):
            leaf = next(
                MANIFEST_RUNNER._flatten_unittest_suite(
                    unittest.defaultTestLoader.loadTestsFromTestCase(case)
                )
            )
            node_id = leaf.id()
            evidence = MANIFEST_RUNNER._UnittestExecutionEvidence(
                run_started=set(),
                run_completed=set(),
                body_started=set(),
                body_completed=set(),
                duplicate_events=set(),
            )
            MANIFEST_RUNNER._instrument_unittest_leaf(leaf, evidence)
            guard = MANIFEST_RUNNER._UnraisableIssueGuard()
            with (
                self.subTest(case=case.__name__),
                guard,
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                result = unittest.TextTestRunner(
                    stream=io.StringIO(),
                    buffer=True,
                    resultclass=MANIFEST_RUNNER._UnittestOutcomeResult,
                ).run(unittest.TestSuite([leaf]))

            self.assertFalse(result.wasSuccessful())
            self.assertEqual(evidence.body_started, {node_id})
            self.assertEqual(evidence.body_completed, set())

    def test_unittest_rejects_lazy_instance_class_and_async_fixtures(self) -> None:
        state = {"hidden": False}

        def lazy_result() -> object:
            def hidden() -> object:
                state["hidden"] = True
                yield None

            return hidden()

        class LazySetUpTests(unittest.TestCase):
            def setUp(self) -> object:
                return lazy_result()

            def test_body(self) -> None:
                pass

        class LazyTearDownTests(unittest.TestCase):
            def tearDown(self) -> object:
                return lazy_result()

            def test_body(self) -> None:
                pass

        class LazySetUpClassTests(unittest.TestCase):
            @classmethod
            def setUpClass(cls) -> object:
                return lazy_result()

            def test_body(self) -> None:
                pass

        class LazyTearDownClassTests(unittest.TestCase):
            @classmethod
            def tearDownClass(cls) -> object:
                return lazy_result()

            def test_body(self) -> None:
                pass

        class LazyAsyncSetUpTests(unittest.IsolatedAsyncioTestCase):
            async def asyncSetUp(self) -> object:
                return lazy_result()

            async def test_body(self) -> None:
                pass

        class LazyAsyncTearDownTests(unittest.IsolatedAsyncioTestCase):
            async def asyncTearDown(self) -> object:
                return lazy_result()

            async def test_body(self) -> None:
                pass

        for case in (
            LazySetUpTests,
            LazyTearDownTests,
            LazySetUpClassTests,
            LazyTearDownClassTests,
            LazyAsyncSetUpTests,
            LazyAsyncTearDownTests,
        ):
            suite = unittest.defaultTestLoader.loadTestsFromTestCase(case)
            with self.subTest(case=case.__name__):
                error = self.execute_synthetic_unittest_suite(suite)
                self.assertIsInstance(error, MANIFEST_RUNNER.ManifestError)

        self.assertFalse(state["hidden"])

    def test_unittest_preserves_inherited_class_fixtures(self) -> None:
        events: list[str] = []

        class BaseTests(unittest.TestCase):
            @classmethod
            def setUpClass(cls) -> None:
                events.append(f"setup:{cls.__name__}")

            @classmethod
            def tearDownClass(cls) -> None:
                events.append(f"teardown:{cls.__name__}")

            def test_base(self) -> None:
                events.append(f"body:{type(self).__name__}:base")

        class DerivedTests(BaseTests):
            def test_derived(self) -> None:
                events.append("body:DerivedTests:derived")

        suite = unittest.TestSuite(
            [
                unittest.defaultTestLoader.loadTestsFromTestCase(BaseTests),
                unittest.defaultTestLoader.loadTestsFromTestCase(DerivedTests),
            ]
        )

        error = self.execute_synthetic_unittest_suite(suite)

        self.assertIsNone(error)
        self.assertEqual(
            events,
            [
                "setup:BaseTests",
                "body:BaseTests:base",
                "teardown:BaseTests",
                "setup:DerivedTests",
                "body:DerivedTests:base",
                "body:DerivedTests:derived",
                "teardown:DerivedTests",
            ],
        )

    def test_unittest_preserves_cooperative_class_fixture_inheritance(self) -> None:
        events: list[str] = []

        class BaseTests(unittest.TestCase):
            @classmethod
            def setUpClass(cls) -> None:
                events.append(f"base+:{cls.__name__}")

            @classmethod
            def tearDownClass(cls) -> None:
                events.append(f"base-:{cls.__name__}")

            def test_base(self) -> None:
                events.append(f"body:{type(self).__name__}:base")

        class DerivedTests(BaseTests):
            @classmethod
            def setUpClass(cls) -> None:
                events.append("derived+")
                super().setUpClass()

            @classmethod
            def tearDownClass(cls) -> None:
                events.append("derived-")
                super().tearDownClass()

            def test_derived(self) -> None:
                events.append("body:DerivedTests:derived")

        suite = unittest.TestSuite(
            [
                unittest.defaultTestLoader.loadTestsFromTestCase(BaseTests),
                unittest.defaultTestLoader.loadTestsFromTestCase(DerivedTests),
            ]
        )

        error = self.execute_synthetic_unittest_suite(suite)

        self.assertIsNone(error)
        self.assertEqual(events.count("base+:BaseTests"), 1)
        self.assertEqual(events.count("base-:BaseTests"), 1)
        self.assertEqual(events.count("base+:DerivedTests"), 1)
        self.assertEqual(events.count("base-:DerivedTests"), 1)
        self.assertEqual(events.count("derived+"), 1)
        self.assertEqual(events.count("derived-"), 1)
        self.assertEqual(events.count("body:BaseTests:base"), 1)
        self.assertEqual(events.count("body:DerivedTests:base"), 1)
        self.assertEqual(events.count("body:DerivedTests:derived"), 1)

    def test_unittest_validates_uncollected_mixin_class_fixtures(self) -> None:
        state = {"hidden": False}

        def lazy_result() -> object:
            def hidden() -> object:
                state["hidden"] = True
                yield None

            return hidden()

        for fixture_name in ("setUpClass", "tearDownClass"):
            if fixture_name == "setUpClass":

                class FixtureMixin:
                    @classmethod
                    def setUpClass(cls) -> object:
                        return lazy_result()

                class DerivedTests(FixtureMixin, unittest.TestCase):
                    @classmethod
                    def setUpClass(cls) -> None:
                        super().setUpClass()

                    def test_body(self) -> None:
                        pass

            else:

                class FixtureMixin:
                    @classmethod
                    def tearDownClass(cls) -> object:
                        return lazy_result()

                class DerivedTests(FixtureMixin, unittest.TestCase):
                    @classmethod
                    def tearDownClass(cls) -> None:
                        super().tearDownClass()

                    def test_body(self) -> None:
                        pass

            with self.subTest(fixture=fixture_name):
                error = self.execute_synthetic_unittest_suite(
                    unittest.defaultTestLoader.loadTestsFromTestCase(DerivedTests)
                )
                self.assertIsInstance(error, MANIFEST_RUNNER.ManifestError)

        self.assertFalse(state["hidden"])

    def test_unittest_validates_cooperative_instance_fixture_inheritance(
        self,
    ) -> None:
        state = {"hidden": False}

        def lazy_result() -> object:
            def hidden() -> object:
                state["hidden"] = True
                yield None

            return hidden()

        class BaseSetUpTests(unittest.TestCase):
            def setUp(self) -> object:
                return lazy_result()

        class DerivedSetUpTests(BaseSetUpTests):
            def setUp(self) -> None:
                super().setUp()

            def test_body(self) -> None:
                pass

        class BaseTearDownTests(unittest.TestCase):
            def tearDown(self) -> object:
                return lazy_result()

        class DerivedTearDownTests(BaseTearDownTests):
            def tearDown(self) -> None:
                super().tearDown()

            def test_body(self) -> None:
                pass

        class BaseAsyncTests(unittest.IsolatedAsyncioTestCase):
            async def asyncSetUp(self) -> object:
                return lazy_result()

            async def asyncTearDown(self) -> object:
                return lazy_result()

        class DerivedAsyncSetUpTests(BaseAsyncTests):
            async def asyncSetUp(self) -> None:
                await super().asyncSetUp()

            async def asyncTearDown(self) -> None:
                pass

            async def test_body(self) -> None:
                pass

        class DerivedAsyncTearDownTests(BaseAsyncTests):
            async def asyncSetUp(self) -> None:
                pass

            async def asyncTearDown(self) -> None:
                await super().asyncTearDown()

            async def test_body(self) -> None:
                pass

        for case in (
            DerivedSetUpTests,
            DerivedTearDownTests,
            DerivedAsyncSetUpTests,
            DerivedAsyncTearDownTests,
        ):
            with self.subTest(case=case.__name__):
                error = self.execute_synthetic_unittest_suite(
                    unittest.defaultTestLoader.loadTestsFromTestCase(case)
                )
                self.assertIsInstance(error, MANIFEST_RUNNER.ManifestError)

        self.assertFalse(state["hidden"])

    def test_unittest_preserves_inherited_callable_object_fixtures(self) -> None:
        events: list[str] = []

        class Fixture:
            def __init__(self, label: str) -> None:
                self.label = label

            def __call__(self) -> int:
                events.append(self.label)
                return 1

        class BaseTests(unittest.TestCase):
            setUp = Fixture("base+")
            tearDown = Fixture("base-")

        class DerivedTests(BaseTests):
            def setUp(self) -> None:
                super().setUp()

            def tearDown(self) -> None:
                super().tearDown()

            def test_body(self) -> None:
                events.append("body")

        error = self.execute_synthetic_unittest_suite(
            unittest.defaultTestLoader.loadTestsFromTestCase(DerivedTests)
        )

        self.assertIsNone(error)
        self.assertEqual(events, ["base+", "body", "base-"])

    def test_unittest_validates_inherited_callable_descriptors(self) -> None:
        state = {"hidden": False}

        def lazy_result() -> object:
            def hidden() -> object:
                state["hidden"] = True
                yield None

            return hidden()

        class FixtureDescriptor:
            def __get__(
                self, instance: object, owner: type[object]
            ) -> Callable[[], object]:
                del instance, owner
                return lazy_result

        class BaseTests(unittest.TestCase):
            setUp = FixtureDescriptor()

        class DerivedTests(BaseTests):
            def setUp(self) -> None:
                super().setUp()

            def test_body(self) -> None:
                pass

        error = self.execute_synthetic_unittest_suite(
            unittest.defaultTestLoader.loadTestsFromTestCase(DerivedTests)
        )

        self.assertIsInstance(error, MANIFEST_RUNNER.ManifestError)
        self.assertFalse(state["hidden"])

    def test_unittest_preserves_inherited_async_callable_descriptors(self) -> None:
        events: list[str] = []

        class AsyncFixtureDescriptor:
            def __call__(self) -> None:
                raise AssertionError("descriptor object itself must not run")

            def __get__(
                self, instance: object, owner: type[object]
            ) -> Callable[[], object]:
                del instance, owner

                async def fixture() -> int:
                    events.append("async+")
                    return 1

                return fixture

        class BaseTests(unittest.IsolatedAsyncioTestCase):
            asyncSetUp = AsyncFixtureDescriptor()

        class DerivedTests(BaseTests):
            async def asyncSetUp(self) -> None:
                await super().asyncSetUp()

            async def test_body(self) -> None:
                events.append("body")

        error = self.execute_synthetic_unittest_suite(
            unittest.defaultTestLoader.loadTestsFromTestCase(DerivedTests)
        )

        self.assertIsNone(error)
        self.assertEqual(events, ["async+", "body"])

    def test_unittest_binds_inherited_fixture_descriptors_once_at_call_time(
        self,
    ) -> None:
        events: list[str] = []

        class OneShotFixtureDescriptor:
            def __init__(self) -> None:
                self.binds = 0

            def __get__(
                self, instance: object, owner: type[object]
            ) -> Callable[[], int]:
                del instance, owner
                self.binds += 1
                events.append("bind")
                if self.binds > 1:
                    raise RuntimeError("descriptor was bound more than once")

                def fixture() -> int:
                    events.append("fixture")
                    return 1

                return fixture

        class BaseTests(unittest.TestCase):
            setUp = OneShotFixtureDescriptor()

        class DerivedTests(BaseTests):
            def setUp(self) -> None:
                events.append("derived")
                super().setUp()

            def test_body(self) -> None:
                events.append("body")

        error = self.execute_synthetic_unittest_suite(
            unittest.defaultTestLoader.loadTestsFromTestCase(DerivedTests)
        )

        self.assertIsNone(error)
        self.assertEqual(events, ["derived", "bind", "fixture", "body"])

    def test_unittest_preserves_sync_super_helper_named_async_setup(self) -> None:
        events: list[str] = []

        class BaseTests(unittest.IsolatedAsyncioTestCase):
            def asyncSetUp(self) -> int:
                events.append("base-sync")
                return 1

        class DerivedTests(BaseTests):
            async def asyncSetUp(self) -> None:
                self.assertEqual(super().asyncSetUp(), 1)
                events.append("derived-async")

            async def test_body(self) -> None:
                events.append("body")

        error = self.execute_synthetic_unittest_suite(
            unittest.defaultTestLoader.loadTestsFromTestCase(DerivedTests)
        )

        self.assertIsNone(error)
        self.assertEqual(events, ["base-sync", "derived-async", "body"])

    def test_unittest_preserves_interleaved_class_and_module_fixtures(self) -> None:
        events: list[str] = []
        module_a = ModuleType("translator_fixture_module_a")
        module_b = ModuleType("translator_fixture_module_b")

        def install_module_fixtures(module: ModuleType, label: str) -> None:
            module.setUpModule = lambda: events.append(f"module+:{label}")
            module.tearDownModule = lambda: events.append(f"module-:{label}")

        install_module_fixtures(module_a, "A")
        install_module_fixtures(module_b, "B")

        class ClassATests(unittest.TestCase):
            @classmethod
            def setUpClass(cls) -> None:
                events.append("class+:A")

            @classmethod
            def tearDownClass(cls) -> None:
                events.append("class-:A")

            def test_one(self) -> None:
                events.append("body:A1")

            def test_two(self) -> None:
                events.append("body:A2")

        class ClassBTests(unittest.TestCase):
            def test_one(self) -> None:
                events.append("body:B1")

        ClassATests.__module__ = module_a.__name__
        ClassBTests.__module__ = module_b.__name__
        module_a.ClassATests = ClassATests
        module_b.ClassBTests = ClassBTests
        sys.modules[module_a.__name__] = module_a
        sys.modules[module_b.__name__] = module_b
        suite = unittest.TestSuite(
            [
                ClassATests("test_one"),
                ClassBTests("test_one"),
                ClassATests("test_two"),
            ]
        )

        try:
            error = self.execute_synthetic_unittest_suite(suite)
        finally:
            sys.modules.pop(module_a.__name__, None)
            sys.modules.pop(module_b.__name__, None)

        self.assertIsNone(error)
        self.assertEqual(events.count("body:A1"), 1)
        self.assertEqual(events.count("body:B1"), 1)
        self.assertEqual(events.count("body:A2"), 1)
        self.assertEqual(events.count("class+:A"), 2)
        self.assertEqual(events.count("class-:A"), 2)
        self.assertEqual(events.count("module+:A"), 2)
        self.assertEqual(events.count("module-:A"), 2)
        self.assertEqual(events.count("module+:B"), 1)
        self.assertEqual(events.count("module-:B"), 1)

    def test_unittest_cleanup_callbacks_may_return_scalar_values(self) -> None:
        values = [1]
        class_values = [2]
        module_values = [3]

        class ScalarCleanupTests(unittest.TestCase):
            @classmethod
            def setUpClass(cls) -> None:
                cls.addClassCleanup(class_values.pop)

            def test_cleanup(self) -> None:
                self.addCleanup(values.pop)

        unittest.addModuleCleanup(module_values.pop)
        try:
            suite = unittest.defaultTestLoader.loadTestsFromTestCase(ScalarCleanupTests)
            error = self.execute_synthetic_unittest_suite(suite)
        finally:
            unittest.case._module_cleanups.clear()

        self.assertIsNone(error)
        self.assertEqual(values, [])
        self.assertEqual(class_values, [])
        self.assertEqual(module_values, [])

    def test_unittest_fixtures_may_return_scalar_values(self) -> None:
        events: list[str] = []

        class ScalarFixtureTests(unittest.TestCase):
            @classmethod
            def setUpClass(cls) -> int:
                events.append("class+")
                return 1

            @classmethod
            def tearDownClass(cls) -> int:
                events.append("class-")
                return 2

            def setUp(self) -> int:
                events.append("instance+")
                return 3

            def tearDown(self) -> int:
                events.append("instance-")
                return 4

            def test_body(self) -> None:
                events.append("body")

        class ScalarAsyncFixtureTests(unittest.IsolatedAsyncioTestCase):
            async def asyncSetUp(self) -> int:
                events.append("async+")
                return 5

            async def asyncTearDown(self) -> int:
                events.append("async-")
                return 6

            async def test_body(self) -> None:
                events.append("async-body")

        module = sys.modules[__name__]
        suite = unittest.TestSuite(
            [
                unittest.defaultTestLoader.loadTestsFromTestCase(ScalarFixtureTests),
                unittest.defaultTestLoader.loadTestsFromTestCase(
                    ScalarAsyncFixtureTests
                ),
            ]
        )
        with (
            mock.patch.object(
                module,
                "setUpModule",
                lambda: events.append("module+") or 7,
                create=True,
            ),
            mock.patch.object(
                module,
                "tearDownModule",
                lambda: events.append("module-") or 8,
                create=True,
            ),
        ):
            error = self.execute_synthetic_unittest_suite(suite)

        self.assertIsNone(error)
        self.assertEqual(events.count("module+"), 1)
        self.assertEqual(events.count("module-"), 1)
        self.assertIn("body", events)
        self.assertIn("async-body", events)

    def test_unittest_rejects_iterator_and_context_manager_fixture_results(
        self,
    ) -> None:
        state = {"entered": False, "iterated": False}

        @contextlib.contextmanager
        def lazy_context() -> object:
            state["entered"] = True
            yield None

        def lazy_iterator() -> object:
            def mark(_value: int) -> None:
                state["iterated"] = True

            return map(mark, [1])

        for name, factory in (
            ("context", lazy_context),
            ("iterator", lazy_iterator),
        ):

            class LazyFixtureTests(unittest.TestCase):
                fixture_factory = staticmethod(factory)

                def setUp(self) -> object:
                    return self.fixture_factory()

                def test_body(self) -> None:
                    pass

            with self.subTest(kind=name):
                error = self.execute_synthetic_unittest_suite(
                    unittest.defaultTestLoader.loadTestsFromTestCase(LazyFixtureTests)
                )
                self.assertIsInstance(error, MANIFEST_RUNNER.ManifestError)

        self.assertFalse(state["entered"])
        self.assertFalse(state["iterated"])

    def test_unittest_cleanup_accepts_unhashable_callable_objects(self) -> None:
        events: list[str] = []

        class UnhashableCleanup:
            __hash__ = None

            def __init__(self, label: str) -> None:
                self.label = label

            def __call__(self) -> None:
                events.append(self.label)

        class CleanupTests(unittest.TestCase):
            @classmethod
            def setUpClass(cls) -> None:
                cls.addClassCleanup(UnhashableCleanup("class"))

            def test_body(self) -> None:
                pass

        unittest.addModuleCleanup(UnhashableCleanup("module"))
        try:
            error = self.execute_synthetic_unittest_suite(
                unittest.defaultTestLoader.loadTestsFromTestCase(CleanupTests)
            )
        finally:
            unittest.case._module_cleanups.clear()

        self.assertIsNone(error)
        self.assertEqual(events, ["class", "module"])

    def test_unittest_preserves_early_shared_cleanup_drains(self) -> None:
        events: list[str] = []
        module = sys.modules[__name__]

        class CleanupTests(unittest.TestCase):
            @classmethod
            def setUpClass(cls) -> None:
                cls.addClassCleanup(lambda: events.append("class-cleanup") or 1)

            @classmethod
            def tearDownClass(cls) -> None:
                cls.doClassCleanups()

            def test_body(self) -> None:
                events.append("body")

        def tear_down_module() -> None:
            unittest.addModuleCleanup(lambda: events.append("module-cleanup") or 2)
            unittest.doModuleCleanups()

        with mock.patch.object(module, "tearDownModule", tear_down_module, create=True):
            error = self.execute_synthetic_unittest_suite(
                unittest.defaultTestLoader.loadTestsFromTestCase(CleanupTests)
            )

        self.assertIsNone(error)
        self.assertEqual(events, ["body", "class-cleanup", "module-cleanup"])

    def test_unittest_cached_shared_cleanup_drains_still_validate_callbacks(
        self,
    ) -> None:
        state = {"hidden": False}

        def lazy_cleanup() -> object:
            def hidden() -> object:
                state["hidden"] = True
                yield None

            return hidden()

        cached_module_add = unittest.addModuleCleanup
        cached_module_drain = unittest.doModuleCleanups

        class ModuleCleanupTests(unittest.TestCase):
            def test_cleanup(self) -> None:
                cached_module_add(lazy_cleanup)
                cached_module_drain()

        error = self.execute_synthetic_unittest_suite(
            unittest.defaultTestLoader.loadTestsFromTestCase(ModuleCleanupTests)
        )
        self.assertIsInstance(error, MANIFEST_RUNNER.ManifestError)
        unittest.case._module_cleanups.clear()

        class ClassCleanupTests(unittest.TestCase):
            def test_cleanup(self) -> None:
                cached_class_add(lazy_cleanup)
                cached_class_drain()

        cached_class_add = ClassCleanupTests.addClassCleanup
        cached_class_drain = ClassCleanupTests.doClassCleanups
        error = self.execute_synthetic_unittest_suite(
            unittest.defaultTestLoader.loadTestsFromTestCase(ClassCleanupTests)
        )
        self.assertIsInstance(error, MANIFEST_RUNNER.ManifestError)
        ClassCleanupTests._class_cleanups.clear()
        self.assertFalse(state["hidden"])

    def test_unittest_preserves_manual_instance_cleanup_drains(self) -> None:
        events: list[str] = []

        class CleanupTests(unittest.TestCase):
            def test_cleanup(self) -> None:
                self.addCleanup(lambda: events.append("cleanup") or 1)
                self.assertTrue(self.doCleanups())
                self.assertEqual(events, ["cleanup"])

        error = self.execute_synthetic_unittest_suite(
            unittest.defaultTestLoader.loadTestsFromTestCase(CleanupTests)
        )

        self.assertIsNone(error)
        self.assertEqual(events, ["cleanup"])

    def test_unittest_cached_instance_cleanup_add_still_validates_callback(
        self,
    ) -> None:
        events: list[str] = []

        class CleanupTests(unittest.TestCase):
            def __init__(self, *args: object, **kwargs: object) -> None:
                super().__init__(*args, **kwargs)
                self.cached_add_cleanup = self.addCleanup

            def test_cleanup(self) -> None:
                self.cached_add_cleanup(lambda: events.append("cached-cleanup") or 1)

        error = self.execute_synthetic_unittest_suite(
            unittest.defaultTestLoader.loadTestsFromTestCase(CleanupTests)
        )

        self.assertIsNone(error)
        self.assertEqual(events, ["cached-cleanup"])

    def test_unittest_preserves_reentrant_instance_cleanup_drains(self) -> None:
        events: list[str] = []

        class CleanupTests(unittest.TestCase):
            def test_cleanup(self) -> None:
                self.addCleanup(lambda: events.append("inner"))

                def outer_cleanup() -> None:
                    events.append("outer-start")
                    self.doCleanups()
                    events.append("outer-end")

                self.addCleanup(outer_cleanup)

        error = self.execute_synthetic_unittest_suite(
            unittest.defaultTestLoader.loadTestsFromTestCase(CleanupTests)
        )

        self.assertIsNone(error)
        self.assertEqual(events, ["outer-start", "inner", "outer-end"])

    def test_unittest_skipped_test_preserves_initial_instance_cleanup(self) -> None:
        events: list[str] = []

        class SkippedTests(unittest.TestCase):
            @unittest.skip("missing_external_prerequisite:private_human_evidence")
            def test_body(self) -> None:
                events.append("body")

        leaf = SkippedTests("test_body")

        def callback() -> None:
            events.append("cleanup")

        leaf.addCleanup(callback)
        error = self.execute_synthetic_unittest_suite(unittest.TestSuite([leaf]))

        self.assertIsNone(error)
        self.assertEqual(events, [])
        self.assertEqual(leaf._cleanups, [(callback, (), {})])

    def test_unittest_preserves_falsey_module_fixtures_and_none_class_fixtures(
        self,
    ) -> None:
        events: list[str] = []
        module = sys.modules[__name__]

        class FalseyFixture:
            def __init__(self, label: str) -> None:
                self.label = label

            def __bool__(self) -> bool:
                return False

            def __call__(self) -> None:
                events.append(self.label)

        class NoneClassFixtureTests(unittest.TestCase):
            setUpClass = None
            tearDownClass = None

            def test_body(self) -> None:
                self.assertIs(type(self).setUpClass, None)
                self.assertIs(type(self).tearDownClass, None)
                events.append("body")

        suite = unittest.defaultTestLoader.loadTestsFromTestCase(NoneClassFixtureTests)
        with (
            mock.patch.object(
                module, "setUpModule", FalseyFixture("module+"), create=True
            ),
            mock.patch.object(
                module, "tearDownModule", FalseyFixture("module-"), create=True
            ),
        ):
            error = self.execute_synthetic_unittest_suite(suite)

        self.assertIsNone(error)
        self.assertEqual(events, ["module+", "body", "module-"])

    def test_unittest_preserves_absent_module_fixture_introspection(self) -> None:
        module = ModuleType("translator_no_module_fixtures")
        state = {"body": False}

        class NoModuleFixtureTests(unittest.TestCase):
            def test_body(self) -> None:
                self.assertFalse(hasattr(module, "setUpModule"))
                self.assertFalse(hasattr(module, "tearDownModule"))
                state["body"] = True

        NoModuleFixtureTests.__module__ = module.__name__
        module.NoModuleFixtureTests = NoModuleFixtureTests
        sys.modules[module.__name__] = module
        try:
            error = self.execute_synthetic_unittest_suite(
                unittest.defaultTestLoader.loadTestsFromTestCase(NoModuleFixtureTests)
            )
        finally:
            sys.modules.pop(module.__name__, None)

        self.assertIsNone(error)
        self.assertTrue(state["body"])

    def test_unittest_skipped_class_preserves_pending_class_cleanup(self) -> None:
        events: list[str] = []

        @unittest.skip("missing_external_prerequisite:private_human_evidence")
        class SkippedTests(unittest.TestCase):
            def test_body(self) -> None:
                events.append("body")

        SkippedTests.addClassCleanup(lambda: events.append("cleanup"))
        try:
            error = self.execute_synthetic_unittest_suite(
                unittest.defaultTestLoader.loadTestsFromTestCase(SkippedTests)
            )
            self.assertIsNone(error)
            self.assertEqual(len(SkippedTests._class_cleanups), 1)
        finally:
            SkippedTests._class_cleanups.clear()

        self.assertEqual(events, [])

    def test_unittest_rejects_lazy_module_fixtures(self) -> None:
        state = {"hidden": False}

        def lazy_fixture() -> object:
            def hidden() -> object:
                state["hidden"] = True
                yield None

            return hidden()

        class ModuleFixtureTests(unittest.TestCase):
            def test_body(self) -> None:
                pass

        module = sys.modules[ModuleFixtureTests.__module__]
        for fixture_name in ("setUpModule", "tearDownModule"):
            suite = unittest.defaultTestLoader.loadTestsFromTestCase(ModuleFixtureTests)
            with (
                self.subTest(fixture=fixture_name),
                mock.patch.object(module, fixture_name, lazy_fixture, create=True),
            ):
                error = self.execute_synthetic_unittest_suite(suite)
                self.assertIsInstance(error, MANIFEST_RUNNER.ManifestError)

        self.assertFalse(state["hidden"])

    def test_unittest_rejects_lazy_cleanup_callbacks(self) -> None:
        state = {"hidden": False}

        def lazy_result() -> object:
            def hidden() -> object:
                state["hidden"] = True
                yield None

            return hidden()

        class InstanceCleanupTests(unittest.TestCase):
            def test_cleanup(self) -> None:
                self.addCleanup(lazy_result)

        class ClassCleanupTests(unittest.TestCase):
            @classmethod
            def setUpClass(cls) -> None:
                cls.addClassCleanup(lazy_result)

            def test_cleanup(self) -> None:
                pass

        class AsyncCleanupTests(unittest.IsolatedAsyncioTestCase):
            async def test_cleanup(self) -> None:
                async def cleanup() -> object:
                    return lazy_result()

                self.addAsyncCleanup(cleanup)

        class SpoofedCleanup:
            _translator_checked_cleanup = True

            def __call__(self) -> object:
                return lazy_result()

        class SpoofedClassCleanupTests(unittest.TestCase):
            @classmethod
            def setUpClass(cls) -> None:
                cls.addClassCleanup(SpoofedCleanup())

            def test_cleanup(self) -> None:
                pass

        for case in (
            InstanceCleanupTests,
            ClassCleanupTests,
            AsyncCleanupTests,
            SpoofedClassCleanupTests,
        ):
            suite = unittest.defaultTestLoader.loadTestsFromTestCase(case)
            with self.subTest(case=case.__name__):
                error = self.execute_synthetic_unittest_suite(suite)
                self.assertIsInstance(error, MANIFEST_RUNNER.ManifestError)

        class ModuleCleanupTests(unittest.TestCase):
            def test_cleanup(self) -> None:
                unittest.addModuleCleanup(active_callback)

        for callback in (lazy_result, SpoofedCleanup()):
            active_callback = callback
            suite = unittest.defaultTestLoader.loadTestsFromTestCase(ModuleCleanupTests)
            try:
                error = self.execute_synthetic_unittest_suite(suite)
                self.assertIsInstance(error, MANIFEST_RUNNER.ManifestError)
            finally:
                unittest.case._module_cleanups.clear()

        self.assertFalse(state["hidden"])

    def test_unittest_rejects_cleanup_registry_deletion(self) -> None:
        state = {"required": False, "eraser": False}

        class CleanupDeletionTests(unittest.TestCase):
            def test_cleanup_conservation(self) -> None:
                def required_cleanup() -> None:
                    state["required"] = True

                def erase_pending_cleanup() -> None:
                    state["eraser"] = True
                    self._cleanups.clear()

                self.addCleanup(required_cleanup)
                self.addCleanup(erase_pending_cleanup)

        suite = unittest.defaultTestLoader.loadTestsFromTestCase(CleanupDeletionTests)
        error = self.execute_synthetic_unittest_suite(suite)

        self.assertIsInstance(error, MANIFEST_RUNNER.ManifestError)
        self.assertTrue(state["eraser"])
        self.assertFalse(state["required"])

    def test_unittest_rejects_self_restoring_fixture_bypass(self) -> None:
        state = {"body": False, "teardown": False, "cleanup": False}

        class SelfRestoringBypassTests(unittest.TestCase):
            def tearDown(self) -> None:
                state["teardown"] = True

            def test_bypass(self) -> None:
                state["body"] = True

                def required_cleanup() -> None:
                    state["cleanup"] = True

                self.addCleanup(required_cleanup)
                trusted_call_teardown = self._callTearDown
                trusted_do_cleanups = self.doCleanups

                def bypass_teardown() -> None:
                    self._callTearDown = trusted_call_teardown

                def bypass_cleanups() -> bool:
                    self.doCleanups = trusted_do_cleanups
                    self._cleanups.clear()
                    return True

                self._callTearDown = bypass_teardown
                self.doCleanups = bypass_cleanups

        suite = unittest.defaultTestLoader.loadTestsFromTestCase(
            SelfRestoringBypassTests
        )
        error = self.execute_synthetic_unittest_suite(suite)

        self.assertIsInstance(error, MANIFEST_RUNNER.ManifestError)
        self.assertTrue(state["body"])
        self.assertFalse(state["teardown"])
        self.assertFalse(state["cleanup"])

    def test_unittest_execution_rejects_custom_suite(self) -> None:
        state = {"body": False, "custom_suite": False}

        class BodyTests(unittest.TestCase):
            def test_body_runs(self) -> None:
                state["body"] = True

        leaf = next(
            MANIFEST_RUNNER._flatten_unittest_suite(
                unittest.defaultTestLoader.loadTestsFromTestCase(BodyTests)
            )
        )

        class ForgedSuite(unittest.TestSuite):
            def run(
                self,
                result: unittest.TestResult,
                debug: bool = False,
            ) -> unittest.TestResult:
                del debug
                state["custom_suite"] = True
                result.startTest(leaf)
                result.addSuccess(leaf)
                result.stopTest(leaf)
                return result

        suite = ForgedSuite([leaf])
        stored = {
            "collections": {
                "unittest": {
                    "nodes": [
                        {"id": leaf.id(), "status": "deterministic", "reason": ""}
                    ]
                }
            }
        }
        with (
            mock.patch.object(
                MANIFEST_RUNNER,
                "_bound_manifest_snapshot",
                return_value=(stored, b"manifest", "a" * 64, b"allowlist"),
            ),
            mock.patch.object(
                MANIFEST_RUNNER, "_discover_unittest_suite", return_value=suite
            ),
            mock.patch.object(MANIFEST_RUNNER, "_verify_manifest_snapshot_unchanged"),
            mock.patch.object(MANIFEST_RUNNER, "_verify_allowlist_snapshot_unchanged"),
            mock.patch.object(MANIFEST_RUNNER, "_print_outcome_receipt"),
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(MANIFEST_RUNNER.ManifestError),
        ):
            MANIFEST_RUNNER._run_unittest_outcomes()

        self.assertFalse(state["body"])
        self.assertFalse(state["custom_suite"])

    def test_unittest_instrumentation_preserves_async_and_declared_skip(self) -> None:
        state = {"async_body": False, "skipped_body": False}

        class AsyncTests(unittest.IsolatedAsyncioTestCase):
            async def test_async_body(self) -> None:
                state["async_body"] = True

        class SkippedTests(unittest.TestCase):
            @unittest.skip("missing_external_prerequisite:private_human_evidence")
            def test_declared_skip(self) -> None:
                state["skipped_body"] = True

        leaves = [
            next(
                MANIFEST_RUNNER._flatten_unittest_suite(
                    unittest.defaultTestLoader.loadTestsFromTestCase(case)
                )
            )
            for case in (AsyncTests, SkippedTests)
        ]
        suite = unittest.TestSuite(leaves)
        stored = {
            "collections": {
                "unittest": {
                    "nodes": [
                        {
                            "id": leaves[0].id(),
                            "status": "deterministic",
                            "reason": "",
                        },
                        {
                            "id": leaves[1].id(),
                            "status": "external_skip",
                            "reason": (
                                "missing_external_prerequisite:private_human_evidence"
                            ),
                        },
                    ]
                }
            }
        }
        with (
            mock.patch.object(
                MANIFEST_RUNNER,
                "_bound_manifest_snapshot",
                return_value=(stored, b"manifest", "a" * 64, b"allowlist"),
            ),
            mock.patch.object(
                MANIFEST_RUNNER, "_discover_unittest_suite", return_value=suite
            ),
            mock.patch.object(MANIFEST_RUNNER, "_verify_manifest_snapshot_unchanged"),
            mock.patch.object(MANIFEST_RUNNER, "_verify_allowlist_snapshot_unchanged"),
            mock.patch.object(MANIFEST_RUNNER, "_print_outcome_receipt"),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            MANIFEST_RUNNER._run_unittest_outcomes()

        self.assertTrue(state["async_body"])
        self.assertFalse(state["skipped_body"])

    def test_unittest_execution_rejects_protocol_mutation_between_leaves(
        self,
    ) -> None:
        state = {"victim_body": False, "forged_run": False}

        class VictimTests(unittest.TestCase):
            def test_body_runs(self) -> None:
                state["victim_body"] = True

        victim = next(
            MANIFEST_RUNNER._flatten_unittest_suite(
                unittest.defaultTestLoader.loadTestsFromTestCase(VictimTests)
            )
        )

        class MutatorTests(unittest.TestCase):
            def test_mutates_next_leaf(self) -> None:
                def forged_run(
                    result: unittest.TestResult | None = None,
                ) -> unittest.TestResult:
                    assert result is not None
                    state["forged_run"] = True
                    result.startTest(victim)
                    result.addSuccess(victim)
                    result.stopTest(victim)
                    return result

                victim.run = forged_run

        mutator = next(
            MANIFEST_RUNNER._flatten_unittest_suite(
                unittest.defaultTestLoader.loadTestsFromTestCase(MutatorTests)
            )
        )
        suite = unittest.TestSuite([mutator, victim])
        stored = {
            "collections": {
                "unittest": {
                    "nodes": [
                        {"id": test.id(), "status": "deterministic", "reason": ""}
                        for test in (mutator, victim)
                    ]
                }
            }
        }
        with (
            mock.patch.object(
                MANIFEST_RUNNER,
                "_bound_manifest_snapshot",
                return_value=(stored, b"manifest", "a" * 64, b"allowlist"),
            ),
            mock.patch.object(
                MANIFEST_RUNNER, "_discover_unittest_suite", return_value=suite
            ),
            mock.patch.object(MANIFEST_RUNNER, "_verify_manifest_snapshot_unchanged"),
            mock.patch.object(MANIFEST_RUNNER, "_verify_allowlist_snapshot_unchanged"),
            mock.patch.object(
                MANIFEST_RUNNER, "_print_outcome_receipt"
            ) as print_receipt,
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(MANIFEST_RUNNER.ManifestError),
        ):
            MANIFEST_RUNNER._run_unittest_outcomes()

        self.assertFalse(state["victim_body"])
        self.assertTrue(state["forged_run"])
        print_receipt.assert_not_called()

    def test_unittest_rejects_class_fixture_replacing_test_body(self) -> None:
        events: list[str] = []

        class MutatingTests(unittest.TestCase):
            @classmethod
            def setUpClass(cls) -> None:
                def replacement(_self: unittest.TestCase) -> None:
                    events.append("replacement-body")

                cls.test_body = replacement

            def test_body(self) -> None:
                events.append("original-body")

        error = self.execute_synthetic_unittest_suite(
            unittest.defaultTestLoader.loadTestsFromTestCase(MutatingTests)
        )

        self.assertIsInstance(error, MANIFEST_RUNNER.ManifestError)
        self.assertEqual(events, [])

    def test_unittest_rejects_setup_replacing_future_teardown(self) -> None:
        events: list[str] = []

        class MutatingTests(unittest.TestCase):
            def setUp(self) -> None:
                def replacement(_self: unittest.TestCase) -> None:
                    events.append("replacement-teardown")

                type(self).tearDown = replacement

            def test_body(self) -> None:
                events.append("body")

            def tearDown(self) -> None:
                events.append("original-teardown")

        error = self.execute_synthetic_unittest_suite(
            unittest.defaultTestLoader.loadTestsFromTestCase(MutatingTests)
        )

        self.assertIsInstance(error, MANIFEST_RUNNER.ManifestError)
        self.assertEqual(events, ["body"])

    def test_unittest_rejects_shared_fixture_replacing_future_fixture(self) -> None:
        events: list[str] = []

        class MutatingTests(unittest.TestCase):
            @classmethod
            def setUpClass(cls) -> None:
                def replacement(_class: type[unittest.TestCase]) -> None:
                    events.append("replacement-class-teardown")

                cls.tearDownClass = classmethod(replacement)

            def test_body(self) -> None:
                events.append("body")

        error = self.execute_synthetic_unittest_suite(
            unittest.defaultTestLoader.loadTestsFromTestCase(MutatingTests)
        )

        self.assertIsInstance(error, MANIFEST_RUNNER.ManifestError)
        self.assertEqual(events, [])

    def test_unittest_rejects_module_fixture_replacing_future_fixture(self) -> None:
        events: list[str] = []
        module = sys.modules[__name__]

        def replacement_teardown() -> None:
            events.append("replacement-module-teardown")

        def mutating_setup() -> None:
            module.tearDownModule = replacement_teardown

        class ModuleFixtureTests(unittest.TestCase):
            def test_body(self) -> None:
                events.append("body")

        with (
            mock.patch.object(module, "setUpModule", mutating_setup, create=True),
            mock.patch.object(module, "tearDownModule", None, create=True),
        ):
            error = self.execute_synthetic_unittest_suite(
                unittest.defaultTestLoader.loadTestsFromTestCase(ModuleFixtureTests)
            )

        self.assertIsInstance(error, MANIFEST_RUNNER.ManifestError)
        self.assertEqual(events, [])

    def test_unittest_execution_rejects_temporal_suite_helper_mutation(self) -> None:
        def mutator_case(helper_name: str) -> type[unittest.TestCase]:
            class MutatorTests(unittest.TestCase):
                def test_mutates_suite_helper(self) -> None:
                    setattr(unittest.TestSuite, helper_name, lambda *args: None)

            return MutatorTests

        for helper_name in (
            "_handleClassSetUp",
            "_handleModuleFixture",
            "_handleModuleTearDown",
            "_tearDownPreviousClass",
            "_addClassOrModuleLevelException",
        ):
            original = dict(MANIFEST_RUNNER._UNITTEST_SUITE_PROTOCOL)[helper_name]
            suite = unittest.defaultTestLoader.loadTestsFromTestCase(
                mutator_case(helper_name)
            )
            with (
                self.subTest(helper=helper_name),
                mock.patch.object(unittest.TestSuite, helper_name, original),
            ):
                error = self.execute_synthetic_unittest_suite(suite)
                self.assertIsInstance(error, MANIFEST_RUNNER.ManifestError)

    def test_unittest_execution_rejects_temporal_suite_global_mutation(self) -> None:
        original = dict(MANIFEST_RUNNER._UNITTEST_SUITE_GLOBAL_PROTOCOL)["_isnotsuite"]

        class MutatorTests(unittest.TestCase):
            def test_mutates_suite_global(self) -> None:
                unittest.suite._isnotsuite = lambda test: False

        suite = unittest.defaultTestLoader.loadTestsFromTestCase(MutatorTests)
        with mock.patch.object(unittest.suite, "_isnotsuite", original):
            error = self.execute_synthetic_unittest_suite(suite)

        self.assertIsInstance(error, MANIFEST_RUNNER.ManifestError)

    def test_unittest_subtest_and_setup_class_outcomes_are_rejected(self) -> None:
        class FailingSubtestTests(unittest.TestCase):
            def test_failing_subtest(self) -> None:
                with self.subTest(case="negative-control"):
                    self.fail("synthetic subtest failure")

        class SkippedSetupTests(unittest.TestCase):
            @classmethod
            def setUpClass(cls) -> None:
                raise unittest.SkipTest("runtime class setup skip")

            def test_never_started_after_skip(self) -> None:
                self.fail("unreachable")

        class FailedSetupTests(unittest.TestCase):
            @classmethod
            def setUpClass(cls) -> None:
                raise RuntimeError("synthetic class setup failure")

            def test_never_started_after_error(self) -> None:
                self.fail("unreachable")

        suite = unittest.TestSuite(
            unittest.defaultTestLoader.loadTestsFromTestCase(case)
            for case in (FailingSubtestTests, SkippedSetupTests, FailedSetupTests)
        )
        collected = {
            MANIFEST_RUNNER._unittest_test_id(test)
            for test in MANIFEST_RUNNER._flatten_unittest_suite(suite)
        }
        result = unittest.TextTestRunner(
            stream=io.StringIO(),
            buffer=True,
            resultclass=MANIFEST_RUNNER._UnittestOutcomeResult,
        ).run(suite)
        outcomes = result.outcomes()

        self.assertFalse(result.wasSuccessful())
        subtest_id = next(
            node_id for node_id in collected if node_id.endswith("test_failing_subtest")
        )
        self.assertEqual(outcomes[subtest_id]["status"], "failed")
        self.assertNotEqual(set(outcomes), collected)
        deterministic = [
            {"id": node_id, "status": "deterministic", "reason": ""}
            for node_id in collected
        ]
        with self.assertRaises(MANIFEST_RUNNER.ManifestError):
            MANIFEST_RUNNER._verify_execution_outcomes(
                "unittest", deterministic, collected, outcomes
            )


_validate_isolated_test_class(ManifestExecutionGateTests)


def _isolated_test_worker_main(test_name: str) -> int:
    if os.environ.get(ISOLATED_TEST_WORKER_ENV) != test_name:
        print("isolated manifest test worker admission failed", file=sys.stderr)
        return 1
    suite = unittest.defaultTestLoader.loadTestsFromName(
        test_name, module=sys.modules[__name__]
    )
    if suite.countTestCases() != 1:
        print("isolated manifest test selection failed", file=sys.stderr)
        return 1
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=0).run(suite)
    clean_pass = (
        result.wasSuccessful()
        and result.testsRun == 1
        and not result.skipped
        and not result.expectedFailures
        and not result.unexpectedSuccesses
        and not result.failures
        and not result.errors
    )
    if not clean_pass:
        print("isolated manifest test did not produce one clean pass", file=sys.stderr)
        sys.stderr.write(stream.getvalue())
        return 1
    print(ISOLATED_TEST_WORKER_RECEIPT_PREFIX + test_name)
    return 0


def _isolated_test_main(test_name: str) -> int:
    if os.environ.get(ISOLATED_TEST_ENV) != test_name:
        print("isolated manifest test supervisor admission failed", file=sys.stderr)
        return 1
    parent_pid = os.environ.get(ISOLATED_TEST_PARENT_PID_ENV, "")
    if not parent_pid.isascii() or not parent_pid.isdigit() or int(parent_pid) <= 0:
        print("isolated manifest test parent admission failed", file=sys.stderr)
        return 1
    try:
        _arm_parent_death_signal(int(parent_pid))
    except (OSError, RuntimeError):
        print("isolated manifest test parent admission failed", file=sys.stderr)
        return 1
    environment = MANIFEST_RUNNER._validation_environment(source=dict(os.environ))
    environment[ISOLATED_TEST_WORKER_ENV] = test_name
    probe_path = os.environ.get(ISOLATED_TEST_PROBE_PATH_ENV)
    if probe_path is not None:
        environment[ISOLATED_TEST_PROBE_PATH_ENV] = probe_path
    command = {
        "argv": [
            sys.executable,
            "-I",
            str(Path(__file__).resolve()),
            "--worker",
            test_name,
        ],
        "cwd": ".",
        "timeout_seconds": ISOLATED_TEST_TIMEOUT_SECONDS,
    }
    try:
        completed = MANIFEST_RUNNER._run_gate_process(
            command,
            environment,
            capture_output=True,
            encoding="utf-8",
            errors="strict",
        )
    except (MANIFEST_RUNNER.ManifestError, subprocess.TimeoutExpired) as error:
        print(f"isolated manifest test containment failed: {error}", file=sys.stderr)
        return 1
    expected = ISOLATED_TEST_WORKER_RECEIPT_PREFIX + test_name + "\n"
    if completed.returncode != 0 or completed.stdout != expected or completed.stderr:
        print("isolated manifest test worker failed", file=sys.stderr)
        if completed.stdout:
            sys.stderr.write(completed.stdout)
        if completed.stderr:
            sys.stderr.write(completed.stderr)
        return 1
    print(ISOLATED_TEST_RECEIPT_PREFIX + test_name)
    return 0


def _outer_signal_probe_main(process_path: str) -> int:
    test_name = (
        "ManifestExecutionGateTests.isolated_probe_hangs_with_detached_descendant"
    )
    environment = MANIFEST_RUNNER._validation_environment(source=dict(os.environ))
    environment.update(
        {
            ISOLATED_TEST_ENV: test_name,
            ISOLATED_TEST_PROBE_PATH_ENV: process_path,
        }
    )
    completed = _run_isolated_test_supervisor(test_name, environment)
    sys.stdout.write(completed.stdout)
    sys.stderr.write(completed.stderr)
    return completed.returncode


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--outer-signal-probe":
        raise SystemExit(_outer_signal_probe_main(sys.argv[2]))
    if (
        len(sys.argv) == 3
        and sys.argv[1] == "--worker"
        and os.environ.get(ISOLATED_TEST_WORKER_ENV)
    ):
        raise SystemExit(_isolated_test_worker_main(sys.argv[2]))
    if len(sys.argv) == 2 and os.environ.get(ISOLATED_TEST_ENV):
        raise SystemExit(_isolated_test_main(sys.argv[1]))
    unittest.main()

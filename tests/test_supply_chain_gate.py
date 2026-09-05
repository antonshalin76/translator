from __future__ import annotations

import datetime
import hashlib
import os
import re
import subprocess
import tempfile
import time
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REVIEWED_RUST_EXCEPTIONS = {
    "RUSTSEC-2024-0370": "proc-macro-error@1.0.4",
    "RUSTSEC-2024-0411": "gdkwayland-sys@0.18.2",
    "RUSTSEC-2024-0412": "gdk@0.18.2",
    "RUSTSEC-2024-0413": "atk@0.18.2",
    "RUSTSEC-2024-0414": "gdkx11-sys@0.18.2",
    "RUSTSEC-2024-0415": "gtk@0.18.2",
    "RUSTSEC-2024-0416": "atk-sys@0.18.2",
    "RUSTSEC-2024-0417": "gdkx11@0.18.2",
    "RUSTSEC-2024-0418": "gdk-sys@0.18.2",
    "RUSTSEC-2024-0419": "gtk3-macros@0.18.2",
    "RUSTSEC-2024-0420": "gtk-sys@0.18.2",
    "RUSTSEC-2024-0429": "glib@0.18.5",
    "RUSTSEC-2025-0075": "unic-char-range@0.9.0",
    "RUSTSEC-2025-0080": "unic-common@0.9.0",
    "RUSTSEC-2025-0081": "unic-char-property@0.9.0",
    "RUSTSEC-2025-0098": "unic-ucd-version@0.9.0",
    "RUSTSEC-2025-0100": "unic-ucd-ident@0.9.0",
}

PINNED_TOOL_DIGESTS = {
    "cargo_audit": "473b9a71e5cb5bde22f69c32f749c9b83931287d92dc36b91cb04f6705640ef2",
    "bun": "92a1cd8b6185f676010bb18e767dfc65c772273e79ae9984480843261939eeaa",
    "osv_scanner": "edcfc41d257db36148f065055655fe3fcfc434b0b423ea67468a84c207524e0c",
}


def _write_executable(path: Path, body: str) -> None:
    path.write_text("#!/usr/bin/env bash\nset -u\n" + body)
    path.chmod(0o700)


class SupplyChainGateTests(unittest.TestCase):
    def test_ci_installs_exact_executables_accepted_by_gate(self) -> None:
        gate = (ROOT / "scripts/translator-sca").read_text()
        workflow = (ROOT / ".github/workflows/ci.yml").read_text()

        for name, digest in PINNED_TOOL_DIGESTS.items():
            self.assertIn(f'expected_{name}_sha256="{digest}"', gate)
            ci_name = name.upper()
            if name == "osv_scanner":
                ci_contract = f'{ci_name}_SHA256: "{digest}"'
            else:
                ci_contract = f'{ci_name}_EXECUTABLE_SHA256: "{digest}"'
            self.assertIn(ci_contract, workflow)

    def test_patched_dependency_versions_are_locked(self) -> None:
        cargo_lock = tomllib.loads((ROOT / "Cargo.lock").read_text())
        cargo_versions = {
            (package["name"], package["version"]) for package in cargo_lock["package"]
        }
        self.assertIn(("h2", "0.4.16"), cargo_versions)
        self.assertNotIn(("h2", "0.4.15"), cargo_versions)

        package_json = (ROOT / "apps/translator-ui/package.json").read_text()
        bun_lock = (ROOT / "apps/translator-ui/bun.lock").read_text()
        self.assertRegex(package_json, r'"nanoid"\s*:\s*"3\.3\.18"')
        self.assertIn('"nanoid": ["nanoid@3.3.18"', bun_lock)
        self.assertNotIn('"nanoid": ["nanoid@3.3.16"', bun_lock)

    def test_reviewed_rust_exceptions_are_exact_expiring_and_evidenced(self) -> None:
        cargo_policy = tomllib.loads((ROOT / ".cargo/audit.toml").read_text())
        osv_policy = tomllib.loads((ROOT / "config/osv-scanner.toml").read_text())

        cargo_ids = set(cargo_policy["advisories"]["ignore"])
        ignored = {item["id"]: item for item in osv_policy["IgnoredVulns"]}
        self.assertEqual(cargo_ids, set(REVIEWED_RUST_EXCEPTIONS))
        self.assertEqual(set(ignored), set(REVIEWED_RUST_EXCEPTIONS))
        self.assertEqual(cargo_policy["output"]["deny"], ["warnings"])

        today = datetime.datetime.now(datetime.UTC).date()
        for advisory_id, package in REVIEWED_RUST_EXCEPTIONS.items():
            exception = ignored[advisory_id]
            reason = exception["reason"]
            self.assertEqual(exception["ignoreUntil"], datetime.date(2026, 12, 1))
            self.assertGreater(exception["ignoreUntil"], today)
            self.assertIn(f"package={package}", reason)
            self.assertIn("owner=translator-desktop", reason)
            self.assertIn("dependency_path=translator-ui@0.1.0", reason)
            self.assertIn("renewal_evidence=", reason)
            self.assertIn("status=temporary-not-resolved", reason)

        self.assertIn(
            "VariantStrIter",
            ignored["RUSTSEC-2024-0429"]["reason"],
        )
        self.assertIn(
            "array_iter_str",
            ignored["RUSTSEC-2024-0429"]["reason"],
        )

    def test_bun_timeout_uses_osv_fallback_and_leaves_no_child(self) -> None:
        result, calls, child_marker, elapsed = self._run_fake_gate(
            bun_mode="timeout",
            osv_exit=0,
        )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertLess(elapsed, 8)
        self.assertIn("bun audit timed out; OSV fallback required", result.stderr)
        self.assertIn("OSV lockfile scan passed", result.stdout)
        self.assertIn("cargo-scan audit --file", calls)
        self.assertIn("osv-scan scan source", calls)
        self.assertIn("--config", calls)
        self.assertIn("config/osv-scanner.toml", calls)
        self.assertIn("Cargo.lock", calls)
        self.assertIn("sidecar/uv.lock", calls)
        self.assertIn("apps/translator-ui/bun.lock", calls)

        time.sleep(0.2)
        marker = str(child_marker).encode()
        surviving = []
        for cmdline_path in Path("/proc").glob("[0-9]*/cmdline"):
            try:
                if marker in cmdline_path.read_bytes():
                    surviving.append(cmdline_path.parent.name)
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
        self.assertEqual(surviving, [], "timed-out audit left a child process")

    def test_osv_finding_blocks_after_bun_timeout(self) -> None:
        result, calls, _, _ = self._run_fake_gate(bun_mode="timeout", osv_exit=1)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("osv-scan", calls)
        self.assertIn("OSV lockfile scan failed", result.stderr)

    def test_non_timeout_bun_finding_is_blocking_and_osv_still_runs(self) -> None:
        result, calls, _, _ = self._run_fake_gate(bun_mode="finding", osv_exit=0)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("bun-scan", calls)
        self.assertIn("osv-scan", calls)
        self.assertIn("bun audit failed", result.stderr)

    def test_timeout_configuration_cannot_exceed_contract_bound(self) -> None:
        result, calls, _, _ = self._run_fake_gate(
            bun_mode="pass",
            osv_exit=0,
            bun_timeout="121",
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("osv-scan", calls)
        self.assertIn(
            "TRANSLATOR_SCA_BUN_TIMEOUT must be an integer from 1 through 120",
            result.stderr,
        )

    def test_glib_public_iterator_reachability_invalidates_exception(self) -> None:
        result, calls, _, _ = self._run_fake_gate(
            bun_mode="pass",
            osv_exit=0,
            glib_reachability_exit=0,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "VariantStrIter/array_iter_str is directly referenced", result.stderr
        )
        self.assertIn("VariantStrIter|array_iter_str", calls)
        self.assertIn("--extended-regexp", calls)
        self.assertIn("glib-reachability -r ", calls)
        self.assertNotIn("glib-reachability -R ", calls)

    def test_glib_reachability_scan_error_fails_closed(self) -> None:
        result, calls, _, _ = self._run_fake_gate(
            bun_mode="pass",
            osv_exit=0,
            glib_reachability_exit=2,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("glib reachability scan failed", result.stderr)
        self.assertIn("glib-reachability", calls)

    def test_glib_reachability_timeout_fails_closed(self) -> None:
        result, calls, _, _ = self._run_fake_gate(
            bun_mode="pass",
            osv_exit=0,
            glib_reachability_exit=124,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("glib reachability scan timed out", result.stderr)
        self.assertIn("glib-reachability -r ", calls)
        self.assertNotIn("glib-reachability -R ", calls)

    def test_glib_false_clean_cannot_bypass_positive_control(self) -> None:
        result, calls, _, _ = self._run_fake_gate(
            bun_mode="pass",
            osv_exit=0,
            glib_positive_exit=1,
            glib_reachability_exit=1,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "glib reachability scan positive control was not detected",
            result.stderr,
        )
        self.assertGreaterEqual(calls.count("glib-reachability -r "), 2)
        control_call = next(
            line for line in calls.splitlines() if "translator-sca-glib." in line
        )
        self.assertIn("/apps ", control_call)
        self.assertTrue(control_call.endswith("/crates"), control_call)
        self.assertNotIn("positive-control.rs", control_call)

    def test_cargo_finding_is_blocking_and_osv_still_runs(self) -> None:
        result, calls, _, _ = self._run_fake_gate(
            bun_mode="pass",
            osv_exit=0,
            cargo_exit=1,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Rust advisory scan failed", result.stderr)
        self.assertIn("osv-scan", calls)

    def test_cargo_timeout_is_blocking(self) -> None:
        result, calls, _, _ = self._run_fake_gate(
            bun_mode="pass",
            osv_exit=0,
            cargo_exit=124,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Rust advisory scan timed out", result.stderr)
        self.assertIn("osv-scan", calls)

    def test_cargo_version_mismatch_is_blocking(self) -> None:
        result, calls, _, _ = self._run_fake_gate(
            bun_mode="pass",
            osv_exit=0,
            cargo_version="cargo-audit 0.22.1",
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("cargo-audit version mismatch; expected 0.22.2", result.stderr)
        self.assertIn("osv-scan", calls)

    def test_version_spoofed_scanners_cannot_satisfy_production_gate(self) -> None:
        result, _, _, _ = self._run_fake_gate(
            bun_mode="pass",
            osv_exit=0,
            patch_tool_identities=False,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("executable identity verification failed", result.stderr)

    def test_scanner_path_replacement_after_version_fails_closed(self) -> None:
        result, calls, _, _ = self._run_fake_gate(
            bun_mode="pass",
            osv_exit=0,
            replace_cargo_after_version=True,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "cargo-audit executable identity verification failed", result.stderr
        )
        self.assertNotIn("cargo-scan", calls)
        self.assertIn("osv-scan", calls)

    def test_unsafe_scanner_file_permissions_fail_before_execution(self) -> None:
        for scanner in ("cargo-audit", "bun", "osv-scanner"):
            for mode in (0o777, 0o2755, 0o4755):
                with self.subTest(scanner=scanner, mode=oct(mode)):
                    result, calls, _, _ = self._run_fake_gate(
                        bun_mode="pass",
                        osv_exit=0,
                        unsafe_scanner=scanner,
                        unsafe_scanner_mode=mode,
                    )

                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(
                        "executable identity verification failed", result.stderr
                    )
                    self.assertEqual(calls, "")

    def _run_fake_gate(
        self,
        *,
        bun_mode: str,
        osv_exit: int,
        bun_timeout: str = "1",
        glib_positive_exit: int = 0,
        glib_reachability_exit: int = 1,
        cargo_exit: int = 0,
        cargo_version: str = "cargo-audit 0.22.2",
        patch_tool_identities: bool = True,
        replace_cargo_after_version: bool = False,
        unsafe_scanner: str | None = None,
        unsafe_scanner_mode: int = 0o755,
    ) -> tuple[subprocess.CompletedProcess[str], str, Path, float]:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary = Path(temporary_directory)
            fake_bin = temporary / "bin"
            fake_bin.mkdir()
            call_log = temporary / "calls.log"
            child_marker = temporary / "bun-audit-child"

            _write_executable(
                fake_bin / "cargo-audit",
                'if [ "${1:-}" = "--version" ]; then\n'
                '  echo "$FAKE_CARGO_VERSION"\n'
                '  if [ "${FAKE_REPLACE_CARGO_AFTER_VERSION:-0}" = 1 ]; then\n'
                '    mv -- "$FAKE_CARGO_REPLACEMENT" "$FAKE_CARGO_PATH"\n'
                "  fi\n"
                "  exit 0\n"
                "fi\n"
                'printf "cargo-scan %s\\n" "$*" >>"$TRANSLATOR_SCA_TEST_LOG"\n'
                'exit "${FAKE_CARGO_EXIT:-0}"\n',
            )
            _write_executable(
                fake_bin / "cargo-audit-replacement",
                'if [ "${1:-}" = "--version" ]; then echo "$FAKE_CARGO_VERSION"; exit 0; fi\n'
                'printf "replacement-cargo-scan %s\\n" "$*" >>"$TRANSLATOR_SCA_TEST_LOG"\n'
                "exit 0\n",
            )
            _write_executable(
                child_marker,
                'exec /usr/bin/python3 -c "import time; time.sleep(30)" "$0"\n',
            )
            _write_executable(
                fake_bin / "bun",
                'if [ "${1:-}" = "--version" ]; then echo "1.3.12"; exit 0; fi\n'
                'echo bun-scan >>"$TRANSLATOR_SCA_TEST_LOG"\n'
                'case "${FAKE_BUN_MODE:-pass}" in\n'
                '  timeout) "$TRANSLATOR_SCA_TEST_CHILD" & wait;;\n'
                "  finding) exit 1;;\n"
                "  pass) exit 0;;\n"
                "esac\n",
            )
            _write_executable(
                fake_bin / "osv-scanner",
                'if [ "${1:-}" = "--version" ]; then\n'
                '  echo "osv-scanner version: 2.5.0"\n'
                '  echo "osv-scalibr version: test"\n'
                "  exit 0\n"
                "fi\n"
                'printf "osv-scan %s\\n" "$*" >>"$TRANSLATOR_SCA_TEST_LOG"\n'
                'exit "${FAKE_OSV_EXIT:-0}"\n',
            )
            if unsafe_scanner is not None:
                (fake_bin / unsafe_scanner).chmod(unsafe_scanner_mode)
            _write_executable(
                fake_bin / "grep",
                'printf "glib-reachability %s\\n" "$*" >>"$TRANSLATOR_SCA_TEST_LOG"\n'
                'case "$*" in *"translator-sca-glib."*)\n'
                '  exit "${FAKE_GLIB_POSITIVE_EXIT:-0}";;\n'
                "esac\n"
                'exit "${FAKE_GLIB_REACHABILITY_EXIT:-1}"\n',
            )

            gate_path = ROOT / "scripts/translator-sca"
            if patch_tool_identities:
                fake_repository = temporary / "repository"
                fake_scripts = fake_repository / "scripts"
                (fake_repository / "apps/translator-ui").mkdir(parents=True)
                fake_scripts.mkdir()
                gate_path = fake_scripts / "translator-sca"
                gate_source = (ROOT / "scripts/translator-sca").read_text()
                digest_replacements = {
                    "expected_cargo_audit_sha256": hashlib.sha256(
                        (fake_bin / "cargo-audit").read_bytes()
                    ).hexdigest(),
                    "expected_bun_sha256": hashlib.sha256(
                        (fake_bin / "bun").read_bytes()
                    ).hexdigest(),
                    "expected_osv_scanner_sha256": hashlib.sha256(
                        (fake_bin / "osv-scanner").read_bytes()
                    ).hexdigest(),
                }
                for constant, digest in digest_replacements.items():
                    gate_source, replacement_count = re.subn(
                        rf'(?m)^(readonly {constant}=")[0-9a-f]{{64}}("$)',
                        rf"\g<1>{digest}\g<2>",
                        gate_source,
                        count=1,
                    )
                    self.assertEqual(replacement_count, 1, constant)
                gate_source, replacement_count = re.subn(
                    r'(?m)^readonly grep_bin="/usr/bin/grep"$',
                    f'readonly grep_bin="{fake_bin / "grep"}"',
                    gate_source,
                    count=1,
                )
                self.assertEqual(replacement_count, 1, "grep_bin")
                gate_path.write_text(gate_source)
                gate_path.chmod(0o700)

            env = os.environ.copy()
            env.update(
                {
                    "FAKE_BUN_MODE": bun_mode,
                    "FAKE_CARGO_EXIT": str(cargo_exit),
                    "FAKE_CARGO_PATH": str(fake_bin / "cargo-audit"),
                    "FAKE_CARGO_REPLACEMENT": str(fake_bin / "cargo-audit-replacement"),
                    "FAKE_REPLACE_CARGO_AFTER_VERSION": (
                        "1" if replace_cargo_after_version else "0"
                    ),
                    "FAKE_CARGO_VERSION": cargo_version,
                    "FAKE_GLIB_POSITIVE_EXIT": str(glib_positive_exit),
                    "FAKE_GLIB_REACHABILITY_EXIT": str(glib_reachability_exit),
                    "FAKE_OSV_EXIT": str(osv_exit),
                    "PATH": f"{fake_bin}:{env['PATH']}",
                    "TRANSLATOR_CARGO_AUDIT": str(fake_bin / "cargo-audit"),
                    "TRANSLATOR_BUN": str(fake_bin / "bun"),
                    "TRANSLATOR_OSV_SCANNER": str(fake_bin / "osv-scanner"),
                    "TRANSLATOR_SCA_BUN_TIMEOUT": bun_timeout,
                    "TRANSLATOR_SCA_TEST_CHILD": str(child_marker),
                    "TRANSLATOR_SCA_TEST_LOG": str(call_log),
                }
            )
            started = time.monotonic()
            result = subprocess.run(
                [str(gate_path)],
                cwd="/tmp",
                env=env,
                text=True,
                capture_output=True,
                timeout=15,
                check=False,
            )
            elapsed = time.monotonic() - started
            calls = call_log.read_text() if call_log.exists() else ""

        return result, calls, child_marker, elapsed


if __name__ == "__main__":
    unittest.main()

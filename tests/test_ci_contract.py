from __future__ import annotations

import os
import re
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
VALIDATOR = ROOT / "scripts" / "translator-validate"
PUBLICATION_CHECK = ROOT / "scripts" / "translator-publication-check.bash"
GITLEAKS_ARCHIVE_SHA256 = (
    "79a3ab579b53f71efd634f3aaf7e04a0fa0cf206b7ed434638d1547a2470a66e"
)
GITLEAKS_EXECUTABLE_SHA256 = (
    "8b6fd684fcd5b4ebe39b68abb072ce59e1063ce7ed4abd556157697845f1f088"
)


class CiContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workflow = WORKFLOW.read_text(encoding="utf-8")

    def test_validation_wrapper_delegates_to_the_manifest_bound_runner(self) -> None:
        validator = VALIDATOR.read_text(encoding="utf-8")

        self.assertTrue(validator.startswith("#!/usr/bin/python3 -I\n"))
        self.assertEqual(validator.count('"/usr/bin/python3"'), 1)
        self.assertEqual(validator.count('"translator-test-manifest"'), 1)
        self.assertIn(
            'argv = ["/usr/bin/python3", "-I", str(runner), "run"]', validator
        )
        self.assertIn("os.execve(argv[0], argv, _sanitized_environment())", validator)
        self.assertNotIn("subprocess", validator)
        self.assertNotIn("shell=True", validator)

    def test_validation_launcher_rejects_hostile_startup_and_path_injection(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            attack_root = Path(temporary)
            marker = attack_root / "executed"
            startup = attack_root / "startup.sh"
            startup.write_text(f': >"{marker}"\nexit 0\n', encoding="utf-8")
            (attack_root / "sitecustomize.py").write_text(
                f"from pathlib import Path\nPath({str(marker)!r}).touch()\n",
                encoding="utf-8",
            )
            for interpreter in ("bash", "python3"):
                fake = attack_root / interpreter
                fake.write_text(f'#!/bin/sh\n: >"{marker}"\nexit 0\n', encoding="utf-8")
                fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
            env = {
                **os.environ,
                "BASH_ENV": str(startup),
                "ENV": str(startup),
                "PYTHONPATH": str(attack_root),
                "PYTHONINSPECT": "1",
                "PATH": f"{attack_root}:{os.environ.get('PATH', '')}",
                "SHELLOPTS": "xtrace",
                "PS4": "hostile-trace ",
            }

            result = subprocess.run(
                [str(VALIDATOR), "definitely-invalid"],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertIn(
                "usage: ./scripts/translator-validate deterministic", result.stderr
            )
            self.assertFalse(marker.exists())

    def test_ci_has_one_deterministic_validation_entrypoint(self) -> None:
        workflows = sorted(
            path.name
            for path in (ROOT / ".github" / "workflows").iterdir()
            if path.is_file()
        )
        self.assertEqual(workflows, ["ci.yml"])
        self.assertEqual(
            self.workflow.count("./scripts/translator-validate deterministic"),
            1,
        )
        for duplicate_gate in (
            "bun test",
            "cargo clippy",
            "cargo fmt",
            "cargo test",
            "ruff check",
            "ruff format",
            "unittest discover",
            "uv run pytest",
        ):
            self.assertNotIn(duplicate_gate, self.workflow)

    def test_ci_attaches_the_exact_pr_checkout_before_validation(self) -> None:
        attach = "git switch --create ci-validation-candidate"
        validate = "./scripts/translator-validate deterministic"

        self.assertEqual(self.workflow.count(attach), 1)
        self.assertLess(
            self.workflow.index("fetch-depth: 0"), self.workflow.index(attach)
        )
        self.assertEqual(self.workflow.count("persist-credentials: false"), 1)
        self.assertLess(self.workflow.index(attach), self.workflow.index(validate))
        self.assertNotIn("github.head_ref", self.workflow)
        self.assertNotIn("github.event.pull_request.head.sha", self.workflow)

    def test_actions_and_toolchain_are_immutable_and_exact(self) -> None:
        action_references = re.findall(
            r"^\s*uses:\s*([^\s#]+)", self.workflow, re.MULTILINE
        )
        self.assertEqual(
            action_references,
            [
                "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1",
                "actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97",
            ],
        )
        for reference in action_references:
            revision = reference.rsplit("@", maxsplit=1)[1]
            self.assertRegex(revision, r"^[0-9a-f]{40}$")

        for exact_contract in (
            "runs-on: ubuntu-24.04",
            "fetch-depth: 0",
            "rustup toolchain install 1.88.0",
            'python-version: "3.12"',
            'CARGO_AUDIT_VERSION: "0.22.2"',
            'UV_VERSION: "0.12.9"',
            'BUN_VERSION: "1.3.12"',
            'GITLEAKS_VERSION: "8.30.0"',
            'OSV_SCANNER_VERSION: "2.5.0"',
            'SHELLCHECK_VERSION: "0.11.0"',
        ):
            self.assertIn(exact_contract, self.workflow)

    def test_downloaded_validation_tools_are_checksum_verified(self) -> None:
        digest_contracts = (
            'CARGO_AUDIT_ARCHIVE_SHA256: "ab28a1bdb54db4d5d8ad5981cf1f959410370b3d28250dbd35f6a44248620e39"',
            'CARGO_AUDIT_EXECUTABLE_SHA256: "473b9a71e5cb5bde22f69c32f749c9b83931287d92dc36b91cb04f6705640ef2"',
            'UV_ARCHIVE_SHA256: "ec7a99cd05e0cd7f80243f135ce1361c76835cb0ee60055d14d20eba8eba1460"',
            'UV_EXECUTABLE_SHA256: "671793498fe0a545432e2524b6691ffb9eea4540d9fda43ca2f978df2dbf8426"',
            'BUN_ARCHIVE_SHA256: "11dc3ee11bc1695e149737c6ca3d5619302cf4346e6b8a6ec7988967ef01ddc5"',
            'BUN_EXECUTABLE_SHA256: "92a1cd8b6185f676010bb18e767dfc65c772273e79ae9984480843261939eeaa"',
            f'GITLEAKS_SHA256: "{GITLEAKS_ARCHIVE_SHA256}"',
            'OSV_SCANNER_SHA256: "edcfc41d257db36148f065055655fe3fcfc434b0b423ea67468a84c207524e0c"',
            'SHELLCHECK_SHA256: "b7af85e41cc99489dcc21d66c6d5f3685138f06d34651e6d34b42ec6d54fe6f6"',
        )
        for contract in digest_contracts:
            self.assertIn(contract, self.workflow)

        self.assertEqual(self.workflow.count("sha256sum --check"), 9)
        self.assertEqual(self.workflow.count("curl --fail --location"), 6)
        self.assertNotIn("cargo install cargo-audit", self.workflow)
        self.assertNotIn("astral-sh/setup-uv@", self.workflow)
        self.assertNotIn("oven-sh/setup-bun@", self.workflow)

    def test_gitleaks_archive_and_executable_have_distinct_digest_pins(
        self,
    ) -> None:
        publication_check = PUBLICATION_CHECK.read_text(encoding="utf-8")

        self.assertNotEqual(GITLEAKS_ARCHIVE_SHA256, GITLEAKS_EXECUTABLE_SHA256)
        self.assertIn(f'GITLEAKS_SHA256: "{GITLEAKS_ARCHIVE_SHA256}"', self.workflow)
        self.assertIn(
            "readonly expected_gitleaks_executable_sha256="
            f'"{GITLEAKS_EXECUTABLE_SHA256}"',
            publication_check,
        )

    def test_ci_has_least_privilege_and_bounded_execution(self) -> None:
        for contract in (
            "permissions:\n  contents: read",
            "cancel-in-progress: true",
            "timeout-minutes: 60",
        ):
            self.assertIn(contract, self.workflow)


if __name__ == "__main__":
    unittest.main()

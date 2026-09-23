from __future__ import annotations

import hashlib
import os
import shutil
import stat
import subprocess
import tempfile
import unittest
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GIT_BIN = "/usr/bin/git"
Content = str | bytes
CandidateHook = Callable[[Path], None]
RunEnvFactory = Callable[[Path], Mapping[str, str]]


def _write_executable(path: Path, body: str) -> None:
    path.write_text("#!/usr/bin/env bash\nset -u\n" + body, encoding="utf-8")
    path.chmod(0o700)


def _synthetic_secret(label: str) -> str:
    suffix = hashlib.sha256(label.encode()).hexdigest()[:36]
    return "gh" + "p_" + suffix


@dataclass(frozen=True)
class CandidateRun:
    result: subprocess.CompletedProcess[str]
    candidate: Path
    candidate_tree: str
    scanner_invocations: tuple[str, ...]


class PublicationGateTests(unittest.TestCase):
    def test_clean_candidate_passes_only_as_precommit_evidence(self) -> None:
        run = self._run_candidate()

        self.assertEqual(run.result.returncode, 0, self._output(run))
        self.assertIn(f"tree={run.candidate_tree}", run.result.stdout)
        self.assertIn("publication precommit receipt: v1", run.result.stdout)
        self.assertIn("release=false", run.result.stdout)
        self.assertNotIn("publication release receipt:", run.result.stdout)

    def test_frozen_release_binds_commit_tag_refs_and_reviewed_tree(self) -> None:
        run = self._run_candidate(gate_mode="release")

        self.assertEqual(run.result.returncode, 0, self._output(run))
        head = self.git(run.candidate, "rev-parse", "HEAD").stdout.strip()
        tag_object = self.git(
            run.candidate, "rev-parse", "refs/tags/v-test^{tag}"
        ).stdout.strip()
        self.assertIn(
            "publication release receipt: v1 "
            f"head={head} tree={run.candidate_tree} refs-sha256=",
            run.result.stdout,
        )
        self.assertIn(f"tag=v-test tag-object={tag_object}", run.result.stdout)
        self.assertIn(
            "publication frozen release commit, tag, refs, history, and tree are clean",
            run.result.stdout,
        )

    def test_release_rejects_wrong_reviewed_tree(self) -> None:
        run = self._run_candidate(gate_mode="release", expected_reviewed_tree="0" * 40)

        self.assert_failed(run, "release tree differs from the reviewed candidate tree")

    def test_release_requires_annotated_tag(self) -> None:
        run = self._run_candidate(gate_mode="release", annotated_release_tag=False)

        self.assert_failed(run, "release mode requires the named annotated tag")

    def test_release_tag_must_target_candidate_head(self) -> None:
        run = self._run_candidate(gate_mode="release", release_tag_target="HEAD^")

        self.assert_failed(
            run, "release tag does not identify the release candidate HEAD"
        )

    def test_release_rejects_annotated_tag_that_directly_targets_another_tag(
        self,
    ) -> None:
        def replace_release_tag_with_tag_of_tag(candidate: Path) -> None:
            head = self.git(candidate, "rev-parse", "HEAD").stdout.strip()
            self.git(
                candidate,
                "tag",
                "-a",
                "nested-target",
                "-m",
                "nested target",
                "HEAD",
            )
            self.git(
                candidate,
                "tag",
                "-f",
                "-a",
                "v-test",
                "-m",
                "outer release tag",
                "refs/tags/nested-target",
            )
            self.assertEqual(
                self.git(
                    candidate, "rev-parse", "refs/tags/v-test^{commit}"
                ).stdout.strip(),
                head,
            )
            outer_tag = self.git(
                candidate, "rev-parse", "refs/tags/v-test^{tag}"
            ).stdout.strip()
            self.assertIn(
                "\ntype tag\n",
                self.git(candidate, "cat-file", "tag", outer_tag).stdout,
            )

        run = self._run_candidate(
            gate_mode="release",
            post_stage=replace_release_tag_with_tag_of_tag,
        )

        self.assert_failed(
            run, "release tag object must directly identify the release candidate HEAD"
        )

    def test_release_rejects_outer_tag_with_mismatched_internal_name(self) -> None:
        def replace_release_tag_with_mismatched_name(candidate: Path) -> None:
            head = self.git(candidate, "rev-parse", "HEAD").stdout.strip()
            tag_object = self.git(
                candidate,
                "mktag",
                input_text=(
                    f"object {head}\n"
                    "type commit\n"
                    "tag v-other\n"
                    "tagger publication-test "
                    "<publication-test@example.invalid> 946684800 +0000\n"
                    "\n"
                    "mismatched release tag name\n"
                ),
            ).stdout.strip()
            self.git(
                candidate,
                "update-ref",
                "refs/tags/v-test",
                tag_object,
            )

        run = self._run_candidate(
            gate_mode="release",
            post_stage=replace_release_tag_with_mismatched_name,
        )

        self.assert_failed(
            run, "release tag object name does not match the requested tag"
        )

    def test_release_ref_drift_is_rejected(self) -> None:
        run = self._run_candidate(gate_mode="release", scanner_mutation="ref")

        self.assert_failed(run, "public refs changed during publication verification")

    def test_clean_candidate_passes_from_outside_repository(self) -> None:
        run = self._run_candidate(run_outside_repository=True)

        self.assertEqual(run.result.returncode, 0, self._output(run))

    def test_relative_scanner_path_is_resolved_before_temporary_chdir(self) -> None:
        def relative_scanner_environment(candidate: Path) -> Mapping[str, str]:
            scanner = candidate.parent / "relative-tools" / "gitleaks"
            scanner.parent.mkdir()
            self.write_fake_scanner(scanner)
            return {"TRANSLATOR_GITLEAKS_BIN": "./relative-tools/gitleaks"}

        run = self._run_candidate(
            run_env_factory=relative_scanner_environment,
            run_outside_repository=True,
        )

        self.assertEqual(run.result.returncode, 0, self._output(run))

    def test_launcher_rejects_invalid_argv_despite_hostile_interpreter_env(
        self,
    ) -> None:
        marker_holder: dict[str, Path] = {}

        def hostile_environment(candidate: Path) -> Mapping[str, str]:
            attack_root = candidate.parent / "interpreter-attack"
            attack_root.mkdir()
            marker = attack_root / "executed"
            marker_holder["path"] = marker
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
            return {
                "BASH_ENV": str(startup),
                "ENV": str(startup),
                "PYTHONPATH": str(attack_root),
                "PATH": f"{attack_root}:{os.environ.get('PATH', '')}",
                "SHELLOPTS": "xtrace",
                "PS4": "hostile-trace ",
            }

        run = self._run_candidate(
            gate_mode="definitely-invalid",
            run_env_factory=hostile_environment,
        )

        self.assertEqual(run.result.returncode, 2, self._output(run))
        self.assertIn(
            "usage: translator-publication-check candidate", run.result.stderr
        )
        self.assertFalse(marker_holder["path"].exists())

    def test_broken_scanner_cannot_false_pass_positive_controls(self) -> None:
        run = self._run_candidate(positive_status=0)

        self.assert_failed(run, "positive control was not detected")

    def test_fake_scanner_is_rejected_when_its_digest_is_not_bound(self) -> None:
        run = self._run_candidate(bind_scanner_identity=False)

        self.assert_failed(run, "gitleaks executable identity verification failed")

    def test_unsafe_scanner_file_permissions_fail_before_execution(self) -> None:
        for mode in (0o777, 0o2755, 0o4755):
            with self.subTest(mode=oct(mode)):
                run = self._run_candidate(scanner_mode=mode)

                self.assert_failed(
                    run, "gitleaks executable identity verification failed"
                )
                self.assertEqual(run.scanner_invocations, ())

    def test_scanner_self_mutation_during_scan_is_rejected(self) -> None:
        run = self._run_candidate(scanner_mutation="scanner")

        self.assert_failed(run, "gitleaks executable identity verification failed")

    def test_scanner_replacement_during_scan_is_rejected(self) -> None:
        run = self._run_candidate(scanner_mutation="scanner-replacement")

        self.assert_failed(run, "gitleaks executable identity verification failed")

    def test_operational_scanner_failure_cannot_false_pass(self) -> None:
        run = self._run_candidate(scan_status=2)

        self.assert_failed(run, "public Git history secret scan failed")

    def test_all_local_path_oracle_branches_are_rejected_without_leaking(self) -> None:
        private_values = {
            "home": "/" + "home/tester/Source/private",
            "root": "/" + "root/private",
            "mac_users": "/" + "Users/tester/src/private",
            "windows_backslash": "C:" + "\\" + "Users\\tester\\Source",
            "windows_slash": "C:" + "/" + "Users/tester/Source",
            "windows_backslash_lower": "c:" + "\\" + "users\\tester\\Source",
            "windows_slash_lower": "c:" + "/" + "users/tester/Source",
            "home_placeholder": "%" + "h/Source/private",
            "unrelated_repository": "uncle-" + "freud" + "-bot",
        }
        for branch, private_value in private_values.items():
            with self.subTest(branch=branch):
                run = self._run_candidate(
                    added_files={"crates/example/src/lib.rs": private_value + "\n"}
                )

                self.assert_failed(
                    run,
                    "candidate content contains a private home or source-checkout path",
                )
                self.assertNotIn(private_value, self._output(run))

    def test_runtime_path_scan_false_clean_cannot_bypass_positive_control(self) -> None:
        private_value = "/" + "home/tester/Source/private"
        run = self._run_candidate(
            added_files={"crates/example/src/lib.rs": private_value + "\n"},
            grep_positive_status=1,
            grep_status=1,
        )

        self.assert_failed(run, "runtime path scan positive control 0 was not detected")
        self.assertNotIn(private_value, self._output(run))

    def test_runtime_path_scan_operational_error_cannot_false_pass(self) -> None:
        run = self._run_candidate(grep_status=2)

        self.assert_failed(run, "candidate path scan failed operationally (2)")

    def test_symlink_modes_are_rejected_from_staged_index(self) -> None:
        cases = {
            "internal": ("docs/readme-link", "../README.md"),
            "dangling": ("scripts/dangling-link", "missing-target"),
            "escape": ("crates/example/escape-link", "/etc/passwd"),
        }
        for mode, (name, target) in cases.items():
            with self.subTest(mode=mode):
                run = self._run_candidate(added_symlinks={name: target})

                self.assert_failed(run, "staged worktree entry is malformed")
                self.assertTrue((run.candidate / name).is_symlink())

    def test_gitlink_mode_is_rejected_from_staged_index(self) -> None:
        name = "vendor/embedded-repository"
        run = self._run_candidate(added_gitlinks=(name,))

        self.assert_failed(run, "staged worktree entry is malformed")
        entry = self.git(run.candidate, "ls-files", "--stage", name)
        self.assertTrue(entry.stdout.startswith("160000 "), entry.stdout)

    def test_other_only_execute_bit_does_not_match_executable_index_mode(self) -> None:
        name = "scripts/example-tool"

        def make_index_executable(candidate: Path) -> None:
            (candidate / name).chmod(0o755)

        def remove_owner_execute(candidate: Path) -> None:
            (candidate / name).chmod(0o641)

        run = self._run_candidate(
            added_files={name: "#!/bin/sh\nexit 0\n"},
            pre_stage=make_index_executable,
            post_stage=remove_owner_execute,
        )

        self.assert_failed(
            run, "worktree bytes or executable mode differ from the staged candidate"
        )

    def test_staged_secret_then_clean_worktree_is_rejected(self) -> None:
        secret = _synthetic_secret("staged-secret")

        def clean_worktree(candidate: Path) -> None:
            (candidate / "docs/value.txt").write_text("safe\n", encoding="utf-8")

        run = self._run_candidate(
            added_files={"docs/value.txt": secret + "\n"}, post_stage=clean_worktree
        )

        self.assert_failed(
            run, "worktree bytes or executable mode differ from the staged candidate"
        )
        self.assertIn(secret, self.git(run.candidate, "show", ":docs/value.txt").stdout)
        self.assertEqual(
            (run.candidate / "docs/value.txt").read_text(encoding="utf-8"), "safe\n"
        )

    def test_clean_index_then_secret_worktree_is_rejected(self) -> None:
        secret = _synthetic_secret("worktree-secret")

        def poison_worktree(candidate: Path) -> None:
            (candidate / "docs/value.txt").write_text(secret + "\n", encoding="utf-8")

        run = self._run_candidate(
            added_files={"docs/value.txt": "safe\n"}, post_stage=poison_worktree
        )

        self.assert_failed(
            run, "worktree bytes or executable mode differ from the staged candidate"
        )
        self.assertEqual(
            self.git(run.candidate, "show", ":docs/value.txt").stdout, "safe\n"
        )
        self.assertIn(
            secret,
            (run.candidate / "docs/value.txt").read_text(encoding="utf-8"),
        )

    def test_staged_symlink_then_regular_worktree_is_rejected(self) -> None:
        name = "docs/value-link"

        def replace_with_regular(candidate: Path) -> None:
            path = candidate / name
            path.unlink()
            path.write_text("safe\n", encoding="utf-8")

        run = self._run_candidate(
            added_symlinks={name: "../README.md"}, post_stage=replace_with_regular
        )

        self.assert_failed(run, "staged worktree entry is malformed")
        entry = self.git(run.candidate, "ls-files", "--stage", name)
        self.assertTrue(entry.stdout.startswith("120000 "), entry.stdout)
        self.assertTrue((run.candidate / name).is_file())
        self.assertFalse((run.candidate / name).is_symlink())

    def test_staged_delete_then_recreated_worktree_file_is_rejected(self) -> None:
        name = "docs/removed.txt"

        def delete_before_stage(candidate: Path) -> None:
            (candidate / name).unlink()

        def recreate_after_stage(candidate: Path) -> None:
            (candidate / name).write_text("recreated\n", encoding="utf-8")

        run = self._run_candidate(
            baseline_files={name: "tracked\n"},
            pre_stage=delete_before_stage,
            post_stage=recreate_after_stage,
        )

        self.assert_failed(run, "untracked files are forbidden")
        self.assertTrue((run.candidate / name).is_file())
        indexed = self.git(
            run.candidate, "ls-files", "--error-unmatch", name, check=False
        )
        self.assertNotEqual(indexed.returncode, 0)

    def test_required_license_only_ignored_and_untracked_is_rejected(self) -> None:
        def remove_license_from_candidate(candidate: Path) -> None:
            with (candidate / ".git/info/exclude").open(
                "a", encoding="utf-8"
            ) as stream:
                stream.write("/LICENSE\n")
            self.git(candidate, "rm", "--cached", "LICENSE")

        run = self._run_candidate(pre_stage=remove_license_from_candidate)

        self.assert_failed(run, "active repository info/exclude rules are forbidden")
        self.assertTrue((run.candidate / "LICENSE").is_file())
        ignored = self.git(run.candidate, "status", "--porcelain", "--ignored")
        self.assertIn("!! LICENSE", ignored.stdout)

    def test_manifest_missing_candidate_path_is_rejected(self) -> None:
        run = self._run_candidate(manifest_remove={"README.md"})

        self.assert_failed(
            run, "candidate paths differ from the authoritative publication manifest"
        )

    def test_manifest_unexpected_path_is_rejected(self) -> None:
        run = self._run_candidate(manifest_add={"docs/not-in-candidate.txt"})

        self.assert_failed(
            run, "candidate paths differ from the authoritative publication manifest"
        )

    def test_root_and_nested_artifact_paths_are_rejected(self) -> None:
        forbidden_paths = (
            ".cache/value.txt",
            "src/.cache/value.txt",
            "dist/value.txt",
            "src/dist/value.txt",
            "audio/sample.flac",
            "models/private.gguf",
            "reports/result.txt",
            "src/reports/result.txt",
        )
        for path in forbidden_paths:
            with self.subTest(path=path):
                run = self._run_candidate(added_files={path: "artifact\n"})

                self.assert_failed(
                    run, "candidate path violates the public artifact policy"
                )
                self.assertNotIn(path, self._output(run))

    def test_archive_suffix_and_magic_are_rejected(self) -> None:
        cases: Mapping[str, Content] = {
            "docs/payload.zip": b"plain text is still an archive by policy",
            "docs/payload.bin": b"PK\x03\x04synthetic archive payload",
        }
        for path, content in cases.items():
            with self.subTest(path=path):
                run = self._run_candidate(added_files={path: content})

                self.assert_failed(
                    run, "compressed archives are forbidden in the candidate snapshot"
                )

    def test_utf16_invalid_utf8_and_nul_payloads_are_rejected(self) -> None:
        private_value = ("/" + "home/tester/Source/private").encode()
        cases = {
            "utf16le": "private".encode("utf-16-le"),
            "utf16be": "private".encode("utf-16-be"),
            "invalid_utf8": b"invalid-\xff-text",
            "nul_and_private_path": b"prefix\0" + private_value,
        }
        for label, content in cases.items():
            with self.subTest(label=label):
                run = self._run_candidate(added_files={f"docs/{label}.bin": content})

                self.assert_failed(
                    run,
                    "non-UTF-8 or control-bearing files are forbidden in the candidate snapshot",
                )
                self.assertNotIn(private_value.decode(), self._output(run))

    def test_repository_gitleaks_controls_are_rejected(self) -> None:
        for name in (".gitleaks.toml", ".gitleaksignore"):
            with self.subTest(name=name):
                run = self._run_candidate(added_files={name: "ignored\n"})

                self.assert_failed(
                    run,
                    "repository-local scanner or content-transform controls are forbidden",
                )

    def test_real_gitleaks_ignores_allow_comment_bypass(self) -> None:
        secret = _synthetic_secret("candidate-allow-comment")
        run = self._run_candidate(
            scanner=self.real_scanner(),
            added_files={"docs/token.txt": f"token={secret} # gitleaks:allow\n"},
            timeout=90,
        )

        self.assert_failed(run, "candidate tree secret scan failed")
        self.assertNotIn(secret, self._output(run))

    def test_shallow_repository_is_rejected(self) -> None:
        def make_shallow(candidate: Path) -> None:
            head = self.git(candidate, "rev-parse", "HEAD").stdout.strip()
            (candidate / ".git/shallow").write_text(head + "\n", encoding="ascii")

        run = self._run_candidate(post_stage=make_shallow)

        self.assert_failed(run, "shallow repository history is forbidden")

    def test_replace_ref_is_rejected(self) -> None:
        def add_replace_ref(candidate: Path) -> None:
            head = self.git(candidate, "rev-parse", "HEAD").stdout.strip()
            self.git(candidate, "update-ref", f"refs/replace/{head}", head)

        run = self._run_candidate(post_stage=add_replace_ref)

        self.assert_failed(run, "Git replace refs are forbidden")

    def test_grafts_file_is_rejected(self) -> None:
        def add_graft(candidate: Path) -> None:
            head = self.git(candidate, "rev-parse", "HEAD").stdout.strip()
            (candidate / ".git/info/grafts").write_text(head + "\n", encoding="ascii")

        run = self._run_candidate(post_stage=add_graft)

        self.assert_failed(run, "Git grafts are forbidden")

    def test_partial_or_promisor_repository_is_rejected(self) -> None:
        def mark_promisor(candidate: Path) -> None:
            self.git(candidate, "config", "remote.origin.promisor", "true")

        run = self._run_candidate(post_stage=mark_promisor)

        self.assert_failed(run, "partial or promisor repository history is forbidden")

    def test_detached_head_with_secret_only_in_detached_lineage_is_rejected(
        self,
    ) -> None:
        secret = _synthetic_secret("detached-lineage")

        def create_detached_lineage(candidate: Path) -> None:
            base = self.git(candidate, "rev-parse", "HEAD").stdout.strip()
            self.git(candidate, "checkout", "--detach", "-q", base)
            self.git(
                candidate,
                "commit",
                "--allow-empty",
                "-m",
                f"detached release {secret}",
            )

        run = self._run_candidate(history_setup=create_detached_lineage)

        self.assert_failed(run, "detached HEAD is forbidden")
        self.assertNotIn(secret, self._output(run))

    def test_real_history_commit_message_secret_is_rejected(self) -> None:
        secret = _synthetic_secret("history-commit-message")

        def add_secret_commit_message(candidate: Path) -> None:
            self.git(
                candidate,
                "commit",
                "--allow-empty",
                "-m",
                f"release marker {secret}",
            )

        run = self._run_candidate(
            scanner=self.real_scanner(),
            history_setup=add_secret_commit_message,
            timeout=90,
        )

        self.assert_failed(run, "secret scan failed")
        self.assertNotIn(secret, self._output(run))

    def test_real_history_annotated_tag_message_secret_is_rejected(self) -> None:
        secret = _synthetic_secret("history-tag-message")

        def add_secret_tag_message(candidate: Path) -> None:
            self.git(candidate, "tag", "-a", "release-test", "-m", f"note {secret}")

        run = self._run_candidate(
            scanner=self.real_scanner(),
            history_setup=add_secret_tag_message,
            timeout=90,
        )

        self.assert_failed(run, "secret scan failed")
        self.assertNotIn(secret, self._output(run))

    def test_precommit_receipt_cannot_cover_later_same_tree_metadata(self) -> None:
        scanner = self.real_scanner()
        run = self._run_candidate(scanner=scanner, timeout=90)

        self.assertEqual(run.result.returncode, 0, self._output(run))
        self.assertIn("release=false", run.result.stdout)
        old_head = self.git(run.candidate, "rev-parse", "HEAD").stdout.strip()
        secret = _synthetic_secret("post-gate-release-metadata")
        self.git(
            run.candidate,
            "commit",
            "--allow-empty",
            "-m",
            f"release marker {secret}",
        )
        self.git(run.candidate, "tag", "-a", "v-post-gate", "-m", "release")
        new_head = self.git(run.candidate, "rev-parse", "HEAD").stdout.strip()
        self.assertNotEqual(old_head, new_head)
        self.assertEqual(
            run.candidate_tree,
            self.git(run.candidate, "rev-parse", "HEAD^{tree}").stdout.strip(),
        )

        env = os.environ.copy()
        env["TRANSLATOR_GITLEAKS_BIN"] = str(scanner)
        result = subprocess.run(
            [
                str(run.candidate / "scripts/translator-publication-check"),
                "release",
                "v-post-gate",
                run.candidate_tree,
            ],
            cwd=run.candidate,
            env=env,
            text=True,
            capture_output=True,
            timeout=90,
            check=False,
        )

        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("secret scan failed", result.stdout + result.stderr)
        self.assertNotIn(secret, result.stdout + result.stderr)

    def test_real_history_ref_name_secret_is_rejected(self) -> None:
        secret = _synthetic_secret("history-ref-name")

        def add_secret_ref(candidate: Path) -> None:
            head = self.git(candidate, "rev-parse", "HEAD").stdout.strip()
            self.git(candidate, "update-ref", f"refs/heads/release/{secret}", head)

        run = self._run_candidate(
            scanner=self.real_scanner(), history_setup=add_secret_ref, timeout=90
        )

        self.assert_failed(run, "secret scan failed")
        self.assertNotIn(secret, self._output(run))

    def test_uploadpack_hide_refs_cannot_omit_secret_ref(self) -> None:
        secret = _synthetic_secret("hidden-history-ref")

        def add_hidden_secret_ref(candidate: Path) -> None:
            head = self.git(candidate, "rev-parse", "HEAD").stdout.strip()
            self.git(candidate, "update-ref", f"refs/heads/hidden/{secret}", head)
            self.git(candidate, "config", "uploadpack.hideRefs", "refs/heads/hidden")

        run = self._run_candidate(
            scanner=self.real_scanner(), history_setup=add_hidden_secret_ref, timeout=90
        )

        self.assert_failed(run, "secret scan failed")
        self.assertNotIn(secret, self._output(run))

    def test_real_raw_reachable_commit_header_secret_is_rejected(self) -> None:
        secret = _synthetic_secret("raw-commit-header")

        def add_raw_commit_header(candidate: Path) -> None:
            head = self.git(candidate, "rev-parse", "HEAD").stdout.strip()
            tree = self.git(candidate, "rev-parse", "HEAD^{tree}").stdout.strip()
            identity = "Publication Test <publication-test@example.invalid> 0 +0000"
            payload = (
                f"tree {tree}\n"
                f"parent {head}\n"
                f"author {identity}\n"
                f"committer {identity}\n"
                f"x-release-token {secret}\n"
                "\nbenign commit message\n"
            )
            commit = self.git(
                candidate,
                "hash-object",
                "-t",
                "commit",
                "-w",
                "--stdin",
                input_text=payload,
            ).stdout.strip()
            self.git(candidate, "update-ref", "refs/heads/raw-header", commit)

        run = self._run_candidate(
            scanner=self.real_scanner(), history_setup=add_raw_commit_header, timeout=90
        )

        self.assert_failed(run, "secret scan failed")
        self.assertNotIn(secret, self._output(run))

    def test_removed_historical_archive_is_rejected(self) -> None:
        def add_then_remove_archive(candidate: Path) -> None:
            path = candidate / "removed.bin"
            path.write_bytes(b"PK\x03\x04historical archive")
            self.git(candidate, "add", "removed.bin")
            self.git(candidate, "commit", "-m", "add historical archive")
            path.unlink()
            self.git(candidate, "add", "-A")
            self.git(candidate, "commit", "-m", "remove historical archive")

        run = self._run_candidate(history_setup=add_then_remove_archive)

        self.assert_failed(run, "compressed archives are forbidden in public history")

    def test_removed_historical_binary_is_rejected(self) -> None:
        def add_then_remove_binary(candidate: Path) -> None:
            path = candidate / "removed.bin"
            path.write_bytes(b"invalid-\xff-history")
            self.git(candidate, "add", "removed.bin")
            self.git(candidate, "commit", "-m", "add historical binary")
            path.unlink()
            self.git(candidate, "add", "-A")
            self.git(candidate, "commit", "-m", "remove historical binary")

        run = self._run_candidate(history_setup=add_then_remove_binary)

        self.assert_failed(
            run, "unapproved binary blobs are forbidden in public history"
        )

    def test_scanner_worktree_mutation_after_snapshot_is_rejected(self) -> None:
        run = self._run_candidate(scanner_mutation="worktree")

        self.assert_failed(
            run, "worktree bytes or executable mode differ from the staged candidate"
        )
        self.assertIn(
            "scanner mutation",
            (run.candidate / "README.md").read_text(encoding="utf-8"),
        )

    def test_scanner_ref_mutation_after_snapshot_is_rejected(self) -> None:
        run = self._run_candidate(scanner_mutation="ref")

        self.assert_failed(run, "public refs changed during publication verification")
        mutated_ref = self.git(
            run.candidate,
            "show-ref",
            "--verify",
            "refs/tags/scanner-mutation",
            check=False,
        )
        self.assertEqual(mutated_ref.returncode, 0, mutated_ref.stderr)

    def test_worktree_gitattributes_filter_is_rejected_before_helper_runs(self) -> None:
        self._assert_external_filter_is_never_run(use_common_attributes=False)

    def test_common_info_attributes_filter_is_rejected_before_helper_runs(self) -> None:
        self._assert_external_filter_is_never_run(use_common_attributes=True)

    def test_empty_and_symlink_common_info_attributes_are_rejected(self) -> None:
        for mode in ("empty", "symlink"):
            with self.subTest(mode=mode):

                def install(candidate: Path, selected_mode: str = mode) -> None:
                    attributes = candidate / ".git/info/attributes"
                    if selected_mode == "empty":
                        attributes.touch()
                    else:
                        attributes.symlink_to("/dev/null")

                run = self._run_candidate(post_stage=install)

                self.assert_failed(
                    run,
                    "repository info/attributes is forbidden during publication verification",
                )

    def test_control_byte_filename_cannot_spoof_success_or_leak_raw_name(self) -> None:
        spoof = "SPOOFED-PUBLICATION-SUCCESS"
        hostile_name = "docs/control\n" + spoof
        run = self._run_candidate(added_files={hostile_name: "safe\n"})

        self.assert_failed(
            run, "candidate paths containing control bytes are forbidden"
        )
        self.assertNotIn(spoof, self._output(run))

    def test_hostile_git_environment_cannot_redirect_candidate_or_configuration(
        self,
    ) -> None:
        run = self._run_candidate(
            run_env={
                "GIT_INDEX_FILE": "/dev/null",
                "GIT_NAMESPACE": "attacker",
                "GIT_CONFIG_COUNT": "1",
                "GIT_CONFIG_KEY_0": "core.repositoryformatversion",
                "GIT_CONFIG_VALUE_0": "999",
            }
        )

        self.assertEqual(run.result.returncode, 0, self._output(run))
        self.assertIn(run.candidate_tree, run.result.stdout)

    def test_hostile_git_attr_source_cannot_replace_candidate_attributes(self) -> None:
        alternate_ref = "refs/heads/alternate-attributes"

        def add_alternate_attributes_tree(candidate: Path) -> None:
            blob = self.git(
                candidate,
                "hash-object",
                "-w",
                "--stdin",
                input_text="README.md export-ignore\n",
            ).stdout.strip()
            tree = self.git(
                candidate,
                "mktree",
                input_text=f"100644 blob {blob}\t.gitattributes\n",
            ).stdout.strip()
            commit = self.git(
                candidate, "commit-tree", tree, "-m", "alternate attributes"
            ).stdout.strip()
            self.git(candidate, "update-ref", alternate_ref, commit)

        run = self._run_candidate(
            history_setup=add_alternate_attributes_tree,
            run_env={"GIT_ATTR_SOURCE": alternate_ref},
        )

        self.assertEqual(run.result.returncode, 0, self._output(run))
        self.assertIn(run.candidate_tree, run.result.stdout)

    def test_git_trace_environment_cannot_leak_repository_paths_or_refs(self) -> None:
        trace_paths: list[Path] = []

        def trace_environment(candidate: Path) -> Mapping[str, str]:
            trace_paths.extend(
                (
                    candidate.parent / "trace-setup.log",
                    candidate.parent / "trace-refs.log",
                    candidate.parent / "trace-packfile.pack",
                )
            )
            return {
                "GIT_TRACE_SETUP": str(trace_paths[0]),
                "GIT_TRACE_REFS": str(trace_paths[1]),
                "GIT_TRACE_PACKFILE": str(trace_paths[2]),
                "GIT_TRACE_REDACT": "0",
            }

        run = self._run_candidate(run_env_factory=trace_environment)

        self.assertEqual(run.result.returncode, 0, self._output(run))
        self.assertTrue(all(not path.exists() for path in trace_paths))
        self.assertNotIn(str(run.candidate), self._output(run))
        branch = self.git(run.candidate, "symbolic-ref", "--short", "HEAD")
        self.assertNotIn(branch.stdout.strip(), self._output(run))

    def _assert_external_filter_is_never_run(
        self, *, use_common_attributes: bool
    ) -> None:
        marker_holder: dict[str, Path] = {}

        def configure_filter(candidate: Path) -> None:
            marker = candidate.parent / "filter-was-run"
            helper = candidate.parent / "hostile-clean-filter"
            _write_executable(helper, f'printf x >"{marker}"\n/usr/bin/cat\n')
            marker_holder["path"] = marker
            self.git(candidate, "config", "filter.hostile.clean", str(helper))
            self.git(candidate, "config", "filter.hostile.required", "true")
            if use_common_attributes:
                (candidate / ".git/info/attributes").write_text(
                    "*.txt filter=hostile\n", encoding="utf-8"
                )

        added_files = (
            {"docs/filter.txt": "safe\n"}
            if use_common_attributes
            else {
                ".gitattributes": "*.txt filter=hostile\n",
                "docs/filter.txt": "safe\n",
            }
        )
        run = self._run_candidate(added_files=added_files, post_stage=configure_filter)

        expected_message = (
            "repository info/attributes is forbidden during publication verification"
            if use_common_attributes
            else "repository-local scanner or content-transform controls are forbidden"
        )
        self.assert_failed(run, expected_message)
        self.assertFalse(marker_holder["path"].exists())

    def _run_candidate(
        self,
        *,
        scanner: Path | None = None,
        positive_status: int = 1,
        scan_status: int = 0,
        grep_positive_status: int = 0,
        grep_status: int | None = None,
        baseline_files: Mapping[str, Content] | None = None,
        added_files: Mapping[str, Content] | None = None,
        added_symlinks: Mapping[str, str] | None = None,
        added_gitlinks: tuple[str, ...] = (),
        manifest_add: set[str] | None = None,
        manifest_remove: set[str] | None = None,
        history_setup: CandidateHook | None = None,
        pre_stage: CandidateHook | None = None,
        post_stage: CandidateHook | None = None,
        gate_mode: str = "candidate",
        expected_reviewed_tree: str | None = None,
        annotated_release_tag: bool = True,
        release_tag_target: str = "HEAD",
        scanner_mutation: str | None = None,
        bind_scanner_identity: bool = True,
        scanner_mode: int | None = None,
        run_env: Mapping[str, str] | None = None,
        run_env_factory: RunEnvFactory | None = None,
        run_outside_repository: bool = False,
        timeout: int = 45,
    ) -> CandidateRun:
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        temporary_root = Path(temporary_directory.name)
        candidate = temporary_root / "candidate"
        candidate.mkdir()

        script = candidate / "scripts/translator-publication-check"
        script.parent.mkdir()
        shutil.copy2(ROOT / "scripts/translator-publication-check", script)
        implementation = candidate / "scripts/translator-publication-check.bash"
        shutil.copy2(ROOT / "scripts/translator-publication-check.bash", implementation)
        if scanner is None:
            scanner = temporary_root / "gitleaks"
            self.write_fake_scanner(scanner)
        if scanner_mode is not None:
            scanner.chmod(scanner_mode)
        if bind_scanner_identity:
            self.bind_scanner_identity(implementation, scanner)

        required: dict[str, Content] = {
            "README.md": "# Candidate\n",
            "LICENSE": "test\n",
            "SECURITY.md": "test\n",
            "CONTRIBUTING.md": "test\n",
            ".env.example": "EXAMPLE=\n",
            "docs/publication/github-description.md": "test\n",
            "docs/publication/release-checklist.md": "test\n",
            "config/publication-files.txt": "placeholder\n",
        }
        required.update(baseline_files or {})
        self.write_files(candidate, required)

        self.git(candidate, "init", "-q")
        self.git(
            candidate,
            "config",
            "user.email",
            "publication-test@example.invalid",
        )
        self.git(candidate, "config", "user.name", "publication-test")
        self.stage_exact_manifest(candidate)
        self.git(candidate, "commit", "-m", "clean candidate")

        if history_setup is not None:
            history_setup(candidate)

        self.write_files(candidate, added_files or {})
        for name, target in (added_symlinks or {}).items():
            path = candidate / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.symlink_to(target)
        for name in added_gitlinks:
            self.add_gitlink(candidate, name)
        if pre_stage is not None:
            pre_stage(candidate)

        self.stage_exact_manifest(
            candidate,
            manifest_add=manifest_add or set(),
            manifest_remove=manifest_remove or set(),
        )
        self.assert_index_matches_worktree(
            candidate,
            check_manifest=not (manifest_add or manifest_remove),
        )
        candidate_tree = self.git(candidate, "write-tree").stdout.strip()

        if gate_mode == "release":
            self.git(
                candidate,
                "commit",
                "--allow-empty",
                "-m",
                "frozen release candidate",
            )
            if annotated_release_tag:
                self.git(
                    candidate,
                    "tag",
                    "-a",
                    "v-test",
                    "-m",
                    "frozen release tag",
                    release_tag_target,
                )
            else:
                self.git(candidate, "tag", "v-test", release_tag_target)

        if post_stage is not None:
            post_stage(candidate)

        fake_state = temporary_root / "fake-scanner-state"
        fake_state.mkdir()
        env = os.environ.copy()
        env.update(
            {
                "FAKE_POSITIVE_STATUS": str(positive_status),
                "FAKE_SCAN_STATUS": str(scan_status),
                "FAKE_STATE_DIR": str(fake_state),
                "FAKE_MUTATION_KIND": scanner_mutation or "",
                "FAKE_SCANNER_PATH": str(scanner),
                "FAKE_SOURCE_REPO": str(candidate),
                "FAKE_MUTATE_WORKTREE_FILE": str(candidate / "README.md"),
                "TRANSLATOR_GITLEAKS_BIN": str(scanner),
            }
        )
        env.update(run_env or {})
        if run_env_factory is not None:
            env.update(run_env_factory(candidate))

        if grep_status is not None:
            fake_grep = temporary_root / "grep"
            _write_executable(
                fake_grep,
                'source_path="${!#}"\n'
                'case "${source_path##*/}" in\n'
                "  exclude) exit 1;;\n"
                "  runtime-path-positive-control-*) "
                'exit "${FAKE_GREP_POSITIVE_STATUS}";;\n'
                '  *) exit "${FAKE_GREP_SCAN_STATUS}";;\n'
                "esac\n",
            )
            script_text = implementation.read_text(encoding="utf-8")
            pinned_grep = 'readonly grep_bin="/usr/bin/grep"'
            self.assertIn(pinned_grep, script_text)
            implementation.write_text(
                script_text.replace(pinned_grep, f'readonly grep_bin="{fake_grep}"', 1),
                encoding="utf-8",
            )
            self.git(candidate, "add", "scripts/translator-publication-check.bash")
            candidate_tree = self.git(candidate, "write-tree").stdout.strip()
            env.update(
                {
                    "FAKE_GREP_POSITIVE_STATUS": str(grep_positive_status),
                    "FAKE_GREP_SCAN_STATUS": str(grep_status),
                }
            )

        run_cwd = temporary_root if run_outside_repository else candidate
        gate_argv = [str(script), gate_mode]
        if gate_mode == "release":
            gate_argv.extend(["v-test", expected_reviewed_tree or candidate_tree])
        result = subprocess.run(
            gate_argv,
            cwd=run_cwd,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        invocation_log = fake_state / "scanner-invocations"
        invocations = (
            tuple(invocation_log.read_text(encoding="utf-8").splitlines())
            if invocation_log.exists()
            else ()
        )
        return CandidateRun(result, candidate, candidate_tree, invocations)

    def stage_exact_manifest(
        self,
        candidate: Path,
        *,
        manifest_add: set[str] | None = None,
        manifest_remove: set[str] | None = None,
    ) -> None:
        self.git(candidate, "add", "-A")
        listed = self.git(candidate, "ls-files", "-z").stdout.split("\0")
        paths = {path for path in listed if path}
        paths.update(manifest_add or set())
        paths.difference_update(manifest_remove or set())
        manifest = candidate / "config/publication-files.txt"
        manifest.write_text("\n".join(sorted(paths)) + "\n", encoding="utf-8")
        self.git(candidate, "add", "config/publication-files.txt")

    def assert_index_matches_worktree(
        self, candidate: Path, *, check_manifest: bool
    ) -> None:
        diff = self.git(
            candidate,
            "diff",
            "--quiet",
            "--ignore-submodules=none",
            check=False,
        )
        self.assertEqual(diff.returncode, 0, diff.stdout + diff.stderr)
        untracked = self.git(candidate, "ls-files", "--others", "--exclude-standard")
        self.assertEqual(untracked.stdout, "")
        cached_paths = [
            path
            for path in self.git(candidate, "ls-files", "-z").stdout.split("\0")
            if path
        ]
        has_control_path = any(
            any(ord(character) < 32 or ord(character) == 127 for character in path)
            for path in cached_paths
        )
        if check_manifest and not has_control_path:
            manifest_paths = (
                (candidate / "config/publication-files.txt")
                .read_text(encoding="utf-8")
                .splitlines()
            )
            self.assertEqual(manifest_paths, sorted(cached_paths))

    def add_gitlink(self, candidate: Path, name: str) -> None:
        nested = candidate / name
        nested.mkdir(parents=True)
        self.git(nested, "init", "-q")
        self.git(nested, "config", "user.email", "gitlink-test@example.invalid")
        self.git(nested, "config", "user.name", "gitlink-test")
        (nested / "tracked.txt").write_text("nested\n", encoding="utf-8")
        self.git(nested, "add", "tracked.txt")
        self.git(nested, "commit", "-m", "nested candidate")

    def write_fake_scanner(self, scanner: Path) -> None:
        _write_executable(
            scanner,
            'printf "%s\\n" "$*" >>"${FAKE_STATE_DIR}/scanner-invocations"\n'
            'if [ "${1:-}" = version ]; then printf "%s\\n" 8.30.0; exit 0; fi\n'
            'command_name="${1:-}"\n'
            'source_path="${!#}"\n'
            'positive_prefix="gh""p_"\n'
            'if [ "${command_name}" = stdin ]; then\n'
            '  stdin_path="${FAKE_STATE_DIR}/stdin.$$"\n'
            '  /usr/bin/cat >"${stdin_path}"\n'
            '  if /usr/bin/grep -a -F -q -- "${positive_prefix}" "${stdin_path}"; then\n'
            '    /usr/bin/rm -f -- "${stdin_path}"\n'
            '    exit "${FAKE_POSITIVE_STATUS:-1}"\n'
            "  fi\n"
            '  /usr/bin/rm -f -- "${stdin_path}"\n'
            "fi\n"
            'case "${source_path}" in\n'
            "  */positive-control|*/archive-positive-control.tar) "
            'exit "${FAKE_POSITIVE_STATUS:-1}";;\n'
            "esac\n"
            'if [ "${command_name}" = dir ] &&\n'
            '   [ "${source_path##*/}" = candidate-tree ] &&\n'
            '   [ ! -e "${FAKE_STATE_DIR}/mutated" ]; then\n'
            '  : >"${FAKE_STATE_DIR}/mutated"\n'
            '  case "${FAKE_MUTATION_KIND:-}" in\n'
            '    worktree) printf "scanner mutation\\n" >>"${FAKE_MUTATE_WORKTREE_FILE}";;\n'
            '    scanner) printf "# scanner mutation\\n" >>"$0";;\n'
            "    scanner-replacement)\n"
            '      replacement="${FAKE_SCANNER_PATH}.replacement"\n'
            '      /usr/bin/cp -- "$0" "${replacement}" || exit 2\n'
            '      /usr/bin/mv -- "${replacement}" "${FAKE_SCANNER_PATH}" || exit 2\n'
            "      ;;\n"
            "    ref)\n"
            '      head="$(/usr/bin/git -C "${FAKE_SOURCE_REPO}" rev-parse HEAD)" || exit 2\n'
            '      /usr/bin/git -C "${FAKE_SOURCE_REPO}" update-ref refs/tags/scanner-mutation "${head}" || exit 2\n'
            "      ;;\n"
            "  esac\n"
            "fi\n"
            'exit "${FAKE_SCAN_STATUS:-0}"\n',
        )

    def bind_scanner_identity(self, implementation: Path, scanner: Path) -> None:
        production_digest = (
            "8b6fd684fcd5b4ebe39b68abb072ce59e1063ce7ed4abd556157697845f1f088"
        )
        scanner_digest = hashlib.sha256(scanner.read_bytes()).hexdigest()
        source = implementation.read_text(encoding="utf-8")
        production_contract = (
            f'readonly expected_gitleaks_executable_sha256="{production_digest}"'
        )
        self.assertEqual(source.count(production_contract), 1)
        implementation.write_text(
            source.replace(
                production_contract,
                f'readonly expected_gitleaks_executable_sha256="{scanner_digest}"',
                1,
            ),
            encoding="utf-8",
        )

    def write_files(self, candidate: Path, files: Mapping[str, Content]) -> None:
        for name, content in files.items():
            path = candidate / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(content, bytes):
                path.write_bytes(content)
            else:
                path.write_text(content, encoding="utf-8")

    def git(
        self,
        candidate: Path,
        *args: str,
        check: bool = True,
        input_text: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        for name in tuple(env):
            if name.startswith("GIT_"):
                env.pop(name)
        env.update({"GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null"})
        return subprocess.run(
            [GIT_BIN, *args],
            cwd=candidate,
            env=env,
            text=True,
            capture_output=True,
            check=check,
            input=input_text,
        )

    def real_scanner(self) -> Path:
        configured = os.environ.get("TRANSLATOR_GITLEAKS_BIN")
        resolved = configured or shutil.which("gitleaks")
        if resolved is None:
            self.fail("real Gitleaks 8.30.0 is required for publication tests")
        scanner = Path(resolved).resolve()
        if not scanner.is_file() or not os.access(scanner, os.X_OK):
            self.fail("configured Gitleaks executable is unavailable")
        return scanner

    def assert_failed(self, run: CandidateRun, message: str) -> None:
        self.assertNotEqual(run.result.returncode, 0, self._output(run))
        self.assertIn(message, run.result.stderr, self._output(run))

    @staticmethod
    def _output(run: CandidateRun) -> str:
        return run.result.stdout + run.result.stderr


if __name__ == "__main__":
    unittest.main()

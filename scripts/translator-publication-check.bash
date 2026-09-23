#!/usr/bin/bash -p
set -euo pipefail

export LC_ALL=C

readonly git_bin="/usr/bin/git"
readonly grep_bin="/usr/bin/grep"
readonly tar_bin="/usr/bin/tar"
readonly python_bin="/usr/bin/python3"
readonly readlink_bin="/usr/bin/readlink"
readonly mktemp_bin="/usr/bin/mktemp"
readonly rm_bin="/usr/bin/rm"
readonly mkdir_bin="/usr/bin/mkdir"
readonly cmp_bin="/usr/bin/cmp"
readonly sha256sum_bin="/usr/bin/sha256sum"
readonly expected_gitleaks_version="8.30.0"
readonly expected_gitleaks_executable_sha256="8b6fd684fcd5b4ebe39b68abb072ce59e1063ce7ed4abd556157697845f1f088"
readonly publication_manifest="config/publication-files.txt"
readonly unrelated_repository_name='uncle-freud''-bot'
readonly slash="/"
readonly backslash="\\"
readonly home_pattern="${slash}"'home/'
readonly root_pattern="${slash}"'root/'
readonly users_pattern="${slash}"'Users/'
readonly windows_users_pattern="[[:alpha:]]:${backslash}${backslash}Users${backslash}${backslash}"
readonly windows_forward_users_pattern="[[:alpha:]]:${slash}"'Users/'
readonly systemd_home_pattern='%h/'"(Source|src)${slash}"
readonly local_path_pattern="(${home_pattern}|${root_pattern}|${users_pattern}|${windows_users_pattern}|${windows_forward_users_pattern}|${systemd_home_pattern}|${unrelated_repository_name})"
readonly public_icon_sha256="d7891c3bbd05e5884b36fd487b349e25565b5d2c47850e97edcef538f0d472c5"
readonly tauri_icon_sha256="53c03ef8d760c49bf582cf72b8f9973315d6d6876606aa9cec4c09c7b573fa81"

usage() {
  printf '%s\n' \
    'usage: translator-publication-check candidate' \
    '       translator-publication-check release <annotated-tag> <reviewed-tree>' \
    >&2
  exit 2
}

publication_mode="${1:-}"
release_tag=""
reviewed_tree=""
case "${publication_mode}" in
  candidate)
    [ "$#" -eq 1 ] || usage
    ;;
  release)
    [ "$#" -eq 3 ] || usage
    release_tag="$2"
    reviewed_tree="$3"
    ;;
  *) usage ;;
esac
readonly publication_mode release_tag reviewed_tree

fail() {
  printf 'publication check: %s\n' "$*" >&2
  exit 1
}

for required_bin in \
  "${git_bin}" \
  "${grep_bin}" \
  "${tar_bin}" \
  "${python_bin}" \
  "${readlink_bin}" \
  "${mktemp_bin}" \
  "${rm_bin}" \
  "${mkdir_bin}" \
  "${cmp_bin}" \
  "${sha256sum_bin}"; do
  [ -x "${required_bin}" ] || fail "required host utility is unavailable"
done

script_path="$("${readlink_bin}" -f -- "${BASH_SOURCE[0]}")" ||
  fail 'script path cannot be resolved'
project_root="$(cd -- "${script_path%/*}/.." && pwd -P)" ||
  fail 'repository path cannot be resolved'
gitleaks_command="${TRANSLATOR_GITLEAKS_BIN:-gitleaks}"
gitleaks_path="$(command -v -- "${gitleaks_command}" 2>/dev/null)" ||
  fail 'gitleaks is unavailable'
gitleaks_bin="$("${readlink_bin}" -f -- "${gitleaks_path}")" ||
  fail 'gitleaks path cannot be resolved'
[ -f "${gitleaks_bin}" ] && [ -x "${gitleaks_bin}" ] ||
  fail 'gitleaks is unavailable'
readonly gitleaks_command gitleaks_path gitleaks_bin

exec {gitleaks_fd}<"${gitleaks_bin}" || fail 'gitleaks cannot be opened'
readonly gitleaks_fd
readonly gitleaks_exec="/proc/self/fd/${gitleaks_fd}"

read_gitleaks_identity() {
  "${python_bin}" -I - \
    "${gitleaks_bin}" \
    "${gitleaks_exec}" \
    "${expected_gitleaks_executable_sha256}" <<'PY' 2>/dev/null
import errno
import hashlib
import hmac
import os
import stat
import sys

source_path, executable_fd_path, expected_digest = sys.argv[1:]
metadata_fields = (
    "st_dev",
    "st_ino",
    "st_mode",
    "st_nlink",
    "st_uid",
    "st_gid",
    "st_size",
    "st_mtime_ns",
    "st_ctime_ns",
)


def metadata(value: os.stat_result) -> tuple[int, ...]:
    return tuple(getattr(value, field) for field in metadata_fields)


def has_security_capability(path: str) -> bool:
    try:
        return bool(os.getxattr(path, "security.capability"))
    except OSError as error:
        unsupported = {errno.ENODATA, errno.ENOTSUP}
        if error.errno in unsupported:
            return False
        raise


flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
source_fd = os.open(source_path, flags)
try:
    before = os.fstat(source_fd)
    held = os.stat(executable_fd_path)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise SystemExit(1)
    unsafe_mode = stat.S_IWGRP | stat.S_IWOTH | stat.S_ISUID | stat.S_ISGID
    if (
        before.st_uid not in {0, os.geteuid()}
        or before.st_mode & unsafe_mode
        or before.st_mode & 0o111 == 0
        or has_security_capability(executable_fd_path)
    ):
        raise SystemExit(1)
    if metadata(before) != metadata(held):
        raise SystemExit(1)

    digest = hashlib.sha256()
    while chunk := os.read(source_fd, 1024 * 1024):
        digest.update(chunk)

    after = os.fstat(source_fd)
    path_after = os.stat(source_path, follow_symlinks=False)
    held_after = os.stat(executable_fd_path)
    if not (
        metadata(before)
        == metadata(after)
        == metadata(path_after)
        == metadata(held_after)
    ) or has_security_capability(executable_fd_path):
        raise SystemExit(1)
    observed_digest = digest.hexdigest()
    if not hmac.compare_digest(observed_digest, expected_digest):
        raise SystemExit(1)
finally:
    os.close(source_fd)

print(":".join(str(value) for value in metadata(before)) + ":" + observed_digest)
PY
}

gitleaks_identity="$(read_gitleaks_identity)" ||
  fail 'gitleaks executable identity verification failed'
[ -n "${gitleaks_identity}" ] ||
  fail 'gitleaks executable identity verification failed'
readonly gitleaks_identity

gitleaks_identity_is_current() {
  local observed_identity
  observed_identity="$(read_gitleaks_identity)" || return 1
  [ "${observed_identity}" = "${gitleaks_identity}" ]
}

assert_gitleaks_identity() {
  gitleaks_identity_is_current ||
    fail 'gitleaks executable identity verification failed'
}

if observed_gitleaks_version="$("${gitleaks_exec}" version 2>/dev/null)"; then
  gitleaks_version_status=0
else
  gitleaks_version_status=$?
fi
assert_gitleaks_identity
[ "${gitleaks_version_status}" -eq 0 ] ||
  fail 'gitleaks version cannot be determined'
observed_gitleaks_version="${observed_gitleaks_version%%$'\n'*}"
[ "${observed_gitleaks_version}" = "${expected_gitleaks_version}" ] ||
  fail "gitleaks version mismatch; expected ${expected_gitleaks_version}"

# Git repository selection, index/ref view, and scanner policy must not be
# replaceable by the caller.
unset \
  GIT_INDEX_FILE \
  GIT_NAMESPACE \
  GIT_CONFIG_COUNT \
  GIT_CONFIG_PARAMETERS \
  GIT_CONFIG_SYSTEM \
  GIT_CONFIG_GLOBAL \
  GIT_SHALLOW_FILE \
  GIT_DIR \
  GIT_WORK_TREE \
  GIT_COMMON_DIR \
  GIT_OBJECT_DIRECTORY \
  GIT_ALTERNATE_OBJECT_DIRECTORIES \
  GIT_REPLACE_REF_BASE \
  GIT_GRAFT_FILE \
  GIT_ATTR_SOURCE \
  GIT_EXEC_PATH \
  GIT_TEMPLATE_DIR \
  GIT_TRACE \
  GIT_TRACE_CURL \
  GIT_TRACE_CURL_NO_DATA \
  GIT_TRACE_FSMONITOR \
  GIT_TRACE_INDEX \
  GIT_TRACE_PACK_ACCESS \
  GIT_TRACE_PACKFILE \
  GIT_TRACE_PACKET \
  GIT_TRACE_PERFORMANCE \
  GIT_TRACE_REFS \
  GIT_TRACE_REDACT \
  GIT_TRACE_SETUP \
  GIT_TRACE_SHALLOW \
  GIT_TRACE2 \
  GIT_TRACE2_CONFIG_PARAMS \
  GIT_TRACE2_EVENT \
  GIT_TRACE2_PARENT_SID \
  GIT_TRACE2_PERF
export \
  GIT_CONFIG_NOSYSTEM=1 \
  GIT_CONFIG_GLOBAL=/dev/null \
  GIT_ATTR_NOSYSTEM=1 \
  GIT_NO_REPLACE_OBJECTS=1 \
  GIT_OPTIONAL_LOCKS=0 \
  GIT_REF_PARANOIA=1
unset GITLEAKS_CONFIG GITLEAKS_CONFIG_TOML TAR_OPTIONS

safe_git() {
  "${git_bin}" "$@"
}

source_git() {
  safe_git \
    -c core.autocrlf=false \
    -c core.attributesFile=/dev/null \
    -c core.excludesFile=/dev/null \
    -c core.fsmonitor=false \
    -c core.untrackedCache=false \
    -C "${project_root}" "$@" 2>/dev/null
}

git_root="$(source_git rev-parse --show-toplevel 2>/dev/null)" ||
  fail 'repository root cannot be resolved'
[ "$("${readlink_bin}" -f -- "${git_root}")" = "${project_root}" ] ||
  fail 'script is not running from its owning repository'

git_common_dir="$(
  source_git rev-parse --path-format=absolute --git-common-dir 2>/dev/null
)" || fail 'repository metadata directory cannot be resolved'
git_common_dir="$("${readlink_bin}" -f -- "${git_common_dir}")" ||
  fail 'repository metadata directory cannot be resolved'
object_format="$(source_git rev-parse --show-object-format 2>/dev/null)" ||
  fail 'repository object format cannot be resolved'
case "${object_format}" in
  sha1 | sha256) ;;
  *) fail 'repository object format is unsupported' ;;
esac
if [ "${publication_mode}" = release ]; then
  [[ "${reviewed_tree}" =~ ^[0-9a-f]+$ ]] ||
    fail 'reviewed candidate tree identifier is invalid'
  if { [ "${object_format}" = sha1 ] && [ "${#reviewed_tree}" -ne 40 ]; } ||
    { [ "${object_format}" = sha256 ] && [ "${#reviewed_tree}" -ne 64 ]; }; then
    fail 'reviewed candidate tree identifier is invalid'
  fi
  source_git check-ref-format "refs/tags/${release_tag}" >/dev/null 2>&1 ||
    fail 'release tag name is invalid'
fi

assert_source_scanner_controls_absent() {
  local control
  local exclude_status
  local info_exclude="${git_common_dir}/info/exclude"
  local local_transform_config
  local local_transform_status
  for control in .gitleaks.toml .gitleaksignore .gitattributes .lfsconfig; do
    if [ -e "${project_root}/${control}" ] || [ -L "${project_root}/${control}" ]; then
      fail 'repository-local scanner or content-transform controls are forbidden'
    fi
  done
  if [ -e "${git_common_dir}/info/attributes" ] ||
    [ -L "${git_common_dir}/info/attributes" ]; then
    fail 'repository info/attributes is forbidden during publication verification'
  fi
  [ ! -L "${info_exclude}" ] ||
    fail 'repository info/exclude symlinks are forbidden'
  if [ -e "${info_exclude}" ]; then
    [ -f "${info_exclude}" ] ||
      fail 'repository info/exclude is not a regular file'
    set +e
    "${grep_bin}" -E -v '^($|#)' "${info_exclude}" >/dev/null 2>&1
    exclude_status=$?
    set -e
    case "${exclude_status}" in
      0) fail 'active repository info/exclude rules are forbidden' ;;
      1) ;;
      *) fail 'repository info/exclude cannot be inspected' ;;
    esac
  fi
  set +e
  local_transform_config="$(
    source_git config --get-regexp \
      '^(filter\..*\.(clean|smudge|process|required)|diff\..*\.(command|textconv))$' \
      2>/dev/null
  )"
  local_transform_status=$?
  set -e
  case "${local_transform_status}" in
    0) fail 'repository-local content transform commands are forbidden' ;;
    1) ;;
    *) fail 'repository-local content transform configuration cannot be inspected' ;;
  esac
  [ -z "${local_transform_config}" ] ||
    fail 'repository-local content transform commands are forbidden'
}

assert_source_scanner_controls_absent

assert_history_source_complete() {
  local config_query
  local config_status
  local config_value
  local replace_refs
  local shallow_state

  shallow_state="$(source_git rev-parse --is-shallow-repository 2>/dev/null)" ||
    fail 'repository history completeness cannot be determined'
  [ "${shallow_state}" = false ] || fail 'shallow repository history is forbidden'

  replace_refs="$(source_git for-each-ref --format='%(refname)' refs/replace)" ||
    fail 'replace refs cannot be enumerated'
  [ -z "${replace_refs}" ] || fail 'Git replace refs are forbidden'
  [ ! -e "${git_common_dir}/info/grafts" ] &&
    [ ! -L "${git_common_dir}/info/grafts" ] || fail 'Git grafts are forbidden'

  for config_query in \
    'extensions.partialClone' \
    '^remote\..*\.promisor$'; do
    set +e
    if [ "${config_query}" = 'extensions.partialClone' ]; then
      config_value="$(
        source_git config --get "${config_query}" 2>/dev/null
      )"
      config_status=$?
    else
      config_value="$(
        source_git config --get-regexp "${config_query}" 2>/dev/null
      )"
      config_status=$?
    fi
    set -e
    case "${config_status}" in
      0) fail 'partial or promisor repository history is forbidden' ;;
      1) ;;
      *) fail 'partial-clone configuration cannot be inspected' ;;
    esac
    [ -z "${config_value}" ] ||
      fail 'partial or promisor repository history is forbidden'
  done
}

assert_history_source_complete

temporary_parent="$("${readlink_bin}" -f -- "${TMPDIR:-/tmp}")" ||
  fail 'temporary directory parent cannot be resolved'
[ -d "${temporary_parent}" ] || fail 'temporary directory parent is unavailable'
case "${temporary_parent}/" in
  "${project_root}/"*) fail 'temporary directory must be outside the repository' ;;
esac

temporary_root="$(
  "${mktemp_bin}" -d -- "${temporary_parent}/translator-publication.XXXXXX"
)" || fail 'temporary publication workspace cannot be created'
[ -d "${temporary_root}" ] && [ ! -L "${temporary_root}" ] &&
  [ -O "${temporary_root}" ] || fail 'temporary publication workspace is unsafe'
[ "${temporary_root%/*}" = "${temporary_parent}" ] ||
  fail 'temporary publication workspace escaped its parent'
case "${temporary_root##*/}" in
  translator-publication.*) ;;
  *) fail 'temporary publication workspace has an invalid name' ;;
esac

cleanup() {
  if [ -n "${temporary_root:-}" ] && [ -d "${temporary_root}" ] &&
    [ ! -L "${temporary_root}" ] && [ -O "${temporary_root}" ] &&
    [ "${temporary_root%/*}" = "${temporary_parent}" ]; then
    "${rm_bin}" -rf -- "${temporary_root}"
  else
    printf '%s\n' 'publication check: refused unsafe temporary cleanup' >&2
  fi
}
trap cleanup EXIT

empty_gitleaks_ignore="${temporary_root}/empty-gitleaks-ignore"
: >"${empty_gitleaks_ignore}"
readonly -a gitleaks_common=(
  --no-banner
  --no-color
  --log-level error
  --redact=100
  --exit-code 1
  --ignore-gitleaks-allow
  --gitleaks-ignore-path "${empty_gitleaks_ignore}"
  --max-decode-depth 5
  --max-target-megabytes 0
)

gitleaks_from_temp() {
  local scan_status
  gitleaks_identity_is_current || return 125
  if (cd -- "${temporary_root}" && "${gitleaks_exec}" "$@"); then
    scan_status=0
  else
    scan_status=$?
  fi
  gitleaks_identity_is_current || return 125
  return "${scan_status}"
}

expect_gitleaks_finding() {
  local label="$1"
  shift
  local status
  set +e
  gitleaks_from_temp "$@" >/dev/null 2>&1
  status=$?
  set -e
  case "${status}" in
    1) ;;
    0) fail "${label} positive control was not detected" ;;
    *) fail "${label} positive control failed operationally (${status})" ;;
  esac
}

# Every alternation of the local-path oracle gets its own behavior control.
path_controls=(
  "$(printf '/%s/%s/%s/%s' home translator-check Source example)"
  "$(printf '/%s/%s' root private)"
  "$(printf '/%s/%s/%s/%s' Users translator-check src example)"
  "$(printf '%s:\\%s\\%s\\%s' C Users translator-check Source)"
  "$(printf '%s:/%s/%s/%s' C Users translator-check Source)"
  "$(printf '%s:\\%s\\%s\\%s' c users translator-check Source)"
  "$(printf '%s:/%s/%s/%s' c users translator-check Source)"
  "$(printf '%%%s/%s/%s' h Source example)"
  "$(printf '%s-%s-%s' uncle freud bot)"
)
for control_index in "${!path_controls[@]}"; do
  control_file="${temporary_root}/runtime-path-positive-control-${control_index}"
  printf '%s\n' "${path_controls[control_index]}" >"${control_file}"
  set +e
  "${grep_bin}" -Ei --binary-files=text \
    "${local_path_pattern}" "${control_file}" >/dev/null 2>&1
  control_status=$?
  set -e
  case "${control_status}" in
    0) ;;
    1) fail "runtime path scan positive control ${control_index} was not detected" ;;
    *) fail "runtime path scan positive control ${control_index} failed operationally" ;;
  esac
done

# Real scanner controls cover git, directory, allow-comment, and binary-stdin modes.
positive_repo="${temporary_root}/positive-control"
safe_git init -q "${positive_repo}"
safe_git -C "${positive_repo}" config user.email translator-publication-check@example.invalid
safe_git -C "${positive_repo}" config user.name translator-publication-check
positive_digest_line="$(
  printf '%s' translator-publication-positive-control | "${sha256sum_bin}"
)" || fail 'scanner positive-control token cannot be constructed'
positive_digest="${positive_digest_line%% *}"
positive_suffix="${positive_digest:0:36}"
positive_token="$(printf 'gh%s_%s' p "${positive_suffix}")"
printf 'token=%s # gitleaks:allow\n' "${positive_token}" \
  >"${positive_repo}/credential.txt"
safe_git -C "${positive_repo}" add credential.txt
safe_git -C "${positive_repo}" commit -qm 'synthetic scanner control'

expect_gitleaks_finding \
  'gitleaks git' \
  git "${gitleaks_common[@]}" --timeout 30 --log-opts='--all --text' \
  "${positive_repo}"
expect_gitleaks_finding \
  'gitleaks directory' \
  dir "${gitleaks_common[@]}" --timeout 30 --max-archive-depth 0 \
  "${positive_repo}"

archive_positive_root="${temporary_root}/archive-positive-control"
"${mkdir_bin}" -p -- "${archive_positive_root}"
printf 'token=%s\n' "${positive_token}" \
  >"${archive_positive_root}/credential.txt"
archive_positive_tar="${temporary_root}/archive-positive-control.tar"
"${tar_bin}" --create --file="${archive_positive_tar}" \
  --directory="${archive_positive_root}" credential.txt ||
  fail 'gitleaks archive positive control could not be created'
expect_gitleaks_finding \
  'gitleaks archive' \
  dir "${gitleaks_common[@]}" --timeout 30 --max-archive-depth 2 \
  "${archive_positive_tar}"

set +e
printf '\0token=%s # gitleaks:allow\n' "${positive_token}" |
  gitleaks_from_temp stdin "${gitleaks_common[@]}" --timeout 30 \
    >/dev/null 2>&1
stdin_control_status=("${PIPESTATUS[@]}")
set -e
[ "${stdin_control_status[0]}" -eq 0 ] ||
  fail 'gitleaks stdin positive control could not be produced'
case "${stdin_control_status[1]}" in
  1) ;;
  0) fail 'gitleaks binary stdin positive control was not detected' ;;
  *) fail 'gitleaks binary stdin positive control failed operationally' ;;
esac

expect_metadata_pipeline_finding() {
  local kind="$1"
  local metadata_repo="${temporary_root}/metadata-${kind}-positive-control"
  local -a pipeline_status
  safe_git init -q "${metadata_repo}"
  safe_git -C "${metadata_repo}" \
    config user.email translator-publication-check@example.invalid
  safe_git -C "${metadata_repo}" \
    config user.name translator-publication-check
  printf '%s\n' 'benign metadata control' >"${metadata_repo}/content.txt"
  safe_git -C "${metadata_repo}" add content.txt
  if [ "${kind}" = commit ]; then
    safe_git -C "${metadata_repo}" commit -qm "${positive_token}"
  else
    safe_git -C "${metadata_repo}" commit -qm 'benign metadata control'
  fi
  case "${kind}" in
    commit) ;;
    tag) safe_git -C "${metadata_repo}" tag -am "${positive_token}" control ;;
    ref) safe_git -C "${metadata_repo}" branch "${positive_token}" ;;
    *) fail 'unknown history metadata positive-control kind' ;;
  esac
  set +e
  safe_git -C "${metadata_repo}" \
    fast-export --signed-commits=verbatim --signed-tags=verbatim --all |
    gitleaks_from_temp stdin "${gitleaks_common[@]}" --timeout 30 \
      >/dev/null 2>&1
  pipeline_status=("${PIPESTATUS[@]}")
  set -e
  [ "${pipeline_status[0]}" -eq 0 ] ||
    fail "history-${kind} positive control could not be exported"
  case "${pipeline_status[1]}" in
    1) ;;
    0) fail "history-${kind} positive control was not detected" ;;
    *) fail "history-${kind} positive control failed operationally" ;;
  esac
}

for metadata_kind in commit tag ref; do
  expect_metadata_pipeline_finding "${metadata_kind}"
done

source_head="$(source_git rev-parse --verify 'HEAD^{commit}' 2>/dev/null)" ||
  fail 'release candidate HEAD cannot be resolved'
candidate_tree="$(source_git write-tree)" ||
  fail 'staged candidate tree cannot be written'
[[ "${candidate_tree}" =~ ^[0-9a-f]{40,64}$ ]] ||
  fail 'candidate snapshot returned an invalid tree identifier'
release_tag_object=""

assert_release_binding() {
  local head_tree
  local tag_binding_status
  local tag_commit
  local tag_object
  local tag_object_contents

  [ "${publication_mode}" = release ] || return 0
  head_tree="$(source_git rev-parse --verify 'HEAD^{tree}' 2>/dev/null)" ||
    fail 'release commit tree cannot be resolved'
  [ "${candidate_tree}" = "${reviewed_tree}" ] ||
    fail 'release tree differs from the reviewed candidate tree'
  [ "${head_tree}" = "${candidate_tree}" ] ||
    fail 'release mode requires a clean committed candidate tree'
  tag_object="$(
    source_git rev-parse --verify "refs/tags/${release_tag}^{tag}" 2>/dev/null
  )" || fail 'release mode requires the named annotated tag'
  tag_commit="$(
    source_git rev-parse --verify "refs/tags/${release_tag}^{commit}" 2>/dev/null
  )" || fail 'release tag does not resolve to a commit'
  [ "${tag_commit}" = "${source_head}" ] ||
    fail 'release tag does not identify the release candidate HEAD'
  tag_object_contents="${temporary_root}/release-tag-object"
  source_git cat-file tag "${tag_object}" >"${tag_object_contents}" ||
    fail 'release tag object cannot be read'
  set +e
  "${python_bin}" -I - \
    "${tag_object_contents}" "${source_head}" "${release_tag}" <<'PY' \
    >/dev/null 2>&1
from __future__ import annotations

import os
import sys
from pathlib import Path

tag_path, source_head, release_tag = sys.argv[1:]
data = Path(tag_path).read_bytes()
header_end = data.find(b"\n\n")
if header_end < 0:
    raise SystemExit(2)
header = data[:header_end]
if b"\x00" in header or b"\r" in header:
    raise SystemExit(2)
lines = header.split(b"\n")
if len(lines) != 4 or not lines[3].startswith(b"tagger "):
    raise SystemExit(2)
if lines[0] != b"object " + source_head.encode("ascii"):
    raise SystemExit(3)
if lines[1] != b"type commit":
    raise SystemExit(3)
if lines[2] != b"tag " + os.fsencode(release_tag):
    raise SystemExit(4)
PY
  tag_binding_status=$?
  set -e
  case "${tag_binding_status}" in
    0) ;;
    2) fail 'release tag object header is malformed' ;;
    3) fail 'release tag object must directly identify the release candidate HEAD' ;;
    4) fail 'release tag object name does not match the requested tag' ;;
    *) fail 'release tag object cannot be verified' ;;
  esac
  if [ -n "${release_tag_object}" ] &&
    [ "${tag_object}" != "${release_tag_object}" ]; then
    fail 'release tag changed during publication verification'
  fi
  release_tag_object="${tag_object}"
}

assert_no_transform_control_paths() {
  local candidate_path
  local directory
  local index_flag
  local index_paths="${temporary_root}/index-paths-before-diff"
  source_git ls-files -z --cached >"${index_paths}" ||
    fail 'staged paths cannot be enumerated'
  while IFS= read -r -d '' candidate_path; do
    [[ ! "${candidate_path}" =~ [[:cntrl:]] ]] ||
      fail 'candidate paths containing control bytes are forbidden'
    case "${candidate_path}" in
      .gitattributes | */.gitattributes)
        fail 'content-transform attribute files are forbidden'
        ;;
    esac
    directory="${candidate_path%/*}"
    while [ "${directory}" != "${candidate_path}" ] && [ -n "${directory}" ]; do
      if [ -e "${project_root}/${directory}/.gitattributes" ] ||
        [ -L "${project_root}/${directory}/.gitattributes" ]; then
        fail 'content-transform attribute files are forbidden'
      fi
      candidate_path="${directory}"
      directory="${candidate_path%/*}"
    done
  done <"${index_paths}"

  source_git ls-files -v -z --cached >"${index_paths}.flags" ||
    fail 'staged index flags cannot be enumerated'
  while IFS= read -r -d '' flagged_path; do
    index_flag="${flagged_path%% *}"
    [ "${index_flag}" = H ] ||
      fail 'assume-unchanged, skip-worktree, or nonstandard index flags are forbidden'
  done <"${index_paths}.flags"
}

assert_worktree_matches_index() {
  local comparison_status
  local index_entries="${temporary_root}/worktree-index-entries"
  local status
  local untracked_paths="${temporary_root}/untracked-paths"

  source_git ls-files --stage -z >"${index_entries}" ||
    fail 'staged worktree entries cannot be enumerated'
  set +e
  "${python_bin}" -I - \
    "${project_root}" "${index_entries}" "${object_format}" <<'PY' \
    >/dev/null 2>&1
from __future__ import annotations

import hashlib
import os
import stat
import sys
from pathlib import Path, PurePosixPath

root_path, entries_path, object_format = sys.argv[1:]
entries = [
    entry
    for entry in Path(entries_path).read_bytes().split(b"\0")
    if entry
]
hash_factory = {"sha1": hashlib.sha1, "sha256": hashlib.sha256}.get(
    object_format
)
if hash_factory is None:
    raise SystemExit(2)

open_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
root_fd = os.open(root_path, open_flags | os.O_DIRECTORY)
try:
    for entry in entries:
        try:
            metadata, raw_path = entry.split(b"\t", 1)
            raw_mode, raw_oid, raw_stage = metadata.split(b" ")
            path = raw_path.decode("utf-8")
            mode = raw_mode.decode("ascii")
            expected_oid = raw_oid.decode("ascii")
        except (UnicodeDecodeError, ValueError):
            raise SystemExit(2)
        pure_path = PurePosixPath(path)
        if (
            raw_stage != b"0"
            or mode not in {"100644", "100755"}
            or len(expected_oid) not in {40, 64}
            or any(character not in "0123456789abcdef" for character in expected_oid)
            or pure_path.is_absolute()
            or any(part in {"", ".", ".."} for part in pure_path.parts)
        ):
            raise SystemExit(2)

        directory_fd = os.dup(root_fd)
        try:
            for component in pure_path.parts[:-1]:
                next_fd = os.open(
                    component,
                    open_flags | os.O_DIRECTORY,
                    dir_fd=directory_fd,
                )
                os.close(directory_fd)
                directory_fd = next_fd
            file_fd = os.open(
                pure_path.parts[-1], open_flags, dir_fd=directory_fd
            )
        except OSError:
            raise SystemExit(3)
        finally:
            os.close(directory_fd)

        try:
            before = os.fstat(file_fd)
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                raise SystemExit(3)
            if before.st_size > 8 * 1024 * 1024:
                raise SystemExit(4)
            executable = bool(before.st_mode & stat.S_IXUSR)
            if executable != (mode == "100755"):
                raise SystemExit(5)
            digest = hash_factory()
            digest.update(f"blob {before.st_size}\0".encode("ascii"))
            while chunk := os.read(file_fd, 1024 * 1024):
                digest.update(chunk)
            after = os.fstat(file_fd)
        finally:
            os.close(file_fd)
        stable_fields = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_nlink",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if any(getattr(before, name) != getattr(after, name) for name in stable_fields):
            raise SystemExit(5)
        if digest.hexdigest() != expected_oid:
            raise SystemExit(5)
finally:
    os.close(root_fd)
PY
  comparison_status=$?
  set -e
  case "${comparison_status}" in
    0) ;;
    2) fail 'staged worktree entry is malformed' ;;
    3) fail 'worktree candidate path is missing, linked, or not a regular file' ;;
    4) fail 'worktree candidate file exceeds the public artifact size limit' ;;
    5) fail 'worktree bytes or executable mode differ from the staged candidate' ;;
    *) fail 'worktree bytes cannot be verified' ;;
  esac
  set +e
  source_git ls-files -z --others --exclude-standard >"${untracked_paths}"
  status=$?
  set -e
  [ "${status}" -eq 0 ] || fail 'untracked candidate paths cannot be enumerated'
  [ ! -s "${untracked_paths}" ] ||
    fail 'untracked files are forbidden in the staged release candidate'
}

capture_source_ref_state() {
  local output="$1"
  local symbolic_head
  local symbolic_status
  set +e
  symbolic_head="$(source_git symbolic-ref -q HEAD 2>/dev/null)"
  symbolic_status=$?
  set -e
  case "${symbolic_status}" in
    0) printf 'symbolic-head %s\n' "${symbolic_head}" >"${output}" ;;
    1) fail 'detached HEAD is forbidden for a release candidate' ;;
    *) fail 'HEAD reference state cannot be determined' ;;
  esac
  source_git show-ref --head --dereference >>"${output}" ||
    fail 'repository refs cannot be captured'
}

capture_public_refs() {
  source_git show-ref --dereference >"$1" ||
    fail 'public refs cannot be captured'
}

assert_no_transform_control_paths
assert_worktree_matches_index
assert_release_binding
refs_before="${temporary_root}/refs-before"
refs_after="${temporary_root}/refs-after"
public_refs_before="${temporary_root}/public-refs-before"
mirror_refs="${temporary_root}/mirror-refs"
capture_source_ref_state "${refs_before}"
capture_public_refs "${public_refs_before}"
refs_digest_line="$("${sha256sum_bin}" "${refs_before}")" ||
  fail 'public ref-set identity cannot be computed'
refs_sha256="${refs_digest_line%% *}"

# Freeze all public refs through a bundle. Unlike a local clone transport,
# bundle creation does not honor uploadpack.hideRefs from repository config.
history_bundle="${temporary_root}/history.bundle"
source_git bundle create "${history_bundle}" --all ||
  fail 'public history bundle cannot be created'
history_repo="${temporary_root}/history.git"
safe_git clone --mirror --quiet "${history_bundle}" "${history_repo}" \
  2>/dev/null ||
  fail 'public history snapshot cannot be created'
[ "$(safe_git --git-dir="${history_repo}" rev-parse --is-shallow-repository)" = false ] ||
  fail 'public history snapshot is incomplete'
safe_git --git-dir="${history_repo}" show-ref --dereference >"${mirror_refs}" ||
  fail 'public history snapshot refs cannot be captured'
"${cmp_bin}" -s "${public_refs_before}" "${mirror_refs}" ||
  fail 'public history snapshot omitted or changed a ref'
safe_git --git-dir="${history_repo}" fsck --full --strict >/dev/null 2>&1 ||
  fail 'public history snapshot failed object-integrity verification'

set +e
gitleaks_from_temp git "${gitleaks_common[@]}" \
  --timeout 120 --max-archive-depth 2 --log-opts='--all --text' \
  "${history_repo}" >/dev/null 2>&1
history_git_scan_status=$?
set -e
[ "${history_git_scan_status}" -eq 0 ] ||
  fail 'public Git history secret scan failed'

# A fast-export stream includes raw binary blobs, names, refs, commit messages,
# and tag messages that patch-oriented scanners can otherwise omit.
set +e
GIT_NO_REPLACE_OBJECTS=1 safe_git --git-dir="${history_repo}" \
  fast-export --signed-commits=verbatim --signed-tags=verbatim --all \
    2>/dev/null |
  gitleaks_from_temp stdin "${gitleaks_common[@]}" --timeout 120 \
    >/dev/null 2>&1
history_stream_status=("${PIPESTATUS[@]}")
set -e
[ "${history_stream_status[0]}" -eq 0 ] ||
  fail 'public history export failed'
[ "${history_stream_status[1]}" -eq 0 ] ||
  fail 'public history raw-stream secret scan failed'

# Compressed historical blobs cannot be certified without an unbounded
# recursive extractor, so this release policy rejects them outright.
history_object_scan_input="${temporary_root}/history-object-scan"
set +e
"${python_bin}" -I - \
  "${git_bin}" "${history_repo}" "${history_object_scan_input}" <<'PY' \
  >/dev/null 2>&1
from __future__ import annotations

import hashlib
import subprocess
import sys

git_bin, git_dir, scan_path = sys.argv[1:]
git = [git_bin, "--git-dir", git_dir]
approved_binary_sha256 = {
    "d7891c3bbd05e5884b36fd487b349e25565b5d2c47850e97edcef538f0d472c5",
    "53c03ef8d760c49bf582cf72b8f9973315d6d6876606aa9cec4c09c7b573fa81",
    "bfbb54d0336d48078c7be8c7b39d9e507500f983281df989e61041748ce0d1ab",
    "1be2b72352fd126904e328764eef36e8f53d1cbf4877542cdf511dfb2a386875",
    "5e053bb3806ed14488f15009deba97f5ac747fec2a3027bf574ddea4911aafe9",
    "7ff4b2f581cba766a372c6a4a84f569e09b92e04d31a273b5b076700fc2ffd8e",
}


def run(*args: str) -> bytes:
    completed = subprocess.run(
        [*git, *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        timeout=120,
    )
    if completed.returncode != 0:
        raise SystemExit(2)
    return completed.stdout


def is_archive(data: bytes) -> bool:
    prefixes = (
        b"PK\x03\x04",
        b"PK\x05\x06",
        b"PK\x07\x08",
        b"\x1f\x8b",
        b"BZh",
        b"\xfd7zXZ\x00",
        b"7z\xbc\xaf'\x1c",
        b"Rar!\x1a\x07",
        b"!<arch>\n",
        b"070701",
        b"070702",
        b"070707",
        b"MSCF",
        b"\xed\xab\xee\xdb",
        b"(\xb5/\xfd",
        b"\x04\x22M\x18",
    )
    return data.startswith(prefixes) or (
        len(data) >= 265 and data[257:262] == b"ustar"
    )


object_rows = run("rev-list", "--objects", "--all").splitlines()
object_ids = sorted(
    {row.split(b" ", 1)[0].decode("ascii") for row in object_rows if row}
)
with open(scan_path, "wb") as scan:
    for object_id in object_ids:
        object_type = run("cat-file", "-t", object_id).strip()
        if object_type not in {b"blob", b"commit", b"tag", b"tree"}:
            raise SystemExit(2)
        try:
            object_size = int(run("cat-file", "-s", object_id))
        except ValueError:
            raise SystemExit(2)
        if object_size > 8 * 1024 * 1024:
            raise SystemExit(13)
        data = run("cat-file", object_type.decode("ascii"), object_id)
        if len(data) != object_size:
            raise SystemExit(2)
        scan.write(object_id.encode("ascii") + b" " + object_type + b"\n")
        scan.write(data + b"\n")
        scan.write(data.replace(b"\0", b"") + b"\n")
        if object_type != b"blob":
            continue
        if is_archive(data[:560]):
            raise SystemExit(10)
        if data.startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise SystemExit(11)
        if hashlib.sha256(data).hexdigest() in approved_binary_sha256:
            continue
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            raise SystemExit(12)
        if any(
            (ord(character) < 32 and character not in "\t\n\r")
            or ord(character) == 127
            for character in text
        ):
            raise SystemExit(12)
PY
history_archive_status=$?
set -e
case "${history_archive_status}" in
  0) ;;
  10) fail 'compressed archives are forbidden in public history' ;;
  11) fail 'Git LFS pointers are forbidden in public history' ;;
  12) fail 'unapproved binary blobs are forbidden in public history' ;;
  13) fail 'public history object exceeds the artifact size limit' ;;
  *) fail 'public history archive policy could not be verified' ;;
esac

set +e
gitleaks_from_temp stdin "${gitleaks_common[@]}" --timeout 120 \
  <"${history_object_scan_input}" >/dev/null 2>&1
history_object_secret_status=$?
set -e
[ "${history_object_secret_status}" -eq 0 ] ||
  fail 'raw reachable Git object secret scan failed'

path_fingerprint() {
  local digest
  digest="$(printf '%s' "$1" | "${sha256sum_bin}")" ||
    fail 'candidate path fingerprint cannot be computed'
  printf '%s' "${digest%% *}"
}

fail_path() {
  local message="$1"
  local candidate_path="$2"
  fail "${message} (path-sha256=$(path_fingerprint "${candidate_path}"))"
}

verify_publication_manifest() {
  local manifest_path="$1"
  local candidate_list="$2"
  set +e
  "${python_bin}" -I - "${manifest_path}" "${candidate_list}" <<'PY' >/dev/null 2>&1
from __future__ import annotations

import sys
from pathlib import PurePosixPath

manifest_path, candidate_path = sys.argv[1:]
payload = open(manifest_path, "rb").read()
if not payload.endswith(b"\n") or b"\0" in payload or b"\r" in payload:
    raise SystemExit(2)
try:
    expected = payload[:-1].decode("utf-8").split("\n")
    actual = [
        value.decode("utf-8")
        for value in open(candidate_path, "rb").read().split(b"\0")
        if value
    ]
except UnicodeDecodeError:
    raise SystemExit(2)
if not expected or any(not value for value in expected):
    raise SystemExit(2)
if expected != sorted(set(expected)):
    raise SystemExit(2)
for value in expected:
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise SystemExit(2)
if set(expected) != set(actual) or len(actual) != len(set(actual)):
    raise SystemExit(3)
PY
  manifest_status=$?
  set -e
  case "${manifest_status}" in
    0) ;;
    2) fail 'authoritative publication manifest is malformed' ;;
    3) fail 'candidate paths differ from the authoritative publication manifest' ;;
    *) fail 'authoritative publication manifest could not be verified' ;;
  esac
}

# The authoritative release candidate is exactly the staged Git index captured
# before the public-history snapshot. No checkout, filter, fsmonitor, hook, or
# custom tree synthesizer runs here.

candidate_entries="${temporary_root}/candidate-entries"
candidate_paths="${temporary_root}/candidate-paths"
: >"${candidate_paths}"
source_git ls-files --stage -z >"${candidate_entries}" ||
  fail 'candidate snapshot entries cannot be enumerated'
while IFS= read -r -d '' candidate_entry; do
  candidate_mode="${candidate_entry%% *}"
  candidate_path="${candidate_entry#*$'\t'}"
  [ "${candidate_path}" != "${candidate_entry}" ] ||
    fail 'candidate snapshot entry is malformed'
  [[ ! "${candidate_path}" =~ [[:cntrl:]] ]] ||
    fail 'candidate paths containing control bytes are forbidden'
  case "${candidate_mode}" in
    100644 | 100755) ;;
    120000) fail_path 'candidate symlink is forbidden' "${candidate_path}" ;;
    *) fail_path 'candidate entry is not a regular file' "${candidate_path}" ;;
  esac
  if [[ "${candidate_path}" == scripts/* ]]; then
    expected_mode=100755
  else
    expected_mode=100644
  fi
  [ "${candidate_mode}" = "${expected_mode}" ] ||
    fail_path 'candidate executable mode differs from publication policy' \
      "${candidate_path}"
  printf '%s\0' "${candidate_path}" >>"${candidate_paths}"
done <"${candidate_entries}"

while IFS= read -r -d '' candidate_path; do
  lower_path="${candidate_path,,}"
  case "${lower_path}" in
    .env.example) ;;
    .env | .env.* | */.env | */.env.* | \
      .gitleaks.toml | */.gitleaks.toml | \
      .gitleaksignore | */.gitleaksignore | \
      .gitattributes | */.gitattributes | \
      .gitmodules | */.gitmodules | .lfsconfig | */.lfsconfig | \
      target | target/* | */target | */target/* | \
      node_modules | node_modules/* | */node_modules | */node_modules/* | \
      dist | dist/* | */dist | */dist/* | \
      .venv | .venv/* | */.venv | */.venv/* | \
      __pycache__ | __pycache__/* | */__pycache__ | */__pycache__/* | \
      .pytest_cache | .pytest_cache/* | */.pytest_cache | */.pytest_cache/* | \
      .ruff_cache | .ruff_cache/* | */.ruff_cache | */.ruff_cache/* | \
      .cache | .cache/* | */.cache | */.cache/* | \
      .model-cache | .model-cache/* | */.model-cache | */.model-cache/* | \
      output | output/* | */output | */output/* | \
      debug-captures | debug-captures/* | \
      */debug-captures | */debug-captures/* | \
      reports | reports/* | */reports | */reports/* | \
      report | report/* | */report | */report/* | \
      artifacts | artifacts/* | */artifacts | */artifacts/* | \
      secrets | secrets/* | */secrets | */secrets/* | \
      runtime | runtime/* | */runtime | */runtime/* | \
      .playwright-cli | .playwright-cli/* | \
      */.playwright-cli | */.playwright-cli/* | \
      .idea | .idea/* | */.idea | */.idea/* | \
      .vscode | .vscode/* | */.vscode | */.vscode/* | \
      docs/benchmarks | docs/benchmarks/* | \
      docs/planning | docs/planning/* | \
      repo-c4.json | */repo-c4.json | \
      coverage.xml | */coverage.xml | .coverage | */.coverage | \
      .ds_store | */.ds_store | *.sock | *.log | *.pem | *.key | \
      *.onnx | *.pt | *.pth | *.safetensors | *.gguf | *.ggml | \
      *.ckpt | *.tflite | *.engine | *.plan | *.mlmodel | \
      *.pcm | *.wav | *.flac | *.mp3 | *.ogg | *.opus | *.m4a | \
      *.aac | *.mp4 | *.mkv | *.webm | *.mov | *.avi | \
      *.csv | *.tsv | *.jsonl | *.parquet | *.arrow | *.feather | \
      *.sqlite | *.sqlite3 | *.db)
      fail_path 'candidate path violates the public artifact policy' "${candidate_path}"
      ;;
    models/*)
      [ "${lower_path}" = 'models/manifest.json' ] ||
        fail_path 'bundled model asset is forbidden' "${candidate_path}"
      ;;
  esac
done <"${candidate_paths}"

candidate_root="${temporary_root}/candidate-tree"
"${mkdir_bin}" -p -- "${candidate_root}"
candidate_archive="${temporary_root}/candidate.tar"
source_git archive --format=tar --output="${candidate_archive}" \
  "${candidate_tree}" || fail 'candidate snapshot cannot be archived'
"${tar_bin}" --extract --file="${candidate_archive}" \
  --directory="${candidate_root}" --no-same-owner --no-same-permissions ||
  fail 'candidate snapshot cannot be extracted'

while IFS= read -r -d '' candidate_path; do
  if [ -L "${candidate_root}/${candidate_path}" ] ||
    [ ! -f "${candidate_root}/${candidate_path}" ]; then
    fail_path 'candidate archive entry is not a regular file' "${candidate_path}"
  fi
done <"${candidate_paths}"

verify_publication_manifest \
  "${candidate_root}/${publication_manifest}" "${candidate_paths}"

require_candidate_file() {
  [ -f "${candidate_root}/$1" ] && [ ! -L "${candidate_root}/$1" ] ||
    fail 'required public file is missing from the candidate snapshot'
}

for required_file in \
  README.md \
  LICENSE \
  SECURITY.md \
  CONTRIBUTING.md \
  .env.example \
  config/publication-files.txt \
  docs/publication/github-description.md \
  docs/publication/release-checklist.md; do
  require_candidate_file "${required_file}"
done

candidate_scan_input="${temporary_root}/candidate-path-scan"
set +e
"${python_bin}" -I - \
  "${candidate_root}" "${candidate_paths}" "${candidate_scan_input}" <<'PY' \
  >/dev/null 2>&1
from __future__ import annotations

import os
import stat
import sys
from pathlib import Path, PurePosixPath

root = Path(sys.argv[1])
paths = [value for value in Path(sys.argv[2]).read_bytes().split(b"\0") if value]
output_path = Path(sys.argv[3])
approved_binary = {
    "apps/translator-ui/public/icon.png",
    "apps/translator-ui/src-tauri/icons/icon.png",
}
archive_suffixes = (
    ".zip", ".tar", ".tgz", ".tar.gz", ".gz", ".bz2", ".xz", ".7z",
    ".rar", ".jar", ".war", ".ear", ".whl", ".egg", ".deb", ".rpm",
    ".apk", ".ipa", ".docx", ".xlsx", ".pptx", ".odt", ".ods",
    ".odp", ".cpio", ".cab", ".zst", ".lz4",
)
archive_prefixes = (
    b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08", b"\x1f\x8b", b"BZh",
    b"\xfd7zXZ\x00", b"7z\xbc\xaf'\x1c", b"Rar!\x1a\x07", b"!<arch>\n",
    b"070701", b"070702", b"070707", b"MSCF", b"\xed\xab\xee\xdb",
    b"(\xb5/\xfd", b"\x04\x22M\x18",
)


def archive_payload(path: str, data: bytes) -> bool:
    lowered = path.lower()
    return (
        lowered.endswith(archive_suffixes)
        or data.startswith(archive_prefixes)
        or (len(data) >= 265 and data[257:262] == b"ustar")
    )


with output_path.open("wb") as output:
    for raw_path in paths:
        try:
            decoded_path = raw_path.decode("utf-8")
        except UnicodeDecodeError:
            raise SystemExit(2)
        pure_path = PurePosixPath(decoded_path)
        if pure_path.is_absolute() or any(
            part in {"", ".", ".."} for part in pure_path.parts
        ):
            raise SystemExit(2)
        candidate_file = root / decoded_path
        file_stat = os.stat(candidate_file, follow_symlinks=False)
        if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_nlink != 1:
            raise SystemExit(2)
        if file_stat.st_size > 8 * 1024 * 1024:
            raise SystemExit(3)
        data = candidate_file.read_bytes()
        if len(data) != file_stat.st_size:
            raise SystemExit(2)
        if archive_payload(decoded_path, data[:560]):
            raise SystemExit(4)
        if data.startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise SystemExit(5)
        if decoded_path not in approved_binary:
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                raise SystemExit(6)
            if any(
                (ord(character) < 32 and character not in "\t\n\r")
                or ord(character) == 127
                for character in text
            ):
                raise SystemExit(6)
        output.write(os.fsencode(decoded_path) + b"\n")
        output.write(data + b"\n")
        output.write(data.replace(b"\0", b"") + b"\n")
PY
candidate_policy_status=$?
set -e
case "${candidate_policy_status}" in
  0) ;;
  2) fail 'candidate path encoding or structure is invalid' ;;
  3) fail 'candidate file exceeds the public artifact size limit' ;;
  4) fail 'compressed archives are forbidden in the candidate snapshot' ;;
  5) fail 'Git LFS pointers are forbidden in the candidate snapshot' ;;
  6) fail 'non-UTF-8 or control-bearing files are forbidden in the candidate snapshot' ;;
  *) fail 'candidate artifact policy could not be verified' ;;
esac

verify_binary_asset() {
  local relative_path="$1"
  local expected_digest="$2"
  local observed_line
  local observed_digest
  if [ ! -e "${candidate_root}/${relative_path}" ]; then
    return
  fi
  observed_line="$("${sha256sum_bin}" "${candidate_root}/${relative_path}")" ||
    fail 'approved binary asset digest cannot be computed'
  observed_digest="${observed_line%% *}"
  [ "${observed_digest}" = "${expected_digest}" ] ||
    fail 'approved binary asset digest differs from policy'
}

verify_binary_asset \
  apps/translator-ui/public/icon.png "${public_icon_sha256}"
verify_binary_asset \
  apps/translator-ui/src-tauri/icons/icon.png "${tauri_icon_sha256}"

set +e
gitleaks_from_temp dir "${gitleaks_common[@]}" \
  --timeout 120 --max-archive-depth 0 "${candidate_root}" \
  >/dev/null 2>&1
candidate_secret_status=$?
set -e
case "${candidate_secret_status}" in
  0) ;;
  125) fail 'gitleaks executable identity verification failed' ;;
  *) fail 'candidate tree secret scan failed' ;;
esac

# Scan the generated artifact as an archive as well as its extracted tree.
# Binary/control-bearing unapproved members have already failed byte policy.
set +e
gitleaks_from_temp dir "${gitleaks_common[@]}" \
  --timeout 120 --max-archive-depth 2 "${candidate_archive}" \
  >/dev/null 2>&1
candidate_archive_secret_status=$?
set -e
[ "${candidate_archive_secret_status}" -eq 0 ] ||
  fail 'candidate archive secret scan failed'

# The normalized stream covers approved binary assets too. NUL/control-bearing
# non-approved candidate files have already failed the deterministic byte policy.
set +e
gitleaks_from_temp stdin "${gitleaks_common[@]}" --timeout 120 \
  <"${candidate_scan_input}" >/dev/null 2>&1
candidate_stream_secret_status=$?
set -e
[ "${candidate_stream_secret_status}" -eq 0 ] ||
  fail 'candidate raw-stream secret scan failed'

set +e
"${grep_bin}" -Ei --binary-files=text \
  "${local_path_pattern}" "${candidate_scan_input}" >/dev/null 2>&1
local_path_status=$?
set -e
case "${local_path_status}" in
  0) fail 'candidate content contains a private home or source-checkout path' ;;
  1) ;;
  *) fail "candidate path scan failed operationally (${local_path_status})" ;;
esac

# Bind the receipt to stable source content, index, HEAD, and public refs.
assert_source_scanner_controls_absent
assert_history_source_complete
assert_no_transform_control_paths
assert_worktree_matches_index
assert_release_binding
[ "$(source_git rev-parse --verify 'HEAD^{commit}')" = "${source_head}" ] ||
  fail 'release candidate HEAD changed during publication verification'
capture_source_ref_state "${refs_after}"
"${cmp_bin}" -s "${refs_before}" "${refs_after}" ||
  fail 'public refs changed during publication verification'
candidate_tree_after="$(source_git write-tree)" ||
  fail 'staged candidate tree cannot be re-read'
[ "${candidate_tree_after}" = "${candidate_tree}" ] ||
  fail 'release candidate tree changed during publication verification'

if [ "${publication_mode}" = release ]; then
  printf \
    'publication release receipt: v1 head=%s tree=%s refs-sha256=%s tag=%s tag-object=%s\n' \
    "${source_head}" "${candidate_tree}" "${refs_sha256}" \
    "${release_tag}" "${release_tag_object}"
  printf '%s\n' \
    'publication frozen release commit, tag, refs, history, and tree are clean'
else
  printf \
    'publication precommit receipt: v1 head=%s tree=%s refs-sha256=%s release=false\n' \
    "${source_head}" "${candidate_tree}" "${refs_sha256}"
  printf '%s\n' \
    'publication precommit candidate only; this is not release evidence'
fi

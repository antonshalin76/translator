# Automatic product acceptance

The user's 2026-10-01 instruction removes manual listening and manual review
handoffs from execution. Production remains unchanged until measured product
gates pass. Automated speech recognition and semantic judgments do not certify
subjective voice naturalness or turn an ambiguous reference into acoustic truth.

## Integrated execution order

1. Capture the actual complete output PCM from unchanged main
   Small -> NLLB -> Piper and all three existing fork arms on the same frozen
   24-case development screen. Compare MT and independently transcribed output
   separately. Use both OpenAI and Google ASR and independent Claude/OpenAI
   semantic judgments; calibrate each evaluator with positive/negative controls.
   Retain uncertainty, errors, drops and every unrun case. No default promotion
   or release decision from this development set alone.
2. Finish the open-speaker path using one retained synchronized physical
   acquisition/AEC graph, not independent resampled `parec` streams. Recheck
   exclusive hardware custody before arming an isolated nondefault physical
   pair. Prove the real 30-second/60-second acoustic gates, revocation,
   Start/Stop/restart and teardown before admitting that graph into runtime.
3. Freeze an independent licensed holdout meeting EVAL-0, then run the full
   voice/mode/direction/fallback matrix, paired baseline comparison, native
   packaged-app controls, authorized real call endpoints and 30-minute soak.
   Missing endpoint credentials or speaker metadata remain explicit unavailable
   prerequisites; virtual fixtures do not replace these gates.
4. Build from the frozen passing candidate, test installation and rollback,
   publish the fork commit and reproducible artifacts, and only then merge,
   activate and release. No human listening gate is scheduled.

## This stage's acceptance and ownership

| Behavior | Owner | Narrow verification |
| --- | --- | --- |
| Critical facts in MT and audible transcript checked independently; missing transcript counted as a drop | `benchmark/task6.py` | 20 new regression cases, complete 88-test suite |
| Capture disabled leaves result/files unchanged; only verified terminal, health and drained session may persist PCM | Existing original/fork runners | Provider event/lifecycle spies plus real WAV readback |
| Private exclusive durable WAV; exact format/hash/count and directory/file identity at send time | `translator_product_pcm.py` | Filesystem faults, cancellation, symlinks, replacement, collisions and readback |
| Whole-run/arm cleanup and exact input bindings before network; custody/remote-cleanup uncertainty stops sends | `translator_cloud_chain_audio.py` | Paired matrix, counted transports and post-preflight mutation |
| Both ASR providers and both semantic judges required; ambiguity or audible-only corruption cannot pass | Cloud diagnostic runner | Distinct eight-output bidirectional fixtures and judgment controls |
| Numeric spelling normalized without erasing amount, unit, sign, time, negation or duplicate identifiers | NeMo/Jiwer scoring boundary | Six deterministic wrapper tests and explicit real RU/EN grammar integration |

Pydantic owns response validation and JSON schema; standard-library WAV handling
owns the container. Existing provider adapters own HTTPS and remote-file cleanup.
The diagnostic owns its private attempt ledger, not product admission, routing,
UI state or release policy. Those product responsibilities are unchanged.

Independent BDD critic/auditor and pre-RED SRP review passed. Task6's initial
15 failures and four missing-transcript regressions became green. PCM capture
had 20 failures and 37 missing-helper errors before implementation. Cloud tests
also reproduced two substantive provenance/custody errors; both became green.
Final independent architecture/source review passed for the bounded diagnostic.
No additional behavior-preserving refactor was required after these fixes.

## Validation ledger

All runs bind exact source/model/input hashes. New capture/evaluation evidence
is necessary because the older receipts contain only output PCM hashes, not
the output WAVs. They cannot be reused for independently transcribing output.

| Gate | Fingerprint / environment | State |
| --- | --- | --- |
| Task6 regression | Two owned source/test files; sidecar environment | 88 PASS |
| Fork runner and PCM custody | Final worker files; sidecar environment | 151 PASS |
| Original runner | Original-sidecar-first isolated process | 26 PASS |
| Cloud orchestration | Final expected-input/custody fixes | 16 PASS |
| Real numeric grammar | NeMo 1.2.0, Pynini 2.1.6.post1, Jiwer 4.0.0, isolated Python 3.12 | 6 wrapper tests and 1 real RU/EN integration PASS |
| Frozen full-chain capture | Original HEAD `9291e8b`; fork source plus helper hashes; 24 frozen inputs; pinned eval cache | Original 24/24, fork 72/72; 96 verified WAVs; both terminal cleanups complete |
| Independent output diagnostic | New capture artifacts, recorded ASR model IDs and dated OpenAI judge, two ASR/two semantic judges | 24/24 COMPLETED; controls PASS; diagnostic acceptance FAIL |
| Aggregate source gate | Updated exact collection manifest and pinned tools | UI/Rust/lint/1153 pytest/335 unittest/SCA/schema PASS; publication packaging correction PASS; exact-source aggregate must produce a terminal receipt before commit |

The artifact root is private and outside Git: `automatic-chain-capture-20261001-uYFSDG`
under the existing product-evidence directory. Outputs remain `release=false`.

The original capture receipt SHA-256 is
`0872b80b973da472d50d58f06ba763f25bc6a69195dd31a93864c4c217083cd5`;
the three-arm candidate receipt SHA-256 is
`027cb8b96b162c12ebf6cafffec20aa4201e9b494b3c35070e4f7d99b163c70d`.
Capture used private offline caches, no audio device, a 9 GB memory ceiling,
1 GB swap ceiling and two CPU cores. Cloud evaluation has a 5 GB memory
ceiling, 512 MB swap ceiling and two-core CPU quota. The evaluator environment
is separate from both the production sidecar and Qanola. Credentials are read
only by the existing provider loader and never copied into receipts or Git.

The generated provider PCM is not physical playback or first-audible evidence.
The 24 inputs are a development screen, not the independent release holdout.

## Actual output result

The evaluator receipt SHA-256 is
`916ed05d7f2e41305dc040d18da106d7bdf79228164c14ee3377768831bf5345`.
All 192 output-ASR and 48 case-judge observations completed; both silence and
both semantic-corruption controls passed. No failed attempt was retried out
of the receipt. The successful command exit means measurement completed,
not that a product-quality gate passed: `diagnostic_accepted=false` and
`release=false`.

| Chain | Both judges accept source-reference to MT text | Both judges accept both output-ASR full chains | Both judges accept both output-ASR TTS layers | First provider PCM median / p90, ms |
| --- | --- | --- | --- | --- |
| Untouched main Small/NLLB/Piper | 5/24 | 5/24 | 18/24 | 7853 / 11500 |
| Fork Small/NLLB/Piper ablation | 8/24 | 8/24 | 20/24 | 689 / 1817 |
| Fork Turbo/NLLB/Piper | 9/24 | 8/24 | 18/24 | 685 / 1057 |
| Fork Turbo/Hy/Piper | 10/24 | 8/24 | 19/24 | 900 / 1193 |

These are strict development judgments against written references, not
population accuracy or independently verified acoustic truth. Disagreement
or uncertainty is not counted as unanimous PASS. All 24 capture attempts
completed in each arm under their development timeout. The serial arm order,
different runtime implementations and absence of physical playback limit
timing attribution; the fast fork-Small ablation precludes crediting the entire
timing difference to Turbo. No EVAL-1 or EVAL-2 comparison is claimed.

The first judgment column includes ASR losses: it compares the written source
reference to the final translation, not actual ASR text to translation. An
independent attribution check found unchanged ASR wording in 22 of 24
original/fork-Small pairs; all three additional unanimous textual passes were
in that unchanged-ASR subset. Some downstream improvement is therefore
observed, but the entire first-column deficit cannot be attributed to MT.
The next private MT comparison must use identical actual-ASR input strings.

Normalized TTS proxy WER across the two independent recognizers was:

| Chain | RU input to EN voice | EN input to RU voice |
| --- | --- | --- |
| Untouched main | 1.56–3.52% | 2.59–5.17% |
| Fork Small | 1.60–4.17% | 3.12–3.91% |
| Turbo/NLLB | 0.97–1.61% | 2.40–3.20% |
| Turbo/Hy | 4.84–6.13% | 2.26–4.51% |

Low aggregate WER does not erase critical errors. Both ASRs and judges
flagged a foreign-name rendering in the Hy output and a joined-word
corruption in the NLLB output. Other mismatches may originate in the output
recognizer rather than Piper. The dominant measured problem is upstream
meaning preservation, not general voice intelligibility. Hy has no demonstrated
full-chain advantage over Turbo/NLLB on this screen. Keep it as an opt-in
challenger; no default promotion or Piper-winner claim follows from this run.

## Reproducible validation prerequisites

The first aggregate stopped at seven publication tests because Gitleaks was
not in the launch environment. A focused 59-test run reproduced the same
seven missing-tool failures; the same suite passed after configuring the
existing pinned Gitleaks executable. No source, assertion or scanner policy
was changed. The full gate must supply the existing pinned tool paths through
`TRANSLATOR_GITLEAKS_BIN`, `TRANSLATOR_CARGO_AUDIT`,
`TRANSLATOR_OSV_SCANNER` and `TRANSLATOR_BUN`. Expected versions are
Gitleaks 8.30.0, cargo-audit 0.22.2, OSV-Scanner 2.5.0 and Bun 1.3.12;
their executable hashes are checked by the existing gates.

With those tools supplied, the complete aggregate passed the Rust/UI/Python,
shell/systemd, SCA and schema gates, then failed at publication. The direct
publication check identified an executable-policy mismatch for the evaluator
requirements file and an incomplete tracked-file inventory. The unchanged
requirements moved to `tests/requirements-automatic-audio-eval.txt`, and the
sorted inventory now matches the exact index. Candidate publication and
test-manifest checks both pass; scanner and publication policy are unchanged.
A later aggregate passed those same test/SCA/schema gates but rejected the
parent's concurrent documentation/index update as a source-freeze violation.
Its exit remains failure. The final aggregate must run with all tracked files
and the index frozen through its terminal result, within 6 GB RAM, 512 MB swap
and two CPU cores. Terminal evidence is written outside Git so recording it
cannot itself change the measured candidate. Earlier failures are retained,
not rewritten as successful runs.

Fresh read-only audio inventory found USB selected as the system default and
no source-output client; the nondefault built-in speaker/internal-microphone
pair and both analog PCM handles were idle. This is a possible isolation path,
not a physical AEC measurement or permission to change the default graph.

Current public/local corpus research has not established EVAL-0 speaker,
gender, accent and per-bucket coverage. Tatoeba offers human bilingual text and
audio authors/license metadata but does not establish all required attributes.
Current development sources cannot be relabelled as an independent holdout.
Four authenticated MDC lease probes returned access for the existing SPS5 RU
and EN datasets, but HTTP 403 for Scripted 27 RU and EN
(`terms_or_access_not_approved`). No new terms were accepted and no audio was
downloaded. After all prior source/speaker exclusions, the accepted RU
archive yields only four quality-eligible fresh recordings from two speakers,
without the required gender/accent coverage. Accepted EN untouched train has
876 eligible recordings from 169 speakers, not a publisher test split.
EVAL-0 does not require human-authored target translations: independent frozen
external references may retain their provenance and uncertainty, but must
not be labelled human gold. Missing RU coverage remains the blocker.

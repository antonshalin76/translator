# Paired saved-audio product diagnostic — 2026-09-27

This fork-only diagnostic feeds the same 24 frozen MDC test WAVs (12 Russian,
12 English) into the real `LocalProvider` twice: Turbo → NLLB → Piper and
Turbo → Hy-MT2 → Piper. It selects the two verified female Piper voices,
`QUALITY_FIRST`, 16-kHz mono input, and 24-kHz mono output. The
[runner](../../scripts/translator_product_audio_pair.py) checks the corpus,
screen, Turbo report, selected WAV bytes and product manifest before starting
models. It records private `0600` JSONL attempts outside Git, including
failures and `NOT_RUN` cases, without printing speech text.

The frozen input identities are manifest SHA-256
`bc2d204c31dbe0cba78187975f02a8a4378cc09407573ca4912ba540bf44ca26`,
screen SHA-256
`005db22422ad98c9c70ecd3d01f578be571ea4bebca81398d5fff7cda86214ae`,
Turbo report SHA-256
`03ebba9576e14e2fd6adb909f25f806583845a185a7df941d0c64b3a6297ac91`,
and current product model manifest SHA-256
`d8f73beb4e9bc2b403405e4b54b18373386ecf05de420702cb7cc9b391c758dc`.
Model files came from the separate evaluation cache, not production. Each
receipt also binds the runner, relevant runtime files, model manifest,
`llama-server` binary, run order and process resources. These 24 origins were
selected before this full-chain result, but after earlier ASR diagnostics;
they are **not an independent release holdout**.
The screen contains only two name, three number and eight negation-candidate
labels, all on clean audio; it does not establish critical-case coverage under
noise or across speakers and domains.

## Observed runs

| Private receipt | SHA-256 | Result |
| --- | --- | --- |
| `full-nllb-hy.jsonl` | `ca6bff455834d8916a6cf2b334579c0e48bcd3b40544f80670b1dc45c2b27ff6` | 48/48 completed, 24/24 pairs |
| `full-hy-nllb.jsonl` | `cf909e37a3749c6b1feadb86dfb69bc11e5f2ed3b46a670edb4134b8d2159d05` | Hy failed on `en-93294` after 22 completions; NLLB `NOT_RUN` |
| `full-hy-nllb-v2.jsonl` | `ca1f020ae5e0f54a1098f0a6901c3f99b0b4503558ea1c5646c3b8af09a23d97` | 48/48 completed, 24/24 pairs |
| `full-hy-nllb-v3.jsonl` | `a2cb0bd632065d8f5a3a09320af128d1a4e3b5d24311bf0f04e818dbaa980a86` | 48/48 completed, 24/24 pairs |

The failed run is retained. Its journal recorded a `ValueError`, but the
original version did not retain a safe event-level failure category, so the
cause cannot be assigned to Hy inference, provider publication, TTS or the
diagnostic validator. The Hy child was still alive at that attempt and was
cleaned up afterward. A separate one-case replay passed, which does not
invalidate the failure. Later runner versions record a safe error category
and observed terminal outcomes if it recurs. No further repeated full runs
will be used to manufacture a green reliability claim (`pc_e78330014d69`).

The three complete runs, with opposite arm orders, observed these medians
from **accelerated saved-PCM submission to first provider PCM frame**:

| Source | NLLB range across runs | Hy range across runs |
| --- | ---: | ---: |
| Russian, 12 cases | 452–494 ms | 580–604 ms |
| English, 12 cases | 318–331 ms | 386–401 ms |

This timing excludes natural speaking duration, microphone capture, endpoint
detection, graph routing and playback. It is not Task 7 first audible. The
current product NLLB path used CUDA, unlike the earlier CPU text-only
comparison; that earlier speed advantage cannot be carried over to this
chain. Peak sampled GPU memory was about 3,254 MiB for the NLLB runner;
with Hy the runner used about 2,318 MiB and its exact child PID about
1,478 MiB. The child exited and GPU use returned to the pre-run level after
each process. These are process observations, not a release resource budget.

## Quality and decision

An independent text review of the first complete
receipt compared each translation with **that arm's actual ASR text**, not
with unheard audio. NLLB: 19 PASS, 5 FAIL, 0 UNCERTAIN. Hy: 22 PASS, 1 FAIL,
1 UNCERTAIN. On 23 cases with the same ASR text, Hy preserved more critical
facts in four, NLLB in none; both failed one. This favors Hy for MT fidelity
on this selected screen, but is not a population estimate or acoustic truth.
Hy's critical failure is `ru-81073`; it still changes a predicate and omits a
fragment. ASR errors affecting both chains include `en-74082`, `ru-71378`,
`ru-71963` and `ru-81073`.

All complete runs flag `en-78643` as ASR-divergent between arms. Across runs,
Turbo also changed how its numerical phrase was segmented on the identical
WAV; the interpreted magnitude can change (`pc_77b677d5d7f0`). Exclude it
from pure MT wins. Another Hy translation varied on `ru-71573`, but the
review found no lost critical fact. Synthesized PCM hashes varied between
runs, as already observed in the separate Piper probe; nonempty PCM is not
proof of pronunciation or intelligibility.

**Decision:** retain Turbo as the input development baseline, NLLB as the
production/default MT, Piper as the voice baseline, and Hy as an opt-in
quality challenger in the fork. A small additional PCM delay would be
acceptable for Hy's factual gains, but the unexplained failed attempt and
critical ASR/MT cases prevent default activation. Do not merge or release.
Next, isolate the failure category without repeating broad benchmarks, test
critical numerical ASR stability, then obtain a true physical first-audible
and acoustic-admission path when hardware allows. No English human listening
or physical microphone/headphone result is claimed here.

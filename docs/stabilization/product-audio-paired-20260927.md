# Paired saved-audio product diagnostic — 2026-09-27

The historical v1 fork-only diagnostic fed the same 24 frozen MDC test WAVs
(12 Russian, 12 English) into the real `LocalProvider` twice: Turbo → NLLB →
Piper and Turbo → Hy-MT2 → Piper. The current v2 runner also names a Small →
NLLB → Piper arm, executed on candidate code as a same-code ablation; it is
not an untouched `main` baseline receipt. One guarded v2 RU saved-audio smoke
completed all three arms on fork SHA `13557cbf6875`; this verifies execution
and cleanup, not comparative quality or reliability. The original paired runs
selected the two verified female Piper voices and `QUALITY_FIRST`; the runner now explicitly selects
male or female voices and any of the three modes. Input is 16-kHz mono and
output is 24-kHz mono. The
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

The runner now accepts an explicit `--mode` (`quality_first`, `balanced`, or
`streaming_first`) and binds it to each session, input frame and journal
record. All historical receipts below remain `quality_first`; selecting
`streaming_first` on a complete saved WAV tests provider policy, not streaming
capture or audible output. Hy-MT and Piper currently ignore this mode argument,
while Turbo ASR and NLLB use it. New mode smoke evidence is separate.

A bounded offline smoke completed 4/4 attempts in each mode on one RU and
one EN WAV (two MT arms each).
A later full `BALANCED` run completed 48/48 attempts and 24/24 pairs.
The full `STREAMING_FIRST` run failed on its first EN attempt after 12
completed RU attempts; the remaining 35 attempts were `NOT_RUN`.
This is a retained product-path failure, not a successful all-mode gate.
The runner now retains at most one matching, sanitized `ProviderLatency`
record on a failed attempt to distinguish request stages without recording
speech, raw errors, or audio. The daemon's Task7 bridge also carries the
existing optional ASR, MT, TTS, and total millisecond milestones through its
runtime observer without adding speech content; zero remains distinct from an
absent value. These are diagnostic fields, not a fix for the failed attempt.
The failed run predates this evidence and its cause is still unknown. No
physical microphone, speaker, or headphone was used.

## Observed runs

| Private receipt | SHA-256 | Result |
| --- | --- | --- |
| `full-nllb-hy.jsonl` | `ca6bff455834d8916a6cf2b334579c0e48bcd3b40544f80670b1dc45c2b27ff6` | 48/48 completed, 24/24 pairs |
| `full-hy-nllb.jsonl` | `cf909e37a3749c6b1feadb86dfb69bc11e5f2ed3b46a670edb4134b8d2159d05` | Hy failed on `en-93294` after 22 completions; NLLB `NOT_RUN` |
| `full-hy-nllb-v2.jsonl` | `ca1f020ae5e0f54a1098f0a6901c3f99b0b4503558ea1c5646c3b8af09a23d97` | 48/48 completed, 24/24 pairs |
| `full-hy-nllb-v3.jsonl` | `a2cb0bd632065d8f5a3a09320af128d1a4e3b5d24311bf0f04e818dbaa980a86` | 48/48 completed, 24/24 pairs |
| `smoke-quality-first.jsonl` | `5c3c9bafc9ba54e0207eda3ac89bff773f68ed43dd2f74c688dc81a79483ddc4` | 4/4 completed, two pairs |
| `smoke-balanced.jsonl` | `2b7672fbbe48bc46873fd443bf3298602bef32479863f8b1914aa59233d72bc5` | 4/4 completed, two pairs |
| `smoke-streaming-first.jsonl` | `cdab0e6de0e53fec7170c1689bf29942dc66fc54bb434d12f807fc5b41bec62b` | 4/4 completed, two pairs |
| `full-balanced.jsonl` | `e573c01f18d6e9e9ca393a7d789e1b696c81dd2cbcb280ba2bd72dff260ed5f6` | 48/48 completed, 24/24 pairs |
| `full-streaming-first.jsonl` | `e62a2354db27005ad857fe1e4b32673572c2e000f4929863831347a0b51d7ff8` | 13/48 attempted; 12 completed, first EN case `en-78643` failed, 35 `NOT_RUN` |
| `replay-streaming-first-en-78643.jsonl` | `77efb12cbc8314624ef2e8ad43bca6d15e3d4753feef2c23a48d564d24c46792` | One frozen-case replay: Hy and NLLB completed; original failure remains unresolved |
| `three-arm-smoke-20261001-nCmtKs/ru-71601.jsonl` | `36812c7a4483a0fb72b15c9aea684e45ed2fee0d701e9f1bb56df279d4efbd4b` | v2, fork SHA `13557cbf6875`: 3/3 complete, one RU comparison; no original-`main` or quality verdict |

A single bounded Hy-only replay then used the exact original sequence of 12
Russian WAVs followed by `en-78643`, one provider and no retry. All 13 attempts
completed; the final English case produced its first provider PCM at 508 ms and
the Hy child was gone after shutdown. The private
`hy-original-prefix-20260927.jsonl` receipt has SHA-256
`cb3474edb98495811886a9dbc114992f7b971b9659ed7d930368ec564038dd70`.
Its short one-off driver was SHA-256
`c75028b827c063f8293592116018b704d54ddea64c9533cac2f71ba57e28c4f9`.
This non-reproduction does not erase the prior failed attempt or estimate a
drop rate. There will be no further blind repeat of this sequence.

The pinned Dmitri and Ryan male voice files were already present in the
isolated evaluation cache and matched all four manifest file hashes. No model
was downloaded or changed in production. With explicit `--voice-gender male`,
the real Turbo → MT → Piper saved-audio chain completed the same two-case
RU/EN smoke in all three modes, with NLLB and Hy run serially (4/4 attempts
per mode, 12/12 total). Each arm released its Hy child where applicable. The
three private receipts are:

| Male-voice smoke | SHA-256 | Result |
| --- | --- | --- |
| `smoke-male-quality-first.jsonl` | `a4a50849eff3914db5d249f2a7b55e2e6bb6d64437a0ab170700f5cc927f8f8d` | 4/4 completed |
| `smoke-male-balanced.jsonl` | `cfb9b82a4b1b071c3b795cb1964a9936bf1d2ac7f37201feb07226f21303bded` | 4/4 completed |
| `smoke-male-streaming-first.jsonl` | `f89d887a0f92514dd9054cd880ae6fb1bc7c2f78da7ce9cae03ef34bdd7f567a` | 4/4 completed |

The male-smoke runner SHA-256 is
`cd969c91b523d41a6497aabbe90276f53b3f7880df5af2f158e37ed386f4e996`;
all three receipts bind the same pinned product manifest and original 24-case
screen. The first-provider-PCM range was 319–1176 ms across these twelve
attempts. This is accelerated saved audio with no playback or independent
transcription of the synthesized PCM. It proves neither the 1-second
physical `STREAMING_FIRST` target nor audible semantic quality; the full
male-voice cells and installed production cache remain unvalidated.

The streaming-first failed attempt retained safe code
`provider_unavailable`, terminal outcome `dropped`, a live Hy child, and
no stage latency. The separate replay on exact fork commit `848978b`
completed both arms and released GPU resources. A passing replay is not a
reliability estimate and does not erase the failed full run. Kernel logs
contain NVIDIA `NV_ERR_NO_MEMORY` during the failed run, but also during
the successful balanced run; temporal association is not causal attribution.
No broad rerun is justified until failure stages and host resource pressure
can be distinguished.

The earlier reverse-order quality-first failed run is also retained. Its
journal recorded a `ValueError`, but that runner version did not retain a
safe event-level failure category, so its cause cannot be assigned to Hy inference, provider publication, TTS or the
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

### Numerical ASR follow-up

The installed faster-whisper 1.2.1 decoder uses a temperature fallback
sequence when a decoded segment fails its quality thresholds. The current
Turbo adapter leaves that library default intact. On the exact frozen
`en-78643` WAV, 20 direct default decodes produced 18 distinct transcript
hashes; an independent 12-decode check produced 10, with observed fallback
temperatures of 0.2 and 0.4. This is a concrete repeatability risk for a
critical numerical utterance, not evidence that either transcript is true.

Twelve direct decodes with `temperature=0.0` were identical, but this is not
a viable quality fix: in a frozen 24-origin comparison it changed only this
English case and produced a repeated-word hallucination. Its WER against the
*unverified written reference* rose from 0.9 for the frozen Turbo output to
7.6 for the temperature-zero output; mean English WER rose from 0.245 to
0.804. The private `asr-temp0-24.json` receipt has SHA-256
`4ca08d03a9a1dc9e9369423b7b6c3df4116841da23a2a0c9c910f2f9a39ad7cb`.
Setting the CTranslate2 random seed before each default decode also did not
stabilize this case (11 distinct outputs in 12 attempts). Neither setting
was applied to the product. The next input-side candidate needs a
predeclared critical-case and independent-speech evaluation; optimizing one
ambiguous utterance would overfit this diagnostic screen.

A metadata-only inventory of the available MDC SPS5 archives also ruled out
using their remaining official-test rows as a new **speaker-disjoint** release
holdout. After the existing duration/text/quality filters and exclusion of
all origins and speakers already exposed by both corpus versions' dev/test
manifests, the remaining official-test count is 0 Russian and 2 English.
Excluding only origins leaves 165 Russian and 138 English eligible rows, but
their speakers overlap earlier evaluations. No audio or model was run for
this inventory. An audio source outside these MDC archives is required for
the independent release holdout; re-labelling these rows would not satisfy
EVAL-0.

Subsequent fork-only NLLB postprocessor fixes restrict 24-hour time and
EN-to-RU document-name correction to sentences with one corresponding numeric
time or document label in each language. The old first-match rewrites could
assign 13:45 to the wrong event or replace the first document name with the
second. Six time and three document-name regressions were RED before the fixes;
all 17 direct cases and the affected MT/provider tests pass afterward.
Ambiguous sentences retain the model output unchanged. This prevents those
code-introduced fact swaps; it does not validate the model's translation. The
saved-audio receipts above were not rerun or re-scored after these changes.

**Decision:** retain Turbo as the input development baseline, NLLB as the
production/default MT, Piper as the voice baseline, and Hy as an opt-in
quality challenger in the fork. A small additional PCM delay would be
acceptable for Hy's factual gains, but the unexplained failed attempt and
critical ASR/MT cases prevent default activation. Do not merge or release.
Next, isolate the failure category without repeating broad benchmarks, test
critical numerical ASR stability, then obtain a true physical first-audible
and acoustic-admission path after a safe physical-audio test is authorized.
No English human listening
or physical microphone/headphone result is claimed here. The new male-voice
smokes and successful failure-prefix replay are bounded diagnostics, not
permission to activate Hy, open-speaker capture, merge, or release.

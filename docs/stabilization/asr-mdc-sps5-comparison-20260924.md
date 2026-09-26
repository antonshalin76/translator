# Independent RU/EN ASR comparison — MDC Spontaneous Speech 5.0

This is an isolated model-selection experiment, not a production model change,
full-chain E2E result, or release gate. The service and its model manifest were
not changed. Turbo remains the single-model development baseline for the next
translation/voice stage; Qwen3-ASR-1.7B remains an English-side candidate.
The selected test subset is not the 120-pair-per-direction, multi-acoustic
release holdout required by `EVAL-0` in `master-bdd.md`.

## Frozen corpus and execution

The two official [Mozilla Data Collective](https://mozilladatacollective.com/)
archives were downloaded after account-level dataset agreement and verified
against the API-provided SHA-256 checksums. Archive SHA-256: RU
`aa0910f5fef7f24afccf5bca075995f0b7a970d96da72626c80507d629d05a6f`;
EN `390b5a54c9afe0cc01da039ad206248f85682f247dd2b27d4cc0ab9a68e860b6`.
The archives, transcripts, audio, and raw outputs remain in a private local
evidence directory outside this repository; no dataset content is published.
MDC prohibits speaker re-identification and dataset re-hosting.

The corpus builder uses the publisher's speaker-disjoint `dev` and `test`
splits, clips 3–20 seconds with 3–65 transcript words and no automated
quality flags, deterministic speaker balancing, and preselected critical-case
IDs based on transcript inspection before model inference. `dev` has 51 RU
clips from 4 pseudonymous speakers and 40 EN clips from 40 speakers. `test`
has 40 RU clips from 11 speakers and 40 EN clips from 37 speakers, plus 8
deterministic 10-dB speech-shaped-noise variants per language. Dev/test
speaker-ID overlap is zero in both languages. Noise variants do not add
independent utterances. Test contains five transcript-tagged name cases and
five number cases per language; negation-candidate counts are 10 RU and 7 EN.
The reference text was not manually adjudicated against the audio.

Manifest SHA-256: dev
`81f0b7f670b13d2778c0ccb3a3dd2907f26cba42038484dd1b69c66c3c070f31`;
test `bc2d204c31dbe0cba78187975f02a8a4378cc09407573ca4912ba540bf44ca26`.
Both models completed all 91 dev and 96 test attempts with pinned local
weights, the same manifest, forced language, and one GPU process at a time.
Each process had a 10-GiB RAM cgroup cap, no swap, 300% CPU quota, and a
15-minute deadline. Raw and scored reports are retained as local evidence; score
SHA-256: dev `88a1d7182034801aa06d00637bd7d75cd449e0c8320401e7ae3ed1c702442684`;
test `4311849f66c42da70011fc01158f4b51b189cc2b005294a25f4d6ee4e221dbe0`.
The evaluator is in `scripts/translator_mdc_*.py`, with focused offline tests.
This exact rerun binds all four reports to source commit `94888e2c70aa93511aa8c18098a62f790fb994c8`,
the corpus-builder/runner hashes, and complete model-directory hashes (not
only weights). SNR estimated from the stored noise WAVs was 9.832–10.162 dB.

## Results

WER is pooled word errors divided by reference words after case/punctuation
normalization and `ё`→`е`; numeric notation is deliberately not converted.
The interval is a paired bootstrap resampling pseudonymous speakers, for
Qwen minus Turbo in percentage points. It reflects this selected corpus, not
a population-level guarantee.
The publisher split and recent release do not prove either model lacked access
to related material during training; this is a new evaluation source for this
project, not a certified model-training-disjoint benchmark.

| Split / language / condition | Clips | Turbo WER | Qwen 1.7B WER | Difference, pp (95% interval) |
| --- | ---: | ---: | ---: | ---: |
| Dev RU clean | 51 | 8.23% | 13.40% | +5.17 (+1.24, +6.40) |
| Dev EN clean | 40 | 9.72% | 9.40% | -0.32 (-2.25, +1.63) |
| Test RU clean | 40 | 10.34% | 11.86% | +1.52 (-0.18, +3.23) |
| Test EN clean | 40 | 10.02% | 6.84% | -3.18 (-6.16, -0.75) |
| Test RU noise, 10 dB | 8 | 16.44% | 20.55% | +4.11 (-4.55, +10.23) |
| Test EN noise, 10 dB | 8 | 30.12% | 9.64% | -20.48 (-38.71, -4.82) |

On the test clean name-tagged subset, Turbo/Qwen WER is 1.63%/6.50% RU
and 6.54%/7.48% EN (five clips per cell). For number-tagged clips it is
16.13%/14.52% RU and 18.39%/4.60% EN. These tiny slices are diagnostic.
The raw-WER gap includes equivalent renderings such as written numbers versus
digits or Roman numerals. One selected EN number case merges uncertain
magnitudes across a pause in both transcripts; neither system has a verified
zero-critical-error result. Excluding the five
EN clean number-tagged clips reduces Qwen's advantage to 1.48 points, with a
speaker-bootstrap interval of -4.65 to +1.00 points. This is a sensitivity
check, not a replacement score. A RU negation was omitted by Qwen on one
selected clip; both systems have unresolved entity/negation transcription
issues, so neither passes a zero-critical-error product gate.

The first retained test run used byte-identical audio and gave 22.89% Turbo
WER on EN noise versus 30.12% in the bound rerun; Qwen remained at 9.64%.
Only one Turbo noisy transcript changed, on a disfluent number case. It
switched between written-number and digit forms and may change the magnitude
interpretation. The eight-clip EN-noise estimate is therefore sensitive to
decoder output variation and should not be treated as a stable regression
size or a product admission result. Clean pooled WER reproduced exactly.

## Targeted semantic-critical replay

After the paired comparison, the isolated Qwen runtime was restored for one
offline replay of the previously identified FLEURS EN `en_us-1.wav` failure.
The WAV SHA-256 was
`5fefdcd12d4c136762cc7841b084126fc60c6d6410f983cc9f82b6cfc45737a2`;
the Qwen model-directory SHA-256 remained
`fed91fc61c395e5cf9e851742942c13e776df1fd1afcdc42a1e4a93a93cc7a8a`.
The eval-only runtime imported `qwen-asr==0.0.6`, `torch==2.9.1+cu130`, and
`transformers==4.57.3` on the RTX 4080 Laptop GPU. With offline mode, two
OpenMP threads, a four-core affinity mask, and a 150-second process deadline,
the inference completed in 2.194 seconds after model load; process elapsed
time was 17.01 seconds and peak RSS was 5,130,020 KiB. This is a single-clip
diagnostic, not a paired latency score. The inference path works, but the
isolated environment is not dependency-metadata clean: `qwen-asr==0.0.6`
declares `transformers==4.57.6`, and `gradio`, `flask`, and `pytz` are absent;
this replay retained the earlier evaluation's `transformers==4.57.3`. It must
not be copied into the service as a production environment.

The published FLEURS transcript says *Javanese* three times. Qwen rendered
all three as *Japanese*, as Turbo did in the earlier local-chain run. Qwen
therefore does not repair this observed meaning-changing input error. A
reference-text inspection of ten selected frozen MDC critical cases found a
Qwen omission of a Russian negation, a wrong English person name where Turbo
omitted the name, and a noisy Russian duration changed from ten years to
decades; these are diagnostic discrepancies, not a blinded or audio-adjudicated
critical-error rate. The private MDC audio and transcripts remain outside Git.

### 2026-09-25 reference-conditioned critical screen

An independent AI text review assessed an explicitly recorded retrospective
subset of ten clean critical-labelled test origins (five RU, five EN) and two
existing 10-dB variants. Its private per-attempt verdict matrix has SHA-256
`2067c0ef209f9f2d895f09ba0b3ac8cb20bcc1615e6852fa344d8239cc2839a2`;
it records the twelve IDs, alias policy, decisions, and matching manifest and
report hashes without raw text. Selection and review happened after model
outputs existed; no model was rerun. Against the publisher's
written reference, Turbo preserved all reviewed critical facts in 8/10 clean
cases and Qwen in 7/10. The two noise variants scored 1/2 and 0/2,
respectively. An unambiguous phonetic spelling variant of a person's name
was accepted for both models. The failures included a changed predicate under
negation (`ru-72182`, Qwen), an omitted or substituted person (`en-20216`,
both), and a changed actor (`en-70798`, both). On noisy `ru-71376`, Qwen also
changed a numeric time span. Noise variants are not independent utterances.

This is a retrospective, non-blinded **reference-conditioned diagnostic**,
not an audio-adjudicated error rate: the source recordings could not be
listened to in this review, and the written references may be wrong. It does
not resolve the existing input errors or justify an EN-specific Qwen route.
The later [blind RU listener follow-up](audio-first-critical-screen-20260925.md#2026-09-26-listener-follow-up-and-english-cpu-triage)
checks six selected RU recordings only; EN remains unreviewed by a listener,
and the clean RU predicate case is rubric-sensitive.

Median isolated inference time over the bound test attempts was 283 ms for
Turbo and 324 ms for Qwen; measured load was 3.9 versus 15.0 seconds, and
peak process RSS was 1.9 versus 5.0 GiB. These measurements exclude
endpointing, MT, TTS, audio routing, and concurrent-model VRAM. They cannot prove that both ASR
models plus MT fit the product's 10-GiB VRAM ceiling or its live latency gate.

## Decision and finite next gate

Keep Turbo as the one-model input baseline for the next MT/TTS quality and
latency comparison. It is stronger on RU in dev, at least competitive on RU
holdout, faster and lower-RAM, while Qwen's EN benefit was observed on this
selected raw-WER subset but is partly notation-sensitive and not yet proven
in a full chain. Do not repin the production manifest, add language routing,
merge, or release on
this evidence. If an English ASR upgrade is later pursued, it needs one
bounded semantic-critical adjudication and a simultaneous ASR+MT+TTS resource
and latency test on the 12-GiB GPU before any language-specific routing.

The immediate product work is now downstream: compare translation and voice
quality using the frozen Turbo input baseline, and carry the existing Task 7
first-audible latency debt forward. Neither this isolated ASR result nor the
prior FLEURS screen closes that debt.

Dataset pages: [RU](https://mozilladatacollective.com/datasets/cmu5mg3pr00simh07epeylc55),
[EN](https://mozilladatacollective.com/datasets/cmu5nqn1h00vwmi07b4dbk085).

# ASR input screen — 2026-09-24

This is a development screen, not EVAL-0/EVAL-1 or release evidence. It does not
change the production service, model manifest, or release admission. The
current Turbo evaluation baseline remains frozen while the input-side
comparison is completed; translation and voice optimization follow a
defensible ASR choice. Turbo is not declared as the deployed model: the current
`models/manifest.json` still lists `faster-whisper-small` and
`faster-whisper-large-v3` as ASR assets.

## Frozen inputs and method

- Candidate source HEAD: `dc468eab0d98752043264077f14096e83208fb2c`.
- Corpus: pinned `google/fleurs` RU and EN validation data at
  `70bb2e84b976b7e960aa89f1c648e09c59f894dd`, CC-BY-4.0. Manifest SHA-256:
  `27e48592c23395b1815b4d4837d435fcbbbcc05516cb37e066901e0bea5f1ef4`.
- 72 unique source utterances per language; six candidate buckets of 12 per
  language (`short`, `negation`, `numbers`, `names`, `long`, `general`). There are
  also 12 deterministic 10-dB speech-shaped-noise variants per language, for
  168 clips total. Variants do not add independent utterances.
- All five models used exactly the same manifest and completed 168/168 attempts.
  Decoding was pinned per runtime; model outputs and failures were retained.
  Scoring case-folds, removes punctuation, maps `ё` to `е`, and preserves names,
  numbers, and negation words. VibeVoice's observed line-leading `Speaker N:`
  structural marker alone is removed for scoring; the raw answer is retained.
- The score artifact SHA-256 is
  `4b815ca33d7025ddfd8045401af350b1408436db0baf7f8bc4e61da4ce7dc9ca`.
  The local evidence packet `translator-product-evidence-20260923` holds the
  manifest, model-specific attempt logs, scorer, and `asr-score-5of5.json`.

| ASR candidate | RU clean WER | EN clean WER | RU 10-dB WER | EN 10-dB WER |
| --- | ---: | ---: | ---: | ---: |
| Whisper large-v3-turbo / CT2 | 4.08% | 6.12% | 5.17% | 6.34% |
| Nemotron 3.5 ASR Streaming 0.6B / Q8 GGUF | 12.69% | 11.86% | 16.38% | 14.18% |
| Qwen3-ASR-0.6B | 8.90% | 5.17% | 12.93% | 6.34% |
| Qwen3-ASR-1.7B | 5.04% | 4.97% | 6.03% | 7.46% |
| VibeVoice-ASR-Streaming-1.5B | 40.73% | 16.84% | 50.43% | 24.25% |

Each clean cell has 72 utterances; each noise cell has 12. WER is the pooled
word-edit count divided by reference words, not the mean of per-utterance WER.
The per-language paired utterance bootstrap for Qwen 1.7B minus Turbo includes
zero in both clean cells: RU `+0.96` percentage point (95% interval `-0.2` to
`+2.2`); EN `-1.15` point (`-2.3` to `+0.1`). These intervals are diagnostic,
not speaker-clustered population confidence intervals.

## Candidate identities and decision

- Turbo CT2 `model.bin` SHA-256:
  `e76620f83d5f5b69efd3d87e3dc180c1bd21df9fbebacfd4335e5e1efcc018da`.
- Nemotron Q8 GGUF SHA-256:
  `3fc991d3badad7277c11030a7519832cddaf2057aafed6d4b25147e953a070b1`;
  official `nemo-speech` 0.1.0 binary SHA-256:
  `3c2edea1ffb9c548c350609b56b48e71ecac6a469285dee9e4845a2099ead98f`.
- Qwen 0.6B revision `5eb144179a02acc5e5ba31e748d22b0cf3e303b0`, weight SHA-256:
  `79d6cbd4c98c7bbffe9db2edac07f56cd6637d0d5944b27f6c2b8353840323ea`.
- Qwen 1.7B revision `7278e1e70fe206f11671096ffdd38061171dd6e5`, weight
  shard SHA-256 values:
  `a4cd1f1a04d90b757dc7f7dd26254e69a013b19e80efe590a83c6a3bde8608d6`,
  `6e0b9d9e09e2e0238e7ef3cc8a484ab387e91b90f1900bedf88bc92d7929ccfc`.
- VibeVoice revision `4262d23d8a539a6530cf64fbd0b1751ef9a30853`, official
  source `1541f590c7099820f10ea012f48d2399282df69f`. All three weight
  shard hashes and the complete local directory digest are in its raw report.

Turbo stays the **current pinned evaluation leader**, not a declared final ASR
winner or production configuration.
Nemotron and VibeVoice do not advance on this diagnostic result. Qwen 0.6B is
not a priority for the quality-first local path. The next paired product-relevant
test is Turbo versus Qwen 1.7B on independent, speaker-identified RU/EN speech.
Neither a model switch nor a larger-model admission is authorized by this screen.

## Limits and next gate

FLEURS has gender codes here but no speaker IDs, so it cannot prove four
distinct speakers or accent coverage. `names` and `numbers` are heuristic
candidate labels, not manually verified critical tokens; some numbered phrases
are not numeric facts. Nemotron's published training inventory includes FLEURS,
so this corpus is not independent for that candidate. No result here proves a
zero critical-error rate, a full-chain latency/memory budget, or translation
quality. Native directory inference timing is not directly comparable to
per-clip model timing. The isolated runs did not exercise microphone/AEC,
translation, TTS, or audible end-to-end output.

Build the next corpus from newly released, terms-compliant speech with at least
four known pseudonymous speaker IDs per language, manually checked critical
number/name/negation cases, and an untouched holdout. Mozilla Common Voice
Spontaneous Speech 5.0 (RU and EN, released 2026-09-17) is a promising small
source with validated transcripts and pseudonymous speaker IDs; its official
download requires an account, dataset-terms acceptance, and API credentials.
Do not use an unofficial re-host or treat a test split as held out if the model
trained on it. Run the paired Turbo/Qwen 1.7B quality-and-resource comparison,
then select the input model and proceed to translation/voice optimization.

Primary sources: [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B),
[Nemotron 3.5 ASR](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b),
[VibeVoice streaming](https://github.com/microsoft/VibeVoice/blob/main/docs/vibevoice-asr-streaming.md),
[Common Voice RU](https://mozilladatacollective.com/datasets/cmu5mg3pr00simh07epeylc55),
[Common Voice EN](https://mozilladatacollective.com/datasets/cmu5nqn1h00vwmi07b4dbk085).

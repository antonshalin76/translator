# AppTek English dialogue ASR diagnostic — 2026-09-27

This is a fork-only, saved-WAV comparison. It does not change the service's
input model or measure microphone, endpointing, translation, playback, or
first audible output. The [AppTek Call-Center Dialogues dataset](https://huggingface.co/datasets/apptek-com/apptek_callcenter_dialogues)
provides publisher-manual English references across 14 accent groups. These
references were not independently adjudicated against audio in this project;
the dataset is a diagnostic, not a release holdout. Its CC BY-SA 4.0 terms
also need review before any redistributed derivative. No audio or transcripts
were committed to Git.

The pinned dataset revision is
`b98967d9946f7f59f58d08624a2a00fe98fe0219`. The deterministic builder
selected two separate calls per accent: two general turns from different
speakers and three critical-label candidates (negation, number, name) from
another call. Each clip is an isolated 3–12-second annotated turn with 250 ms
padding. The selected 28 source WAVs were checked against the pinned LFS
inventory before extracting 70 clips. This is 28 calls, not 70 independent
conversations. The labels come from reference-text regexes and are not
semantic ground truth.

Private evidence under `translator-product-evidence-20260923/apptek-en-20260927`:

| Artifact | SHA-256 |
| --- | --- |
| Source inventory | `869b00a1654823360aaf51c81e3d0c876712a6d9afd9c1fdc7e912b0cc30cedd` |
| Frozen manifest | `9020ff7f8a7bb8b8d9c5a5416453d4c0f9c2bc4d99eda73ad3d1328047f23c42` |
| Turbo report | `d539191c4f50219fe2a615a1654c9c65c833164c697901ddd1c9d58a2298400f` |
| Qwen3-ASR-1.7B report | `57674373b9435ed3a015e183fb691234447924cc3218cb5c1bf1987790d6e096` |
| Reviewed score | `92029108248a022b30437fc7f5bba0537b36f5c4dc74b36575505002114a5dab` |

Both models completed 70/70 attempts on the same saved clips, sequentially
on the local GPU. Their report identities include pinned model-directory and
weight hashes. The scorer now rejects a relabelled model identity, changed
manifest rows, missing attempts and invalid elapsed values. A separate
post-run check matched all 28 manifest source IDs and digests to the pinned
inventory and verified the two-call/five-turn structure per accent. These
checks strengthen evidence binding; they do not prove the publisher's text
matches what a listener hears.

## Result and sensitivity

| Cohort | Clips | Turbo WER | Qwen WER | Turbo/Qwen median saved-WAV-to-text |
| --- | ---: | ---: | ---: | ---: |
| General | 28 | 16.07% | 13.39% | 206/309 ms |
| Critical-label | 42 | 17.80% | 11.49% | 204/329 ms |
| Number-label | 14 | 28.63% | 12.45% | — |

These primary WER values lowercase and separate punctuation but do not
convert number words into digits. As a sensitivity check, the exact same
outputs were rescored with `EnglishTextNormalizer` from
`openai-whisper==20250625` (installed with `more-itertools` in an isolated
evaluation directory, not the product environment). The general result
became **11.11% Turbo / 12.24% Qwen**; critical became **9.84% / 9.30%**;
number-label became **7.35% / 7.84%**. This is the Whisper normalizer alone,
not the publisher's entire mapping-enhanced scoring script. The ranking's
sensitivity to legitimate normalization rules prevents a WER-only decision.

An independent reference-conditioned text review of the 42 critical-label
clips found clear meaning-changing departures in 6 Turbo and 5 Qwen outputs;
Qwen was better on 3 paired clips, Turbo on 2, and 37 were tied. Both models
corrupted the same numerical code, and each had other critical errors. Four
publisher-reference/audio ambiguities were left unresolved rather than
counted as wins. This review was not English human listening or an acoustic
truth measurement.

**Decision:** keep Turbo as the single input baseline. Qwen remains an
English diagnostic candidate, not a product route. The new comparison does
not resolve the previously observed numerical instability or justify a model
switch. Product work should now concentrate on the paired ASR→MT→TTS failure,
safe acoustic admission without headphones, and Task 7 physical first-audible
latency. No release, merge or production activation follows from this test.

# RU/EN voice pilot — 2026-09-24

No production TTS change. This diagnostic compares direct synthesis, not the
daemon's audio graph, resampling, acoustic admission, or first audible sound.
The candidate files, two report scripts, and all 16 WAVs were produced in the
isolated product fork and ignored evaluation cache. No microphone, speaker,
headphone, or production model directory was written.

## Frozen setup

The [four source texts](../../scripts/translator_tts_voice_pilot.py) contain
two RU and two EN cases with negation, names, money, train/flight numbers, and
time corrections. Each text was synthesized by female and male Piper profiles
and the F1/M1 styles of [Supertonic 3](https://huggingface.co/supertone-oss-archive/supertonic-3),
for eight matched text/style cells and 16 WAVs. The model assets were downloaded
at pinned revisions into a separate eval directory. The four Piper model and
config hashes all match `models/manifest.json`. Supertonic used revision
`aafc6e32416a594460b32413efc49d7fe4ce6d46`; its vector estimator and
vocoder SHA-256 values are respectively
`883ac868ea0275ef0e991524dc64f16b3c0376efd7c320af6b53f5b780d7c61c`
and `085de76dd8e8d5836d6ca66826601f615939218f90e519f70ee8a36ed2a4c4ba`.

Both direct synthesis runs were CPU-only on two pinned cores, warmed once per
language/style. Piper used the production-pinned `piper-tts==1.4.2` in the fork
sidecar environment; Supertonic used `supertonic==1.3.1`, ONNX Runtime 1.30.0,
eight default synthesis steps, and two intra-op threads. Their separate Python
processes and model-loading schemes differ: Piper loads four voices (2,665 ms),
while Supertonic loads one shared engine (946 ms). Do not compare these cold-load
numbers as a product Start measurement.

The local raw reports are `docs/benchmarks/tts-voice-pilot-20260924/`:
Piper SHA-256 `8f2c4f85019dfbe21a594b040b84120dbb136e37e6f5807f9c09f2946c31b9ca`,
Supertonic SHA-256 `7ef74cf2e3ac40f2c02867ff8c3021553e5fd95e9f063253a0eacd100ae446d9`.
WAV hashes and per-case details are in those reports. The WAVs are kept out of
Git because they are local evaluation artifacts, not a redistributable fixture.

| Warm direct synthesis, eight clips each | Piper | Supertonic 3 |
| --- | ---: | ---: |
| Median time to first available PCM | 297 ms | 2,472 ms |
| Median full-utterance synthesis | 431 ms | 2,472 ms |
| Median real-time factor | 0.087 | 0.417 |
| Peak process RSS | 796 MiB | 512 MiB |

Piper yields sentence chunks; Supertonic's tested SDK returns only a complete
waveform, so its first-PCM and full-utterance times coincide. These are direct
library timings, not end-to-end playback latency. Piper's four loaded voices
make the RSS figures non-equivalent. All WAVs were nonempty and finite; Piper
had a tiny fraction of normalized peak samples, while Supertonic's peaks were
below 0.5. Loudness and synthesis randomness were not equalized.

## Intelligibility diagnostic and limits

The [CPU ASR proxy](../../scripts/translator_tts_asr_proxy.py) transcribes the
same WAVs with a frozen Whisper small model and then the project's pinned
Whisper large-v3-turbo on all 16 clips. Turbo's `model.bin` SHA-256 is
`e76620f83d5f5b69efd3d87e3dc180c1bd21df9fbebacfd4335e5e1efcc018da`;
the complete Turbo report SHA-256 is
`766310e0c5de61ff9b8818245c223958de0e1dc7176a47cb561a512a74d54221`.
It is a second model's reading of synthesized audio, **not** a listening test
or a certified word-error score.
Automatic punctuation, currency, and number formatting distort word error
rate; exact critical tokens were inspected in the raw transcripts instead.

The initial small-model run raised possible numeral and word-pronunciation
errors. In the full Turbo run, all eight Piper clips retained their intended
numeric values, even when transcription formatting differed. Turbo rendered
four of eight Supertonic clips with changed numeric values: the RU amount
`15 420` became `15-20` with F1 and `15-14-20` with M1; RU F1's `18:05`
became `17:05`; EN F1's `$1,540` became `1,040`. In addition, EN M1's amount
lost the dollar sign in the transcript. One Piper RU male clip was heard by
Turbo as `Вейст` rather than `Поезд`, though it preserved train number and
both times. These are concrete ASR-proxy observations, not proof of what a
human listener would hear.

The agent runtime does not accept local audio as perceptual input. Therefore
no naturalness, pronunciation, equal-loudness, or human critical-error verdict
is claimed. Do not promote Supertonic or repin Piper from this pilot. Human
blinded listening with exact numeric/negation transcription, plus real
audio-chain latency, remains the voice-selection gate. The recorded Task 7
5,968-ms first-audible debt is unchanged.

## Distribution constraint

Voice provenance needs review before any release switch. The pinned Piper
[Ryan model card](https://huggingface.co/rhasspy/piper-voices/blob/main/en/en_US/ryan/medium/MODEL_CARD)
marks its training dataset CC BY-NC-SA 4.0, as does the
[HFC female card](https://huggingface.co/rhasspy/piper-voices/blob/main/en/en_US/hfc_female/medium/MODEL_CARD).
The pinned [Irina card](https://huggingface.co/rhasspy/piper-voices/blob/main/ru/ru_RU/irina/medium/MODEL_CARD)
lists the dataset license as unknown. The
[Supertonic model](https://huggingface.co/Supertone/supertonic-3) uses
[OpenRAIL-M](https://huggingface.co/Supertone/supertonic-3/blob/main/LICENSE),
with use and redistribution conditions. Repository-level MIT labels alone do
not settle model or dataset rights. This is a licensing flag, not legal advice
or a determination of permitted use.

## Decision

Keep the current Piper product configuration unchanged. Supertonic 3 is a
research candidate with a compact shared engine, but the tested nonstreaming
CPU path is roughly eight times slower to first PCM and has unresolved
critical-number pronunciation signals. A voice change is not the next release
action. The male Piper assets used here were acquired only into the isolated
eval cache; the production cache still has only the two female profiles and
cannot satisfy the existing four-voice bootstrap. Close the translation
critical-error gate and restore a safe GPU test environment before paired
ASR→MT→TTS measurement; physical open-speaker validation still requires the
separately designed acoustic calibration path.

# Output-path model screen — 2026-09-24

Research-only shortlist. No production model, manifest, or service change.

The user-provided `speech_recognition_ru` export was inspected through
2026-09-24 01:24 GMT+08. Its recent pages `messages89.html`,
`messages90.html`, and `messages91.html` are bound by SHA-256
`d2e826eda2a544742673a756280080d49da661095ca8d1f1845b01f344ac1ea2`,
`bc30f8b1923b22c016f671a709340364564e2793a7364c9d45da16840db16eb3`,
and `6f42172dbc29835ad76781269144fd2059f87a194cc7c8f05789346a422a9ad2`.
The private export stays outside the repository. The chat is a lead source,
not a product benchmark; some linked audio was absent from the export.

| Role | Candidate | Disposition |
| --- | --- | --- |
| MT | [Hy-MT2-1.8B](https://huggingface.co/tencent/Hy-MT2-1.8B-GGUF), Apache-2.0, RU/EN | Offline paired comparison against pinned NLLB; no repin yet. |
| TTS | [Supertonic 3](https://huggingface.co/supertone-oss-archive/supertonic-3), RU/EN, 99M, CPU, five male and five female fixed styles | First Piper challenger; audit OpenRAIL-M terms and archived-maintenance risk. |
| TTS | [xVibePocketTTS](https://github.com/GenVoice/xVibePocketTTS), recent RU-only CPU streaming fine-tune with stress controls | Research-only; available model card lacks a fine-tuned-weight license and English needs another engine. |
| TTS | [Higgs TTS 3 4B](https://huggingface.co/bosonai/higgs-tts-3-4b) | Excluded: embedding in a product/service requires a separate commercial license. |
| ASR | [Russian ASR leaderboard](https://huggingface.co/spaces/ESpeech/russian_asr_leaderboard) | Diagnostic RU-only result, not a replacement for our paired RU/EN input evaluation. |

The export reports pronunciation/stress problems, clipped endings, chunk
joins, and stalls in community streaming TTS. A participant's roughly 1.5-s
Qwen3-TTS latency on another machine is not a Translator measurement. Vendor
WER and speed claims also do not establish audible product quality.

Next bounded experiment: freeze independent RU→EN and EN→RU natural text with
negation, numbers, names, and roles; compare NLLB and Hy-MT2 on identical text
for raw output, critical errors, chrF2, latency, and resources. Then compare
Piper and Supertonic 3 in all four language/gender voice cells for first PCM,
repeated-output stability, proxy ASR WER, clipped content, and blind listening.
The Task6-v4 template corpus is only a regression fixture, not an independent
release holdout. Test the winner as a simultaneous ASR+MT+TTS chain against
the 10-GiB used-VRAM ceiling and first-audible latency. Task 7's recorded
5968-ms failure remains open. No merge or release follows from this screen.

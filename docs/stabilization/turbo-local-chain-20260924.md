# Fork-only Turbo local-chain smoke — 2026-09-24

This is a development measurement on branch `codex/product-clean-20260923`
from base `69a3d7d19f693fe603a9b4d8f7cebcb7971da4d2`. The production
checkout, service configuration, and model cache were not changed. The fork
adds a pinned, opt-in `faster-whisper-large-v3-turbo` ASR entry; its default
remains `faster-whisper-small`. Qwen3-ASR-1.7B remains a retained evaluation
candidate, not a local-provider runtime choice.

The [CT2 model](https://huggingface.co/deepdml/faster-whisper-large-v3-turbo-ct2)
was pinned at revision `4df90f75321148c3a29a9e2351b7ddf8f5b115a8`;
`model.bin` SHA-256 is
`e76620f83d5f5b69efd3d87e3dc180c1bd21df9fbebacfd4335e5e1efcc018da`.
All five Turbo files were size- and SHA-256-checked through the repository
manifest. A separate 2.8-GiB evaluation cache held the exact selected model
assets; the existing large-v3 asset was not available there and was not used.

The machine had an RTX 4080 Laptop GPU (12 GiB) with driver 580.178.04.
Both runs used two CPU cores, offline mode, the same staged NLLB and Piper
assets, `quality_first`, mono 16-kHz PCM input, and female Piper output in
20-ms frames. The two FLEURS validation files were `ru_ru-1.wav` (SHA-256
`363087f90513f5484750d8076da3cf7d029065d6b09f7ea68170bcf10b487d27`)
and `en_us-1.wav` (SHA-256
`5fefdcd12d4c136762cc7841b084126fc60c6d6410f983cc9f82b6cfc45737a2`).
Each process loaded its ASR, MT, and all four TTS profiles, then processed the
RU and EN files in that order. Timings are single observations, not medians.

| ASR | Cold bootstrap | RU→EN ASR / MT / first PCM | EN→RU ASR / MT / first PCM |
| --- | ---: | ---: | ---: |
| small | 5409 ms | 213 / 69 / 365 ms | 213 / 252 / 940 ms |
| Turbo | 8975 ms | 318 / 59 / 493 ms | 249 / 266 / 1011 ms |

Both directions returned nonempty transcription, translation, and PCM, and
both providers shut down cleanly. The fork's full sidecar test suite and
scoped Ruff checks passed. Deterministic tests cover explicit Turbo selection,
CPU fallback to small, and CUDA OOM fallback during model load and inference.

These numbers start when an already captured audio file is handed to ASR;
they exclude capture, endpointing, HTTP Start, acoustic admission/AEC, playback,
and a live round trip. Two FLEURS clips cannot establish translation or
speech quality. Turbo cost 128 ms and 71 ms more to first PCM than small on
these two clips, while its cold bootstrap cost about 3.6 s more. The broader
[speaker-disjoint ASR comparison](asr-mdc-sps5-comparison-20260924.md) remains
the input-model evidence; its unresolved semantic-critical cases still bar a
final quality claim. Task 7's 5968-ms first-audible debt, physical-audio gate,
human listening, merge, and release remain open. The model loads also raised
host swap use to about 2.6 GiB, so subsequent validation should be serialized
and memory-bounded.

## Local provider session check

The opt-in Turbo chain was also exercised through `LocalProvider` session open,
100-ms `ProviderInputFrame` submission, event publication, idle, health, and
shutdown on the same two FLEURS files. The files were decoded to mono 16-kHz
S16LE before submission; all frames were submitted from an already captured
file, without real-time capture or playback. The run used the separate verified
model cache, offline mode, two CPU cores, `quality_first`, and female Piper.
Both sessions began and ended `ready`, emitted one `completed` final event,
nonempty transcript and translation, and audio frames; neither emitted a
provider error. The provider shut down cleanly.

| Input | Source frames | First audio after final input frame | Provider audio | Provider total |
| --- | ---: | ---: | ---: | ---: |
| RU→EN `ru_ru-1.wav` | 75 × 100 ms | 516 ms | 135 × 20 ms | 532 ms |
| EN→RU `en_us-1.wav` | 164 × 100 ms | 1159 ms | 951 × 20 ms | 1268 ms |

Cold provider bootstrap was 7483 ms in this single run. These timings exclude
the duration of recording the speech and do not measure HTTP Start, endpointing,
acoustic admission, AEC, hardware playback, or first audible sound. They are
not a paired baseline comparison.

There is a confirmed semantic-critical error in the EN→RU path: the FLEURS
reference says *Javanese cuisine* and *Javanese coconut sugar*, but Turbo
recognized *Japanese* in both places; NLLB then translated this as
`японская кухня` and `японский кокосовый сахар`. The earlier small and
large-v3-v1 transcripts made the same Javanese→Japanese substitution on this
clip. Thus a successful provider completion is not a quality pass. This
example remains open for input-model evaluation and critical-error scoring;
it is not a reason to patch a specific phrase into the runtime.

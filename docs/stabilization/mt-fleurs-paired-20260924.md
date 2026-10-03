# Paired FLEURS RU/EN translation diagnostic — 2026-09-24

No production model or service change. This is a text-only comparison, not
ASR+MT+TTS or a release holdout.

The existing FLEURS ASR-selection manifest at revision
`70bb2e84b976b7e960aa89f1c648e09c59f894dd` was selected by acoustic
properties before either MT output was seen. Its SHA-256 is
`27e48592c23395b1815b4d4837d435fcbbbcc05516cb37e066901e0bea5f1ef4`.
Joining clean RU and EN samples by their original parallel-sentence ID yields
all 33 available pairs, 66 directed translations. The derived local corpus SHA-256 is
`56699822c70761c75681c89d69bb3e3848cc7c3b7ca5aaf307fe76181fb1ed5a`.
The source text and raw model outputs remain under ignored `docs/benchmarks/`;
the report SHA-256 is
`357859460840a240380f476b80533d818115393271b84c235fc3bd68bc1719b4`.
FLEURS is [parallel read speech](https://huggingface.co/datasets/google/fleurs),
not spontaneous dialogue, and these public sentences may have been seen by
either model during development. One aligned pair, ID `1566`, has an extra
English sentence absent from the Russian reference.

Both models used the same input/reference pair in each direction and one
CPU-only workstation. NLLB was the manifest-pinned int8 CTranslate2 adapter
with its current `QUALITY_FIRST` behavior; Hy-MT2 was the pinned Q4_K_M GGUF
(`dc5f44fcf1fa496ee7ad725982c0c8c553a4de00259b53af84c4b89fb0c06699`)
through `llama-server --jinja`. Both were warmed per direction and then called
serially on two CPU cores. Hy-MT2 used temperature 0, top-p 0.6, top-k 20,
and repetition penalty 1.05. This deterministic temperature differs from the
vendor's recommended 0.7. Timing is request-to-text only, excluding ASR,
voice, audio graph, and first audible output.

| Direction, 33 phrases | NLLB chrF2 | Hy-MT2 chrF2 | NLLB median | Hy-MT2 median |
| --- | ---: | ---: | ---: | ---: |
| RU→EN | 56.91 | 57.97 | 1,308 ms | 2,993 ms |
| EN→RU | 49.01 | 53.24 | 1,561 ms | 3,329 ms |

Paired bootstrap resampling the 33 sentence IDs 1,000 times with Python
`random.Random(20260924)` and the 2.5/97.5 percentile order statistics gave
an interval for Hy-MT2 minus NLLB chrF2 of `[-2.23, +4.78]` RU→EN and
`[-0.84, +9.45]` EN→RU. Both include zero. These intervals describe only
this selected sample, not product-domain performance.

Before reading outputs, IDs `1514, 1516, 1520, 1536, 1546, 1564, 1566,
1572, 1573, 1586, 1609, 1629, 1650` were marked for critical-case review.
In a randomized A/B self-review of their 26 directed outputs, NLLB omitted
an entire second sentence in five inspected cases (`1516` both directions,
`1546` both directions, and `1586` EN→RU). ID `1516` lost a direct prohibition;
Hy-MT2 retained these clauses. Hy-MT2 used "casualties" where the English
reference says "death toll" on `1629` RU→EN; the Russian source is less
specific, so this is an adjudication case rather than a counted error. Both
models mistranslated the crocodile species on `1650` RU→EN. This review was
label-blinded but not independently adjudicated; it is not a certified
zero-critical-error result.

Hy-MT2 is therefore a serious quality challenger, especially for complete
multi-sentence output, but neither the small aggregate difference nor this
manual subset justifies a production switch. It costs roughly twice the CPU
text latency. Keep NLLB as incumbent, retain Hy-MT2 for a product-domain
critical-error test, and proceed to an isolated Piper-versus-Supertonic voice
comparison. Full-chain GPU validation is temporarily unavailable: the loaded
NVIDIA module is `580.173.02` while NVML userspace is `580.178.04`. Do not
alter the production driver during this evaluation. Task 7's recorded
5,968-ms first-audible failure remains open.

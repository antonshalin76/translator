# RU/EN text-translation pilot — 2026-09-24

Diagnostic only. The production checkout, service, and model manifest were not
changed. Input ASR remains frozen to Whisper large-v3-turbo for the downstream
comparison; this run did not include ASR, TTS, audio routing, or live latency.

## Frozen inputs and execution

The [12-case natural-text fixture](../../sidecar/tests/quality_corpus/mt-natural-pilot-20260924.json)
was authored before model outputs were inspected. It has six RU→EN and six
EN→RU phrases covering negation, names and roles, numbers, time changes, and
device state. Its SHA-256 is
`e4517097a2d4c538d5537e269c362bd7fdf5c9b560da37b37a1d22305a2a98ab`.
The fixture is a small diagnostic set with one human-written reference per
phrase, not an independent release holdout or a statistically powered sample.

The incumbent was the manifest-pinned NLLB-200-distilled-600M-ct2-int8 through
`NllbTranslator`, with its existing postprocessing, `QUALITY_FIRST`, and CPU
inference. The challenger was Tencent
[Hy-MT2-1.8B-GGUF Q4_K_M](https://huggingface.co/tencent/Hy-MT2-1.8B-GGUF)
through local `llama-server` with the model's Jinja chat template. The GGUF
SHA-256 was
`dc5f44fcf1fa496ee7ad725982c0c8c553a4de00259b53af84c4b89fb0c06699`;
server version was `1 (6f3a9f3de)`. Both ran on two CPU cores, without GPU
offload. Each direction was warmed before timed requests. Hy-MT2 used
temperature 0, top-p 0.6, top-k 20, and repetition penalty 1.05. Temperature
0 differs from Tencent's recommended 0.7 to make this first comparison
deterministic. The [runner](../../scripts/translator_mt_natural_pilot.py)
and raw JSON report are retained locally; the report SHA-256 is
`188583eb335ffeedb019b3320f842b5d9be21dd291f573e45c5eb8b9a3b82676`.

Ollama 0.30.7's direct GGUF import produced a malformed template and an
unrelated Chinese response on the first RU→EN smoke. Those outputs were
excluded. The bundled `llama-server --jinja` exposed the official template
and correctly translated the same smoke phrase before this paired run.

## Results

| Direction | NLLB chrF2 | Hy-MT2 chrF2 | NLLB median text latency | Hy-MT2 median text latency |
| --- | ---: | ---: | ---: | ---: |
| RU→EN, 6 phrases | 67.04 | 81.14 | 648 ms | 1,252 ms |
| EN→RU, 6 phrases | 58.76 | 61.63 | 709 ms | 1,471 ms |

These are paired text-only measurements, not first-audible latency. A single
reference can undercount valid paraphrases or alternative time notation.
Manual inspection found no obvious negation or numeric-value reversal in
these 12 outputs. Both systems mishandled participant wording on the RU
"call with Maria" case (`to Maria` in NLLB, `from Maria` in Hy-MT2). Hy-MT2
rendered the EN "review with Roman" as `с Романа`, changing Roman's role;
NLLB preserved that role but translated `review` awkwardly as `обзор`.
Hy-MT2 also produced a typo in its Russian invoice output. These examples
prevent declaring it a quality winner from the higher aggregate chrF2.

## Follow-up on natural ASR output

Ten clean, critical-label-tagged transcripts from the frozen MDC test report
were selected for a text-only diagnostic after that ASR report already existed.
The same Turbo ASR text was sent to the existing NLLB adapter and the same
Hy-MT2 GGUF through CPU-only `llama-server --jinja` with the pilot's sampling
settings. This was not blinded, the publisher's
references were not checked against audio, and no translated-reference corpus
was created. Private transcripts and outputs remain outside Git.

NLLB silently omitted later clauses on several multi-sentence Russian inputs:
one output lost the fact that somebody was training and the time of day;
another lost the speaker's lack of money and phone. It also omitted the
subject of an English algebra anecdote. Hy-MT2 retained more of those source
clauses, but changed a named television title and rendered a description of
spoiled dogs as lazy. Neither candidate has a verified zero-critical-error
result. Direct single-process median text latency on these ten selected cases
was 1,111 ms for NLLB and 3,134 ms for Hy-MT2; this is not a first-audible or
population latency estimate.

An exploratory sentence-by-sentence NLLB run recovered the omitted clauses,
but mistranslated some short utterances after losing their surrounding
context. The current adapter feeds the full utterance to NLLB once; its
decoding limit already scales with input length, so merely raising that limit
is unsupported. No sentence splitting, model routing, or runtime change was
made. A broader paired critical-error gate must compare completeness and
meaning before either change is safe.

## Decision

Keep NLLB as the current product translation baseline and Hy-MT2 as a
challenger. No repin, provider integration, merge, or release is justified by
12 hand-authored phrases. The next bounded gate is a larger independently
frozen RU/EN natural-text set with human critical-error adjudication, followed
by a paired ASR+MT+TTS resource and first-audible test if a challenger wins.
The recorded Task 7 first-audible result of 5,968 ms remains open.

The isolated test servers on ports 11577 and 11578 were stopped; no test
runner or listener remained. The GGUF remains in a separate eval cache under
`translator-eval-cache-20260924` outside the repository for reproducibility.

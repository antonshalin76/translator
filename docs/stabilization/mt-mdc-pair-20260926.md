# Translation on fixed Turbo speech — 2026-09-26

The original 12-row CPU diagnostic is retained below. The prospective 24-row
CPU/GPU update and current decision are at the end of this document.

The fork's current sentence-wise NLLB translator and Hy-MT2 were compared on
the same 12 saved Turbo transcripts from the MDC critical screen. This is a
text-only diagnostic, not an ASR, TTS, live-service, or release test. The ten
clean clips have distinct origins; the two noisy clips repeat two of them.
The selection was made after earlier ASR output existed, so these are not
independent holdout observations.

The [runner](../../scripts/translator_mt_mdc_pair.py) checked the selected
screen SHA-256 `2067c0ef209f9f2d895f09ba0b3ac8cb20bcc1615e6852fa344d8239cc2839a2`
and the bound Turbo report SHA-256
`03ebba9576e14e2fd6adb909f25f806583845a185a7df941d0c64b3a6297ac91`
before loading a model. The protected v3 run used runner SHA-256
`5cb5dee65d63b3134d826cc2276158e95f7b9787d8301fb4b26fa4ab62593902`.
The fork HEAD before this diagnostic was `a2b09cd3557b545144160af684be87b311f4d069`;
the model manifest SHA-256 was
`36398ee5dc5c4c2fadcf906b54edc7519590dccfb778d838677f46af2133f5d8`.
Hy-MT2 Q4_K_M's GGUF SHA-256 was
`dc5f44fcf1fa496ee7ad725982c0c8c553a4de00259b53af84c4b89fb0c06699`.
The private, mode-0600 v3 NLLB and Hy reports have SHA-256
`053df9f4b1bd6d52beaea74d1bb03549c4eb1b37209d9b4add111621febc0f71`
and `938b4613e88600db1f49881060c0c2444a277b785c5884341a84daeb0340669f`,
respectively. No private speech text or listener response was committed.

Both backends ran serially on two CPU cores, without GPU use. NLLB used the
fork's `QUALITY_FIRST` adapter and verified model files. Hy ran in
`llama-server` version `1 (6f3a9f3de)` with the GGUF Jinja template, 2048-token
context, temperature 0, top-p 0.6, top-k 20, repetition penalty 1.05, and
128 output tokens. The runner disabled HTTP proxies and redirects, checked the
local listener's PID, executable, model path and GGUF hash before sending
text, and rechecked that listener before each request. Each direction was
warmed before timing. All 12 Hy replies reported `finish_reason=stop`; none
reported length truncation. The v3 outputs matched both earlier runs exactly.

| Clean direction, five clips | NLLB median / maximum | Hy median / maximum |
| --- | ---: | ---: |
| RU→EN | 1.34 / 1.75 s | 4.93 / 6.32 s |
| EN→RU | 1.60 / 1.95 s | 4.20 / 10.14 s |

`/usr/bin/time` measured peak RSS of 1,978,636 KiB for the NLLB process and
2,103,600 KiB for the Hy server. The v3 Hy client briefly reached 1,142,528
KiB because its preflight hash read the whole GGUF. The current runner instead
streams the same SHA-256 verification; a focused hash check produced the same
digest with 34,956 KiB peak RSS and its four fail-closed tests passed. Current
runner SHA-256 is `59fda72c8671d0fcb6118b96837ad04049a7b28060baec0d843e3447b0a18995`;
the full inference was not rerun after that memory-only change. These timings
exclude cold model load, ASR, synthesis, audio
routing, and first audible output. The attempted user-systemd memory scope was
unavailable (`No medium found`); the runs used a wall-clock timeout and CPU
affinity but **no enforced memory cap**. Both finished without swap activity
reported by `/usr/bin/time`. The test server was stopped and port 11578 was
verified free afterward.

## Critical-fact review

An independent read-only text review compared both outputs against each saved
Turbo transcript. `PASS` means no identified loss in the selected critical
facts; it is not a claim of naturalness or of acoustic correctness.

| Clean clip | NLLB | Hy-MT2 | Decisive observation |
| --- | --- | --- | --- |
| RU 71376 | FAIL | PASS | NLLB turned "everyone has become stupid" into "head over heels". |
| RU 72182 | UNCERTAIN | PASS | NLLB made the negated lighting statement ambiguous. |
| RU 71366 | FAIL | FAIL | NLLB changed rewatching to reviewing; Hy changed the title *Doctor House* to *House of the Doctor*. |
| RU 72019 | PASS | PASS | The 5–6 o'clock range survived. |
| RU 72227 | FAIL | PASS | NLLB changed hitchhiking into "on a bus stop". |
| EN 66815 | PASS | PASS | Teacher assistance and the conditional remained. |
| EN 20216 | PASS | PASS | Both translated the Prime Minister mention present in clean Turbo text. |
| EN 20253 | PASS | PASS | One person, one vote remained. |
| EN 70813 | FAIL | FAIL | Both lost *half* in *half-siblings*; Hy also changed *spoiled* to "lazy". |
| EN 70798 | FAIL | PASS | NLLB changed the name *Common Voice* to "ordinary voice". |

On these ten clean origins, NLLB has 4 PASS, 5 FAIL, and 1 UNCERTAIN;
Hy-MT2 has 8 PASS and 2 FAIL. The noisy RU 71376 repeats NLLB FAIL / Hy PASS.
The noisy EN 20216 gives PASS / PASS **relative to Turbo text only**: Turbo
had already omitted the Prime Minister fact. Counting that row as a product
success would hide an upstream ASR error.

One Russian-speaking listener previously checked the six selected RU audio
clips, including the noisy clip. No English clip has a human transcription;
there is no English listening gate in this decision. For English, the table
asserts only translation fidelity to unverified Turbo text. It cannot settle
whether the original English speaker said the same words.

## Initial CPU-only decision (superseded by the update below)

Hy-MT2 is the stronger **text-fidelity challenger on this selected set**, but
its observed translation stage takes roughly three times as long and still
has two critical errors. Neither result establishes a population error rate.
Keep the current product model pin and Piper baseline unchanged. The next
product gate is a prospectively frozen, non-duplicated RU/EN text set with
critical-fact review, then a paired Turbo→MT→Piper first-audible measurement
on identical inputs and hardware. English acoustic truth remains explicitly
unverified; it will not be replaced by machine agreement or an invented
listening result. The existing Task 7 first-audible result of 5,968 ms is
still open. No merge, release, or production activation follows from this
diagnostic.

## Prospective clean-origin update

Before inspecting the new MT outputs, a separate private screen selected 12
clean, completed Turbo transcripts per source language from the frozen MDC
report. It excluded the ten origins used in the first comparison, sorted the
remaining eligible origins by SHA-256 of `mt-next-20260926-v1:<origin_id>`, and
took the first 12 per language. There are 24 distinct origins and no noisy
duplicates. The private mode-0600 screen SHA-256 is
`005db22422ad98c9c70ecd3d01f578be571ea4bebca81398d5fff7cda86214ae`.
Its bound Turbo report SHA-256 is
`03ebba9576e14e2fd6adb909f25f806583845a185a7df941d0c64b3a6297ac91`.
The manifest SHA-256 remained
`36398ee5dc5c4c2fadcf906b54edc7519590dccfb778d838677f46af2133f5d8`.
The final protected NLLB/GPU runs both recorded runner SHA-256
`ea32d0f80bc8e6a4c400f77c2d764a632772105c6b1b73edf42a31584c7c4fcb`.
Their private reports bind the imported MT and model-verification module hashes;
`mt.py` was
`3460325651b61e11dd7480bf04083cf21746d4bc7ffe60855f5cee71c626ece4`.
The reports' `source_head=94888e2c70aa93511aa8c18098a62f790fb994c8`
belongs to the frozen **Turbo ASR
report**, not the MT checkout. The code hashes, not that field, bind this
diagnostic runtime.

NLLB and Hy-MT2 used the same selected Turbo texts; each retained its own
pinned model and decoding settings. The final NLLB and Hy GPU reports recorded
runner and Hy-server affinity to CPU 0–1. Their private, mode-0600 report
hashes are respectively
`cd9ac14fc633db2e8f346b4da23e9c3c1203960f3ca141a9a1fd9179e9effa82`
and `da5fb41004288b66cc1bbe8fa2d7fa199c8e5a24c9b12836c82c5bcf5837f43d`.
Its GGUF hash was the same pinned
`dc5f44fcf1fa496ee7ad725982c0c8c553a4de00259b53af84c4b89fb0c06699`.
All 24 GPU Hy replies ended with `finish_reason=stop`. The previous CPU Hy
report remains an exploratory observation but lacked exact runner/command
provenance, so it is excluded from the final paired timing table.

| Warm text translation, 12 per direction | NLLB CPU median / max | Hy GPU median / max |
| --- | ---: | ---: |
| RU→EN | 1,192 / 2,224 ms | 218 / 359 ms |
| EN→RU | 620 / 1,206 ms | 116 / 334 ms |

The GPU run used the locally installed `llama-server` version `1 (6f3a9f3de)`
with the CUDA backend and `--gpu-layers 99`, one local server and one request
at a time. The hardened runner required one exact server argument vector, PID,
executable, binary SHA-256
`dbfeea380cdc1de9bbfe32399befbcd8381a3c5ac83e573d79d0fa41bdc40037`,
GGUF hash, and loopback listener; it rejects extra LoRA, control-vector, and
chat-template options. `nvidia-smi` observed that exact server PID using 1,472
MiB of GPU memory. Used GPU memory returned to its 1,115 MiB pre-run level
after exact-PID shutdown. The server's peak RSS was 1,414,028 KiB, the client
peak RSS was 36,784 KiB, and `/usr/bin/time` reported zero swaps. No new
kernel GPU Xid was found after the run. The NLLB CPU process peaked at
1,978,552 KiB. A separate sanitized, operator-observed resource receipt has
SHA-256 `7fcff3a86aa576f7088f682c710c890a34fc3ed71db44afa8378fc357866ee12`.
These resource figures have different process and device boundaries and are not
an end-to-end memory budget. Model load, ASR, TTS, playback, and cold-start
latency are excluded.

An independent read-only reviewer recorded a private, per-row verdict and
rationale receipt (SHA-256
`8bbcd64eb71f0ae45e860305f9c5707b3d7f266ad750c2c45f5b95115b369d7a`)
bound to both final report hashes. Under its **Turbo-source-text fidelity**
rubric, NLLB had 16 PASS, 7 FAIL, 1 UNCERTAIN; GPU Hy had 23 PASS, 1 FAIL,
0 UNCERTAIN. A previous informal review classified more rows as uncertain
(15/6/3 versus 20/1/3); its per-row receipt was not retained. The numerical
score is therefore reviewer-sensitive, though both readings favor Hy on this
screen. The single GPU Hy FAIL is `ru-81073`, whose Turbo source is already
garbled; Hy changes the negated ship-maneuver predicate and drops later
fragments. No English audio has a human transcript. None of these text
verdicts is acoustic truth, a population error estimate, or a pronunciation
score.

## Current decision

Hy-MT2 with GPU offload is the **development MT leader on this frozen
text-fidelity screen**: it preserved more critical facts than the current
NLLB path while warm translation was faster on this machine. It is not yet a
product model switch. Hy is not wired into `LocalProvider`, CPU and GPU
outputs are not strictly identical, GPU coexistence with Turbo and Piper has
not been measured, and the complete ASR→MT→Piper→audio path has not passed a
paired first-audible or live semantic gate. The Task 7 5,968-ms first-audible
debt therefore stays open. Preserve the production configuration and fork
boundary. Do not invent an English listening verdict or release from
component timing.

## Saved-MT-output to product Piper frame probe

The [bounded probe](../../scripts/translator_mt_piper_probe.py) replayed eight
preselected clean origins from the same 24-row screen: `ru-71601`, `ru-71597`,
`ru-71963`, `ru-71599`, `en-87466`, `en-71488`, `en-78643`, and `en-20265`.
It verified the complete NLLB and **GPU Hy** report hashes above, equal source
text for each pair, and the pinned manifest before loading voices. The
product `PiperVoiceRegistry` acquired verified Irina and HFC female ONNX/JSON
snapshots. Both voices were warmed once, then the saved NLLB and Hy outputs
were synthesized serially in NLLB→Hy order, followed by the reverse order.
No MT or ASR model ran in this process. The private mode-0600 probe report has
SHA-256 `eb1e7bc1cd8c9f892efb886d7eeaae32e4d38e5ae86edac55c0ce387c2f4db8a`;
the probe source SHA-256 is
`2338305cfc207c2016b115b296eaa0ca83caffe74a7fe178e51c0fc45319340d`.
The report also binds the product `tts.py` SHA-256
`3765772ea54b61c323b6598efae48c3af9e356a06bf03c2f4e40ae8ba8c57784`
and the verified voice ONNX/JSON hashes. It hashes and parses each saved MT
report from the same bytes before synthesis.

All 32 product `synthesize_frames` calls yielded nonempty, non-silent 24-kHz
mono s16le output; every 20-ms frame was exactly 960 bytes, and no utterance
exceeded the product 30-second bound. The longest generated speech was 11.56
seconds; peak process RSS was 730,800 KiB and `/usr/bin/time` reported no
swaps. The observed first-PCM medians across the two passes were 379 ms
(NLLB-derived text) versus 394 ms (Hy-derived text) for EN output, and 175 ms
versus 200 ms for RU output. These values start **after** a saved translation
is available; they are neither combined MT→TTS latency nor first audible
playback. Repeated synthesis of the same text produced different PCM hashes in
all 16 paired cells, so sample identity cannot be assumed.

The command requested affinity to CPU 0–1, but `/usr/bin/time` recorded 386.54
CPU-seconds in 21.52 wall-seconds (1806% CPU). Installed Piper constructs
default ONNX Runtime session options; [ONNX Runtime's thread-management
documentation](https://onnxruntime.ai/docs/performance/tune-performance/threading.html)
explains that its default intra-op pool creates workers across physical cores
with their own affinity. Thus the observed first-PCM numbers are an
**unconstrained-thread diagnostic**, not a valid two-core latency benchmark.
An explicit bounded Piper thread-pool contract and a verified resource check
are needed before comparing full-chain timing under a fixed CPU budget.

This probe confirms format compatibility for these saved translations only. It
does not adjudicate voice naturalness or pronunciation, original English
speech, physical audio, acoustic admission, or Task 7. The next useful gate is
a resource-safe paired full-chain run on identical audio with the candidate MT
properly attached and the current NLLB baseline retained; a product model
switch still requires that evidence and a safe rollback path.

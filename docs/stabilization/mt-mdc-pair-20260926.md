# Translation on fixed Turbo speech — 2026-09-26

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

## Decision

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

# Local audio-first critical screen — 2026-09-25

This is an eval-only diagnostic in the product fork. It does not change the
Translator service, ASR routing, models, production audio graph, or release
status. The private MDC audio, reference text, questions, and model answers
remain outside Git.

## Contract

[`translator_audio_adjudication.py`](../../scripts/translator_audio_adjudication.py)
checks the frozen manifest and the previously selected 12 critical cases.
Each private question encodes a hypothesis from the **written** reference,
not an audio-verified truth. The model first receives the question without
audio, then with a different same-language utterance selected because its
written reference omits the fact, then with the target WAV. The reference,
expected aliases, and Turbo/Qwen outputs are never included in the prompt.
Either `NO_AUDIO` or `NOT_STATED` is accepted as no-audio abstention; the
contrasting WAV must return `NOT_STATED`. Both audio requests must have more
input tokens than the no-audio control. Exact normalized alias agreement is
recorded as `AGREES_WITH_WRITTEN_REFERENCE`; it is not audio truth or product
PASS. Every other answer or failed control stays `UNRESOLVED`.

The runner verifies WAV bytes immediately before sending them, pins the local
GGUF audio-capable Ollama model and digest, uses loopback without environment
proxies or redirects, creates a new private `0600` JSONL attempt ledger, and
exits nonzero if any request fails. Tests cover substituted audio, silent
audio loss, controls, model identity, private paths, redirects, and errors.

## Retained evidence

- Frozen test manifest SHA-256:
  `bc2d204c31dbe0cba78187975f02a8a4378cc09407573ca4912ba540bf44ca26`.
- Existing 12-case selection SHA-256:
  `2067c0ef209f9f2d895f09ba0b3ac8cb20bcc1615e6852fa344d8239cc2839a2`.
- Private question-spec SHA-256:
  `6f97be48ee916a30acfb0dd42a68a8fc4eebec89e66c980ed11eb173d7d4edf8`.
- Local `gemma4-e4b-16k:latest` digest:
  `d63abc61f41004c33d6453c73f09b3cd82ac155c37acc52a6033c23e06662c70`.
- Final runner SHA-256:
  `db510e0b1f3e853456d4e8078de77a41c9f9f69643ad0ccd3d6cd026f44875fa`.

The pre-hardening second attempt ledger SHA-256 is
`a20b1b9a1de88def1c9cc57df7baa1e4ff99460f2da562c5a178064857a83d99`:
12/12 attempts completed, with eight diagnostic agreements and four
`UNRESOLVED`. Its runner was subsequently strengthened for send-time byte
identity, redirect rejection, and explicit error exit; this earlier receipt
is not an exact-final-source gate.

The exact-final-source ledger SHA-256 is
`72c269640d6e2f350b5e4e6d89d962ffc2f53a478122c1ee9d9cc6e34541bdb7`:
12/12 attempts were recorded; seven agreed with the written reference,
four remained unresolved by answer/control, and one was an HTTP error. The
process exited **1** with `errors=1`. The failed attempt must not be retried
away or converted into a quality result. Ollama logged HTTP 500, followed
by `llama-server` abort on a CUDA illegal instruction; the kernel recorded
NVIDIA Xid 13/43. The model was unloaded after the run. The GPU was visible
again, but this does not establish a stable inference path.

In the small initial feasibility check, the same model also changed a
Russian lighting predicate in free transcription and substituted an English
person's name. Focused questions can recover some facts, but this model is
not a certified independent oracle. Eight agreements in an earlier run are
not an 8/12 accuracy estimate; the written references have not been
audio-adjudicated. None of these results measures TTS naturalness or the
live ASR→MT→TTS product chain.

## Gate and next action

The new harness has 11 focused tests passing, Ruff check and format passing,
Python compilation passing, and `git diff --check` passing. The older MDC
evidence test did not run in the available Python environments (`numpy` or
`soundfile` missing); it is not counted as a pass. Independent architecture
review accepted the eval-only ownership and found no remaining P1/P2 source
issues before the exact-source run. The recorded CUDA failure now blocks a
claim that this local audio judge is stable.

Keep Turbo as the input baseline and leave production unchanged. Resolve the
GPU/runtime instability through a separate, bounded diagnostic before more
Gemma GPU judging. The four unresolved cases and the written references
still need blinded listening for a release-grade critical-error verdict.
Task 7 first-audible latency and full-chain quality gates remain open. No
merge, model repin, or release is authorized by this diagnostic.

## 2026-09-26 resource-safe listening handoff

The 9.6-GB local Gemma model was not rerun after the CUDA failure. At this
checkpoint, `free -h` reported 9.2 GiB available RAM and 16 GiB already used
swap, with another workstation build and a VM active. A separate CPU-only
load would have left inadequate headroom for the model and its runtime. This
is a resource stop, not evidence that CPU inference works or fails.

[`translator_blind_audio_review.py`](../../scripts/translator_blind_audio_review.py)
now builds an offline, private listening packet from the exact frozen test
manifest and critical-case screen. It verifies every WAV hash, shuffles the 12
cases, embeds only audio and empty transcription fields in the review page,
and stores written references in a different owner-only directory. The review
directory contains only the HTML page, never the answer key or model answers.
The page requires a response or explicit `UNINTELLIGIBLE` for every clip.
Exported responses must be kept private too; opening the answer key before
listening defeats blinding. The first combined-directory packet was retained
privately under a `superseded-` name and must not be handed to a listener.

The final split-directory packet is outside Git. Its review-page SHA-256 is
`edaaa138528da33615b73b2a3d9be41168e2167724d139337e019058b38db7e6`;
the separate key SHA-256 is
`6599687e670bdebdb3f7bf18982d1cc9eda9264c6423823b03ced830c748f02a`.
Both private directories are mode `0700`, their files are mode `0600`, and an
independent decode check matched all 12 embedded WAV hashes to the key with
zero written-reference leaks. A 16-MiB bounded read prevents a large WAV
from consuming unbounded memory while building the packet. The export
JavaScript passed a syntax check; actual browser playback and export have not
been exercised. The available Playwright CLI rejects local `file:` URLs, and
serving private audio over HTTP solely for this smoke test was declined.
Seventeen focused tests pass across the packet and prior audio-judge harness.
This prepares a blind listening review; it is not a human review result or a
product-quality pass.

## 2026-09-26 listener follow-up and English CPU triage

One Russian-speaking listener completed all six RU fields in the blind page
and explicitly marked the six EN fields `NOT_REVIEWED_EN`. The exported JSON
matched the packet ID and all 12 labels. A private `0600` copy has SHA-256
`0eae652ccb20df922d9053e1cabffa02895597cfde23577a62b9cea6fb6a4ba7`.
The listener reported that the Russian words were intelligible, including
the speech-shaped-noise case. This is a single-listener judgment of a selected
set, not a population accuracy estimate.

Compared against that listener's own transcript, Turbo retained the selected
critical facts in all five clean RU cases and the one noisy RU variant. Qwen
retained the selected facts in four uncontroversial clean cases but changed
the noisy clip's ten-year span to decades and its final characterization.
The fifth clean case is rubric-sensitive: Qwen changed the predicate
`включено` to `яркое` under negation while retaining the broader inference
that someone was training under the lit field at 9 p.m. An independent text
reviewer counted the broader fact as preserved; the earlier strict screen
counted the predicate substitution as a failure. Thus Qwen is 4/5 on clean
RU under the strict predicate rubric or 5/5 under the broader semantic rubric;
its noisy variant is 0/1 in either reading. The same clip's opening
interjection appears in the written reference and Turbo but not in the
listener transcript; its audibility remains unresolved. The noisy RU case
is a variant of a clean origin, not an independent utterance.

No English case received human transcription. To triangulate without another
GPU load, a pinned local `faster-whisper-small` snapshot transcribed the six
EN clips on CPU only. The transient user scope launch requested a 3-GiB
`MemoryMax`, zero `MemorySwapMax`, and 200% `CPUQuota`; the runner used two
CPU threads. These scope settings are launch evidence, not fields in the
model report.
The run completed with a 742,220-KiB peak process RSS. Its private report
SHA-256 is
`7482aee364096c6dcd8f45b69809eec2e98efea7a1af2b0edf0369c9d5ad2c59`;
the one-off runner SHA-256 is
`f8705b636cd584514d995133c29e90b9e0bb899d47fa0a713cf60ded564ea9cd`.
Small's agreement with Turbo/Qwen on simple facts is not independent audio
truth: it shares Whisper ancestry with Turbo and made a time error itself.
All three ASR outputs differ from the written reference on the actor in one
English clip, raising reference-versus-audio uncertainty; none reliably
recovered the person's name in another. These remain unresolved until an
English-capable listener or a validated independent audio oracle checks the
recordings.

Turbo remains the single input-side evaluation baseline. This new RU evidence
does not authorize production model replacement, merge, or release. The
English critical cases, full-chain audio quality, and Task 7 first-audible
latency are still open gates.

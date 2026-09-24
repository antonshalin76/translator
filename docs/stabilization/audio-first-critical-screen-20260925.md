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

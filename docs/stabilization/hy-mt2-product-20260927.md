# Hy-MT2 product adapter in the fork (2026-09-27)

Hy-MT2 Q4_K_M is now an explicit, offline MT choice for `LocalProvider`.
`TRANSLATOR_MT_MODEL_ID=hy-mt2-1.8b-gguf-q4-k-m` selects it; the default remains
NLLB. The GGUF is pinned by size and SHA-256 in the manifest and is not part
of `ModelFetcher.fetch_all()`. It must be fetched explicitly with
`ModelFetcher.fetch(model_id, file_path)` or supplied at the verified cache
path. The source is [Tencent's Apache-2.0 GGUF repository](https://huggingface.co/tencent/Hy-MT2-1.8B-GGUF/tree/main).

The adapter holds a sealed verified model snapshot, starts its own offline
`llama-server` through a private Unix socket, runs RU→EN and EN→RU bootstrap
checks, and owns process, socket and model cleanup. It requires the installed
Ollama `llama-server` and CUDA backend on this machine. The selected ASR model
still uses its own CUDA check; failure of CTranslate2's CUDA probe no longer
blocks a working Hy backend. A truncated or rejected translation drops only
that utterance; transport/process failure still makes shared MT unavailable.
The scheduler and source-commit boundary preserve this distinction without
exposing the rejected text. The 10-second inference socket timeout bounds
individual blocking operations, not the complete request wall time.

## Fixed-input evidence

The existing private prospective screen has 24 distinct, clean Turbo-source
texts (12 per direction), SHA-256
`005db22422ad98c9c70ecd3d01f578be571ea4bebca81398d5fff7cda86214ae`.
The product adapter returned 24/24 complete translations. All 24 outputs
matched the prior pinned GPU Hy diagnostic byte-for-byte in the final run
and five additional serial repetitions. Median warm MT time in that final
run was 212 ms RU→EN and 111 ms EN→RU. The private 0600 receipt is
`mt-hy-product-24-20260927-v3.json`, SHA-256
`f5014cb2d304d854cfdd420ccd9bad2c956d5b536178fb35b22e4651dd845d0e`.
Its hashes bind the screen, Turbo report, manifest, runner, adapter, model and
server binary. These timings have no causal comparison against the current
product path and exclude ASR, TTS and cold start.

One earlier 128-token product-adapter attempt failed with an incomplete
translation before writing a report. Its exact reason and origin were not
captured by the original runner; it is not counted as a pass. Five subsequent
128-token repeats succeeded, but this does not erase the failure. The adapter
now permits 256 output tokens; six serial 24-case runs then completed with
identical outputs. A longer soak is still needed to estimate reliability.

The isolated local-provider path also ran Turbo → Hy → Piper on the same two
pinned FLEURS WAVs previously used for the Turbo/NLLB check. Both utterances
completed with nonempty transcript, translation and PCM, no safe provider
error, and clean shutdown. Cold construction was 7.5 s. The RU→EN recording
yielded 3.3 s of synthesized audio and 810 ms provider work after the saved
PCM was submitted; EN→RU yielded 18.7 s and 1968 ms. Re-transcribing the
generated speech gave WER 0.00 and 0.097 against the generated text. This is
an automated pronunciation diagnostic, not an English listening result. The
exact Hy child PID held 1,472 MiB of GPU memory while loaded; memory returned
to baseline after shutdown. This shows GPU use, not an exact layer count.

The full Python sidecar test suite, the runner's focused tests, Ruff and
`git diff --check` passed on this fork. Unit tests cover missing model/runtime,
private socket and inherited FD, cold readiness, child death, request timeout,
explicit CUDA visibility, nonfatal request rejection, subsequent utterance,
and cleanup. The default NLLB bulk-fetch budget remains 759,782,786 bytes;
the optional 1,133,080,448-byte Hy download does not enter its automatic
download or free-space reservation.

One earlier full-suite run intermittently failed the existing immediate
gRPC session-reopen scenario with `protocol_error`; the isolated scenario
then passed 10/10 and a subsequent complete suite passed. Its lifecycle
cause is unresolved (`pc_624e929c5f78`) and the green rerun does not close it.

## Limits and decision

This is an **opt-in development path**, not production activation. The two
saved WAVs are not physical microphone/headphone E2E; no HTTP Start,
endpointing, AEC, playback, English human listening, or first-audible timing
was measured. `--gpu-layers 99` and driver presence request CUDA, but the
adapter does not yet independently prove the number of layers offloaded on
every start; health's CUDA label must not be cited as that proof. The isolated
run above did verify memory for its exact child PID.
Task 7's 5968 ms first-audible debt remains open. These results do not justify
merge, release, or replacing NLLB by default. The next gate is a paired,
larger saved-audio Turbo→MT→Piper comparison with critical-fact review and
process-level GPU/resource proof, followed by the physical product path when
its prerequisites exist.

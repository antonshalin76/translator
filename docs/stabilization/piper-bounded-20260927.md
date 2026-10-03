# Bounded Piper product loader (fork only, 2026-09-27)

The product voice registry now constructs Piper's CPU ONNX session with two
intra-op threads and one inter-op thread per voice. Piper 1.4.2's stock loader
creates a default ONNX session with no thread bound. The verified model lease,
voice profiles, existing test factory seam, and output framing are unchanged.

Focused `test_local_tts.py`: 56 passed, including an assertion on the actual
`SessionOptions` passed to the installed ONNX Runtime. Four verified voices
prepared from the isolated model cache in 2.44 seconds, at 370 MiB RSS. The
process had 32 threads before voice preparation and 36 after it: four new ONNX
workers, not four machine-wide pools.

The existing pinned saved-output runner completed all 32 product Piper
utterances with nonempty 24-kHz mono s16le frames, no swaps, and 726,160 KiB
peak RSS. The new report is SHA-256
`30239b87ed10cf5b5bf2c234c4cb16f68c54345a947f8321d6f641da4c1b524b`.
Its median first-PCM time was 122 ms, versus 249 ms in the earlier
unconstrained-thread run on the same saved texts; median synthesis time was
294 ms versus 412 ms. The new process used 26.08 CPU-seconds in 10.83 wall
seconds. The earlier affinity-labeled run used 386.54 CPU-seconds in 21.52
wall seconds because ONNX's default workers escaped that requested CPU budget.
These single runs show resource control and no apparent throughput regression,
not a statistically proven acoustic or latency improvement. Piper's output is
stochastic, so PCM hashes need not match.

No physical audio, ASR, live MT, English listening, or first-audible boundary
was exercised. Task7 and release gates remain open; production is untouched.

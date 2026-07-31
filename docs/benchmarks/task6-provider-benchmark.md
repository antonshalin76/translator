# Task 6 Provider Benchmark

Run timestamp: `2026-07-29T05:32:58+08:00`

Machine-readable evidence:

- `docs/benchmarks/task6-results.json`, SHA-256
  `a563b986db097e89236e3ad0b92b884bbbea56f562339d08cbba1d50cd7e98c6`;
- `docs/benchmarks/task6-critical-review.json`, SHA-256
  `f01a7181da1513a4271b3429f279a8fe7b03021f7c7f5b01ffb89e6c82164958`.

## Environment

- NVIDIA GeForce RTX 4080 Laptop GPU, 12,282 MiB VRAM.
- NVIDIA driver `580.159.03`.
- CUDA toolkit `13.0`; CTranslate2 `4.7.1`.
- Reused CUDA 12 cuBLAS and cuDNN 9 libraries documented in
  `task6-model-inventory.md`.
- 32 logical CPUs and 33,439,776,768 bytes RAM.
- Offline local models only. The benchmark performed no download.

Process CPU percentages may exceed 100 because psutil reports aggregate use
across logical CPUs. GPU memory is total device memory used, including the
desktop baseline. A 20 ms periodic sampler collected CPU, RSS, GPU and VRAM
during active ASR and simultaneous-duplex operations; explicit boundary samples
were retained as additional checkpoints.

## Voice Smoke

All four configured presets emitted non-empty PCM without downloads or
cross-gender fallback.

| Language | Gender | Frames | PCM bytes |
| --- | --- | ---: | ---: |
| Russian | Male | 12 | 38,400 |
| Russian | Female | 14 | 44,800 |
| English | Male | 6 | 19,200 |
| English | Female | 9 | 28,800 |

## ASR Candidates

Ten warmups were excluded and 100 utterances were measured per candidate. The
2.3-second English Piper fixture was identical for both candidates.

| Candidate | Cold load + inference | Warm p95 | Throughput | Peak RSS | Peak GPU | Peak VRAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `faster-whisper-small` | 675.87 ms | 92.32 ms | 29.47x realtime | 1,253,040,128 bytes | 66% | 2,296 MiB |
| `faster-whisper-large-v3` | 1,601.15 ms | 289.41 ms | 8.27x realtime | 3,950,452,736 bytes | 100% | 5,466 MiB |

`faster-whisper-small` is selected for normal residency. `large-v3` is measured
as an isolated ASR candidate, retained for the offline synthesized-output
quality pass and its own duplex candidate run, then explicitly released before
the normal small model is created.

## Quality

Corpus: `translator.quality-corpus.v4`, ID `task6-v4`, SHA-256
`5048ac82565edd3e64392f1b28ae2092fb61199c27883d856788e67da84c03eb`.

The run excluded 10 warmups and measured 100 cases per direction. chrF2 compares
NLLB output with the reviewed reference. Synthesized-output WER compares the
actual translated text passed to Piper with the alternate-ASR transcript.

| Direction | chrF2 | Synthesized WER | Automated critical violations | Drops | Quality-chain p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ru -> En | 72.87 | 5.35% | 0 | 0% | 875.19 ms |
| En -> Ru | 69.61 | 12.60% | 0 | 0% | 1,200.11 ms |

An independent critic exhaustively reviewed all 200 directional rows and 280
critical number/count/time/identifier, name identity/role and scoped-negation
judgments. The final review found zero meaning-changing failures and zero
ambiguities. The permanent ledger contains only case IDs, direction, critical
labels and verdicts; source, reference and output text remain absent.
Its canonical review-content SHA-256 matches the digest generated independently
inside the benchmark result.

Both directions pass chrF2 >=45, WER <=15%, automated and manual critical
review, and drop rate <1%.

## Simultaneous Duplex

For each ASR candidate, ten simultaneous warmup pairs were excluded, then 100
simultaneous pairs were measured through an isolated `LocalProvider`. Every
utterance used a distinct session, stream and utterance identity. Latency starts
before provider session opening and ends at the first committed
`ProviderAudioDelta`.

| Candidate | Ru -> En p95 | En -> Ru p95 | Peak CPU | Peak RSS | Peak GPU | Peak VRAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `faster-whisper-small` | 569.27 ms | 683.86 ms | 3,208.5% | 2,109,657,088 bytes | 81% | 4,420 MiB |
| `faster-whisper-large-v3` | 892.80 ms | 1,063.71 ms | 3,294.0% | 1,630,019,584 bytes | 100% | 6,052 MiB |

Both candidates pass the 10 GiB VRAM gate. The selected normal runtime remains
`faster-whisper-small`; final residency evidence reports `resident_model_id`
`small`.

## Boundary

This report characterizes the local provider. It does not include physical
capture, PipeWire queues, virtual microphone playback, physical sink playback
or audible-output detection. Task 7 must measure
`speech_onset_to_first_audible_ms` at the graph boundary before assigning
`meets_target`, `usable_degraded` or `fails_usable_limit`.

The benchmark JSON contains aggregate timings, resource samples, counts and
typed violations only. The review ledger contains no speech or translation
text. Neither artifact contains PCM.

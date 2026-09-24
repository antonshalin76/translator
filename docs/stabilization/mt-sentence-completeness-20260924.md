# Fork-only NLLB sentence-completeness candidate — 2026-09-24

The local NLLB adapter previously sent each complete ASR utterance to the
model in one request. On natural multi-sentence input, the model sometimes
returned a grammatical but incomplete translation. The fixed FLEURS review
already recorded five omitted second sentences, including a direct
prohibition; a later ten-case natural-ASR diagnostic found further lost
clauses. Increasing `max_decoding_length` is not supported by this evidence:
the adapter already scaled that limit with source-piece count, and the model
ended those translations early.

This branch now uses pinned [pySBD](https://github.com/nipunsadvilkar/pySBD)
`0.3.4` to find RU/EN sentence boundaries while verifying that their joined
text equals the normalized input. The NLLB adapter translates up to 16
nonempty sentences in one native batch, with a 512-source-piece request cap,
preserves their order, and refuses a missing or empty sentence result. Inputs
beyond the cap fail closed instead of returning a partial translation.
Single-sentence inputs continue through one model call. The change is
confined to the development fork; production models, services, and manifests
are unchanged.

## Evidence

The existing 66-direction FLEURS text corpus was byte-identical to SHA-256
`56699822c70761c75681c89d69bb3e3848cc7c3b7ca5aaf307fe76181fb1ed5a`.
Both baseline and candidate used the pinned NLLB int8 model, CPU,
and `QUALITY_FIRST`. Only 11 of 66 inputs contained more than one detected
sentence. All 66 candidate translations completed. Corpus chrF2 changed as
follows; this selected public corpus is not a product release holdout.

| Direction | Complete-utterance NLLB | Sentence-wise NLLB |
| --- | ---: | ---: |
| RU→EN, 33 inputs | 56.91 | 58.70 |
| EN→RU, 33 inputs | 49.01 | 52.35 |

In FLEURS pair `1516`, the formerly omitted prohibition about jokes was
present in both directions through the changed adapter. The other previously
omitted second sentences in pairs `1546` and `1586` were also returned by the
sentence-wise diagnostic. All 12 one-sentence natural-text pilot outputs were
byte-identical to the prior NLLB report. A native model test now asserts the
prohibition in both directions; unit tests assert ordered sentences, empty
fragment failure, bounded work, and abbreviation/time boundaries. A separate
file-driven local-provider run with the frozen Turbo ASR, NLLB, and Piper on
one private 8.82-second multi-sentence clip completed with four detected
sentences and 384 output-audio frames. Its translated text preserved both the
training fact and the 9 p.m. time that the old single-request NLLB diagnostic
had omitted. The clip and raw text remain outside Git. This was not physical
capture/playback or a paired first-audible measurement. The complete sidecar
test suite and scoped Ruff checks passed.

This is a completeness improvement, **not** a zero-critical-error claim.
Short segments sometimes lose context: a Russian social-commentary case was
translated worse, and the dinosaur-feather example retained its second
sentence but still mistranslated some terms. The ten-case natural diagnostic
was selected after ASR outputs existed and was not blinded or audio-adjudicated.
Latency observations across separate processes are not paired, and no
post-change ASR→MT→TTS first-audible measurement or physical audio test has
been made. Further natural-dialogue critical-error adjudication, product-chain
latency/resource measurement, human listening, and Task 7 acceptance remain
required before merge or release.

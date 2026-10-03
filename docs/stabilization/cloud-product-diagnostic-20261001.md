# Cloud-assisted diagnostic of the frozen Translator development set

Date: 2026-10-01. This is a bounded diagnostic, not a product e2e test or a
release gate. The fork and production model configuration were not changed.

## Scope and evidence

The runner compared the unchanged original-main Small -> NLLB -> Piper receipt
with three fork receipts on the same 24 saved RU/EN WAVs (12 per language).
The inputs were bound by SHA-256 before network requests: manifest
`bc2d204c31dbe0cba78187975f02a8a4378cc09407573ca4912ba540bf44ca26`,
screen `005db22422ad98c9c70ecd3d01f578be571ea4bebca81398d5fff7cda86214ae`,
original-main receipt `6d161fee9a1adc96776d5ed0afb3430e53d1d2af97f6c14fcbbaad05bf2c5f36`,
and fork receipt `fab049e7952d84b6bbcf7265e0cde1e41bf8cc692e3193fe9c32ae60ec5ede8e`.
Each WAV was rehashed at send time. Model IDs, device, and ready state were
checked at both receipt boundaries. The full private JSONL output is
`cloud-diagnostic-final-20261001.jsonl`, SHA-256
`8374bef88b81ceeea1b698c110e74a59452afdf4ba080cff95ff5a7068482323`,
mode 0600 under the Translator product-evidence state directory. It records
source runner SHA `51cee1311da5e4bc9861a86951d4d738d3e7addecc51b4025910c0b60e526081`.

Both silent-WAV controls passed. OpenAI `gpt-transcribe`, Google
`gemini-3.5-transcribe`, and Anthropic `claude-sonnet-5-5` completed all 24
cases with no provider errors in this final run. Google uploaded files were
deleted per case; a subsequent list found zero matching diagnostic files.
The three provider credentials were read only from the Qanola Agent `.env`;
that application, its database, and production audio were not modified.

The text judge compared each translation against both the written source and
the corresponding arm's own ASR text. These are separate questions: the
own-ASR score can miss a transcription error. Counts below are against the
written reference; the 12 critical-tagged cases overlap the 24 total.

| Saved chain | 24-case PASS / FAIL / UNCERTAIN | Critical PASS / FAIL / UNCERTAIN |
| --- | ---: | ---: |
| Original main: Small -> NLLB -> Piper | 8 / 16 / 0 | 4 / 8 / 0 |
| Fork Small -> NLLB, same-code ablation | 11 / 13 / 0 | 5 / 7 / 0 |
| Fork Turbo -> NLLB | 16 / 7 / 1 | 7 / 4 / 1 |
| Fork Turbo -> Hy | 18 / 4 / 2 | 10 / 1 / 1 |

ASR word error rate against the *written* references, RU / EN: original-main
Small 0.2145 / 0.1074; fork Small 0.2215 / 0.1544; fork Turbo 0.1280 /
0.1007-0.1074 (the tiny EN arm difference is in the saved receipts); cloud
OpenAI 0.1142 / 0.0738; cloud Google 0.1107 / 0.1074. The cloud ASR numbers
are diagnostic comparators, not measured local runtime latency, privacy, or
deployment fitness. The written references have not been checked against the
audio independently: at least `ru-71599` appears to contain a surname
disagreement, and `en-78643` contains a self-corrected number. Judge labels
also varied by one or two cases between two runs, so they are not an exact
human truth oracle.

## Direct voice pilot

The separately saved Piper and Supertonic 3 direct-synthesis pilot comprised
eight matched language/gender/text cells per backend. The private cloud-ASR
proxy receipt `cloud-tts-pilot-20261001.jsonl` (SHA-256
`410a525dcea16a37b91fd77339d03ef18d1e5763deebb7b49f7a2fbe27024ea1`)
contains 16/16 completed transcriptions and a passed silent control. Raw WER
on this *direct synthesis* was Piper RU 0.35 / EN 0.24 and Supertonic RU 0.45 /
EN 0.54. Numeric written-vs-spoken forms inflate those values; the proxy
cannot distinguish TTS errors from ASR errors or assess naturalness. It is
not the actual 24-case translated-audio chain. Piper remains the configured
voice; no voice switch is justified by this pilot alone.

## Decision boundary

Turbo remains the local input development leader. Hy has the strongest
automatic meaning score on this development set, but its four FAIL and two
UNCERTAIN cases need targeted adjudication before any default switch. The
original-main baseline is now included explicitly. This evidence supports a
focused output-translation investigation, not a release or cloud-provider
migration. The independent RU/EN speaker-disjoint holdout, physical open-
speaker AEC, audible output check, paired first-audible timing, live-app
matrix, soak, and installation/rollback gates remain open. The cloud judge
and ASR proxy cannot close them.

The final run predates a failure-path-only improvement to the runner: a
failed silent control now writes a private receipt containing any uncertain
remote cleanup identity. Focused tests cover that path; the successful
24-case result was not rerun merely to exercise an unchanged success path.

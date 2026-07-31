# Task 6 Model Inventory

Inventory timestamp: `2026-07-28T21:06:10+08:00`

No model was downloaded while preparing this inventory.

## Workstation

| Resource | Inventory result |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4080 Laptop GPU, 12,282 MiB VRAM, compute capability 8.9 |
| CPU | Intel Core i9-13900HX, 32 logical CPUs |
| RAM | 31 GiB total |
| Target filesystem | `/dev/nvme0n1p2`, mounted at `/`, ext4 |
| Target filesystem free space | 238,198,165,504 bytes |
| CTranslate2 reference runtime | 4.7.1; CUDA supports `float16`, `int8_float16`, and `int8` |

Task 6 permits at most `2,147,483,648` new model bytes and requires at least
`21,474,836,480` free bytes after the planned download.

The planned `759,782,786` bytes leave `237,438,382,718` bytes free, which is
`215,963,546,238` bytes above the floor. This snapshot is evidence only. The
downloader must run `statvfs` against
`/home/anton/Source/translator/models/cache/.staging` before every transfer,
confirm that staging and final paths share the target device, and fail closed
when either path or device cannot be verified.

## Local Search Evidence

The absence gate used these roots:

- `/home/anton/.cache/huggingface/hub`;
- `/home/anton/Source/uncle-freud-bot/.data`;
- all other readable directories below `/home/anton`.

Two unreadable PostgreSQL data directories were discovered on the first pass:

```text
/home/anton/.local/share/sgr-delivery-online-poc/data/postgres
/home/anton/scripts/sgr-delivery-online-poc/data/postgres
```

They are explicit allowed exclusions because they are database storage, not
model roots. No other error or exclusion is accepted. The signature scan:

```bash
find /home/anton -xdev \
  \( -path /home/anton/.local/share/sgr-delivery-online-poc/data/postgres \
     -o -path /home/anton/scripts/sgr-delivery-online-poc/data/postgres \) \
  -prune -o \( -type f -o -type l \) -name model.bin -print

find /home/anton -xdev \
  \( -path /home/anton/.local/share/sgr-delivery-online-poc/data/postgres \
     -o -path /home/anton/scripts/sgr-delivery-online-poc/data/postgres \) \
  -prune -o -type f -name '*.onnx.json' -print
```

The first result was filtered to directories that also contain `config.json`.
It exited `0`, wrote zero stderr lines and returned only the two Whisper CT2
ASR snapshots listed below. Their configs contain Whisper
`alignment_heads`; neither is MT. The second result was filtered to configs
with numeric `audio.sample_rate`, string `phoneme_type`, numeric `num_symbols`,
and an adjacent ONNX file. It also exited `0` with zero stderr lines and
returned only:

```text
/home/anton/Source/uncle-freud-bot/.data/piper-voices/en_US-ryan-medium.onnx
/home/anton/Source/uncle-freud-bot/.data/piper-voices/ru_RU-dmitri-medium.onnx
```

The source-checkpoint pass then inventoried every Hugging Face repository root:

```bash
find /home/anton -xdev \
  \( -path /home/anton/.local/share/sgr-delivery-online-poc/data/postgres \
     -o -path /home/anton/scripts/sgr-delivery-online-poc/data/postgres \) \
  -prune -o -type d -name 'models--*' -print
```

It exited `0` with zero stderr lines and found 19 roots, including lock
directories. Their repository IDs covered ASR, OCR, document/chart models,
Qwen LLM/TTS, ViT and DistilBERT. None had an MT repository identifier.

A repository name is not sufficient evidence, so every readable regular or
symlinked `config.json` with an adjacent standard Transformers weight file was
also checked. Standard weights included `model.safetensors`, sharded `model-*.safetensors`,
`pytorch_model.bin`, sharded `pytorch_model-*.bin`, `tf_model.h5`, and
`flax_model.msgpack`. A candidate matched when `is_encoder_decoder=true`,
`model_type` was one of `marian`, `m2m_100`, `m2m100`, `nllb`, `mbart`, `t5`,
or `mt5`, or an architecture named conditional generation/translation. This
scan exited `0`, wrote zero stderr lines and returned one broad-signature
candidate:

```text
Qwen3-TTS-12Hz-1.7B-CustomVoice
model_type=qwen3_tts
architectures=["Qwen3TTSForConditionalGeneration"]
```

It is the already inventoried TTS cache, not a text MT checkpoint, and is
excluded from the MT candidate set. No encoder-decoder or translation
model type was found.

This reproducible result permits one MT download and two missing female voice
downloads. The manifest validator remains authoritative if local state changes.

## Reused Local Models

| Role | Model and revision | Absolute runtime path | Bytes | License and integrity evidence |
| --- | --- | --- | ---: | --- |
| ASR candidate | [`Systran/faster-whisper-small@536b0662742c02347bc0e980a01041f333bce120`](https://huggingface.co/Systran/faster-whisper-small/tree/536b0662742c02347bc0e980a01041f333bce120) | `/home/anton/.cache/huggingface/hub/models--Systran--faster-whisper-small/snapshots/536b0662742c02347bc0e980a01041f333bce120` | 486,212,372 | MIT. All four runtime files are covered by the integrity table below. |
| ASR candidate | [`Systran/faster-whisper-large-v3@edaa852ec7e145841d8ffdb056a99866b5f0a478`](https://huggingface.co/Systran/faster-whisper-large-v3/tree/edaa852ec7e145841d8ffdb056a99866b5f0a478) | `/home/anton/Source/uncle-freud-bot/.data/faster-whisper/models--Systran--faster-whisper-large-v3/snapshots/edaa852ec7e145841d8ffdb056a99866b5f0a478` | 3,090,835,702 | MIT. All five runtime files are covered by the integrity table below. |
| Russian male TTS | [`rhasspy/piper-voices@0d907f.../ru_RU-dmitri-medium`](https://huggingface.co/rhasspy/piper-voices/tree/0d907f158acc877ddeebcbf827659ee13bea8bcd/ru/ru_RU/dmitri/medium) | `/home/anton/Source/uncle-freud-bot/.data/piper-voices/ru_RU-dmitri-medium.onnx` | 63,201,294 | Repository MIT; source dataset CC0; SHA-256 `f073356ebc4bd0f80c5af58df2953a5988bd5bdab1eb38635ce960b071fbefcb`. |
| Russian male TTS config | Same pinned voice | `/home/anton/Source/uncle-freud-bot/.data/piper-voices/ru_RU-dmitri-medium.onnx.json` | 4,824 | SHA-256 `667ef3117bc642c2892dff7690d8bdc8ca4228aeaa783b2dc1416df632855e0d`. |
| English male TTS | [`rhasspy/piper-voices@0d907f.../en_US-ryan-medium`](https://huggingface.co/rhasspy/piper-voices/tree/0d907f158acc877ddeebcbf827659ee13bea8bcd/en/en_US/ryan/medium) | `/home/anton/Source/uncle-freud-bot/.data/piper-voices/en_US-ryan-medium.onnx` | 63,201,294 | Repository MIT; source dataset CC-BY-NC-SA-4.0; SHA-256 `abf4c274862564ed647ba0d2c47f8ee7c9b717d27bdad9219100eb310db4047a`. |
| English male TTS config | Same pinned voice | `/home/anton/Source/uncle-freud-bot/.data/piper-voices/en_US-ryan-medium.onnx.json` | 4,883 | SHA-256 `44034c056cb15681b2ad494307c7f3f2e4499d1253c700c711fa0a4607ffe78d`. |

Reused ASR runtime integrity:

| Model | File | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| small | `config.json` | 2,370 | `b55496ac7940a7ae47d2c01eab40edfd8701feec1229d9cce3b40014383fb828` |
| small | `model.bin` | 483,546,902 | `3e305921506d8872816023e4c273e75d2419fb89b24da97b4fe7bce14170d671` |
| small | `tokenizer.json` | 2,203,239 | `fb7b63191e9bb045082c79fd742a3106a12c99513ab30df4a0d47fa6cb6fd0ab` |
| small | `vocabulary.txt` | 459,861 | `34ce3fe1c5041027b3f8d42912270993f986dbc4bb34cf27f951e34a1e453913` |
| large-v3 | `config.json` | 2,394 | `a9306624f5ec14270a014b647e5c316b6e03a662c369758d1b90697a7b0655b9` |
| large-v3 | `model.bin` | 3,087,284,237 | `69f74147e3334731bc3a76048724833325d2ec74642fb52620eda87352e3d4f1` |
| large-v3 | `preprocessor_config.json` | 340 | `7ccc62c6f2765af1f3b46c00c9b5894426835a05021c8b9c01eecb6dfb542711` |
| large-v3 | `tokenizer.json` | 2,480,617 | `6d8cbd7cd0d8d5815e478dac67b85a26bbe77c1f5e0c6d76d1ce2abc0e5f21ca` |
| large-v3 | `vocabulary.json` | 1,068,114 | `c69260f2ab26d659b7c398f9a2b2b48ed0df16c3b47d7326782fd9cba71690c1` |

The local Qwen3 TTS cache is excluded from MVP-A because its approximately
4.3 GiB residency and custom-voice scope do not fit the Piper bootstrap.

## Runtime Selection

The measured normal-residency model is
`Systran/faster-whisper-small@536b0662742c02347bc0e980a01041f333bce120`.
`large-v3` is retained on disk as the offline synthesized-output quality oracle,
not as a second normal resident model. The measured comparison and quality
results are in `task6-provider-benchmark.md` and
`task6-results.json`.

CTranslate2 `4.7.1` is built against CUDA 12 libraries. The workstation's
system CUDA 13 installation does not provide `libcublas.so.12`. No additional
runtime was downloaded: Task 6 reused CUDA 12 cuBLAS from
`/usr/local/lib/ollama/cuda_v12` and cuDNN 9 from
`/home/anton/Source/uncle-freud-bot/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib`.
The systemd user unit sets this `LD_LIBRARY_PATH`; direct benchmark runs must
set the same path. CPU fallback remains the supported path when these
compatibility libraries are absent.

## Usage Policy

The manifest-wide mode is `personal_noncommercial` with
`redistribution=false` and `certified_or_safety_critical=false`. These limits
apply to NLLB, both female voices, and the reused Ryan voice. Any commercial,
redistribution, certified or safety-critical mode fails closed. MIT/CC0 assets
remain subject to their own terms but do not weaken the manifest-wide mode.

## MT Selection

| Priority | Candidate | Runtime and disk shape | License | Decision |
| ---: | --- | --- | --- | --- |
| 1 | [`mijuanlo/nllb-200-distilled-600M-ct2-int8`](https://huggingface.co/mijuanlo/nllb-200-distilled-600M-ct2-int8) | One CTranslate2 INT8 model for both directions; 633,370,400 allowlisted bytes | CC-BY-NC-4.0 | Selected for personal, non-commercial use. One artifact covers Ru and En and fits the download budget. |
| 2 | [`facebook/m2m100_418M`](https://huggingface.co/facebook/m2m100_418M), converted locally | Requires downloading an original Transformers model and producing a second CT2 artifact | CC-BY-NC-4.0 | Rejected before download: no pinned preconverted artifact was selected and local conversion adds a second provenance surface. |
| 3 | Paired Helsinki OPUS `rus-eng` and `eng-rus` | Requires two directional model instances | Apache-2.0 | Rejected before download: it does not meet the selected one-bidirectional-model residency contract. |

Selected revision:
`16bc5ff0482f9f1c0d35bdef950721ce58640789`.
The publisher does not identify the converter version; provenance records it as
`unknown`. Upstream is `facebook/nllb-200-distilled-600M`; its exact upstream
revision is not declared by the converter publisher. The published conversion
command is
`ct2-transformers-converter --model facebook/nllb-200-distilled-600M --quantization int8`.
The artifact card declares CTranslate2 compatibility from 3.22.0.

| Allowlisted file | Bytes | SHA-256 |
| --- | ---: | --- |
| `config.json` | 1,065 | `bf8ade7c3f1683e5f13001bab18b04a1ccd1a6801208efd227ed13b2ff6f15e7` |
| `model.bin` | 622,596,105 | `398726640cc2a02cc6a35277fa3cf2159ce8a1a66b48aa1b6c8837a47e3dd00c` |
| `sentencepiece.bpe.model` | 4,852,054 | `14bb8dfb35c0ffdea7bc01e56cea38b9e3d5efcdcb9c251d6b40538e1aab555a` |
| `shared_vocabulary.json` | 5,921,176 | `af53bfd0e6f726209e7325e45b87ab3b14e5856f7d42d7b9be91de3287c45267` |

Absolute planned cache:
`/home/anton/Source/translator/models/cache/nllb-200-distilled-600M-ct2-int8`.
Runtime language codes are `rus_Cyrl` and `eng_Latn`.

## Female Voice Selection

Both voices come from
[`rhasspy/piper-voices`](https://huggingface.co/rhasspy/piper-voices) at
revision `0d907f158acc877ddeebcbf827659ee13bea8bcd`.

| Language | Voice | Files and SHA-256 | License decision |
| --- | --- | --- | --- |
| Russian | [`ru_RU-irina-medium`](https://huggingface.co/rhasspy/piper-voices/tree/0d907f158acc877ddeebcbf827659ee13bea8bcd/ru/ru_RU/irina/medium) | ONNX: 63,201,294 bytes, `8ff38212d23da300bbe3705c645e6e5b9475f0bfde01558eb17813e22acaaaaa`; config: 4,765 bytes, `c2ec28bb38e2b59e93b959b3e40348c1afebbd272f30fed5d41205d08e98a9d7` | Repository MIT; source dataset license is `Unknown`. Accepted only through waiver `PIPER_RU_IRINA_PERSONAL_LOCAL_V1`. |
| English | [`en_US-hfc_female-medium`](https://huggingface.co/rhasspy/piper-voices/tree/0d907f158acc877ddeebcbf827659ee13bea8bcd/en/en_US/hfc_female/medium) | ONNX: 63,201,294 bytes, `914c473788fc1fa8b63ace1cdcdb44588f4ae523d3ab37df1536616835a140b7`; config: 5,033 bytes, `03f1fa0622b80463283592d97aca9f6e89aec345a5c56b7257723e0093c58b6c` | Repository MIT; source dataset CC-BY-NC-SA-4.0. Accepted for personal, non-commercial use. |

Both voices are 22,050 Hz mono Piper medium voices. No cross-gender fallback
is permitted if either asset is unavailable.

Pinned source and target paths:

| Asset | Pinned source URL | Absolute target |
| --- | --- | --- |
| Irina ONNX | `https://huggingface.co/rhasspy/piper-voices/resolve/0d907f158acc877ddeebcbf827659ee13bea8bcd/ru/ru_RU/irina/medium/ru_RU-irina-medium.onnx` | `/home/anton/Source/translator/models/cache/piper/ru_RU-irina-medium.onnx` |
| Irina config | `https://huggingface.co/rhasspy/piper-voices/resolve/0d907f158acc877ddeebcbf827659ee13bea8bcd/ru/ru_RU/irina/medium/ru_RU-irina-medium.onnx.json` | `/home/anton/Source/translator/models/cache/piper/ru_RU-irina-medium.onnx.json` |
| HFC female ONNX | `https://huggingface.co/rhasspy/piper-voices/resolve/0d907f158acc877ddeebcbf827659ee13bea8bcd/en/en_US/hfc_female/medium/en_US-hfc_female-medium.onnx` | `/home/anton/Source/translator/models/cache/piper/en_US-hfc_female-medium.onnx` |
| HFC female config | `https://huggingface.co/rhasspy/piper-voices/resolve/0d907f158acc877ddeebcbf827659ee13bea8bcd/en/en_US/hfc_female/medium/en_US-hfc_female-medium.onnx.json` | `/home/anton/Source/translator/models/cache/piper/en_US-hfc_female-medium.onnx.json` |

Waiver `PIPER_RU_IRINA_PERSONAL_LOCAL_V1` is valid only when
`usage_mode=personal_noncommercial`, `redistribution=false`, and the exact
pinned revision and hashes above match. An unknown license without this exact
waiver is rejected. Commercial or redistribution modes always reject the
waiver.

## Download Gate

The planned download is `759,782,786` bytes (`724.59 MiB`):

- MT: 633,370,400 bytes;
- Russian female voice: 63,206,059 bytes;
- English female voice: 63,206,327 bytes.

The download remains blocked until `models/manifest.json` and its validator
tests enforce all of these conditions:

- exact pinned repository revisions and file allowlists;
- exact byte count and SHA-256 for every file;
- cumulative new-model bytes no greater than `2,147,483,648`;
- separate planned and transferred byte ledgers; transferred bytes include
  existing `.part` bytes and every retry byte;
- at least `21,474,836,480` free bytes after all remaining bytes;
- same-filesystem `.part` staging opened with exclusive creation, file and
  directory `fsync`, checksum verification, and atomic rename;
- fail-closed redirect handling: only the pinned Hugging Face repository URL
  its exact same-repository `/api/resolve-cache` path, and allowlisted
  content-delivery hosts may serve bytes. The live gate observed
  `us.aws.cdn.hf.co` for the pinned NLLB weight on 2026-07-28 and added only
  that exact hostname;
- no runtime network path and absolute local-only model paths;
- rejection of unknown sizes, checksums, licenses, or unapproved usage modes,
  except the exact typed personal-use waiver documented above.

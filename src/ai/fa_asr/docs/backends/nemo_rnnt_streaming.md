# nemo_rnnt_streaming Backend

## Backend Name

`nemo_rnnt_streaming`

## Status

この backend は、local `.nemo` file を `nemo_rnnt_streaming_worker` process で読み込み、NeMo RNNT / Transducer model の cache-aware streaming inference を行うための backend slot です。

ただし、2026-05-22 時点の実測では、手元の Parakeet 1.1B multilingual `.nemo` は、現行 FluentAudio worker policy では local streaming backend として未成立です。

確認済みの事実は次です。

- 現行 worker は `[70, 1]` を無条件に `set_default_att_context_size(...)` へ渡しません。
- 現行 worker は model が公開する supported attention contexts を検査します。
- `fluent-audio-runtime` container で `nvidia/riva/parakeet-rnnt-riva-1-1b-unified-ml-cs-universal:trainable_v1.0` 由来の `.nemo` に `health` を投げると、model restore 後に worker が `returncode=1` で fail closed しました。
- stderr の本質行は次です。

```text
model encoder does not support a finite attention context; supported_contexts=[[-1, -1]]
```

したがって、この artifact については、現時点で次を完了扱いしてはいけません。

- `health_ok`
- `start`
- `audio_accepted`
- non-empty partial transcript
- non-empty final transcript
- full ROS graph 上の ASR 成功

NIM / Riva の公式資料では Parakeet 1.1B RNNT Multilingual は streaming + offline、多言語、`ja-JP` support を持つ model family として扱われます。
しかし NIM / Riva は TensorRT、Triton、model download、optimization、serving profile、pre/post-processing を含む serving stack です。
FluentAudio は NIM / Riva server を使わず、local `.nemo` を NeMo runtime で直接 restore します。
そのため、NIM / Riva 上の streaming support は、FluentAudio local worker の成功証明ではありません。

## 根拠の分離

この document では、公式一次情報、local 実測、FluentAudio の設計判断を混ぜません。
同じ Parakeet / RNNT / streaming という語が出ていても、成立を証明している範囲が違います。

| 分類 | この document で使う根拠 | 言えること | 言えないこと |
| --- | --- | --- | --- |
| 公式一次情報 | NVIDIA ASR NIM docs、support matrix、deploy docs、NeMo docs、ConformerEncoder source docs、NGC CLI docs | NIM / Riva serving stack、model family、support matrix、cache-aware streaming concept、NGC download command の意味 | FluentAudio local `.nemo` worker の `health` / `start` / `audio` / `finish` 成功 |
| local 実測 | NGC metadata、local `.nemo` config、worker `health` 実行結果、stderr、container 内 runtime facts | 対象 artifact と現行 worker policy の実際の相性 | 別 artifact、別 NIM/Riva profile、別 future policy の成功 |
| FluentAudio 設計判断 | fail-closed policy、health-first 完了条件、ASR-ready stream contract、file-source gating | この backend を完了扱いする条件、未成立を未成立として止める理由 | NVIDIA docs の capability 表そのものの否定 |

したがって、NIM / Riva / support matrix は model selection と serving stack の根拠です。
`nemo_rnnt_streaming` backend の成立は、local `.nemo` に対する worker protocol と full ROS graph の代表検証でだけ判定します。

## Runtime Boundary

`fa_asr` node 本体は NeMo / PyTorch / Parakeet を import しません。
NeMo 依存は `nemo_rnnt_streaming_worker` process の内部へ閉じ込めます。

境界は JSON Lines protocol です。

- `fa_asr` は worker stdin へ JSON object を 1 行ずつ送る。
- worker は stdout へ JSON object を 1 行ずつ返す。
- NeMo warning、traceback、telemetry は stdout ではなく stderr に出す。
- stdout が閉じた場合、`fa_asr` は stderr の理由を backend failure として扱う。
- failure 時に Whisper、OpenAI、`local_command`、別 Parakeet backend へ fallback しない。

raw worker protocol としては、`start` command が prior `health` なしで届いた場合にも、worker は同じ `.nemo` restore / capability validation を実行してから stream を開始します。
これは transport protocol 上の許容経路であり、system としての完了条件を弱めるものではありません。
`fa_asr` backend としては、起動時 `health` によって model、sample rate、attention context、streaming params を検証し、そこで失敗するものを runtime audio 到着まで遅延させないことを基本契約にします。

worker protocol の command は `health`、`start`、`audio`、`drain`、`finish`、`cancel` です。
transport として `start` が単独で restore / validation を起こせても、system 完了条件は health-first です。
`audio_accepted` は audio chunk が active stream state に受理されたことだけを意味し、partial / final transcript の成功ではありません。
`drain` は新しい audio を追加せず、現時点の non-empty partial があれば返します。
`finish` は `final` と `finished` を返す protocol completion ですが、final text が空の場合は speech ASR 成功とは扱いません。
local backend wrapper は final text が空で、直前までに non-empty partial がある場合、その latest partial を final commit 候補として扱えます。
final も prior partial も空の場合、`fa_asr` publish 経路で final result 自体は publish され得ますが、これは transport completion であり speech ASR 成功ではありません。
代表検証の合格条件は non-empty text です。

この backend は streaming 専用です。
`transcribe()` 的な complete-file offline request はこの backend の責務ではありません。
現行 `NemoRnntStreamingAsrBackend.transcribe()` は常に失敗します。
`TranscribeAudio` service が ASR-ready rolling timeline から slice を作れても、この backend は non-streaming transcription backend としては未対応です。
full-context / offline transcribe を行う場合は別 backend policy として設計し、この streaming backend の成功条件に混ぜません。

## NIM / Riva / NeMo Local の責務差分

| 領域 | 役割 | FluentAudio での扱い |
| --- | --- | --- |
| NIM ASR | NVIDIA が配布する ASR serving container。pre-trained NeMo model と TensorRT / Triton stack を self-contained container に package し、model download、optimization、serving を扱う。streaming mode では audio 到着に応じて partial transcript を返す。 | 使わない。`nemo_rnnt_streaming` は NIM gRPC / WebSocket endpoint へ接続しない。 |
| Riva | NVIDIA の speech serving / deployment stack。model repository、Triton、serving pipeline、profile selection、pre/post-processing と結びつく。 | server としては使わない。artifact selection の参考にはするが、Riva serving 成功を FluentAudio local worker 成功とは扱わない。 |
| NGC artifact | Riva / NIM 用 model artifact を取得する配布単位。model card や version metadata を持つ。`ngc registry model download-version` は version contents を local disk に置くだけで、artifact を `.nemo` / RMIR / ONNX / TensorRT engine へ変換しない。 | checkpoint の入手元。metadata は model 選定根拠だが、runtime contract は local worker で別途検証する。 |
| local `.nemo` | NeMo / PyTorch で直接 restore できる checkpoint。 | `nemo_rnnt_streaming_worker` が直接 restore する対象。streaming params、attention context、cache state、decoder state を FluentAudio worker が明示的に扱う。 |

公式資料上の事実は次のように扱います。

- NVIDIA ASR NIM docs は、ASR NIM が pre-trained NeMo model と TensorRT / Triton inference stack を self-contained container に package し、download / optimization / serving を扱うと説明している。
- NVIDIA ASR NIM docs は、streaming inference を audio arrival に応じて partial transcripts を返す mode と説明している。
- NVIDIA ASR NIM docs / support matrix は、Parakeet 1.1B RNNT Multilingual を streaming speech-to-text transcription 対応の multilingual model として扱い、`ja-JP` を supported language に含めている。
- NVIDIA ASR NIM support matrix は、Parakeet 1.1B RNNT Multilingual に `type=default` / `type=prompt` / `type=indic` を定義している。`ja-JP` は default / prompt に含まれ、indic には含まれない。
- NVIDIA ASR NIM support matrix では、`type=default` は language を自動検出して language code を出力し、`type=prompt` は client が language code を渡す前提でより高精度、`type=indic` は Indic 言語向け最適化として扱われる。
- NVIDIA ASR NIM deploy docs は、Parakeet RNNT Multilingual の serving profile として `CONTAINER_ID=parakeet-1-1b-rnnt-multilingual` を使い、`NIM_TAGS_SELECTOR` を support matrix から選ぶ構成を説明している。
- `NIM_TAGS_SELECTOR` の例には、`mode=ofl`、`mode=str`、`mode=str-thr`、`mode=all`、`diarizer=sortformer`、`vad=silero`、`type=default` / `type=prompt` / `type=indic` がある。これらは NIM serving profile の概念であり、local worker の `backend.*` config ではない。
- NIM の `CONTAINER_ID` / `NIM_TAGS_SELECTOR` は serving profile selection であり、local NeMo worker の `backend.language`、`backend.sample_rate_hz`、attention context、streaming params へそのまま移植しない。
- NeMo docs は、cache-aware streaming Conformer が limited right context と caching を使い、Conformer-CTC と Conformer-Transducer で support されると説明している。
- NeMo docs は、offline full-context model を streaming simulation に使う道も示しているが、accuracy / latency tradeoff があり、limited-context streaming model とは別物として扱う必要がある。

これらは model family / serving stack / algorithm concept の根拠です。
FluentAudio backend の完了判定は、local `.nemo` に対する `health -> start -> audio -> finish` と full ROS graph の代表検証で行います。

Parakeet 1.1B RNNT Multilingual については、公式資料上の `Streaming + Offline`、25+ languages / auto-detect、`ja-JP` support は NIM/Riva serving profile の capability です。
これは local `.nemo` worker が finite attention context を持ち、`health`、`start`、`audio`、non-empty `final` に成功することとは別の事実です。

## Artifact

現時点で検証対象にしている artifact は次です。

```text
nvidia/riva/parakeet-rnnt-riva-1-1b-unified-ml-cs-universal:trainable_v1.0
```

local runtime では次の `.nemo` を参照しています。

```text
src/ai/fa_asr/models/nemo_rnnt_streaming/
  parakeet-rnnt-riva-1-1b-unified-ml-cs-universal_vtrainable_v1.0/
    Parakeet-RNNT-XXL-1.1b_merged_universal_spe8.5k_1.0.nemo
```

model checkpoint は git 管理しません。
host または prepare container で取得し、runtime container には volume として渡します。
host の通常 `PATH` には `ngc` が無く、repo-local `./ngc-cli/ngc` は NGC CLI `4.18.0` として存在することを確認しています。
`fluent-audio-runtime` container 内 PATH に `ngc` CLI がある前提で runtime node が model download する設計にはしません。

NGC CLI でこの artifact を取得する場合の操作単位は、次の pinned version です。

```text
ngc registry model download-version \
  nvidia/riva/parakeet-rnnt-riva-1-1b-unified-ml-cs-universal:trainable_v1.0 \
  --dest <models_dir>
```

この command の意味は、NGC model version の中身を `<models_dir>` 以下へ取得することだけです。
`--dest` は取得先、`--file` は取得 file の wildcard、`--exclude` は除外 wildcard、`--format_type` は CLI output の format です。
`download-version` は Riva artifact を NeMo local streaming 用に検証せず、`.nemo` の model config を変更せず、有限 attention context を追加せず、worker `health` も実行しません。
version を省略すると NGC 側の latest version を対象にするため、FluentAudio の backend 検証では version を省略しません。
`download-version` の成功は local disk 上に version contents が置かれたことだけを意味し、`ASRModel.restore_from(...)`、finite attention context 追加、worker health、NIM / Triton / Riva runtime 再現は行いません。

NGC metadata で確認した version facts は次です。

| NGC field | 確認値 | backend 上の扱い |
| --- | --- | --- |
| version | `trainable_v1.0` | 取得対象は latest ではなくこの pinned version として扱う。 |
| `totalFileCount` | `1` | artifact version は単一 file で構成される。 |
| file path | `Parakeet-RNNT-XXL-1.1b_merged_universal_spe8.5k_1.0.nemo` | local worker が restore する候補 file。 |
| file size | `4011233560` bytes | runtime container には volume として渡す。git 管理しない。 |
| file SHA256 base64 | `UjMulu9o/4z+/R2Ne4xde1Mz+qPPrIftTMe17D1YIcA=` | 取得後の integrity 確認に使えるが、ASR 成功の証明ではない。 |
| model format | `riva` | NGC model card の配布分類。local `.nemo` restore 可能性とは別に実測する。 |
| runtime engine | `Riva 2.19.0 or higher` / Triton | Riva serving stack 向けの記述であり、FluentAudio local worker の runtime ではない。 |
| model detail | `NeMo Version=1.23.0`, `Dataset Size=94k hrs`, `Vocabulary Size=8.5k` | model 選定情報。local NeMo 2.7.3 worker での streaming 成立は別検証。 |
| model card language | 25 languages, including `ja-JP` | 日本語対応の根拠。ただし transcript 成功は speech fixture で検証する。 |
| model card input | mono audio required, input format `wav` | Riva serving 入力の説明。FluentAudio worker は WAV ではなく ASR-ready float32 chunks を受ける。 |

これらの model card / metadata / integrity 値は selection、pinning、download 後の整合性確認の根拠です。
`model format=riva`、`Riva 2.19.0 or higher` / Triton、`NeMo Version=1.23.0`、file count、file size、SHA256 が揃っても、ASR 成功や local worker health 成功は証明しません。

この `.nemo` の backend-relevant facts は次です。

| 項目 | 確認値 | backend 上の意味 |
| --- | --- | --- |
| `sample_rate` | `16000` | `fa_asr` が backend へ渡す ASR-ready audio は 16 kHz mono `FLOAT32LE` でなければならない。 |
| `preprocessor.window_stride` | `0.01` | feature frame は 10 ms stride。raw samples と encoder token の対応計算に使う。 |
| `encoder.subsampling_factor` | `8` | encoder output token は raw feature frame より粗い時間単位になる。 |
| `encoder.att_context_size` | `[-1, -1]` | restore 直後は full context。現行 worker はこれを local streaming 成功扱いしない。 |
| `encoder.att_context_style` | `regular` | limited/chunked context を示す config ではない。 |
| model target | `nemo.collections.asr.models.rnnt_bpe_models.EncDecRNNTBPEModel` | RNNT/BPE 系 model として restore される。 |
| decoder / joint | `RNNTDecoder` / `RNNTJoint` | RNNT/Transducer 系である根拠。 |
| tokenizer | `merged_universal_tokenizer` | universal tokenizer artifact。language support の一部だが streaming 成立の根拠ではない。 |
| supported attention contexts | `[[-1, -1]]` | finite context が support されていないため、現行 worker policy では fail closed する。 |
| vocabulary | `<ja-JP>` を含む | 日本語 token の存在は日本語 ASR 成功の証明ではない。 |

local file と runtime image について確認済みの事実は次です。

| 項目 | 確認値 | backend 上の意味 |
| --- | --- | --- |
| local `.nemo` path | `src/ai/fa_asr/models/nemo_rnnt_streaming/parakeet-rnnt-riva-1-1b-unified-ml-cs-universal_vtrainable_v1.0/Parakeet-RNNT-XXL-1.1b_merged_universal_spe8.5k_1.0.nemo` | worker が restore した local file。git 管理しない。 |
| local file size | `4011233560` bytes | NGC metadata の size と整合するが、ASR 成功の証明ではない。 |
| local SHA256 | `52332e96ef68ff8cfefd1d8d7b8c5d7b5333faa3cfac87ed4cc7b5ec3d5821c0` | local file integrity の証跡。worker capability 成立とは別。 |
| repo-local NGC CLI | `./ngc-cli/ngc`, version `4.18.0` | artifact 取得用 tooling。runtime worker の依存ではない。 |
| host `PATH` | `ngc` absent | host shell で `ngc` が常に使える前提にしない。 |
| runtime image | `vlabor-fluent-audio:local` | worker、NeMo、Torch、CUDA が入っている検証対象 image。 |
| GPU | RTX 5070 Ti | CUDA runtime が存在する検証環境。health failure の主因ではない。 |

`vlabor-fluent-audio:local` で `backend.language=ja-JP`、16 kHz mono `FLOAT32LE`、`emit_partial=true`、`chunk_size_samples=1600`、`max_partial_interval_ms=300` の `health` を実行した結果、worker は code `1` で終了しました。
stderr では model が `EncDecRNNTBPEModel` として restore された後、次の理由で fail closed しています。

```text
model encoder does not support a finite attention context; supported_contexts=[[-1, -1]]
```

trace には `command_received health`、`health_start`、同じ理由の `worker_error` が残ります。
したがって、この失敗は CUDA、Torch、NeMo、worker executable、model file 欠落ではなく、現行 worker policy と model capability の契約不一致です。
`health` が失敗しているため、speech fixture、partial emission、streaming transcript、full ROS graph 上の ASR 成功は未検証です。

注意すべき operational drift は次です。

- `src/ai/fa_asr/scripts/prepare_nemo_rnnt_streaming_asr` は現時点では Parakeet 1.1B multilingual の NGC preparer ではありません。
- 同 script の default は `nemotron-speech-streaming-en-0.6b` で、Hugging Face URL から `curl` で `.nemo` を取得します。
- 同 script の `validate_model_id()` は `nemotron-speech-streaming-en-0.6b` 以外を拒否します。
- 同 script は `${MODELS_DIR}/${MODEL_ID}.nemo` を出力 path にするため、上記 Parakeet NGC artifact の directory layout とは一致しません。

したがって、Parakeet 1.1B multilingual の取得、配置、integrity check、worker `health` は、現行 prepare script の成功とは別の作業として扱います。
prepare script を実行して env block が出たことを、Parakeet backend の準備完了や `health_ok` 成功として扱ってはいけません。

## Input Contract

`fa_asr` から worker へ渡す audio は、すでに ASR-ready でなければなりません。

- sample rate: `backend.sample_rate_hz`
- channels: `1`
- audio encoding: `FLOAT32LE`
- value range: normalized float32 `[-1.0, 1.0]`
- wire encoding: `base64_float32le`
- stream identity: `fa_asr.expected_source_id` / `fa_asr.expected_stream_id` と一致済み
- time range: `AudioFrame` timeline contract で連続性確認済み

この backend は以下を行いません。

- resample
- downmix
- bit-depth conversion
- PCM16 conversion
- WAV / OGG / MP3 / FLAC decode
- value range normalization
- missing frame 補完
- VAD / KWS / TD の代替判定
- 別 ASR backend への fallback

未対応形式を受け取った場合は、変換して継続せず fail closed します。

## Required Config

現行実装が要求する設定は次です。

- `backend.command`: `nemo_rnnt_streaming_worker` executable
- `backend.model_path`: local readable `.nemo` file
- `backend.language`: request language
- `backend.sample_rate_hz`: model sample rate と一致する値
- `backend.channels`: `1`
- `backend.chunk_size_samples`: worker JSONL contract 上の raw audio chunk size
- `backend.emit_partial`: partial result を返すか
- `backend.max_partial_interval_ms`: partial interval contract。health / config に保持するが、実 transcript 成功とは別。
- `backend.timeout_sec`: JSONL response timeout

`backend.chunk_size_samples` は NeMo encoder chunk size ではありません。
raw sample chunk と encoder token / feature chunk は単位が違うため、`window_stride` と `subsampling_factor` を通して橋渡しします。

## Backend Algorithm Detail

`nemo_rnnt_streaming_worker` は JSONL wrapper ではありません。
NIM / Riva serving stack を使わない以上、model restore、capability validation、streaming setup、cache state、decoder state、error reporting を backend algorithm として所有します。

### Phase 1: Model Restore

入力:

- readable `.nemo` path
- worker config

処理:

1. `ASRModel.restore_from(...)` で `.nemo` を restore する。
2. NeMo が stdout へ出し得る message を JSONL protocol から隔離する。
3. model class を記録する。

拒否条件:

- `.nemo` path が無い、読めない。
- restore が例外を出す。
- restore 後に model object が ASR model として必要な属性を持たない。

この phase の成功は model load の成功であり、streaming ASR 成功ではありません。

### Phase 2: Artifact And Config Extraction

入力:

- restored model
- model config
- backend config

取得する値:

- model sample rate
- `preprocessor.window_stride`
- `encoder.subsampling_factor`
- `encoder.att_context_size`
- supported attention contexts
- decoder strategy
- language token / vocabulary
- cache-aware streaming API の有無

拒否条件:

- 必須値が取れない。
- sample rate が `backend.sample_rate_hz` と一致しない。
- `backend.channels != 1`。
- RNNT / Transducer として扱えない。
- encoder が streaming cache API を持たない。

不足値を default で補って進めません。

### Phase 3: Capability Validation

入力:

- extracted model capability
- requested backend config

検査すること:

1. model が RNNT / Transducer 系であること。
2. encoder が cache-aware streaming に必要な API を持つこと。
3. model sample rate と backend sample rate が一致すること。
4. language request が model の language contract に反しないこと。
5. partial support を backend contract と矛盾なく扱えること。

拒否条件:

- capability が満たされない。
- capability が曖昧で、成功時の意味を説明できない。

この phase で `health_ok` を返すには、次の attention context phase も成立している必要があります。

language validation は条件付きです。
worker は model config が `languages` または `language` を公開している場合だけ `backend.language` と照合します。
model config が language metadata を公開しない場合、worker は language を理由に拒否しません。
その場合でも vocabulary に `<ja-JP>` があることは「日本語を扱う前提と矛盾しない」根拠に留まり、日本語 speech fixture の transcription 成功を代替しません。
NIM support matrix の `ja-JP`、local vocabulary token の `<ja-JP>`、FluentAudio config の `backend.language` は同一 contract ではありません。
現行 SO101 profile は `backend.language: en` であり、日本語 Parakeet 検証には未適合です。
日本語検証では canonical language value を Product Owner が決める必要があります。
model が language metadata を公開する場合、worker は `backend.language` を exact match で検査し、不一致なら fail closed します。

### Phase 4: Attention Context Validation

入力:

- encoder
- encoder config
- current `att_context_size`
- supported attention contexts

現行 worker policy:

1. restore 直後の context が full context `[-1, -1]` かを調べる。
2. full context の場合、supported contexts を `encoder.att_context_size_all` などから取得する。
3. requested finite context は現行固定 policy として `[70, 1]` を使う。
4. `[70, 1]` が supported contexts に含まれる場合だけ `set_default_att_context_size([70, 1])` を呼ぶ。
5. supported contexts が full context だけなら fail closed する。
6. supported contexts に `[70, 1]` が無ければ fail closed する。

重要な NeMo source contract:

- NeMo の `set_default_att_context_size(...)` は unsupported look-ahead の場合 warning を出す。
- しかし warning 後も `att_context_size` をセットし、`setup_streaming_params()` を呼ぶ。
- そのため FluentAudio worker は NeMo warning を信用して継続せず、呼び出し前に自前で supported contexts を検査する。

実測:

```text
supported_contexts=[[-1, -1]]
model encoder does not support a finite attention context; supported_contexts=[[-1, -1]]
```

この結果により、手元の Parakeet `.nemo` は現行 worker policy では local streaming backend として未成立です。

### Phase 5: Streaming Params Setup

入力:

- `backend.chunk_size_samples`
- `backend.sample_rate_hz`
- `preprocessor.window_stride`
- `encoder.subsampling_factor`
- selected finite attention context

処理:

1. raw sample chunk と encoder token の単位を分ける。
2. `encoder_token_samples = sample_rate_hz * window_stride * subsampling_factor` を計算する。
3. `requested_encoder_chunk_size = ceil(chunk_size_samples / encoder_token_samples)` を計算する。
4. 現行 policy では `effective_encoder_chunk_size = max(10, requested_encoder_chunk_size)` とする。
5. `shift_size = max(1, effective_encoder_chunk_size // 2)` とする。
6. `left_chunks=2` として `encoder.setup_streaming_params(...)` を呼ぶ。
7. effective `streaming_cfg` を trace できる形で記録する。

拒否条件:

- `setup_streaming_params(...)` が無い。
- `setup_streaming_params(...)` が失敗する。
- `streaming_cfg.last_channel_cache_size` が positive integer ではない。
- attention context が未成立のまま streaming params を成立扱いしようとする。

full-context model を simulated streaming で使う道は NeMo docs 上存在します。
しかしそれは accuracy / latency tradeoff を持つ別 policy です。
現行 `nemo_rnnt_streaming` backend の成功条件とは分け、明示仕様なしに full context を受け入れません。

### Phase 6: Stream Start And Cache State

入力:

- `start` command
- session id
- start command に含まれる worker config

処理:

1. active session が無いことを確認する。
2. `CacheAwareStreamingAudioBuffer` を作る。
3. `encoder.get_initial_cache_state(batch_size=1)` で encoder cache を初期化する。
4. previous hypothesis / previous decoder output を初期化する。
5. accepted sample count を 0 にする。
6. trace に stream start と effective `streaming_cfg` を記録する。

拒否条件:

- start command の config が model restore / capability validation / streaming setup を成立させられない。
- cache state が 3 要素 tuple として得られない。
- active session がすでに存在する。

cache は session-local です。
別 user turn / 別 session と共有しません。
backend wrapper は health と start に同じ resolved config を送りますが、raw worker protocol は「過去の health config と start config の比較状態」を持つ contract ではありません。
start 単体でも restore / capability validation を行える transport であり、health-first は system 側の完了条件です。

### Phase 7: Audio Validation

入力:

- `audio` command
- `session_id`
- `encoding`
- `sample_count`
- base64 float32le payload

検査:

1. `session_id` が active stream と一致する。
2. `encoding == base64_float32le`。
3. `sample_count * channels * 4` と payload byte length が一致する。
4. decoded samples が finite。
5. decoded samples が normalized `[-1.0, 1.0]` 内。

拒否条件:

- session mismatch
- unsupported encoding
- malformed base64
- byte length mismatch
- NaN / Inf
- range violation

worker は PCM16、WAV、別 sample rate、stereo を受け取って変換しません。
変換は upstream processing node の責務です。

### Phase 8: Buffer Drain

入力:

- validated float32 samples
- stream-local buffer

処理:

1. samples を `CacheAwareStreamingAudioBuffer` に append する。
2. buffer が推論可能 chunk を返すまで待たない。
3. 返された chunk だけを順に streaming step へ渡す。
4. chunk が無い場合は transcript なしの accepted state として扱う。
5. append / drain / chunk count / duration を trace に残す。

注意:

- `audio_accepted` は transcript 成功ではありません。
- `audio_accepted` は worker が chunk を stream state へ受理したことだけを意味します。
- partial / final transcript は別 event です。

### Phase 9: Streaming Step

入力:

- processed chunk
- chunk length
- encoder cache
- previous hypothesis
- previous decoder output

処理:

1. `streaming_cfg.last_channel_cache_size` を取得する。
2. chunk の time length を取得する。
3. `required_max_audio_length = last_channel_cache_size + chunk_time_length` を計算する。
4. `encoder.set_max_audio_length(required_max_audio_length)` を呼び、positional encoding length を成立させる。
5. `model.conformer_stream_step(...)` を呼ぶ。
6. returned cache / hypothesis / decoder output を stream-local state として更新する。
7. non-empty text があれば partial candidate として保持する。
8. duration、text length、exception を trace する。

拒否条件:

- `set_max_audio_length(...)` が無い。
- maximum length 設定に失敗する。
- `conformer_stream_step(...)` が無い。
- return value が期待 tuple と一致しない。
- tensor shape mismatch が出る。

tensor shape mismatch を empty transcript として扱いません。
worker error として fail closed します。

### Phase 10: Partial / Drain / Finish

`partial`:

- non-empty text があり、`backend.emit_partial=true` の場合だけ publish 候補になる。
- empty partial は transcript 成功ではない。

`drain`:

- 新しい audio を追加しない。
- 現在の latest non-empty partial state を返す。
- active session が無ければ拒否する。

`finish`:

1. accepted audio が 1 sample 以上あることを確認する。
2. buffer の残りの推論可能 chunk を処理する。
3. final transcript を返す。
4. final が空でも prior non-empty partial がある場合、backend wrapper は latest partial を final commit 候補にできます。
5. final も prior partial も空の場合、transport completion としての `finished` は成立し得ますが、speech window の ASR 成功として扱わない。

silence smoke では empty final が transport / state-machine の確認として成立する場合があります。
しかし VAD / KWS / TD / TurnContext によって speech window として開いた区間で empty final が返り、last partial commit も成立していない場合、それを「ユーザーが何も言わなかった」と暗黙解釈してはいけません。

### Phase 11: Error Reporting And Trace

現行 worker は opt-in JSONL trace を持ちます。

- `FLUENT_AUDIO_NEMO_RNNT_WORKER_TRACE_FILE`
- `FLUENT_AUDIO_NEMO_RNNT_WORKER_TRACE_STDERR`

trace すべき情報:

- command 受信
- audio chunk validation / accept
- buffer append / drain
- effective streaming config
- selected attention context
- `conformer_stream_step` start / finish / duration / exception
- partial / final generation
- worker error / exception

trace は問題を隠すための fallback ではありません。
失敗した phase、拒否理由、stderr の本質行を追跡するための証跡です。

公開 JSONL `health_ok` response と trace / stderr は責務が違います。
`health_ok` response keys は model class、cache-aware streaming の有無、sample rate、channels、audio encoding、streaming flag、partial capability、language、chunk size、partial interval です。
selected finite attention context と effective `streaming_cfg` は現行 `health_ok` response keys には含まれません。
それらは worker 内で `health_ok` 返却前に検査 / 設定されますが、`fa_asr` が health response field として検証しているわけではありません。
selected context と effective `streaming_cfg` は opt-in trace、stderr、worker internal validation の証跡として扱い、公開 JSONL health contract と混同しません。

## Preparation Script Boundary

`prepare_nemo_rnnt_streaming_asr` は backend algorithm の検証器ではありません。
この script の成功が意味するのは、次だけです。

1. `nemo_rnnt_streaming_worker` が存在し executable である。
2. configured model file が既に存在する、または configured URL から download できた。
3. trace file を作成できた。
4. host / container path 用の environment export block を出力できた。

この script は次を行いません。

- NGC CLI による Parakeet 1.1B multilingual artifact download。
- `ASRModel.restore_from(...)` による `.nemo` restore。
- RNNT / Transducer capability validation。
- sample rate / mono / float32 worker contract validation。
- supported attention contexts の検査。
- finite attention context selection。
- `health -> start -> audio -> finish` の worker protocol smoke。
- full ROS graph の起動や ASR result 確認。

したがって、prepare trace の `status=success` は file preparation の成功であり、ASR backend の成功ではありません。
backend 完了判定は次章の verification requirements で行います。

## Observed Validation

2026-05-22 の実測事実:

1. 現行 worker は supported contexts を検査する。
2. Parakeet `.nemo` は model restore 後に full-context only と判定された。
3. worker は `health_ok` を返さず、`returncode=1` で fail closed した。
4. 本質行は次。

```text
model encoder does not support a finite attention context; supported_contexts=[[-1, -1]]
```

このため、過去資料にあった次の記述は現状と矛盾します。

- 「worker health check は成功する」
- 「無音 minimal streaming smoke は通過済み」
- 「現行 worker は full-context を検出した場合 `[70,1]` を設定する」
- 「1、3、4、5、6、7、8、9、10 は基本形実装済み」

正しくは、現行 worker は unsupported finite context を検出して早期 fail closed します。
これは失敗ではありますが、意味を壊して進まないという点では正しい挙動です。

## Verification Requirements

この backend は、次の順に検証しなければ完了扱いしません。

### 1. Worker Health

`health` が成功し、公開 JSONL health response で次を確認する。

- model class
- sample rate
- cache-aware streaming support
- partial support

selected finite attention context と effective streaming config は公開 health response の field ではありません。
これらは worker internal validation と opt-in trace / stderr で確認する証跡です。
`health_ok` が返るには内部 validation が先に成立している必要がありますが、`fa_asr` が response key として selected context や `streaming_cfg` を照合するわけではありません。

現時点の Parakeet `.nemo` はここで fail closed しているため、後続検証へ進めません。

### 2. Worker Minimal Streaming Step

ROS を使わず、worker 単体で次を流す。

```text
health
start
audio
audio
finish
```

無音でもよいが、少なくとも worker が落ちずに `audio_accepted` と `final` を返す必要があります。
これは transcript 品質検証ではありません。

### 3. Worker Speech Fixture

既知の短い speech fixture を使い、partial または final に non-empty transcript が出ることを確認する。
日本語 model として使う場合は、日本語 speech fixture を使う。

### 4. Full ROS Graph

次の graph で実音声または file source 音声を流す。

```text
fa_in
  -> fa_sample_format
  -> fa_resample
  -> fa_vad
  -> fa_asr
```

確認対象:

- `voice/vad_state`: `start=true` / `end=true`
- `voice/asr/event`: `EVENT_STREAM_OPENED` / `EVENT_STREAM_AUDIO_PUSHED` / `EVENT_STREAM_FINAL_RESULT` / `EVENT_STREAM_CLOSED`
- `voice/asr/result`: `STATUS_FINAL` and non-empty `text`

`STATUS_ERROR`、empty final、backend crash、VAD が開かない状態は未完了です。

file source でこの graph を検証する場合、`fa_in` は `backend.name=pcm_file_reader` として headerless raw PCM だけを読みます。
WAV / OGG / MP3 / FLAC decode、resample、downmix、gain、normalization は `fa_in` で行いません。
file の sample rate、channel count、bit depth、encoding、layout は `audio.*` config によって与えられ、file から推定しません。
file size は configured frame byte size で割り切れる必要があります。
`file.path` は source identity ではなく、`AudioFrame.stream_id` は `audio.stream_id` から設定されます。

finite file source では、最初の frame が downstream subscriber discovery 前に publish されると検証が壊れます。
このため、file-source E2E では `fa_in.startup.required_subscribers >= 1` とし、subscriber が matched してから source read を開始します。
これは replay や fallback ではなく、有限入力を見えないまま捨てないための source-read gating です。

ASR に渡る identity は raw file stream ではなく、最後に ASR-ready になった stream identity です。
`fa_sample_format` と `fa_resample` を通した場合、`fa_vad` input と `fa_asr.expected_stream_id` は 16 kHz mono `FLOAT32LE` の出力 stream に合わせます。
`audio_topic` は ROS transport topic、`expected_stream_id` は `AudioFrame.stream_id` です。
topic 名、file path、raw input stream id を ASR-ready stream identity と混同しません。

現行 SO101 voice frontend profile では、full pipeline は次の binding です。

```text
audio/frame (stream_id=audio/raw/mic)
  -> fa_sample_format
  -> audio/sample_format/mic (stream_id=audio/float32/mic)
  -> fa_resample
  -> audio/resample16k/mic (stream_id=audio/preprocessed/mono16k)
  -> fa_dc_offset_removal
  -> audio/dc_offset_removed/frame (stream_id=audio/dc_offset_removed/mic)
  -> fa_high_pass
  -> audio/high_pass/frame (stream_id=audio/high_pass/mic)
  -> fa_vad / fa_asr / fa_kws / fa_turn_detector
```

ASR-ready stream は `audio/high_pass/frame` + `AudioFrame.stream_id=audio/high_pass/mic` です。
`fa_asr.expected_source_id=mic`、`fa_asr.expected_stream_id=audio/high_pass/mic`、`control.speech_control.source_id=mic`、`control.speech_control.stream_id=audio/high_pass/mic` を profile contract として揃えます。
現行 code は control config の source/stream と `expected_*` の一致を起動時に強制していないため、これは profile 側で守る binding contract です。

archive branch は `audio/high_pass/frame` から `fa_archive_sample_format` で分岐し、`audio/archive_pcm16/frame` + `AudioFrame.stream_id=audio/archive_pcm16/mic` を `fa_audio_window` に渡します。
これは export / archive 用の PCM16 window であり、ASR backend input ではありません。

ASR-ready rolling timeline は backend に渡す前の `fa_asr` contract です。
`timeline.timestamp_alignment_tolerance_ms` 内の bounded overlap / gap は sample boundary に寄せます。
tolerance 超過 overlap / gap や selected range の coverage failure は、backend に空 audio や短い audio を渡さず、timeline error として扱います。

## Failure Conditions

次の場合、この backend は fail closed します。

- `backend.command` missing / not executable
- `backend.model_path` missing / unreadable
- preparation script success is treated as backend readiness without worker health
- model restore failure
- model is not RNNT / Transducer
- model sample rate mismatch
- channels is not `1`
- audio encoding is not `FLOAT32LE`
- input sample is NaN / Inf / outside `[-1.0, 1.0]`
- cache-aware streaming API unavailable
- supported attention contexts unavailable
- requested finite attention context unsupported
- only full-context `[-1,-1]` is supported under current policy
- streaming params cannot be established
- encoder maximum audio / positional encoding length cannot be established
- tensor shape mismatch in streaming step
- worker stdout closes before JSONL response
- health response does not match backend config
- start command config cannot establish the same model / streaming capability contract
- audio / drain / finish session id differs from active session
- finish is called before any accepted audio

失敗時に別 ASR backend へ切り替えません。

## Next Decisions

Product Owner が次に判断すべきこと:

1. 現行 policy のまま、finite attention context を support する別 `.nemo` / model artifact を選ぶ。
2. full-context/offline model を simulated streaming として使う別 backend policy を設計する。
3. Riva / NIM serving stack を使う別 backend として切り分ける。
4. NeMo local worker で supported contexts のある model を前提に、trace と representative validation を進める。

1 と 4 は `nemo_rnnt_streaming` の現行思想に近いです。
2 は latency / accuracy tradeoff を持つ別仕様です。
3 は NIM/Riva を使わないという現在方針と衝突するため、別 decision として扱います。

## Research Sources

この backend document は、次の情報源に基づいています。

- NVIDIA ASR NIM docs
  - https://docs.nvidia.com/nim/speech/26.02.0/asr/index.html
  - ASR NIM が pre-trained NeMo model と TensorRT / Triton inference stack を self-contained container に package し、model download / optimization / serving を扱うこと。
  - streaming mode が audio arrival に応じて partial transcripts を返すこと。
  - Parakeet RNNT Multilingual が 25+ languages、Streaming + Offline、auto language detection を持つ model family として扱われること。
- NVIDIA ASR NIM Support Matrix
  - https://docs.nvidia.com/nim/speech/latest/reference/support-matrix/asr.html
  - Parakeet 1.1B RNNT Multilingual が streaming speech-to-text transcription 対応として扱われること。
  - `ja-JP` が default / prompt type の supported language に含まれること。
- NVIDIA ASR NIM deploy docs
  - https://docs.nvidia.com/nim/speech/latest/asr/deploy-asr-models/index.html
  - `CONTAINER_ID=parakeet-1-1b-rnnt-multilingual` と `NIM_TAGS_SELECTOR` による serving profile selection の説明。
  - これは NIM serving の説明であり、local `.nemo` worker の成功証明ではないこと。
- NGC CLI registry docs
  - https://docs.nvidia.com/dgx/ngc-registry-cli-user-guide/index-bak.html
  - `ngc registry model download-version <org>/[<team>/]<model-name:version>` が registry から local disk へ指定 version を取得する操作であること。
  - `download-version` が artifact contents の取得であり、format conversion や worker health check ではないこと。
- NVIDIA NeMo Framework ASR Models docs
  - https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/models.html
  - cache-aware streaming Conformer が limited right context と caching を使い、Conformer-CTC / Conformer-Transducer で support されること。
  - full-context model を streaming simulation に使う場合、accuracy / latency tradeoff があること。
- NeMo ConformerEncoder source docs
  - https://docs.nvidia.com/nemo/speech/nightly/_modules/nemo/collections/asr/modules/conformer_encoder.html
  - `set_default_att_context_size(...)` が unsupported look-ahead を warning しつつ、その後 `att_context_size` をセットし、`setup_streaming_params()` を呼ぶこと。
- local `fluent-audio-runtime` NeMo 2.7.3 source
  - `ConformerEncoder.setup_streaming_params(...)`
  - `ConformerEncoder.get_initial_cache_state(...)`
  - `ConformerEncoder.set_max_audio_length(...)`
  - `CacheAwareStreamingAudioBuffer`
  - `conformer_stream_step(...)`
- local worker / container 実測
  - NGC CLI metadata for `nvidia/riva/parakeet-rnnt-riva-1-1b-unified-ml-cs-universal:trainable_v1.0`
  - local `.nemo` `model_config.yaml`
  - `model encoder does not support a finite attention context; supported_contexts=[[-1, -1]]`
  - `returncode=1`

外部 docs は model capability と serving stack の説明として使います。
FluentAudio backend の完了判定は、local worker と full ROS graph の実検証に基づきます。

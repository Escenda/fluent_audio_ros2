# nemo_offline_transcribe Backend

## Backend Name

`nemo_offline_transcribe`

## Status

`nemo_offline_transcribe` は local `.nemo` file を NeMo offline / full-context `transcribe(...)` API で呼ぶ non-streaming command / worker backend です。
working tree 上では backend loader、worker、unit tests、profile work が作成済みです。

2026-05-22 時点で PO による real worker health と raw `FLOAT32LE` fixture transcription は通過済みです。
ただし、file-source full ROS graph validation と accuracy 評価は未完了です。
file-source full graph 試行は VAD start/end と ASR start まで到達しましたが、ASR-ready rolling timeline の gap により backend へ連続 audio を渡せず、non-empty final result には到達していません。
したがって、この backend を本番相当経路で検証済みとは扱いません。
この document は、実装済みの backend contract、local runtime evidence、未検証範囲を分けて記録します。

## Evidence Classes

この backend では根拠を 3 種類に分けます。

| 分類 | 根拠 | この backend で言えること | 言えないこと |
| --- | --- | --- | --- |
| 公式一次情報 | NVIDIA NeMo / NIM / Riva / NGC docs | NeMo offline / streaming の概念、NIM/Riva が service runtime であること、NGC artifact が取得元であること。 | FluentAudio worker が実機で動いたこと。 |
| local 実測 | `fluent-audio-runtime`、Torch `2.11.0+cu130`、CUDA available、`EncDecRNNTBPEModel` restore、offline fixture output | 対象 local `.nemo` を NeMo offline API で restore/transcribe できること。 | streaming backend 成功、accuracy、full ROS graph 成功。 |
| FluentAudio 設計判断 | no fallback、ASR-ready stream、backend boundary、unsupported input fail-closed | worker の入力契約、WAV bridge の責務、失敗時の扱い、完了条件。 | NVIDIA 公式 capability の代替や、未検証経路の成功宣言。 |

公式情報だけで local backend が動いたとは書きません。
local offline 実測だけで streaming 対応とは書きません。

## 調査結果を offline backend algorithm に落とす

`nemo_offline_transcribe` は、local `.nemo` を NeMo offline / full-context `transcribe(...)` API で呼ぶ backend です。
この backend の document では、NGC / Riva / NIM / gRPC / local NeMo / streaming を次のように分けます。

### 1. NGC `download-version` は model file の取得だけである

`ngc registry model download-version <artifact>` は、指定した model version の contents を local disk に取得する preparation command です。
`--dest` は取得先、`--file` は取得 file wildcard、`--exclude` は除外 wildcard、`--format_type` は CLI output format です。
この command は `.nemo` restore を実行せず、temporary WAV bridge を作らず、offline transcript を生成せず、ASR worker health も実行しません。

したがって、この backend で NGC download 成功を `nemo_offline_transcribe` readiness として扱いません。
download 成功後に必要なのは、local file integrity、worker health、real `.nemo` restore、offline `transcribe(...)` availability、speech fixture、full ROS graph の順の検証です。

### 2. Riva/NGC artifact を local `.nemo` として読むことは Riva/NIM serving ではない

現在の local 実験では、Riva/NGC artifact 由来の `.nemo` を NeMo runtime へ直接渡しています。
これは Riva server、RMIR、Triton、NIM container、gRPC endpoint を使う構成ではありません。

この backend が所有するのは、local command process、`.nemo` restore、model sample rate / language metadata validation、raw `FLOAT32LE` input validation、temporary PCM16 mono WAV bridge、NeMo offline `transcribe(...)` call、stdout / output file emission です。
Riva/NIM が所有する serving profile、server-side stream state、Riva client request、Triton/TensorRT execution はこの backend に含めません。

### 3. NIM/Riva support matrix は local offline worker の成功証明ではない

NIM support matrix 上の Parakeet RNNT Multilingual capability は、NIM/Riva serving stack 上での model family / profile / language support を説明します。
`ja-JP` support が matrix にあることは、日本語 model selection の根拠として有用です。
しかし、この backend の local worker が real `.nemo` を restore し、`transcribe(...)` が non-empty text を返すことは、local runtime で別に検証します。

support matrix は次を証明しません。

- worker process が NeMo を import できること。
- `ASRModel.restore_from(...)` が local `.nemo` を読めること。
- model sample rate と configured sample rate が一致すること。
- worker の raw float32le input validation が通ること。
- temporary WAV bridge が作れること。
- offline `transcribe(...)` result が non-empty になること。
- full ROS graph で `AsrResult` が publish されること。

### 4. Offline worker health

`nemo_offline_transcribe_worker health` は、download 済み file の存在確認だけではありません。
worker environment に NeMo ASR module があり、`ASRModel.restore_from(...)` が使え、model が `eval()` と offline `transcribe` API を持ち、model sample rate と language metadata が config と矛盾しないことを確認します。

health が通っても transcript success はまだ未証明です。
health は「この backend が audio request を受ける準備がある」ことの startup gate であり、speech content の認識結果は transcribe smoke / ROS graph validation で確認します。

### 5. Temporary WAV bridge は hidden DSP ではない

NeMo offline API は file path input を取るため、worker は validated raw `FLOAT32LE` mono samples を temporary PCM16 mono WAV に書きます。
この bridge は model API adapter です。
`fa_asr` node 本体の hidden resample / downmix / normalization ではありません。

bridge の前提は次です。

- input sample rate は backend config と一致済み。
- channel count は 1。
- samples は finite。
- samples は normalized `[-1.0, 1.0]`。
- empty audio ではない。

worker は sample rate を変えません。
channel 数を変えません。
denoise、gain、AGC、VAD、semantic correction も行いません。
壊れた input を WAV に包んで成功扱いしません。

### 6. Offline/full-context と cache-aware streaming は別物である

この backend は complete audio を渡して complete transcript を得る offline / full-context backend です。
streaming session、encoder cache、partial result、`audio_accepted`、`finish` protocol、low-latency commit は持ちません。

現行 local Parakeet `.nemo` では、offline direct API と worker smoke が non-empty Japanese text を返した evidence があります。
しかし `nemo_rnnt_streaming` は finite attention context 不成立で fail closed しています。
offline success は streaming fail-closed 判定を覆しません。
逆に、streaming backend が fail closed していても、この offline backend の full-context viability は別 evidence として扱います。

### 7. Current local evidence and remaining work

現在 docs に残してよい evidence は次です。

- local `.nemo` file size / SHA256。
- `ASRModel.restore_from(...)` が `EncDecRNNTBPEModel` として restore できたこと。
- direct offline API が Japanese fixture に対して non-empty text を返したこと。
- `nemo_offline_transcribe_worker health` が real Parakeet `.nemo` restore まで到達して `ok` を返したこと。
- raw `FLOAT32LE` 16 kHz mono fixture から worker 経由で non-empty Japanese text が出たこと。
- targeted unit tests が通ったこと。

現在 docs で完了扱いしてはいけない evidence は次です。

- output accuracy。
- file-source full ROS graph validation の完了。現時点の file-source 試行は VAD start/end と ASR start までで、ASR timeline gap により未完了。
- `TranscribeAudio` service integration。
- `nemo_rnnt_streaming` `health_ok`。
- streaming partial / final transcript。
- NIM/Riva serving readiness。

## Runtime Boundary

`nemo_offline_transcribe` は command process backend です。
`fa_asr` node 本体は NeMo / Torch / Parakeet を import せず、ASR-ready mono `FLOAT32LE` samples と sample rate を backend boundary に渡します。

worker は次を所有します。

- local `.nemo` file の restore。
- NeMo ASR module / `ASRModel.restore_from(...)` / model `transcribe` API の検証。
- model sample rate と configured sample rate の照合。
- model が language metadata を公開する場合の language 照合。
- raw normalized `FLOAT32LE` mono samples の再検証。
- NeMo offline API に渡す temporary PCM16 mono WAV bridge。
- `plain_text` / `segments_json_v1` output emission。

worker は次を所有しません。

- NIM server 起動。
- Riva server / model repository / RMIR 管理。
- gRPC endpoint 接続。
- streaming session / cache state / partial transcript。
- VAD / KWS / turn detection。
- upstream resample / downmix / sample-format conversion。
- 別 ASR backend への fallback。

## NIM / Riva / gRPC / NGC / local `.nemo`

NIM は containerized service です。
application は model file を直接触らず、gRPC / HTTP API で NIM container に request を送ります。
container 側が model loading、GPU execution、batching、streaming state を扱います。
`nemo_offline_transcribe` は NIM container を使いません。

Riva は server、model repository、RMIR、Triton、gRPC client を含む serving stack です。
Riva ASR は offline / streaming recognition を扱えますが、Riva の success は Riva server / model repository / RecognitionConfig / gRPC response の成功です。
`nemo_offline_transcribe` は Riva server を使いません。

gRPC は通信方式です。
NIM/Riva の API protocol として使われますが、gRPC 自体が model runtime ではありません。
FluentAudio で gRPC を使う場合は、別 backend protocol として定義します。
この backend の protocol は local command process と temporary file / stdout / stderr です。

NGC artifact は model artifact の取得元です。
file size、SHA256、model card metadata は selection / integrity の根拠ですが、worker health や transcript success ではありません。
`download-version` 成功を backend readiness として扱いません。

local `.nemo` + NeMo は Python process 内で `.nemo` を restore し、model object に API call する方式です。
`nemo_offline_transcribe` はこの方式を使います。

### backend-adjacent 用語の判定表

この backend は offline / full-context local NeMo worker です。
したがって、support matrix、artifact acquisition、worker health、transcript smoke、full ROS graph validation を同じ合格条件として扱いません。

| 用語 | owned boundary | 証明すること | 証明しないこと | FluentAudio failure mode | この doc での位置 |
| --- | --- | --- | --- | --- | --- |
| NIM | NVIDIA ASR NIM container / serving API。 | NIM serving stack 上の ASR capability。 | この local command worker の `.nemo` restore、WAV bridge、offline transcript、ROS graph 成功。 | NIM を使わないため、NIM matrix を worker success としない。NIM backend は別 contract。 | この節の混同禁止項目。 |
| Riva | Riva server / model repository / Riva gRPC client。 | Riva ASR server での model deployment と recognition API。 | local NeMo process の restore / transcribe 成功、temporary WAV bridge、worker stdout。 | Riva artifact metadata を runtime success としない。Riva failure はこの worker の fallback 先にしない。 | この節の混同禁止項目。 |
| gRPC | service transport protocol。 | channel / request / response / deadline の通信境界。 | model runtime、artifact conversion、transcript success。 | この backend は local command / file / stdout / stderr の protocol なので、gRPC readiness を状態に入れない。 | `Runtime Boundary` とこの節。 |
| NGC artifact / `download-version` | NGC registry から local disk への model acquisition。 | pinned artifact contents が取得されたこと。 | `.nemo` restore、temporary WAV bridge、worker health、transcript success、full ROS graph。 | download / integrity failure は preparation failure。download success だけでは backend readiness としない。 | artifact / evidence の前提。 |
| local `.nemo` | `ASRModel.restore_from(...)` の input file。 | file presence、integrity、restore 到達時は NeMo が model object として読めたこと。 | transcript success、accuracy、streaming capability、ROS graph 成功。 | missing / unreadable / suffix mismatch / restore failure は startup failure。 | `Runtime Evidence` と `Required Config`。 |
| NeMo offline / full-context transcribe | worker 内の complete audio transcription API。 | selected audio から non-empty transcript を得る offline policy。 | cache-aware streaming、partial result、`audio_accepted`、low-latency streaming。 | empty transcript、NeMo exception、single transcript 化不能は fail closed。 | `Algorithm` と `Relation To Streaming`。 |
| NeMo cache-aware streaming | 別 backend が所有する streaming cache / partial / final state。 | streaming prerequisites が満たされる場合の stateful streaming 処理。 | この backend の offline transcript success。 | この backend 内で simulated streaming fallback として隠さない。必要なら別 backend policy。 | `Relation To Streaming`。 |
| finite attention context | streaming backend の encoder capability。 | local streaming policy の前提。 | offline/full-context transcript success。 | この backend の合格条件にはしないが、streaming 成功にも昇格しない。 | `Relation To Streaming`。 |
| worker health | offline worker startup readiness。 | real `.nemo` restore、sample rate、language metadata、offline `transcribe` API が起動時に成立すること。 | transcript success、accuracy、ROS graph 成功、streaming readiness。 | timeout / non-zero exit / restore failure / config mismatch は backend unavailable。 | `Verification Requirements` 1。 |
| transcript success | worker が non-empty transcript を返し、schema が valid であること。 | 指定 fixture から text が返ったこと。 | accuracy、language support 全般、ROS graph 成功、streaming readiness。 | empty text、invalid `segments_json_v1`、range mismatch は fail closed。 | `Runtime Evidence` と `Verification Requirements` 2。 |
| full ROS graph validation | ROS graph 上の ASR-ready stream、VAD close、backend invocation、result publish。 | `fa_in -> processing -> fa_vad -> fa_asr` 相当の経路で non-empty final result が出ること。 | worker health や fixture smoke の代替ではない。 | identity / timeline / VAD / backend / publish のどこかが未確認なら未完了。現行 file-source 試行は VAD start/end と ASR start まで到達後、ASR timeline gap で止まっている。 | `Verification Requirements` 5。 |

## Runtime Evidence

`fluent-audio-runtime` container 内で、local Parakeet 1.1B multilingual `.nemo` を NeMo offline API へ直接渡す検証を行いました。

確認済みの runtime facts は次です。

| 項目 | 確認値 | 扱い |
| --- | --- | --- |
| runtime | `fluent-audio-runtime` | NeMo / Torch / CUDA を含む検証環境。 |
| Torch | `2.11.0+cu130` | NeMo offline restore / transcribe を実行した Python runtime。 |
| CUDA | available | local GPU runtime が利用可能な状態。 |
| restored model class | `EncDecRNNTBPEModel` | `ASRModel.restore_from(...)` が local `.nemo` を RNNT/BPE model として restore した証跡。 |
| local `.nemo` path | `models/nemo_rnnt_streaming/parakeet-rnnt-riva-1-1b-unified-ml-cs-universal_vtrainable_v1.0/Parakeet-RNNT-XXL-1.1b_merged_universal_spe8.5k_1.0.nemo` | local file restore 対象。 |
| local file size | `4011233560` bytes | file integrity / selection の証跡。 |
| local SHA256 | `52332e96ef68ff8cfefd1d8d7b8c5d7b5333faa3cfac87ed4cc7b5ec3d5821c0` | local file integrity の証跡。 |
| fixture | `/tmp/fluent_audio_asr_fixture/ja-pronunciation-practice5.ogg` | offline API へ渡した日本語 speech fixture。 |

PO が direct API evidence として使用した NeMo offline API は次です。

```text
ASRModel.restore_from(...);
model.transcribe(
    audio=[...],
    batch_size=1,
    return_hypotheses=False,
    num_workers=0,
    verbose=False,
)
```

fixture に対して返った text は次です。

```text
天 気 練 習 残 業 安 ん な り 電 波 宣 兵 電 米 宣 本 専 用 本 屋 三 円 単 位
```

この結果は、local `.nemo` を NIM / Riva / gRPC なしに NeMo offline / full-context API で restore し、non-empty Japanese output を返せることを示します。
accuracy は証明しません。
`nemo_rnnt_streaming` backend の `health_ok`、streaming `start/audio/finish`、partial/final transcript、full ROS graph 成功も証明しません。

この direct API evidence は worker algorithm そのものではありません。
現行 `nemo_offline_transcribe_worker` は runtime signature を調べ、`batch_size`、`return_hypotheses`、`verbose`、および受け付けられる場合の language key だけを `model.transcribe(...)` へ渡します。
worker が `num_workers` を渡すとは記載しません。

## Input Contract

`fa_asr` から backend へ渡す audio は、すでに ASR-ready でなければなりません。

- encoding: `FLOAT32LE`
- channels: `1`
- sample rate: `backend.sample_rate_hz`
- samples: finite
- range: normalized `[-1.0, 1.0]`
- shape: 1-dimensional non-empty mono samples
- identity/timeline: `fa_asr` 側で source / stream / selected range を検証済み

`fa_asr` node 本体は、PCM16、WAV、stereo、別 sample rate、non-finite samples、range violation を変換して通しません。
worker も raw `.f32` file を再検証し、byte length、empty audio、NaN / Inf、range violation を fail closed にします。

## Model API Bridge

NeMo offline `transcribe(...)` API は file path input を受ける runtime 形態です。
このため worker は validated raw normalized `FLOAT32LE` mono samples を temporary PCM16 mono WAV に書きます。

この WAV bridge の意味は次です。

- backend adapter 内で NeMo model API へ接続するための serialization bridge。
- sample rate と channel count は configured contract のまま保持する。
- `[-1.0, 1.0]` の normalized sample value を PCM16 表現へ写す。
- denoise、gain、normalization、resample、downmix、VAD、KWS、semantic correction は行わない。
- bridge 作成前に input contract を検証し、壊れた音を WAV に包んで成功扱いしない。

したがって、この bridge は `fa_asr` node 本体の hidden resample / downmix / sample-format conversion ではありません。
NeMo API の入口へ渡すための backend boundary 内 adapter です。

## Required Config

必要な設定は次です。

- `backend.name`: `nemo_offline_transcribe`
- `backend.command`: `nemo_offline_transcribe_worker`
- `backend.model_path`: local readable `.nemo` file
- `backend.language`: non-empty language value
- `backend.sample_rate_hz`: positive integer。model config が sample rate を公開する場合は一致必須。
- `backend.channels`: `1`
- `backend.result_format`: `plain_text` または `segments_json_v1`
- `backend.timeout_sec`: health / transcribe response timeout

backend loader は `.nemo` suffix、language、sample rate、channels を起動前に検証します。
default config で backend selection や output contract を暗黙指定してはいけません。

## Algorithm

1. `fa_asr` node 起動時に backend config を読み、`nemo_offline_transcribe` loader を選ぶ。
2. loader は command executable、local `.nemo` model path、language、sample rate、channels、result format を検証する。
3. loader は command process common layer に health args を渡す。
4. startup health は worker に `health --model {model} --language {language} --sample-rate <hz> --channels 1` を実行させる。
5. worker health は NeMo ASR module、`ASRModel.restore_from(...)`、model `eval()`、offline `transcribe` API、model sample rate、language metadata を検証する。
6. `fa_asr` は `AsrRequest.payload` が backend capability と一致することを確認し、raw `float32le` samples を temporary `.f32` file として worker に渡す。
7. worker は `.f32` file を読み、byte length、empty audio、finite samples、normalized range を検証する。
8. worker は temporary PCM16 mono WAV bridge を作る。
9. worker は restore 済み model に `transcribe([wav_path], batch_size=1, return_hypotheses=False, verbose=False, ...)` を呼ぶ。runtime signature が受け付ける場合だけ `language` / `language_id` / `source_lang` / `source_language` のいずれかを渡す。
10. worker は result が single transcript として取り出せることを検証する。
11. empty transcript は fail closed。
12. `plain_text` では transcript text を stdout または output file に出す。
13. `segments_json_v1` では selected request samples 全体 `[0, sample_count)` を覆う 1 segment として strict JSON を出す。
14. command process common layer が `plain_text` / `segments_json_v1` を `AsrTranscript` に変換する。

## Relation To Streaming

`nemo_offline_transcribe` は streaming backend ではありません。
stream session、encoder cache、partial result、`audio_accepted`、`finish` protocol を持ちません。

NeMo docs には offline model を overlapping chunks / buffered streaming 的に使う説明があります。
ただし、この方式は compute duplication と training/inference mismatch を持ち、chunk size や latency/accuracy tradeoff の影響を受けます。
FluentAudio では、この backend の offline success を low-latency streaming success として扱いません。

`nemo_rnnt_streaming` は別 backend です。
現行 Parakeet `.nemo` は streaming worker `health` で finite attention context が成立せず fail closed しています。
`nemo_offline_transcribe` の runtime evidence は、この streaming fail-closed 判定を覆しません。

| 状態 | 現在の扱い |
| --- | --- |
| local `.nemo` file integrity | size / SHA256 確認済み。 |
| NeMo offline restore | `EncDecRNNTBPEModel` として restore 済み。 |
| NeMo offline Japanese fixture transcription | non-empty Japanese text を返すことを確認済み。accuracy は未評価。 |
| `nemo_offline_transcribe` backend / worker / unit tests | working tree 上で作成済み。 |
| `nemo_offline_transcribe` real worker health | `fluent-audio-runtime` で real Parakeet `.nemo` restore まで到達し、`ok` を返すことを確認済み。 |
| `nemo_offline_transcribe` real worker transcription smoke | raw `FLOAT32LE` 16 kHz mono fixture から stdout に `天 気 練 習 残 業 安 心 す ん な り 電 波` が出ることを確認済み。accuracy は未評価。 |
| targeted unit tests | runtime container で `57 passed, 5 warnings in 0.66s`。 |
| file-source full ROS graph validation | 試行済みだが未完了。VAD start/end と ASR start まで到達し、ASR-ready rolling timeline gap で停止。 |
| `nemo_rnnt_streaming` `health_ok` | 未成立。finite attention context 不成立で fail closed。 |
| `nemo_rnnt_streaming` partial / final transcript | 未検証。 |

## Verification Requirements

この backend を完了扱いするには、少なくとも次を確認します。

1. worker health
   - real `.nemo` restore、model class、sample rate、language metadata、device/runtime dependency を検証する。
   - PO 検証では `fluent-audio-runtime` で `nemo_offline_transcribe_worker health` が real Parakeet `.nemo` restore まで到達し、`ok` を返した。
2. worker offline transcription smoke
   - known fixture で worker 経由の non-empty transcript を返す。
   - PO 検証では raw `FLOAT32LE` 16 kHz mono fixture から stdout に `天 気 練 習 残 業 安 心 す ん な り 電 波` が出た。
   - output accuracy は別評価とし、non-empty Japanese output と accuracy を混同しない。
3. backend contract tests
   - unsupported sample rate、stereo、PCM16 input、empty audio、NaN / Inf、range violation、timeout、empty transcript を fail closed にする。
   - model API bridge が hidden resample/downmix として扱われていないことを性質として確認する。
4. `TranscribeAudio` service integration
   - ASR-ready rolling timeline から selected range を切り出し、offline backend へ渡せることを確認する。
5. file-source full ROS graph validation
   - `fa_in -> processing -> fa_vad -> fa_asr` 相当の graph で ASR-ready stream identity、VAD close、backend invocation、non-empty final result を確認する。
   - 現行 file-source 試行では VAD start/end と ASR start までは確認したが、ASR timeline gap により backend へ連続 audio を渡せず、non-empty final result は未確認。

本番相当の ROS graph を検証するまで、backend 単体の smoke を full ROS graph success として扱いません。

## Failure Conditions

次の場合は fail closed します。

- `backend.command` missing / not executable
- `backend.model_path` missing / unreadable / `.nemo` ではない
- `backend.language` empty
- `backend.sample_rate_hz` non-positive
- `backend.channels != 1`
- NeMo ASR module unavailable
- `ASRModel.restore_from(...)` unavailable / failure
- model が `eval()` または offline `transcribe` を持たない
- model sample rate と configured sample rate が一致しない
- model が language metadata を公開し、configured language が含まれない
- input encoding が `FLOAT32LE` ではない
- payload byte length 不一致
- empty audio
- NaN / Inf
- normalized range violation
- temporary WAV bridge 作成失敗
- NeMo transcribe exception
- NeMo transcribe result が single transcript として解釈できない
- backend output が empty text
- backend output が invalid `segments_json_v1`
- timeout / non-zero exit / worker crash

失敗時に `nemo_rnnt_streaming`、Whisper、OpenAI、NIM/Riva backend へ fallback しません。

## Research Sources

この backend document で参照する公式一次情報は次です。

- NVIDIA ASR NIM docs
  - https://docs.nvidia.com/nim/speech/latest/asr/index.html
  - NIM が ASR serving container / API 境界であり、local command worker ではないこと。
- NVIDIA ASR NIM support matrix
  - https://docs.nvidia.com/nim/speech/latest/reference/support-matrix/asr.html
  - support matrix capability が NIM/Riva serving stack の model capability であり、local `.nemo` worker success ではないこと。
- NVIDIA ASR NIM deploy docs
  - https://docs.nvidia.com/nim/speech/latest/asr/deploy-asr-models/index.html
  - container id / serving profile / deployment selection が NIM serving の概念であり、この backend の local NeMo restore ではないこと。
- NVIDIA NIM for Speech overview
  - https://docs.nvidia.com/nim/speech/latest/about/index.html
  - NIM を application-facing API server として扱う根拠。
- NVIDIA NeMo ASR model docs
  - https://docs.nvidia.com/nemo/speech/nightly/asr/models.html
  - NeMo ASR model、offline transcription、cache-aware / limited-context streaming を分けて扱う根拠。
- NVIDIA NeMo ASR source docs
  - https://github.com/NVIDIA/NeMo/blob/main/docs/source/asr/models.rst
  - NeMo ASR model family と `transcribe(...)` 系 API の一次参照。
- NVIDIA Riva ASR overview
  - https://docs.nvidia.com/deeplearning/riva/archives/2-24-0/public/asr/asr-overview.html
  - Riva が server / gRPC client / streaming and offline recognition を含む serving stack であること。
- NGC CLI registry docs
  - https://docs.ngc.nvidia.com/cli/cmd_registry.html
  - `ngc registry model download-version <artifact>` が registry から local disk へ artifact を取得する操作であり、format conversion、model restore、worker health、transcript success ではないこと。

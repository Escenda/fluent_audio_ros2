# nemo_rnnt_streaming Backend

## Backend Name

`nemo_rnnt_streaming`

## Status

この backend は local `.nemo` file を JSONL worker process で読み込み、NeMo RNNT / Transducer model の cache-aware streaming inference を行うための backend slot です。

2026-05-22 時点の実機検証では、次の事実が確認されています。

- `fa_asr` から `nemo_rnnt_streaming_worker` を起動できる。
- `nvidia/riva/parakeet-rnnt-riva-1-1b-unified-ml-cs-universal:trainable_v1.0` 由来の `.nemo` file を worker が restore できる。
- worker health check は `cache_aware_streaming=true` として成功する。
- full ROS graph では `fa_vad` が speech start を検出し、`fa_asr` が control window を開き、stream を open できる。
- ただし最初の audio push で worker が `RuntimeError` により終了し、`fa_asr` は `stream_audio_push_failed` として fail closed する。

このため、現状の backend は「model load / health check / ROS graph open までは検証済み」ですが、「実 audio chunk を継続的に受理して partial / final transcript を返す streaming ASR backend」としては未成立です。

## Runtime Boundary

`fa_asr` node 本体は NeMo / PyTorch / Parakeet を import しません。
NeMo 依存は `nemo_rnnt_streaming_worker` process の内部へ閉じ込めます。

境界は JSON Lines protocol です。

- `fa_asr` は worker stdin へ JSON object を 1 行ずつ送る。
- worker は stdout へ JSON object を 1 行ずつ返す。
- NeMo warning / traceback / telemetry message は stderr に出す。
- stdout が閉じた場合、`fa_asr` は stderr を failure reason に含めて fail closed する。

この backend は streaming 専用です。
`transcribe()` による non-streaming request は受け付けません。

## NIM / Riva / NeMo Local Backend の境界

Parakeet RNNT multilingual を FluentAudio で扱うとき、似た名前の技術が複数あります。
これらを混同すると、model artifact は存在するのに streaming backend としては成立しない、という状態を見落とします。

| 領域 | 役割 | FluentAudio での扱い |
| --- | --- | --- |
| NIM | NVIDIA が配布する speech ASR serving container。model download、optimization、serving stack を container 内で持つ。 | 使わない。FluentAudio の `nemo_rnnt_streaming` は NIM server へ gRPC/WebSocket 接続しない。 |
| Riva | NVIDIA の speech serving SDK / model deployment stack。NGC 上に Riva 用 model artifact / RMIR / trainable artifact がある。 | model artifact の入手元として扱う。Riva server が動くことは FluentAudio worker の成功証明ではない。 |
| NeMo local `.nemo` | PyTorch / NeMo で直接 restore できる checkpoint。 | `nemo_rnnt_streaming_worker` が local file として restore し、cache-aware streaming algorithm を直接実行する。 |

NVIDIA Speech NIM docs では Parakeet RNNT Multilingual は 25+ languages、Streaming + Offline、auto language detection を持つ model として扱われています。
NVIDIA Riva docs でも multilingual universal-rnnt は `ja-JP` を含み、streaming multilingual ASR の推奨 model として扱われています。
しかし、それは NIM / Riva serving stack 上の contract です。

FluentAudio の contract は別です。
FluentAudio は local `.nemo` を worker process 内で直接 restore するため、Riva/NIM が内部で行っている model setup、streaming params setup、decoder strategy、cache state 管理、positional encoding length 更新を、FluentAudio 側の backend algorithm として明示的に成立させる必要があります。
したがって「NIM/Riva の model が streaming 対応」という事実だけで、`nemo_rnnt_streaming_worker` を完了扱いしてはいけません。

## NGC Artifact と Local `.nemo` の扱い

現時点で検証対象としている artifact は次です。

```text
nvidia/riva/parakeet-rnnt-riva-1-1b-unified-ml-cs-universal:trainable_v1.0
```

取得は NGC CLI による次の操作を想定します。

```text
ngc registry model download-version nvidia/riva/parakeet-rnnt-riva-1-1b-unified-ml-cs-universal:trainable_v1.0
```

FluentAudio repository では model checkpoint 自体を git 管理対象にしません。
checkpoint は prepare / download 手順で local volume に配置し、backend config から `backend.model_path` として参照します。

現在の検証環境では、次の `.nemo` を使っています。

```text
src/ai/fa_asr/models/nemo_rnnt_streaming/
  parakeet-rnnt-riva-1-1b-unified-ml-cs-universal_vtrainable_v1.0/
    Parakeet-RNNT-XXL-1.1b_merged_universal_spe8.5k_1.0.nemo
```

この `.nemo` の `model_config.yaml` から確認した backend-relevant fact は次です。

| 項目 | 値 | backend 上の意味 |
| --- | --- | --- |
| `sample_rate` | `16000` | `fa_asr` が受け取る ASR-ready audio と `backend.sample_rate_hz` は 16 kHz で一致しなければならない。 |
| `preprocessor.window_stride` | `0.01` | feature frame は 10 ms stride として扱われる。subsampling 後の token duration 計算に使う。 |
| `encoder.subsampling_factor` | `8` | encoder 出力 token は raw feature frame より 8 倍粗い時間解像度になる。runtime chunk と NeMo chunk の対応計算に必須。 |
| `encoder.att_context_size` | `[-1, -1]` | restore 直後は full context 相当。streaming backend としてはこのまま成功扱いしない。finite context の設定可否を検査する。 |
| `encoder.att_context_style` | `regular` | right context / look-ahead と latency の関係を regular attention style として扱う。 |
| `encoder.pos_emb_max_len` | `5000` | restore 直後の positional encoding 長。cache を含む streaming step では不足し得る。 |
| `decoding.strategy` | `greedy_batch` | RNNT decoding strategy の初期値。streaming partial/final semantics と一致するか worker で検査する。 |
| vocabulary | `<en-US>`, `<ja-JP>` を含む | 日本語音声を流す前提と矛盾しない。ただし token の存在は日本語 ASR 成功の証明ではない。 |

ここで重要なのは、`.nemo` config の存在をそのまま runtime contract と見なさないことです。
`att_context_size=[-1,-1]` や `pos_emb_max_len=5000` は、restore 直後の model 状態です。
FluentAudio backend は、この状態から streaming inference に必要な finite context / streaming config / cache / positional encoding length を明示的に成立させる責務を持ちます。

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
- WAV decode / encode
- value range normalization
- missing frame 補完
- VAD / TD / KWS の代替判定
- 別 ASR backend への fallback

未対応形式を受け取った場合は、変換して継続せず fail closed します。

## Required Config

現行実装が要求する設定は次です。

- `backend.command`: `nemo_rnnt_streaming_worker` executable
- `backend.model_path`: local readable `.nemo` file
- `backend.language`: request language
- `backend.sample_rate_hz`: model sample rate と一致する値
- `backend.channels`: `1`
- `backend.chunk_size_samples`: worker JSONL contract 上の audio chunk size
- `backend.emit_partial`: partial result を返すか
- `backend.max_partial_interval_ms`: partial result emission interval
- `backend.timeout_sec`: JSONL response timeout

ただし、Parakeet cache-aware streaming を正しく成立させるには、現行設定だけでは不足しています。
backend algorithm としては、少なくとも次の streaming model configuration を明示的に扱う必要があります。

- attention context size
- maximum audio / positional encoding length
- model window stride
- encoder subsampling factor
- model chunk size
- shift size
- pre-encode cache size
- drop-extra-pre-encoded frames
- feature buffer length
- valid output length
- decoder strategy

これらを暗黙に NeMo default へ任せた状態で full-context model を streaming step に流すと、cache-aware streaming contract が破綻します。

## Local NeMo 2.7.3 Source から見た実行契約

現在の `fluent-audio-runtime` container では NeMo `2.7.3` が使われています。
この backend は NeMo の public high-level description だけではなく、実際に import される local source の contract に従って実装される必要があります。

調査した NeMo 実装上の要点は次です。

| NeMo API | 実装上の意味 | FluentAudio backend が守ること |
| --- | --- | --- |
| `ConformerEncoder.set_max_audio_length(max_audio_length)` | encoder の `max_audio_length` を更新し、positional encoding を `pos_enc.extend_pe(max_audio_length, device, dtype)` で拡張する。 | cache を含む streaming step の必要長を計算し、step 前に不足しない長さを設定する。API が無い場合は fail closed。 |
| `ConformerEncoder.setup_streaming_params(...)` | `CacheAwareStreamingConfig` を作り、`chunk_size`、`shift_size`、`cache_drop_size`、`last_channel_cache_size`、`valid_out_len`、`pre_encode_cache_size`、`drop_extra_pre_encoded` を設定する。 | restore 直後の default に任せず、backend config と model config から effective streaming config を明示的に作る。 |
| `ConformerEncoder.get_initial_cache_state(...)` | `streaming_cfg.last_channel_cache_size` を使って last-channel cache tensor を作る。 | `start` ごとに stream-local cache を初期化し、別 stream と共有しない。 |
| `CacheAwareStreamingAudioBuffer` | `streaming_cfg.chunk_size` / `shift_size` と pre-encode cache を使い、streaming step 用の chunk と length を返す。 | raw audio を単純な固定 sample chunk として `conformer_stream_step` に直結しない。NeMo が要求する feature/cache 境界を尊重する。 |
| `conformer_stream_step(...)` | RNNT/CTC かつ `StreamingEncoder` を前提に `encoder.cache_aware_stream_step(...)` を呼び、cache と hypothesis を更新する。 | previous hypothesis、previous decoder output、encoder cache を stream-local state として保持し、step ごとに更新する。 |

この表は「NeMo がそのように動いているらしい」という推測ではありません。
`fluent-audio-runtime` 内に install されている NeMo 2.7.3 の source と、現在の worker 実装が実際に呼ぶ API から確認した contract です。

特に `set_max_audio_length(...)` は、今回観測した tensor shape mismatch と直結します。
restore 直後の `.nemo` config では `pos_emb_max_len=5000` ですが、実機で観測した `streaming_cfg.last_channel_cache_size` は `10000` でした。
cache-aware attention は current chunk だけではなく過去 cache を含むため、current chunk 長だけを前提にした positional encoding では不足します。
direct investigation では、step 直前に `last_channel_cache_size + chunk_feature_len` 相当を `set_max_audio_length(...)` へ渡すことで、少なくとも最初の `conformer_stream_step(...)` は shape mismatch なしに通りました。

したがって `nemo_rnnt_streaming_worker` の algorithm は、次を必須処理として持ちます。

1. `encoder.streaming_cfg.last_channel_cache_size` を読み取る。
2. 実際に step へ渡す chunk の feature/time length を取得する。
3. `last_channel_cache_size + chunk_feature_len` を minimum required sequence length として計算する。
4. `encoder.set_max_audio_length(...)` で positional encoding を拡張する。
5. 設定できない場合、empty transcript ではなく worker error として fail closed する。

## JSONL Protocol

### health

`fa_asr` は起動時に `health` を送ります。

worker は model を restore し、model capability を検査します。

検査項目は次です。

- model class が RNNT / Transducer である。
- encoder が cache-aware streaming を提供している。
- model sample rate が `backend.sample_rate_hz` と一致する。
- requested language が model language contract に反しない。
- requested partial mode が model capability と矛盾しない。

health が成功した場合、worker は `health_ok` を返します。
health で model が読み込めない、または capability が満たされない場合、backend は起動完了扱いにしません。

### start

`fa_asr` は ASR control window が開くと `start` を送ります。

worker は stream-local state を初期化します。

- active session id
- accepted sample count
- NeMo cache state
- feature / audio buffer
- previous RNNT hypothesis
- previous decoder output
- last partial text

`start` の config は health 時の config と完全一致しなければなりません。
health と start の streaming contract が一致しない場合、worker は stream を開始しません。

### audio

`fa_asr` は `AudioFrame` から取り出した validated float32 samples を `audio` として送ります。

worker は次の順に処理します。

1. `session_id` が active stream と一致することを確認する。
2. `encoding == base64_float32le` を確認する。
3. `sample_count * channels * 4` と payload byte length が一致することを確認する。
4. float32 sample が finite かつ `[-1.0, 1.0]` 内であることを確認する。
5. streaming buffer へ sample を追加する。
6. buffer が inference step を実行できる長さになった場合、cache-aware RNNT step を実行する。
7. partial text があり、`backend.emit_partial=true` の場合は `partial` を返す。
8. audio chunk を受理したことを `audio_accepted` として返す。

重要なのは、`audio_accepted` は transcript 成功ではなく「worker がこの chunk を streaming state に受理した」という意味だけを持つ点です。
backend は `audio_accepted` と `partial` / `final` を混同してはいけません。

### drain

`drain` は現時点の partial hypothesis を取り出すための command です。
新しい audio sample を追加しません。

streaming model がまだ text を返せない場合は、空 partial を成功扱いしてはいけません。
返すべき text がない場合は `drained` のみを返します。

### finish

`finish` は active stream の finalization です。

worker は受理済み audio sample が 1 つ以上存在することを確認し、buffer に残った推論可能部分を処理し、final transcript を返します。

final transcript が空の場合の扱いは backend policy として明確にする必要があります。
現行 `NemoRnntStreamingSession` は、final が空で、かつ直前に non-empty partial が存在する場合、その partial を final commit に使います。
これは streaming commit semantics であり、ASR model output の推測補正ではありません。
この仕様を使う場合は、最後に観測した non-empty partial を final result として commit する条件を明示する必要があります。

### cancel

`cancel` は active stream を破棄します。
worker は cache / buffer / accepted sample count / previous hypothesis を破棄し、以降の `audio` / `finish` を受け付けません。

## NeMo Cache-Aware Streaming Algorithm

Parakeet RNNT streaming は、単に audio samples を `model.conformer_stream_step()` に渡せば成立する処理ではありません。

cache-aware streaming では、少なくとも次の状態が互いに整合している必要があります。

- raw audio chunk length
- preprocessor window stride
- encoder subsampling factor
- model chunk size
- shift size
- pre-encode cache size
- attention context
- encoder cache tensors
- RNNT previous hypothesis
- decoder strategy

NeMo 側の streaming implementation は、restore 直後の model をそのまま呼ぶ前提ではなく、streaming 用の model setup を行ってから step 実行します。

### Phase-by-Phase Backend Algorithm

`nemo_rnnt_streaming_worker` は、JSONL command を受け取る単なる wrapper ではありません。
local `.nemo` を direct restore する以上、NIM/Riva serving stack が内部で行う初期化を worker 自身の algorithm として所有します。
各 phase の責務は次です。

| phase | 入力 | 必須処理 | 失敗時 |
| --- | --- | --- | --- |
| model restore | readable `.nemo` path | `ASRModel.restore_from(...)` で model を読み込み、stdout ではなく stderr 側へ NeMo log を逃がす。 | JSONL protocol を壊す前に worker error。 |
| config extraction | restored model config | `sample_rate`、`window_stride`、`subsampling_factor`、`att_context_size`、`pos_emb_max_len`、decoder strategy、language token を取得する。 | 不足を default で埋めない。capability failure。 |
| capability validation | extracted config + backend config | RNNT/Transducer、StreamingEncoder、cache-aware API、sample rate、mono/float32 contract を検証する。 | `health_ok` を返さない。 |
| finite attention setup | encoder + requested streaming policy | full context `[-1,-1]` をそのまま streaming 成功扱いせず、finite context を設定できるか確認する。 | unsupported streaming context として fail closed。 |
| streaming params setup | window stride / subsampling / chunk config | `setup_streaming_params(...)` を使い、effective `chunk_size`、`shift_size`、cache sizes、valid output length を確立する。 | default に任せて継続しない。 |
| health response | effective capability | model class、sample rate、effective streaming config、partial support を返す。 | health success を audio success と混同しない。 |
| start | session id + same backend config | stream-local audio buffer、encoder cache、previous hypotheses、previous decoder output、accepted sample count を初期化する。 | health と config が違えば start を拒否する。 |
| audio validation | base64 float32 payload | byte length、sample count、finite/normalized range、active session を検証する。 | 変換せず reject。 |
| audio buffering | validated samples | `CacheAwareStreamingAudioBuffer` に追加し、推論可能な chunk だけ取り出す。 | 足りない chunk は transcript なしで accepted とする。 |
| streaming step | chunk + cache state | `set_max_audio_length(...)`、`conformer_stream_step(...)`、cache/hypothesis 更新を行う。 | tensor mismatch は empty text ではなく worker error。 |
| partial emission | step result | non-empty text かつ `emit_partial=true` の場合だけ partial を返す。 | 空 partial を成功 transcript として扱わない。 |
| drain | active session | 新規 audio を追加せず、直近の non-empty partial 状態を返す。 | active session 不一致は reject。 |
| finish | accepted audio + buffer state | 残りの推論可能 chunk を処理し、final transcript を返す。 | accepted audio なし、empty final policy 不成立は fail closed。 |
| cancel | active session | cache、buffer、previous hypothesis、accepted sample count を破棄する。 | 破棄後の audio/finish は拒否する。 |

この phase 分解は、debug のためだけではありません。
どの phase が失敗したのかを `fa_asr` の event / result / trace で説明できるようにするための backend contract です。
たとえば `health` が通って `audio` で落ちる場合、model file や ROS graph ではなく streaming step の成立条件を疑うべきです。

### Model Restore

worker は `.nemo` を `ASRModel.restore_from()` で読み込みます。

restore 後に最低限確認するものは次です。

- model class が `EncDecRNNTBPEModel` など RNNT 系である。
- `model.encoder` が streaming encoder である。
- `model.encoder.streaming_cfg` が存在する。
- `model.encoder.get_initial_cache_state(batch_size=1)` が使用可能である。
- `model.conformer_stream_step(...)` または同等の cache-aware inference API が使用可能である。

`streaming_cfg` が存在するだけでは十分ではありません。
実際に `audio` command で 1 step 以上の cache-aware inference が成功することを、代表検証で確認する必要があります。

### Positional Encoding Length

実機調査では、Parakeet `.nemo` の `encoder.streaming_cfg.last_channel_cache_size` が `10000` でした。
cache-aware attention は current chunk だけではなく、過去 cache を含む sequence length を扱います。
そのため positional encoding も current chunk 長だけでは足りません。

現行 worker は `CacheAwareStreamingAudioBuffer` から得た chunk を `model.conformer_stream_step(...)` に渡していますが、step 直前に encoder の maximum audio / positional encoding length を cache 長込みで更新していません。
このため attention 内で current chunk 側の長さと cache を含む長さが一致せず、次のような mismatch が発生しました。

```text
RuntimeError: The size of tensor a (10001) must match the size of tensor b (5001) at non-singleton dimension 3
```

追加調査では、worker と同じ NeMo API を直接呼び、`encoder.set_max_audio_length(last_channel_cache_size + chunk_feature_len)` 相当を streaming step 直前に呼ぶと、少なくとも最初の `conformer_stream_step(...)` は tensor shape mismatch なしに完了しました。

したがって、`nemo_rnnt_streaming_worker` は streaming model setup の一部として、次を明示的に扱う必要があります。

1. `encoder.streaming_cfg.last_channel_cache_size` を取得する。
2. 実際に step へ渡す chunk / feature length を取得する。
3. `last_channel_cache_size + chunk_length` を minimum required maximum length として計算する。
4. encoder が `set_max_audio_length(...)` を提供しない場合は fail closed する。
5. step 前に maximum length を設定し、設定不能なら transcript を空にせず error として扱う。

これは音声補正ではなく、NeMo cache-aware encoder を streaming mode で呼び出すための推論 backend 内部契約です。
ROS node、VAD、pipeline profile、upstream processing node に漏らしてはいけません。

### Attention Context

実機で確認した Parakeet `.nemo` の restore 直後の `att_context_size` は `[-1, -1]` でした。
これは full context を意味するため、そのまま streaming step に使うと cache-aware streaming の intent と矛盾します。

NeMo 同梱の streaming service 例では、streaming service 側で finite attention context を設定します。
代表例は `[70, 1]` です。

ただし、この値を FluentAudio が無条件に固定してよいわけではありません。
backend algorithm としては、次のどちらかを明示する必要があります。

- model が support する streaming attention context を読み取り、その範囲内で明示設定する。
- 必要な finite attention context を model が support しない場合、起動時に fail closed する。

`[-1, -1]` のまま streaming backend として成功扱いすることは禁止です。
これは「動いているように見えるが意味が壊れている」状態を生むためです。

### Streaming Params

NeMo の cache-aware streaming は、model chunk と runtime chunk を分けて扱います。

実機で確認した model の restore 直後の `streaming_cfg` は次のような形でした。

- `chunk_size=[1009, 1016]`
- `shift_size=[1, 8]`
- `cache_drop_size=126`
- `last_channel_cache_size=10000`
- `valid_out_len=1`
- `pre_encode_cache_size=[0, 9]`
- `drop_extra_pre_encoded=2`

`last_channel_cache_size=10000` のような full-context 相当の値は、streaming runtime としては疑って扱う必要があります。
finite attention context を設定して `setup_streaming_params()` をやり直すと、cache size は変化します。

NeMo の streaming service 例では、次のような考え方を使います。

1. `window_stride = model.cfg.preprocessor.window_stride`
2. `subsampling_factor = model.cfg.encoder.subsampling_factor`
3. `chunk_size_in_secs` から `tokens_per_frame` を計算する。
4. model chunk size を subsampling factor で割り、encoder streaming chunk size として渡す。
5. `encoder.setup_streaming_params(chunk_size=..., shift_size=...)` を呼ぶ。

したがって FluentAudio の backend 設定も、単なる `chunk_size_samples` だけでは足りません。
少なくとも algorithm document 上は、`chunk_size_samples` が NeMo の `chunk_size_in_secs` / `tokens_per_frame` / `shift_size` とどう対応するかを定義する必要があります。

### Feature Buffer

現行 worker は `CacheAwareStreamingAudioBuffer` を使って raw audio を buffer し、得られた chunk を `model.conformer_stream_step()` に渡します。

一方、NeMo 同梱の voice-agent streaming service は、raw audio から feature buffer を作り、その feature buffer を `encoder.cache_aware_stream_step()` に渡します。

この違いは重要です。
Parakeet 系の streaming model では、preprocessor window、look-back、pre-encode cache、feature buffer length が推論 step の tensor shape と直結します。
そのため、raw audio buffer を使うのか、feature buffer を使うのかを backend algorithm として決め、決めた方式に対して tensor shape contract を検証する必要があります。

現時点の実機失敗は、feature / context / cache shape の不整合を示しています。
追加調査により、少なくとも positional encoding / maximum audio length の不足が直接の failure path であることが分かっています。

## Observed Failure

full ROS graph 検証では、次の順序までは到達しました。

1. `fa_in` が `hw:CARD=S3,DEV=0` から audio を取得した。
2. `fa_sample_format` が `PCM16LE` から `FLOAT32LE` へ変換した。
3. `fa_resample` が 16 kHz へ変換した。
4. `fa_dc_offset_removal` / `fa_high_pass` が ASR-ready stream を生成した。
5. `fa_vad` が `voice/vad_state` で `start=true` を publish した。
6. `fa_asr` が control window を開いた。
7. `fa_asr` が worker stream を open した。
8. 最初の audio push で worker が終了した。

`AsrResult` は次でした。

```text
status: STATUS_ERROR
reason: stream_audio_push_failed
text: ""
```

`AsrEvent` では次が観測されました。

```text
EVENT_CONTROL_WINDOW_OPENED
EVENT_STREAM_OPENED
EVENT_STREAM_ERROR
EVENT_FAIL_CLOSED
```

worker stderr の本質的な error は次です。

```text
RuntimeError: The size of tensor a (10001) must match the size of tensor b (5001) at non-singleton dimension 3
```

これは audio device、ROS topic、VAD control、TurnContext の問題ではありません。
`nemo_rnnt_streaming_worker` 内部の cache-aware streaming step contract が未成立であることを示す failure です。
追加の単体調査では、`pad_and_drop_preencoded` の切り替えだけでは同系統の mismatch は解消せず、`encoder.set_max_audio_length(...)` 相当を呼んだ場合に最初の streaming step が通ることを確認しました。

## Required Correction Direction

修正は `fa_asr` node ではなく、`nemo_rnnt_streaming_worker` の model runner 層で行います。

責務境界は次です。

- `fa_asr` node: topic control、AudioFrame validation、timeline、AsrEvent / AsrResult publish
- `NemoRnntStreamingAsrBackend`: JSONL protocol、worker lifecycle、health / start / audio / drain / finish / cancel contract
- `nemo_rnnt_streaming_worker`: NeMo model restore、streaming model setup、cache-aware inference algorithm
- upstream processing node: resample、channel conversion、sample format conversion

`fa_asr` node に NeMo model setup を持ち込んではいけません。
`nemo_rnnt_streaming_worker` が NeMo-specific algorithm を所有します。

修正時に必要な algorithm steps は次です。

1. `.nemo` restore 後に model class と streaming capability を検査する。
2. RNNT decoder strategy を streaming 用に明示設定する。
3. finite attention context を明示設定する。設定できない場合は fail closed する。
4. `window_stride`、`subsampling_factor`、model chunk size、requested runtime chunk duration から shift size を決める。
5. `encoder.setup_streaming_params(...)` を呼び、実際の `streaming_cfg` を health response または trace に記録する。
6. `start` で encoder cache、feature/audio buffer、previous hypothesis を初期化する。
7. `audio` で chunk を buffer に入れ、推論可能な step だけ処理する。
8. step 直前に cache 長込みの maximum audio / positional encoding length を encoder に設定する。
9. `partial` は non-empty text がある場合だけ返す。
10. `finish` で残りを処理し、final transcript を返す。
11. tensor shape mismatch、unsupported context、unsupported decoder、empty accepted audio は fail closed する。

## Verification Requirements

この backend は、次の順に検証しなければ完了扱いしません。

### 1. Worker Health

`health` が成功し、次を確認します。

- model class
- sample rate
- cache-aware streaming support
- configured attention context
- effective streaming config
- partial support

health success は model load の証跡であり、streaming ASR 成功の証跡ではありません。

### 2. Worker Minimal Streaming Step

ROS を使わず、worker 単体で次を流します。

```text
health
start
audio
audio
finish
```

この検証では、実 audio が無音でもよいですが、少なくとも worker が落ちずに `audio_accepted` と `final` を返す必要があります。

### 3. Worker Speech Fixture

既知の短い speech fixture を使い、partial または final に non-empty transcript が出ることを確認します。
日本語 model として使う場合は、日本語 speech fixture を使います。

### 4. Full ROS Graph

次の graph で実音声を流し、VAD start/end と ASR result を確認します。

```text
fa_in
  -> fa_sample_format
  -> fa_resample
  -> fa_dc_offset_removal
  -> fa_high_pass
  -> fa_vad
  -> fa_asr
```

確認対象は次です。

- `voice/vad_state`: `start=true` / `end=true`
- `voice/asr/event`: `EVENT_STREAM_OPENED` / `EVENT_STREAM_AUDIO_PUSHED` / `EVENT_STREAM_FINAL_RESULT` / `EVENT_STREAM_CLOSED`
- `voice/asr/result`: `STATUS_FINAL` and non-empty `text`

`STATUS_ERROR`、empty final、backend crash は未完了です。

## Failure Conditions

次の場合、この backend は fail closed します。

- `backend.command` missing / not executable
- `backend.model_path` missing / unreadable
- model is not RNNT / Transducer
- model sample rate mismatch
- channels is not `1`
- audio encoding is not `FLOAT32LE`
- input sample is NaN / Inf / outside `[-1.0, 1.0]`
- cache-aware streaming API unavailable
- finite attention context cannot be established
- streaming params cannot be established
- encoder maximum audio / positional encoding length cannot be established
- tensor shape mismatch in streaming step
- worker stdout closes before JSONL response
- health response does not match backend config
- start config differs from health config
- audio / drain / finish session id differs from active session
- finish is called before any accepted audio

失敗時に Whisper、OpenAI、local_command、別 Parakeet worker へ切り替えません。

## Research Sources

この backend document は、次の情報源に基づいて更新しています。

- NVIDIA NeMo Framework ASR Models docs
  - https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/models.html
  - Parakeet が FastConformer encoder と CTC/RNNT/TDT decoder の model family であること。
  - Cache-aware Streaming Conformer が Conformer-CTC / Conformer-Transducer を support し、cache によって重複計算を避けること。
  - full-context checkpoint を limited-context streaming inference に使う場合、chunk size / latency / accuracy tradeoff があること。
- NVIDIA Speech NIM ASR docs
  - https://docs.nvidia.com/nim/speech/latest/asr/index.html
  - Parakeet RNNT Multilingual が 25+ languages、Streaming + Offline、auto language detection を持つこと。
  - real-time streaming model として Parakeet RNNT Multilingual が挙げられていること。
- NVIDIA Riva ASR docs
  - https://docs.nvidia.com/deeplearning/riva/archives/2-24-0/public/asr/asr-overview.html
  - Multilingual universal-rnnt が `ja-JP` を含む複数言語を support し、streaming multilingual ASR 推奨 model として扱われていること。
- local `fluent-audio-runtime` NeMo 2.7.3 source
  - `ConformerEncoder.set_max_audio_length(...)`
  - `ConformerEncoder.setup_streaming_params(...)`
  - `ConformerEncoder.get_initial_cache_state(...)`
  - `CacheAwareStreamingAudioBuffer`
  - `conformer_stream_step(...)`
- local `.nemo` `model_config.yaml`
  - sample rate、window stride、subsampling factor、attention context、positional embedding length、decoder strategy、language tokens。

外部 docs は model capability と NVIDIA serving stack の説明として使います。
FluentAudio backend の完了判定は、local worker の `health -> start -> audio -> finish` と full ROS graph の non-empty transcript 検証に基づきます。

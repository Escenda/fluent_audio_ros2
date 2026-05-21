# AudioFrame Stream Contract PO 設計資料

## 位置づけ

この資料は、FluentAudioROS2 における `fa_interfaces/msg/AudioFrame` の
stream accounting / stream contract を定義するための Product Owner 設計資料である。

これは node 仕様書そのものではない。
この資料は、今後 `fa_interfaces` の仕様書、アルゴリズム説明、テスト設計、
および `fa_sample_format`、`fa_resample`、`fa_frame_buffer`、`fa_vad`、`fa_asr`
などの node 仕様へ落とし込むための上位契約である。

現時点では、この資料の作成によって `AudioFrame.msg` や各 node 実装が完了したことを意味しない。
message 変更、node 実装、テスト、代表検証は後続作業である。

## 目的

FluentAudio の pipeline では、音声 frame が複数の node を通る。
ただし、VAD と ASR は単純な直列関係ではない。

悪い表現:

```text
fa_in
  -> fa_sample_format
  -> fa_resample
  -> fa_frame_buffer
  -> fa_vad
  -> fa_asr
```

この表現では、`fa_vad` が音声 data を変換して `fa_asr` に渡すように見える。
それは FluentAudio の責務境界として正しくない。

VAD は音声区間を判定する node であり、ASR に音声 data を渡す node ではない。
ASR は、自分の backend contract に合う audio stream または audio window を読む。
VAD の結果は、ASR を実行するかどうか、どの時間範囲を読むかを決める side signal として扱う。

正しい表現:

```text
fa_in
  -> fa_sample_format
       -> fa_resample_for_vad
            -> fa_frame_buffer_for_vad
            -> fa_vad
                 -> VadState / speech range

       -> fa_resample_for_asr
            -> fa_frame_buffer_for_asr
            -> fa_audio_window
                 -> fa_asr
```

VAD と ASR が同じ sample rate を要求する場合は、途中の stream を共有してもよい。
しかし、VAD が 16kHz、ASR が 48kHz を要求する場合は、用途別 stream に分岐する。

```text
source stream
  -> normalized source stream
       -> vad input stream, e.g. 16kHz
       -> asr input stream, e.g. 48kHz
```

この場合、VAD の `start_sample` を ASR の `start_sample` として直接使ってはいけない。
VAD の sample index は VAD stream の sample domain に属し、
ASR の sample index は ASR stream の sample domain に属する。

ASR が読む範囲は、VAD result の sample range から直接作るのではなく、
いったん source media time range または resolved time range に解決し、
それと同等の時間範囲を ASR stream 側の sample range として再解決する。

この pipeline では、音声 data だけでなく、時間軸と sample 範囲も正しく流れる必要がある。
`AudioFrame.data` が存在していても、それがどの stream の、どの時点の、どの sample 範囲なのかが
曖昧であれば、後段 node は正しい判断ができない。

この資料で扱う stream contract の目的は、次の状態を検出できるようにすることである。

- frame が欠落した。
- frame が重複した。
- frame の順序が入れ替わった。
- sample 範囲が overlap した。
- sample 範囲が rewind した。
- 同じ stream の途中で sample rate / channel / format が変わった。
- `fa_resample` 後の出力 sample 範囲が入力 stream と説明できない。
- `fa_frame_buffer` の chunk が、どの入力範囲から作られたのか追えない。
- VAD が検出した発話範囲と同等の範囲を、ASR stream 側で安全に切り出せない。

重要なのは、壊れた状態を勝手に直すことではない。
壊れた状態を、壊れた状態として見えるようにすることである。

FluentAudio が必要としているのは、信用を保証する台帳ではない。
必要なのは、**ズレたら分かる帳簿**である。

## 非目的

この契約は、銀行 system や blockchain のような仕組みではない。

次のものは非目的である。

- 改ざん不能性
- hash chain
- 署名
- cryptographic integrity
- 永続的な監査台帳
- 取引履歴の真正性保証
- 音声 data の所有権証明

`AudioFrame Stream Contract` は、音声 stream の実務的な帳尻合わせである。
近い考え方は、sequence number、sample counter、transport manifest、
RTP timestamp のような media timeline accounting である。

「あとから誰かに証明するための台帳」ではなく、
「pipeline の中で、前後の node が互いに矛盾を検出するための帳簿」である。

## 現状

現在の `AudioFrame.msg` は次の field を持つ。

```text
std_msgs/Header header
string source_id
string stream_id
string encoding
uint32 sample_rate
uint32 channels
uint32 bit_depth
string layout
uint8[] data
uint32 epoch
```

現状でも、source identity、stream identity、format metadata は表現できる。
そのため、次のような検証は可能である。

- `source_id` が空でないこと。
- `stream_id` が空でないこと。
- `stream_id` が node の `input_stream_id` と一致すること。
- `sample_rate` が期待値と一致すること。
- `channels` が期待値と一致すること。
- `encoding` / `bit_depth` / `layout` が期待値と一致すること。
- `data` の byte size が format と整合すること。
- `epoch` によって stream restart の単位を表すこと。

一方で、現在の `AudioFrame` には stream continuity と sample range を表す field が足りない。

不足しているものは、少なくとも次である。

| field | 不足によって起きる問題 |
| --- | --- |
| `seq` | frame の欠落、重複、順序入れ替わりを検出しにくい。 |
| `start_sample` | frame が stream timeline のどこから始まるか分からない。 |
| `frame_count` | `data.size()` から推測できるが、契約として明示されていない。 |

`data.size()` から frame 数を計算することはできる。
しかし、それは payload の長さを読んでいるだけであり、
stream timeline 上の位置を表していない。

`header.stamp` も時間 anchor として使えるが、sample accounting の代替にはならない。
timestamp には clock の揺れ、publish 遅延、device capture 時刻と ROS publish 時刻の差がある。
sample の連続性は、sample counter と frame count で検証する必要がある。

## 提案する AudioFrame contract field

将来的な `AudioFrame` は、少なくとも次の contract を持つべきである。

| field | 種別 | 意味 |
| --- | --- | --- |
| `header.stamp` | time anchor | capture または publish の時刻 anchor。sample accounting の代替ではない。 |
| `source_id` | identity | device / app / loopback など、音声 source の identity。 |
| `stream_id` | identity | logical media stream identity。ROS topic 名ではない。 |
| `epoch` | stream generation | stream restart / device change / clock reset / continuity break の世代。 |
| `seq` | stream accounting | `stream_id + epoch` 内で publish された `AudioFrame` の連番。 |
| `start_sample` | stream accounting | `stream_id + epoch` の sample domain における先頭 sample-frame index。 |
| `frame_count` | stream accounting | この frame が持つ per-channel sample-frame 数。byte count ではない。 |
| `sample_rate` | format contract | 1 秒あたりの sample-frame 数。 |
| `channels` | format contract | channel 数。 |
| `encoding` | format contract | PCM sample representation。例: `PCM16LE`, `FLOAT32LE`。 |
| `bit_depth` | format contract | 1 sample あたりの bit depth。 |
| `layout` | format contract | `interleaved` などの channel layout。 |
| `data` | payload | contract と一致する waveform payload。 |

`duration_ns` を `AudioFrame` に直接持たせるかは未決とする。

基本方針として、duration は次で導出できる。

```text
duration_seconds = frame_count / sample_rate
```

そのため、`AudioFrame` 本体に `duration_ns` を重複して持たせると、
`frame_count / sample_rate` と `duration_ns` が矛盾する可能性がある。

現段階の推奨は、`AudioFrame` では duration を派生値として扱い、
必要な node が diagnostics や side message で duration を出す形である。
ただし、network / encoded payload / media container との境界で `duration_ns` が必要になる場合は、
`EncodedAudioChunk` や専用 message 側で扱う。

## field semantics

### `source_id`

`source_id` は音声がどこから来たかを表す。

例:

- `mic/front`
- `mic/realsense`
- `system/playback`
- `app/browser`
- `loopback/main`

`source_id` は ROS topic 名ではない。
同じ source から複数 topic へ publish されても、source identity は変わらない。

### `stream_id`

`stream_id` は logical media stream identity である。

例:

- `audio/raw/mic`
- `audio/float32/mic`
- `audio/preprocessed/mono16k`
- `audio/buffered/mic`
- `audio/asr/input`

`stream_id` は ROS topic 名ではない。
ROS topic は transport の経路であり、`stream_id` は音声 data の意味上の identity である。

topic 名を `stream_id` の代わりに使うと、remap、namespace、launch profile の変更で
stream identity が壊れる。
そのため、FluentAudio では topic と stream identity を分ける。

### `epoch`

`epoch` は `stream_id` 内の世代である。

次のような場合、同じ `stream_id` を使い続けるとしても `epoch` を進める。

- device が切り替わった。
- stream が restart した。
- clock contract が reset された。
- sample continuity を保持できなくなった。
- buffer flush によって過去との連続性を保証できなくなった。
- output path が stop / resume で別世代になった。

`epoch` は、continuity break を隠すためのものではない。
continuity break を「ここから別世代」と明示するための field である。

### `seq`

`seq` は `stream_id + epoch` 内の `AudioFrame` 連番である。

基本規則:

- 最初の frame は `seq = 0` を推奨する。
- 次の frame は `previous.seq + 1` である。
- frame が drop された場合、後段が gap を検出できるようにする。
- 同じ `stream_id + epoch + seq` を別 frame として再利用しない。

`seq` は sample 数ではない。
`seq` は frame packet の連番である。

### `start_sample`

`start_sample` は、`stream_id + epoch` の sample domain における
この `AudioFrame` の先頭 sample-frame index である。

ここでいう sample-frame は、channel をまとめた時間方向の 1 sample を指す。
stereo interleaved の `[L0, R0, L1, R1]` では、sample-frame は 2 個である。
interleaved element 数は 4 であり、sample-frame 数とは違う。

基本規則:

```text
next.start_sample == current.start_sample + current.frame_count
```

連続していない場合は gap または overlap である。
node はそれを暗黙に補正しない。

### `frame_count`

`frame_count` は、この `AudioFrame` が持つ per-channel sample-frame 数である。
byte 数ではない。
interleaved element 数でもない。

`data.size()` は次と整合しなければならない。

```text
data.size() == frame_count * channels * bytes_per_sample
```

`bytes_per_sample` は `encoding` と `bit_depth` から決まる。
node が対応していない encoding / bit depth の場合、推測で処理しない。
未対応 contract として reject する。

### `sample_rate`

`sample_rate` は `stream_id + epoch` 内では変わらない。

同一 `stream_id + epoch` の途中で sample rate が変わる場合、
それは stream contract violation である。

sample rate を変える場合は、`fa_resample` のような明示的な node が
別 `stream_id` の出力として publish する。

### `header.stamp`

`header.stamp` は時刻 anchor である。
ただし、sample continuity の唯一の根拠にしてはいけない。

`header.stamp` が必要なのは、ROS graph 上の他 sensor、camera、robot state などと
大まかな時刻整合を取るためである。

音声の連続性は `seq`、`start_sample`、`frame_count`、`sample_rate` で検証する。

## invariant

`AudioFrame` は、少なくとも次の invariant を満たす必要がある。

| ID | invariant |
| --- | --- |
| `FA-AUDIOFRAME-CONTRACT-001` | `source_id` は non-empty。 |
| `FA-AUDIOFRAME-CONTRACT-002` | `stream_id` は non-empty。 |
| `FA-AUDIOFRAME-CONTRACT-003` | `sample_rate > 0`。 |
| `FA-AUDIOFRAME-CONTRACT-004` | `channels > 0`。 |
| `FA-AUDIOFRAME-CONTRACT-005` | `encoding` は受信 node が明示的に support する値。 |
| `FA-AUDIOFRAME-CONTRACT-006` | `bit_depth` は `encoding` と一致し、受信 node が support する値。 |
| `FA-AUDIOFRAME-CONTRACT-007` | `layout` は受信 node が明示的に support する値。 |
| `FA-AUDIOFRAME-CONTRACT-008` | `frame_count > 0`。 |
| `FA-AUDIOFRAME-CONTRACT-009` | `data.size() == frame_count * channels * bytes_per_sample`。 |
| `FA-AUDIOFRAME-CONTRACT-010` | 同じ `stream_id + epoch` で `sample_rate / channels / encoding / bit_depth / layout` は変わらない。 |
| `FA-AUDIOFRAME-CONTRACT-011` | contiguous frame では `next.seq == current.seq + 1`。 |
| `FA-AUDIOFRAME-CONTRACT-012` | contiguous frame では `next.start_sample == current.start_sample + current.frame_count`。 |
| `FA-AUDIOFRAME-CONTRACT-013` | 同じ `stream_id + epoch` で sample range を silent overlap しない。 |
| `FA-AUDIOFRAME-CONTRACT-014` | 同じ `stream_id + epoch` で sample range を silent rewind しない。 |
| `FA-AUDIOFRAME-CONTRACT-015` | sample rate 変更は、別 `stream_id` または別 `epoch` として扱う。 |

## violation handling

contract violation の扱いは、node の責務と severity によって変わる。
ただし、共通して次は禁止する。

- hidden fallback
- guessed timestamp
- missing `seq` の自動補完
- missing `start_sample` の自動補完
- `data.size()` だけを根拠に timeline を作ったことにする処理
- sample rate 変更を同一 stream の継続として扱う処理
- unsupported encoding を別 encoding とみなす処理
- gap / overlap を warning だけで成功扱いする処理

許容される扱いは、明示的なものに限る。

| 状況 | 扱い |
| --- | --- |
| 起動時 config が invalid | `std::runtime_error` などで fail closed。 |
| runtime frame の format が unsupported | frame drop + warning / diagnostics。 |
| continuity が必須の node で `seq` / `start_sample` が missing | explicit error / frame rejection。 |
| gap を許容する設計の node | gap として diagnostics に出し、後段が分かる形にする。 |
| overlap / rewind | 原則 reject。自動 trim しない。 |
| sample rate 変更 | 同一 stream では reject。別 stream / epoch として扱う。 |

ここで重要なのは、失敗を消すことではない。
失敗の場所を後段と operator が理解できるようにすることである。

## node-by-node transformation rules

### `fa_in`

`fa_in` は source boundary である。
DSP、AI、format conversion を隠して実装しない。

責務:

- device / loopback / app source から audio payload を受け取る。
- `source_id` を決める。
- raw input の `stream_id` を決める。
- `epoch` を管理する。
- `seq` を発行する。
- `start_sample` を monotonic に進める。
- `frame_count` を payload から計算し、明示する。
- source の sample rate / channels / encoding / bit depth / layout を明示する。

`fa_in` が continuity を維持できなくなった場合、
同じ `epoch` のまま何事もなかったように続けてはならない。
`epoch` を進めるか、stream failure として明示する。

### `fa_sample_format`

`fa_sample_format` は sample representation だけを変換する node である。
resample、channel conversion、gain、normalize、denoise、AI 推論を隠して実装しない。

保存する field:

- `source_id`
- 入力 sample domain を維持する場合の `epoch`
- `seq`
- `start_sample`
- `frame_count`
- `sample_rate`
- `channels`
- `layout`
- `header.stamp`

更新する field:

- `stream_id`
- `encoding`
- `bit_depth`
- `data`

変換前後で sample 数は変わらない。
そのため、`start_sample` と `frame_count` は保持できる。

`data.size()` は、出力 encoding / bit depth / channels / frame_count と一致しなければならない。
未対応変換は fail closed または frame rejection であり、暗黙変換しない。

### `fa_resample`

`fa_resample` は sample rate だけを変換する format processing node である。

保存する field:

- `source_id`
- `header.stamp` を time anchor として引き継ぐ。ただし resampling delay の意味は diagnostics に出す。

更新する field:

- `stream_id`
- `sample_rate`
- `start_sample`
- `frame_count`
- `data`

`fa_resample` の出力 `start_sample` は、出力 stream の sample domain で表す。
入力 `start_sample` をそのままコピーしてはいけない。
48kHz 入力の sample 48000 と 16kHz 出力の sample 16000 は同じ数値体系ではない。

resampling では、入力範囲と出力範囲の対応が ratio と backend state によって決まる。
丸め誤差、flush、filter delay、algorithmic delay が発生し得る。

そのため、`fa_resample` は次を明示する必要がある。

- 出力 `start_sample` は出力 sample domain に属する。
- 出力 `frame_count` は実際に生成した output frame 数である。
- 入力 frame count と期待 output frame count の差は diagnostics に出す。
- algorithmic delay は sample range identity とは別に report する。
- backend の rounding error は隠さず、metrics / diagnostics に出す。

`fa_resample` は sample rate 変更 node なので、
入力 stream と出力 stream は別 `stream_id` とするのが基本である。
同じ `stream_id` のまま sample rate を変えてはならない。

### `fa_frame_buffer`

`fa_frame_buffer` は可変長 frame を固定長 chunk にまとめる streaming node である。
format conversion、resample、DSP、AI 推論を隠して実装しない。

入力 contract:

- 入力 frame は同一 `source_id / stream_id / epoch / sample_rate / channels / encoding / bit_depth / layout` である。
- `seq` と `start_sample` が連続していることを検証する。
- gap / overlap / rewind を silent に混ぜない。

出力 contract:

- 出力 `stream_id` は `output.stream_id` に変わる。
- 出力 `start_sample` は出力 stream の sample domain で表す。
- sample rate は変わらないため、基本的には入力 sample domain と 1:1 で対応できる。
- 出力 `frame_count` は `frames_per_chunk` と一致する。
- partial chunk は publish しない。
- padding しない。

`fa_frame_buffer` の output chunk は複数 input frame から作られる可能性がある。
そのため、将来的には output chunk に対して、どの input range が寄与したかを
diagnostics または side message で説明できる必要がある。

初期実装で `AudioFrame` に input range reference を持たせない場合でも、
node 内 diagnostics と test では、input range -> output chunk の対応を検証する。

### `fa_vad`

`fa_vad` は音声区間を判定する AI / analysis node である。
format conversion、resample、channel conversion を隠して実装しない。

入力 contract:

- backend が要求する sample rate / channels / encoding / bit depth / layout と一致すること。
- `stream_id / epoch / start_sample / frame_count` によって判定対象範囲が分かること。

出力 contract:

- `VadState` は判定元 `AudioFrame.source_id` / `stream_id` / `epoch` を保持するべきである。
- 判定対象の sample range を表せるようにするべきである。
- 後段 consumer は topic だけで VAD state を信じず、audio stream identity と一致するか検証する。
- `VadState` は ASR stream の sample index を直接指定しない。
- VAD result は、ASR が読む範囲を決めるための発話区間 signal であり、音声 payload ではない。

`fa_vad` は missing accounting を勝手に推測してはいけない。
continuity が必要な backend で accounting が欠けている場合は explicit error とする。

### `fa_asr`

`fa_asr` は ASR backend へ audio window / stream を渡し、文字起こし結果を publish する AI node である。
resample、downmix、sample format conversion を暗黙に行わない。

入力 contract:

- ASR backend が要求する sample rate / channels / encoding / bit depth / layout と一致すること。
- ASR に渡す window が `stream_id + epoch` 内で連続していること。
- gap がある window を、連続音声として backend に渡さないこと。
- VAD topic を使う場合、VAD の発話区間を resolved time range に変換し、ASR stream 側で同等の sample range を切り出すこと。
- VAD stream と ASR stream が異なる場合、VAD sample index を ASR sample index として直接使わないこと。

出力 contract:

- `AsrResult` / `TranscriptSegment` は、文字起こし対象の stream identity と time/sample range を参照できるべきである。
- partial / final transcript は、どの audio range から得られたものか説明できるべきである。

ASR backend が streaming に対応している場合でも、
FluentAudio 側では input stream accounting を失ってはならない。
backend protocol の timestamp と `AudioFrame` の sample accounting は別契約として扱う。

## resampling range mapping

resampling では、入力と出力の sample domain が異なる。

例:

```text
input:  48000 Hz, start_sample=48000, frame_count=480
output: 16000 Hz, start_sample=16000, frame_count=160
```

この例では、おおよそ同じ時間範囲を表しているが、
sample index の数値体系は違う。

`fa_resample` は、入力 `start_sample` を出力 `start_sample` にそのままコピーしない。
出力 stream の sample counter として再計算する。

基本式:

```text
output_time_start = input_start_sample / input_sample_rate
output_start_sample ~= output_time_start * output_sample_rate
```

ただし、実際には streaming resampler の内部 state、filter delay、rounding、flush によって
単純な式だけでは説明できない場合がある。

そのため、`fa_resample` の contract は次を要求する。

- output sample domain を明示する。
- output frame count は実際に publish した sample-frame 数にする。
- ideal output frame count と実 output frame count の差を diagnostics に出す。
- algorithmic delay は別 metrics として出す。
- rounding error を隠さない。
- backend ごとの delay / quality tradeoff を docs と metrics で説明する。

## VAD / ASR alignment

VAD は ASR の音声 payload 前段ではない。
VAD は、ASR 実行可否と ASR 対象範囲を決めるための side signal を publish する。

基本形:

```text
audio stream for vad
  -> fa_vad
       -> VadState / speech range

audio stream for asr
  -> fa_audio_window
       -> fa_asr
```

`fa_asr` または ASR session manager は、VAD topic を購読し、
発話開始、発話中、発話終了を判断する。
ただし、VAD が publish した sample range を ASR stream の sample range として直接使わない。

VAD result は次の範囲を指す。

```text
vad_stream_id
vad_epoch
vad_start_sample
vad_frame_count
vad_sample_rate
```

ASR が読むべき範囲は次である。

```text
asr_stream_id
asr_epoch
asr_start_sample
asr_frame_count
asr_sample_rate
```

VAD stream と ASR stream が同一であれば、range を直接使える場合がある。
ただし、それでも `stream_id`、`epoch`、`sample_rate`、`frame_count` が一致することを検証する。

VAD stream と ASR stream が異なる場合は、次の順序で解決する。

```text
VAD sample range
  -> source media time range / resolved time range
  -> ASR stream sample range
```

例:

```text
VAD:
  stream_id = audio/vad_input/16k
  sample_rate = 16000
  start_sample = 16000
  frame_count = 320

Resolved range:
  media time = [1.000s, 1.020s)

ASR:
  stream_id = audio/asr_input/48k
  sample_rate = 48000
  start_sample = 48000
  frame_count = 960
```

この変換では、整数 ns、sample boundary、resampling delay、rounding が問題になる。
そのため、resolved range は単なる float 秒ではなく、uncertainty と rounding rule を持つ契約として扱う。

ASR は、次を満たす場合だけ backend に audio を渡す。

- VAD が示した発話区間と同等の ASR stream range を解決できる。
- ASR stream range が retention window 内に存在する。
- ASR stream range が gap なく連続している。
- ASR backend の sample rate / channel / encoding / bit depth / layout と一致する。
- VAD result の source identity が ASR 対象 source と一致する。

解決できない場合、ASR は空音声、近い範囲、別 stream、zero padding で代用しない。
`range_not_continuous`、`range_outside_window`、`window_not_found` などの明示 error として扱う。

## frame buffering / windowing range mapping

`fa_frame_buffer` は、複数の input frame から 1 つの output chunk を作る。

例:

```text
input A: start_sample=0,   frame_count=160
input B: start_sample=160, frame_count=160
input C: start_sample=320, frame_count=160
input D: start_sample=480, frame_count=160

output chunk: start_sample=0, frame_count=512
```

この場合、output chunk は input A-D の一部または全部から作られる。
chunk 境界は input frame 境界と一致するとは限らない。

したがって `fa_frame_buffer` は、次を守る。

- input sample range の連続性を検証する。
- output sample range を明示する。
- partial chunk を publish しない。
- padding しない。
- gap を埋めたことにしない。
- overlap を trim して成功扱いしない。

input range reference を `AudioFrame` に入れるか、diagnostics / side message に分けるかは未決である。
ただし、どちらの形でも、test は input range から output chunk が正しく作られることを検証する。

## diagnostics direction

stream contract violation は operator と後段 node が分かる形で出す必要がある。

diagnostics は少なくとも次を表せるべきである。

| field | 意味 |
| --- | --- |
| `stream_id` | 対象 stream。 |
| `epoch` | 対象 epoch。 |
| `last_seq` | 直前に受理した seq。 |
| `received_seq` | 今回受け取った seq。 |
| `expected_start_sample` | 期待した start sample。 |
| `received_start_sample` | 実際に受け取った start sample。 |
| `violation_type` | `seq_gap`, `sample_gap`, `overlap`, `rewind`, `format_change` など。 |
| `action` | `drop`, `fail_closed`, `reset_epoch_required` など。 |

diagnostics schema は未決である。
既存の `/diagnostics` topic に string key/value として出すか、
専用 message を作るかは、後続設計で決める。

## test design direction

テストは、source text や Markdown を読むためのものではない。
テストは contract の性質を検証するためのものである。

少なくとも次の property test / unit test / graph test が必要である。

| test | 検証する性質 |
| --- | --- |
| contiguous frames pass | `seq` と `start_sample` が連続する frame を受理する。 |
| seq gap detected | `seq` が飛んだ場合に gap として検出する。 |
| start_sample gap detected | `start_sample` が飛んだ場合に gap として検出する。 |
| overlap detected | `start_sample` が過去範囲と重なる場合に reject する。 |
| rewind detected | `start_sample` が巻き戻る場合に reject する。 |
| format change rejected | 同一 `stream_id + epoch` で sample rate / channel / encoding が変わる場合に reject する。 |
| byte-size mismatch rejected | `data.size()` と `frame_count * channels * bytes_per_sample` が一致しない場合に reject する。 |
| sample format preserves range | `fa_sample_format` が `start_sample / frame_count` を保持する。 |
| resample recalculates output range | `fa_resample` が出力 sample domain の `start_sample / frame_count` を出す。 |
| resample rounding bounded | resample の丸め差が定義範囲内で diagnostics に出る。 |
| frame buffer range output | `fa_frame_buffer` が連続入力から正しい chunk range を作る。 |
| frame buffer rejects mixed stream | 別 stream / format / epoch を混ぜて chunk を作らない。 |
| VAD controls ASR by range alignment | VAD result が ASR 実行可否を決め、同等の time range を ASR stream 側で解決する。 |
| ASR rejects direct cross-domain sample index | VAD stream と ASR stream が違う場合、VAD sample index を ASR sample index として使う経路を reject する。 |
| ASR rejects unresolved speech range | VAD 発話区間に対応する ASR stream range が無い場合に明示 error を返す。 |
| ASR rejects discontinuous window | gap を含む audio window を連続音声として ASR に渡さない。 |

テストの中心は、実行経路と public contract である。
source file の import 文字列、Markdown の自然言語、package.xml の存在だけを検査するテストは、
この contract の証明にならない。

## implementation staging

この contract は、段階的に導入する。

### Stage 1: PO design

この資料を作成し、stream accounting の目的、非目的、field semantics、invariant、
node ごとの責務を固定する。

完了条件:

- この資料が存在する。
- 非目的として改ざん不能性や blockchain 的性質を明確に除外している。
- `AudioFrame.msg` の現状と不足 field が明記されている。
- 後続作業の未決事項が明記されている。

### Stage 2: interface design

`fa_interfaces` の仕様書、アルゴリズム説明、テスト設計へ落とし込む。
`AudioFrame.msg` に field を追加する場合、ROS2 message 生成と dependent package build への影響を確認する。

候補 field:

```text
uint64 seq
uint64 start_sample
uint32 frame_count
```

型は未決である。
長時間 stream では `start_sample` は `uint64` が必要になる。
`frame_count` は単一 message 内の frame 数なので `uint32` で足りる可能性が高いが、
最大 payload 設計と一致させる必要がある。

### Stage 3: shared validator

複数 node が同じ validation を重複実装しないように、
shared contract validator / helper を作るか検討する。

ただし、汎用 helper が責務を曖昧にするなら作らない。
helper は node の責務を隠すためではなく、同じ invariant を同じ言葉で検証するために使う。

候補:

- `fa_interfaces` に validation utility を置くか。
- `fluent_audio_common` のような共通 package を作るか。
- 各 C++ package 内で最小実装し、後から統合するか。

これは未決である。

### Stage 4: processing node adoption

まず次の node で adoption する。

- `fa_sample_format`
- `fa_resample`
- `fa_frame_buffer`

理由:

- これらは `AudioFrame` を入力し、`AudioFrame` を出力する。
- sample range の保存、変換、集約が明確に発生する。
- 後段 AI node の信頼性の土台になる。

### Stage 5: AI node consumption

次に、AI node が stream accounting を消費する。

- `fa_vad`
- `fa_asr`
- `fa_kws`
- `fa_turn_detector`

AI node は、backend に渡す audio が連続しているかを検証する。
backend が要求する format と一致しない場合、暗黙変換しない。
必要な変換は前段 processing node で明示する。

### Stage 6: pipeline-level verification

代表 pipeline で contract を検証する。

```text
fa_in
  -> fa_sample_format
       -> fa_resample_for_vad
            -> fa_frame_buffer_for_vad
            -> fa_vad
                 -> VadState / speech range

       -> fa_resample_for_asr
            -> fa_frame_buffer_for_asr
            -> fa_audio_window
                 -> fa_asr
```

確認すること:

- `stream_id` が node ごとに意味のある identity として変化している。
- `epoch` が不必要に変化していない。
- `seq` gap が検出される。
- `start_sample` と `frame_count` が破綻しない。
- `fa_resample` 後の output sample domain が説明できる。
- `fa_frame_buffer` が gap を含む window を作らない。
- VAD stream と ASR stream が異なる場合、VAD sample range を resolved time range 経由で ASR sample range に変換できる。
- `fa_asr` が VAD topic を ASR 実行可否と window selection に使い、VAD を音声 payload の前段 node として扱わない。
- `fa_asr` が discontinuous window を連続音声として扱わない。

## open decisions

未決事項は次である。

| ID | 内容 | 判断が必要な理由 |
| --- | --- | --- |
| `OD-001` | `duration_ns` を `AudioFrame` に追加するか。 | `frame_count / sample_rate` と重複し、矛盾 field になる可能性がある。 |
| `OD-002` | input range reference を `AudioFrame` に持たせるか。 | `fa_frame_buffer` や `fa_resample` の説明力は上がるが、message が複雑になる。 |
| `OD-003` | stream violation diagnostics を既存 `/diagnostics` に載せるか、専用 message にするか。 | operator/debug の使いやすさと package 依存が変わる。 |
| `OD-004` | missing `seq` / `start_sample` を全 node で即 reject するか、package ごとに staged adoption するか。 | message 変更直後は既存 node との移行衝突が大きい。後方互換を残さない方針との整理が必要。 |
| `OD-005` | validator helper をどの package に置くか。 | interface package は runtime dependency を持たない方がきれいだが、重複実装も避けたい。 |
| `OD-006` | `seq` の drop policy をどう表現するか。 | drop を gap として後段に見せるか、drop 済み frame を diagnostics のみに残すかで consumer の扱いが変わる。 |
| `OD-007` | `epoch` の increment owner をどこまで node ごとに許すか。 | source boundary 以外でも buffer reset / resampler reset が continuity break になる場合がある。 |
| `OD-008` | VAD speech range を表す message を `VadState` 拡張にするか、別 message にするか。 | VAD の bool state と発話区間 range contract を同じ message に混ぜるべきか判断が必要。 |
| `OD-009` | VAD stream range から ASR stream range への alignment owner をどの node にするか。 | `fa_asr`、`fa_audio_window`、session manager のどこが resolved time range を責任を持って作るか決める必要がある。 |

## PO decision summary

現時点の Product Owner 判断は次である。

- この仕組みは banking / blockchain ではない。
- 改ざん不能性、hash chain、署名、監査台帳は不要である。
- 必要なのは stream accounting / stream contract である。
- `AudioFrame` は waveform payload だけでなく、stream continuity を検証できる metadata を持つべきである。
- `header.stamp` は必要だが、sample accounting の代替ではない。
- `seq`、`start_sample`、`frame_count` は追加候補として強い。
- sample rate 変更は同一 `stream_id + epoch` の継続として扱わない。
- gap / overlap / rewind は暗黙補正しない。
- `fa_sample_format` は range を保存する。
- `fa_resample` は出力 sample domain として range を再計算する。
- `fa_frame_buffer` は入力 range から出力 chunk range を説明できるようにする。
- `fa_vad` は ASR へ音声 payload を渡す前段 node ではなく、ASR 実行可否と発話区間を示す side signal を出す。
- `fa_asr` は VAD topic を読み、VAD が特定した発話区間と同等の ASR stream 範囲を解決してから文字起こしする。
- VAD stream と ASR stream が異なる場合、sample index を直渡しせず、resolved time range を経由する。
- `fa_vad` / `fa_asr` は、連続していない audio window や解決不能な発話範囲を連続音声として扱わない。
- 実装完了と設計資料作成を混同しない。

この資料をもって、次の作業は `fa_interfaces` の正式仕様・message 変更影響調査・node adoption 計画へ進める。

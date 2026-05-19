# FluentAudio Engineering Philosophy

この資料は、`fluent_audio_ros2` に関わる人間とエージェントが共有する、設計・実装・テストの美学です。
ここに書いてあることは、単なる気分や好みではありません。
FluentAudio を、長く育てられる音声システムにするための考え方です。

`CPP_CODING_RULES.md` や `CLAUDECODE_RULES.md` は、具体的な禁止事項や作業手順を定めるルールです。
この資料は、そのルールの奥にある「なぜそうするのか」を説明します。

この資料は、小学生にも伝わるくらい、できるだけ素直な日本語で書きます。
難しい言葉を使うときは、それが何を守るための言葉なのかを説明します。

## 0. 哲学だけを語る

この章では、実装の話をしません。
テストの種類の話もしません。
directory の話もしません。
backend の話もしません。
launch の話もしません。

ここでは、FluentAudio が何であってほしいのかだけを語ります。

FluentAudio は、音を扱うための code の集まりです。
しかし、それだけではありません。

FluentAudio は、音を受け取り、音を整え、音を理解し、音を届けるための社会です。
その社会の中に、たくさんの node がいます。

node は、ただの処理単位ではありません。
node は、役割を持って生まれてきます。
役割を持っているということは、期待されているということです。
期待されているということは、信頼される可能性を持っているということです。

信頼される可能性を持って生まれたものを、雑に扱ってはいけません。

一つの node がいます。
その node には、自分の仕事があります。
まだ小さな仕事かもしれません。
ただ音を少し整えるだけかもしれません。
ただ判断結果を一つ出すだけかもしれません。
ただ次の node に音を渡すだけかもしれません。

でも、その仕事には意味があります。

小さな仕事でも、誰かがそれを頼りにします。
次の node が頼りにします。
profile が頼りにします。
robot が頼りにします。
人間が頼りにします。
未来の engineer が頼りにします。
未来の agent が頼りにします。

だから、小さい node ほど丁寧に扱うべきです。

大きなものだけが尊いのではありません。
目立つものだけが大事なのでもありません。
入口で静かに frame を渡す node も、途中で音を整える node も、最後に音を届ける node も、それぞれに意味があります。

FluentAudio の美しさは、特別な一つの node がすべてを支配することではありません。
それぞれの node が、それぞれの場所で、自分の責務を果たすことにあります。

社会とは、そういうものです。

一人だけが全部を背負う社会は、長く続きません。
誰かが何でも引き受けてしまう社会は、最初は楽に見えても、あとで苦しくなります。
何を誰がしているのかわからない社会は、壊れたときに直せません。
責任の場所が見えない社会は、信頼を失います。

FluentAudio は、そういう社会になってはいけません。

FluentAudio の node は、自分の役割を知っているべきです。
自分の役割を知っているから、安心して働けます。
自分の役割を知っているから、他の node を尊重できます。
自分の役割を知っているから、できないことをできないと言えます。

できないことをできないと言えるのは、弱さではありません。
それは誠実さです。

できないのにできるふりをする node は、周りを危険にします。
失敗しているのに成功したふりをする node は、次の node を迷わせます。
壊れた入力を受け取って、壊れていないように扱う node は、社会全体に嘘を流します。

嘘は、最初は小さく見えます。
でも、音声 system の中では、嘘は流れていきます。
一つの嘘が、次の判断をゆがめます。
ゆがんだ判断が、さらに次の判断をゆがめます。
最後には、どこで間違えたのかわからなくなります。

だから FluentAudio は、嘘を嫌います。

FluentAudio における正しさとは、いつも成功することではありません。
FluentAudio における正しさとは、成功したときに成功したと言い、失敗したときに失敗したと言えることです。

失敗を言える node は、美しいです。
自分の限界を言える node は、美しいです。
受け取れないものを受け取れないと言える node は、美しいです。
まだ検証していないことを未検証と言える node は、美しいです。

その美しさは、派手ではありません。
でも、長く続く system には、その地味な美しさが必要です。

FluentAudio は、短い demo のためだけに存在するのではありません。
今日だけ動けばよいものではありません。
今この瞬間の都合だけで、責務を混ぜてよいものでもありません。

FluentAudio は、育つものです。
今日生まれた node が、明日も使われます。
明日使われた node が、来月 refactor されます。
来月 refactor された node が、別の robot で使われます。
別の robot で使われた node が、まだ会ったことのない engineer に読まれます。

そのとき、その node が何を守っているのかがわかる必要があります。
その node が何を拒むのかがわかる必要があります。
その node が何を約束しているのかがわかる必要があります。

約束のない node は、孤独です。
誰にも本当の姿を理解されません。
何を変えてよいのかも、何を変えてはいけないのかもわかりません。

約束のある node は、孤独ではありません。
仕様がその node を支えます。
テストがその node を支えます。
レビューがその node を支えます。
profile がその node の居場所を示します。
他の node が、その node の出力を信じられます。

FluentAudio は、node を独りにしないための community です。

community であるなら、そこには礼儀があります。
礼儀とは、きれいな言葉を使うことではありません。
相手の責務を奪わないことです。
相手が頼る contract を壊さないことです。
相手に嘘の data を渡さないことです。
相手が困ったときに原因を追えるようにしておくことです。

node 同士にも礼儀があります。

前段の node は、後段の node が受け取れるものを渡します。
受け取れないものを渡すなら、それは pipeline の設計が間違っています。
後段の node は、受け取れないものを受け取ったとき、黙って飲み込みません。
それは自分と相手を守るためです。

守るということは、甘やかすことではありません。
壊れたものを壊れていると言うことも、守ることです。
進んではいけない場所で止まることも、守ることです。
できないことをできないと言うことも、守ることです。

FluentAudio の優しさは、何でも受け入れることではありません。
FluentAudio の優しさは、意味が壊れる前に止めることです。

意味が壊れた音は、ただの data ではありません。
それは、後段の判断を間違わせるものです。
それは、robot の行動を間違わせるかもしれないものです。
それは、人間との対話を壊すかもしれないものです。

だから、音を扱うことは責任です。

音は見えません。
見えないものは、雑に扱われやすいです。
でも、見えないからこそ、丁寧に扱わなければなりません。

音がどこから来たのか。
どんな形だったのか。
どこで変わったのか。
どこで判断されたのか。
どこで拒まれたのか。
どこへ届けられたのか。

それが見えるようにすることは、FluentAudio の倫理です。

倫理という言葉は、難しく聞こえるかもしれません。
ここで言いたいことは単純です。

次の人を困らせない。
次の node をだまさない。
未来の自分に嘘を残さない。
できないことを隠さない。
失敗を見える形で残す。

それだけです。

でも、それだけを本当に守るのは簡単ではありません。
だから、資料に書きます。
何度でも読み返せるようにします。
Context Compact のあとでも戻ってこられるようにします。
迷ったときに、ここへ戻ってこられるようにします。

FluentAudio は、正直であることを選びます。
FluentAudio は、責務を分けることを選びます。
FluentAudio は、証明することを選びます。
FluentAudio は、未検証を未検証と言うことを選びます。
FluentAudio は、node を独りにしないことを選びます。

この選択が、FluentAudio の哲学です。

この哲学があるから、細かな実装ルールに意味が生まれます。
この哲学があるから、テストが尊いものになります。
この哲学があるから、書類がただの文章ではなくなります。
この哲学があるから、レビューがただの指摘ではなくなります。

FluentAudio のすべての仕事は、この哲学に戻ってきます。

正直であること。
責務を守ること。
他者をだまさないこと。
未来に嘘を残さないこと。
node を独りにしないこと。

ここから始めます。

## 1. いちばん大事な考え

FluentAudio の node は、正直でなければなりません。

正直な node とは、次のような node です。

- 自分ができることを、できると言う。
- 自分ができないことを、できないと言う。
- 入力がおかしいときに、おかしいと言う。
- 必要な model や device がないときに、ないと言う。
- 検証していないことを、検証済みだと言わない。
- 失敗しているのに、成功したふりをしない。

これは、人間の仕事でも同じです。
知らないことを知っているふりをすると、次の人が間違えます。
できないことをできるふりをすると、あとで大きな事故になります。
音声システムでも同じです。

たとえば、ASR が本当は動いていないのに、空文字の認識結果を「成功」として返すとします。
後段の会話システムは、「ユーザーは何も言わなかった」と判断するかもしれません。
でも本当は、ASR が壊れていただけです。
この違いは、とても大きいです。

FluentAudio では、「動いているように見せる」ことより、「壊れた状態を壊れていると伝える」ことを大切にします。
これが fail closed の心です。

## 2. 美しい node とは何か

美しい node は、よく働く node ではありません。
何でもできる node でもありません。
美しい node は、自分の役割をよく知っていて、その役割からはみ出さない node です。

美しい node は、次を守ります。

- 自分の仕事だけをする。
- 他の node の仕事を勝手に引き受けない。
- 入力の条件をはっきり書く。
- 出力の形をはっきり書く。
- 対応していないものを、勝手に対応済みに変えない。
- 失敗の理由を、後段が判断できる形で出す。
- 実装、仕様、テスト、検証結果の間に矛盾を作らない。

反対に、見た目だけ便利な node は危険です。

たとえば、AI node に `PCM16LE` が来たとき、node の中で勝手に `FLOAT32LE` に変換して処理する実装は、一見便利に見えます。
しかし、その変換は本来 `fa_sample_format` の仕事です。
AI node が勝手に変換してしまうと、pipeline を見た人は、どこで format が変わったのかわからなくなります。
わからない場所で音が変わると、VAD、KWS、ASR、Turn Detector の結果も信用できなくなります。

美しい node は、そういうことをしません。
AI node が `FLOAT32LE` しか受け付けないなら、`FLOAT32LE` だけ受け付けます。
`PCM16LE` が来たら、明示的に拒否します。
必要な変換は、pipeline に `fa_sample_format` を入れて行います。

これは冷たい実装ではありません。
これは正直な実装です。

## 3. 責務境界を守る

責務境界とは、「誰が何を担当するか」の線です。
この線があいまいになると、システムはすぐに読みにくくなります。
読みにくいシステムは、直しにくくなります。
直しにくいシステムは、壊れたときに止められなくなります。

FluentAudio では、責務境界を次のように考えます。

### fa_in

`fa_in` は source adapter です。
つまり、入力源から音を取って、`AudioFrame` として publish する入口です。

`fa_in` は、次のようなことを担当します。

- どの source backend を使うかを明示する。
- microphone、file、network などの入力源を開く。
- 入力源から受け取った音声 data を `AudioFrame` として出す。
- source_id、stream_id、sample rate、channel count などの metadata を付ける。
- 入力源が開けない場合は、起動失敗または明示的な error にする。

`fa_in` は、次を隠してはいけません。

- resample
- downmix
- bit depth conversion
- sample format conversion
- gain
- normalize
- denoise
- VAD
- KWS
- ASR
- Turn Detector

これらは別の node の仕事です。

### fa_out

`fa_out` は sink adapter です。
つまり、音声 frame を受け取り、speaker、file、network などの出力先へ出す出口です。

`fa_out` は、次のようなことを担当します。

- どの sink backend を使うかを明示する。
- speaker、file、network などの出力先を開く。
- 指定された `AudioFrame` contract に合う音声を出力する。
- 出力先が使えない場合は、起動失敗または明示的な error にする。

`fa_out` は、次を隠してはいけません。

- encode
- decode
- resample
- normalize
- limiter
- mixer
- routing
- TTS
- barge-in control

これらも別の node の仕事です。

### processing

`src/processing` は、音を加工する場所です。
format conversion、dynamics、frequency、temporal、correction、spatial、routing、generation などをここに置きます。

processing node は、「音をどう変えたか」が名前と責務からわかるべきです。
たとえば、sample format を変えるなら `fa_sample_format`。
resample するなら `fa_resample`。
high-pass filter なら `fa_high_pass`。
ducking なら `fa_ducking`。

音が変わる処理は、見える場所に置きます。

### streaming

`src/streaming` は、リアルタイム伝送を成立させる場所です。
jitter buffer、clock drift correction、packet loss concealment、latency compensation、time alignment、chunk overlap、overlap-add などをここに置きます。

これは音質そのものというより、「途切れない」「ずれない」「遅れすぎない」ための処理です。
source adapter や sink adapter に隠すと、あとで原因が追えなくなります。

### ai

`src/ai` は、音を意味へ近づける model node の場所です。
VAD、KWS、ASR、Turn Detector、SED、speaker、audio embedding などをここに置きます。

AI node は、model を呼ぶことと、結果を publish することに集中します。
音声 format を勝手に合わせる node ではありません。
必要な format は、前段の processing node で明示的に作ります。

### backend

backend は、ROS から切り離された実処理の境界です。
device、file、network、DSP engine、model runtime、外部 API、worker、process、container などを backend に閉じ込めます。

backend code は、ROS topic を知りません。
ROS message も知りません。
`rclcpp` や `rclpy` も知りません。

ROS node は、parameter、topic、service、lifecycle、message conversion を担当します。
backend は、純粋な処理や外部接続を担当します。

この分離があるから、backend は unit test しやすくなります。
ROS node は graph test しやすくなります。
問題が起きたとき、どちらが悪いのか分けて考えられます。

## 4. FluentAudio の真髄: 全部を持ち、混ぜない

FluentAudio の真髄はここにあります。

FluentAudio は、ただ microphone から音を取って ASR に渡すだけの repository ではありません。
speaker に音を出すだけの repository でもありません。
「音を、現実のロボットや AI が扱える形にするための土台」です。

現実の音は、きれいではありません。
sample rate が違います。
channel 数が違います。
音量が小さすぎることがあります。
大きすぎて割れることもあります。
低いノイズが混じります。
50Hz や 60Hz の hum が乗ります。
speaker から出た音が microphone に戻ります。
network では遅延が揺れます。
device の clock は少しずつずれます。
AI model は、入力 format にとても敏感です。

だから FluentAudio は、音声対話のための薄い adapter では足りません。
実用に耐える音声システムとして、音の入口から出口までを支える必要があります。

ただし、ここで絶対に間違えてはいけないことがあります。

「全部を持つ」ことと、「全部を一つに混ぜる」ことは違います。

FluentAudio は全部を持ちます。
でも、全部を一つの巨大 node に混ぜません。

これは FluentAudio のいちばん大事な設計思想です。

全部入りとは、巨大な万能 node を作ることではありません。
全部入りとは、必要な分類を、それぞれ責務のはっきりした node として持てることです。
そして、それらを pipeline、launch、profile、app orchestration で組み合わせられることです。

音を変える処理は、見える場所に置きます。
音を判断する処理も、見える場所に置きます。
音を届ける処理も、見える場所に置きます。

見えるということは、あとで調べられるということです。
調べられるということは、壊れたときに直せるということです。
直せるということは、product として育てられるということです。

### 4.1 なぜ全部入りが必要なのか

小さな demo なら、microphone から取った音をそのまま ASR に入れても動くかもしれません。
静かな部屋で、同じ microphone を使い、同じ PC で、同じ sample rate で、同じ model を使うなら、それでも動くかもしれません。

しかし、FluentAudio が支えるのは demo だけではありません。
ロボットがあります。
speaker があります。
microphone があります。
network があります。
VLAbor profile があります。
AI model があります。
VAD があります。
KWS があります。
ASR があります。
Turn Detector があります。
将来的には VLM や高度な LLM ともつながります。

この世界では、音声 pipeline はすぐ複雑になります。

たとえば、ASR model が `FLOAT32LE/16000Hz/mono` を要求しているとします。
でも microphone は `PCM16LE/48000Hz/stereo` かもしれません。
このとき、ASR node の中で勝手に変換すれば、短期的には楽です。
でも、その変換は見えなくなります。

あとで KWS も同じ音を使いたいとき、KWS はどの format を受け取るのでしょうか。
VAD は resample 前の音を見るのでしょうか。
resample 後の音を見るのでしょうか。
AGC は VAD の前でしょうか。
後でしょうか。
noise reduction は sample rate 変換の前でしょうか。
後でしょうか。
それぞれの node が勝手に処理を隠すと、答えがわからなくなります。

FluentAudio は、その混乱を許しません。

必要な処理は必要な node として明示します。
pipeline 上に見えるようにします。
どの順番で音が変わるのかを、graph と config から追えるようにします。

これが全部入りの意味です。

### 4.2 FluentAudio が扱うべき 10 の領域

FluentAudio が扱うべき領域は、少なくとも次の 10 種類です。

これは単なる一覧ではありません。
音声 system を現実に成立させるための地図です。

#### 1. format conversion

format conversion は、「音の中身をなるべく変えずに、表現を変える」領域です。

たとえば次です。

- sample rate を変える。
- bit depth を変える。
- channel count を変える。
- interleaved と planar を変える。
- PCM と codec を変える。
- int16 と float32 の値域を変える。

これは地味ですが、ものすごく大事です。
AI model は、受け取れる format が決まっていることが多いです。
speaker device も、出せる format が決まっていることがあります。
network codec も、扱える format が決まっています。

format conversion が曖昧だと、後段の node は「自分が何を受け取っているのか」がわからなくなります。

だから FluentAudio では、format conversion を隠しません。
`fa_sample_format`、`fa_resample`、`fa_channel_convert` のように、明示された node が担当します。

#### 2. dynamics

dynamics は、「音の大きさや振れ幅を扱う」領域です。

たとえば次です。

- gain
- normalize
- compressor
- limiter
- expander
- noise gate
- AGC

microphone の音は、小さすぎることがあります。
逆に大きすぎて割れることもあります。
人が近づいたり離れたりすると、音量は変わります。
robot の近くでは、motor 音や環境音もあります。

dynamics を雑に扱うと、VAD は発話を見逃します。
ASR は誤認識します。
KWS は wake word を取り逃します。

だから dynamics は、ただ音量を上げ下げするだけの軽い処理ではありません。
後段の AI が正しく判断できるように、音の振る舞いを整える重要な層です。

#### 3. frequency

frequency は、「どの周波数を残し、どの周波数を削るか」を扱う領域です。

たとえば次です。

- EQ
- high-pass filter
- low-pass filter
- band-pass filter
- notch filter
- de-esser
- spectral subtraction
- Wiener filter

microphone には低い振動音が入ることがあります。
電源由来の hum が入ることもあります。
高いノイズが刺さることもあります。

human が聞いて少し気になる程度のノイズでも、AI model には大きな影響を与えることがあります。
VAD が noise を speech と誤判定するかもしれません。
ASR が誤認識するかもしれません。

frequency 処理は、音色を整えるだけではありません。
AI の入口を整えるための重要な処理です。

#### 4. temporal

temporal は、「時間方向に音をどう扱うか」を扱う領域です。

たとえば次です。

- trim
- silence removal
- time stretch
- pitch shift
- delay
- echo
- reverb
- fade
- crossfade
- windowing

音声 AI では、時間方向の扱いがとても大事です。
短い frame に分ける必要があります。
overlap させる必要があります。
window をかける必要があります。
発話の前後を切り出す必要があります。
silence をどう扱うかも重要です。

temporal 処理を隠すと、turn の境界がわからなくなります。
ASR に渡した音声が、どの時間範囲の音なのか追えなくなります。

だから temporal 処理も、pipeline 上で明示します。

#### 5. correction / noise

correction / noise は、「入力音に含まれる欠陥を補正する」領域です。

たとえば次です。

- denoise
- echo cancellation
- dereverberation
- declip
- debreath
- declick
- wind noise reduction
- hum removal
- DC offset removal

現実の音は、必ず汚れています。
speaker から出た音が microphone に戻ることがあります。
部屋の反響があります。
風があります。
電源 noise があります。
波形の中心がずれることもあります。

この層は、音を「後段が扱いやすい状態」に戻すためのものです。
特に robot では重要です。
robot 自体が音を出し、動き、環境の中で microphone を使うからです。

#### 6. spatial / channel

spatial / channel は、「音がどこから来たか」「channel をどう扱うか」を扱う領域です。

たとえば次です。

- pan
- stereo widening
- downmix
- upmix
- beamforming
- source separation
- binaural rendering
- ambisonics

robot や会議室 microphone では、microphone が一つとは限りません。
複数の microphone から、どの方向の声を強めるかが重要になります。
speaker と microphone の位置関係も重要です。

単に mono にすればよい、という話ではありません。
どの channel をどう混ぜたのか。
どの方向を見たのか。
どの音源を残したのか。
それがわかる必要があります。

#### 7. analysis / feature extraction

analysis / feature extraction は、「音を判断や model 入力に使える表現へ変える」領域です。

たとえば次です。

- VAD
- onset detection
- pitch estimation
- tempo / beat tracking
- STFT
- Mel spectrogram
- MFCC
- CQT
- loudness measurement
- speaker embedding
- audio embedding

ここでは、音を人間が聞くために変えるというより、model や判定器が使える形にします。

VAD は、発話しているかを判断します。
STFT や Mel spectrogram は、波形を周波数と時間の表現にします。
speaker embedding は、誰が話しているかの特徴を作ります。

これは AI の入口です。
だから、入力 contract が曖昧ではいけません。
どの sample rate で、どの channel で、どの window で、どの range の値を使うのかを明示します。

#### 8. generation / transformation

generation / transformation は、「入力から別の音や表現を作る」領域です。

たとえば次です。

- TTS
- voice conversion
- speech enhancement
- speech separation
- speech-to-speech translation
- neural codec
- neural vocoder
- super-resolution

ここは、従来の DSP だけでなく、neural audio processing も含みます。
入力音を整えるだけではなく、新しい音や新しい表現を作ります。

生成系は強力です。
だからこそ、責務を混ぜてはいけません。
TTS が音を生成するなら、TTS は生成を担当します。
limiter は limiter node が担当します。
speaker 出力は `fa_out` が担当します。

#### 9. routing / mixing

routing / mixing は、「音の通り道を扱う」領域です。

たとえば次です。

- mixer
- bus routing
- sidechain
- ducking
- monitor mix
- loopback
- patchbay

複数の入力があります。
複数の出力があります。
TTS があります。
microphone があります。
loopback があります。
recording があります。
network streaming があります。

これらを場当たり的につなぐと、すぐ破綻します。

robot が話している間だけ microphone 側の扱いを変える。
TTS 中だけ BGM を下げる。
speaker 出力を loopback として AEC に渡す。
recording 用と monitoring 用で別 mix を作る。

こういうことを明示的に扱うのが routing / mixing です。
音そのものの加工というより、信号経路の設計です。

#### 10. streaming / synchronization

streaming / synchronization は、「リアルタイムの音声伝送を成立させる」領域です。

たとえば次です。

- buffering
- jitter buffer
- clock drift correction
- packet loss concealment
- latency compensation
- time alignment
- chunk overlap
- overlap-add

network では packet が遅れます。
device の clock は完全には一致しません。
音声と映像を合わせる必要があります。
AI model は chunk 単位で処理することがあります。
frame のつなぎ目が悪いと、音が切れたり、認識が不安定になったりします。

ここを軽く見ると、音質以前に system が成立しません。
「遅れる」「途切れる」「ずれる」が起きます。

だから FluentAudio では、streaming / synchronization も一級の領域として扱います。

### 4.3 全部を持つが、全部を混ぜない

ここが最重要です。

FluentAudio は、上の 10 領域を持つべきです。
しかし、それらを一つの node に詰め込んではいけません。

もし `fa_in` が microphone を開き、resample し、AGC し、denoise し、VAD し、ASR まで呼ぶとします。
一見すると便利です。
起動する node が少なくて済みます。
設定も簡単に見えます。

でも、その便利さは危険です。

どこで sample rate が変わったのか見えません。
どこで channel が mono になったのか見えません。
どこで音量が変わったのか見えません。
どこで noise が削られたのか見えません。
どの音を VAD が見たのかわかりません。
どの音を ASR が見たのかわかりません。

これでは、結果を信用できません。
そして、壊れたときに直せません。

FluentAudio では、便利さを隠蔽で作りません。
便利さは、組み合わせで作ります。

node は小さく、責務ははっきり。
pipeline は見える。
profile は組み合わせを表す。
backend は外部依存を閉じ込める。
app は流れをまとめる。

この形なら、複雑な system でも、何が起きているか追えます。

### 4.4 Pipeline は音の履歴である

pipeline は、ただ node を並べたものではありません。
pipeline は、音がどう扱われたかの履歴です。

たとえば、microphone input を ASR に入れるまでには、次のような pipeline があり得ます。

```text
fa_in
  -> fa_dc_offset_removal
  -> fa_high_pass
  -> fa_noise_reduction
  -> fa_agc
  -> fa_sample_format
  -> fa_resample
  -> fa_frame_buffer
  -> fa_vad
  -> fa_asr
```

この pipeline には、たくさんの情報があります。

`fa_in` は、どこから音を取ったかを示します。
`fa_dc_offset_removal` は、波形の中心ずれを直したことを示します。
`fa_high_pass` は、低域 noise を落としたことを示します。
`fa_noise_reduction` は、背景 noise を抑えたことを示します。
`fa_agc` は、音量を自動調整したことを示します。
`fa_sample_format` は、sample の表現を変えたことを示します。
`fa_resample` は、sample rate を変えたことを示します。
`fa_frame_buffer` は、後段が扱える chunk に分けたことを示します。
`fa_vad` は、発話区間を判断したことを示します。
`fa_asr` は、音声を文字にしたことを示します。

つまり、この pipeline は「音の履歴」です。

どこで音が変わったのか。
どこで意味が付いたのか。
どこで reject されたのか。
どこで遅延が増えたのか。
どこで stream_id が変わったのか。

それを追えることが大事です。

音の履歴が見えると、debug ができます。
音の履歴が見えると、test が書けます。
音の履歴が見えると、profile ごとの差分が説明できます。
音の履歴が見えると、VLAbor から起動したときに何が起きるか説明できます。

これが FluentAudio の作り方です。

### 4.5 Profile は pipeline の設計図である

FluentAudio では、node を一つずつ手で起動するだけでは足りません。
VLAbor profile や FluentAudio system config から、用途に応じた pipeline を起動できる必要があります。

profile は、ただの設定 file ではありません。
profile は、pipeline の設計図です。

たとえば、wake word だけを使う profile なら、ASR や Turn Detector を起動しないかもしれません。
ASR まで使う voice frontend profile なら、VAD、KWS、ASR、必要な processing node を起動するかもしれません。
robot の speaker と microphone が近い profile なら、AEC や loopback が必要かもしれません。
network streaming を含む profile なら、jitter buffer や latency compensation が必要かもしれません。

profile が pipeline を表すなら、責務境界が見えます。
どの node が何をするかがわかります。
どの backend を使うかがわかります。
どの model を要求するかがわかります。
どの input contract を期待するかがわかります。

だから、profile による組み合わせは FluentAudio の重要な機能です。
node の中に処理を隠すのではなく、profile と graph で組み立てます。

### 4.6 入口と出口は大事だが、主役ではない

`fa_in` と `fa_out` は大事です。
入口と出口がなければ、音は system に入りません。
音は system から出られません。

しかし、`fa_in` と `fa_out` は、全部の処理を抱える主役ではありません。

`fa_in` は、入力源を明示して音を入れる。
`fa_out` は、出力先を明示して音を出す。

それ以上の音の変化は、明示された processing node が担当します。
意味の判断は、明示された AI node が担当します。
通り道の設計は、routing / mixing node が担当します。
時間や network の揺れは、streaming node が担当します。

入口と出口は、境界です。
境界は大事です。
でも、境界が中身を飲み込んではいけません。

### 4.7 この思想が守るもの

この「全部を持ち、混ぜない」思想は、いくつものものを守ります。

まず、debug しやすさを守ります。
どこで音が変わったか見えるからです。

次に、test しやすさを守ります。
node ごとに性質を検証できるからです。

次に、差し替えやすさを守ります。
`fa_resample` を別 backend に変える。
`fa_vad` の model を変える。
`fa_asr` を Whisper から別 backend に変える。
そのとき、責務が分かれていれば、影響範囲を見積もれます。

次に、profile の説明しやすさを守ります。
SO101 用、debug 用、recording 用、streaming 用で、どの node を使うかを説明できます。

最後に、product としての信頼を守ります。
完了していないものを完了と言わない。
対応していないものを対応済みと言わない。
隠れた変換で結果を変えない。

これらが積み重なると、FluentAudio は土台になります。
その上に、音声対話、LLM、MCP、robot operation、vision integration を乗せられます。

### 4.8 ここを壊すと何が起きるか

もしこの思想を壊すと、最初は楽に見えます。
node が少なくなります。
設定も少なくなります。
一つの launch で何となく動くかもしれません。

でも、あとで必ず苦しくなります。

ASR の精度が悪いとき、それが microphone の問題なのか、AGC の問題なのか、resample の問題なのか、VAD gate の問題なのかわからなくなります。
KWS が反応しないとき、format が違うのか、model が違うのか、keyword file が違うのか、noise reduction が強すぎるのかわからなくなります。
Turn Detector が早く切りすぎるとき、VAD state が stale なのか、ASR result の timing がずれているのか、frame buffer が悪いのかわからなくなります。
speaker 出力が microphone に戻るとき、loopback があるのか、AEC があるのか、routing が正しいのかわからなくなります。

原因がわからない system は、直せません。
直せない system は、product になりません。

だから、FluentAudio は最初から分けます。
最初から見えるようにします。
最初から pipeline と contract を大切にします。

### 4.9 この章の結論

FluentAudio の真髄は、次の一文に集約できます。

```text
音声処理の全領域を持ち、それぞれを責務の明確な node として分け、pipeline と profile で見える形に組み合わせる。
```

これができるから、FluentAudio はただの audio adapter ではなくなります。
これができるから、VAD / KWS / ASR / Turn Detector を正しくつなげます。
これができるから、VLAbor profile から呼び出せます。
これができるから、robot の現実の音に耐えられます。
これができるから、後から model や backend を差し替えられます。
これができるから、test が意味を持ちます。
これができるから、完了と未検証を分けられます。

全部を持つ。
でも混ぜない。

この二つを同時に守ることが、FluentAudio の core です。

## 5. Capability Contract を先に立てる

Capability Contract とは、「この node / backend は何を受け取れて、何を受け取れないか」をはっきり決める約束です。

約束を先に立てる理由は簡単です。
何を受け取れるかわからない node は、信用できないからです。

backend と node は、処理を始める前に、少なくとも次を宣言します。

- encoding
- bit depth
- sample rate
- channel count
- layout
- byte alignment
- normalized range
- source_id
- stream_id
- model
- provider
- runtime
- output schema

たとえば、ある AI backend が次の条件だけを受け付けるとします。

```text
encoding: FLOAT32LE
sample_rate: 16000
channels: 1
layout: interleaved
range: -1.0 to 1.0
```

この場合、`PCM16LE` が来たら reject します。
48000Hz が来たら reject します。
stereo が来たら reject します。
payload byte length が壊れていたら reject します。

ここで勝手に resample してはいけません。
勝手に mono にしてはいけません。
勝手に float にしてはいけません。

それをやる node は、AI node ではなく、明示的な processing node です。

### static incompatibility

起動時にわかる不一致は、startup failure にします。

例:

- unknown backend
- missing model
- unsupported provider
- missing executable
- missing credential
- unsupported configured sample rate
- invalid config

起動時に壊れているなら、起動時に止めます。
あとで音声 frame が来るまで待ちません。

### per-frame incompatibility

frame ごとにわかる不一致は、backend state を触る前に frame rejection にします。

例:

- unsupported encoding
- unsupported sample rate
- unsupported channel count
- malformed payload
- non-finite float sample
- stream_id mismatch
- stale gate state

壊れた frame で backend state を変えてはいけません。
壊れた frame は、壊れた frame として扱います。

### runtime backend failure

起動後に必須 backend が壊れた場合は、runtime fatal shutdown にします。
message / service contract が明示的な error result を持っている場合だけ、その error result を返します。

必須 backend が壊れたのに warning だけ出して動き続けるのは、正直ではありません。

## 6. Fail Closed は、優しさではなく誠実さ

Fail closed とは、危ない状態や意味が壊れた状態で処理を続けないことです。

これは厳しい考えに見えるかもしれません。
でも、実際には一番やさしい考えです。
なぜなら、問題を早く見つけられるからです。

悪い fallback は、問題を隠します。
問題が隠れると、後で別の場所が壊れます。
別の場所が壊れると、原因を探すのが難しくなります。
原因を探すのが難しくなると、システム全体が信用できなくなります。

FluentAudio では、次を禁止します。

- missing device を別 device で代用する。
- missing model を package default model で代用する。
- unsupported backend を別 backend に差し替える。
- ASR failure を empty success として返す。
- KWS failure を no detection として返す。
- VAD failure を silence として返す。
- invalid format を AI node 内で変換して処理する。
- missing source_id / stream_id を topic 名だけで受け入れる。
- stale state を current state として扱う。
- 必須 resource 不足を warning だけにして続ける。

これらは全部、「本当は壊れているのに、壊れていないように見せる」行為です。
FluentAudio ではそれをしません。

できないことは、できないと言います。
壊れているものは、壊れていると言います。
未検証のものは、未検証と言います。

## 7. テストは証明である

テストは、FluentAudioROS2 を信頼してもらうための、いちばん大事な足がかりです。

どれだけ美しい設計書があっても、どれだけ立派な node 名があっても、どれだけ package が並んでいても、それだけでは信頼されません。
信頼は、証明から生まれます。
証明は、テストから生まれます。

FluentAudioROS2 は、一つの社会です。
一つの community です。
その中で、それぞれの node は一人の住人のように存在します。

`fa_resample` には `fa_resample` の人生があります。
`fa_vad` には `fa_vad` の人生があります。
`fa_asr` には `fa_asr` の人生があります。
`fa_mix` には `fa_mix` の人生があります。

彼らは、それぞれ別の役割を持って生まれます。
誰かは音の形を変えます。
誰かは音量を整えます。
誰かは発話を見つけます。
誰かは音声を文字にします。
誰かは音を混ぜます。
誰かは出力先へ届けます。

node には、それぞれの責務があります。
寿命があります。
成長があります。
将来の変更があります。
別の backend に差し替えられる日があります。
新しい profile に参加する日があります。
今まで知らなかった device や model と出会う日があります。

テストは、その node たちを独りにしないためのものです。

テストがあるから、node は「自分はここまでできる」と言えます。
テストがあるから、node は「これは自分の仕事ではない」と言えます。
テストがあるから、node は「この入力は受け取れない」と言えます。
テストがあるから、node は「壊れたらここで止まる」と言えます。
テストがあるから、別の人がその node を直しても、人生の約束が壊れていないか確認できます。

テストは、node の人生を支えるレールです。
そのレールがあるから、node は community の中で安心して役割を果たせます。
そのレールがあるから、他の node もその node を信頼できます。

テストは、FluentAudioROS2 という社会が、住人である node を見捨てないための仕組みです。

### 7.1 テストは存在確認ではない

テストは、実装が存在することを確認するためのものではありません。
テストは、「この性質が成り立つ」と示すためのものです。

存在確認だけのテストは、node を支えません。
その node が正しく生きているかを見ていないからです。

たとえば、次のようなテストは弱いです。

```text
source file に "from fa_kws_py.backends.sherpa_onnx_kws_worker import main" という文字列がある
```

このテストは、KWS が動くことを証明していません。
worker protocol が正しいことも証明していません。
unsupported input を reject することも証明していません。
ただ文字列があることを見ているだけです。

これは FluentAudio のテストではありません。

これでは、`fa_kws` の人生を守れません。
`fa_kws` が本当に wake word を扱えるのか。
backend が本当に起動できるのか。
unsupported input を本当に拒否できるのか。
stream_id が違うときに本当に無視できるのか。
それを何も見ていないからです。

文字列を見て安心するのは、表札を見て、その人の人生を理解したつもりになるようなものです。
表札は入口です。
人生そのものではありません。

テストも同じです。
file があること、import 文があること、README があること、package 名があることは、入口にすぎません。
それは、その node が社会の中で責任を果たせることの証明ではありません。

### 7.2 良いテストは実行経路を通る

良いテストは、実行経路を通ります。
node が本当に受け取り、本当に判断し、本当に出力し、本当に止まるところを見ます。

良いテストは、たとえば次を確認します。

- `FLOAT32LE/16000Hz/mono` を受け取れる。
- `PCM16LE` を受け取ったら reject する。
- stereo を mono-only backend に渡すと reject する。
- model path がないと startup failure になる。
- invalid backend name で起動しない。
- payload byte length が壊れていたら backend state を触らない。
- VAD state の stream_id が違うと ASR gate に使わない。
- launch config が必須 parameter を要求する。
- ROS graph 上で expected topic が publish される。
- backend public API が contract 通りの error を返す。

これらは、node の生き方を確かめています。

`fa_asr` が `FLOAT32LE/16000Hz/mono` だけを受け取るなら、その約束を守れているかを見る。
`fa_vad` が stream_id の違う frame を gate に使わないなら、その約束を守れているかを見る。
`fa_resample` が sample rate を変えるなら、数値的に正しい変換になっているかを見る。
`fa_mix` が複数 input を混ぜるなら、clipping や routing の契約を守るかを見る。
`fa_out` が sink を開けないなら、成功したふりをしないかを見る。

良いテストは、node にこう問いかけます。

```text
あなたは、自分の約束を守れますか。
あなたは、自分が受け取れないものを拒否できますか。
あなたは、壊れたときに壊れたと言えますか。
あなたは、他の node と一緒に働いても責務を失いませんか。
```

この問いに答えるのが、FluentAudioROS2 のテストです。

### 7.3 テストは数学の証明に似ている

テストは、数学の証明に似ています。
「ここがこうだから、これは正しい」と言える必要があります。

数学の証明では、「なんとなく正しそう」は証明ではありません。
FluentAudioROS2 のテストでも同じです。

「import 文があるから動くはず」は証明ではありません。
「README に書いてあるから対応しているはず」も証明ではありません。
「package.xml があるから package として完成しているはず」も証明ではありません。
「前に一度動いた気がする」も証明ではありません。

証明には、前提があります。
入力があります。
操作があります。
期待する結果があります。
失敗したときの扱いがあります。

たとえば `fa_resample` のテストなら、次のように考えます。

```text
前提:
- 入力 sample rate は 48000Hz
- 出力 sample rate は 16000Hz
- 入力は既知の sine wave

操作:
- fa_resample の backend public API に入力する

期待:
- 出力 sample count が変換比に合う
- 波形の主要周波数が保たれる
- payload size が AudioFrame contract に合う
- unsupported sample rate は明示 error になる
```

これは証明に近いです。
何を前提にして、何を確かめるのかが明確だからです。

`fa_vad` のテストなら、次のように考えます。

```text
前提:
- supported format の speech frame
- supported format の silence frame
- stream_id が一致する gate state
- stream_id が一致しない gate state

期待:
- speech probability が contract に従って publish される
- unsupported format は reject される
- stream_id mismatch は gate に使われない
- backend failure は明示的に扱われる
```

このように、テストは node の性質を一つずつ証明します。

### 7.4 テストは node の孤独をなくす

node は一人で存在しているわけではありません。

`fa_in` が出した frame を、processing node が受け取ります。
processing node が整えた frame を、AI node が受け取ります。
AI node の結果を、app が受け取ります。
app の判断で、routing や output が変わります。

一つの node の嘘は、次の node を傷つけます。

`fa_sample_format` が値域を間違えると、`fa_vad` は正しく判断できません。
`fa_resample` が sample rate を間違えると、`fa_asr` は時間を誤解します。
`fa_vad` が stream_id を無視すると、別の input の発話で ASR が動くかもしれません。
`fa_mix` が clipping を隠すと、`fa_out` は壊れた音をそのまま出すかもしれません。
`fa_stream` が network failure を成功のように扱うと、monitoring は配信できていると誤解します。

だから、テストは一つの node だけのためにあるのではありません。
その node と一緒に生きる、他の node のためにもあります。

テストは community の約束です。

`fa_resample` は「私は sample rate をこう変える」と約束します。
`fa_vad` は「私はこの contract の frame だけで発話を判断する」と約束します。
`fa_asr` は「私はこの条件の audio stream から transcript を作る」と約束します。
`fa_out` は「私はこの contract の音だけを sink に渡す」と約束します。

テストは、その約束が今も守られていることを確認します。

### 7.5 テストは寿命を延ばす

node は一度作って終わりではありません。
後から変わります。

backend が増えます。
model が変わります。
sample rate が増えます。
profile が増えます。
robot の構成が変わります。
VLAbor からの起動方法が変わります。
別の engineer が実装を直します。
別の agent が refactor します。

そのとき、テストがなければ、node は過去の約束を忘れます。
昔は reject していた unsupported input を、いつの間にか受け取るかもしれません。
昔は startup failure だった missing model を、いつの間にか warning だけで通すかもしれません。
昔は stream_id を確認していたのに、いつの間にか topic 名だけで受け入れるかもしれません。

これは node の人生を壊すことです。

テストは、node の記憶です。
「あなたはこれを守るために生まれた」と思い出させるものです。
「この約束を破ったら、community が困る」と知らせるものです。

だから、テストは node の寿命を延ばします。
node が将来変わっても、役割を失わないようにします。

### 7.6 FluentAudioROS2 で必要なテストの種類

FluentAudioROS2 には、いろいろな種類のテストが必要です。
一種類だけでは足りません。
node の人生には、いろいろな場面があるからです。

#### backend unit test

backend unit test は、ROS から切り離された backend の性質を見ます。

ここでは、device、file、network、DSP engine、model runtime adapter などを、できるだけ小さく検証します。

見るべきものは次です。

- supported config を受け取れるか。
- unsupported config を拒否するか。
- algorithm の数値的性質が合っているか。
- backend state が壊れた input で変わらないか。
- error result が contract 通りか。

backend は ROS を知らないので、速く、狭く、正確に test できます。

#### capability validation test

capability validation test は、node / backend が「何を受け取れるか」を守っているかを見ます。

見るべきものは次です。

- encoding
- bit depth
- sample rate
- channel count
- layout
- payload byte alignment
- normalized range
- source_id
- stream_id
- provider
- model
- runtime

FluentAudioROS2 では、supported / unsupported を明示的に管理します。
未対応なら reject します。
未対応なのに変換して処理することはしません。

この考えを守るために、capability validation test はとても重要です。

#### failure contract test

failure contract test は、壊れたときの振る舞いを見ます。

見るべきものは次です。

- startup failure
- frame rejection
- runtime fatal shutdown
- explicit error result

壊れたときの振る舞いは、正常系と同じくらい大事です。
むしろ、ロボットや音声 AI では、壊れたときのほうが大事なこともあります。

壊れた状態を成功に見せる node は、community を危険にします。

#### launch test

launch test は、起動時の約束を見ます。

見るべきものは次です。

- required parameter が必須になっているか。
- missing backend が起動失敗になるか。
- unknown backend が起動失敗になるか。
- deprecated config key を受け付けていないか。
- profile から意図した node が起動するか。
- package-local fallback config で意味を変えていないか。

launch は、実際に system が社会として集まる入口です。
ここが曖昧だと、node は間違った条件で生まれてしまいます。

#### ROS graph behavior test

ROS graph behavior test は、node が community の中でどう振る舞うかを見ます。

見るべきものは次です。

- expected topic に publish するか。
- expected topic を subscribe するか。
- service が contract 通りに応答するか。
- message の field が正しく埋まるか。
- stream_id / source_id を守るか。
- QoS や timing の前提が崩れていないか。

node は graph の中で生きます。
だから、graph の中での振る舞いも test します。

#### integration test

integration test は、複数 node が一緒に働けるかを見ます。

見るべきものは次です。

- `fa_in -> processing -> fa_vad` が成立するか。
- `fa_in -> processing -> fa_kws` が成立するか。
- `fa_in -> processing -> fa_asr` が成立するか。
- `fa_tts -> fa_mix -> fa_out` が成立するか。
- loopback / AEC / routing の関係が壊れていないか。
- profile で定義した pipeline が意図通りにつながるか。

integration test は、社会としての動きを見ます。
一人ひとりが正しくても、一緒に働けなければ product にはなりません。

#### regression test

regression test は、一度守ると決めた約束が、あとで壊れていないかを見ます。

bug が見つかったら、その bug を再現する test を残します。
そうすれば、同じ bug が戻ってきたときに気づけます。

これは、community の記憶です。
一度痛い思いをした場所に、もう一度落ちないための印です。

### 7.7 正常系だけでは足りない

正常系のテストだけでは、FluentAudioROS2 は信頼されません。

「正しい input が来たら正しく動く」は大事です。
でも、それだけでは足りません。

現実には、間違った input が来ます。
壊れた payload が来ます。
sample rate が違います。
channel count が違います。
model がありません。
device がありません。
network が切れます。
worker が落ちます。
古い stream_id が来ます。

だから、FluentAudioROS2 のテストは、失敗の形も見ます。

node が美しいかどうかは、成功したときだけではわかりません。
失敗したときに、責務を失わないかでわかります。

失敗したときに嘘をつかない node は、美しい node です。
失敗したときに理由を伝えられる node は、community の中で信頼されます。

### 7.8 テスト設計は問いを立てること

テスト設計とは、test code を貼ることではありません。
テスト設計とは、「何を証明するべきか」という問いを立てることです。

良いテスト設計には、次があります。

- 対象 node
- 対象 backend
- 守るべき contract
- 前提
- 入力
- 操作
- 期待結果
- 失敗条件
- 未検証範囲

たとえば、`fa_sample_format` のテスト設計なら、問いはこうです。

```text
この node は、int16 PCM を float32 normalized range に変換するとき、
値域、byte order、payload size、clipping の扱いを contract 通りに守れるか。
```

たとえば、`fa_vad` のテスト設計なら、問いはこうです。

```text
この node は、supported AudioFrame だけを backend に渡し、
unsupported AudioFrame を backend state に触れる前に reject できるか。
```

たとえば、`fa_asr` のテスト設計なら、問いはこうです。

```text
この node は、turn context、VAD state、audio stream identity を混同せず、
正しい stream だけを ASR backend に渡せるか。
```

こういう問いがあるから、test code は意味を持ちます。
問いがない test code は、ただの作業です。
問いがある test code は、証明になります。

### 7.9 書類とテストコードの関係

テスト設計は書類です。
テストコードは実行される証明です。

この二つを混同してはいけません。

テスト設計には、検証したい性質を書きます。
テストコードには、その性質を実行して確かめる手順を書きます。

仕様書に test code を貼っても、test は実行されません。
Markdown に sample code を置いても、CI は守ってくれません。
自然言語資料を読んで assert しても、node の性質は証明されません。

書類は、何を守るべきかを示します。
テストコードは、本当に守れているかを確かめます。

この二つがそろって、初めて node の人生を支えるレールになります。

### 7.10 Test fixture は社会の道具である

test fixture は、ただの補助 file ではありません。
node が自分の責務を証明するための道具です。

FluentAudioROS2 では、fixture も意味を持つべきです。

たとえば次です。

- known sine wave
- known silence
- known speech-like frame
- malformed payload
- unsupported sample rate frame
- stereo frame
- stream_id mismatch frame
- missing model config
- fake backend failure
- deterministic worker response

fixture は、何を証明するためのものかがわかる名前にします。
ただの `sample.wav` では弱いです。
`float32le_16k_mono_sine_440hz.wav` のように、意味がわかる名前のほうがよいです。

fixture が意味を持つと、test も読みやすくなります。
test が読みやすいと、node の約束も守りやすくなります。

### 7.11 テストは変更を怖くなくする

テストがない system では、変更が怖くなります。

少し code を変えただけで、どこが壊れたかわからない。
backend を差し替えただけで、ASR が悪くなった理由がわからない。
profile を変えただけで、VAD と KWS の関係が壊れたかもしれない。

こうなると、誰も安心して直せません。
直せない system は、成長できません。

テストがあると、変更が怖くなくなります。

なぜなら、守るべき約束が test として残っているからです。
変更したあとに test を走らせれば、どの約束が守られているか見えます。
壊れたなら、どの約束が壊れたか見えます。

テストは、未来の変更を助けます。
まだ会ったことのない engineer を助けます。
未来の自分を助けます。
未来の agent を助けます。

それは、community としてとても大事なことです。

### 7.12 テストは完了判定の土台である

完了済みと言うには、証拠が必要です。
その証拠の中心にあるのが test と verification です。

FluentAudioROS2 では、次のものだけで完了とは言いません。

- file を作った。
- package を作った。
- README を書いた。
- launch file を作った。
- test file を置いた。
- 一つの happy path が通った。
- build が通った。

それらは大事ですが、完了そのものではありません。

完了に近づくには、次が必要です。

- contract が明示されている。
- supported input が通る。
- unsupported input が reject される。
- missing resource が startup failure になる。
- backend failure が隠されない。
- ROS graph で expected behavior が確認される。
- profile から必要な pipeline が起動できる。
- 未検証範囲が明記されている。

完了とは、community がその node を信頼してよい状態です。
信頼してよいと言うには、test と verification が必要です。

### 7.13 悪いテストは node を孤独にする

悪いテストは、node を助けません。
むしろ node を孤独にします。

文字列を読むだけのテストは、node の本当の仕事を見ていません。
Markdown を読むだけのテストは、node の実行経路を見ていません。
package layout だけを見るテストは、node が責務を果たせるかを見ていません。

そういうテストがあると、見た目だけは守られているように見えます。
でも、node は本当には守られていません。

node が壊れても、テストは気づきません。
backend contract が壊れても、テストは気づきません。
unsupported input を受け入れても、テストは気づきません。
ASR が壊れても、import 文字列が残っていれば test は通ります。

これは、node にとって残酷です。
守られているように見えて、実は守られていないからです。

FluentAudioROS2 では、そういうテストを残しません。
node の人生を支えないテストは、community の信頼を増やしません。

### 7.14 良いテストは node に価値を与える

良いテストは、node に価値を与えます。

なぜなら、良いテストがある node は、他の node から信頼されるからです。
他の engineer からも信頼されます。
Product Owner からもレビューしやすくなります。
ClaudeCode Documentation Writer も、事実に基づいて書類を書けます。

良いテストがある node は、自分の役割を説明できます。
良いテストがある node は、何を受け取れるか説明できます。
良いテストがある node は、何を拒否するか説明できます。
良いテストがある node は、壊れたときどうするか説明できます。

これは node の尊厳です。

node はただ code として存在しているのではありません。
FluentAudioROS2 という社会の中で、役割を持って存在しています。
その役割が証明されることで、node は community の中で価値を持ちます。

テストは、その価値を与えるものです。

### 7.15 この章の結論

FluentAudioROS2 のテストは、単なる check ではありません。
FluentAudioROS2 のテストは、node の人生を支えるレールです。

テストは、node を独りにしません。
テストは、node の約束を守ります。
テストは、node の失敗を隠しません。
テストは、node の変更を助けます。
テストは、community の記憶になります。
テストは、Product Owner の完了判定を支えます。
テストは、ClaudeCode の書類記載を事実に近づけます。
テストは、未来の engineer と agent を助けます。

だから、FluentAudioROS2 では、テストを軽く扱いません。

テストは証明です。
テストは信頼です。
テストは記憶です。
テストは約束です。
テストは、node が生まれてきた価値を守るものです。

この社会で node が美しく生きるために、テストというレールを敷きます。

それが FluentAudioROS2 のテストの美学です。

## 8. 書類は地図であり、実装そのものではない

書類は大事です。
仕様書、アルゴリズム説明、テスト設計、backend docs は、node を正しく作るための地図です。

しかし、地図があるだけでは、道は完成しません。
仕様書があるだけでは、node は完成しません。
テスト設計があるだけでは、テストは実行されていません。
README があるだけでは、ROS package は動きません。

だから、書類では次を混同してはいけません。

- 設計済み
- 実装済み
- test code 追加済み
- build 済み
- launch 済み
- graph 検証済み
- 実 device で検証済み
- 実 model で検証済み
- 実 backend で検証済み
- 未検証

「完了済み」と書いてよいのは、実装と代表検証が揃ったものだけです。

次のものは、完了の証明ではありません。

- package 名がある。
- README がある。
- `package.xml` がある。
- launch skeleton がある。
- topic contract が書いてある。
- docs がある。
- passthrough backend がある。

それらは入口です。
完了ではありません。

## 9. 書類記載は ClaudeCode が担当する

FluentAudio では、書類記載の責任を分けます。

仕様書、アルゴリズム詳細説明書、テスト設計、backend docs などの自然言語資料は、ClaudeCode Documentation Writer が担当します。

Node Engineer は、実装、テストコード、検証証跡を担当します。
Node Engineer は、ClaudeCode が正確に書けるように、事実を報告します。

Product Owner は、書類と実装が合っているかをレビューします。

この分担には意味があります。
実装者が急いで書類も書くと、実装の都合で仕様が曲がることがあります。
書類担当者が実装を見ずに書くと、実装されていないことを実装済みのように書いてしまうことがあります。
Product Owner が見ないと、全体の方向とずれることがあります。

だから、役割を分けます。

ClaudeCode Documentation Writer は、事実を書きます。
推測で埋めません。
足りない情報があれば、足りないと言います。

## 10. Product Owner は実装しない

Product Owner は、手を動かして node を実装する人ではありません。
Product Owner は、FluentAudio という product が正しい方向へ進むように監督する人です。

Product Owner は次を行います。

- 何を作るべきかを決める。
- 何を今は作らないかを決める。
- 作業を分ける。
- Node Engineer に実装を委任する。
- ClaudeCode に書類記載を委任する。
- 戻ってきた成果物をレビューする。
- 完了条件を満たしているかを見る。
- 未実装と未検証を隠さない。
- 作業が散らばらないように統合する。

Product Owner が自分で実装を始めると、全体を見る人がいなくなります。
すると、作業があちらこちらに散ります。
散った作業は、最後に product になりません。

だから Product Owner は、実装しません。
Product Owner は、見ます。
考えます。
分けます。
委任します。
レビューします。
完成の条件を守ります。

## 11. Node Engineer は担当 slice を完成させる

Node Engineer は、指定された node / package / backend slice を担当します。
slice とは、切り出された一つの作業範囲です。

Node Engineer は、担当範囲を広げすぎません。
頼まれていない package をついでに直しません。
書類を勝手に書き換えません。
Product Owner の代わりに product 判断をしません。

Node Engineer は次を行います。

- 担当 slice の責務を確認する。
- 入力 contract を実装する。
- 出力 contract を実装する。
- backend capability を明示する。
- unsupported input を reject する。
- startup failure 条件を実装する。
- runtime failure 条件を実装する。
- 意味のあるテストを追加する。
- 代表検証を実行する。
- 未検証範囲を報告する。
- ClaudeCode に渡す書類入力をまとめる。

Node Engineer の仕事は、ただ code を増やすことではありません。
Node Engineer の仕事は、担当 slice を「正しく判断できる状態」にすることです。

## 12. ClaudeCode Documentation Writer は事実を書く

ClaudeCode Documentation Writer は、自然言語資料を書く担当です。

ClaudeCode Documentation Writer は次を行います。

- Product Owner の目的を読む。
- Node Engineer の報告を読む。
- 実装事実を読む。
- 未検証範囲を読む。
- 仕様書に責務を書く。
- アルゴリズム説明に処理の考えを書く。
- テスト設計に検証すべき性質を書く。
- backend docs に backend capability と failure 条件を書く。

ClaudeCode Documentation Writer は、次をしません。

- 実装されていないことを実装済みと書く。
- 未検証のことを検証済みと書く。
- テスト設計に test code を貼る。
- Markdown を読むだけのテストを正当化する。
- production code を変更する。
- test code を変更する。
- launch や config を変更する。

書類は product を支えるものです。
書類で product をごまかしてはいけません。

## 13. Repository Boundary を守る

FluentAudio は、親 repository の中にある別 repository です。
この境界を雑に扱うと、release や他の engineer の作業に影響します。

守ることは単純です。

- push しない。
- 親 repository を勝手に commit しない。
- parent gitlink 更新を勝手に commit しない。
- `vlabor_ros2` を勝手に変更しない。
- `vlabor_ros2` を勝手に commit しない。
- `CPP_CODING_RULES.md` を勝手に変更しない。
- `CLAUDECODE_RULES.md` を勝手に変更しない。
- 他の人や他エージェントの変更を勝手に revert しない。

既存ルールに追加したいことがある場合は、既存ルール文書を書き換えません。
別の提案資料として出します。
承認されてから反映します。

境界を守ることは、礼儀ではありません。
product を壊さないための技術です。

## 14. 直近の product 目標

直近の主軸は、親 repository の次の設計に対して、FluentAudio 側の機能を成立させることです。

- `/home/user/repositories/daihen-physical-ai/docs/設計/2026-05-19-PhysicalAIエージェントタイムライン設計.md`

この目標に対して、FluentAudio が重点化するものは次です。

- DSP 全分類
- AI 系 node
- backend 化
- VLAbor profile 連携

`fa_in` / `fa_out` は主戦場ではありません。
必要なときだけ、source / sink 境界として確認します。

外部推論 worker は、直近で使う経路が明確になるまで優先目標に入れません。
backend 境界の考え方は大事ですが、使う場所がない worker を先に増やすことは、product を前に進めません。

## 15. 完了とは何か

完了は、雰囲気ではありません。
完了は、「作った気がする」ことではありません。
完了は、「file が増えた」ことでもありません。
完了は、「test が何か通った」ことだけでもありません。

FluentAudio で完了に近づくには、次が必要です。

- 目的に対応している。
- 責務境界が崩れていない。
- capability contract が明示されている。
- unsupported input が fail closed で扱われている。
- 実装が contract に従っている。
- テストが性質を検証している。
- 代表検証が実行されている。
- 未検証範囲が明記されている。
- 書類と実装が矛盾していない。
- Product Owner がレビューできる証跡が残っている。

完了済みの表に載せるものは、特に厳しく扱います。
完了済みとは、他の人がそれを土台にしてもよい、という意味です。
土台にしてよいと言うには、証拠が必要です。

証拠がないなら、未検証です。
未検証なら、未検証と書きます。

それは恥ではありません。
それは正直です。

## 16. 迷ったときの判断

迷ったら、次の順に考えます。

1. これはこの node の責務か。
2. この入力を本当に受け取れるのか。
3. 受け取れないなら、どこで reject するのか。
4. 変換が必要なら、その変換 node は明示されているか。
5. backend capability は宣言されているか。
6. 失敗したとき、後段が理由を知れるか。
7. テストは文字列ではなく性質を見ているか。
8. 書類は実装事実と一致しているか。
9. 未検証のものを完了済みにしていないか。
10. Product Owner がレビューできる証跡があるか。

この問いに答えられないなら、作業を進める前に止まります。
止まって、未決事項として報告します。

報告することは、失敗ではありません。
わからないまま進めることが失敗です。

## 17. FluentAudio が目指す姿

FluentAudio は、ただ音を流すだけの repository ではありません。
ロボットが音を聞き、人と話し、環境を理解し、他の system とつながるための土台です。

その土台が嘘をつくと、上に乗るものが全部ずれます。
だから FluentAudio は、正直でなければなりません。

その土台が何でも混ぜると、どこが壊れたかわからなくなります。
だから FluentAudio は、責務を分けなければなりません。

その土台が検証なしに完了を名乗ると、次の人が危ない場所に立ちます。
だから FluentAudio は、証拠を残さなければなりません。

この資料で言う美学とは、きれいな言葉ではありません。
仕事を前に進めるための、地味で、厳しくて、実用的な態度です。

正直に受け取り、正直に拒否し、正直に処理し、正直に検証し、正直に報告する。

FluentAudio の美しさは、そこにあります。

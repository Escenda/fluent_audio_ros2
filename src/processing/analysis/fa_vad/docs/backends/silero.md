# silero Backend

## Backend Name

`silero`

## Runtime

External Silero VAD process。

`fa_vad` は PyTorch / Silero を ROS2 node process 内で import しません。`backend.command` で指定した外部 worker process に audio window を WAV file として渡し、stdout に出力された probability float を読みます。

## Input

- mono PCM16 bytes
- target sample rate
- local WAV path passed as `{audio}`
- model path passed as `{model}`
- execution provider string passed as `{provider}`

## Output

- probability
- is_speech
- start
- end

## Failure Conditions

- `backend.model_path` が空
- `backend.model_path` が存在しない local torch.hub repository directory を指す
- `backend.execution_provider` が空
- `backend.execution_provider` が未対応
- `backend.command` が空
- `backend.command` が実行不能
- `backend.args` に `{audio}`, `{model}`, `{provider}`, `{sample_rate}` が含まれない
- external command timeout / non-zero exit
- external command stdout が probability float ではない
- probability が `[0.0, 1.0]` の範囲外

`backend.model_path` は必須です。空の場合は `~/.cache/torch/hub` などを推測せず起動失敗します。online download fallback はありません。`backend.execution_provider` は `cpu`, `cuda`, `cuda:<index>` のいずれかを明示します。

`backend.command` は ROS2 node と異なる Python / venv / container runtime を指すための境界です。command が失敗しても別 backend へ fallback しません。

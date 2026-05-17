# silero Backend

## Backend Name

`silero`

## Runtime

Python / PyTorch Silero VAD。

## Input

- mono PCM16 bytes
- target sample rate

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
- model load failure

`backend.model_path` は必須です。空の場合は `~/.cache/torch/hub` などを推測せず起動失敗します。online download fallback はありません。`backend.execution_provider` は `cpu`, `cuda`, `cuda:<index>` のいずれかを明示します。

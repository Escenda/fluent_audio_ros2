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

- repo dir が存在しない
- online download が disabled かつ local repo がない
- model load failure

`silero.repo_dir` は `silero.allow_online=false` のとき必須です。空の場合は
`~/.cache/torch/hub` などを推測せず起動失敗します。online download は
fallback ではありません。`silero.allow_online=true` かつ repo dir 未指定の
場合だけ、明示された online source として使います。

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

online download は fallback ではありません。`silero.allow_online` を明示した場合だけ許可します。

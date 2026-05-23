# FA Interfaces

`fa_interfaces`は FluentAudio の msg/srv を集約するインターフェースパッケージです。

## Messages
- `fa_interfaces/msg/AsrControl`
- `fa_interfaces/msg/AsrTranscript`
- `fa_interfaces/msg/AudioClipRef`
- `fa_interfaces/msg/AudioEmbeddingFrame`
- `fa_interfaces/msg/AudioFrame`
- `fa_interfaces/msg/AudioWindowRef`
- `fa_interfaces/msg/CqtFrame`
- `fa_interfaces/msg/EncodedAudioChunk`
- `fa_interfaces/msg/LogMelFrame`
- `fa_interfaces/msg/LoudnessFrame`
- `fa_interfaces/msg/MfccFrame`
- `fa_interfaces/msg/OnsetFrame`
- `fa_interfaces/msg/PitchFrame`
- `fa_interfaces/msg/PlaybackDone`
- `fa_interfaces/msg/ResolvedTimeRange`
- `fa_interfaces/msg/StftFrame`
- `fa_interfaces/msg/TempoFrame`
- `fa_interfaces/msg/TurnContext`
- `fa_interfaces/msg/TurnEnd`
- `fa_interfaces/msg/TurnEndRequest`
- `fa_interfaces/msg/WakeWordResult`

## Services
- `fa_interfaces/srv/ArchiveAudioWindow`
- `fa_interfaces/srv/ExportAudioWindow`
- `fa_interfaces/srv/ListDevices`
- `fa_interfaces/srv/PlaybackControl`
- `fa_interfaces/srv/Record`
- `fa_interfaces/srv/Speak`
- `fa_interfaces/srv/SwitchDevice`

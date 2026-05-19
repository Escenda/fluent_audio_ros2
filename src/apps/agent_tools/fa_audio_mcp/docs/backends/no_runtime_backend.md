# no runtime backend 境界

`fa_audio_mcp` は runtime backend package ではありません。audio DSP、model inference、codec、archive storage の backend を持ちません。

## 境界

| 項目 | 所有者 |
| --- | --- |
| MCP transport / tool schema | `fa_audio_mcp` |
| numeric time range validation | `fa_audio_mcp` |
| Agent-facing scope mapping | `fa_audio_mcp` |
| ROS service client | `fa_audio_mcp` |
| ASR-ready timeline | `fa_asr` |
| `TranscribeAudio` service 実行 | `fa_asr` |
| archive PCM16 window | `fa_audio_window` |
| `ArchiveAudioWindow` service 実行 | `fa_audio_window` |
| audio format conversion | `fa_sample_format` などの processing package |
| resample | `fa_resample` |
| durable artifact / World Station 連携 | 別 runtime / integration layer |

## 禁止事項

- `fa_audio_mcp` 内で ASR model を load しない。
- `fa_audio_mcp` 内で PCM decode / WAV encode / resample / channel conversion を実装しない。
- `fa_audio_mcp` 内で `fa_audio_window` の代わりに window buffer を持たない。
- `fa_audio_mcp` 内で unsupported scope を default scope に落とさない。
- `fa_audio_mcp` 内で service failure を成功 result に変換しない。

## SO101 voice frontend との関係

SO101 voice frontend では、ASR は `audio/high_pass/frame` / `audio/high_pass/mic` を入力にします。archive branch は同じ high-pass stream から `fa_archive_sample_format` で PCM16 に変換し、`audio/archive_pcm16/frame` / `audio/archive_pcm16/mic` を `fa_audio_window` に渡します。

`fa_audio_mcp` はこの 2 系統を隠して統合しません。transcribe tool は transcribe scope mapping に従って ASR service を呼び、archive tool は archive scope mapping に従って archive service を呼びます。

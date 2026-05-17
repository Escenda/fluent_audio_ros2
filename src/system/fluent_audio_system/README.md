# fluent_audio_system

`fluent_audio_system` は FluentAudio の node group を YAML から展開する launch 専用 package です。VLAbor profile からは `type: include` で `launch/run.py` を呼び、詳細な backend / model / node 構成は FluentAudio 側 config に閉じ込めます。

```bash
ros2 launch fluent_audio_system run.py config:=/path/to/fluent_audio_system.yaml
```

Missing config や missing params file は起動失敗にします。暗黙の device 推測、model fallback、temporary YAML 書き換えは行いません。

## Profiles

- `config/profiles/so101.yaml`: SO101 の site-bound I/O profile。format pipeline は定義済みだが default では無効。
- `config/profiles/so101_mic_frontend.yaml`: `fa_in -> fa_sample_format -> fa_resample` を明示した microphone frontend profile。`fa_resample` へ `PCM16LE` を直結せず、`fa_sample_format` で `FLOAT32LE/32/interleaved` に変換してから 16kHz 化する。

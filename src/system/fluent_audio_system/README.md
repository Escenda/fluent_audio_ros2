# fluent_audio_system

`fluent_audio_system` は FluentAudio の node group を YAML から展開する launch 専用 package です。VLAbor profile からは `type: include` で `launch/run.py` を呼び、詳細な backend / model / node 構成は FluentAudio 側 config に閉じ込めます。

```bash
ros2 launch fluent_audio_system run.py config:=/path/to/fluent_audio_system.yaml
```

Missing config や missing params file は起動失敗にします。暗黙の device 推測、model fallback、temporary YAML 書き換えは行いません。

# fluent_audio_system

`fluent_audio_system` は FluentAudio の node group を YAML から展開する launch 専用 package です。VLAbor profile からは `type: include` で `launch/run.py` を呼び、詳細な backend / model / node 構成は FluentAudio 側 config に閉じ込めます。

```bash
ros2 launch fluent_audio_system run.py \
  config:=/path/to/fluent_audio_system.yaml \
  fa_in_enabled:=true \
  fa_out_enabled:=true \
  fa_in_source_id:=hw:1,0 \
  fa_out_sink_id:=hw:2,0
```

`config` は単一 YAML に加えて、カンマ区切りの明示 config path list を受け付けます。複数指定時は左から右の順に compose し、group / node の展開順と required package list の順序もその順序を保持します。

```bash
ros2 launch fluent_audio_system run.py \
  config:=/path/to/so101_voice_frontend.yaml,/path/to/so101_agent_audio_tools.yaml \
  fa_in_enabled:=false \
  fa_out_enabled:=false \
  fa_in_source_id:=disabled \
  fa_out_sink_id:=disabled
```

I/O node を起動しない debug では、次のように site binding を明示的に無効化します。

```bash
ros2 launch fluent_audio_system run.py \
  config:=/path/to/fluent_audio_system.yaml \
  fa_in_enabled:=false \
  fa_out_enabled:=false \
  fa_in_source_id:=disabled \
  fa_out_sink_id:=disabled
```

Missing config、invalid YAML、config list の空 segment、missing params file は起動失敗にします。複数 config を compose する場合、`system.default_start_delay` / `system.inter_group_delay` の不一致、group id 重複、enabled node id 重複も起動失敗にします。`params_file` は対象 node 用の `ros__parameters` block を持つ必要があります。暗黙の device 推測、model fallback、temporary YAML 書き換えは行いません。
`fa_in` / `fa_out`、codec / correction / deterministic analysis、AI / TTS の runtime backend package は effective `backend.name` を必須にし、runtime default backend へ落としません。
`config`、`fa_in_enabled`、`fa_out_enabled`、`fa_in_source_id`、`fa_out_sink_id` に launch default はありません。profile で enabled な IO を起動しない場合は、site profile または debug launch で `fa_in_enabled:=false` / `fa_out_enabled:=false` を明示します。

VLAbor / Docker 側で system config から build 対象 package を解決する場合は、次の CLI を使います。出力は `fa_interfaces`、`fluent_audio_system`、enabled node package の順で 1 行 1 package です。

```bash
ros2 run fluent_audio_system list_required_packages --config /path/to/fluent_audio_system.yaml
```

`--config` も同じカンマ区切り config path list を受け付けます。child-side FluentAudio config composition は package-local config / CLI で検証済みです。さらに Docker/VLAbor ROS2 環境で、combined launch smoke は `2 passed in 11.57s` として確認済みです。

この smoke suite は、`config:=owner.yaml,adapter.yaml` 形式の generated owner/adapter config を使う launch-managed MCP HTTP relative-time tool-call smoke と、実 profile pair `${share:fluent_audio_system}/config/profiles/so101_voice_frontend.yaml,${share:fluent_audio_system}/config/profiles/so101_agent_audio_tools.yaml` を `fluent_audio_system/run.py` 経由で起動する SO101 profile pair runtime smoke を含みます。後者は `fa_audio_mcp -> fa_audio_window` の runtime 経路を確認します。ただし、この検証は child repo 内の実 FluentAudio profile pair smoke であり、親 VLAbor profile integration、親 Agent Runtime / MCP client integration、実 R2/S3 client upload、World Station evidence linkage、実 SO101 device / model provisioning の検証完了を意味しません。

## Profiles

- `config/profiles/so101.yaml`: SO101 の site-bound I/O profile。format pipeline は定義済みだが default では無効。
- `config/profiles/so101_mic_frontend.yaml`: `fa_in -> fa_sample_format -> fa_resample -> fa_dc_offset_removal -> fa_high_pass` を明示した microphone frontend profile。`fa_resample` へ `PCM16LE` を直結せず、`fa_sample_format` で `FLOAT32LE/32/interleaved` に変換してから 16kHz 化し、AI 前処理は dedicated processing node として挟む。
- `config/profiles/so101_kws_frontend.yaml`: `fa_in -> fa_sample_format -> fa_resample -> fa_dc_offset_removal -> fa_high_pass -> fa_kws` を明示した KWS frontend profile。KWS の worker command、model path / model files、provider、worker args、health args、QoS は FluentAudio system config に閉じる。Turn Detector はこの profile では起動しない。
- `config/profiles/so101_voice_frontend.yaml`: `fa_in -> fa_sample_format -> fa_resample -> fa_dc_offset_removal -> fa_high_pass -> fa_archive_sample_format -> fa_audio_window` と `fa_kws / fa_turn_detector / fa_dialogue` を明示した voice frontend profile。会話 orchestration は app layer が担う。
- `config/profiles/so101_tts_output.yaml`: `fa_tts -> fa_resample -> fa_sample_format -> fa_mix -> fa_out` を明示した TTS playback profile。`fa_tts` の `FLOAT32LE/32` を 48kHz に揃えてから `PCM16LE/16` に変換し、`fa_out` に hidden resample / format conversion を持たせない。OpenJTalk dictionary path は `FLUENT_AUDIO_OPENJTALK_DICT_DIR` で明示する。

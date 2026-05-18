# src/system

`src/system` contains launch and system-composition packages for FluentAudio.

System packages read explicit FluentAudio system configuration, validate the
configured groups and nodes, and expand them into ROS 2 launch actions.

Current packages:

- `fluent_audio_system/`: expands FluentAudio node groups from system YAML.

System packages do not infer devices, select model backends, rewrite temporary
YAML, or substitute fallback models. Site-specific bindings such as source and
sink identifiers must be passed explicitly by the caller or profile, while
backend and model details remain in FluentAudio configuration files.

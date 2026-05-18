# no_runtime_backend

`fa_safety_policy` は deterministic policy evaluation package であり、runtime backend を選択しない。
No external runtime backend is selected by this package.

## Boundary

- policy rules are package configuration, not backend selection
- dialogue / voice command proposal は安全判断前の input として扱う
- robot state と operator confirmation は typed input として検証する

## Forbidden

- model backend fallback
- plugin backend fallback
- default allow fallback
- delegating final safety accept / reject to a dialogue backend
- treating stale robot state as current state

No model backend fallback is allowed.

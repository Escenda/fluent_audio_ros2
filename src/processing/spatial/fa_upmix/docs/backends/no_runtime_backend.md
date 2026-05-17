# no_runtime_backend

`fa_upmix` は外部 runtime backend を持たない。

処理は C++ node 内の deterministic duplicate algorithm で完結する。モデル、外部 worker、device API、network API、runtime plugin は使用しない。

この backend 境界により、`fa_upmix` は以下を行わない。

- device I/O
- resample
- format conversion
- channel 推定
- ML model inference
- 外部 process 起動

未対応 config または runtime contract 違反は fallback せず、startup failure または frame drop として扱う。

# AEC Challenge (ICASS 2023) ベンチマーク

参照: https://www.microsoft.com/en-us/research/academic-program/acoustic-echo-cancellation-challenge-icassp-2023/

## ディレクトリ構成（標準）
- `benchmarks/aec_challenge_2023/data/`（git管理しない）
- `benchmarks/aec_challenge_2023/runs/<run_id>/`（git管理しない）
- `benchmarks/aec_challenge_2023/results/`（必要ならgit管理）

## 重要（保存）
使用したデータ/生成物/評価結果/評価に用いた全データは必ず保存します（`docs/fa_benchmark_policy.md`）。

## 実行（雛形）
`benchmarks/scripts/run_aec_challenge_2023.sh` を使用し、`data/` を指して `runs/<run_id>/` に成果物をまとめます。


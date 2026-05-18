# fa_safety_policy

音声入力による操作を安全に制御するためのポリシー（例: 起動/停止/モード切替のガード、危険操作の拒否、確認要求）を配置する予定地です。

現時点では ROS 2 package ではありません。`package.xml`、launch、config、node 実装を持たないため、実装済みとして扱いません。

## Documents

- `docs/仕様書.md`
- `docs/アルゴリズム詳細説明書.md`
- `docs/テスト設計.md`
- `docs/backends/no_runtime_backend.md`

## Backend Documents

この package は deterministic safety policy を扱い、model runtime backend や plugin backend を選択しません。backend 化が必要になった時点で、`docs/backends/no_runtime_backend.md` ではなく具体 backend contract を追加します。

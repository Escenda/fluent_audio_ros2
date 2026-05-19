# Fluent Vision ROS2 C++ コーディング規約（実装ベース）

本ドキュメントは、`fluent_vision_ros2` 配下の既存 C++ 実装から観察したスタイル・設計パターンを整理したものです。新規コードは **既存実装と一貫したスタイル** を守ることを目標とします。

## 1. ファイル構成・基本方針

- 1クラス（もしくは密接に関連する小さな構造体群）につき1ヘッダ/1ソースを基本とする。
- パッケージごとに `include/` と `src/` を分離し、公開インターフェースは `include/<package>/` 配下に配置する。
- `src/` 配下に Markdown などドキュメントファイルを置かない（ドキュメントは `docs/` やリポジトリルートに置く）。
- 例外ベースのエラー処理と ROS2 ロギング（`RCLCPP_*`）を併用し、「落とすべきかどうか」で使い分ける。

## 2. インクルードとガード

- ヘッダガードには `#pragma once` を用いる。
- `.cpp` の先頭では、基本的に次の順序でインクルードする:
  - 対応する自ヘッダ（例: `#include "fv_audio/fv_audio_node.hpp"`）
  - C++ 標準ライブラリヘッダ
  - サードパーティ（OpenCV, ALSA, tf2 など）
  - ROS2 / メッセージ型のヘッダ
  - プロジェクト内の他ヘッダ（`fluent_lib/*`, `fv_*` など）
- 不要なヘッダは極力インクルードしない。

## 3. 名前空間とスコープ

- パッケージ単位で名前空間を定義する:
  - 例: `namespace fv_audio { ... }`, `namespace fv_instance_seg { ... }`
  - 共通ライブラリは `namespace fluent_lib { ... }` やサブ名前空間 `fluent_lib::async` などを利用。
- 無名名前空間は **このファイルだけで使う静的関数・定数** に限定して使う。
- 名前空間終端にはコメントを付ける: `} // namespace fluent_image`

## 4. 命名規則

- クラス名: `CamelCase`（例: `FvAudioNode`, `InstanceSegNode`, `Worker`）
- 構造体・型エイリアス: クラスと同様に `CamelCase`
- メンバ関数: `camelCase`（例: `loadParameters`, `configureDevice`, `imageCallback`）
- 自由関数: `camelCase` または用途に応じた短い動詞＋目的語
- 変数名: `snake_case`
- クラスメンバ変数: `snake_case_`（末尾にアンダースコア）
- 定数: `kCamelCase` 形式の `constexpr`（例: `constexpr double kInt16Scale`）
- ROS2 パラメータ名: `group.key` のドット区切り（例: `"audio.device_selector.mode"`, `"metrics.length_unit"`）

## 5. フォーマット・記法

- インデントはスペース 2〜4 だが、**既存ファイルのスタイルに合わせる**（`fv_audio` 系は 2、`fluent_lib` 系は 4 が多い）。
- ブレースは宣言行と同じ行に置く:
  - `class Foo { ... };`
  - `void func() { ... }`
- `if`, `for` などの制御構文でも例外なくブレースを付ける。
- 参照・ポインタは型側に結合: `const std::string& name`, `Foo* ptr`
- 行はおおよそ 120 文字を目安に折り返し、長い式は論理単位で改行する。

## 6. ROS2 ノード実装パターン

- ノードクラスは `rclcpp::Node` を継承し、クラス名は `***Node` とする:
  - 例: `class FVObjectDetectorNode : public rclcpp::Node`
  - 例: `class AsparaUiCppNode : public rclcpp::Node`
- コンストラクタで以下を行う:
  - パラメータ宣言（`declare_parameter`）と既定値設定
  - パラメータ取得（`get_parameter` や `declare_parameter` の戻り値）による設定構造体の初期化
  - パブリッシャ・サブスクライバ・サービス・タイマーの作成
  - モデルロード・デバイス初期化などの重い処理
- QoS はパラメータから制御可能にし、`best_effort` / `reliable` を切り替えられるようにする。
- ログ出力には `RCLCPP_INFO/WARN/ERROR/DEBUG` を使い、重要な設定値やエラー理由を具体的に出す。

## 7. パラメータと設定の扱い

- ノード起動時にすべてのパラメータを `declare_parameter` で宣言し、既定値をコード側に明示する。
- 取得時は:
  - 直接戻り値を変数に代入するパターン（`auto x = declare_parameter<T>("name", default)`）
  - 一旦宣言のみしてから `get_parameter` で構造体に詰めるパターン
- パラメータ名には論理的な階層を持たせる:
  - `audio.device_selector.*`, `tracking.*`, `metrics.*`, `selection.*` など
- パラメータから得た生値に対しては **妥当性チェックと補正** を行い、ログで警告する（例: パレット長が3の倍数か、閾値の範囲など）。

## 8. 画像処理・メッセージ変換

- OpenCV の行列型は `cv::Mat` を用い、`cv_bridge` / ラッパークラスで ROS メッセージと相互変換する。
- 画像エンコーディングは `BGR8` を基本とし、`sensor_msgs::image_encodings` を利用して明示する。
- `fluent_image::Image` や `fluent_lib` のユーティリティが存在する場合は、それらを優先的に利用し重複実装を避ける。
- 未知エンコーディングや特殊ケースでは「フォールバック処理」を実装し、クラッシュではなくログ＋安全なデフォルト動作を優先する。

## 9. 例外・エラー処理

- 「致命的で起動継続が難しい」ケースは `std::runtime_error` などの例外を投げる（例: モデルロードに失敗したとき）。
- 「リカバリ可能」または「一部機能だけ失敗」するケースでは例外は投げず、`RCLCPP_WARN/ERROR` ログを残して処理をスキップ・リトライする。
- 外部ライブラリ呼び出し（ALSA, OpenVINO 等）の戻り値は必ずチェックし、失敗時にはエラー内容を `snd_strerror` などでログに出す。
- `try/catch` ブロックでは、キャッチした例外メッセージをそのままログに書き出す。

## 10. 並行処理・スレッド

- 背景処理が必要な場合は:
  - `std::thread` と RAII なラッパクラス（`fluent_lib::async::Worker`）を利用し、デストラクタで `join()` する。
  - 共有状態の保護には `std::mutex` と `std::lock_guard<std::mutex>` を用いる。
- ALSA キャプチャなどの長時間処理は専用スレッドに分離し、ROS コールバックはレスポンス良く保つ。
- 静的変数やグローバル変数は必要最小限とし、基本はクラスメンバで状態を管理する。

## 11. コメント・ドキュメント

- 公開クラス・主要なノード実装には Doxygen 形式のコメントを付ける:
  - `@brief`, `@details`, `@param`, `@return` など
  - 日本語で「何を・なぜ行っているか」を簡潔に説明する。
- 実装内のブロックは `// ===== セクション名 =====` のような区切りコメントで整理することが多い。
- ログメッセージもユーザー・オペレータ視点で分かりやすい日本語・簡潔な英語を心がける。

## 12. その他の実装方針

- 既存コードが採用している C++ 機能（`auto`, `std::unique_ptr`, `std::make_unique`, 範囲for 等）を積極的に利用し、古い C スタイルよりもモダン C++ を優先する。
- マジックナンバーは極力 `constexpr` 定数やパラメータに切り出す。
- 新しいユーティリティは、使い回しが見込める場合は `fluent_lib` などの共通ライブラリに配置する。
- 既存の設計・API と矛盾する変更を行う場合は、事前にドキュメント（`docs/` 配下）で意図と影響範囲を明文化する。


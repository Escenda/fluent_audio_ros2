// FluentLib unified public header (compat layer)
#pragma once

// 集約ヘッダー（既存プロジェクト互換）
// - 既存コードはこの1つをインクルードすればOK
// - 依存が見つからないもの（未実装のUI等）は読み込みを避けてビルドを通す

// ルートの簡易 API 集約
#include <fluent.hpp>

// FluentCloud（ヘッダー実装の軽量ファサード）
#include "fluent_lib/fluent_cloud/io.hpp"
#include "fluent_lib/fluent_cloud/filters.hpp"
#include "fluent_lib/fluent_cloud/pipeline.hpp"

// ROS ヘルパ
#include "fluent_lib/ros/timer.hpp"
#include "fluent_lib/ros/params.hpp"
#include "fluent_lib/ros/param_binder.hpp"
#include "fluent_lib/ros/param_dict.hpp"
#include "fluent_lib/ros/pubsub.hpp"
#include "fluent_lib/ros/log.hpp"
#include "fluent_lib/ros/qos.hpp"
#include "fluent_lib/ros/dsl.hpp"
#include "fluent_lib/ros/timer_registry.hpp"
#include "fluent_lib/ros/fluent_node.hpp"

// 非必須の UI / utils は未使用のため除外（不足でビルド落ちを避ける）

// 画像ラッパ
#include "fluent_lib/fluent_image/image.hpp"

// エイリアス（既存コード互換）
using FluentImage = fluent_image::Image;


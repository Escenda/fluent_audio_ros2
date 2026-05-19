#pragma once

#include <functional>
#include <memory>
#include <string>

#include "fa_in/backends/source_backend.hpp"

namespace fa_in::backends
{

struct SourceBackendSettings
{
  std::string name;
};

using SourceBackendFactory = std::function<std::unique_ptr<SourceBackend>()>;

/**
 * @brief production 用 ALSA source backend factory を返す。
 *
 * `fa_in_node` の contract test では fake backend factory を注入するため、
 * default factory は concrete ALSA backend 生成だけを担当する ROS-free 境界に置く。
 */
SourceBackendFactory defaultAlsaCaptureBackendFactory();

/**
 * @brief `backend.name` から concrete source backend を生成する。
 *
 * missing / unknown backend name は別 source へ fallback せず例外で fail closed にする。
 */
std::unique_ptr<SourceBackend> buildSourceBackend(const SourceBackendSettings & settings);

/**
 * @brief ALSA backend factory injection 付き source backend 生成。
 *
 * 実 ALSA device を使わない node core test のために ALSA backend だけ注入可能にする。
 * file / network backend はこの ROS-free factory 境界で直接生成し、node から concrete class を隠す。
 */
std::unique_ptr<SourceBackend> buildSourceBackend(
  const SourceBackendSettings & settings,
  const SourceBackendFactory & alsa_capture_backend_factory);

}  // namespace fa_in::backends

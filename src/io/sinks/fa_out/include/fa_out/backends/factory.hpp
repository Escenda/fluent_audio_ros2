#pragma once

#include <functional>
#include <memory>
#include <string>

#include "fa_out/backends/alsa_playback_backend.hpp"
#include "fa_out/backends/network_pcm_sender_backend.hpp"
#include "fa_out/backends/pcm_file_writer_backend.hpp"
#include "fa_out/backends/sink_backend.hpp"

namespace fa_out::backends
{

struct SinkBackendSettings
{
  std::string name{};
  AlsaPlaybackConfig alsa_playback{};
  PcmFileWriterConfig pcm_file_writer{};
  NetworkPcmSenderConfig network_pcm_sender{};
};

using AlsaPlaybackBackendFactory =
  std::function<std::unique_ptr<SinkBackend>(const AlsaPlaybackConfig &)>;

/**
 * @brief production 用 ALSA sink backend factory を返す。
 *
 * `fa_out_node` の contract test では fake backend factory を注入するため、
 * default factory は concrete ALSA backend 生成だけを担当する ROS-free 境界に置く。
 */
AlsaPlaybackBackendFactory defaultAlsaPlaybackBackendFactory();

/**
 * @brief `backend.name` から concrete sink backend を生成する。
 *
 * missing / unknown backend name は別 sink へ fallback せず例外で fail closed にする。
 */
std::unique_ptr<SinkBackend> buildSinkBackend(const SinkBackendSettings & settings);

/**
 * @brief ALSA backend factory injection 付き sink backend 生成。
 *
 * 実 ALSA device を使わない node core test のために ALSA backend だけ注入可能にする。
 * file / network backend はこの ROS-free factory 境界で直接生成し、node から concrete class を隠す。
 */
std::unique_ptr<SinkBackend> buildSinkBackend(
  const SinkBackendSettings & settings,
  const AlsaPlaybackBackendFactory & alsa_playback_backend_factory);

}  // namespace fa_out::backends

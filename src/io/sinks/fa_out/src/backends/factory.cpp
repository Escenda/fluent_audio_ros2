#include "fa_out/backends/factory.hpp"

#include <stdexcept>

namespace fa_out::backends
{

namespace
{
constexpr const char * kBackendAlsaPlayback = "alsa_playback";
constexpr const char * kBackendPcmFileWriter = "pcm_file_writer";
constexpr const char * kBackendNetworkPcmSender = "network_pcm_sender";
}  // namespace

AlsaPlaybackBackendFactory defaultAlsaPlaybackBackendFactory()
{
  return [](const AlsaPlaybackConfig & config) {
    return std::make_unique<AlsaPlaybackBackend>(config);
  };
}

std::unique_ptr<SinkBackend> buildSinkBackend(const SinkBackendSettings & settings)
{
  return buildSinkBackend(settings, defaultAlsaPlaybackBackendFactory());
}

std::unique_ptr<SinkBackend> buildSinkBackend(
  const SinkBackendSettings & settings,
  const AlsaPlaybackBackendFactory & alsa_playback_backend_factory)
{
  if (settings.name.empty()) {
    throw std::runtime_error("backend.name is required");
  }
  if (settings.name == kBackendAlsaPlayback) {
    if (!alsa_playback_backend_factory) {
      throw std::runtime_error("fa_out ALSA backend factory is required");
    }
    auto backend = alsa_playback_backend_factory(settings.alsa_playback);
    if (!backend) {
      throw std::runtime_error("fa_out backend factory returned null backend");
    }
    return backend;
  }
  if (settings.name == kBackendPcmFileWriter) {
    return std::make_unique<PcmFileWriterBackend>(settings.pcm_file_writer);
  }
  if (settings.name == kBackendNetworkPcmSender) {
    return std::make_unique<NetworkPcmSenderBackend>(settings.network_pcm_sender);
  }
  throw std::runtime_error("unsupported fa_out backend.name: " + settings.name);
}

}  // namespace fa_out::backends

#include "fa_in/backends/factory.hpp"

#include <stdexcept>

#include "fa_in/backends/alsa_capture_backend.hpp"
#include "fa_in/backends/network_pcm_receiver_backend.hpp"
#include "fa_in/backends/pcm_file_reader_backend.hpp"

namespace fa_in::backends
{

namespace
{
constexpr const char * kBackendAlsaCapture = "alsa_capture";
constexpr const char * kBackendPcmFileReader = "pcm_file_reader";
constexpr const char * kBackendNetworkPcmReceiver = "network_pcm_receiver";
}  // namespace

SourceBackendFactory defaultAlsaCaptureBackendFactory()
{
  return []() {
    return std::make_unique<AlsaCaptureBackend>();
  };
}

std::unique_ptr<SourceBackend> buildSourceBackend(const SourceBackendSettings & settings)
{
  return buildSourceBackend(settings, defaultAlsaCaptureBackendFactory());
}

std::unique_ptr<SourceBackend> buildSourceBackend(
  const SourceBackendSettings & settings,
  const SourceBackendFactory & alsa_capture_backend_factory)
{
  if (settings.name.empty()) {
    throw std::runtime_error("backend.name is required");
  }
  if (settings.name == kBackendAlsaCapture) {
    if (!alsa_capture_backend_factory) {
      throw std::runtime_error("fa_in ALSA backend factory is required");
    }
    auto backend = alsa_capture_backend_factory();
    if (!backend) {
      throw std::runtime_error("fa_in backend factory returned null backend");
    }
    return backend;
  }
  if (settings.name == kBackendPcmFileReader) {
    return std::make_unique<PcmFileReaderBackend>();
  }
  if (settings.name == kBackendNetworkPcmReceiver) {
    return std::make_unique<NetworkPcmReceiverBackend>();
  }
  throw std::runtime_error("unsupported fa_in backend.name: " + settings.name);
}

}  // namespace fa_in::backends

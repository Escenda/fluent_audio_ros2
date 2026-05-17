#include "fa_in/backends/alsa_capture_backend.hpp"

#include <alsa/asoundlib.h>

#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace fa_in
{
namespace backends
{

namespace
{
constexpr const char* kEncodingPcm16 = "PCM16LE";
constexpr const char* kEncodingPcm32 = "PCM32LE";
constexpr const char* kEncodingFloat32 = "FLOAT32LE";

void silenceAlsaErrors(
  const char* /*file*/, int /*line*/, const char* /*function*/, int /*err*/, const char* /*fmt*/, ...)
{
  // ALSA prints directly to stderr when devices are missing. fa_in surfaces these failures through BackendError.
}

bool isRawAlsaHardwareSource(const std::string& source_id)
{
  return source_id.rfind("hw:", 0) == 0;
}

std::string displayName(const DeviceInfo& device)
{
  if (device.name.empty()) {
    return device.id;
  }
  return device.name;
}

snd_pcm_format_t alsaFormatForConfig(const AudioFormat& format)
{
  if (format.encoding == kEncodingPcm16 && format.bit_depth == 16) {
    return SND_PCM_FORMAT_S16_LE;
  }
  if (format.encoding == kEncodingPcm32 && format.bit_depth == 32) {
    return SND_PCM_FORMAT_S32_LE;
  }
  if (format.encoding == kEncodingFloat32 && format.bit_depth == 32) {
    return SND_PCM_FORMAT_FLOAT_LE;
  }
  return SND_PCM_FORMAT_UNKNOWN;
}

std::string alsaError(const std::string& operation, int err)
{
  return operation + ": " + snd_strerror(err);
}
}  // namespace

class AlsaCaptureBackend::Impl
{
public:
  Impl()
  {
    snd_lib_error_set_handler(silenceAlsaErrors);
  }

  ~Impl()
  {
    close();
  }

  std::vector<DeviceInfo> listDevices() const
  {
    std::vector<DeviceInfo> devices;
    void** hints = nullptr;
    if (snd_device_name_hint(-1, "pcm", &hints) != 0) {
      return devices;
    }

    for (void** hint = hints; *hint != nullptr; ++hint) {
      char* name = snd_device_name_get_hint(*hint, "NAME");
      char* desc = snd_device_name_get_hint(*hint, "DESC");
      char* io = snd_device_name_get_hint(*hint, "IOID");
      const bool is_input = (io == nullptr) || (std::strcmp(io, "Input") == 0);
      if (name != nullptr && is_input) {
        const std::string source_id{name};
        if (isRawAlsaHardwareSource(source_id)) {
          devices.push_back(DeviceInfo{source_id, desc != nullptr ? std::string(desc) : ""});
        }
      }
      if (name != nullptr) {
        free(name);
      }
      if (desc != nullptr) {
        free(desc);
      }
      if (io != nullptr) {
        free(io);
      }
    }
    snd_device_name_free_hint(hints);

    return devices;
  }

  DeviceInfo selectDevice(const DeviceSelector& selector) const
  {
    const auto devices = listDevices();
    if (devices.empty()) {
      throw BackendError("No ALSA input source candidates were enumerated");
    }

    if (selector.mode == "index") {
      if (selector.index >= 0 && static_cast<size_t>(selector.index) < devices.size()) {
        return devices[selector.index];
      }
      throw BackendError("Configured ALSA input source index is invalid: " + std::to_string(selector.index));
    }

    if (selector.mode == "name") {
      if (selector.identifier.empty()) {
        throw BackendError("audio.device_selector.identifier is required when mode is name");
      }
      for (const auto& device : devices) {
        const std::string label = displayName(device);
        if (device.id == selector.identifier || label == selector.identifier) {
          return device;
        }
      }
      throw BackendError("Configured ALSA input source name was not found: " + selector.identifier);
    }

    throw BackendError("Unsupported audio.device_selector.mode: " + selector.mode);
  }

  size_t open(const std::string& device_id, const AudioFormat& format, size_t requested_frames)
  {
    close();

    if (device_id.empty()) {
      throw BackendError("audio input source id is required");
    }

    int err = snd_pcm_open(&pcm_handle_, device_id.c_str(), SND_PCM_STREAM_CAPTURE, 0);
    if (err < 0) {
      pcm_handle_ = nullptr;
      throw BackendError("snd_pcm_open failed for " + device_id + ": " + snd_strerror(err));
    }

    snd_pcm_hw_params_t* params = nullptr;
    err = snd_pcm_hw_params_malloc(&params);
    if (err < 0) {
      close();
      throw BackendError(alsaError("snd_pcm_hw_params_malloc failed", err));
    }

    try {
      configureHardwareParams(params, format, requested_frames);
    } catch (...) {
      snd_pcm_hw_params_free(params);
      close();
      throw;
    }

    snd_pcm_hw_params_free(params);

    err = snd_pcm_prepare(pcm_handle_);
    if (err < 0) {
      close();
      throw BackendError(alsaError("snd_pcm_prepare failed", err));
    }

    return configured_period_size_;
  }

  void close()
  {
    if (pcm_handle_ == nullptr) {
      return;
    }
    snd_pcm_drop(pcm_handle_);
    snd_pcm_close(pcm_handle_);
    pcm_handle_ = nullptr;
  }

  void drop()
  {
    if (pcm_handle_ != nullptr) {
      snd_pcm_drop(pcm_handle_);
    }
  }

  ReadResult read(uint8_t* data, size_t frames)
  {
    if (pcm_handle_ == nullptr) {
      return ReadResult{ReadStatus::kError, 0, "Capture loop started without ALSA handle"};
    }

    const snd_pcm_sframes_t captured_frames = snd_pcm_readi(pcm_handle_, data, frames);
    if (captured_frames == -EPIPE) {
      return ReadResult{ReadStatus::kXrun, 0, "ALSA capture XRUN"};
    }
    if (captured_frames < 0) {
      return ReadResult{
        ReadStatus::kError,
        0,
        snd_strerror(static_cast<int>(captured_frames))};
    }
    if (captured_frames == 0) {
      return ReadResult{ReadStatus::kZeroFrames, 0, "snd_pcm_readi returned zero frames"};
    }

    return ReadResult{ReadStatus::kOk, static_cast<size_t>(captured_frames), ""};
  }

private:
  void configureHardwareParams(snd_pcm_hw_params_t* params, const AudioFormat& format, size_t requested_frames)
  {
    int err = snd_pcm_hw_params_any(pcm_handle_, params);
    if (err < 0) {
      throw BackendError(alsaError("snd_pcm_hw_params_any failed", err));
    }

    err = snd_pcm_hw_params_set_rate_resample(pcm_handle_, params, 0);
    if (err < 0) {
      throw BackendError(alsaError("Failed to disable ALSA software resampling", err));
    }

    err = snd_pcm_hw_params_set_access(pcm_handle_, params, SND_PCM_ACCESS_RW_INTERLEAVED);
    if (err < 0) {
      throw BackendError(alsaError("Failed to set access", err));
    }

    const snd_pcm_format_t alsa_format = alsaFormatForConfig(format);
    if (alsa_format == SND_PCM_FORMAT_UNKNOWN) {
      throw BackendError("audio.encoding/audio.bit_depth must be one of PCM16LE/16, PCM32LE/32, FLOAT32LE/32");
    }
    err = snd_pcm_hw_params_set_format(pcm_handle_, params, alsa_format);
    if (err < 0) {
      throw BackendError(alsaError("Failed to set format", err));
    }

    unsigned int channels = format.channels;
    err = snd_pcm_hw_params_set_channels(pcm_handle_, params, channels);
    if (err < 0) {
      throw BackendError(alsaError("Failed to set channels", err));
    }

    unsigned int rate = format.sample_rate;
    int dir = 0;
    err = snd_pcm_hw_params_set_rate(pcm_handle_, params, rate, dir);
    if (err < 0) {
      throw BackendError(alsaError("Failed to set rate", err));
    }

    snd_pcm_uframes_t period_size = static_cast<snd_pcm_uframes_t>(requested_frames);
    err = snd_pcm_hw_params_set_period_size_near(pcm_handle_, params, &period_size, &dir);
    if (err < 0) {
      throw BackendError(alsaError("Failed to set period size", err));
    }

    err = snd_pcm_hw_params(pcm_handle_, params);
    if (err < 0) {
      throw BackendError(alsaError("Failed to apply hw params", err));
    }

    configured_period_size_ = static_cast<size_t>(period_size);
  }

  snd_pcm_t* pcm_handle_{nullptr};
  size_t configured_period_size_{0};
};

AlsaCaptureBackend::AlsaCaptureBackend()
: impl_(std::make_unique<Impl>())
{
}

AlsaCaptureBackend::~AlsaCaptureBackend() = default;

std::vector<DeviceInfo> AlsaCaptureBackend::listDevices() const
{
  return impl_->listDevices();
}

DeviceInfo AlsaCaptureBackend::selectDevice(const DeviceSelector& selector) const
{
  return impl_->selectDevice(selector);
}

size_t AlsaCaptureBackend::open(const std::string& device_id, const AudioFormat& format, size_t requested_frames)
{
  return impl_->open(device_id, format, requested_frames);
}

void AlsaCaptureBackend::close()
{
  impl_->close();
}

void AlsaCaptureBackend::drop()
{
  impl_->drop();
}

ReadResult AlsaCaptureBackend::read(uint8_t* data, size_t frames)
{
  return impl_->read(data, frames);
}

}  // namespace backends
}  // namespace fa_in

#include "fa_out/backends/alsa_playback_backend.hpp"

#include <alsa/asoundlib.h>

#include <cerrno>
#include <limits>
#include <utility>

namespace fa_out::backends
{

namespace
{
constexpr const char * kEncodingPcm16 = "PCM16LE";

std::string alsaErrorMessage(const char * operation, const int error_code)
{
  return std::string(operation) + " failed: " + snd_strerror(error_code);
}

void closeHandle(snd_pcm_t * handle)
{
  if (handle != nullptr) {
    snd_pcm_close(handle);
  }
}

snd_pcm_uframes_t toAlsaFrames(const char * config_name, const size_t frames)
{
  if (frames > std::numeric_limits<snd_pcm_uframes_t>::max()) {
    throw AlsaPlaybackError(std::string(config_name) + " exceeds ALSA frame count range");
  }
  return static_cast<snd_pcm_uframes_t>(frames);
}
}  // namespace

struct AlsaPlaybackBackend::Impl
{
  snd_pcm_t * pcm_handle{nullptr};
};

AlsaPlaybackError::AlsaPlaybackError(const std::string & message)
: SinkBackendError(message)
{
}

AlsaPlaybackBackend::AlsaPlaybackBackend(AlsaPlaybackConfig config)
: config_(std::move(config)), impl_(std::make_unique<Impl>())
{
}

AlsaPlaybackBackend::~AlsaPlaybackBackend()
{
  close();
}

bool AlsaPlaybackBackend::isRawHardwareDevice(const std::string & device_id)
{
  return device_id.rfind("hw:", 0) == 0;
}

SinkOpenInfo AlsaPlaybackBackend::open()
{
  validateConfig();
  close();

  const snd_pcm_uframes_t requested_buffer_size =
    toAlsaFrames("audio.alsa.buffer_frames", config_.buffer_frames);
  const snd_pcm_uframes_t requested_period_size =
    toAlsaFrames("audio.alsa.period_frames", config_.period_frames);

  snd_pcm_t * handle = nullptr;
  int err = snd_pcm_open(&handle, config_.device_id.c_str(), SND_PCM_STREAM_PLAYBACK, 0);
  if (err < 0) {
    throw AlsaPlaybackError(alsaErrorMessage("snd_pcm_open", err));
  }

  snd_pcm_hw_params_t * hw_params;
  snd_pcm_hw_params_alloca(&hw_params);

  err = snd_pcm_hw_params_any(handle, hw_params);
  if (err < 0) {
    closeHandle(handle);
    throw AlsaPlaybackError(alsaErrorMessage("snd_pcm_hw_params_any", err));
  }

  err = snd_pcm_hw_params_set_rate_resample(handle, hw_params, 0);
  if (err < 0) {
    closeHandle(handle);
    throw AlsaPlaybackError(
      std::string("Failed to disable ALSA software resampling: ") + snd_strerror(err));
  }

  err = snd_pcm_hw_params_set_access(handle, hw_params, SND_PCM_ACCESS_RW_INTERLEAVED);
  if (err < 0) {
    closeHandle(handle);
    throw AlsaPlaybackError(alsaErrorMessage("snd_pcm_hw_params_set_access", err));
  }

  err = snd_pcm_hw_params_set_format(handle, hw_params, SND_PCM_FORMAT_S16_LE);
  if (err < 0) {
    closeHandle(handle);
    throw AlsaPlaybackError(alsaErrorMessage("snd_pcm_hw_params_set_format", err));
  }

  err = snd_pcm_hw_params_set_channels(handle, hw_params, config_.channels);
  if (err < 0) {
    closeHandle(handle);
    throw AlsaPlaybackError(alsaErrorMessage("snd_pcm_hw_params_set_channels", err));
  }

  unsigned int rate = config_.sample_rate;
  err = snd_pcm_hw_params_set_rate(handle, hw_params, rate, 0);
  if (err < 0) {
    closeHandle(handle);
    throw AlsaPlaybackError(alsaErrorMessage("snd_pcm_hw_params_set_rate", err));
  }

  snd_pcm_uframes_t buffer_size = requested_buffer_size;
  err = snd_pcm_hw_params_set_buffer_size_near(handle, hw_params, &buffer_size);
  if (err < 0) {
    closeHandle(handle);
    throw AlsaPlaybackError(alsaErrorMessage("snd_pcm_hw_params_set_buffer_size_near", err));
  }
  if (buffer_size != requested_buffer_size) {
    closeHandle(handle);
    throw AlsaPlaybackError(
      "ALSA buffer size negotiation changed requested playback buffer from " +
      std::to_string(static_cast<unsigned long>(requested_buffer_size)) + " frames to " +
      std::to_string(static_cast<unsigned long>(buffer_size)) + " frames");
  }

  snd_pcm_uframes_t period_size = requested_period_size;
  err = snd_pcm_hw_params_set_period_size_near(handle, hw_params, &period_size, 0);
  if (err < 0) {
    closeHandle(handle);
    throw AlsaPlaybackError(alsaErrorMessage("snd_pcm_hw_params_set_period_size_near", err));
  }
  if (period_size != requested_period_size) {
    closeHandle(handle);
    throw AlsaPlaybackError(
      "ALSA period size negotiation changed requested playback period from " +
      std::to_string(static_cast<unsigned long>(requested_period_size)) + " frames to " +
      std::to_string(static_cast<unsigned long>(period_size)) + " frames");
  }

  err = snd_pcm_hw_params(handle, hw_params);
  if (err < 0) {
    closeHandle(handle);
    throw AlsaPlaybackError(alsaErrorMessage("snd_pcm_hw_params", err));
  }

  SinkOpenInfo open_info;
  open_info.info_messages.push_back(
    "ALSA buffer: " + std::to_string(static_cast<unsigned long>(buffer_size)) +
    " frames, period: " + std::to_string(static_cast<unsigned long>(period_size)) + " frames");

  snd_pcm_sw_params_t * sw_params;
  snd_pcm_sw_params_alloca(&sw_params);

  err = snd_pcm_sw_params_current(handle, sw_params);
  if (err < 0) {
    closeHandle(handle);
    throw AlsaPlaybackError(alsaErrorMessage("snd_pcm_sw_params_current", err));
  }

  const snd_pcm_uframes_t start_threshold = (buffer_size / 2) + (buffer_size % 2);
  err = snd_pcm_sw_params_set_start_threshold(handle, sw_params, start_threshold);
  if (err < 0) {
    closeHandle(handle);
    throw AlsaPlaybackError(alsaErrorMessage("snd_pcm_sw_params_set_start_threshold", err));
  }

  err = snd_pcm_sw_params_set_avail_min(handle, sw_params, period_size);
  if (err < 0) {
    closeHandle(handle);
    throw AlsaPlaybackError(alsaErrorMessage("snd_pcm_sw_params_set_avail_min", err));
  }

  err = snd_pcm_sw_params(handle, sw_params);
  if (err < 0) {
    closeHandle(handle);
    throw AlsaPlaybackError(alsaErrorMessage("snd_pcm_sw_params", err));
  }
  open_info.info_messages.push_back(
    "ALSA software params: start_threshold=" +
    std::to_string(static_cast<unsigned long>(start_threshold)) + " frames");

  impl_->pcm_handle = handle;
  return open_info;
}

void AlsaPlaybackBackend::close()
{
  closeHandle(impl_->pcm_handle);
  impl_->pcm_handle = nullptr;
}

bool AlsaPlaybackBackend::isOpen() const
{
  return impl_->pcm_handle != nullptr;
}

bool AlsaPlaybackBackend::isRunning() const
{
  if (impl_->pcm_handle == nullptr) {
    return false;
  }
  return snd_pcm_state(impl_->pcm_handle) == SND_PCM_STATE_RUNNING;
}

void AlsaPlaybackBackend::discardBuffer(const std::string & operation)
{
  if (impl_->pcm_handle == nullptr) {
    throw AlsaPlaybackError(operation + " requested without an open ALSA playback device");
  }

  int err = snd_pcm_drop(impl_->pcm_handle);
  if (err < 0) {
    throw AlsaPlaybackError(
      std::string("snd_pcm_drop failed during ") + operation + ": " + snd_strerror(err));
  }

  err = snd_pcm_prepare(impl_->pcm_handle);
  if (err < 0) {
    throw AlsaPlaybackError(
      std::string("snd_pcm_prepare failed during ") + operation + ": " + snd_strerror(err));
  }
}

size_t AlsaPlaybackBackend::writeFrames(const uint8_t * data, const size_t frame_count)
{
  if (impl_->pcm_handle == nullptr) {
    throw AlsaPlaybackError("ALSA playback handle closed while the configured sink is required");
  }

  const snd_pcm_sframes_t result = snd_pcm_writei(
    impl_->pcm_handle,
    data,
    static_cast<snd_pcm_uframes_t>(frame_count));

  if (result == -EPIPE) {
    throw AlsaPlaybackError("ALSA playback XRUN on required sink " + config_.device_id);
  }
  if (result == -EAGAIN) {
    throw AlsaPlaybackError("snd_pcm_writei returned EAGAIN on required sink " + config_.device_id);
  }
  if (result < 0) {
    throw AlsaPlaybackError(
      "snd_pcm_writei failed on required sink " + config_.device_id + ": " +
      snd_strerror(static_cast<int>(result)));
  }
  if (result == 0) {
    throw AlsaPlaybackError("snd_pcm_writei wrote zero frames on required sink " + config_.device_id);
  }

  return static_cast<size_t>(result);
}

void AlsaPlaybackBackend::validateConfig() const
{
  if (config_.device_id.empty()) {
    throw AlsaPlaybackError("audio.device_id is required for backend.name=alsa_playback");
  }
  if (config_.encoding != kEncodingPcm16) {
    throw AlsaPlaybackError("audio.encoding must be PCM16LE for backend.name=alsa_playback");
  }
  if (!isRawHardwareDevice(config_.device_id)) {
    throw AlsaPlaybackError(
      "audio.device_id must be an ALSA raw hardware id starting with hw: for backend.name=alsa_playback");
  }
  if (config_.sample_rate == 0) {
    throw AlsaPlaybackError("audio.sample_rate must be > 0");
  }
  if (config_.channels == 0) {
    throw AlsaPlaybackError("audio.channels must be > 0");
  }
  if (config_.bit_depth != 16) {
    throw AlsaPlaybackError("audio.bit_depth must be 16 for PCM16LE playback");
  }
  if (config_.buffer_frames == 0) {
    throw AlsaPlaybackError("audio.alsa.buffer_frames must be > 0");
  }
  if (config_.period_frames == 0) {
    throw AlsaPlaybackError("audio.alsa.period_frames must be > 0");
  }
  if (config_.period_frames > config_.buffer_frames) {
    throw AlsaPlaybackError("audio.alsa.period_frames must be <= audio.alsa.buffer_frames");
  }
}

}  // namespace fa_out::backends

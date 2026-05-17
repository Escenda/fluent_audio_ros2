#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace fa_out::validation
{

inline uint32_t requirePositiveUint32(const char * parameter_name, const int64_t value)
{
  if (value <= 0) {
    throw std::invalid_argument(std::string(parameter_name) + " must be > 0");
  }
  if (static_cast<uint64_t>(value) > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument(std::string(parameter_name) + " exceeds uint32 range");
  }
  return static_cast<uint32_t>(value);
}

inline size_t requirePositiveSize(const char * parameter_name, const int64_t value)
{
  if (value <= 0) {
    throw std::invalid_argument(std::string(parameter_name) + " must be > 0");
  }
  if (static_cast<uint64_t>(value) > std::numeric_limits<size_t>::max()) {
    throw std::invalid_argument(std::string(parameter_name) + " exceeds size_t range");
  }
  return static_cast<size_t>(value);
}

inline size_t playbackChunkFrames(const uint32_t sample_rate, const uint32_t chunk_duration_ms)
{
  const uint64_t frame_product =
    static_cast<uint64_t>(sample_rate) * static_cast<uint64_t>(chunk_duration_ms);
  if (frame_product % 1000u != 0u) {
    throw std::invalid_argument(
      "audio.sample_rate * audio.chunk_duration_ms must produce an integer playback chunk");
  }

  const uint64_t frames = frame_product / 1000u;
  if (frames == 0u) {
    throw std::invalid_argument("audio.chunk_duration_ms produces zero playback frames");
  }
  if (frames > std::numeric_limits<size_t>::max()) {
    throw std::invalid_argument("audio.chunk_duration_ms produces too many playback frames");
  }
  return static_cast<size_t>(frames);
}

inline size_t bytesPerFrame(const uint32_t channels, const uint32_t bit_depth)
{
  if ((bit_depth % 8u) != 0u) {
    throw std::invalid_argument("audio.bit_depth must be byte-aligned");
  }
  const uint64_t bytes_per_sample = bit_depth / 8u;
  if (bytes_per_sample == 0u) {
    throw std::invalid_argument("audio.bit_depth produces zero bytes per sample");
  }
  if (static_cast<uint64_t>(channels) > (std::numeric_limits<size_t>::max() / bytes_per_sample)) {
    throw std::invalid_argument("audio.channels * audio.bit_depth exceeds size_t range");
  }
  return static_cast<size_t>(channels) * static_cast<size_t>(bytes_per_sample);
}

inline size_t bytesForFrames(const char * frame_count_name, const size_t frames, const size_t bytes_per_frame)
{
  if (bytes_per_frame == 0u) {
    throw std::invalid_argument("bytes_per_frame must be > 0");
  }
  if (frames > (std::numeric_limits<size_t>::max() / bytes_per_frame)) {
    throw std::invalid_argument(std::string(frame_count_name) + " produces too many bytes");
  }
  return frames * bytes_per_frame;
}

inline bool isRawAlsaHardwareSink(const std::string & sink_id)
{
  return sink_id.rfind("hw:", 0) == 0;
}

inline void requireRawAlsaHardwareSink(const std::string & sink_id)
{
  if (sink_id.empty()) {
    throw std::invalid_argument("audio.device_id is required for backend.name=alsa_playback");
  }
  if (!isRawAlsaHardwareSink(sink_id)) {
    throw std::invalid_argument(
      "audio.device_id must be an ALSA raw hardware id starting with hw: for backend.name=alsa_playback");
  }
}

}  // namespace fa_out::validation

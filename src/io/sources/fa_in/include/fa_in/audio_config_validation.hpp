#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace fa_in::validation
{

inline uint32_t requirePositiveUint32(const char * parameter_name, const int64_t value)
{
  if (value <= 0) {
    throw std::runtime_error(std::string(parameter_name) + " must be > 0");
  }
  if (static_cast<uint64_t>(value) > std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error(std::string(parameter_name) + " exceeds uint32 range");
  }
  return static_cast<uint32_t>(value);
}

inline void requireDeviceSelector(
  const std::string & mode, const std::string & identifier, const int64_t index)
{
  if (mode == "name") {
    if (identifier.empty()) {
      throw std::runtime_error(
        "audio.device_selector.identifier is required when audio.device_selector.mode=name");
    }
    return;
  }

  if (mode == "index") {
    if (index < 0) {
      throw std::runtime_error(
        "audio.device_selector.index must be >= 0 when audio.device_selector.mode=index");
    }
    return;
  }

  throw std::runtime_error("unsupported audio.device_selector.mode: " + mode);
}

inline size_t captureFramesPerBuffer(const uint32_t sample_rate, const uint32_t chunk_ms)
{
  const uint64_t frame_product =
    static_cast<uint64_t>(sample_rate) * static_cast<uint64_t>(chunk_ms);
  if (frame_product % 1000u != 0u) {
    throw std::runtime_error("audio.sample_rate * audio.chunk_ms must produce an integer frame count");
  }

  const uint64_t frames = frame_product / 1000u;
  if (frames == 0u) {
    throw std::runtime_error("audio.chunk_ms produces zero capture frames");
  }
  if (frames > std::numeric_limits<size_t>::max()) {
    throw std::runtime_error("audio.chunk_ms produces too many capture frames");
  }
  return static_cast<size_t>(frames);
}

inline size_t bytesPerFrame(const uint32_t channels, const uint32_t bit_depth)
{
  if ((bit_depth % 8u) != 0u) {
    throw std::runtime_error("audio.bit_depth must be byte-aligned");
  }
  const uint64_t bytes_per_sample = bit_depth / 8u;
  if (bytes_per_sample == 0u) {
    throw std::runtime_error("audio.bit_depth produces zero bytes per sample");
  }
  if (static_cast<uint64_t>(channels) > (std::numeric_limits<size_t>::max() / bytes_per_sample)) {
    throw std::runtime_error("audio.channels * audio.bit_depth exceeds size_t range");
  }
  return static_cast<size_t>(channels) * static_cast<size_t>(bytes_per_sample);
}

inline size_t bytesForFrames(const char * frame_count_name, const size_t frames, const size_t bytes_per_frame)
{
  if (bytes_per_frame == 0u) {
    throw std::runtime_error("bytes_per_frame must be > 0");
  }
  if (frames > (std::numeric_limits<size_t>::max() / bytes_per_frame)) {
    throw std::runtime_error(std::string(frame_count_name) + " produces too many bytes");
  }
  return frames * bytes_per_frame;
}

}  // namespace fa_in::validation

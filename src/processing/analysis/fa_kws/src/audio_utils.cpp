#include "fa_kws/audio_utils.hpp"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace fa_kws
{

std::vector<float> frameToMonoFloat(const fa_interfaces::msg::AudioFrame &msg)
{
  if (msg.data.empty()) {
    return {};
  }

  std::vector<float> samples;
  samples.reserve(msg.data.size() / sizeof(std::int16_t));

  if (msg.bit_depth == 16) {
    const auto *raw = reinterpret_cast<const std::int16_t *>(msg.data.data());
    const std::size_t count = msg.data.size() / sizeof(std::int16_t);
    constexpr float kScale = 1.0f / 32768.0f;
    samples.resize(count);
    for (std::size_t i = 0; i < count; ++i) {
      samples[i] = static_cast<float>(raw[i]) * kScale;
    }
  } else if (msg.bit_depth == 32) {
    const auto *raw = reinterpret_cast<const float *>(msg.data.data());
    const std::size_t count = msg.data.size() / sizeof(float);
    samples.resize(count);
    for (std::size_t i = 0; i < count; ++i) {
      samples[i] = raw[i];
    }
  } else {
    throw std::invalid_argument("unsupported AudioFrame bit_depth=" + std::to_string(msg.bit_depth));
  }

  if (msg.channels <= 1u) {
    return samples;
  }

  if (samples.size() % msg.channels != 0u) {
    throw std::invalid_argument("AudioFrame sample count is not divisible by channels");
  }
  const std::size_t frames = samples.size() / msg.channels;
  if (frames == 0) {
    return {};
  }

  std::vector<float> mono(frames, 0.0f);
  for (std::size_t i = 0; i < frames; ++i) {
    float sum = 0.0f;
    for (std::size_t ch = 0; ch < msg.channels; ++ch) {
      sum += samples[i * msg.channels + ch];
    }
    mono[i] = sum / static_cast<float>(msg.channels);
  }
  return mono;
}

std::vector<float> resampleLinear(const std::vector<float> &samples,
                                  std::int32_t src_rate,
                                  std::int32_t dst_rate)
{
  if (samples.empty() || src_rate <= 0 || dst_rate <= 0 || src_rate == dst_rate) {
    if (src_rate <= 0 || dst_rate <= 0) {
      throw std::invalid_argument("sample rates must be positive");
    }
    return samples;
  }

  const double ratio = static_cast<double>(dst_rate) / static_cast<double>(src_rate);
  const std::size_t dst_len =
    std::max<std::size_t>(1, static_cast<std::size_t>(std::floor(samples.size() * ratio)));

  std::vector<float> out(dst_len);
  const double inv_ratio = 1.0 / ratio;

  for (std::size_t i = 0; i < dst_len; ++i) {
    const double pos = static_cast<double>(i) * inv_ratio;
    const std::size_t idx0 = static_cast<std::size_t>(std::floor(pos));
    const std::size_t idx1 = std::min(idx0 + 1, samples.size() - 1);
    const double t = pos - static_cast<double>(idx0);
    out[i] = static_cast<float>((1.0 - t) * samples[idx0] + t * samples[idx1]);
  }

  return out;
}

}  // namespace fa_kws

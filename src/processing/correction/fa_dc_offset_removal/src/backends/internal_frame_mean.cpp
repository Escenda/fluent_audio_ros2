#include "fa_dc_offset_removal/backends/internal_frame_mean.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <utility>

#if !defined(__BYTE_ORDER__) || !defined(__ORDER_LITTLE_ENDIAN__) || \
  (__BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__)
#error "fa_dc_offset_removal internal_frame_mean requires a little-endian target for FLOAT32LE"
#endif

namespace fa_dc_offset_removal::backends
{

InternalFrameMeanBackend::InternalFrameMeanBackend(const InternalFrameMeanConfig & config)
: config_(config)
{
  if (config_.channels <= 0) {
    throw std::runtime_error("expected.channels must be > 0");
  }
}

ProcessResult InternalFrameMeanBackend::process(
  const std::vector<uint8_t> & input,
  std::vector<uint8_t> & output) const
{
  const size_t channel_count = static_cast<size_t>(config_.channels);
  const size_t bytes_per_frame = channel_count * sizeof(float);
  if (input.empty()) {
    return ProcessResult{ProcessStatus::kEmptyInput};
  }
  if ((input.size() % bytes_per_frame) != 0) {
    return ProcessResult{ProcessStatus::kMisalignedInput};
  }

  const size_t sample_count = input.size() / sizeof(float);
  const size_t frame_count = sample_count / channel_count;
  std::vector<double> channel_sums(channel_count, 0.0);
  std::vector<float> samples(sample_count, 0.0F);

  for (size_t i = 0; i < sample_count; ++i) {
    float sample = 0.0F;
    std::memcpy(&sample, input.data() + (i * sizeof(float)), sizeof(float));
    if (!std::isfinite(sample)) {
      return ProcessResult{ProcessStatus::kNonFiniteInput};
    }
    samples.at(i) = sample;
    channel_sums.at(i % channel_count) += static_cast<double>(sample);
  }

  std::vector<double> channel_means(channel_count, 0.0);
  for (size_t channel = 0; channel < channel_count; ++channel) {
    const double mean = channel_sums.at(channel) / static_cast<double>(frame_count);
    if (!std::isfinite(mean)) {
      return ProcessResult{ProcessStatus::kNonFiniteMean};
    }
    channel_means.at(channel) = mean;
  }

  std::vector<uint8_t> next_output(input.size());
  for (size_t i = 0; i < sample_count; ++i) {
    const double corrected =
      static_cast<double>(samples.at(i)) - channel_means.at(i % channel_count);
    if (!std::isfinite(corrected)) {
      return ProcessResult{ProcessStatus::kNonFiniteOutput};
    }
    const float out_sample = static_cast<float>(corrected);
    if (!std::isfinite(out_sample)) {
      return ProcessResult{ProcessStatus::kNonFiniteOutput};
    }
    std::memcpy(next_output.data() + (i * sizeof(float)), &out_sample, sizeof(float));
  }

  output = std::move(next_output);
  return ProcessResult{ProcessStatus::kOk};
}

const char * processStatusMessage(ProcessStatus status)
{
  switch (status) {
    case ProcessStatus::kOk:
      return "ok";
    case ProcessStatus::kEmptyInput:
      return "input data is empty";
    case ProcessStatus::kMisalignedInput:
      return "input byte length is not aligned to FLOAT32LE interleaved frames";
    case ProcessStatus::kNonFiniteInput:
      return "input sample is not finite";
    case ProcessStatus::kNonFiniteMean:
      return "computed channel mean is not finite";
    case ProcessStatus::kNonFiniteOutput:
      return "output sample is not finite FLOAT32LE";
  }
  throw std::logic_error("unhandled DC offset backend process status");
}

}  // namespace fa_dc_offset_removal::backends

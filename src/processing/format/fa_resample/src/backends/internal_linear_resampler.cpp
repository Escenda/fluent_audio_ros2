#include "fa_resample/backends/internal_linear_resampler.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace fa_resample::backends
{

namespace
{
uint64_t elapsedNs(const std::chrono::steady_clock::time_point & start)
{
  const auto elapsed = std::chrono::steady_clock::now() - start;
  return static_cast<uint64_t>(
    std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count());
}

}  // namespace

std::vector<float> resampleLinear(
  const std::vector<float> & interleaved,
  const uint32_t in_rate,
  const uint32_t out_rate,
  const uint32_t channels,
  const uint32_t in_frames,
  uint32_t & out_frames)
{
  out_frames = 0;
  if (in_rate == 0 || out_rate == 0 || channels == 0 || in_frames == 0 || interleaved.empty()) {
    return {};
  }
  if (interleaved.size() != static_cast<size_t>(in_frames) * channels) {
    return {};
  }
  if (!containsOnlyFiniteNormalizedSamples(interleaved)) {
    return {};
  }

  if (in_rate == out_rate) {
    out_frames = in_frames;
    return interleaved;
  }

  const double ratio = static_cast<double>(out_rate) / static_cast<double>(in_rate);
  const double out_frames_f = static_cast<double>(in_frames) * ratio;
  const uint32_t frames = static_cast<uint32_t>(std::max<double>(1.0, std::lround(out_frames_f)));

  std::vector<float> out;
  out.resize(static_cast<size_t>(frames) * channels);

  const double step = static_cast<double>(in_rate) / static_cast<double>(out_rate);
  for (uint32_t i = 0; i < frames; ++i) {
    const double src_pos = static_cast<double>(i) * step;
    uint32_t idx0 = static_cast<uint32_t>(std::floor(src_pos));
    const double frac = src_pos - static_cast<double>(idx0);

    if (idx0 >= in_frames) {
      idx0 = in_frames - 1;
    }
    const uint32_t idx1 = std::min<uint32_t>(idx0 + 1, in_frames - 1);

    for (uint32_t ch = 0; ch < channels; ++ch) {
      const float s0 = interleaved.at(static_cast<size_t>(idx0) * channels + ch);
      const float s1 = interleaved.at(static_cast<size_t>(idx1) * channels + ch);
      const float value = static_cast<float>(
        (1.0 - frac) * static_cast<double>(s0) + frac * static_cast<double>(s1));
      out.at(static_cast<size_t>(i) * channels + ch) = value;
    }
  }

  if (!containsOnlyFiniteNormalizedSamples(out)) {
    return {};
  }

  out_frames = frames;
  return out;
}

InternalLinearResamplerBackend::InternalLinearResamplerBackend(
  const InternalLinearResamplerConfig & config)
: config_(config)
{
  if (config_.target_sample_rate <= 0) {
    throw std::runtime_error("target_sample_rate must be > 0");
  }
}

std::string InternalLinearResamplerBackend::name() const
{
  return kName;
}

std::string InternalLinearResamplerBackend::quality() const
{
  return kQuality;
}

int InternalLinearResamplerBackend::targetSampleRate() const
{
  return config_.target_sample_rate;
}

BackendMetrics InternalLinearResamplerBackend::metrics() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  return metrics_;
}

ProcessResult InternalLinearResamplerBackend::process(
  const ProcessRequest & request,
  std::vector<uint8_t> & output)
{
  const FrameContractStatus frame_contract_status = validateFloat32InterleavedContract(request.contract);
  if (frame_contract_status != FrameContractStatus::kOk) {
    return ProcessResult{ProcessStatus::kInvalidFrameContract, frame_contract_status, 0};
  }
  if (request.stream_id.empty()) {
    return ProcessResult{
      ProcessStatus::kInvalidFrameContract,
      FrameContractStatus::kInvalidStreamIdentity,
      0};
  }

  const std::vector<float> in_f32 = decodeFloat32Le(request.input);
  if (!containsOnlyFiniteNormalizedSamples(in_f32)) {
    return ProcessResult{ProcessStatus::kInvalidInputSamples, FrameContractStatus::kOk, 0};
  }

  const uint32_t in_frames = frameCountFromContract(request.contract);
  if (in_frames == 0) {
    return ProcessResult{ProcessStatus::kInvalidFrameContract, FrameContractStatus::kUnalignedData, 0};
  }

  const auto started_at = std::chrono::steady_clock::now();
  std::lock_guard<std::mutex> lock(mutex_);
  const StreamContract next_contract = makeStreamContract(
    request.stream_id,
    request.contract,
    config_.target_sample_rate,
    kName,
    kQuality);

  auto stream_it = streams_.find(request.stream_id);
  if (stream_it == streams_.end()) {
    StreamState state;
    state.contract = next_contract;
    state.buffer_start_frame_index = 0;
    stream_it = streams_.emplace(request.stream_id, std::move(state)).first;
  } else if (!streamContractMatches(stream_it->second.contract, next_contract)) {
    return ProcessResult{ProcessStatus::kStreamContractViolation, FrameContractStatus::kOk, 0};
  }

  ProcessResult result = processValidatedFrame(stream_it->second, in_f32, in_frames, output);
  recordProcessingTime(metrics_, elapsedNs(started_at));
  return result;
}

ProcessResult InternalLinearResamplerBackend::processValidatedFrame(
  StreamState & state,
  const std::vector<float> & samples,
  const uint32_t input_frames,
  std::vector<uint8_t> & output)
{
  const uint32_t channels = state.contract.frame.channels;
  const uint32_t input_rate = state.contract.frame.sample_rate;
  const uint32_t output_rate = static_cast<uint32_t>(config_.target_sample_rate);

  if (samples.size() != static_cast<size_t>(input_frames) * channels) {
    return ProcessResult{ProcessStatus::kInvalidFrameContract, FrameContractStatus::kUnalignedData, 0};
  }

  if (state.buffer.empty()) {
    state.buffer = samples;
    state.buffer_start_frame_index = state.input_frames_total;
  } else {
    std::vector<float> next_buffer;
    next_buffer.reserve(channels + samples.size());
    const size_t previous_tail_offset = state.buffer.size() - channels;
    next_buffer.insert(
      next_buffer.end(),
      state.buffer.begin() + static_cast<std::vector<float>::difference_type>(previous_tail_offset),
      state.buffer.end());
    next_buffer.insert(next_buffer.end(), samples.begin(), samples.end());
    state.buffer = std::move(next_buffer);
    state.buffer_start_frame_index = state.input_frames_total - 1;
  }
  state.input_frames_total += input_frames;

  const uint64_t last_available_frame = state.input_frames_total - 1;
  const double input_per_output = static_cast<double>(input_rate) / static_cast<double>(output_rate);
  std::vector<float> out_f32;
  const auto sample_at = [&state](const uint64_t frame_index, const uint32_t channel) {
      const uint64_t relative_frame = frame_index - state.buffer_start_frame_index;
      return state.buffer.at(
        static_cast<size_t>(relative_frame) * state.contract.frame.channels + channel);
    };

  while (true) {
    const double src_pos = static_cast<double>(state.next_output_frame_index) * input_per_output;
    if (src_pos > static_cast<double>(last_available_frame)) {
      break;
    }

    uint64_t idx0 = static_cast<uint64_t>(std::floor(src_pos));
    const double frac = src_pos - static_cast<double>(idx0);
    if (idx0 < state.buffer_start_frame_index) {
      idx0 = state.buffer_start_frame_index;
    }
    const uint64_t idx1 = std::min<uint64_t>(idx0 + 1, last_available_frame);

    for (uint32_t ch = 0; ch < channels; ++ch) {
      const float s0 = sample_at(idx0, ch);
      const float s1 = sample_at(idx1, ch);
      out_f32.push_back(static_cast<float>(
        (1.0 - frac) * static_cast<double>(s0) + frac * static_cast<double>(s1)));
    }
    state.next_output_frame_index += 1;
  }

  if (out_f32.empty() || !containsOnlyFiniteNormalizedSamples(out_f32)) {
    return ProcessResult{ProcessStatus::kBackendProcessFailed, FrameContractStatus::kOk, 0};
  }

  const std::vector<uint8_t> out_bytes = encodeFloat32Le(out_f32);
  if (out_bytes.empty()) {
    return ProcessResult{ProcessStatus::kEncodeFailed, FrameContractStatus::kOk, 0};
  }

  const uint32_t output_frames = static_cast<uint32_t>(out_f32.size() / channels);
  state.output_frames_total += output_frames;
  updateFrameCountMetrics(
    metrics_,
    state.input_frames_total,
    state.output_frames_total,
    input_rate,
    output_rate);

  output = out_bytes;
  return ProcessResult{ProcessStatus::kOk, FrameContractStatus::kOk, output_frames};
}

}  // namespace fa_resample::backends

#include "fa_resample/backends/speexdsp_resampler.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <dlfcn.h>
#include <stdexcept>
#include <utility>

namespace fa_resample::backends
{

namespace
{
using SpeexState = void;
using SpeexInitFn = SpeexState * (*)(uint32_t, uint32_t, uint32_t, int, int *);
using SpeexDestroyFn = void (*)(SpeexState *);
using SpeexProcessInterleavedFloatFn = int (*)(
  SpeexState *,
  const float *,
  uint32_t *,
  float *,
  uint32_t *);
using SpeexGetLatencyFn = int (*)(SpeexState *);
using SpeexStrerrorFn = const char * (*)(int);

uint64_t elapsedNs(const std::chrono::steady_clock::time_point & start)
{
  const auto elapsed = std::chrono::steady_clock::now() - start;
  return static_cast<uint64_t>(
    std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count());
}

template<typename Function>
Function loadSymbol(void * handle, const char * name, const std::string & library_path)
{
  dlerror();
  void * symbol = dlsym(handle, name);
  const char * error = dlerror();
  if (error != nullptr || symbol == nullptr) {
    throw std::runtime_error(
      "speexdsp runtime dependency is missing symbol " + std::string(name) +
      " in " + library_path);
  }
  return reinterpret_cast<Function>(symbol);
}

uint32_t outputCapacityFrames(
  const uint32_t input_frames,
  const uint32_t input_rate,
  const uint32_t output_rate,
  const int output_latency)
{
  const double ratio = static_cast<double>(output_rate) / static_cast<double>(input_rate);
  const double estimated = std::ceil(static_cast<double>(input_frames + 32U) * ratio);
  return static_cast<uint32_t>(std::max<double>(1.0, estimated)) +
         static_cast<uint32_t>(std::max(0, output_latency)) + 64U;
}
}  // namespace

struct SpeexDspResamplerBackend::Library
{
  explicit Library(const std::string & library_path)
  : path(library_path)
  {
    handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {
      const char * error = dlerror();
      throw std::runtime_error(
        "speexdsp backend selected but runtime dependency is unavailable: " +
        path + " (" + (error == nullptr ? "dlopen failed" : std::string(error)) + ")");
    }

    init = loadSymbol<SpeexInitFn>(handle, "speex_resampler_init", path);
    destroy = loadSymbol<SpeexDestroyFn>(handle, "speex_resampler_destroy", path);
    process_interleaved_float = loadSymbol<SpeexProcessInterleavedFloatFn>(
      handle,
      "speex_resampler_process_interleaved_float",
      path);
    get_input_latency = loadSymbol<SpeexGetLatencyFn>(
      handle,
      "speex_resampler_get_input_latency",
      path);
    get_output_latency = loadSymbol<SpeexGetLatencyFn>(
      handle,
      "speex_resampler_get_output_latency",
      path);
    strerror = loadSymbol<SpeexStrerrorFn>(handle, "speex_resampler_strerror", path);
  }

  ~Library()
  {
    if (handle != nullptr) {
      dlclose(handle);
    }
  }

  std::string path;
  void * handle{nullptr};
  SpeexInitFn init{nullptr};
  SpeexDestroyFn destroy{nullptr};
  SpeexProcessInterleavedFloatFn process_interleaved_float{nullptr};
  SpeexGetLatencyFn get_input_latency{nullptr};
  SpeexGetLatencyFn get_output_latency{nullptr};
  SpeexStrerrorFn strerror{nullptr};
};

struct SpeexDspResamplerBackend::StreamState
{
  StreamContract contract;
  SpeexState * state{nullptr};
  uint64_t input_frames_total{0};
  uint64_t output_frames_total{0};
  int input_latency{0};
  int output_latency{0};
};

SpeexDspResamplerBackend::SpeexDspResamplerBackend(const SpeexDspResamplerConfig & config)
: config_(config),
  library_(std::make_unique<Library>(config.library_path))
{
  if (config_.target_sample_rate <= 0) {
    throw std::runtime_error("target_sample_rate must be > 0");
  }
  if (config_.quality < 0 || config_.quality > 10) {
    throw std::runtime_error("backend.quality for speexdsp must be an integer in 0..10");
  }
}

SpeexDspResamplerBackend::~SpeexDspResamplerBackend()
{
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto & stream : streams_) {
    if (stream.second && stream.second->state != nullptr) {
      library_->destroy(stream.second->state);
      stream.second->state = nullptr;
    }
  }
}

std::string SpeexDspResamplerBackend::name() const
{
  return kName;
}

std::string SpeexDspResamplerBackend::quality() const
{
  return std::to_string(config_.quality);
}

int SpeexDspResamplerBackend::targetSampleRate() const
{
  return config_.target_sample_rate;
}

BackendMetrics SpeexDspResamplerBackend::metrics() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  return metrics_;
}

SpeexDspResamplerBackend::StreamState SpeexDspResamplerBackend::makeStreamState(
  const ProcessRequest & request) const
{
  int error = 0;
  SpeexState * state = library_->init(
    request.contract.channels,
    request.contract.sample_rate,
    static_cast<uint32_t>(config_.target_sample_rate),
    config_.quality,
    &error);
  if (state == nullptr || error != 0) {
    const char * message = library_->strerror == nullptr ? "unknown speexdsp error" : library_->strerror(error);
    throw std::runtime_error("speexdsp resampler init failed: " + std::string(message));
  }

  StreamState stream_state;
  stream_state.contract = makeStreamContract(
    request.stream_id,
    request.contract,
    config_.target_sample_rate,
    kName,
    quality());
  stream_state.state = state;
  stream_state.input_latency = library_->get_input_latency(state);
  stream_state.output_latency = library_->get_output_latency(state);
  return stream_state;
}

ProcessResult SpeexDspResamplerBackend::process(
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
    quality());

  auto stream_it = streams_.find(request.stream_id);
  if (stream_it == streams_.end()) {
    stream_it = streams_.emplace(
      request.stream_id,
      std::make_unique<StreamState>(makeStreamState(request))).first;
  } else if (!streamContractMatches(stream_it->second->contract, next_contract)) {
    return ProcessResult{ProcessStatus::kStreamContractViolation, FrameContractStatus::kOk, 0};
  }

  ProcessResult result = processValidatedFrame(*stream_it->second, in_f32, in_frames, output);
  recordProcessingTime(metrics_, elapsedNs(started_at));
  return result;
}

ProcessResult SpeexDspResamplerBackend::processValidatedFrame(
  StreamState & state,
  const std::vector<float> & samples,
  const uint32_t input_frames,
  std::vector<uint8_t> & output)
{
  const uint32_t channels = state.contract.frame.channels;
  const uint32_t input_rate = state.contract.frame.sample_rate;
  const uint32_t output_rate = static_cast<uint32_t>(config_.target_sample_rate);

  uint32_t input_consumed = 0;
  std::vector<float> out_f32;
  while (input_consumed < input_frames) {
    uint32_t in_len = input_frames - input_consumed;
    uint32_t out_len = outputCapacityFrames(in_len, input_rate, output_rate, state.output_latency);
    std::vector<float> out_chunk(static_cast<size_t>(out_len) * channels);

    const int error = library_->process_interleaved_float(
      state.state,
      samples.data() + static_cast<size_t>(input_consumed) * channels,
      &in_len,
      out_chunk.data(),
      &out_len);
    if (error != 0) {
      const char * message = library_->strerror == nullptr ? "unknown speexdsp error" : library_->strerror(error);
      (void)message;
      return ProcessResult{ProcessStatus::kBackendProcessFailed, FrameContractStatus::kOk, 0};
    }
    if (in_len == 0 && out_len == 0) {
      return ProcessResult{ProcessStatus::kBackendProcessFailed, FrameContractStatus::kOk, 0};
    }

    input_consumed += in_len;
    out_f32.insert(
      out_f32.end(),
      out_chunk.begin(),
      out_chunk.begin() + static_cast<std::vector<float>::difference_type>(
        static_cast<size_t>(out_len) * channels));
  }

  if (!out_f32.empty() && !containsOnlyFiniteNormalizedSamples(out_f32)) {
    return ProcessResult{ProcessStatus::kEncodeFailed, FrameContractStatus::kOk, 0};
  }

  const uint32_t output_frames = static_cast<uint32_t>(out_f32.size() / channels);
  state.input_frames_total += input_frames;
  state.output_frames_total += output_frames;
  metrics_.algorithmic_delay_input_samples = static_cast<double>(state.input_latency);
  metrics_.algorithmic_delay_output_samples = static_cast<double>(state.output_latency);
  metrics_.algorithmic_delay_ms =
    static_cast<double>(state.output_latency) * 1000.0 / static_cast<double>(output_rate);
  updateFrameCountMetrics(
    metrics_,
    state.input_frames_total,
    state.output_frames_total,
    input_rate,
    output_rate);

  if (out_f32.empty()) {
    output.clear();
    return ProcessResult{ProcessStatus::kOk, FrameContractStatus::kOk, 0};
  }

  const std::vector<uint8_t> out_bytes = encodeFloat32Le(out_f32);
  if (out_bytes.empty()) {
    return ProcessResult{ProcessStatus::kEncodeFailed, FrameContractStatus::kOk, 0};
  }

  output = out_bytes;
  return ProcessResult{ProcessStatus::kOk, FrameContractStatus::kOk, output_frames};
}

}  // namespace fa_resample::backends

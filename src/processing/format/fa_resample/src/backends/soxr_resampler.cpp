#include "fa_resample/backends/soxr_resampler.hpp"

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
using SoxrState = void;
using SoxrError = const char *;

enum SoxrDataType
{
  kSoxrFloat32Interleaved = 0,
};

struct SoxrIoSpec
{
  int itype{0};
  int otype{0};
  double scale{1.0};
  void * reserved{nullptr};
  unsigned long flags{0};
};

struct SoxrQualitySpec
{
  double precision{20.0};
  double phase_response{50.0};
  double passband_end{0.913};
  double stopband_begin{1.0};
  void * reserved{nullptr};
  unsigned long flags{0};
};

struct SoxrRuntimeSpec
{
  unsigned log2_min_dft_size{10};
  unsigned log2_large_dft_size{17};
  unsigned coef_size_kbytes{400};
  unsigned num_threads{1};
  void * reserved{nullptr};
  unsigned long flags{0};
};

using SoxrCreateFn = SoxrState * (*)(
  double,
  double,
  unsigned,
  SoxrError *,
  const SoxrIoSpec *,
  const SoxrQualitySpec *,
  const SoxrRuntimeSpec *);
using SoxrDeleteFn = void (*)(SoxrState *);
using SoxrProcessFn = SoxrError (*)(
  SoxrState *,
  const void *,
  size_t,
  size_t *,
  void *,
  size_t,
  size_t *);
using SoxrDelayFn = double (*)(SoxrState *);
using SoxrIoSpecFn = SoxrIoSpec (*)(int, int);
using SoxrQualitySpecFn = SoxrQualitySpec (*)(unsigned long, unsigned long);
using SoxrRuntimeSpecFn = SoxrRuntimeSpec (*)(unsigned);

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
      "soxr runtime dependency is missing symbol " + std::string(name) +
      " in " + library_path);
  }
  return reinterpret_cast<Function>(symbol);
}

uint32_t outputCapacityFrames(
  const uint32_t input_frames,
  const uint32_t input_rate,
  const uint32_t output_rate,
  const double output_delay)
{
  const double ratio = static_cast<double>(output_rate) / static_cast<double>(input_rate);
  const double estimated = std::ceil(static_cast<double>(input_frames + 64U) * ratio);
  return static_cast<uint32_t>(std::max<double>(1.0, estimated + output_delay)) + 128U;
}
}  // namespace

struct SoxrResamplerBackend::Library
{
  explicit Library(const std::string & library_path)
  : path(library_path)
  {
    handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {
      const char * error = dlerror();
      throw std::runtime_error(
        "soxr backend selected but runtime dependency is unavailable: " +
        path + " (" + (error == nullptr ? "dlopen failed" : std::string(error)) + ")");
    }

    create = loadSymbol<SoxrCreateFn>(handle, "soxr_create", path);
    destroy = loadSymbol<SoxrDeleteFn>(handle, "soxr_delete", path);
    process = loadSymbol<SoxrProcessFn>(handle, "soxr_process", path);
    delay = loadSymbol<SoxrDelayFn>(handle, "soxr_delay", path);
    io_spec = loadSymbol<SoxrIoSpecFn>(handle, "soxr_io_spec", path);
    quality_spec = loadSymbol<SoxrQualitySpecFn>(handle, "soxr_quality_spec", path);
    runtime_spec = loadSymbol<SoxrRuntimeSpecFn>(handle, "soxr_runtime_spec", path);
  }

  ~Library()
  {
    if (handle != nullptr) {
      dlclose(handle);
    }
  }

  std::string path;
  void * handle{nullptr};
  SoxrCreateFn create{nullptr};
  SoxrDeleteFn destroy{nullptr};
  SoxrProcessFn process{nullptr};
  SoxrDelayFn delay{nullptr};
  SoxrIoSpecFn io_spec{nullptr};
  SoxrQualitySpecFn quality_spec{nullptr};
  SoxrRuntimeSpecFn runtime_spec{nullptr};
};

struct SoxrResamplerBackend::StreamState
{
  StreamContract contract;
  SoxrState * state{nullptr};
  uint64_t input_frames_total{0};
  uint64_t output_frames_total{0};
  double output_delay{0.0};
};

SoxrResamplerBackend::SoxrResamplerBackend(const SoxrResamplerConfig & config)
: config_(config),
  library_(std::make_unique<Library>(config.library_path))
{
  if (config_.target_sample_rate <= 0) {
    throw std::runtime_error("target_sample_rate must be > 0");
  }
}

SoxrResamplerBackend::~SoxrResamplerBackend()
{
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto & stream : streams_) {
    if (stream.second && stream.second->state != nullptr) {
      library_->destroy(stream.second->state);
      stream.second->state = nullptr;
    }
  }
}

std::string SoxrResamplerBackend::name() const
{
  return kName;
}

std::string SoxrResamplerBackend::quality() const
{
  return soxrQualityName(config_.quality);
}

int SoxrResamplerBackend::targetSampleRate() const
{
  return config_.target_sample_rate;
}

BackendMetrics SoxrResamplerBackend::metrics() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  return metrics_;
}

SoxrResamplerBackend::StreamState SoxrResamplerBackend::makeStreamState(
  const ProcessRequest & request) const
{
  SoxrError error = nullptr;
  const SoxrIoSpec io_spec = library_->io_spec(
    kSoxrFloat32Interleaved,
    kSoxrFloat32Interleaved);
  const SoxrQualitySpec quality_spec = library_->quality_spec(
    soxrQualityRecipe(config_.quality),
    0UL);
  const SoxrRuntimeSpec runtime_spec = library_->runtime_spec(1U);

  SoxrState * state = library_->create(
    static_cast<double>(request.contract.sample_rate),
    static_cast<double>(config_.target_sample_rate),
    request.contract.channels,
    &error,
    &io_spec,
    &quality_spec,
    &runtime_spec);
  if (state == nullptr || error != nullptr) {
    throw std::runtime_error(
      "soxr resampler init failed: " +
      std::string(error == nullptr ? "unknown soxr error" : error));
  }

  StreamState stream_state;
  stream_state.contract = makeStreamContract(
    request.stream_id,
    request.contract,
    config_.target_sample_rate,
    kName,
    quality());
  stream_state.state = state;
  stream_state.output_delay = library_->delay(state);
  return stream_state;
}

ProcessResult SoxrResamplerBackend::process(
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

ProcessResult SoxrResamplerBackend::processValidatedFrame(
  StreamState & state,
  const std::vector<float> & samples,
  const uint32_t input_frames,
  std::vector<uint8_t> & output)
{
  const uint32_t channels = state.contract.frame.channels;
  const uint32_t input_rate = state.contract.frame.sample_rate;
  const uint32_t output_rate = static_cast<uint32_t>(config_.target_sample_rate);

  size_t input_consumed = 0;
  std::vector<float> out_f32;
  while (input_consumed < input_frames) {
    size_t in_done = 0;
    size_t out_done = 0;
    const uint32_t out_capacity = outputCapacityFrames(
      static_cast<uint32_t>(input_frames - input_consumed),
      input_rate,
      output_rate,
      state.output_delay);
    std::vector<float> out_chunk(static_cast<size_t>(out_capacity) * channels);

    const SoxrError error = library_->process(
      state.state,
      samples.data() + input_consumed * channels,
      input_frames - input_consumed,
      &in_done,
      out_chunk.data(),
      out_capacity,
      &out_done);
    if (error != nullptr) {
      return ProcessResult{ProcessStatus::kBackendProcessFailed, FrameContractStatus::kOk, 0};
    }
    if (in_done == 0 && out_done == 0) {
      return ProcessResult{ProcessStatus::kBackendProcessFailed, FrameContractStatus::kOk, 0};
    }

    input_consumed += in_done;
    out_f32.insert(
      out_f32.end(),
      out_chunk.begin(),
      out_chunk.begin() + static_cast<std::vector<float>::difference_type>(out_done * channels));
    state.output_delay = library_->delay(state.state);
  }

  if (!out_f32.empty() && !containsOnlyFiniteNormalizedSamples(out_f32)) {
    return ProcessResult{ProcessStatus::kEncodeFailed, FrameContractStatus::kOk, 0};
  }

  const uint32_t output_frames = static_cast<uint32_t>(out_f32.size() / channels);
  state.input_frames_total += input_frames;
  state.output_frames_total += output_frames;
  metrics_.algorithmic_delay_output_samples = state.output_delay;
  metrics_.algorithmic_delay_input_samples =
    state.output_delay * static_cast<double>(input_rate) / static_cast<double>(output_rate);
  metrics_.algorithmic_delay_ms =
    state.output_delay * 1000.0 / static_cast<double>(output_rate);
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

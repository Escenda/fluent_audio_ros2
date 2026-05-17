#include "fa_ns/dtln_onnx_engine.hpp"

#include <array>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "kiss_fftr.h"

#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

namespace fa_ns
{

namespace
{
std::vector<int64_t> concreteShape(const std::vector<int64_t> & shape)
{
  std::vector<int64_t> out = shape;
  for (auto & dim : out) {
    if (dim <= 0) {
      dim = 1;
    }
  }
  return out;
}

size_t elementCount(const std::vector<int64_t> & shape)
{
  size_t count = 1;
  for (int64_t dim : shape) {
    if (dim <= 0) {
      throw std::runtime_error("Invalid tensor shape (non-positive dim)");
    }
    const size_t d = static_cast<size_t>(dim);
    if (d != 0 && count > (std::numeric_limits<size_t>::max() / d)) {
      throw std::runtime_error("Tensor shape overflow");
    }
    count *= d;
  }
  return count;
}

#ifdef _WIN32
std::wstring toOrtPath(const std::string & path)
{
  return std::wstring(path.begin(), path.end());
}
#endif

}  // namespace

struct DtlnOnnxEngine::Impl
{
  explicit Impl(const DtlnOnnxConfig & config)
  : config_(config),
    env_(ORT_LOGGING_LEVEL_WARNING, "fa_ns_dtln"),
    memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
  {
    if (config_.block_len <= 0 || config_.block_shift <= 0) {
      throw std::runtime_error("DTLN config requires block_len/block_shift > 0");
    }
    if (config_.block_shift > config_.block_len) {
      throw std::runtime_error("DTLN config requires block_shift <= block_len");
    }
    if (config_.model_1_path.empty() || config_.model_2_path.empty()) {
      throw std::runtime_error("DTLN requires model_1_path and model_2_path");
    }

    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(std::max<int>(1, config_.intra_op_num_threads));
    options.SetInterOpNumThreads(std::max<int>(1, config_.inter_op_num_threads));
    options.SetGraphOptimizationLevel(
      config_.enable_ort_optimizations ? GraphOptimizationLevel::ORT_ENABLE_ALL
                                      : GraphOptimizationLevel::ORT_DISABLE_ALL);

#ifdef _WIN32
    const std::wstring model_1_w = toOrtPath(config_.model_1_path);
    const std::wstring model_2_w = toOrtPath(config_.model_2_path);
    session_1_ = std::make_unique<Ort::Session>(env_, model_1_w.c_str(), options);
    session_2_ = std::make_unique<Ort::Session>(env_, model_2_w.c_str(), options);
#else
    session_1_ = std::make_unique<Ort::Session>(env_, config_.model_1_path.c_str(), options);
    session_2_ = std::make_unique<Ort::Session>(env_, config_.model_2_path.c_str(), options);
#endif

    initNamesAndShapes();
  }

  void reset()
  {
    std::fill(state_1_.begin(), state_1_.end(), 0.0f);
    std::fill(state_2_.begin(), state_2_.end(), 0.0f);
  }

  void initNamesAndShapes()
  {
    Ort::AllocatorWithDefaultOptions allocator;

    // Model 1: [mag, state] -> [mask, state]
    const size_t in_count_1 = session_1_->GetInputCount();
    const size_t out_count_1 = session_1_->GetOutputCount();
    if (in_count_1 < 2 || out_count_1 < 2) {
      throw std::runtime_error("DTLN model_1 must have at least 2 inputs and 2 outputs");
    }

    input_names_1_.reserve(in_count_1);
    for (size_t i = 0; i < in_count_1; ++i) {
      auto name = session_1_->GetInputNameAllocated(i, allocator);
      input_names_1_.push_back(std::string(name.get()));
    }
    output_names_1_.reserve(out_count_1);
    for (size_t i = 0; i < out_count_1; ++i) {
      auto name = session_1_->GetOutputNameAllocated(i, allocator);
      output_names_1_.push_back(std::string(name.get()));
    }

    mag_shape_ = concreteShape(
      session_1_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape());
    state_shape_1_ = concreteShape(
      session_1_->GetInputTypeInfo(1).GetTensorTypeAndShapeInfo().GetShape());

    if (mag_shape_.size() != 3 || mag_shape_[0] != 1 || mag_shape_[1] != 1) {
      throw std::runtime_error("DTLN model_1 mag input must be [1,1,bins]");
    }
    const int64_t bins = mag_shape_[2];
    if (bins <= 0) {
      throw std::runtime_error("DTLN model_1 bins dimension invalid");
    }
    const int expected_bins = (config_.block_len / 2) + 1;
    if (bins != expected_bins) {
      throw std::runtime_error(
              "DTLN model_1 bins mismatch: model=" + std::to_string(bins) +
              " expected=" + std::to_string(expected_bins));
    }

    if (state_shape_1_.size() != 4 || state_shape_1_[0] != 1 || state_shape_1_[3] != 2) {
      throw std::runtime_error("DTLN model_1 state input must be [1,layers,units,2]");
    }

    // Model 2: [time, state] -> [time, state]
    const size_t in_count_2 = session_2_->GetInputCount();
    const size_t out_count_2 = session_2_->GetOutputCount();
    if (in_count_2 < 2 || out_count_2 < 2) {
      throw std::runtime_error("DTLN model_2 must have at least 2 inputs and 2 outputs");
    }

    input_names_2_.reserve(in_count_2);
    for (size_t i = 0; i < in_count_2; ++i) {
      auto name = session_2_->GetInputNameAllocated(i, allocator);
      input_names_2_.push_back(std::string(name.get()));
    }
    output_names_2_.reserve(out_count_2);
    for (size_t i = 0; i < out_count_2; ++i) {
      auto name = session_2_->GetOutputNameAllocated(i, allocator);
      output_names_2_.push_back(std::string(name.get()));
    }

    time_shape_ = concreteShape(
      session_2_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape());
    state_shape_2_ = concreteShape(
      session_2_->GetInputTypeInfo(1).GetTensorTypeAndShapeInfo().GetShape());

    if (time_shape_.size() != 3 || time_shape_[0] != 1 || time_shape_[1] != 1) {
      throw std::runtime_error("DTLN model_2 time input must be [1,1,block_len]");
    }
    const int64_t block_len = time_shape_[2];
    if (block_len != config_.block_len) {
      throw std::runtime_error(
              "DTLN model_2 block_len mismatch: model=" + std::to_string(block_len) +
              " config=" + std::to_string(config_.block_len));
    }
    if (state_shape_2_ != state_shape_1_) {
      throw std::runtime_error("DTLN model_2 state shape must match model_1 state shape");
    }

    const size_t bins_count = elementCount(mag_shape_);
    mag_.assign(bins_count, 0.0f);
    mask_.assign(bins_count, 0.0f);

    state_1_.assign(elementCount(state_shape_1_), 0.0f);
    state_2_.assign(elementCount(state_shape_2_), 0.0f);

    time_in_.assign(elementCount(time_shape_), 0.0f);
    time_out_.assign(elementCount(time_shape_), 0.0f);

    const size_t fft_bins = static_cast<size_t>((config_.block_len / 2) + 1);
    fft_bins_.assign(fft_bins, kiss_fft_cpx{0.0f, 0.0f});
    est_bins_.assign(fft_bins, kiss_fft_cpx{0.0f, 0.0f});

    fwd_cfg_ = kiss_fftr_alloc(config_.block_len, 0, nullptr, nullptr);
    inv_cfg_ = kiss_fftr_alloc(config_.block_len, 1, nullptr, nullptr);
    if (!fwd_cfg_ || !inv_cfg_) {
      throw std::runtime_error("Failed to allocate kissfft configs");
    }
  }

  ~Impl()
  {
    if (fwd_cfg_) {
      kiss_fftr_free(fwd_cfg_);
      fwd_cfg_ = nullptr;
    }
    if (inv_cfg_) {
      kiss_fftr_free(inv_cfg_);
      inv_cfg_ = nullptr;
    }
  }

  void processHop(const float * in_buffer, float * out_block, const float inv_scale)
  {
    // FFT -> magnitude
    kiss_fftr(fwd_cfg_, in_buffer, fft_bins_.data());
    for (size_t k = 0; k < fft_bins_.size(); ++k) {
      const float re = fft_bins_[k].r;
      const float im = fft_bins_[k].i;
      mag_[k] = std::sqrt(re * re + im * im);
    }

    // Model 1
    Ort::Value mag_tensor = Ort::Value::CreateTensor<float>(
      memory_info_, mag_.data(), mag_.size(), mag_shape_.data(), mag_shape_.size());
    Ort::Value state1_tensor = Ort::Value::CreateTensor<float>(
      memory_info_, state_1_.data(), state_1_.size(), state_shape_1_.data(), state_shape_1_.size());

    std::array<const char *, 2> input_names = {
      input_names_1_[0].c_str(),
      input_names_1_[1].c_str()
    };
    std::array<Ort::Value, 2> inputs = {std::move(mag_tensor), std::move(state1_tensor)};
    std::array<const char *, 2> output_names = {
      output_names_1_[0].c_str(),
      output_names_1_[1].c_str()
    };

    std::vector<Ort::Value> outputs = session_1_->Run(
      Ort::RunOptions{nullptr},
      input_names.data(), inputs.data(), inputs.size(),
      output_names.data(), output_names.size());

    if (outputs.size() < 2) {
      throw std::runtime_error("DTLN model_1 Run returned insufficient outputs");
    }
    const float * mask_ptr = outputs[0].GetTensorData<float>();
    const float * state1_ptr = outputs[1].GetTensorData<float>();
    std::memcpy(mask_.data(), mask_ptr, mask_.size() * sizeof(float));
    std::memcpy(state_1_.data(), state1_ptr, state_1_.size() * sizeof(float));

    // Apply mask in frequency domain: estimated_fft = mask * fft
    for (size_t k = 0; k < fft_bins_.size(); ++k) {
      est_bins_[k].r = fft_bins_[k].r * mask_[k];
      est_bins_[k].i = fft_bins_[k].i * mask_[k];
    }

    // IFFT -> time domain (numpy irfft scales by 1/N)
    kiss_fftri(inv_cfg_, est_bins_.data(), time_in_.data());
    for (float & v : time_in_) {
      v *= inv_scale;
    }

    // Model 2
    Ort::Value time_tensor = Ort::Value::CreateTensor<float>(
      memory_info_, time_in_.data(), time_in_.size(), time_shape_.data(), time_shape_.size());
    Ort::Value state2_tensor = Ort::Value::CreateTensor<float>(
      memory_info_, state_2_.data(), state_2_.size(), state_shape_2_.data(), state_shape_2_.size());

    std::array<const char *, 2> input_names2 = {
      input_names_2_[0].c_str(),
      input_names_2_[1].c_str()
    };
    std::array<Ort::Value, 2> inputs2 = {std::move(time_tensor), std::move(state2_tensor)};
    std::array<const char *, 2> output_names2 = {
      output_names_2_[0].c_str(),
      output_names_2_[1].c_str()
    };

    std::vector<Ort::Value> outputs2 = session_2_->Run(
      Ort::RunOptions{nullptr},
      input_names2.data(), inputs2.data(), inputs2.size(),
      output_names2.data(), output_names2.size());

    if (outputs2.size() < 2) {
      throw std::runtime_error("DTLN model_2 Run returned insufficient outputs");
    }
    const float * out_ptr = outputs2[0].GetTensorData<float>();
    const float * state2_ptr = outputs2[1].GetTensorData<float>();
    std::memcpy(time_out_.data(), out_ptr, time_out_.size() * sizeof(float));
    std::memcpy(state_2_.data(), state2_ptr, state_2_.size() * sizeof(float));

    std::memcpy(out_block, time_out_.data(), static_cast<size_t>(config_.block_len) * sizeof(float));
  }

  DtlnOnnxConfig config_;

  Ort::Env env_;
  Ort::MemoryInfo memory_info_;
  std::unique_ptr<Ort::Session> session_1_;
  std::unique_ptr<Ort::Session> session_2_;

  std::vector<std::string> input_names_1_;
  std::vector<std::string> output_names_1_;
  std::vector<std::string> input_names_2_;
  std::vector<std::string> output_names_2_;

  std::vector<int64_t> mag_shape_;
  std::vector<int64_t> time_shape_;
  std::vector<int64_t> state_shape_1_;
  std::vector<int64_t> state_shape_2_;

  std::vector<float> mag_;
  std::vector<float> mask_;

  std::vector<float> state_1_;
  std::vector<float> state_2_;

  std::vector<float> time_in_;
  std::vector<float> time_out_;

  std::vector<kiss_fft_cpx> fft_bins_;
  std::vector<kiss_fft_cpx> est_bins_;

  kiss_fftr_cfg fwd_cfg_{nullptr};
  kiss_fftr_cfg inv_cfg_{nullptr};
};

DtlnOnnxEngine::DtlnOnnxEngine(const DtlnOnnxConfig & config)
: config_(config),
  pending_input_(),
  in_buffer_(static_cast<size_t>(config.block_len), 0.0f),
  out_buffer_(static_cast<size_t>(config.block_len), 0.0f),
  impl_(new Impl(config))
{
  reset();
}

DtlnOnnxEngine::~DtlnOnnxEngine()
{
  delete impl_;
  impl_ = nullptr;
}

void DtlnOnnxEngine::reset()
{
  pending_input_.clear();
  pending_offset_ = 0;
  std::fill(in_buffer_.begin(), in_buffer_.end(), 0.0f);
  std::fill(out_buffer_.begin(), out_buffer_.end(), 0.0f);
  if (impl_) {
    impl_->reset();
  }
}

size_t DtlnOnnxEngine::pendingInputSamples() const
{
  if (pending_offset_ >= pending_input_.size()) {
    return 0;
  }
  return pending_input_.size() - pending_offset_;
}

std::vector<float> DtlnOnnxEngine::process(const float * samples, size_t sample_count)
{
  if (!samples || sample_count == 0) {
    return {};
  }
  if (!impl_) {
    throw std::runtime_error("DTLN engine not initialized");
  }

  pending_input_.insert(pending_input_.end(), samples, samples + sample_count);

  std::vector<float> out;
  out.reserve(sample_count);

  const size_t hop = static_cast<size_t>(config_.block_shift);
  const size_t len = static_cast<size_t>(config_.block_len);
  const float inv_scale = 1.0f / static_cast<float>(config_.block_len);

  std::vector<float> out_block(len, 0.0f);

  while ((pending_input_.size() - pending_offset_) >= hop) {
    const float * hop_in = pending_input_.data() + pending_offset_;

    // shift input buffer and append hop
    std::memmove(in_buffer_.data(), in_buffer_.data() + hop, (len - hop) * sizeof(float));
    std::memcpy(in_buffer_.data() + (len - hop), hop_in, hop * sizeof(float));

    // run DTLN for this hop (produces a full block_len output block)
    impl_->processHop(in_buffer_.data(), out_block.data(), inv_scale);

    // overlap-add buffer update (same as upstream python reference)
    std::memmove(out_buffer_.data(), out_buffer_.data() + hop, (len - hop) * sizeof(float));
    std::fill(out_buffer_.begin() + (len - hop), out_buffer_.end(), 0.0f);
    for (size_t i = 0; i < len; ++i) {
      out_buffer_[i] += out_block[i];
    }

    // output first hop samples
    out.insert(out.end(), out_buffer_.begin(), out_buffer_.begin() + hop);

    pending_offset_ += hop;
  }

  if (pending_offset_ > 0) {
    pending_input_.erase(
      pending_input_.begin(),
      pending_input_.begin() + static_cast<std::vector<float>::difference_type>(pending_offset_));
    pending_offset_ = 0;
  }

  return out;
}

}  // namespace fa_ns

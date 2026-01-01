#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace fa_ns
{

struct DtlnOnnxConfig
{
  int block_len = -1;
  int block_shift = -1;

  std::string model_1_path;
  std::string model_2_path;

  int intra_op_num_threads = 1;
  int inter_op_num_threads = 1;
  bool enable_ort_optimizations = true;
};

class DtlnOnnxEngine
{
public:
  explicit DtlnOnnxEngine(const DtlnOnnxConfig & config);
  ~DtlnOnnxEngine();

  DtlnOnnxEngine(const DtlnOnnxEngine &) = delete;
  DtlnOnnxEngine & operator=(const DtlnOnnxEngine &) = delete;
  DtlnOnnxEngine(DtlnOnnxEngine &&) = delete;
  DtlnOnnxEngine & operator=(DtlnOnnxEngine &&) = delete;

  void reset();

  // Process samples and return enhanced samples.
  // Output length is a multiple of block_shift.
  std::vector<float> process(const float * samples, size_t sample_count);

  size_t pendingInputSamples() const;

  const DtlnOnnxConfig & config() const { return config_; }

private:
  void processHop(const float * hop, float * out_hop);

  DtlnOnnxConfig config_;

  // streaming buffers
  std::vector<float> pending_input_;
  size_t pending_offset_{0};

  std::vector<float> in_buffer_;
  std::vector<float> out_buffer_;

  // ONNX models (opaque to keep onnxruntime headers out of this header file)
  struct Impl;
  Impl * impl_{nullptr};
};

}  // namespace fa_ns


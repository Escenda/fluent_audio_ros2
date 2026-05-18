#include "fa_denoise/backends/dtln_onnx_backend.hpp"

#include <filesystem>
#include <stdexcept>

namespace fa_denoise::backends
{

namespace
{
std::string resolveModelPathOrThrow(const std::string & path_or_empty, const std::string & parameter_name)
{
  if (path_or_empty.empty()) {
    throw std::runtime_error(parameter_name + " is required for dtln_onnx backend");
  }
  std::filesystem::path path(path_or_empty);
  std::error_code ec;
  if (!std::filesystem::exists(path, ec) || ec) {
    throw std::runtime_error("Model file not found: " + path.string());
  }
  return path.string();
}
}  // namespace

DtlnOnnxBackend::DtlnOnnxBackend(const DtlnOnnxBackendConfig & config)
: expected_format_(config.expected_format),
  output_format_(config.output_format),
  engine_config_(config.engine_config)
{
  if (expected_format_.sample_rate != 16000) {
    throw std::runtime_error("fa_denoise dtln_onnx requires expected sample rate 16000");
  }
  if (expected_format_.channels != 1) {
    throw std::runtime_error("fa_denoise dtln_onnx requires expected_channels=1");
  }
  if (!isSupportedAudioFormatPair(expected_format_.encoding, expected_format_.bit_depth)) {
    throw std::runtime_error("fa_denoise dtln_onnx expected format must be PCM16LE/16 or FLOAT32LE/32");
  }
  if (!isSupportedAudioFormatPair(output_format_.encoding, output_format_.bit_depth)) {
    throw std::runtime_error("fa_denoise dtln_onnx output format must be PCM16LE/16 or FLOAT32LE/32");
  }
  engine_config_.model_1_path =
    resolveModelPathOrThrow(engine_config_.model_1_path, "dtln.model_1_path");
  engine_config_.model_2_path =
    resolveModelPathOrThrow(engine_config_.model_2_path, "dtln.model_2_path");
  engine_ = std::make_unique<DtlnOnnxEngine>(engine_config_);
}

DtlnOnnxBackend::~DtlnOnnxBackend() = default;

const char * DtlnOnnxBackend::name() const
{
  return kName;
}

ProcessResult DtlnOnnxBackend::process(const AudioBuffer & input)
{
  ProcessResult result;
  const ProcessStatus validation_status = validateAudioBuffer(input);
  if (validation_status != ProcessStatus::kOk) {
    result.status = validation_status;
    return result;
  }
  if (input.format.sample_rate != expected_format_.sample_rate ||
      input.format.channels != expected_format_.channels ||
      input.format.encoding != expected_format_.encoding ||
      input.format.bit_depth != expected_format_.bit_depth)
  {
    result.status = ProcessStatus::kUnsupportedInputFormat;
    return result;
  }

  std::vector<float> input_samples;
  const ProcessStatus decode_status = decodeToFloat(input, input_samples);
  if (decode_status != ProcessStatus::kOk) {
    result.status = decode_status;
    return result;
  }
  if ((input_samples.size() % static_cast<size_t>(engine_config_.block_shift)) != 0) {
    result.status = ProcessStatus::kSampleCountNotBlockAligned;
    return result;
  }

  std::vector<float> output_samples;
  try {
    output_samples = engine_->process(input_samples.data(), input_samples.size());
  } catch (const std::exception &) {
    result.status = ProcessStatus::kProcessingFailed;
    return result;
  }
  if (output_samples.size() != input_samples.size()) {
    result.status = ProcessStatus::kOutputSampleCountMismatch;
    return result;
  }

  std::vector<uint8_t> output_bytes;
  const ProcessStatus encode_status = encodeFromFloat(output_samples, output_format_, output_bytes);
  if (encode_status != ProcessStatus::kOk) {
    result.status = encode_status;
    return result;
  }

  result.status = ProcessStatus::kOk;
  result.output.format = output_format_;
  result.output.data = std::move(output_bytes);
  return result;
}

size_t DtlnOnnxBackend::pendingInputSamples() const
{
  if (!engine_) {
    throw std::logic_error("fa_denoise dtln_onnx backend engine is not initialized");
  }
  return engine_->pendingInputSamples();
}

}  // namespace fa_denoise::backends

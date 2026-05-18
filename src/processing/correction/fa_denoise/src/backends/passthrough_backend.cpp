#include "fa_denoise/backends/passthrough_backend.hpp"

#include <stdexcept>
#include <utility>

namespace fa_denoise::backends
{

namespace
{
bool formatsMatch(const AudioFormat & lhs, const AudioFormat & rhs)
{
  return lhs.sample_rate == rhs.sample_rate &&
         lhs.channels == rhs.channels &&
         lhs.encoding == rhs.encoding &&
         lhs.bit_depth == rhs.bit_depth;
}
}  // namespace

PassthroughBackend::PassthroughBackend(AudioFormat expected_format, AudioFormat output_format)
: expected_format_(std::move(expected_format)),
  output_format_(std::move(output_format))
{
  if (!isSupportedAudioFormatPair(expected_format_.encoding, expected_format_.bit_depth)) {
    throw std::runtime_error("passthrough expected format must be PCM16LE/16 or FLOAT32LE/32");
  }
  if (!isSupportedAudioFormatPair(output_format_.encoding, output_format_.bit_depth)) {
    throw std::runtime_error("passthrough output format must be PCM16LE/16 or FLOAT32LE/32");
  }
  if (!formatsMatch(expected_format_, output_format_)) {
    throw std::runtime_error("fa_denoise passthrough requires output format to match expected input format");
  }
}

const char * PassthroughBackend::name() const
{
  return kName;
}

ProcessResult PassthroughBackend::process(const AudioBuffer & input)
{
  ProcessResult result;
  const ProcessStatus validation_status = validateAudioBuffer(input);
  if (validation_status != ProcessStatus::kOk) {
    result.status = validation_status;
    return result;
  }
  if (!formatsMatch(input.format, expected_format_)) {
    result.status = ProcessStatus::kPassthroughFormatMismatch;
    return result;
  }

  result.status = ProcessStatus::kOk;
  result.output.format = output_format_;
  result.output.data = input.data;
  return result;
}

}  // namespace fa_denoise::backends

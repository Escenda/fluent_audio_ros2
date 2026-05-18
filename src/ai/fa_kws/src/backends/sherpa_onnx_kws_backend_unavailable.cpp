#include "fa_kws/backends/sherpa_onnx_kws_backend.hpp"

#include <stdexcept>
#include <string>

namespace
{

std::runtime_error unavailableSherpaOnnxBackendError(const char *operation)
{
  return std::runtime_error(
    std::string("fa_kws was built without sherpa-onnx support; cannot ") +
    operation +
    " backend.name=sherpa_onnx_kws. Rebuild fa_kws with "
    "-DFA_KWS_SHERPA_ONNX=ON and a valid SHERPA_ONNX_PREFIX.");
}

}  // namespace

namespace fa_kws
{

struct SherpaOnnxKwsBackendState
{
};

bool isSupportedSherpaOnnxExecutionProvider(const std::string &execution_provider)
{
  return execution_provider == "cpu" ||
         execution_provider == "cuda" ||
         execution_provider == "coreml";
}

std::string supportedSherpaOnnxExecutionProvidersForMessage()
{
  return "cpu, cuda, coreml";
}

SherpaOnnxKwsBackend::SherpaOnnxKwsBackend(const SherpaOnnxKwsBackendConfig &config)
: config_(config),
  last_detect_time_(std::chrono::steady_clock::now()),
  has_detect_time_(false)
{
  throw unavailableSherpaOnnxBackendError("construct");
}

SherpaOnnxKwsBackend::~SherpaOnnxKwsBackend() = default;

std::optional<KwsDetection> SherpaOnnxKwsBackend::process(
  const std::vector<float> &,
  std::int32_t,
  float,
  std::chrono::steady_clock::time_point)
{
  throw unavailableSherpaOnnxBackendError("process");
}

void SherpaOnnxKwsBackend::reset()
{
  throw unavailableSherpaOnnxBackendError("reset");
}

void SherpaOnnxKwsBackend::resetHard()
{
  throw unavailableSherpaOnnxBackendError("resetHard");
}

void SherpaOnnxKwsBackend::validateConfig() const
{
  throw unavailableSherpaOnnxBackendError("validate");
}

void SherpaOnnxKwsBackend::requireReady(const char *operation) const
{
  throw unavailableSherpaOnnxBackendError(operation);
}

}  // namespace fa_kws

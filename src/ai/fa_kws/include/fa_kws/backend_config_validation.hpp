#pragma once

#include <stdexcept>
#include <string>

namespace fa_kws::validation
{

constexpr const char * kBackendSherpaOnnxKws = "sherpa_onnx_kws";

inline void requireSupportedBackendName(const std::string & backend_name)
{
  if (backend_name.empty()) {
    throw std::runtime_error("backend.name is required");
  }
  if (backend_name != kBackendSherpaOnnxKws) {
    throw std::runtime_error("unsupported fa_kws backend.name: " + backend_name);
  }
}

}  // namespace fa_kws::validation

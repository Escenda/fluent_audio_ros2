#pragma once

#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace fa_kws
{

/**
 * @brief Wake word detection result returned by a KWS backend.
 */
struct KwsDetection
{
  std::string keyword;
  float score{1.0f};
  double start_time_sec{0.0};
};

/**
 * @brief ROS2-free keyword spotting engine interface.
 */
class KwsBackend
{
public:
  virtual ~KwsBackend() = default;

  virtual std::optional<KwsDetection> process(const std::vector<float> &samples,
                                             std::int32_t sample_rate,
                                             std::chrono::steady_clock::time_point now) = 0;

  virtual void reset() = 0;
  virtual void resetHard() = 0;
};

}  // namespace fa_kws

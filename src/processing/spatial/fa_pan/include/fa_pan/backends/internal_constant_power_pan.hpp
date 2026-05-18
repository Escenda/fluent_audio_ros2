#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace fa_pan::backends
{

struct InternalConstantPowerPanConfig
{
  double position{-2.0};
};

enum class ProcessStatus
{
  kOk,
  kInvalidInputSize,
  kInvalidInputSample,
  kInvalidOutputSample,
};

class InternalConstantPowerPanBackend
{
public:
  static constexpr const char * kName = "internal_constant_power_pan";

  explicit InternalConstantPowerPanBackend(const InternalConstantPowerPanConfig & config);

  [[nodiscard]] double leftGain() const;
  [[nodiscard]] double rightGain() const;
  [[nodiscard]] ProcessStatus process(
    const std::vector<uint8_t> & input,
    std::vector<uint8_t> & output) const;

private:
  double left_gain_{1.0};
  double right_gain_{1.0};
};

[[nodiscard]] const char * processStatusMessage(ProcessStatus status);
[[nodiscard]] bool isNormalizedFinite(float sample);
[[nodiscard]] float readFloat32Le(const std::vector<uint8_t> & bytes, size_t sample_index);
void appendFloat32Le(float sample, std::vector<uint8_t> & out_bytes);

}  // namespace fa_pan::backends

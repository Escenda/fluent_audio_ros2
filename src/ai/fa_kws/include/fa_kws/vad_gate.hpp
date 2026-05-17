#pragma once

#include <cmath>

namespace fa_kws
{

inline bool isValidVadProbability(float probability)
{
  return std::isfinite(probability) && probability >= 0.0f && probability <= 1.0f;
}

inline bool isValidVadGateThreshold(double threshold)
{
  return std::isfinite(threshold) && threshold > 0.0 && threshold <= 1.0;
}

inline bool passesVadGate(float probability, float threshold)
{
  return isValidVadProbability(probability) &&
         isValidVadGateThreshold(static_cast<double>(threshold)) &&
         probability >= threshold;
}

}  // namespace fa_kws

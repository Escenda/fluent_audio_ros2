#include <limits>

#include <gtest/gtest.h>

#include "fa_kws/vad_gate.hpp"

namespace fa_kws
{
namespace
{

TEST(VadGateContract, ProbabilityMustBeFiniteAndNormalized)
{
  EXPECT_TRUE(isValidVadProbability(0.0f));
  EXPECT_TRUE(isValidVadProbability(0.5f));
  EXPECT_TRUE(isValidVadProbability(1.0f));

  EXPECT_FALSE(isValidVadProbability(-0.001f));
  EXPECT_FALSE(isValidVadProbability(1.001f));
  EXPECT_FALSE(isValidVadProbability(std::numeric_limits<float>::quiet_NaN()));
  EXPECT_FALSE(isValidVadProbability(std::numeric_limits<float>::infinity()));
}

TEST(VadGateContract, ThresholdCannotDisableGate)
{
  EXPECT_TRUE(isValidVadGateThreshold(0.001));
  EXPECT_TRUE(isValidVadGateThreshold(1.0));

  EXPECT_FALSE(isValidVadGateThreshold(0.0));
  EXPECT_FALSE(isValidVadGateThreshold(-0.001));
  EXPECT_FALSE(isValidVadGateThreshold(1.001));
  EXPECT_FALSE(isValidVadGateThreshold(std::numeric_limits<double>::quiet_NaN()));
  EXPECT_FALSE(isValidVadGateThreshold(std::numeric_limits<double>::infinity()));
}

TEST(VadGateContract, GateIsInclusiveAndFailClosed)
{
  EXPECT_TRUE(passesVadGate(0.35f, 0.35f));
  EXPECT_TRUE(passesVadGate(0.90f, 0.35f));

  EXPECT_FALSE(passesVadGate(0.349f, 0.35f));
  EXPECT_FALSE(passesVadGate(1.0f, 0.0f));
  EXPECT_FALSE(passesVadGate(std::numeric_limits<float>::quiet_NaN(), 0.35f));
  EXPECT_FALSE(passesVadGate(0.35f, std::numeric_limits<float>::quiet_NaN()));
}

}  // namespace
}  // namespace fa_kws

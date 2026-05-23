#include "fa_kws/vad_gate.hpp"

#include <chrono>
#include <stdexcept>

#include <gtest/gtest.h>

namespace
{

fa_kws::VadGateConfig baseConfig()
{
  fa_kws::VadGateConfig config;
  config.enabled = true;
  config.default_enabled = false;
  config.source_id = "mic";
  config.stream_id = "audio/preprocessed/mono16k";
  config.probability_gate = 0.35;
  config.max_age = std::chrono::milliseconds{250};
  return config;
}

fa_kws::VoiceActivityUpdate speechUpdate(double probability)
{
  fa_kws::VoiceActivityUpdate update;
  update.source_id = "mic";
  update.stream_id = "audio/preprocessed/mono16k";
  update.probability = probability;
  update.is_speech = true;
  update.speech_ended = false;
  return update;
}

}  // namespace

TEST(KwsVadGateTest, AllowsAudioOnlyWhileSpeechProbabilityPassesGate)
{
  fa_kws::VadGate gate(baseConfig());
  const auto now = std::chrono::steady_clock::now();

  EXPECT_FALSE(gate.allows(now));
  EXPECT_TRUE(gate.update(speechUpdate(0.34), now));
  EXPECT_FALSE(gate.allows(now));

  EXPECT_TRUE(gate.update(speechUpdate(0.35), now));
  EXPECT_TRUE(gate.allows(now));

  auto ended = speechUpdate(0.80);
  ended.speech_ended = true;
  EXPECT_TRUE(gate.update(ended, now));
  EXPECT_FALSE(gate.allows(now));
}

TEST(KwsVadGateTest, FallsBackToDefaultWhenUpdateIsStale)
{
  auto config = baseConfig();
  config.default_enabled = false;
  fa_kws::VadGate gate(config);
  const auto now = std::chrono::steady_clock::now();

  EXPECT_TRUE(gate.update(speechUpdate(0.90), now));
  EXPECT_TRUE(gate.allows(now + std::chrono::milliseconds{250}));
  EXPECT_FALSE(gate.allows(now + std::chrono::milliseconds{251}));
}

TEST(KwsVadGateTest, DisabledGateAlwaysAllowsAudio)
{
  auto config = baseConfig();
  config.enabled = false;
  fa_kws::VadGate gate(config);

  EXPECT_TRUE(gate.allows(std::chrono::steady_clock::now()));
}

TEST(KwsVadGateTest, RejectsUnexpectedVoiceActivityIdentity)
{
  fa_kws::VadGate gate(baseConfig());
  auto update = speechUpdate(0.90);
  update.stream_id = "audio/other";

  EXPECT_FALSE(gate.update(update, std::chrono::steady_clock::now()));
  EXPECT_FALSE(gate.allows(std::chrono::steady_clock::now()));
}

TEST(KwsVadGateTest, RejectsInvalidConfig)
{
  auto config = baseConfig();
  config.probability_gate = 1.2;
  EXPECT_THROW({ fa_kws::VadGate gate(config); }, std::invalid_argument);

  config = baseConfig();
  config.max_age = std::chrono::milliseconds{0};
  EXPECT_THROW({ fa_kws::VadGate gate(config); }, std::invalid_argument);
}

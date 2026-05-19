#include "fa_kws/backends/factory.hpp"

#include <chrono>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

namespace
{

fa_kws::KwsBackendSettings baseSettings()
{
  fa_kws::KwsBackendSettings settings;
  settings.name = "sherpa_onnx_kws";
  settings.target_sample_rate = 16000;
  settings.model_num_threads = 1;
  settings.execution_provider = "bogus";
  settings.max_active_paths = 4;
  settings.num_trailing_blanks = 2;
  settings.keywords_score = 1.0f;
  settings.keywords_threshold = 0.25f;
  settings.vad_threshold = 0.5f;
  settings.cooldown = std::chrono::milliseconds{0};
  settings.timeout_sec = 1.0;
  settings.cleanup_audio_files = true;
  return settings;
}

}  // namespace

TEST(KwsBackendFactoryTest, RejectsMissingBackendName)
{
  auto settings = baseSettings();
  settings.name = "";

  EXPECT_THROW(
    { static_cast<void>(fa_kws::buildKwsBackend(settings)); },
    std::runtime_error);
}

TEST(KwsBackendFactoryTest, RejectsUnknownBackendName)
{
  auto settings = baseSettings();
  settings.name = "bogus";

  EXPECT_THROW(
    { static_cast<void>(fa_kws::buildKwsBackend(settings)); },
    std::runtime_error);
}

TEST(KwsBackendFactoryTest, DelegatesSherpaOnnxConfigValidation)
{
  auto settings = baseSettings();

  EXPECT_THROW(
    { static_cast<void>(fa_kws::buildKwsBackend(settings)); },
    std::invalid_argument);
}

#include "fa_kws/backends/factory.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

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
  settings.cooldown = std::chrono::milliseconds{0};
  settings.timeout_sec = 1.0;
  settings.cleanup_audio_files = true;
  return settings;
}

void writeRequiredModelFiles(
  fa_kws::KwsBackendSettings &settings,
  const std::filesystem::path &directory)
{
  std::filesystem::create_directories(directory);
  settings.encoder_path = (directory / "encoder.onnx").string();
  settings.decoder_path = (directory / "decoder.onnx").string();
  settings.joiner_path = (directory / "joiner.onnx").string();
  settings.tokens_path = (directory / "tokens.txt").string();
  settings.keywords_path = (directory / "keywords.txt").string();
  for (const auto *path : {
      settings.encoder_path.c_str(),
      settings.decoder_path.c_str(),
      settings.joiner_path.c_str(),
      settings.tokens_path.c_str(),
      settings.keywords_path.c_str(),
    }) {
    std::ofstream(path) << "fixture\n";
  }
}

std::vector<std::string> validInferenceArgs()
{
  return {
    "detect",
    "--audio",
    "{audio}",
    "--encoder",
    "{encoder}",
    "--decoder",
    "{decoder}",
    "--joiner",
    "{joiner}",
    "--tokens",
    "{tokens}",
    "--keywords",
    "{keywords}",
    "--provider",
    "{provider}",
    "--sample-rate",
    "{sample_rate}",
    "--num-threads",
    "{num_threads}",
    "--max-active-paths",
    "{max_active_paths}",
    "--num-trailing-blanks",
    "{num_trailing_blanks}",
    "--keywords-score",
    "{keywords_score}",
    "--keywords-threshold",
    "{keywords_threshold}",
  };
}

std::vector<std::string> validHealthArgs()
{
  return {
    "health",
    "--encoder",
    "{encoder}",
    "--decoder",
    "{decoder}",
    "--joiner",
    "{joiner}",
    "--tokens",
    "{tokens}",
    "--keywords",
    "{keywords}",
    "--provider",
    "{provider}",
    "--sample-rate",
    "{sample_rate}",
    "--num-threads",
    "{num_threads}",
    "--max-active-paths",
    "{max_active_paths}",
    "--num-trailing-blanks",
    "{num_trailing_blanks}",
    "--keywords-score",
    "{keywords_score}",
    "--keywords-threshold",
    "{keywords_threshold}",
  };
}

void expectInvalidArgument(
  const fa_kws::KwsBackendSettings &settings,
  const std::string &expected_message)
{
  try {
    static_cast<void>(fa_kws::buildKwsBackend(settings));
    FAIL() << "expected invalid_argument: " << expected_message;
  } catch (const std::invalid_argument &error) {
    EXPECT_STREQ(expected_message.c_str(), error.what());
  }
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

TEST(KwsBackendFactoryTest, RejectsMissingSherpaOnnxInferenceArgs)
{
  auto settings = baseSettings();
  settings.execution_provider = "cpu";
  settings.command = "/bin/true";
  settings.health_args = validHealthArgs();
  settings.workspace_dir = "/tmp/fa_kws_backend_factory_contract";
  writeRequiredModelFiles(
    settings,
    std::filesystem::temp_directory_path() / "fa_kws_backend_factory_missing_args");

  expectInvalidArgument(settings, "backend.args must not be empty");
}

TEST(KwsBackendFactoryTest, RejectsMissingSherpaOnnxHealthArgs)
{
  auto settings = baseSettings();
  settings.execution_provider = "cpu";
  settings.command = "/bin/true";
  settings.args = validInferenceArgs();
  settings.workspace_dir = "/tmp/fa_kws_backend_factory_contract";
  writeRequiredModelFiles(
    settings,
    std::filesystem::temp_directory_path() / "fa_kws_backend_factory_missing_health_args");

  expectInvalidArgument(settings, "backend.health_args must not be empty");
}

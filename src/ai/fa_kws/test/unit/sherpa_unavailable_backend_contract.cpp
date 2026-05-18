#include <chrono>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

#include "fa_kws/backends/sherpa_onnx_kws_backend.hpp"

namespace
{

fa_kws::SherpaOnnxKwsBackendConfig makeCompleteConfig()
{
  fa_kws::SherpaOnnxKwsBackendConfig config;
  config.target_sample_rate = 16000;
  config.model_num_threads = 1;
  config.execution_provider = "cpu";
  config.encoder_path = "/tmp/fa_kws_encoder.onnx";
  config.decoder_path = "/tmp/fa_kws_decoder.onnx";
  config.joiner_path = "/tmp/fa_kws_joiner.onnx";
  config.tokens_path = "/tmp/fa_kws_tokens.txt";
  config.keywords_path = "/tmp/fa_kws_keywords.txt";
  config.max_active_paths = 4;
  config.num_trailing_blanks = 1;
  config.keywords_score = 1.0f;
  config.keywords_threshold = 0.25f;
  config.vad_threshold = 0.5f;
  config.cooldown = std::chrono::milliseconds{1000};
  return config;
}

}  // namespace

TEST(SherpaUnavailableBackendContract, ConstructorFailsClosedWhenSherpaSupportIsDisabled)
{
  const fa_kws::SherpaOnnxKwsBackendConfig config = makeCompleteConfig();

  try {
    fa_kws::SherpaOnnxKwsBackend backend(config);
    FAIL() << "expected unavailable sherpa-onnx backend to fail closed";
  } catch (const std::runtime_error &err) {
    const std::string message = err.what();
    EXPECT_NE(message.find("fa_kws was built without sherpa-onnx support"), std::string::npos);
    EXPECT_NE(message.find("backend.name=sherpa_onnx_kws"), std::string::npos);
  }
}


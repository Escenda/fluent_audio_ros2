#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>

#include "fa_kws/backends/kws_backend.hpp"

namespace fa_kws
{

struct SherpaOnnxKwsBackendState;

struct SherpaOnnxKwsBackendConfig
{
  std::int32_t target_sample_rate;
  int model_num_threads;
  std::string execution_provider;

  std::string encoder_path;
  std::string decoder_path;
  std::string joiner_path;
  std::string tokens_path;
  std::string keywords_path;

  int max_active_paths;
  int num_trailing_blanks;
  float keywords_score;
  float keywords_threshold;

  float vad_threshold;
  std::chrono::milliseconds cooldown;
};

bool isSupportedSherpaOnnxExecutionProvider(const std::string &execution_provider);

std::string supportedSherpaOnnxExecutionProvidersForMessage();

class SherpaOnnxKwsBackend final : public KwsBackend
{
public:
  explicit SherpaOnnxKwsBackend(const SherpaOnnxKwsBackendConfig &config);
  ~SherpaOnnxKwsBackend() override;

  SherpaOnnxKwsBackend(const SherpaOnnxKwsBackend &) = delete;
  SherpaOnnxKwsBackend &operator=(const SherpaOnnxKwsBackend &) = delete;

  std::optional<KwsDetection> process(const std::vector<float> &samples,
                                      std::int32_t sample_rate,
                                      float vad_prob,
                                      std::chrono::steady_clock::time_point now) override;

  // Soft reset: clear stream state (may not fully reset internal buffers)
  void reset() override;

  // Hard reset: destroy and recreate stream (guarantees clean state)
  void resetHard() override;

private:
  void validateConfig() const;
  void requireReady(const char *operation) const;

  SherpaOnnxKwsBackendConfig config_;
  std::unique_ptr<SherpaOnnxKwsBackendState> state_;

  std::chrono::steady_clock::time_point last_detect_time_;
  bool has_detect_time_{false};
};

}  // namespace fa_kws

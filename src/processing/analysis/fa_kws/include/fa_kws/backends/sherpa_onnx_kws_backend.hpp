#pragma once

#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <sherpa-onnx/c-api/c-api.h>

namespace fa_kws
{

struct SherpaOnnxKwsBackendConfig
{
  std::int32_t target_sample_rate{16000};
  int model_num_threads{4};
  std::string model_provider{"cpu"};

  std::string encoder_path;
  std::string decoder_path;
  std::string joiner_path;
  std::string tokens_path;
  std::string keywords_path;

  int max_active_paths{4};
  int num_trailing_blanks{1};
  float keywords_score{1.0f};
  float keywords_threshold{0.25f};

  float vad_threshold{0.35f};
  std::chrono::milliseconds cooldown{std::chrono::milliseconds{2000}};
};

struct KwsDetection
{
  std::string keyword;
  double start_time_sec{0.0};
};

class SherpaOnnxKwsBackend
{
public:
  explicit SherpaOnnxKwsBackend(const SherpaOnnxKwsBackendConfig &config);
  ~SherpaOnnxKwsBackend();

  SherpaOnnxKwsBackend(const SherpaOnnxKwsBackend &) = delete;
  SherpaOnnxKwsBackend &operator=(const SherpaOnnxKwsBackend &) = delete;

  std::optional<KwsDetection> process(const std::vector<float> &samples,
                                      std::int32_t sample_rate,
                                      float vad_prob,
                                      std::chrono::steady_clock::time_point now);

  // Soft reset: clear stream state (may not fully reset internal buffers)
  void reset();

  // Hard reset: destroy and recreate stream (guarantees clean state)
  void resetHard();

private:
  SherpaOnnxKwsBackendConfig config_;
  const SherpaOnnxKeywordSpotter *spotter_{nullptr};
  SherpaOnnxOnlineStream *stream_{nullptr};  // non-const for resetHard()

  std::chrono::steady_clock::time_point last_detect_time_;
  bool has_detect_time_{false};
};

}  // namespace fa_kws

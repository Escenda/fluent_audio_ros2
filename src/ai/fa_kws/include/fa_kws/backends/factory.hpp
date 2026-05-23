#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "fa_kws/backends/kws_backend.hpp"

namespace fa_kws
{

struct KwsBackendSettings
{
  std::string name;

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

  std::chrono::milliseconds cooldown;

  std::string command;
  std::vector<std::string> args;
  std::vector<std::string> stream_args;
  std::vector<std::string> health_args;
  double timeout_sec;
  std::string workspace_dir;
  bool cleanup_audio_files;
};

std::unique_ptr<KwsBackend> buildKwsBackend(const KwsBackendSettings &settings);

}  // namespace fa_kws

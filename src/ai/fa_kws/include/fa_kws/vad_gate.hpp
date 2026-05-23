#pragma once

#include <chrono>
#include <string>

namespace fa_kws
{

struct VadGateConfig
{
  bool enabled{false};
  bool default_enabled{true};
  std::string source_id;
  std::string stream_id;
  double probability_gate{0.0};
  std::chrono::milliseconds max_age{std::chrono::milliseconds{0}};
};

struct VoiceActivityUpdate
{
  std::string source_id;
  std::string stream_id;
  double probability{0.0};
  bool is_speech{false};
  bool speech_ended{false};
};

class VadGate
{
public:
  explicit VadGate(VadGateConfig config);

  bool update(
    const VoiceActivityUpdate & update,
    std::chrono::steady_clock::time_point now);

  bool allows(std::chrono::steady_clock::time_point now) const;
  double probability() const;
  bool isSpeech() const;
  bool hasUpdate() const;

private:
  VadGateConfig config_;
  bool has_update_{false};
  bool allows_audio_{false};
  bool is_speech_{false};
  double probability_{0.0};
  std::chrono::steady_clock::time_point last_update_{};
};

}  // namespace fa_kws

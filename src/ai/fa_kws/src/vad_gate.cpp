#include "fa_kws/vad_gate.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace fa_kws
{

VadGate::VadGate(VadGateConfig config)
: config_(std::move(config)),
  allows_audio_(config_.default_enabled)
{
  if (!config_.enabled) {
    return;
  }
  if (config_.source_id.empty()) {
    throw std::invalid_argument("VAD gate source_id is required");
  }
  if (config_.stream_id.empty()) {
    throw std::invalid_argument("VAD gate stream_id is required");
  }
  if (!std::isfinite(config_.probability_gate) ||
    config_.probability_gate < 0.0 ||
    config_.probability_gate > 1.0)
  {
    throw std::invalid_argument("VAD gate probability_gate must be in [0.0, 1.0]");
  }
  if (config_.max_age.count() <= 0) {
    throw std::invalid_argument("VAD gate max_age must be positive");
  }
}

bool VadGate::update(
  const VoiceActivityUpdate & update,
  std::chrono::steady_clock::time_point now)
{
  if (!config_.enabled) {
    return true;
  }
  if (update.source_id != config_.source_id || update.stream_id != config_.stream_id) {
    return false;
  }
  if (!std::isfinite(update.probability) ||
    update.probability < 0.0 ||
    update.probability > 1.0)
  {
    return false;
  }

  has_update_ = true;
  last_update_ = now;
  probability_ = update.probability;
  is_speech_ = update.is_speech && !update.speech_ended;
  allows_audio_ = is_speech_ && probability_ >= config_.probability_gate;
  return true;
}

bool VadGate::allows(std::chrono::steady_clock::time_point now) const
{
  if (!config_.enabled) {
    return true;
  }
  if (!has_update_) {
    return config_.default_enabled;
  }
  if (now - last_update_ > config_.max_age) {
    return config_.default_enabled;
  }
  return allows_audio_;
}

double VadGate::probability() const
{
  return probability_;
}

bool VadGate::isSpeech() const
{
  return is_speech_;
}

bool VadGate::hasUpdate() const
{
  return has_update_;
}

}  // namespace fa_kws

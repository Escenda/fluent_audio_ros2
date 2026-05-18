#include "fa_kws/backends/sherpa_onnx_kws_backend.hpp"

#include <sherpa-onnx/c-api/c-api.h>

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>

#include "fa_kws/vad_gate.hpp"

namespace
{

bool traceEnabled()
{
  static const bool enabled = []() {
    const char *v = std::getenv("FA_KWS_TRACE");
    return v != nullptr && v[0] != '\0' && std::atoi(v) != 0;
  }();
  return enabled;
}

void tracef(const char *fmt, ...)
{
  if (!traceEnabled()) {
    return;
  }
  va_list ap;
  va_start(ap, fmt);
  std::vfprintf(stderr, fmt, ap);
  va_end(ap);
  std::fflush(stderr);
}

void requireReadableRegularFile(const char *config_name, const std::string &path_value)
{
  if (path_value.empty()) {
    throw std::invalid_argument(std::string(config_name) + " is required");
  }

  const std::filesystem::path path(path_value);
  std::error_code ec;
  if (!std::filesystem::is_regular_file(path, ec)) {
    throw std::invalid_argument(std::string(config_name) + " must be a regular readable file: " +
                                path_value);
  }

  std::ifstream probe(path, std::ios::binary);
  if (!probe.good()) {
    throw std::invalid_argument(std::string(config_name) + " is not readable: " + path_value);
  }
}

}  // namespace

namespace fa_kws
{

struct SherpaOnnxKwsBackendState
{
  const SherpaOnnxKeywordSpotter *spotter{nullptr};
  SherpaOnnxOnlineStream *stream{nullptr};
};

bool isSupportedSherpaOnnxExecutionProvider(const std::string &execution_provider)
{
  return execution_provider == "cpu" ||
         execution_provider == "cuda" ||
         execution_provider == "coreml";
}

std::string supportedSherpaOnnxExecutionProvidersForMessage()
{
  return "cpu, cuda, coreml";
}

SherpaOnnxKwsBackend::SherpaOnnxKwsBackend(const SherpaOnnxKwsBackendConfig &config)
: config_(config),
  state_(std::make_unique<SherpaOnnxKwsBackendState>()),
  last_detect_time_(std::chrono::steady_clock::now()),
  has_detect_time_(false)
{
  validateConfig();

  SherpaOnnxFeatureConfig feat_config;
  std::memset(&feat_config, 0, sizeof(feat_config));
  feat_config.sample_rate = config_.target_sample_rate;
  feat_config.feature_dim = 80;

  SherpaOnnxOnlineTransducerModelConfig transducer;
  std::memset(&transducer, 0, sizeof(transducer));
  transducer.encoder = config_.encoder_path.c_str();
  transducer.decoder = config_.decoder_path.c_str();
  transducer.joiner = config_.joiner_path.c_str();

  SherpaOnnxOnlineModelConfig model_config;
  std::memset(&model_config, 0, sizeof(model_config));
  model_config.transducer = transducer;
  model_config.tokens = config_.tokens_path.c_str();
  model_config.num_threads = config_.model_num_threads;
  model_config.provider = config_.execution_provider.c_str();
  model_config.debug = 0;
  model_config.model_type = "";
  model_config.modeling_unit = "";
  model_config.bpe_vocab = "";
  model_config.tokens_buf = nullptr;
  model_config.tokens_buf_size = 0;

  SherpaOnnxKeywordSpotterConfig kws_config;
  std::memset(&kws_config, 0, sizeof(kws_config));
  kws_config.feat_config = feat_config;
  kws_config.model_config = model_config;
  kws_config.max_active_paths = config_.max_active_paths;
  kws_config.num_trailing_blanks = config_.num_trailing_blanks;
  kws_config.keywords_score = config_.keywords_score;
  kws_config.keywords_threshold = config_.keywords_threshold;
  kws_config.keywords_file = config_.keywords_path.c_str();
  kws_config.keywords_buf = nullptr;
  kws_config.keywords_buf_size = 0;

  state_->spotter = SherpaOnnxCreateKeywordSpotter(&kws_config);
  if (!state_->spotter) {
    throw std::runtime_error("SherpaOnnxKwsBackend: SherpaOnnxCreateKeywordSpotter() failed");
  }

  state_->stream = const_cast<SherpaOnnxOnlineStream*>(
    SherpaOnnxCreateKeywordStream(state_->spotter));
  if (!state_->stream) {
    SherpaOnnxDestroyKeywordSpotter(state_->spotter);
    state_->spotter = nullptr;
    throw std::runtime_error("SherpaOnnxKwsBackend: SherpaOnnxCreateKeywordStream() failed");
  }
}

SherpaOnnxKwsBackend::~SherpaOnnxKwsBackend()
{
  if (!state_) {
    return;
  }
  if (state_->stream != nullptr) {
    SherpaOnnxDestroyOnlineStream(state_->stream);
    state_->stream = nullptr;
  }
  if (state_->spotter != nullptr) {
    SherpaOnnxDestroyKeywordSpotter(state_->spotter);
    state_->spotter = nullptr;
  }
}

std::optional<KwsDetection> SherpaOnnxKwsBackend::process(
  const std::vector<float> &samples,
  std::int32_t sample_rate,
  float vad_prob,
  std::chrono::steady_clock::time_point now)
{
  requireReady("process");
  if (samples.empty()) {
    throw std::invalid_argument("KWS backend samples are required");
  }
  if (sample_rate != config_.target_sample_rate) {
    throw std::invalid_argument(
      "KWS backend sample_rate must match configured target_sample_rate " +
      std::to_string(config_.target_sample_rate) + ", got " + std::to_string(sample_rate));
  }
  if (!isValidVadProbability(vad_prob)) {
    throw std::invalid_argument("vad_prob must be finite and in [0.0, 1.0]");
  }
  if (!passesVadGate(vad_prob, config_.vad_threshold)) {
    tracef("fa_kws: VAD gate dropped (vad_prob=%.4f threshold=%.4f)\n",
           static_cast<double>(vad_prob),
           static_cast<double>(config_.vad_threshold));
    SherpaOnnxResetKeywordStream(state_->spotter, state_->stream);
    return std::nullopt;
  }

  tracef(
    "fa_kws: process begin samples=%zu sr=%d vad_prob=%.4f\n",
    samples.size(),
    static_cast<int>(sample_rate),
    static_cast<double>(vad_prob));

  tracef("fa_kws: calling SherpaOnnxOnlineStreamAcceptWaveform()\n");
  SherpaOnnxOnlineStreamAcceptWaveform(
    state_->stream,
    sample_rate,
    samples.data(),
    static_cast<int32_t>(samples.size()));
  tracef("fa_kws: returned SherpaOnnxOnlineStreamAcceptWaveform()\n");

  while (SherpaOnnxIsKeywordStreamReady(state_->spotter, state_->stream)) {
    tracef("fa_kws: SherpaOnnxDecodeKeywordStream()\n");
    SherpaOnnxDecodeKeywordStream(state_->spotter, state_->stream);
  }

  tracef("fa_kws: SherpaOnnxGetKeywordResult()\n");
  const SherpaOnnxKeywordResult *result =
    SherpaOnnxGetKeywordResult(state_->spotter, state_->stream);
  if (!result) {
    tracef("fa_kws: result=null\n");
    return std::nullopt;
  }

  const bool has_keyword =
    result->keyword != nullptr && std::strlen(result->keyword) > 0;

  if (!has_keyword) {
    tracef("fa_kws: keyword empty, destroying result\n");
    SherpaOnnxDestroyKeywordResult(result);
    return std::nullopt;
  }

  const auto elapsed_ms =
    has_detect_time_
      ? std::chrono::duration_cast<std::chrono::milliseconds>(now - last_detect_time_).count()
      : std::numeric_limits<long long>::max();

  if (elapsed_ms < config_.cooldown.count()) {
    tracef(
      "fa_kws: cooldown active elapsed_ms=%lld cooldown_ms=%lld\n",
      static_cast<long long>(elapsed_ms),
      static_cast<long long>(config_.cooldown.count()));
    SherpaOnnxDestroyKeywordResult(result);
    return std::nullopt;
  }

  last_detect_time_ = now;
  has_detect_time_ = true;

  KwsDetection det;
  det.keyword = result->keyword ? result->keyword : "";
  det.score = 1.0f;
  det.start_time_sec = result->start_time;

  tracef("fa_kws: detected keyword='%s' start_time=%.3f\n",
         det.keyword.c_str(),
         det.start_time_sec);
  SherpaOnnxDestroyKeywordResult(result);

  // Reset stream state after a detection, as recommended by the sherpa-onnx API.
  SherpaOnnxResetKeywordStream(state_->spotter, state_->stream);

  return det;
}

void SherpaOnnxKwsBackend::reset()
{
  requireReady("reset");
  SherpaOnnxResetKeywordStream(state_->spotter, state_->stream);
  has_detect_time_ = false;
}

void SherpaOnnxKwsBackend::resetHard()
{
  if (!state_ || !state_->spotter) {
    throw std::runtime_error("SherpaOnnxKwsBackend resetHard requested without keyword spotter");
  }
  if (state_->stream) {
    SherpaOnnxDestroyOnlineStream(state_->stream);
    state_->stream = nullptr;
  }
  state_->stream = const_cast<SherpaOnnxOnlineStream*>(
    SherpaOnnxCreateKeywordStream(state_->spotter));
  if (!state_->stream) {
    throw std::runtime_error(
      "SherpaOnnxKwsBackend: SherpaOnnxCreateKeywordStream() failed during resetHard");
  }
  has_detect_time_ = false;
}

void SherpaOnnxKwsBackend::validateConfig() const
{
  if (config_.target_sample_rate <= 0) {
    throw std::invalid_argument("backend.target_sample_rate must be > 0");
  }
  if (config_.model_num_threads <= 0) {
    throw std::invalid_argument("backend.model_num_threads must be > 0");
  }
  if (!isSupportedSherpaOnnxExecutionProvider(config_.execution_provider)) {
    throw std::invalid_argument(
      "unsupported backend.execution_provider for sherpa_onnx_kws: " +
      config_.execution_provider +
      "; supported providers: " +
      supportedSherpaOnnxExecutionProvidersForMessage());
  }
  requireReadableRegularFile("backend.encoder", config_.encoder_path);
  requireReadableRegularFile("backend.decoder", config_.decoder_path);
  requireReadableRegularFile("backend.joiner", config_.joiner_path);
  requireReadableRegularFile("backend.tokens", config_.tokens_path);
  requireReadableRegularFile("backend.keywords", config_.keywords_path);
  if (config_.max_active_paths <= 0) {
    throw std::invalid_argument("backend.max_active_paths must be > 0");
  }
  if (config_.num_trailing_blanks < 0) {
    throw std::invalid_argument("backend.num_trailing_blanks must be >= 0");
  }
  if (!std::isfinite(config_.keywords_score) || config_.keywords_score <= 0.0f) {
    throw std::invalid_argument("backend.keywords_score must be finite and > 0");
  }
  if (!std::isfinite(config_.keywords_threshold) || config_.keywords_threshold <= 0.0f) {
    throw std::invalid_argument("backend.keywords_threshold must be finite and > 0");
  }
  if (!isValidVadGateThreshold(static_cast<double>(config_.vad_threshold))) {
    throw std::invalid_argument("vad_threshold must be finite and in (0.0, 1.0]");
  }
  if (config_.cooldown.count() < 0) {
    throw std::invalid_argument("backend.cooldown must be >= 0 ms");
  }
}

void SherpaOnnxKwsBackend::requireReady(const char *operation) const
{
  if (!state_ || !state_->spotter) {
    throw std::runtime_error(std::string("SherpaOnnxKwsBackend ") + operation +
                             " requested without keyword spotter");
  }
  if (!state_->stream) {
    throw std::runtime_error(std::string("SherpaOnnxKwsBackend ") + operation +
                             " requested without keyword stream");
  }
}

}  // namespace fa_kws

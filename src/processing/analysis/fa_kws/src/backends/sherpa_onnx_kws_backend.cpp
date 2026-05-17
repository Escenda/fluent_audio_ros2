#include "fa_kws/backends/sherpa_onnx_kws_backend.hpp"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>

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

}  // namespace

namespace fa_kws
{

SherpaOnnxKwsBackend::SherpaOnnxKwsBackend(const SherpaOnnxKwsBackendConfig &config)
: config_(config),
  last_detect_time_(std::chrono::steady_clock::now()),
  has_detect_time_(false)
{
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
  model_config.provider = config_.model_provider.c_str();
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

  spotter_ = SherpaOnnxCreateKeywordSpotter(&kws_config);
  if (!spotter_) {
    throw std::runtime_error("SherpaOnnxKwsBackend: SherpaOnnxCreateKeywordSpotter() failed");
  }

  stream_ = const_cast<SherpaOnnxOnlineStream*>(SherpaOnnxCreateKeywordStream(spotter_));
  if (!stream_) {
    SherpaOnnxDestroyKeywordSpotter(spotter_);
    spotter_ = nullptr;
    throw std::runtime_error("SherpaOnnxKwsBackend: SherpaOnnxCreateKeywordStream() failed");
  }
}

SherpaOnnxKwsBackend::~SherpaOnnxKwsBackend()
{
  if (stream_ != nullptr) {
    SherpaOnnxDestroyOnlineStream(stream_);
    stream_ = nullptr;
  }
  if (spotter_ != nullptr) {
    SherpaOnnxDestroyKeywordSpotter(spotter_);
    spotter_ = nullptr;
  }
}

std::optional<KwsDetection> SherpaOnnxKwsBackend::process(
  const std::vector<float> &samples,
  std::int32_t sample_rate,
  float vad_prob,
  std::chrono::steady_clock::time_point now)
{
  if (!spotter_ || !stream_ || samples.empty()) {
    return std::nullopt;
  }

  tracef(
    "fa_kws: process begin samples=%zu sr=%d vad_prob=%.4f\n",
    samples.size(),
    static_cast<int>(sample_rate),
    static_cast<double>(vad_prob));

  tracef("fa_kws: calling SherpaOnnxOnlineStreamAcceptWaveform()\n");
  SherpaOnnxOnlineStreamAcceptWaveform(
    stream_,
    sample_rate,
    samples.data(),
    static_cast<int32_t>(samples.size()));
  tracef("fa_kws: returned SherpaOnnxOnlineStreamAcceptWaveform()\n");

  while (SherpaOnnxIsKeywordStreamReady(spotter_, stream_)) {
    tracef("fa_kws: SherpaOnnxDecodeKeywordStream()\n");
    SherpaOnnxDecodeKeywordStream(spotter_, stream_);
  }

  tracef("fa_kws: SherpaOnnxGetKeywordResult()\n");
  const SherpaOnnxKeywordResult *result = SherpaOnnxGetKeywordResult(spotter_, stream_);
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

  // If vad_threshold <= 0.0, treat it as "no VAD gating".
  if (config_.vad_threshold > 0.0f && vad_prob < config_.vad_threshold) {
    tracef("fa_kws: VAD gate dropped (vad_prob=%.4f threshold=%.4f)\n",
           static_cast<double>(vad_prob),
           static_cast<double>(config_.vad_threshold));
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
  det.start_time_sec = result->start_time;

  tracef("fa_kws: detected keyword='%s' start_time=%.3f\n",
         det.keyword.c_str(),
         det.start_time_sec);
  SherpaOnnxDestroyKeywordResult(result);

  // Reset stream state after a detection, as recommended by the sherpa-onnx API.
  SherpaOnnxResetKeywordStream(spotter_, stream_);

  return det;
}

void SherpaOnnxKwsBackend::reset()
{
  if (spotter_ && stream_) {
    SherpaOnnxResetKeywordStream(spotter_, stream_);
    has_detect_time_ = false;
  }
}

void SherpaOnnxKwsBackend::resetHard()
{
  if (stream_) {
    SherpaOnnxDestroyOnlineStream(stream_);
    stream_ = nullptr;
  }
  if (spotter_) {
    stream_ = const_cast<SherpaOnnxOnlineStream*>(SherpaOnnxCreateKeywordStream(spotter_));
  }
  has_detect_time_ = false;
}

}  // namespace fa_kws

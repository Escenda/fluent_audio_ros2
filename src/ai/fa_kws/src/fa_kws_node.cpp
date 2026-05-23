#include <atomic>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "fa_interfaces/msg/audio_frame.hpp"
#include "fa_interfaces/msg/voice_activity.hpp"
#include "fa_interfaces/msg/wake_word_result.hpp"
#include "rclcpp/rclcpp.hpp"

#include "fa_kws/audio_utils.hpp"
#include "fa_kws/backends/factory.hpp"
#include "fa_kws/backends/kws_backend.hpp"
#include "fa_kws/vad_gate.hpp"

namespace fa_kws
{

using fa_interfaces::msg::AudioFrame;
using fa_interfaces::msg::VoiceActivity;
using fa_interfaces::msg::WakeWordResult;

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

std::string comparableIdentity(const std::string &value)
{
  if (!value.empty() && value.front() == '/') {
    return value.substr(1);
  }
  return value;
}

bool sameIdentityString(const std::string &left, const std::string &right)
{
  return comparableIdentity(left) == comparableIdentity(right);
}

std::size_t millisecondsToSamples(int milliseconds, int sample_rate)
{
  return static_cast<std::size_t>(
    (static_cast<long long>(milliseconds) * static_cast<long long>(sample_rate)) / 1000LL);
}

rclcpp::QoS makeExplicitQos(int depth, bool reliable)
{
  if (depth <= 0) {
    throw std::runtime_error("qos depth must be greater than zero");
  }
  rclcpp::QoS qos(rclcpp::KeepLast(static_cast<std::size_t>(depth)));
  if (reliable) {
    qos.reliable();
  } else {
    qos.best_effort();
  }
  return qos;
}

struct KwsStreamGateConfig
{
  bool enabled{true};
  int sample_rate{16000};
  int pre_roll_ms{500};
  int hangover_ms{900};
  int merge_gap_ms{400};
};

struct KwsStreamGateDecision
{
  bool used_audio{false};
  bool should_process{false};
  bool reset_stream{false};
  bool opened{false};
  bool merged{false};
  bool closed{false};
  std::uint64_t segment_id{0};
  std::size_t buffered_samples{0};
  std::size_t probe_samples{0};
  std::vector<float> samples;
};

class KwsStreamGate
{
public:
  explicit KwsStreamGate(KwsStreamGateConfig config)
  : config_(config),
    pre_roll_samples_(millisecondsToSamples(config.pre_roll_ms, config.sample_rate))
  {
    if (config_.sample_rate <= 0) {
      throw std::invalid_argument("KWS stream gate sample_rate must be positive");
    }
    if (config_.pre_roll_ms < 0 || config_.hangover_ms < 0 || config_.merge_gap_ms < 0) {
      throw std::invalid_argument(
              "KWS stream gate pre_roll_ms, hangover_ms, and merge_gap_ms must be >= 0");
    }
  }

  KwsStreamGateDecision push(
    const std::vector<float> &samples,
    bool vad_open,
    std::chrono::steady_clock::time_point now)
  {
    KwsStreamGateDecision decision;
    decision.segment_id = segment_id_;
    decision.buffered_samples = pre_roll_.size();

    if (vad_open) {
      last_voice_time_ = now;
      has_voice_time_ = true;
      if (!in_segment_) {
        ++segment_id_;
        decision.segment_id = segment_id_;
        decision.opened = true;
        in_segment_ = true;
        waiting_for_merge_ = false;
        decision.samples = copyPreRoll();
        decision.samples.insert(decision.samples.end(), samples.begin(), samples.end());
      } else {
        decision.merged = waiting_for_merge_;
        waiting_for_merge_ = false;
        decision.samples = samples;
      }
      decision.used_audio = true;
      decision.should_process = !decision.samples.empty();
      decision.probe_samples = decision.samples.size();
      appendPreRoll(samples);
      decision.buffered_samples = pre_roll_.size();
      return decision;
    }

    if (in_segment_ && has_voice_time_) {
      const auto silence = now - last_voice_time_;
      if (silence <= std::chrono::milliseconds(config_.hangover_ms)) {
        waiting_for_merge_ = false;
        decision.used_audio = true;
        decision.should_process = true;
        decision.samples = samples;
        decision.probe_samples = decision.samples.size();
      } else if (
        silence > std::chrono::milliseconds(config_.hangover_ms + config_.merge_gap_ms))
      {
        in_segment_ = false;
        waiting_for_merge_ = false;
        decision.closed = true;
        decision.reset_stream = true;
      } else {
        waiting_for_merge_ = true;
        decision.used_audio = true;
      }
    }

    appendPreRoll(samples);
    decision.segment_id = segment_id_;
    decision.buffered_samples = pre_roll_.size();
    return decision;
  }

  void forceClose()
  {
    in_segment_ = false;
    waiting_for_merge_ = false;
  }

private:
  void appendPreRoll(const std::vector<float> &samples)
  {
    for (const float sample : samples) {
      pre_roll_.push_back(sample);
    }
    while (pre_roll_.size() > pre_roll_samples_) {
      pre_roll_.pop_front();
    }
  }

  std::vector<float> copyPreRoll() const
  {
    std::vector<float> out;
    out.reserve(pre_roll_.size());
    for (const float sample : pre_roll_) {
      out.push_back(sample);
    }
    return out;
  }

  KwsStreamGateConfig config_;
  std::deque<float> pre_roll_;
  std::size_t pre_roll_samples_{0};
  std::uint64_t segment_id_{0};
  bool in_segment_{false};
  bool has_voice_time_{false};
  bool waiting_for_merge_{false};
  std::chrono::steady_clock::time_point last_voice_time_{};
};

}  // namespace

class FaKwsNode : public rclcpp::Node
{
public:
  FaKwsNode()
  : rclcpp::Node("fa_kws")
  {
    loadParameters();
    validateTopicBindingsOrThrow();
    validateBackendContractOrThrow();
    vad_gate_ = std::make_unique<VadGate>(vad_gate_config_);
    stream_gate_config_.sample_rate = target_sample_rate_;
    if (stream_gate_config_.enabled) {
      stream_gate_ = std::make_unique<KwsStreamGate>(stream_gate_config_);
    }

    KwsBackendSettings backend_settings;
    backend_settings.name = backend_name_;
    backend_settings.target_sample_rate = target_sample_rate_;
    backend_settings.model_num_threads = model_num_threads_;
    backend_settings.execution_provider = execution_provider_;
    backend_settings.encoder_path = encoder_path_;
    backend_settings.decoder_path = decoder_path_;
    backend_settings.joiner_path = joiner_path_;
    backend_settings.tokens_path = tokens_path_;
    backend_settings.keywords_path = keywords_path_;
    backend_settings.max_active_paths = kws_max_active_paths_;
    backend_settings.num_trailing_blanks = kws_num_trailing_blanks_;
    backend_settings.keywords_score = static_cast<float>(kws_keywords_score_);
    backend_settings.keywords_threshold = static_cast<float>(kws_keywords_threshold_);
    backend_settings.cooldown = std::chrono::milliseconds(cooldown_ms_);
    backend_settings.command = backend_command_;
    backend_settings.args = backend_args_;
    backend_settings.stream_args = backend_stream_args_;
    backend_settings.health_args = backend_health_args_;
    backend_settings.timeout_sec = backend_timeout_sec_;
    backend_settings.workspace_dir = backend_workspace_dir_;
    backend_settings.cleanup_audio_files = backend_cleanup_audio_files_;

    kws_backend_ = buildKwsBackend(backend_settings);

    setupCommunication();
    setupDebug();

    RCLCPP_INFO(
      this->get_logger(),
      "fa_kws initialized: audio_topic=%s expected_source_id=%s expected_stream_id=%s output_topic=%s target_sr=%d provider=%s vad_gate=%s stream_gate=%s pre_roll_ms=%d hangover_ms=%d merge_gap_ms=%d",
      audio_topic_.c_str(),
      expected_source_id_.c_str(),
      expected_stream_id_.c_str(),
      output_topic_.c_str(),
      target_sample_rate_,
      execution_provider_.c_str(),
      vad_gate_config_.enabled ? "enabled" : "disabled",
      stream_gate_config_.enabled ? "enabled" : "disabled",
      stream_gate_config_.pre_roll_ms,
      stream_gate_config_.hangover_ms,
      stream_gate_config_.merge_gap_ms);
  }

  ~FaKwsNode() override
  = default;

private:
  void validateTopicBindingsOrThrow() const
  {
    if (audio_topic_.empty()) {
      throw std::runtime_error("audio_topic is required");
    }
    if (output_topic_.empty()) {
      throw std::runtime_error("output_topic is required");
    }
    if (expected_source_id_.empty()) {
      throw std::runtime_error("expected_source_id is required");
    }
    if (expected_stream_id_.empty()) {
      throw std::runtime_error("expected_stream_id is required");
    }
    if (sameIdentityString(expected_stream_id_, audio_topic_)) {
      throw std::runtime_error("expected_stream_id must be distinct from ROS audio_topic");
    }
    if (sameIdentityString(expected_stream_id_, output_topic_)) {
      throw std::runtime_error("expected_stream_id must be distinct from ROS output_topic");
    }
    if (sameIdentityString(audio_topic_, output_topic_)) {
      throw std::runtime_error("audio_topic must be distinct from output_topic");
    }
    if (!vad_topic_.empty() && sameIdentityString(vad_topic_, audio_topic_)) {
      throw std::runtime_error("control VAD topic must be distinct from audio_topic");
    }
    if (!vad_topic_.empty() && sameIdentityString(vad_topic_, output_topic_)) {
      throw std::runtime_error("control VAD topic must be distinct from output_topic");
    }
  }

  void loadParameters()
  {
    audio_topic_ = this->declare_parameter<std::string>("audio_topic");
    expected_stream_id_ = this->declare_parameter<std::string>("expected_stream_id");
    output_topic_ = this->declare_parameter<std::string>("output_topic");
    expected_source_id_ = this->declare_parameter<std::string>("expected_source_id");
    backend_name_ = this->declare_parameter<std::string>("backend.name");

    target_sample_rate_ = this->declare_parameter<int>("target_sample_rate");
    cooldown_ms_ = this->declare_parameter<int>("cooldown_ms");
    debug_status_period_sec_ = this->declare_parameter<double>("debug.status_period_sec");
    audio_qos_depth_ = this->declare_parameter<int>("audio.qos.depth");
    audio_qos_reliable_ = this->declare_parameter<bool>("audio.qos.reliable");
    output_qos_depth_ = this->declare_parameter<int>("output.qos.depth");
    output_qos_reliable_ = this->declare_parameter<bool>("output.qos.reliable");

    encoder_path_ = this->declare_parameter<std::string>("model.encoder");
    decoder_path_ = this->declare_parameter<std::string>("model.decoder");
    joiner_path_ = this->declare_parameter<std::string>("model.joiner");
    tokens_path_ = this->declare_parameter<std::string>("model.tokens");
    keywords_path_ = this->declare_parameter<std::string>("kws.keywords_file");

    model_num_threads_ = this->declare_parameter<int>("model.num_threads");
    execution_provider_ = this->declare_parameter<std::string>("backend.execution_provider");
    backend_command_ = this->declare_parameter<std::string>("backend.command");
    // Declare an empty typed sentinel so missing backend args reach backend validation
    // as a clear required-contract error instead of ROS's untyped empty-array failure.
    backend_args_ =
      this->declare_parameter<std::vector<std::string>>(
        "backend.args",
        std::vector<std::string>{});
    backend_stream_args_ =
      this->declare_parameter<std::vector<std::string>>(
        "backend.stream_args",
        std::vector<std::string>{});
    backend_health_args_ =
      this->declare_parameter<std::vector<std::string>>(
        "backend.health_args",
        std::vector<std::string>{});
    backend_timeout_sec_ = this->declare_parameter<double>("backend.timeout_sec");
    backend_workspace_dir_ = this->declare_parameter<std::string>("backend.workspace_dir");
    backend_cleanup_audio_files_ = this->declare_parameter<bool>("backend.cleanup_audio_files");

    kws_max_active_paths_ = this->declare_parameter<int>("kws.max_active_paths");
    kws_num_trailing_blanks_ = this->declare_parameter<int>("kws.num_trailing_blanks");
    kws_keywords_score_ = this->declare_parameter<double>("kws.keywords_score");
    kws_keywords_threshold_ = this->declare_parameter<double>("kws.keywords_threshold");

    stream_gate_config_.enabled =
      this->declare_parameter<bool>("stream_gate.enabled", true);
    stream_gate_config_.pre_roll_ms =
      this->declare_parameter<int>("stream_gate.pre_roll_ms", 500);
    stream_gate_config_.hangover_ms =
      this->declare_parameter<int>("stream_gate.hangover_ms", 900);
    stream_gate_config_.merge_gap_ms =
      this->declare_parameter<int>("stream_gate.merge_gap_ms", 400);
    kws_diagnostics_enabled_ =
      this->declare_parameter<bool>("diagnostics.enabled", false);

    loadVoiceActivityGateParameters();
  }

  void loadVoiceActivityGateParameters()
  {
    control_default_enabled_ = this->declare_parameter<bool>("control.default_enabled", true);
    control_inputs_ =
      this->declare_parameter<std::vector<std::string>>(
        "control.inputs",
        std::vector<std::string>{});
    if (control_inputs_.empty()) {
      vad_gate_config_ = VadGateConfig{};
      return;
    }
    if (control_inputs_.size() != 1) {
      throw std::runtime_error("fa_kws supports exactly one control input");
    }

    vad_control_id_ = control_inputs_.front();
    if (vad_control_id_.empty()) {
      throw std::runtime_error("fa_kws control input id must be non-empty");
    }

    const std::string prefix = "control." + vad_control_id_ + ".";
    const std::string action = this->declare_parameter<std::string>(prefix + "action");
    if (action != "gate") {
      throw std::runtime_error("fa_kws control action must be gate");
    }
    const std::string msg_type = this->declare_parameter<std::string>(prefix + "msg_type");
    if (msg_type != "fa_interfaces/msg/VoiceActivity") {
      throw std::runtime_error(
              "fa_kws control msg_type must be fa_interfaces/msg/VoiceActivity");
    }

    vad_topic_ = this->declare_parameter<std::string>(prefix + "topic");
    const std::string control_source_id =
      this->declare_parameter<std::string>(prefix + "source_id", "");
    const std::string control_stream_id = this->declare_parameter<std::string>(prefix + "stream_id");
    vad_probability_field_ =
      this->declare_parameter<std::string>(prefix + "probability_field");
    if (vad_probability_field_ != "probability" &&
      vad_probability_field_ != "smoothed_probability")
    {
      throw std::runtime_error(
              "fa_kws control probability_field must be probability or smoothed_probability");
    }

    vad_probability_gate_ = this->declare_parameter<double>(prefix + "probability_gate");
    vad_max_age_ms_ = this->declare_parameter<int>(prefix + "max_age_ms");
    vad_qos_depth_ = this->declare_parameter<int>(prefix + "qos.depth");
    vad_qos_reliable_ = this->declare_parameter<bool>(prefix + "qos.reliable");

    vad_gate_config_.enabled = true;
    vad_gate_config_.default_enabled = control_default_enabled_;
    vad_gate_config_.source_id =
      control_source_id.empty() ? expected_source_id_ : control_source_id;
    vad_gate_config_.stream_id = control_stream_id;
    vad_gate_config_.probability_gate = vad_probability_gate_;
    vad_gate_config_.max_age = std::chrono::milliseconds(vad_max_age_ms_);
  }

  void validateBackendContractOrThrow() const
  {
    if (backend_name_ != "sherpa_onnx_kws") {
      return;
    }
    if (backend_args_.empty() && backend_stream_args_.empty()) {
      throw std::runtime_error("backend.args or backend.stream_args is required for backend.name=sherpa_onnx_kws");
    }
    if (backend_health_args_.empty()) {
      throw std::runtime_error("backend.health_args is required for backend.name=sherpa_onnx_kws");
    }
  }

  void setupCommunication()
  {
    const rclcpp::QoS qos_audio = makeExplicitQos(audio_qos_depth_, audio_qos_reliable_);
    const rclcpp::QoS qos_output = makeExplicitQos(output_qos_depth_, output_qos_reliable_);

    wake_pub_ = this->create_publisher<WakeWordResult>(output_topic_, qos_output);

    if (vad_gate_config_.enabled) {
      const rclcpp::QoS qos_vad = makeExplicitQos(vad_qos_depth_, vad_qos_reliable_);
      vad_sub_ = this->create_subscription<VoiceActivity>(
        vad_topic_, qos_vad,
        std::bind(&FaKwsNode::onVoiceActivity, this, std::placeholders::_1));
    }

    audio_sub_ = this->create_subscription<AudioFrame>(
      audio_topic_, qos_audio,
      std::bind(&FaKwsNode::onAudio, this, std::placeholders::_1));
  }

  void setupDebug()
  {
    if (debug_status_period_sec_ <= 0.0) {
      return;
    }
    debug_timer_ = this->create_wall_timer(
      std::chrono::duration<double>(debug_status_period_sec_),
      std::bind(&FaKwsNode::onDebugTimer, this));
  }

  void onDebugTimer()
  {
    const auto frames = audio_frames_received_.load(std::memory_order_relaxed);
    const auto samples = audio_samples_received_.load(std::memory_order_relaxed);
    const auto gated = audio_frames_vad_rejected_.load(std::memory_order_relaxed);
    const auto worker_calls = kws_worker_calls_.load(std::memory_order_relaxed);
    const auto no_detections = kws_no_detections_.load(std::memory_order_relaxed);
    const auto detections = kws_detections_.load(std::memory_order_relaxed);
    const auto last_rx_ns = last_audio_rx_ns_.load(std::memory_order_relaxed);

    double since_audio_sec = -1.0;
    if (last_rx_ns > 0) {
      const auto now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now().time_since_epoch())
                            .count();
      since_audio_sec = static_cast<double>(now_ns - last_rx_ns) / 1e9;
    }

    const int32_t sr = last_sample_rate_.load(std::memory_order_relaxed);

    if (vad_gate_config_.enabled && vad_gate_) {
      const bool vad_allowed = vad_gate_->allows(std::chrono::steady_clock::now());
      RCLCPP_INFO(
        this->get_logger(),
        "fa_kws status: frames=%llu samples=%llu vad_gated=%llu worker_calls=%llu no_detection=%llu detected=%llu last_audio=%.2fs sr=%d vad_allowed=%s vad_prob=%.3f",
        static_cast<unsigned long long>(frames),
        static_cast<unsigned long long>(samples),
        static_cast<unsigned long long>(gated),
        static_cast<unsigned long long>(worker_calls),
        static_cast<unsigned long long>(no_detections),
        static_cast<unsigned long long>(detections),
        since_audio_sec,
        static_cast<int>(sr),
        vad_allowed ? "true" : "false",
        vad_gate_->probability());
      return;
    }

    RCLCPP_INFO(
      this->get_logger(),
      "fa_kws status: frames=%llu samples=%llu vad_gated=%llu worker_calls=%llu no_detection=%llu detected=%llu last_audio=%.2fs sr=%d",
      static_cast<unsigned long long>(frames),
      static_cast<unsigned long long>(samples),
      static_cast<unsigned long long>(gated),
      static_cast<unsigned long long>(worker_calls),
      static_cast<unsigned long long>(no_detections),
      static_cast<unsigned long long>(detections),
      since_audio_sec,
      static_cast<int>(sr));
  }

  void onVoiceActivity(const VoiceActivity::SharedPtr msg)
  {
    if (!msg || !vad_gate_) {
      return;
    }

    const double probability =
      vad_probability_field_ == "probability" ? msg->probability : msg->smoothed_probability;
    const auto now = std::chrono::steady_clock::now();
    const bool accepted = vad_gate_->update(
      VoiceActivityUpdate{
        msg->source_id,
        msg->stream_id,
        probability,
        msg->is_speech,
        msg->speech_ended,
      },
      now);
    if (!accepted) {
      RCLCPP_WARN(
        this->get_logger(),
        "Dropping VoiceActivity before KWS gate: source=%s stream=%s probability=%.3f",
        msg->source_id.c_str(),
        msg->stream_id.c_str(),
        probability);
      return;
    }
  }

  void onAudio(const AudioFrame::SharedPtr msg)
  {
    if (!msg) {
      throw std::runtime_error("fa_kws received null AudioFrame message");
    }
    if (!kws_backend_) {
      throw std::runtime_error("fa_kws backend is not initialized");
    }

    tracef(
      "fa_kws: onAudio stamp=%d.%09u source=%s stream=%s sr=%u ch=%u bit=%u layout=%s bytes=%zu\n",
      static_cast<int>(msg->header.stamp.sec),
      static_cast<unsigned int>(msg->header.stamp.nanosec),
      msg->source_id.c_str(),
      msg->stream_id.c_str(),
      msg->sample_rate,
      msg->channels,
      msg->bit_depth,
      msg->layout.c_str(),
      msg->data.size());

    audio_frames_received_.fetch_add(1, std::memory_order_relaxed);
    last_sample_rate_.store(static_cast<int32_t>(msg->sample_rate), std::memory_order_relaxed);
    const auto now = std::chrono::steady_clock::now();
    const auto now_rx_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                             now.time_since_epoch())
                             .count();
    last_audio_rx_ns_.store(now_rx_ns, std::memory_order_relaxed);

    std::vector<float> samples;
    try {
      samples = frameToCanonicalFloat(*msg, expected_source_id_, expected_stream_id_);
    } catch (const std::invalid_argument &e) {
      RCLCPP_ERROR(this->get_logger(), "Dropping invalid AudioFrame: %s", e.what());
      return;
    }
    const std::int32_t src_rate = static_cast<std::int32_t>(msg->sample_rate);
    if (src_rate <= 0) {
      RCLCPP_ERROR(this->get_logger(), "Dropping AudioFrame with invalid sample_rate=%d", src_rate);
      return;
    }

    if (src_rate != target_sample_rate_) {
      RCLCPP_ERROR(
        this->get_logger(),
        "Dropping AudioFrame with sample_rate=%d; expected target_sample_rate=%d",
        static_cast<int>(src_rate),
        target_sample_rate_);
      return;
    }

    const bool vad_allowed = !vad_gate_ || vad_gate_->allows(now);
    std::vector<float> backend_samples;
    std::uint64_t segment_id = 0;
    std::size_t buffered_samples = samples.size();

    if (stream_gate_) {
      const KwsStreamGateDecision decision = stream_gate_->push(samples, vad_allowed, now);
      segment_id = decision.segment_id;
      buffered_samples = decision.buffered_samples;
      if (kws_diagnostics_enabled_ && (decision.opened || decision.merged || decision.closed)) {
        RCLCPP_INFO(
          this->get_logger(),
          "KWS gate: segment=%llu opened=%s merged=%s closed=%s reset_stream=%s vad_allowed=%s buffered_ms=%.1f",
          static_cast<unsigned long long>(decision.segment_id),
          decision.opened ? "true" : "false",
          decision.merged ? "true" : "false",
          decision.closed ? "true" : "false",
          decision.reset_stream ? "true" : "false",
          vad_allowed ? "true" : "false",
          1000.0 * static_cast<double>(decision.buffered_samples) /
          static_cast<double>(target_sample_rate_));
      }
      if (decision.reset_stream) {
        try {
          kws_backend_->reset();
        } catch (const std::exception &e) {
          RCLCPP_FATAL(this->get_logger(), "KWS backend reset failed: %s", e.what());
          rclcpp::shutdown();
          throw;
        }
      }
      if (!decision.used_audio) {
        audio_frames_vad_rejected_.fetch_add(1, std::memory_order_relaxed);
        tracef(
          "fa_kws: KWS stream gate skipped audio vad_prob=%.4f vad_speech=%d\n",
          vad_gate_ ? vad_gate_->probability() : 1.0,
          vad_gate_ && vad_gate_->isSpeech() ? 1 : 0);
        return;
      }
      if (!decision.should_process) {
        return;
      }
      backend_samples = decision.samples;
    } else {
      if (!vad_allowed) {
        audio_frames_vad_rejected_.fetch_add(1, std::memory_order_relaxed);
        tracef(
          "fa_kws: VAD gate skipped audio vad_prob=%.4f vad_speech=%d\n",
          vad_gate_ ? vad_gate_->probability() : 1.0,
          vad_gate_ && vad_gate_->isSpeech() ? 1 : 0);
        return;
      }
      backend_samples = samples;
    }

    audio_samples_received_.fetch_add(samples.size(), std::memory_order_relaxed);

    std::optional<KwsDetection> detection;
    try {
      kws_worker_calls_.fetch_add(1, std::memory_order_relaxed);
      detection = kws_backend_->process(backend_samples, target_sample_rate_, now);
    } catch (const std::exception &e) {
      RCLCPP_FATAL(this->get_logger(), "KWS backend failed: %s", e.what());
      rclcpp::shutdown();
      throw;
    }
    if (!detection) {
      kws_no_detections_.fetch_add(1, std::memory_order_relaxed);
      return;
    }

    kws_detections_.fetch_add(1, std::memory_order_relaxed);
    if (kws_diagnostics_enabled_) {
      RCLCPP_INFO(
        this->get_logger(),
        "KWS stream: segment=%llu result=DETECTED keyword=%s score=%.3f chunk_ms=%.1f buffered_ms=%.1f calls=%llu",
        static_cast<unsigned long long>(segment_id),
        detection->keyword.c_str(),
        static_cast<double>(detection->score),
        1000.0 * static_cast<double>(backend_samples.size()) /
        static_cast<double>(target_sample_rate_),
        1000.0 * static_cast<double>(buffered_samples) /
        static_cast<double>(target_sample_rate_),
        static_cast<unsigned long long>(kws_worker_calls_.load(std::memory_order_relaxed)));
    }

    WakeWordResult out;
    out.header = msg->header;
    out.keyword = detection->keyword;
    out.score = detection->score;
    out.detected = true;

    wake_pub_->publish(out);
    if (stream_gate_) {
      stream_gate_->forceClose();
    }

    RCLCPP_INFO(
      this->get_logger(),
      "Wake word detected: keyword=%s",
      out.keyword.c_str());
  }

  std::string audio_topic_;
  std::string output_topic_;
  std::string expected_source_id_;
  std::string expected_stream_id_;
  std::string backend_name_;
  std::string vad_control_id_;
  std::string vad_topic_;
  std::string vad_probability_field_;

  int target_sample_rate_{};
  int cooldown_ms_{};
  int audio_qos_depth_{};
  bool audio_qos_reliable_{};
  int output_qos_depth_{};
  bool output_qos_reliable_{};
  bool control_default_enabled_{true};
  std::vector<std::string> control_inputs_;
  double vad_probability_gate_{};
  int vad_max_age_ms_{};
  int vad_qos_depth_{};
  bool vad_qos_reliable_{};

  std::string encoder_path_;
  std::string decoder_path_;
  std::string joiner_path_;
  std::string tokens_path_;
  std::string keywords_path_;

  int model_num_threads_{};
  std::string execution_provider_;
  std::string backend_command_;
  std::vector<std::string> backend_args_;
  std::vector<std::string> backend_stream_args_;
  std::vector<std::string> backend_health_args_;
  double backend_timeout_sec_{};
  std::string backend_workspace_dir_;
  bool backend_cleanup_audio_files_{};

  int kws_max_active_paths_{};
  int kws_num_trailing_blanks_{};
  double kws_keywords_score_{};
  double kws_keywords_threshold_{};

  VadGateConfig vad_gate_config_;
  std::unique_ptr<VadGate> vad_gate_;
  KwsStreamGateConfig stream_gate_config_;
  std::unique_ptr<KwsStreamGate> stream_gate_;
  std::unique_ptr<KwsBackend> kws_backend_;

  bool kws_diagnostics_enabled_{false};
  double debug_status_period_sec_{0.0};
  rclcpp::TimerBase::SharedPtr debug_timer_;
  std::atomic<std::uint64_t> audio_frames_received_{0};
  std::atomic<std::uint64_t> audio_frames_vad_rejected_{0};
  std::atomic<std::uint64_t> audio_samples_received_{0};
  std::atomic<std::uint64_t> kws_worker_calls_{0};
  std::atomic<std::uint64_t> kws_no_detections_{0};
  std::atomic<std::uint64_t> kws_detections_{0};
  std::atomic<std::int64_t> last_audio_rx_ns_{0};
  std::atomic<std::int32_t> last_sample_rate_{0};

  rclcpp::Subscription<AudioFrame>::SharedPtr audio_sub_;
  rclcpp::Subscription<VoiceActivity>::SharedPtr vad_sub_;
  rclcpp::Publisher<WakeWordResult>::SharedPtr wake_pub_;
};

}  // namespace fa_kws

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<fa_kws::FaKwsNode>();
    rclcpp::spin(node);
  } catch (const std::exception &e) {
    RCLCPP_FATAL(rclcpp::get_logger("fa_kws"), "Exception in fa_kws: %s", e.what());
    rclcpp::shutdown();
    return EXIT_FAILURE;
  }
  rclcpp::shutdown();
  return EXIT_SUCCESS;
}

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "fa_interfaces/msg/audio_frame.hpp"
#include "fa_interfaces/msg/vad_state.hpp"
#include "fa_interfaces/msg/wake_word_result.hpp"
#include "rclcpp/rclcpp.hpp"

#include "fa_kws/audio_utils.hpp"
#include "fa_kws/backends/factory.hpp"
#include "fa_kws/backends/kws_backend.hpp"
#include "fa_kws/vad_gate.hpp"
#include "fa_kws/vad_state_identity.hpp"

namespace fa_kws
{

using fa_interfaces::msg::AudioFrame;
using fa_interfaces::msg::VadState;
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

std::string trimCopy(const std::string &value)
{
  const auto first = value.find_first_not_of(" \t\n\r");
  if (first == std::string::npos) {
    return "";
  }
  const auto last = value.find_last_not_of(" \t\n\r");
  return value.substr(first, last - first + 1);
}

bool hasSurroundingWhitespace(const std::string &value)
{
  return trimCopy(value) != value;
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

}  // namespace

class FaKwsNode : public rclcpp::Node
{
public:
  FaKwsNode()
  : rclcpp::Node("fa_kws"),
    current_control_prob_(0.0f)
  {
    loadParameters();
    validateTopicBindingsOrThrow();
    validateControlInputOrThrow();
    validateBackendContractOrThrow();

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
    backend_settings.vad_threshold = static_cast<float>(control_probability_gate_);
    backend_settings.cooldown = std::chrono::milliseconds(cooldown_ms_);
    backend_settings.command = backend_command_;
    backend_settings.args = backend_args_;
    backend_settings.health_args = backend_health_args_;
    backend_settings.timeout_sec = backend_timeout_sec_;
    backend_settings.workspace_dir = backend_workspace_dir_;
    backend_settings.cleanup_audio_files = backend_cleanup_audio_files_;

    kws_backend_ = buildKwsBackend(backend_settings);

    setupCommunication();
    setupDebug();

    RCLCPP_INFO(
      this->get_logger(),
      "fa_kws initialized: audio_topic=%s expected_source_id=%s expected_stream_id=%s control_input=%s control_topic=%s output_topic=%s target_sr=%d provider=%s",
      audio_topic_.c_str(),
      expected_source_id_.c_str(),
      expected_stream_id_.c_str(),
      control_id_.c_str(),
      control_topic_.c_str(),
      output_topic_.c_str(),
      target_sample_rate_,
      execution_provider_.c_str());
  }

  ~FaKwsNode() override
  = default;

private:
  void validateTopicBindingsOrThrow() const
  {
    if (audio_topic_.empty()) {
      throw std::runtime_error("audio_topic is required");
    }
    if (control_topic_.empty()) {
      throw std::runtime_error("control input topic is required");
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
    if (sameIdentityString(expected_stream_id_, control_topic_)) {
      throw std::runtime_error("expected_stream_id must be distinct from ROS control input topic");
    }
    if (sameIdentityString(expected_stream_id_, output_topic_)) {
      throw std::runtime_error("expected_stream_id must be distinct from ROS output_topic");
    }
    if (sameIdentityString(audio_topic_, output_topic_)) {
      throw std::runtime_error("audio_topic must be distinct from output_topic");
    }
  }

  void validateControlInputOrThrow() const
  {
    if (control_default_enabled_) {
      throw std::runtime_error(
        "control.default_enabled=true is unsupported for fa_kws; explicit fresh control input is required");
    }
    if (control_id_.empty()) {
      throw std::runtime_error("control.inputs must contain exactly one control input ID");
    }
    if (hasSurroundingWhitespace(control_action_)) {
      throw std::runtime_error("control input action must not contain surrounding whitespace");
    }
    if (control_action_ != "topic") {
      throw std::runtime_error("control input action must be topic");
    }
    if (hasSurroundingWhitespace(control_topic_)) {
      throw std::runtime_error("control input topic must not contain surrounding whitespace");
    }
    if (hasSurroundingWhitespace(control_msg_type_)) {
      throw std::runtime_error("control input msg_type must not contain surrounding whitespace");
    }
    if (control_msg_type_ != "fa_interfaces/msg/VadState") {
      throw std::runtime_error("control input msg_type must be fa_interfaces/msg/VadState");
    }
    if (hasSurroundingWhitespace(control_source_id_)) {
      throw std::runtime_error("control input source_id must not contain surrounding whitespace");
    }
    if (control_source_id_.empty()) {
      throw std::runtime_error("control input source_id is required");
    }
    if (control_source_id_ != expected_source_id_) {
      throw std::runtime_error("control input source_id must match expected_source_id");
    }
    if (hasSurroundingWhitespace(control_stream_id_)) {
      throw std::runtime_error("control input stream_id must not contain surrounding whitespace");
    }
    if (control_stream_id_.empty()) {
      throw std::runtime_error("control input stream_id is required");
    }
    if (control_stream_id_ != expected_stream_id_) {
      throw std::runtime_error("control input stream_id must match expected_stream_id");
    }
    if (hasSurroundingWhitespace(control_probability_field_)) {
      throw std::runtime_error(
        "control input probability_field must not contain surrounding whitespace");
    }
    if (control_probability_field_ != "probability") {
      throw std::runtime_error("control input probability_field must be probability");
    }
    if (!isValidVadGateThreshold(control_probability_gate_)) {
      throw std::runtime_error("control input probability_gate must be finite and in (0.0, 1.0]");
    }
    if (control_max_age_ms_ <= 0) {
      throw std::runtime_error("control input max_age_ms must be greater than zero");
    }
    if (cooldown_ms_ < 0) {
      throw std::runtime_error("cooldown_ms must be zero or greater");
    }
    if (audio_qos_depth_ <= 0) {
      throw std::runtime_error("audio.qos.depth must be greater than zero");
    }
    if (control_qos_depth_ <= 0) {
      throw std::runtime_error("control input qos.depth must be greater than zero");
    }
    if (output_qos_depth_ <= 0) {
      throw std::runtime_error("output.qos.depth must be greater than zero");
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
    control_default_enabled_ = this->declare_parameter<bool>("control.default_enabled");
    loadControlInputParameters();
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
  }

  void loadControlInputParameters()
  {
    const auto control_ids = this->declare_parameter<std::vector<std::string>>("control.inputs");
    if (control_ids.empty()) {
      throw std::runtime_error("control.inputs must contain exactly one control input ID");
    }
    std::unordered_set<std::string> seen;
    for (const auto &control_id : control_ids) {
      const auto normalized_id = trimCopy(control_id);
      if (normalized_id.empty()) {
        throw std::runtime_error("control.inputs must not contain empty IDs");
      }
      if (normalized_id != control_id) {
        throw std::runtime_error("control.inputs IDs must not contain surrounding whitespace");
      }
      if (!seen.insert(control_id).second) {
        throw std::runtime_error("control.inputs must not contain duplicate IDs");
      }
    }
    if (control_ids.size() != 1U) {
      throw std::runtime_error("control.inputs must contain exactly one control input ID");
    }

    control_id_ = control_ids.front();
    const std::string prefix = "control." + control_id_;
    control_action_ = this->declare_parameter<std::string>(prefix + ".action");
    control_topic_ = this->declare_parameter<std::string>(prefix + ".topic");
    control_msg_type_ = this->declare_parameter<std::string>(prefix + ".msg_type");
    control_source_id_ = this->declare_parameter<std::string>(prefix + ".source_id");
    control_stream_id_ = this->declare_parameter<std::string>(prefix + ".stream_id");
    control_probability_field_ =
      this->declare_parameter<std::string>(prefix + ".probability_field");
    control_probability_gate_ = this->declare_parameter<double>(prefix + ".probability_gate");
    control_max_age_ms_ = this->declare_parameter<int>(prefix + ".max_age_ms");
    control_qos_depth_ = this->declare_parameter<int>(prefix + ".qos.depth");
    control_qos_reliable_ = this->declare_parameter<bool>(prefix + ".qos.reliable");
  }

  void validateBackendContractOrThrow() const
  {
    if (backend_name_ != "sherpa_onnx_kws") {
      return;
    }
    if (backend_args_.empty()) {
      throw std::runtime_error("backend.args is required for backend.name=sherpa_onnx_kws");
    }
    if (backend_health_args_.empty()) {
      throw std::runtime_error("backend.health_args is required for backend.name=sherpa_onnx_kws");
    }
  }

  void setupCommunication()
  {
    const rclcpp::QoS qos_audio = makeExplicitQos(audio_qos_depth_, audio_qos_reliable_);
    const rclcpp::QoS qos_control = makeExplicitQos(control_qos_depth_, control_qos_reliable_);
    const rclcpp::QoS qos_output = makeExplicitQos(output_qos_depth_, output_qos_reliable_);

    wake_pub_ = this->create_publisher<WakeWordResult>(output_topic_, qos_output);

    control_sub_ = this->create_subscription<VadState>(
      control_topic_, qos_control,
      std::bind(&FaKwsNode::onControl, this, std::placeholders::_1));

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
    const auto last_rx_ns = last_audio_rx_ns_.load(std::memory_order_relaxed);

    double since_audio_sec = -1.0;
    if (last_rx_ns > 0) {
      const auto now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now().time_since_epoch())
                            .count();
      since_audio_sec = static_cast<double>(now_ns - last_rx_ns) / 1e9;
    }

    const float control_prob = current_control_prob_.load(std::memory_order_relaxed);
    const int32_t sr = last_sample_rate_.load(std::memory_order_relaxed);

    RCLCPP_INFO(
      this->get_logger(),
      "fa_kws status: frames=%llu samples=%llu last_audio=%.2fs control_prob=%.3f sr=%d",
      static_cast<unsigned long long>(frames),
      static_cast<unsigned long long>(samples),
      since_audio_sec,
      static_cast<double>(control_prob),
      static_cast<int>(sr));
  }

  void onControl(const VadState::SharedPtr msg)
  {
    if (!msg) {
      return;
    }
    if (!vadStateMatchesAudioBinding(*msg, control_source_id_, control_stream_id_)) {
      clearControlState();
      RCLCPP_ERROR_THROTTLE(
        this->get_logger(),
        *this->get_clock(),
        2000,
        "Rejecting control input %s identity mismatch: source_id='%s' control_source_id='%s' stream_id='%s' control_stream_id='%s'",
        control_id_.c_str(),
        msg->source_id.c_str(),
        control_source_id_.c_str(),
        msg->stream_id.c_str(),
        control_stream_id_.c_str());
      return;
    }
    if (!isValidVadProbability(msg->probability)) {
      clearControlState();
      RCLCPP_ERROR_THROTTLE(
        this->get_logger(),
        *this->get_clock(),
        2000,
        "Rejecting invalid control input %s probability %.6f; probability must be finite in [0.0, 1.0]",
        control_id_.c_str(),
        static_cast<double>(msg->probability));
      return;
    }
    current_control_prob_.store(msg->probability, std::memory_order_relaxed);
    const auto now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                          std::chrono::steady_clock::now().time_since_epoch())
                          .count();
    last_control_rx_ns_.store(now_ns, std::memory_order_relaxed);
  }

  void clearControlState()
  {
    current_control_prob_.store(0.0f, std::memory_order_relaxed);
    last_control_rx_ns_.store(0, std::memory_order_relaxed);
  }

  bool readFreshControlProbability(std::int64_t now_ns, float &control_prob)
  {
    const auto last_control_ns = last_control_rx_ns_.load(std::memory_order_relaxed);
    if (last_control_ns <= 0) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(),
        *this->get_clock(),
        2000,
        "Dropping AudioFrame until first control input %s is received on %s",
        control_id_.c_str(),
        control_topic_.c_str());
      return false;
    }

    const auto control_age_ms = (now_ns - last_control_ns) / 1000000;
    if (control_age_ms > control_max_age_ms_) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(),
        *this->get_clock(),
        2000,
        "Dropping AudioFrame because latest control input %s is stale: age_ms=%lld max_age_ms=%d",
        control_id_.c_str(),
        static_cast<long long>(control_age_ms),
        control_max_age_ms_);
      return false;
    }

    control_prob = current_control_prob_.load(std::memory_order_relaxed);
    return true;
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

    audio_samples_received_.fetch_add(samples.size(), std::memory_order_relaxed);

    float control_prob = 0.0f;
    if (!readFreshControlProbability(now_rx_ns, control_prob)) {
      return;
    }
    if (!passesVadGate(control_prob, static_cast<float>(control_probability_gate_))) {
      RCLCPP_DEBUG(
        this->get_logger(),
        "Dropping AudioFrame below control probability gate: control_prob=%.6f threshold=%.6f",
        static_cast<double>(control_prob),
        control_probability_gate_);
      return;
    }

    std::optional<KwsDetection> detection;
    try {
      detection = kws_backend_->process(samples, target_sample_rate_, control_prob, now);
    } catch (const std::exception &e) {
      RCLCPP_FATAL(this->get_logger(), "KWS backend failed: %s", e.what());
      rclcpp::shutdown();
      throw;
    }
    if (!detection) {
      return;
    }

    WakeWordResult out;
    out.header = msg->header;
    out.keyword = detection->keyword;
    out.score = detection->score;
    out.detected = true;

    wake_pub_->publish(out);

    RCLCPP_INFO(
      this->get_logger(),
      "Wake word detected: keyword=%s control_prob=%.3f",
      out.keyword.c_str(),
      static_cast<double>(control_prob));
  }

  std::string audio_topic_;
  std::string output_topic_;
  std::string expected_source_id_;
  std::string expected_stream_id_;
  std::string backend_name_;

  int target_sample_rate_{};
  bool control_default_enabled_{};
  std::string control_id_;
  std::string control_action_;
  std::string control_topic_;
  std::string control_msg_type_;
  std::string control_source_id_;
  std::string control_stream_id_;
  std::string control_probability_field_;
  double control_probability_gate_{};
  int control_max_age_ms_{};
  int cooldown_ms_{};
  int audio_qos_depth_{};
  bool audio_qos_reliable_{};
  int control_qos_depth_{};
  bool control_qos_reliable_{};
  int output_qos_depth_{};
  bool output_qos_reliable_{};

  std::string encoder_path_;
  std::string decoder_path_;
  std::string joiner_path_;
  std::string tokens_path_;
  std::string keywords_path_;

  int model_num_threads_{};
  std::string execution_provider_;
  std::string backend_command_;
  std::vector<std::string> backend_args_;
  std::vector<std::string> backend_health_args_;
  double backend_timeout_sec_{};
  std::string backend_workspace_dir_;
  bool backend_cleanup_audio_files_{};

  int kws_max_active_paths_{};
  int kws_num_trailing_blanks_{};
  double kws_keywords_score_{};
  double kws_keywords_threshold_{};

  std::atomic<float> current_control_prob_;
  std::atomic<std::int64_t> last_control_rx_ns_{0};

  std::unique_ptr<KwsBackend> kws_backend_;

  double debug_status_period_sec_{0.0};
  rclcpp::TimerBase::SharedPtr debug_timer_;
  std::atomic<std::uint64_t> audio_frames_received_{0};
  std::atomic<std::uint64_t> audio_samples_received_{0};
  std::atomic<std::int64_t> last_audio_rx_ns_{0};
  std::atomic<std::int32_t> last_sample_rate_{0};

  rclcpp::Subscription<AudioFrame>::SharedPtr audio_sub_;
  rclcpp::Subscription<VadState>::SharedPtr control_sub_;
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

#include "fa_clock_drift/fa_clock_drift_node.hpp"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

#include "diagnostic_msgs/msg/diagnostic_status.hpp"
#include "diagnostic_msgs/msg/key_value.hpp"

namespace fa_clock_drift
{

namespace
{
constexpr int64_t kNanosecondsPerSecond = 1000000000LL;
constexpr long double kNanosecondsPerMillisecond = 1000000.0L;
constexpr long double kMaxBuiltinTimeNanoseconds =
  (static_cast<long double>(2147483647LL) * static_cast<long double>(kNanosecondsPerSecond)) +
  999999999.0L;

void pushKeyValue(
  diagnostic_msgs::msg::DiagnosticStatus & status,
  const std::string & key,
  const std::string & value)
{
  diagnostic_msgs::msg::KeyValue kv;
  kv.key = key;
  kv.value = value;
  status.values.push_back(kv);
}

builtin_interfaces::msg::Time nanosecondsToStamp(const int64_t nanoseconds)
{
  builtin_interfaces::msg::Time stamp;
  stamp.sec = static_cast<int32_t>(nanoseconds / kNanosecondsPerSecond);
  stamp.nanosec = static_cast<uint32_t>(nanoseconds % kNanosecondsPerSecond);
  return stamp;
}
}  // namespace

FaClockDriftNode::FaClockDriftNode()
: rclcpp::Node("fa_clock_drift")
{
  RCLCPP_INFO(this->get_logger(), "Starting FA Clock Drift node");
  loadParameters();
  setupInterfaces();
}

void FaClockDriftNode::loadParameters()
{
  this->declare_parameter<std::string>("input_topic");
  this->declare_parameter<std::string>("output_topic");
  this->declare_parameter<int>("expected.sample_rate");
  this->declare_parameter<int>("expected.channels");
  this->declare_parameter<std::string>("expected.encoding");
  this->declare_parameter<int>("expected.bit_depth");
  this->declare_parameter<std::string>("expected.layout");
  this->declare_parameter<double>("drift.ema_alpha");
  this->declare_parameter<double>("drift.max_correction_ms_per_frame");
  this->declare_parameter<double>("drift.reset_threshold_ms");
  this->declare_parameter<int>("qos.depth");
  this->declare_parameter<bool>("qos.reliable");
  this->declare_parameter<int>("diagnostics.publish_period_ms");

  config_.input_topic = this->get_parameter("input_topic").as_string();
  config_.output_topic = this->get_parameter("output_topic").as_string();
  config_.expected_sample_rate = this->get_parameter("expected.sample_rate").as_int();
  config_.expected_channels = this->get_parameter("expected.channels").as_int();
  config_.expected_encoding = this->get_parameter("expected.encoding").as_string();
  config_.expected_bit_depth = this->get_parameter("expected.bit_depth").as_int();
  config_.expected_layout = this->get_parameter("expected.layout").as_string();
  config_.drift_ema_alpha = this->get_parameter("drift.ema_alpha").as_double();
  config_.drift_max_correction_ms_per_frame =
    this->get_parameter("drift.max_correction_ms_per_frame").as_double();
  config_.drift_reset_threshold_ms = this->get_parameter("drift.reset_threshold_ms").as_double();
  config_.qos_depth = this->get_parameter("qos.depth").as_int();
  config_.qos_reliable = this->get_parameter("qos.reliable").as_bool();
  config_.diagnostics_publish_period_ms =
    this->get_parameter("diagnostics.publish_period_ms").as_int();

  if (config_.input_topic.empty()) {
    throw std::runtime_error("input_topic is required");
  }
  if (config_.output_topic.empty()) {
    throw std::runtime_error("output_topic is required");
  }
  if (config_.expected_sample_rate <= 0) {
    throw std::runtime_error("expected.sample_rate must be > 0");
  }
  if (config_.expected_channels <= 0) {
    throw std::runtime_error("expected.channels must be > 0");
  }
  if (config_.expected_encoding.empty()) {
    throw std::runtime_error("expected.encoding is required");
  }
  if (config_.expected_bit_depth <= 0) {
    throw std::runtime_error("expected.bit_depth must be > 0");
  }
  if ((config_.expected_bit_depth % 8) != 0) {
    throw std::runtime_error("expected.bit_depth must be byte-aligned");
  }
  if (config_.expected_layout.empty()) {
    throw std::runtime_error("expected.layout is required");
  }
  if (!std::isfinite(config_.drift_ema_alpha) ||
      config_.drift_ema_alpha <= 0.0 || config_.drift_ema_alpha > 1.0)
  {
    throw std::runtime_error("drift.ema_alpha must be finite and 0.0 < alpha <= 1.0");
  }
  if (!std::isfinite(config_.drift_max_correction_ms_per_frame) ||
      config_.drift_max_correction_ms_per_frame < 0.0)
  {
    throw std::runtime_error("drift.max_correction_ms_per_frame must be finite and >= 0");
  }
  if (!std::isfinite(config_.drift_reset_threshold_ms) ||
      config_.drift_reset_threshold_ms <= 0.0)
  {
    throw std::runtime_error("drift.reset_threshold_ms must be finite and > 0");
  }
  if (config_.qos_depth <= 0) {
    throw std::runtime_error("qos.depth must be > 0");
  }
  if (config_.diagnostics_publish_period_ms <= 0) {
    throw std::runtime_error("diagnostics.publish_period_ms must be > 0");
  }

  RCLCPP_INFO(
    this->get_logger(),
    "Clock drift config: input=%s output=%s expected=%dHz/%d/%s/%d/%s alpha=%.6f "
    "max_correction=%.6fms reset_threshold=%.6fms qos_depth=%d reliable=%s diag=%dms",
    config_.input_topic.c_str(),
    config_.output_topic.c_str(),
    config_.expected_sample_rate,
    config_.expected_channels,
    config_.expected_encoding.c_str(),
    config_.expected_bit_depth,
    config_.expected_layout.c_str(),
    config_.drift_ema_alpha,
    config_.drift_max_correction_ms_per_frame,
    config_.drift_reset_threshold_ms,
    config_.qos_depth,
    config_.qos_reliable ? "true" : "false",
    config_.diagnostics_publish_period_ms);
}

void FaClockDriftNode::setupInterfaces()
{
  rclcpp::QoS qos(static_cast<size_t>(config_.qos_depth));
  if (config_.qos_reliable) {
    qos.reliable();
  } else {
    qos.best_effort();
  }

  audio_pub_ = this->create_publisher<fa_interfaces::msg::AudioFrame>(config_.output_topic, qos);
  audio_sub_ = this->create_subscription<fa_interfaces::msg::AudioFrame>(
    config_.input_topic,
    qos,
    std::bind(&FaClockDriftNode::handleFrame, this, std::placeholders::_1));

  diag_pub_ = this->create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
    "diagnostics",
    rclcpp::SystemDefaultsQoS());
  diag_timer_ = this->create_wall_timer(
    std::chrono::milliseconds(config_.diagnostics_publish_period_ms),
    std::bind(&FaClockDriftNode::publishDiagnostics, this));
}

void FaClockDriftNode::handleFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg)
{
  frames_in_.fetch_add(1);
  if (!msg) {
    frames_dropped_.fetch_add(1);
    return;
  }

  if (!validateFrame(*msg)) {
    frames_dropped_.fetch_add(1);
    return;
  }

  fa_interfaces::msg::AudioFrame out;
  if (!correctFrame(*msg, out)) {
    frames_dropped_.fetch_add(1);
    return;
  }

  audio_pub_->publish(out);
  frames_out_.fetch_add(1);
}

bool FaClockDriftNode::validateFrame(const fa_interfaces::msg::AudioFrame & msg)
{
  if (msg.source_id.empty() || msg.stream_id.empty()) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "AudioFrame source_id and stream_id are required");
    return false;
  }
  if (msg.stream_id != config_.input_topic) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "AudioFrame stream_id mismatch: %s != %s",
      msg.stream_id.c_str(),
      config_.input_topic.c_str());
    return false;
  }
  if (msg.sample_rate != static_cast<uint32_t>(config_.expected_sample_rate) ||
      msg.channels != static_cast<uint32_t>(config_.expected_channels))
  {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "AudioFrame format mismatch: frame=%uHz/%u config=%dHz/%d",
      msg.sample_rate,
      msg.channels,
      config_.expected_sample_rate,
      config_.expected_channels);
    return false;
  }
  if (msg.encoding != config_.expected_encoding ||
      msg.bit_depth != static_cast<uint32_t>(config_.expected_bit_depth) ||
      msg.layout != config_.expected_layout)
  {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "AudioFrame encoding/layout mismatch: %s/%u/%s != %s/%d/%s",
      msg.encoding.c_str(),
      msg.bit_depth,
      msg.layout.c_str(),
      config_.expected_encoding.c_str(),
      config_.expected_bit_depth,
      config_.expected_layout.c_str());
    return false;
  }
  if (msg.data.empty() || (msg.data.size() % bytesPerSampleFrame()) != 0U) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "AudioFrame data size is empty or not aligned to configured sample frames");
    return false;
  }
  return true;
}

bool FaClockDriftNode::correctFrame(
  const fa_interfaces::msg::AudioFrame & in,
  fa_interfaces::msg::AudioFrame & out)
{
  std::lock_guard<std::mutex> lock(timeline_mutex_);

  long double input_timestamp_ns = 0.0L;
  if (!stampToNanoseconds(in.header.stamp, input_timestamp_ns)) {
    resetTimeline();
    timeline_resets_.fetch_add(1);
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping AudioFrame because input header.stamp is outside the supported timestamp range");
    return false;
  }
  if (input_timestamp_ns < 0.0L || input_timestamp_ns > kMaxBuiltinTimeNanoseconds) {
    resetTimeline();
    timeline_resets_.fetch_add(1);
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping AudioFrame because input header.stamp is negative or outside builtin_interfaces/Time range");
    return false;
  }

  if (hasDifferentStreamIdentity(in)) {
    RCLCPP_WARN(
      this->get_logger(),
      "AudioFrame source or format contract changed; resetting clock drift timeline");
    resetTimeline();
    timeline_resets_.fetch_add(1);
  }

  long double current_frame_duration_ns = 0.0L;
  if (!frameDurationNanoseconds(in, current_frame_duration_ns)) {
    resetTimeline();
    timeline_resets_.fetch_add(1);
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping AudioFrame because frame duration is not finite");
    return false;
  }

  if (!previous_output_timestamp_ns_.has_value()) {
    return publishBaselineFrame(in, out, 0.0L, current_frame_duration_ns);
  }
  if (!previous_frame_duration_ns_.has_value()) {
    resetTimeline();
    timeline_resets_.fetch_add(1);
    RCLCPP_WARN(
      this->get_logger(),
      "Dropping AudioFrame because previous frame duration is missing from drift timeline state");
    return false;
  }

  const long double expected_timestamp_ns =
    *previous_output_timestamp_ns_ + *previous_frame_duration_ns_;
  if (!std::isfinite(expected_timestamp_ns)) {
    resetTimeline();
    timeline_resets_.fetch_add(1);
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping AudioFrame because expected timestamp is not finite");
    return false;
  }

  const long double observed_drift_ns = input_timestamp_ns - expected_timestamp_ns;
  const long double reset_threshold_ns =
    static_cast<long double>(config_.drift_reset_threshold_ms) * kNanosecondsPerMillisecond;

  if (std::fabs(observed_drift_ns) > reset_threshold_ns) {
    timeline_resets_.fetch_add(1);
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Observed clock drift exceeded reset threshold; resetting timeline");
    return publishBaselineFrame(in, out, observed_drift_ns, current_frame_duration_ns);
  }

  const long double alpha = static_cast<long double>(config_.drift_ema_alpha);
  const long double next_estimate_ns =
    ((1.0L - alpha) * drift_estimate_ns_) + (alpha * observed_drift_ns);
  if (!std::isfinite(next_estimate_ns)) {
    resetTimeline();
    timeline_resets_.fetch_add(1);
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping AudioFrame because drift estimate is not finite");
    return false;
  }

  const long double bounded_correction_ns =
    boundCorrectionNanoseconds(next_estimate_ns, *previous_frame_duration_ns_);
  if (bounded_correction_ns != next_estimate_ns) {
    correction_limited_frames_.fetch_add(1);
  }

  const long double output_timestamp_ns = expected_timestamp_ns + bounded_correction_ns;
  builtin_interfaces::msg::Time output_stamp;
  if (!buildStamp(output_timestamp_ns, output_stamp)) {
    resetTimeline();
    timeline_resets_.fetch_add(1);
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping AudioFrame because computed output header.stamp is invalid");
    return false;
  }

  out = in;
  out.header.stamp = output_stamp;
  out.stream_id = config_.output_topic;
  previous_output_timestamp_ns_ = output_timestamp_ns;
  previous_frame_duration_ns_ = current_frame_duration_ns;
  drift_estimate_ns_ = next_estimate_ns;
  last_observed_drift_ns_ = observed_drift_ns;
  return true;
}

bool FaClockDriftNode::publishBaselineFrame(
  const fa_interfaces::msg::AudioFrame & in,
  fa_interfaces::msg::AudioFrame & out,
  const long double last_observed_drift_ns,
  const long double current_frame_duration_ns)
{
  long double input_timestamp_ns = 0.0L;
  if (!stampToNanoseconds(in.header.stamp, input_timestamp_ns)) {
    resetTimeline();
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping AudioFrame because baseline header.stamp is invalid");
    return false;
  }

  builtin_interfaces::msg::Time output_stamp;
  if (!buildStamp(input_timestamp_ns, output_stamp)) {
    resetTimeline();
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping AudioFrame because baseline output header.stamp is invalid");
    return false;
  }

  out = in;
  out.header.stamp = output_stamp;
  out.stream_id = config_.output_topic;
  activateStream(in, input_timestamp_ns);
  previous_frame_duration_ns_ = current_frame_duration_ns;
  drift_estimate_ns_ = 0.0L;
  last_observed_drift_ns_ = last_observed_drift_ns;
  return true;
}

bool FaClockDriftNode::hasDifferentStreamIdentity(
  const fa_interfaces::msg::AudioFrame & msg) const
{
  if (!active_stream_.has_value()) {
    return false;
  }
  return msg.source_id != active_stream_->source_id ||
         msg.stream_id != active_stream_->stream_id ||
         msg.sample_rate != active_stream_->sample_rate ||
         msg.channels != active_stream_->channels ||
         msg.encoding != active_stream_->encoding ||
         msg.bit_depth != active_stream_->bit_depth ||
         msg.layout != active_stream_->layout;
}

void FaClockDriftNode::activateStream(
  const fa_interfaces::msg::AudioFrame & msg,
  const long double output_timestamp_ns)
{
  ActiveStreamIdentity identity;
  identity.source_id = msg.source_id;
  identity.stream_id = msg.stream_id;
  identity.sample_rate = msg.sample_rate;
  identity.channels = msg.channels;
  identity.encoding = msg.encoding;
  identity.bit_depth = msg.bit_depth;
  identity.layout = msg.layout;
  active_stream_ = identity;
  previous_output_timestamp_ns_ = output_timestamp_ns;
}

void FaClockDriftNode::resetTimeline()
{
  active_stream_.reset();
  previous_output_timestamp_ns_.reset();
  previous_frame_duration_ns_.reset();
  drift_estimate_ns_ = 0.0L;
  last_observed_drift_ns_ = 0.0L;
}

bool FaClockDriftNode::frameDurationNanoseconds(
  const fa_interfaces::msg::AudioFrame & msg,
  long double & duration_ns) const
{
  const size_t bytes_per_sample_frame = bytesPerSampleFrame();
  if (bytes_per_sample_frame == 0U || msg.data.empty()) {
    return false;
  }

  const size_t sample_count = msg.data.size() / bytes_per_sample_frame;
  duration_ns =
    (static_cast<long double>(sample_count) /
     static_cast<long double>(config_.expected_sample_rate)) *
    static_cast<long double>(kNanosecondsPerSecond);
  return std::isfinite(duration_ns) && duration_ns > 0.0L;
}

bool FaClockDriftNode::stampToNanoseconds(
  const builtin_interfaces::msg::Time & stamp,
  long double & timestamp_ns) const
{
  if (stamp.nanosec >= static_cast<uint32_t>(kNanosecondsPerSecond)) {
    return false;
  }

  timestamp_ns =
    (static_cast<long double>(stamp.sec) * static_cast<long double>(kNanosecondsPerSecond)) +
    static_cast<long double>(stamp.nanosec);
  return std::isfinite(timestamp_ns);
}

bool FaClockDriftNode::buildStamp(
  const long double timestamp_ns,
  builtin_interfaces::msg::Time & stamp) const
{
  if (!std::isfinite(timestamp_ns)) {
    return false;
  }
  if (timestamp_ns < 0.0L) {
    return false;
  }
  if (timestamp_ns > kMaxBuiltinTimeNanoseconds) {
    return false;
  }

  const int64_t rounded_ns = static_cast<int64_t>(std::llround(timestamp_ns));
  if (rounded_ns < 0 ||
      static_cast<long double>(rounded_ns) > kMaxBuiltinTimeNanoseconds)
  {
    return false;
  }

  stamp = nanosecondsToStamp(rounded_ns);
  return true;
}

long double FaClockDriftNode::boundCorrectionNanoseconds(
  const long double correction_ns,
  const long double previous_frame_duration_ns) const
{
  const long double max_correction_ns =
    static_cast<long double>(config_.drift_max_correction_ms_per_frame) *
    kNanosecondsPerMillisecond;
  if (correction_ns > max_correction_ns) {
    return max_correction_ns;
  }
  long double negative_limit_ns = max_correction_ns;
  if (previous_frame_duration_ns <= 1.0L) {
    negative_limit_ns = 0.0L;
  } else if (negative_limit_ns > previous_frame_duration_ns - 1.0L) {
    negative_limit_ns = previous_frame_duration_ns - 1.0L;
  }
  if (correction_ns < -negative_limit_ns) {
    return -negative_limit_ns;
  }
  return correction_ns;
}

size_t FaClockDriftNode::bytesPerSampleFrame() const
{
  return static_cast<size_t>(config_.expected_channels) *
         (static_cast<size_t>(config_.expected_bit_depth) / 8U);
}

void FaClockDriftNode::publishDiagnostics()
{
  diagnostic_msgs::msg::DiagnosticArray array_msg;
  array_msg.header.stamp = this->now();

  long double drift_estimate_ms = 0.0L;
  long double last_observed_drift_ms = 0.0L;
  {
    std::lock_guard<std::mutex> lock(timeline_mutex_);
    drift_estimate_ms = drift_estimate_ns_ / kNanosecondsPerMillisecond;
    last_observed_drift_ms = last_observed_drift_ns_ / kNanosecondsPerMillisecond;
  }

  diagnostic_msgs::msg::DiagnosticStatus status;
  status.name = "fa_clock_drift";
  status.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
  status.message = "running";
  status.values.reserve(18);
  pushKeyValue(status, "input_topic", config_.input_topic);
  pushKeyValue(status, "output_topic", config_.output_topic);
  pushKeyValue(status, "expected_sample_rate", std::to_string(config_.expected_sample_rate));
  pushKeyValue(status, "expected_channels", std::to_string(config_.expected_channels));
  pushKeyValue(status, "expected_encoding", config_.expected_encoding);
  pushKeyValue(status, "expected_bit_depth", std::to_string(config_.expected_bit_depth));
  pushKeyValue(status, "expected_layout", config_.expected_layout);
  pushKeyValue(status, "drift_ema_alpha", std::to_string(config_.drift_ema_alpha));
  pushKeyValue(
    status,
    "drift_max_correction_ms_per_frame",
    std::to_string(config_.drift_max_correction_ms_per_frame));
  pushKeyValue(status, "drift_reset_threshold_ms", std::to_string(config_.drift_reset_threshold_ms));
  pushKeyValue(status, "qos_depth", std::to_string(config_.qos_depth));
  pushKeyValue(status, "qos_reliable", config_.qos_reliable ? "true" : "false");
  pushKeyValue(status, "frames_in", std::to_string(frames_in_.load()));
  pushKeyValue(status, "frames_out", std::to_string(frames_out_.load()));
  pushKeyValue(status, "frames_dropped", std::to_string(frames_dropped_.load()));
  pushKeyValue(status, "timeline_resets", std::to_string(timeline_resets_.load()));
  pushKeyValue(status, "drift_estimate_ms", std::to_string(static_cast<double>(drift_estimate_ms)));
  pushKeyValue(
    status,
    "last_observed_drift_ms",
    std::to_string(static_cast<double>(last_observed_drift_ms)));
  pushKeyValue(
    status,
    "correction_limited_frames",
    std::to_string(correction_limited_frames_.load()));

  array_msg.status.push_back(status);
  diag_pub_->publish(array_msg);
}

}  // namespace fa_clock_drift

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<fa_clock_drift::FaClockDriftNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return EXIT_SUCCESS;
  } catch (const std::exception & e) {
    RCLCPP_FATAL(rclcpp::get_logger("fa_clock_drift"), "Exception: %s", e.what());
    rclcpp::shutdown();
    return EXIT_FAILURE;
  }
}

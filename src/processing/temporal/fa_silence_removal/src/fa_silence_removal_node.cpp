#include "fa_silence_removal/fa_silence_removal_node.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "diagnostic_msgs/msg/diagnostic_status.hpp"
#include "diagnostic_msgs/msg/key_value.hpp"

namespace fa_silence_removal
{

namespace
{
constexpr const char * kEncodingFloat32 = "FLOAT32LE";
constexpr const char * kInterleavedLayout = "interleaved";
constexpr float kMinNormalizedSample = -1.0F;
constexpr float kMaxNormalizedSample = 1.0F;
constexpr double kMillisecondsPerSecond = 1000.0;

bool isFinite(double value)
{
  return std::isfinite(value);
}

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

float readFloatSample(const std::vector<uint8_t> & data, size_t sample_index)
{
  float sample = 0.0F;
  std::memcpy(&sample, data.data() + (sample_index * sizeof(float)), sizeof(float));
  return sample;
}
}  // namespace

FaSilenceRemovalNode::FaSilenceRemovalNode()
: rclcpp::Node("fa_silence_removal")
{
  RCLCPP_INFO(this->get_logger(), "Starting FA Silence Removal node");
  loadParameters();
  setupInterfaces();
}

void FaSilenceRemovalNode::loadParameters()
{
  this->declare_parameter("input_topic", config_.input_topic);
  this->declare_parameter("output_topic", config_.output_topic);
  this->declare_parameter<double>("threshold.rms", config_.threshold_rms);
  this->declare_parameter<double>("hangover_ms", config_.hangover_ms);
  this->declare_parameter<int>("expected.sample_rate", config_.expected_sample_rate);
  this->declare_parameter<int>("expected.channels", config_.expected_channels);
  this->declare_parameter("expected.encoding", config_.expected_encoding);
  this->declare_parameter<int>("expected.bit_depth", config_.expected_bit_depth);
  this->declare_parameter("expected.layout", config_.expected_layout);
  this->declare_parameter<int>("qos.depth", config_.qos_depth);
  this->declare_parameter<bool>("qos.reliable", config_.qos_reliable);
  this->declare_parameter<int>(
    "diagnostics.publish_period_ms",
    config_.diagnostics_publish_period_ms);

  config_.input_topic = this->get_parameter("input_topic").as_string();
  config_.output_topic = this->get_parameter("output_topic").as_string();
  config_.threshold_rms = this->get_parameter("threshold.rms").as_double();
  config_.hangover_ms = this->get_parameter("hangover_ms").as_double();
  config_.expected_sample_rate = this->get_parameter("expected.sample_rate").as_int();
  config_.expected_channels = this->get_parameter("expected.channels").as_int();
  config_.expected_encoding = this->get_parameter("expected.encoding").as_string();
  config_.expected_bit_depth = this->get_parameter("expected.bit_depth").as_int();
  config_.expected_layout = this->get_parameter("expected.layout").as_string();
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
  if (!isFinite(config_.threshold_rms) ||
      config_.threshold_rms < 0.0 ||
      config_.threshold_rms > 1.0)
  {
    throw std::runtime_error("threshold.rms must be finite and in [0.0, 1.0]");
  }
  if (!isFinite(config_.hangover_ms) || config_.hangover_ms < 0.0) {
    throw std::runtime_error("hangover_ms must be finite and >= 0");
  }
  if (config_.expected_sample_rate <= 0) {
    throw std::runtime_error("expected.sample_rate must be > 0");
  }
  if (config_.expected_channels <= 0) {
    throw std::runtime_error("expected.channels must be > 0");
  }
  if (config_.expected_encoding != kEncodingFloat32) {
    throw std::runtime_error("fa_silence_removal requires expected.encoding=FLOAT32LE");
  }
  if (config_.expected_bit_depth != 32) {
    throw std::runtime_error("fa_silence_removal requires expected.bit_depth=32");
  }
  if (config_.expected_layout != kInterleavedLayout) {
    throw std::runtime_error("fa_silence_removal requires expected.layout=interleaved");
  }
  if (config_.qos_depth <= 0) {
    throw std::runtime_error("qos.depth must be > 0");
  }
  if (config_.diagnostics_publish_period_ms <= 0) {
    throw std::runtime_error("diagnostics.publish_period_ms must be > 0");
  }

  const double raw_hangover_samples =
    config_.hangover_ms * static_cast<double>(config_.expected_sample_rate) / kMillisecondsPerSecond;
  if (!isFinite(raw_hangover_samples) ||
      raw_hangover_samples > static_cast<double>(std::numeric_limits<size_t>::max()))
  {
    throw std::runtime_error("hangover_ms converts to an unsupported sample count");
  }
  hangover_samples_ = static_cast<size_t>(std::ceil(raw_hangover_samples));

  RCLCPP_INFO(
    this->get_logger(),
    "Silence removal config: input=%s output=%s threshold_rms=%f hangover=%.3fms "
    "hangover_samples=%zu expected=%dHz/%d/%s/%d/%s qos_depth=%d reliable=%s diag=%dms",
    config_.input_topic.c_str(),
    config_.output_topic.c_str(),
    config_.threshold_rms,
    config_.hangover_ms,
    hangover_samples_,
    config_.expected_sample_rate,
    config_.expected_channels,
    config_.expected_encoding.c_str(),
    config_.expected_bit_depth,
    config_.expected_layout.c_str(),
    config_.qos_depth,
    config_.qos_reliable ? "true" : "false",
    config_.diagnostics_publish_period_ms);
}

void FaSilenceRemovalNode::setupInterfaces()
{
  rclcpp::QoS qos(std::max<int>(1, config_.qos_depth));
  if (config_.qos_reliable) {
    qos.reliable();
  } else {
    qos.best_effort();
  }

  audio_pub_ = this->create_publisher<fa_interfaces::msg::AudioFrame>(config_.output_topic, qos);
  audio_sub_ = this->create_subscription<fa_interfaces::msg::AudioFrame>(
    config_.input_topic,
    qos,
    std::bind(&FaSilenceRemovalNode::handleFrame, this, std::placeholders::_1));

  diag_pub_ = this->create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
    "diagnostics",
    rclcpp::SystemDefaultsQoS());
  diag_timer_ = this->create_wall_timer(
    std::chrono::milliseconds(config_.diagnostics_publish_period_ms),
    std::bind(&FaSilenceRemovalNode::publishDiagnostics, this));
}

void FaSilenceRemovalNode::handleFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg)
{
  messages_in_.fetch_add(1);

  if (!msg) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Received null AudioFrame pointer");
    invalid_frames_dropped_.fetch_add(1);
    messages_dropped_.fetch_add(1);
    return;
  }
  if (!validateFrame(*msg)) {
    invalid_frames_dropped_.fetch_add(1);
    messages_dropped_.fetch_add(1);
    return;
  }

  double rms = 0.0;
  if (!computeRms(*msg, rms)) {
    invalid_frames_dropped_.fetch_add(1);
    messages_dropped_.fetch_add(1);
    return;
  }
  last_rms_ = rms;

  if (rms >= config_.threshold_rms) {
    active_frames_.fetch_add(1);
    hangover_samples_remaining_ = hangover_samples_;
    publishAcceptedFrame(*msg);
    return;
  }

  if (hangover_samples_remaining_ > 0) {
    hangover_frames_.fetch_add(1);
    publishAcceptedFrame(*msg);
    consumeHangoverSamples(frameCount(*msg));
    return;
  }

  silent_frames_dropped_.fetch_add(1);
  messages_dropped_.fetch_add(1);
}

bool FaSilenceRemovalNode::validateFrame(const fa_interfaces::msg::AudioFrame & msg)
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
  if (msg.layout != config_.expected_layout) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "AudioFrame layout mismatch: %s != %s",
      msg.layout.c_str(),
      config_.expected_layout.c_str());
    return false;
  }
  if (msg.encoding != config_.expected_encoding ||
      msg.bit_depth != static_cast<uint32_t>(config_.expected_bit_depth))
  {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "AudioFrame encoding mismatch: %s/%u != %s/%d",
      msg.encoding.c_str(),
      msg.bit_depth,
      config_.expected_encoding.c_str(),
      config_.expected_bit_depth);
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
  if (msg.data.empty() || (msg.data.size() % bytesPerFrame()) != 0) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "AudioFrame data size is invalid for FLOAT32LE interleaved samples");
    return false;
  }
  return true;
}

bool FaSilenceRemovalNode::computeRms(
  const fa_interfaces::msg::AudioFrame & msg,
  double & rms)
{
  const size_t sample_count = msg.data.size() / sizeof(float);
  double sum_squares = 0.0;
  for (size_t sample_index = 0; sample_index < sample_count; ++sample_index) {
    const float sample = readFloatSample(msg.data, sample_index);
    if (!std::isfinite(sample) ||
        sample < kMinNormalizedSample ||
        sample > kMaxNormalizedSample)
    {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 3000,
        "Dropping frame because input sample is outside normalized FLOAT32LE range");
      return false;
    }

    const double sample_value = static_cast<double>(sample);
    sum_squares += sample_value * sample_value;
  }

  rms = std::sqrt(sum_squares / static_cast<double>(sample_count));
  if (!isFinite(rms) || rms < 0.0 || rms > 1.0) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping frame because computed RMS is outside normalized range");
    return false;
  }
  return true;
}

void FaSilenceRemovalNode::publishAcceptedFrame(const fa_interfaces::msg::AudioFrame & msg)
{
  fa_interfaces::msg::AudioFrame out = msg;
  out.stream_id = config_.output_topic;
  audio_pub_->publish(out);
  messages_out_.fetch_add(1);
}

void FaSilenceRemovalNode::consumeHangoverSamples(size_t frame_count)
{
  if (frame_count >= hangover_samples_remaining_) {
    hangover_samples_remaining_ = 0;
    return;
  }
  hangover_samples_remaining_ -= frame_count;
}

size_t FaSilenceRemovalNode::bytesPerFrame() const
{
  return static_cast<size_t>(config_.expected_channels) * sizeof(float);
}

size_t FaSilenceRemovalNode::frameCount(const fa_interfaces::msg::AudioFrame & msg) const
{
  return msg.data.size() / bytesPerFrame();
}

void FaSilenceRemovalNode::publishDiagnostics()
{
  diagnostic_msgs::msg::DiagnosticArray array_msg;
  array_msg.header.stamp = this->now();

  diagnostic_msgs::msg::DiagnosticStatus status;
  status.name = "fa_silence_removal";
  status.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
  status.message = "running";
  status.values.reserve(16);
  pushKeyValue(status, "threshold_rms", std::to_string(config_.threshold_rms));
  pushKeyValue(status, "hangover_ms", std::to_string(config_.hangover_ms));
  pushKeyValue(status, "hangover_samples", std::to_string(hangover_samples_));
  pushKeyValue(status, "hangover_samples_remaining", std::to_string(hangover_samples_remaining_));
  pushKeyValue(status, "last_rms", std::to_string(last_rms_));
  pushKeyValue(status, "input_topic", config_.input_topic);
  pushKeyValue(status, "output_topic", config_.output_topic);
  pushKeyValue(status, "expected_sample_rate", std::to_string(config_.expected_sample_rate));
  pushKeyValue(status, "expected_channels", std::to_string(config_.expected_channels));
  pushKeyValue(status, "messages_in", std::to_string(messages_in_.load()));
  pushKeyValue(status, "messages_out", std::to_string(messages_out_.load()));
  pushKeyValue(status, "messages_dropped", std::to_string(messages_dropped_.load()));
  pushKeyValue(status, "invalid_frames_dropped", std::to_string(invalid_frames_dropped_.load()));
  pushKeyValue(status, "silent_frames_dropped", std::to_string(silent_frames_dropped_.load()));
  pushKeyValue(status, "hangover_frames", std::to_string(hangover_frames_.load()));
  pushKeyValue(status, "active_frames", std::to_string(active_frames_.load()));

  array_msg.status.push_back(status);
  diag_pub_->publish(array_msg);
}

}  // namespace fa_silence_removal

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<fa_silence_removal::FaSilenceRemovalNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return EXIT_SUCCESS;
  } catch (const std::exception & e) {
    RCLCPP_FATAL(rclcpp::get_logger("fa_silence_removal"), "Exception: %s", e.what());
    rclcpp::shutdown();
    return EXIT_FAILURE;
  }
}

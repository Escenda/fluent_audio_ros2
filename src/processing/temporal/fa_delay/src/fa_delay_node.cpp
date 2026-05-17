#include "fa_delay/fa_delay_node.hpp"

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

#include "diagnostic_msgs/msg/diagnostic_status.hpp"
#include "diagnostic_msgs/msg/key_value.hpp"

namespace fa_delay
{

namespace
{
constexpr const char * kEncodingFloat32 = "FLOAT32LE";
constexpr const char * kInterleavedLayout = "interleaved";
constexpr float kSilenceSample = 0.0F;
constexpr float kMinNormalizedSample = -1.0F;
constexpr float kMaxNormalizedSample = 1.0F;

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

void writeFloatSample(std::vector<uint8_t> & data, size_t sample_index, float sample)
{
  std::memcpy(data.data() + (sample_index * sizeof(float)), &sample, sizeof(float));
}
}  // namespace

FaDelayNode::FaDelayNode()
: rclcpp::Node("fa_delay")
{
  RCLCPP_INFO(this->get_logger(), "Starting FA Delay node");
  loadParameters();
  setupInterfaces();
}

void FaDelayNode::loadParameters()
{
  this->declare_parameter("input_topic", config_.input_topic);
  this->declare_parameter("output_topic", config_.output_topic);
  this->declare_parameter<double>("delay.ms", config_.delay_ms);
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
  config_.delay_ms = this->get_parameter("delay.ms").as_double();
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
  if (!std::isfinite(config_.delay_ms) || config_.delay_ms <= 0.0) {
    throw std::runtime_error("delay.ms must be > 0 and finite");
  }
  if (config_.expected_sample_rate <= 0) {
    throw std::runtime_error("expected.sample_rate must be > 0");
  }
  if (config_.expected_channels <= 0) {
    throw std::runtime_error("expected.channels must be > 0");
  }
  if (config_.expected_encoding != kEncodingFloat32) {
    throw std::runtime_error("fa_delay requires expected.encoding=FLOAT32LE");
  }
  if (config_.expected_bit_depth != 32) {
    throw std::runtime_error("fa_delay requires expected.bit_depth=32");
  }
  if (config_.expected_layout != kInterleavedLayout) {
    throw std::runtime_error("fa_delay requires expected.layout=interleaved");
  }
  if (config_.qos_depth <= 0) {
    throw std::runtime_error("qos.depth must be > 0");
  }
  if (config_.diagnostics_publish_period_ms <= 0) {
    throw std::runtime_error("diagnostics.publish_period_ms must be > 0");
  }

  const double raw_delay_samples =
    config_.delay_ms * static_cast<double>(config_.expected_sample_rate) / 1000.0;
  if (!std::isfinite(raw_delay_samples) ||
      raw_delay_samples > static_cast<double>(std::numeric_limits<long long>::max()))
  {
    throw std::runtime_error("delay.ms converts to an unsupported sample count");
  }
  delay_samples_ = static_cast<size_t>(std::llround(raw_delay_samples));
  if (delay_samples_ == 0) {
    throw std::runtime_error("delay.ms must convert to at least 1 sample");
  }

  RCLCPP_INFO(
    this->get_logger(),
    "Delay config: input=%s output=%s delay=%.3fms delay_samples=%zu "
    "expected=%dHz/%d/%s/%d/%s qos_depth=%d reliable=%s diag=%dms",
    config_.input_topic.c_str(),
    config_.output_topic.c_str(),
    config_.delay_ms,
    delay_samples_,
    config_.expected_sample_rate,
    config_.expected_channels,
    config_.expected_encoding.c_str(),
    config_.expected_bit_depth,
    config_.expected_layout.c_str(),
    config_.qos_depth,
    config_.qos_reliable ? "true" : "false",
    config_.diagnostics_publish_period_ms);
}

void FaDelayNode::setupInterfaces()
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
    std::bind(&FaDelayNode::handleFrame, this, std::placeholders::_1));

  diag_pub_ = this->create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
    "diagnostics",
    rclcpp::SystemDefaultsQoS());
  diag_timer_ = this->create_wall_timer(
    std::chrono::milliseconds(config_.diagnostics_publish_period_ms),
    std::bind(&FaDelayNode::publishDiagnostics, this));
}

void FaDelayNode::handleFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg)
{
  messages_in_.fetch_add(1);

  if (!msg) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Received null AudioFrame pointer");
    messages_dropped_.fetch_add(1);
    return;
  }
  if (!validateFrame(*msg)) {
    messages_dropped_.fetch_add(1);
    return;
  }
  if (!validateSamples(*msg)) {
    messages_dropped_.fetch_add(1);
    return;
  }

  ensureDelayState(msg->source_id);

  fa_interfaces::msg::AudioFrame out;
  if (!applyDelay(*msg, out)) {
    messages_dropped_.fetch_add(1);
    return;
  }

  audio_pub_->publish(out);
  messages_out_.fetch_add(1);
}

bool FaDelayNode::validateFrame(const fa_interfaces::msg::AudioFrame & msg)
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

bool FaDelayNode::validateSamples(const fa_interfaces::msg::AudioFrame & msg)
{
  const size_t sample_count = msg.data.size() / sizeof(float);
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
  }
  return true;
}

void FaDelayNode::ensureDelayState(const std::string & source_id)
{
  if (source_id == current_source_id_ &&
      delay_buffers_.size() == static_cast<size_t>(config_.expected_channels))
  {
    return;
  }

  if (!current_source_id_.empty() && source_id != current_source_id_) {
    source_resets_.fetch_add(1);
    RCLCPP_WARN(
      this->get_logger(),
      "AudioFrame source_id changed: %s -> %s; resetting delay buffers",
      current_source_id_.c_str(),
      source_id.c_str());
  }

  resetDelayState(source_id);
}

void FaDelayNode::resetDelayState(const std::string & source_id)
{
  current_source_id_ = source_id;
  delay_buffers_.assign(
    static_cast<size_t>(config_.expected_channels),
    std::deque<float>(delay_samples_, kSilenceSample));
}

bool FaDelayNode::applyDelay(
  const fa_interfaces::msg::AudioFrame & in,
  fa_interfaces::msg::AudioFrame & out)
{
  if (delay_buffers_.size() != static_cast<size_t>(config_.expected_channels)) {
    RCLCPP_WARN(
      this->get_logger(),
      "Dropping frame because delay buffer state does not match expected channel count");
    return false;
  }

  out = in;
  out.stream_id = config_.output_topic;
  out.data.resize(in.data.size());

  const size_t channels = static_cast<size_t>(config_.expected_channels);
  const size_t frame_count = in.data.size() / bytesPerFrame();
  for (size_t frame_index = 0; frame_index < frame_count; ++frame_index) {
    for (size_t channel_index = 0; channel_index < channels; ++channel_index) {
      const size_t sample_index = (frame_index * channels) + channel_index;
      const float input_sample = readFloatSample(in.data, sample_index);
      const float delayed_sample = delay_buffers_[channel_index].front();
      delay_buffers_[channel_index].pop_front();
      delay_buffers_[channel_index].push_back(input_sample);
      writeFloatSample(out.data, sample_index, delayed_sample);
    }
  }

  return true;
}

size_t FaDelayNode::bytesPerFrame() const
{
  return static_cast<size_t>(config_.expected_channels) * sizeof(float);
}

void FaDelayNode::publishDiagnostics()
{
  diagnostic_msgs::msg::DiagnosticArray array_msg;
  array_msg.header.stamp = this->now();

  diagnostic_msgs::msg::DiagnosticStatus status;
  status.name = "fa_delay";
  status.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
  status.message = "running";
  status.values.reserve(11);
  pushKeyValue(status, "delay_ms", std::to_string(config_.delay_ms));
  pushKeyValue(status, "delay_samples", std::to_string(delay_samples_));
  pushKeyValue(status, "current_source_id", current_source_id_);
  pushKeyValue(status, "input_topic", config_.input_topic);
  pushKeyValue(status, "output_topic", config_.output_topic);
  pushKeyValue(status, "messages_in", std::to_string(messages_in_.load()));
  pushKeyValue(status, "messages_out", std::to_string(messages_out_.load()));
  pushKeyValue(status, "messages_dropped", std::to_string(messages_dropped_.load()));
  pushKeyValue(status, "source_resets", std::to_string(source_resets_.load()));
  pushKeyValue(status, "expected_channels", std::to_string(config_.expected_channels));

  array_msg.status.push_back(status);
  diag_pub_->publish(array_msg);
}

}  // namespace fa_delay

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<fa_delay::FaDelayNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return EXIT_SUCCESS;
  } catch (const std::exception & e) {
    RCLCPP_FATAL(rclcpp::get_logger("fa_delay"), "Exception: %s", e.what());
    rclcpp::shutdown();
    return EXIT_FAILURE;
  }
}

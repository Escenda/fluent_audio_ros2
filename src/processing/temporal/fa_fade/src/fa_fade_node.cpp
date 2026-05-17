#include "fa_fade/fa_fade_node.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>

#include "diagnostic_msgs/msg/diagnostic_status.hpp"
#include "diagnostic_msgs/msg/key_value.hpp"

namespace fa_fade
{

namespace
{
constexpr const char * kEncodingFloat32 = "FLOAT32LE";
constexpr const char * kInterleavedLayout = "interleaved";
constexpr const char * kFadeIn = "fade_in";
constexpr const char * kFadeOut = "fade_out";
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
}  // namespace

FaFadeNode::FaFadeNode()
: rclcpp::Node("fa_fade")
{
  RCLCPP_INFO(this->get_logger(), "Starting FA Fade node");
  loadParameters();
  position_frames_ = static_cast<uint64_t>(config_.initial_position_frames);
  setupInterfaces();
}

void FaFadeNode::loadParameters()
{
  this->declare_parameter("input_topic", config_.input_topic);
  this->declare_parameter("output_topic", config_.output_topic);
  this->declare_parameter("fade.mode", config_.mode);
  this->declare_parameter<int>("fade.duration_frames", config_.duration_frames);
  this->declare_parameter<int>("fade.initial_position_frames", config_.initial_position_frames);
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
  config_.mode = this->get_parameter("fade.mode").as_string();
  config_.duration_frames = this->get_parameter("fade.duration_frames").as_int();
  config_.initial_position_frames = this->get_parameter("fade.initial_position_frames").as_int();
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
  if (config_.mode != kFadeIn && config_.mode != kFadeOut) {
    throw std::runtime_error("fade.mode must be one of fade_in, fade_out");
  }
  if (config_.duration_frames <= 0) {
    throw std::runtime_error("fade.duration_frames must be > 0");
  }
  if (config_.initial_position_frames < 0) {
    throw std::runtime_error("fade.initial_position_frames must be >= 0");
  }
  if (config_.expected_sample_rate <= 0) {
    throw std::runtime_error("expected.sample_rate must be > 0");
  }
  if (config_.expected_channels <= 0) {
    throw std::runtime_error("expected.channels must be > 0");
  }
  if (config_.expected_encoding != kEncodingFloat32) {
    throw std::runtime_error("fa_fade requires expected.encoding=FLOAT32LE");
  }
  if (config_.expected_bit_depth != 32) {
    throw std::runtime_error("fa_fade requires expected.bit_depth=32");
  }
  if (config_.expected_layout != kInterleavedLayout) {
    throw std::runtime_error("fa_fade requires expected.layout=interleaved");
  }
  if (config_.qos_depth <= 0) {
    throw std::runtime_error("qos.depth must be > 0");
  }
  if (config_.diagnostics_publish_period_ms <= 0) {
    throw std::runtime_error("diagnostics.publish_period_ms must be > 0");
  }

  RCLCPP_INFO(
    this->get_logger(),
    "Fade config: input=%s output=%s mode=%s duration=%d initial_position=%d "
    "expected=%dHz/%d/%s/%d/%s qos_depth=%d reliable=%s diag=%dms",
    config_.input_topic.c_str(),
    config_.output_topic.c_str(),
    config_.mode.c_str(),
    config_.duration_frames,
    config_.initial_position_frames,
    config_.expected_sample_rate,
    config_.expected_channels,
    config_.expected_encoding.c_str(),
    config_.expected_bit_depth,
    config_.expected_layout.c_str(),
    config_.qos_depth,
    config_.qos_reliable ? "true" : "false",
    config_.diagnostics_publish_period_ms);
}

void FaFadeNode::setupInterfaces()
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
    std::bind(&FaFadeNode::handleFrame, this, std::placeholders::_1));

  diag_pub_ = this->create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
    "diagnostics",
    rclcpp::SystemDefaultsQoS());
  diag_timer_ = this->create_wall_timer(
    std::chrono::milliseconds(config_.diagnostics_publish_period_ms),
    std::bind(&FaFadeNode::publishDiagnostics, this));
}

void FaFadeNode::handleFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg)
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
  if (!applyFade(*msg, out)) {
    frames_dropped_.fetch_add(1);
    return;
  }

  audio_pub_->publish(out);
  frames_out_.fetch_add(1);
}

bool FaFadeNode::validateFrame(const fa_interfaces::msg::AudioFrame & msg)
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

bool FaFadeNode::applyFade(
  const fa_interfaces::msg::AudioFrame & in,
  fa_interfaces::msg::AudioFrame & out)
{
  out = in;
  out.stream_id = config_.output_topic;
  out.data.resize(in.data.size());

  const size_t sample_count = in.data.size() / sizeof(float);
  const size_t frame_count = in.data.size() / bytesPerFrame();
  uint64_t sample_position_frames = position_frames_;

  for (size_t i = 0; i < sample_count; ++i) {
    float sample = 0.0F;
    std::memcpy(&sample, in.data.data() + (i * sizeof(float)), sizeof(float));
    if (!std::isfinite(sample) || sample < kMinNormalizedSample || sample > kMaxNormalizedSample) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 3000,
        "Dropping frame because input sample is outside normalized FLOAT32LE range");
      return false;
    }

    sample_position_frames = position_frames_ + static_cast<uint64_t>(i / static_cast<size_t>(config_.expected_channels));
    const double gain = gainAtPosition(sample_position_frames);
    const double faded = static_cast<double>(sample) * gain;
    if (!std::isfinite(faded) || faded < kMinNormalizedSample || faded > kMaxNormalizedSample) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 3000,
        "Dropping frame because fade output would exceed normalized FLOAT32LE range");
      return false;
    }

    const float out_sample = static_cast<float>(faded);
    if (!std::isfinite(out_sample)) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 3000,
        "Dropping frame because fade output is not finite FLOAT32LE");
      return false;
    }
    std::memcpy(out.data.data() + (i * sizeof(float)), &out_sample, sizeof(float));
  }

  position_frames_ += static_cast<uint64_t>(frame_count);
  return true;
}

double FaFadeNode::gainAtPosition(uint64_t position_frames) const
{
  const double progress =
    static_cast<double>(position_frames) / static_cast<double>(config_.duration_frames);
  if (config_.mode == kFadeIn) {
    return std::min(1.0, progress);
  }
  return std::max(0.0, 1.0 - progress);
}

size_t FaFadeNode::bytesPerFrame() const
{
  return static_cast<size_t>(config_.expected_channels) * sizeof(float);
}

void FaFadeNode::publishDiagnostics()
{
  diagnostic_msgs::msg::DiagnosticArray array_msg;
  array_msg.header.stamp = this->now();

  diagnostic_msgs::msg::DiagnosticStatus status;
  status.name = "fa_fade";
  status.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
  status.message = "running";
  status.values.reserve(9);
  pushKeyValue(status, "mode", config_.mode);
  pushKeyValue(status, "duration_frames", std::to_string(config_.duration_frames));
  pushKeyValue(status, "current_position_frames", std::to_string(position_frames_));
  pushKeyValue(status, "input_topic", config_.input_topic);
  pushKeyValue(status, "output_topic", config_.output_topic);
  pushKeyValue(status, "frames_in", std::to_string(frames_in_.load()));
  pushKeyValue(status, "frames_out", std::to_string(frames_out_.load()));
  pushKeyValue(status, "frames_dropped", std::to_string(frames_dropped_.load()));
  pushKeyValue(status, "expected_channels", std::to_string(config_.expected_channels));

  array_msg.status.push_back(status);
  diag_pub_->publish(array_msg);
}

}  // namespace fa_fade

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<fa_fade::FaFadeNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return EXIT_SUCCESS;
  } catch (const std::exception & e) {
    RCLCPP_FATAL(rclcpp::get_logger("fa_fade"), "Exception: %s", e.what());
    rclcpp::shutdown();
    return EXIT_FAILURE;
  }
}

#include "fa_interleave/fa_interleave_node.hpp"

#include <chrono>
#include <cstdlib>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "diagnostic_msgs/msg/diagnostic_status.hpp"
#include "diagnostic_msgs/msg/key_value.hpp"

namespace fa_interleave
{

namespace
{
constexpr const char * kEncodingPcm16 = "PCM16LE";
constexpr const char * kEncodingPcm32 = "PCM32LE";
constexpr const char * kEncodingFloat32 = "FLOAT32LE";
constexpr const char * kInterleavedLayout = "interleaved";
constexpr const char * kPlanarLayout = "planar";

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

FaInterleaveNode::FaInterleaveNode()
: rclcpp::Node("fa_interleave")
{
  RCLCPP_INFO(this->get_logger(), "Starting FA Interleave node");
  loadParameters();
  setupInterfaces();
}

void FaInterleaveNode::loadParameters()
{
  this->declare_parameter("input_topic", config_.input_topic);
  this->declare_parameter("output_topic", config_.output_topic);
  this->declare_parameter("input.layout", config_.input_layout);
  this->declare_parameter("output.layout", config_.output_layout);
  this->declare_parameter<int>("expected.sample_rate", config_.expected_sample_rate);
  this->declare_parameter<int>("expected.channels", config_.expected_channels);
  this->declare_parameter("expected.encoding", config_.expected_encoding);
  this->declare_parameter<int>("expected.bit_depth", config_.expected_bit_depth);
  this->declare_parameter<int>("qos.depth", config_.qos_depth);
  this->declare_parameter<bool>("qos.reliable", config_.qos_reliable);
  this->declare_parameter<int>(
    "diagnostics.publish_period_ms",
    config_.diagnostics_publish_period_ms);

  config_.input_topic = this->get_parameter("input_topic").as_string();
  config_.output_topic = this->get_parameter("output_topic").as_string();
  config_.input_layout = this->get_parameter("input.layout").as_string();
  config_.output_layout = this->get_parameter("output.layout").as_string();
  config_.expected_sample_rate = this->get_parameter("expected.sample_rate").as_int();
  config_.expected_channels = this->get_parameter("expected.channels").as_int();
  config_.expected_encoding = this->get_parameter("expected.encoding").as_string();
  config_.expected_bit_depth = this->get_parameter("expected.bit_depth").as_int();
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
  if (!isSupportedLayoutConversion(config_.input_layout, config_.output_layout)) {
    throw std::runtime_error("fa_interleave requires interleaved->planar or planar->interleaved layout conversion");
  }
  if (config_.expected_sample_rate <= 0) {
    throw std::runtime_error("expected.sample_rate must be > 0");
  }
  if (config_.expected_channels <= 0) {
    throw std::runtime_error("expected.channels must be > 0");
  }
  if (!isSupportedFormat(config_.expected_encoding, config_.expected_bit_depth)) {
    throw std::runtime_error("fa_interleave supports only FLOAT32LE/32, PCM16LE/16, or PCM32LE/32");
  }
  if (config_.qos_depth <= 0) {
    throw std::runtime_error("qos.depth must be > 0");
  }
  if (config_.diagnostics_publish_period_ms <= 0) {
    throw std::runtime_error("diagnostics.publish_period_ms must be > 0");
  }

  RCLCPP_INFO(
    this->get_logger(),
    "Interleave config: input=%s output=%s layout=%s->%s expected=%dHz/%d/%s/%d qos_depth=%d "
    "reliable=%s diag=%dms",
    config_.input_topic.c_str(),
    config_.output_topic.c_str(),
    config_.input_layout.c_str(),
    config_.output_layout.c_str(),
    config_.expected_sample_rate,
    config_.expected_channels,
    config_.expected_encoding.c_str(),
    config_.expected_bit_depth,
    config_.qos_depth,
    config_.qos_reliable ? "true" : "false",
    config_.diagnostics_publish_period_ms);
}

void FaInterleaveNode::setupInterfaces()
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
    std::bind(&FaInterleaveNode::handleFrame, this, std::placeholders::_1));

  diag_pub_ = this->create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
    "diagnostics",
    rclcpp::SystemDefaultsQoS());
  diag_timer_ = this->create_wall_timer(
    std::chrono::milliseconds(config_.diagnostics_publish_period_ms),
    std::bind(&FaInterleaveNode::publishDiagnostics, this));
}

void FaInterleaveNode::handleFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg)
{
  frames_in_.fetch_add(1);
  if (!msg) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping null AudioFrame pointer");
    frames_dropped_.fetch_add(1);
    return;
  }

  if (!validateFrame(*msg)) {
    frames_dropped_.fetch_add(1);
    return;
  }

  fa_interfaces::msg::AudioFrame out;
  if (!convertFrame(*msg, out)) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping AudioFrame because layout reorder failed");
    frames_dropped_.fetch_add(1);
    return;
  }

  audio_pub_->publish(out);
  frames_out_.fetch_add(1);
}

bool FaInterleaveNode::validateFrame(const fa_interfaces::msg::AudioFrame & msg)
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
  if (msg.layout != config_.input_layout) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "AudioFrame layout mismatch: %s != %s",
      msg.layout.c_str(),
      config_.input_layout.c_str());
    return false;
  }
  if (msg.encoding != config_.expected_encoding ||
      msg.bit_depth != static_cast<uint32_t>(config_.expected_bit_depth))
  {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "AudioFrame format mismatch: %s/%u != %s/%d",
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
      "AudioFrame rate/channel mismatch: frame=%uHz/%u config=%dHz/%d",
      msg.sample_rate,
      msg.channels,
      config_.expected_sample_rate,
      config_.expected_channels);
    return false;
  }

  const size_t bytes_per_sample = bytesPerSample(config_.expected_encoding, config_.expected_bit_depth);
  const size_t bytes_per_frame = static_cast<size_t>(config_.expected_channels) * bytes_per_sample;
  if (msg.data.empty() || (msg.data.size() % bytes_per_frame) != 0) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "AudioFrame data size is invalid for configured layout conversion");
    return false;
  }
  return true;
}

bool FaInterleaveNode::convertFrame(
  const fa_interfaces::msg::AudioFrame & in,
  fa_interfaces::msg::AudioFrame & out)
{
  const size_t bytes_per_sample = bytesPerSample(config_.expected_encoding, config_.expected_bit_depth);
  const size_t channel_count = static_cast<size_t>(config_.expected_channels);
  const size_t frame_count = in.data.size() / (channel_count * bytes_per_sample);

  std::vector<uint8_t> output_data;
  if (config_.input_layout == kInterleavedLayout && config_.output_layout == kPlanarLayout) {
    output_data = reorderInterleavedToPlanar(in.data, frame_count, channel_count, bytes_per_sample);
  } else if (config_.input_layout == kPlanarLayout && config_.output_layout == kInterleavedLayout) {
    output_data = reorderPlanarToInterleaved(in.data, frame_count, channel_count, bytes_per_sample);
  } else {
    return false;
  }

  if (output_data.size() != in.data.size()) {
    return false;
  }

  out = in;
  out.stream_id = config_.output_topic;
  out.layout = config_.output_layout;
  out.data = output_data;
  return true;
}

bool FaInterleaveNode::isSupportedLayout(const std::string & layout)
{
  return layout == kInterleavedLayout || layout == kPlanarLayout;
}

bool FaInterleaveNode::isSupportedLayoutConversion(
  const std::string & input_layout,
  const std::string & output_layout)
{
  return isSupportedLayout(input_layout) && isSupportedLayout(output_layout) && input_layout != output_layout;
}

bool FaInterleaveNode::isSupportedFormat(const std::string & encoding, int bit_depth)
{
  return (encoding == kEncodingFloat32 && bit_depth == 32) ||
         (encoding == kEncodingPcm16 && bit_depth == 16) ||
         (encoding == kEncodingPcm32 && bit_depth == 32);
}

size_t FaInterleaveNode::bytesPerSample(const std::string & encoding, int bit_depth)
{
  if (encoding == kEncodingPcm16 && bit_depth == 16) {
    return sizeof(uint16_t);
  }
  if ((encoding == kEncodingPcm32 && bit_depth == 32) ||
      (encoding == kEncodingFloat32 && bit_depth == 32))
  {
    return sizeof(uint32_t);
  }
  throw std::runtime_error("unsupported fa_interleave encoding/bit_depth");
}

std::vector<uint8_t> FaInterleaveNode::reorderInterleavedToPlanar(
  const std::vector<uint8_t> & input_bytes,
  size_t frame_count,
  size_t channel_count,
  size_t bytes_per_sample)
{
  std::vector<uint8_t> output_bytes;
  output_bytes.reserve(input_bytes.size());

  for (size_t channel_index = 0; channel_index < channel_count; ++channel_index) {
    for (size_t frame_index = 0; frame_index < frame_count; ++frame_index) {
      const size_t input_sample_index = frame_index * channel_count + channel_index;
      appendSampleBytes(input_bytes, input_sample_index, bytes_per_sample, output_bytes);
    }
  }
  return output_bytes;
}

std::vector<uint8_t> FaInterleaveNode::reorderPlanarToInterleaved(
  const std::vector<uint8_t> & input_bytes,
  size_t frame_count,
  size_t channel_count,
  size_t bytes_per_sample)
{
  std::vector<uint8_t> output_bytes;
  output_bytes.reserve(input_bytes.size());

  for (size_t frame_index = 0; frame_index < frame_count; ++frame_index) {
    for (size_t channel_index = 0; channel_index < channel_count; ++channel_index) {
      const size_t input_sample_index = channel_index * frame_count + frame_index;
      appendSampleBytes(input_bytes, input_sample_index, bytes_per_sample, output_bytes);
    }
  }
  return output_bytes;
}

void FaInterleaveNode::appendSampleBytes(
  const std::vector<uint8_t> & input_bytes,
  size_t sample_index,
  size_t bytes_per_sample,
  std::vector<uint8_t> & output_bytes)
{
  const size_t byte_offset = sample_index * bytes_per_sample;
  for (size_t byte_index = 0; byte_index < bytes_per_sample; ++byte_index) {
    output_bytes.push_back(input_bytes.at(byte_offset + byte_index));
  }
}

void FaInterleaveNode::publishDiagnostics()
{
  diagnostic_msgs::msg::DiagnosticArray array_msg;
  array_msg.header.stamp = this->now();

  diagnostic_msgs::msg::DiagnosticStatus status;
  status.name = "fa_interleave";
  status.hardware_id = "dsp";
  status.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
  status.message = "running";

  status.values.reserve(13);
  pushKeyValue(status, "input_topic", config_.input_topic);
  pushKeyValue(status, "output_topic", config_.output_topic);
  pushKeyValue(status, "input.layout", config_.input_layout);
  pushKeyValue(status, "output.layout", config_.output_layout);
  pushKeyValue(status, "expected.sample_rate", std::to_string(config_.expected_sample_rate));
  pushKeyValue(status, "expected.channels", std::to_string(config_.expected_channels));
  pushKeyValue(status, "expected.encoding", config_.expected_encoding);
  pushKeyValue(status, "expected.bit_depth", std::to_string(config_.expected_bit_depth));
  pushKeyValue(status, "frames.in", std::to_string(frames_in_.load()));
  pushKeyValue(status, "frames.out", std::to_string(frames_out_.load()));
  pushKeyValue(status, "frames.drop", std::to_string(frames_dropped_.load()));

  array_msg.status.push_back(status);
  diag_pub_->publish(array_msg);
}

}  // namespace fa_interleave

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<fa_interleave::FaInterleaveNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return EXIT_SUCCESS;
  } catch (const std::exception & e) {
    RCLCPP_FATAL(rclcpp::get_logger("fa_interleave"), "Exception: %s", e.what());
    rclcpp::shutdown();
    return EXIT_FAILURE;
  }
}

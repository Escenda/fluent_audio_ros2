#include "fa_bit_depth/fa_bit_depth_node.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "diagnostic_msgs/msg/diagnostic_status.hpp"
#include "diagnostic_msgs/msg/key_value.hpp"

namespace fa_bit_depth
{

namespace
{
constexpr const char * kEncodingPcm16 = "PCM16LE";
constexpr const char * kEncodingPcm32 = "PCM32LE";
constexpr const char * kInterleavedLayout = "interleaved";

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

FaBitDepthNode::FaBitDepthNode()
: rclcpp::Node("fa_bit_depth")
{
  RCLCPP_INFO(this->get_logger(), "Starting FA Bit Depth node");
  loadParameters();
  setupInterfaces();
}

void FaBitDepthNode::loadParameters()
{
  this->declare_parameter("input_topic", config_.input_topic);
  this->declare_parameter("output_topic", config_.output_topic);
  this->declare_parameter("input.encoding", config_.input_encoding);
  this->declare_parameter<int>("input.bit_depth", config_.input_bit_depth);
  this->declare_parameter("output.encoding", config_.output_encoding);
  this->declare_parameter<int>("output.bit_depth", config_.output_bit_depth);
  this->declare_parameter<int>("expected.sample_rate", config_.expected_sample_rate);
  this->declare_parameter<int>("expected.channels", config_.expected_channels);
  this->declare_parameter("expected.layout", config_.expected_layout);
  this->declare_parameter<int>("qos.depth", config_.qos_depth);
  this->declare_parameter<bool>("qos.reliable", config_.qos_reliable);
  this->declare_parameter<int>(
    "diagnostics.publish_period_ms",
    config_.diagnostics_publish_period_ms);

  config_.input_topic = this->get_parameter("input_topic").as_string();
  config_.output_topic = this->get_parameter("output_topic").as_string();
  config_.input_encoding = this->get_parameter("input.encoding").as_string();
  config_.input_bit_depth = this->get_parameter("input.bit_depth").as_int();
  config_.output_encoding = this->get_parameter("output.encoding").as_string();
  config_.output_bit_depth = this->get_parameter("output.bit_depth").as_int();
  config_.expected_sample_rate = this->get_parameter("expected.sample_rate").as_int();
  config_.expected_channels = this->get_parameter("expected.channels").as_int();
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
  if (!isSupportedConversion(
      config_.input_encoding,
      config_.input_bit_depth,
      config_.output_encoding,
      config_.output_bit_depth))
  {
    throw std::runtime_error(
            "fa_bit_depth supports only PCM16LE/16 <-> PCM32LE/32 integer bit-depth conversion");
  }
  if (config_.expected_sample_rate <= 0) {
    throw std::runtime_error("expected.sample_rate must be > 0");
  }
  if (config_.expected_channels <= 0) {
    throw std::runtime_error("expected.channels must be > 0");
  }
  if (config_.expected_layout != kInterleavedLayout) {
    throw std::runtime_error("fa_bit_depth requires expected.layout=interleaved");
  }
  if (config_.qos_depth <= 0) {
    throw std::runtime_error("qos.depth must be > 0");
  }
  if (config_.diagnostics_publish_period_ms <= 0) {
    throw std::runtime_error("diagnostics.publish_period_ms must be > 0");
  }

  RCLCPP_INFO(
    this->get_logger(),
    "Bit depth config: input=%s output=%s conversion=%s/%d -> %s/%d expected=%dHz/%d/%s "
    "qos_depth=%d reliable=%s diag=%dms",
    config_.input_topic.c_str(),
    config_.output_topic.c_str(),
    config_.input_encoding.c_str(),
    config_.input_bit_depth,
    config_.output_encoding.c_str(),
    config_.output_bit_depth,
    config_.expected_sample_rate,
    config_.expected_channels,
    config_.expected_layout.c_str(),
    config_.qos_depth,
    config_.qos_reliable ? "true" : "false",
    config_.diagnostics_publish_period_ms);
}

void FaBitDepthNode::setupInterfaces()
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
    std::bind(&FaBitDepthNode::handleFrame, this, std::placeholders::_1));

  diag_pub_ = this->create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
    "diagnostics",
    rclcpp::SystemDefaultsQoS());
  diag_timer_ = this->create_wall_timer(
    std::chrono::milliseconds(config_.diagnostics_publish_period_ms),
    std::bind(&FaBitDepthNode::publishDiagnostics, this));
}

void FaBitDepthNode::handleFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg)
{
  frames_in_.fetch_add(1);
  if (!msg) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping null AudioFrame");
    frames_dropped_.fetch_add(1);
    return;
  }

  if (!validateFrame(*msg)) {
    frames_dropped_.fetch_add(1);
    return;
  }

  fa_interfaces::msg::AudioFrame out;
  if (!convertFrame(*msg, out)) {
    frames_dropped_.fetch_add(1);
    return;
  }

  audio_pub_->publish(out);
  frames_out_.fetch_add(1);
}

bool FaBitDepthNode::validateFrame(const fa_interfaces::msg::AudioFrame & msg)
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
  if (msg.encoding != config_.input_encoding ||
      msg.bit_depth != static_cast<uint32_t>(config_.input_bit_depth))
  {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "AudioFrame input bit-depth format mismatch: %s/%u != %s/%d",
      msg.encoding.c_str(),
      msg.bit_depth,
      config_.input_encoding.c_str(),
      config_.input_bit_depth);
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

  const size_t bytes_per_sample = bytesPerSample(config_.input_bit_depth);
  const size_t bytes_per_frame = static_cast<size_t>(config_.expected_channels) * bytes_per_sample;
  if (msg.data.empty() || (msg.data.size() % bytes_per_frame) != 0) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "AudioFrame data size is invalid for configured interleaved PCM bit depth");
    return false;
  }
  return true;
}

bool FaBitDepthNode::convertFrame(
  const fa_interfaces::msg::AudioFrame & in,
  fa_interfaces::msg::AudioFrame & out)
{
  std::vector<uint8_t> output_data;
  if (config_.input_encoding == kEncodingPcm16 && config_.input_bit_depth == 16 &&
      config_.output_encoding == kEncodingPcm32 && config_.output_bit_depth == 32)
  {
    output_data = convertPcm16ToPcm32(in.data);
  } else if (
    config_.input_encoding == kEncodingPcm32 && config_.input_bit_depth == 32 &&
    config_.output_encoding == kEncodingPcm16 && config_.output_bit_depth == 16)
  {
    output_data = convertPcm32ToPcm16(in.data);
  } else {
    RCLCPP_ERROR_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping frame because configured PCM bit-depth conversion is unsupported");
    return false;
  }

  if (output_data.empty()) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping frame because PCM bit-depth conversion produced no output");
    return false;
  }

  out = in;
  out.stream_id = config_.output_topic;
  out.encoding = config_.output_encoding;
  out.bit_depth = static_cast<uint32_t>(config_.output_bit_depth);
  out.sample_rate = in.sample_rate;
  out.channels = in.channels;
  out.layout = in.layout;
  out.data = output_data;
  return true;
}

bool FaBitDepthNode::isSupportedConversion(
  const std::string & input_encoding,
  int input_bit_depth,
  const std::string & output_encoding,
  int output_bit_depth)
{
  return (input_encoding == kEncodingPcm16 && input_bit_depth == 16 &&
         output_encoding == kEncodingPcm32 && output_bit_depth == 32) ||
         (input_encoding == kEncodingPcm32 && input_bit_depth == 32 &&
         output_encoding == kEncodingPcm16 && output_bit_depth == 16);
}

size_t FaBitDepthNode::bytesPerSample(int bit_depth)
{
  return static_cast<size_t>(bit_depth / 8);
}

std::vector<uint8_t> FaBitDepthNode::convertPcm16ToPcm32(const std::vector<uint8_t> & input_bytes)
{
  if (input_bytes.empty() || (input_bytes.size() % sizeof(uint16_t)) != 0) {
    return {};
  }

  std::vector<uint8_t> out_bytes;
  out_bytes.reserve((input_bytes.size() / sizeof(uint16_t)) * sizeof(uint32_t));
  for (size_t i = 0; i < input_bytes.size(); i += sizeof(uint16_t)) {
    const uint16_t raw =
      static_cast<uint16_t>(input_bytes.at(i)) |
      (static_cast<uint16_t>(input_bytes.at(i + 1)) << 8U);
    const uint32_t aligned_sample = static_cast<uint32_t>(raw) << 16U;
    appendPcm32Le(aligned_sample, out_bytes);
  }
  return out_bytes;
}

std::vector<uint8_t> FaBitDepthNode::convertPcm32ToPcm16(const std::vector<uint8_t> & input_bytes)
{
  if (input_bytes.empty() || (input_bytes.size() % sizeof(uint32_t)) != 0) {
    return {};
  }

  std::vector<uint8_t> out_bytes;
  out_bytes.reserve((input_bytes.size() / sizeof(uint32_t)) * sizeof(uint16_t));
  for (size_t i = 0; i < input_bytes.size(); i += sizeof(uint32_t)) {
    const uint32_t raw =
      static_cast<uint32_t>(input_bytes.at(i)) |
      (static_cast<uint32_t>(input_bytes.at(i + 1)) << 8U) |
      (static_cast<uint32_t>(input_bytes.at(i + 2)) << 16U) |
      (static_cast<uint32_t>(input_bytes.at(i + 3)) << 24U);
    const uint16_t high_word = static_cast<uint16_t>((raw >> 16U) & 0xFFFFU);
    appendPcm16Le(high_word, out_bytes);
  }
  return out_bytes;
}

void FaBitDepthNode::appendPcm16Le(uint16_t sample, std::vector<uint8_t> & out_bytes)
{
  out_bytes.push_back(static_cast<uint8_t>(sample & 0xFFU));
  out_bytes.push_back(static_cast<uint8_t>((sample >> 8U) & 0xFFU));
}

void FaBitDepthNode::appendPcm32Le(uint32_t sample, std::vector<uint8_t> & out_bytes)
{
  out_bytes.push_back(static_cast<uint8_t>(sample & 0xFFU));
  out_bytes.push_back(static_cast<uint8_t>((sample >> 8U) & 0xFFU));
  out_bytes.push_back(static_cast<uint8_t>((sample >> 16U) & 0xFFU));
  out_bytes.push_back(static_cast<uint8_t>((sample >> 24U) & 0xFFU));
}

void FaBitDepthNode::publishDiagnostics()
{
  diagnostic_msgs::msg::DiagnosticArray array_msg;
  array_msg.header.stamp = this->now();

  diagnostic_msgs::msg::DiagnosticStatus status;
  status.name = "fa_bit_depth";
  status.hardware_id = "dsp";
  status.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
  status.message = "running";

  status.values.reserve(10);
  pushKeyValue(status, "input_topic", config_.input_topic);
  pushKeyValue(status, "output_topic", config_.output_topic);
  pushKeyValue(status, "input.encoding", config_.input_encoding);
  pushKeyValue(status, "input.bit_depth", std::to_string(config_.input_bit_depth));
  pushKeyValue(status, "output.encoding", config_.output_encoding);
  pushKeyValue(status, "output.bit_depth", std::to_string(config_.output_bit_depth));
  pushKeyValue(status, "frames.in", std::to_string(frames_in_.load()));
  pushKeyValue(status, "frames.out", std::to_string(frames_out_.load()));
  pushKeyValue(status, "frames.drop", std::to_string(frames_dropped_.load()));

  array_msg.status.push_back(status);
  diag_pub_->publish(array_msg);
}

}  // namespace fa_bit_depth

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<fa_bit_depth::FaBitDepthNode>();
    rclcpp::spin(node);
  } catch (const std::exception & e) {
    RCLCPP_FATAL(rclcpp::get_logger("fa_bit_depth"), "Exception: %s", e.what());
    return EXIT_FAILURE;
  }
  rclcpp::shutdown();
  return EXIT_SUCCESS;
}

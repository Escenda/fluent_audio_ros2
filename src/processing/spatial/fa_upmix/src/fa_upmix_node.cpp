#include "fa_upmix/fa_upmix_node.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
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

namespace fa_upmix
{

namespace
{
constexpr const char * kEncodingFloat32 = "FLOAT32LE";
constexpr const char * kInterleavedLayout = "interleaved";
constexpr const char * kModeMonoDuplicate = "mono_duplicate";
constexpr const char * kModeStereoDuplicatePairs = "stereo_duplicate_pairs";
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

bool isRequiredParameterSet(const rclcpp::Parameter & parameter)
{
  return parameter.get_type() != rclcpp::ParameterType::PARAMETER_NOT_SET;
}

rclcpp::Parameter getRequiredParameter(const rclcpp::Node & node, const std::string & name)
{
  rclcpp::Parameter parameter;
  if (!node.get_parameter(name, parameter) || !isRequiredParameterSet(parameter)) {
    throw std::runtime_error(name + " is required");
  }
  return parameter;
}

std::string readRequiredString(const rclcpp::Node & node, const std::string & name)
{
  const rclcpp::Parameter parameter = getRequiredParameter(node, name);
  if (parameter.get_type() != rclcpp::ParameterType::PARAMETER_STRING) {
    throw std::runtime_error(name + " must be a string");
  }
  return parameter.as_string();
}

int readRequiredInt(const rclcpp::Node & node, const std::string & name)
{
  const rclcpp::Parameter parameter = getRequiredParameter(node, name);
  if (parameter.get_type() != rclcpp::ParameterType::PARAMETER_INTEGER) {
    throw std::runtime_error(name + " must be an integer");
  }
  const int64_t value = parameter.as_int();
  const int64_t min_value = std::numeric_limits<int>::min();
  const int64_t max_value = std::numeric_limits<int>::max();
  if (value < min_value || value > max_value) {
    throw std::runtime_error(name + " is outside supported integer range");
  }
  return static_cast<int>(value);
}

bool readRequiredBool(const rclcpp::Node & node, const std::string & name)
{
  const rclcpp::Parameter parameter = getRequiredParameter(node, name);
  if (parameter.get_type() != rclcpp::ParameterType::PARAMETER_BOOL) {
    throw std::runtime_error(name + " must be a bool");
  }
  return parameter.as_bool();
}

std::string identityWithoutLeadingSlash(const std::string & value)
{
  if (!value.empty() && value.front() == '/') {
    return value.substr(1);
  }
  return value;
}

bool sameIdentityString(const std::string & left, const std::string & right)
{
  return identityWithoutLeadingSlash(left) == identityWithoutLeadingSlash(right);
}
}  // namespace

FaUpmixNode::FaUpmixNode()
: rclcpp::Node("fa_upmix")
{
  RCLCPP_INFO(this->get_logger(), "Starting FA Upmix node");
  loadParameters();
  setupInterfaces();
}

void FaUpmixNode::loadParameters()
{
  this->declare_parameter<std::string>("input_topic");
  this->declare_parameter<std::string>("output_topic");
  this->declare_parameter<std::string>("input_stream_id");
  this->declare_parameter<std::string>("output.stream_id");
  this->declare_parameter<int>("expected.sample_rate");
  this->declare_parameter<int>("expected.input_channels");
  this->declare_parameter<int>("output.channels");
  this->declare_parameter<std::string>("expected.encoding");
  this->declare_parameter<int>("expected.bit_depth");
  this->declare_parameter<std::string>("expected.layout");
  this->declare_parameter<std::string>("mode");
  this->declare_parameter<int>("qos.depth");
  this->declare_parameter<bool>("qos.reliable");
  this->declare_parameter<int>("diagnostics.publish_period_ms");
  this->declare_parameter<int>("diagnostics.qos.depth");
  this->declare_parameter<bool>("diagnostics.qos.reliable");

  config_.input_topic = readRequiredString(*this, "input_topic");
  config_.output_topic = readRequiredString(*this, "output_topic");
  config_.input_stream_id = readRequiredString(*this, "input_stream_id");
  config_.output_stream_id = readRequiredString(*this, "output.stream_id");
  config_.expected_sample_rate = readRequiredInt(*this, "expected.sample_rate");
  config_.expected_input_channels = readRequiredInt(*this, "expected.input_channels");
  config_.output_channels = readRequiredInt(*this, "output.channels");
  config_.expected_encoding = readRequiredString(*this, "expected.encoding");
  config_.expected_bit_depth = readRequiredInt(*this, "expected.bit_depth");
  config_.expected_layout = readRequiredString(*this, "expected.layout");
  config_.mode = readRequiredString(*this, "mode");
  config_.qos_depth = readRequiredInt(*this, "qos.depth");
  config_.qos_reliable = readRequiredBool(*this, "qos.reliable");
  config_.diagnostics_publish_period_ms = readRequiredInt(
    *this, "diagnostics.publish_period_ms");
  config_.diagnostics_qos_depth = readRequiredInt(*this, "diagnostics.qos.depth");
  config_.diagnostics_qos_reliable = readRequiredBool(*this, "diagnostics.qos.reliable");

  if (config_.input_topic.empty()) {
    throw std::runtime_error("input_topic is required");
  }
  if (config_.output_topic.empty()) {
    throw std::runtime_error("output_topic is required");
  }
  if (config_.input_stream_id.empty()) {
    throw std::runtime_error("input_stream_id is required");
  }
  if (config_.output_stream_id.empty()) {
    throw std::runtime_error("output.stream_id is required");
  }
  const std::string resolved_input_topic =
    this->get_node_topics_interface()->resolve_topic_name(config_.input_topic);
  const std::string resolved_output_topic =
    this->get_node_topics_interface()->resolve_topic_name(config_.output_topic);
  if (sameIdentityString(config_.input_stream_id, config_.input_topic) ||
      sameIdentityString(config_.input_stream_id, config_.output_topic) ||
      sameIdentityString(config_.input_stream_id, resolved_input_topic) ||
      sameIdentityString(config_.input_stream_id, resolved_output_topic)) {
    throw std::runtime_error("input_stream_id must be distinct from ROS topics");
  }
  if (sameIdentityString(config_.output_stream_id, config_.input_topic) ||
      sameIdentityString(config_.output_stream_id, config_.output_topic) ||
      sameIdentityString(config_.output_stream_id, resolved_input_topic) ||
      sameIdentityString(config_.output_stream_id, resolved_output_topic)) {
    throw std::runtime_error("output.stream_id must be distinct from ROS topics");
  }
  if (sameIdentityString(config_.input_stream_id, config_.output_stream_id)) {
    throw std::runtime_error("input_stream_id and output.stream_id must be distinct");
  }
  if (config_.expected_sample_rate <= 0) {
    throw std::runtime_error("expected.sample_rate must be > 0");
  }
  if (config_.expected_input_channels <= 0) {
    throw std::runtime_error("expected.input_channels must be > 0");
  }
  if (config_.output_channels <= 0) {
    throw std::runtime_error("output.channels must be > 0");
  }
  if (config_.expected_encoding != kEncodingFloat32) {
    throw std::runtime_error("fa_upmix requires expected.encoding=FLOAT32LE");
  }
  if (config_.expected_bit_depth != 32) {
    throw std::runtime_error("fa_upmix requires expected.bit_depth=32");
  }
  if (config_.expected_layout != kInterleavedLayout) {
    throw std::runtime_error("fa_upmix requires expected.layout=interleaved");
  }
  if (!isSupportedUpmix(config_.mode, config_.expected_input_channels, config_.output_channels)) {
    throw std::runtime_error(
            "fa_upmix supports only mono_duplicate 1->N with N>1 or stereo_duplicate_pairs 2->even N with N>2");
  }
  if (config_.qos_depth <= 0) {
    throw std::runtime_error("qos.depth must be > 0");
  }
  if (config_.diagnostics_publish_period_ms <= 0) {
    throw std::runtime_error("diagnostics.publish_period_ms must be > 0");
  }
  if (config_.diagnostics_qos_depth <= 0) {
    throw std::runtime_error("diagnostics.qos.depth must be > 0");
  }

  RCLCPP_INFO(
    this->get_logger(),
    "Upmix config: input=%s/%s output=%s/%s mode=%s expected=%dHz/%dch/%s/%d/%s output_channels=%d "
    "qos_depth=%d reliable=%s diag=%dms",
    config_.input_topic.c_str(),
    config_.input_stream_id.c_str(),
    config_.output_topic.c_str(),
    config_.output_stream_id.c_str(),
    config_.mode.c_str(),
    config_.expected_sample_rate,
    config_.expected_input_channels,
    config_.expected_encoding.c_str(),
    config_.expected_bit_depth,
    config_.expected_layout.c_str(),
    config_.output_channels,
    config_.qos_depth,
    config_.qos_reliable ? "true" : "false",
    config_.diagnostics_publish_period_ms);
}

void FaUpmixNode::setupInterfaces()
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
    std::bind(&FaUpmixNode::handleFrame, this, std::placeholders::_1));

  rclcpp::QoS diagnostics_qos(static_cast<size_t>(config_.diagnostics_qos_depth));
  if (config_.diagnostics_qos_reliable) {
    diagnostics_qos.reliable();
  } else {
    diagnostics_qos.best_effort();
  }

  diag_pub_ = this->create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
    "diagnostics",
    diagnostics_qos);
  diag_timer_ = this->create_wall_timer(
    std::chrono::milliseconds(config_.diagnostics_publish_period_ms),
    std::bind(&FaUpmixNode::publishDiagnostics, this));
}

void FaUpmixNode::handleFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg)
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
  if (!upmixFrame(*msg, out)) {
    frames_dropped_.fetch_add(1);
    return;
  }

  audio_pub_->publish(out);
  frames_out_.fetch_add(1);
}

bool FaUpmixNode::validateFrame(const fa_interfaces::msg::AudioFrame & msg)
{
  if (msg.source_id.empty() || msg.stream_id.empty()) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "AudioFrame source_id and stream_id are required");
    return false;
  }
  if (msg.stream_id != config_.input_stream_id) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "AudioFrame stream_id mismatch: %s != %s",
      msg.stream_id.c_str(),
      config_.input_stream_id.c_str());
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
      msg.channels != static_cast<uint32_t>(config_.expected_input_channels))
  {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "AudioFrame format mismatch: frame=%uHz/%u config=%dHz/%d",
      msg.sample_rate,
      msg.channels,
      config_.expected_sample_rate,
      config_.expected_input_channels);
    return false;
  }

  const size_t bytes_per_frame = static_cast<size_t>(config_.expected_input_channels) * sizeof(float);
  if (msg.data.empty() || (msg.data.size() % bytes_per_frame) != 0) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "AudioFrame data size is invalid for FLOAT32LE interleaved input channels");
    return false;
  }
  return true;
}

bool FaUpmixNode::upmixFrame(
  const fa_interfaces::msg::AudioFrame & in,
  fa_interfaces::msg::AudioFrame & out)
{
  std::vector<uint8_t> output_data;
  const size_t input_channels = static_cast<size_t>(config_.expected_input_channels);
  const size_t output_channels = static_cast<size_t>(config_.output_channels);
  const size_t frame_count = in.data.size() / (input_channels * sizeof(float));
  output_data.reserve(frame_count * output_channels * sizeof(float));

  if (config_.mode == kModeMonoDuplicate) {
    for (size_t frame_index = 0; frame_index < frame_count; ++frame_index) {
      const float sample = readFloat32Le(in.data, frame_index);
      if (!isNormalizedFinite(sample)) {
        RCLCPP_WARN_THROTTLE(
          this->get_logger(), *this->get_clock(), 3000,
          "Dropping frame because input sample is outside normalized FLOAT32LE range");
        return false;
      }
      for (size_t channel_index = 0; channel_index < output_channels; ++channel_index) {
        if (!isNormalizedFinite(sample)) {
          RCLCPP_WARN_THROTTLE(
            this->get_logger(), *this->get_clock(), 3000,
            "Dropping frame because upmix output is outside normalized FLOAT32LE range");
          return false;
        }
        appendFloat32Le(sample, output_data);
      }
    }
  } else if (config_.mode == kModeStereoDuplicatePairs) {
    const size_t output_pairs = output_channels / 2U;
    for (size_t frame_index = 0; frame_index < frame_count; ++frame_index) {
      const size_t frame_sample_offset = frame_index * input_channels;
      const float left = readFloat32Le(in.data, frame_sample_offset);
      const float right = readFloat32Le(in.data, frame_sample_offset + 1U);
      if (!isNormalizedFinite(left) || !isNormalizedFinite(right)) {
        RCLCPP_WARN_THROTTLE(
          this->get_logger(), *this->get_clock(), 3000,
          "Dropping frame because input sample is outside normalized FLOAT32LE range");
        return false;
      }
      for (size_t pair_index = 0; pair_index < output_pairs; ++pair_index) {
        if (!isNormalizedFinite(left) || !isNormalizedFinite(right)) {
          RCLCPP_WARN_THROTTLE(
            this->get_logger(), *this->get_clock(), 3000,
            "Dropping frame because upmix output is outside normalized FLOAT32LE range");
          return false;
        }
        appendFloat32Le(left, output_data);
        appendFloat32Le(right, output_data);
      }
    }
  } else {
    return false;
  }

  out = in;
  out.stream_id = config_.output_stream_id;
  out.channels = static_cast<uint32_t>(config_.output_channels);
  out.data = output_data;
  return true;
}

bool FaUpmixNode::isSupportedUpmix(
  const std::string & mode,
  int input_channels,
  int output_channels)
{
  return (mode == kModeMonoDuplicate && input_channels == 1 && output_channels > 1) ||
         (mode == kModeStereoDuplicatePairs && input_channels == 2 &&
          output_channels > 2 && (output_channels % 2) == 0);
}

float FaUpmixNode::readFloat32Le(const std::vector<uint8_t> & bytes, size_t sample_index)
{
  uint32_t raw =
    static_cast<uint32_t>(bytes.at(sample_index * sizeof(float))) |
    (static_cast<uint32_t>(bytes.at((sample_index * sizeof(float)) + 1U)) << 8U) |
    (static_cast<uint32_t>(bytes.at((sample_index * sizeof(float)) + 2U)) << 16U) |
    (static_cast<uint32_t>(bytes.at((sample_index * sizeof(float)) + 3U)) << 24U);
  float sample = 0.0F;
  std::memcpy(&sample, &raw, sizeof(float));
  return sample;
}

void FaUpmixNode::appendFloat32Le(float sample, std::vector<uint8_t> & out_bytes)
{
  uint32_t raw = 0;
  std::memcpy(&raw, &sample, sizeof(float));
  out_bytes.push_back(static_cast<uint8_t>(raw & 0xFFU));
  out_bytes.push_back(static_cast<uint8_t>((raw >> 8U) & 0xFFU));
  out_bytes.push_back(static_cast<uint8_t>((raw >> 16U) & 0xFFU));
  out_bytes.push_back(static_cast<uint8_t>((raw >> 24U) & 0xFFU));
}

bool FaUpmixNode::isNormalizedFinite(double sample)
{
  return std::isfinite(sample) && sample >= kMinNormalizedSample && sample <= kMaxNormalizedSample;
}

void FaUpmixNode::publishDiagnostics()
{
  diagnostic_msgs::msg::DiagnosticArray array_msg;
  array_msg.header.stamp = this->now();

  diagnostic_msgs::msg::DiagnosticStatus status;
  status.name = "fa_upmix";
  status.hardware_id = "dsp";
  status.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
  status.message = "running";

  status.values.reserve(14);
  pushKeyValue(status, "input_topic", config_.input_topic);
  pushKeyValue(status, "output_topic", config_.output_topic);
  pushKeyValue(status, "input_stream_id", config_.input_stream_id);
  pushKeyValue(status, "output_stream_id", config_.output_stream_id);
  pushKeyValue(status, "mode", config_.mode);
  pushKeyValue(status, "expected.sample_rate", std::to_string(config_.expected_sample_rate));
  pushKeyValue(status, "expected.input_channels", std::to_string(config_.expected_input_channels));
  pushKeyValue(status, "output.channels", std::to_string(config_.output_channels));
  pushKeyValue(status, "expected.encoding", config_.expected_encoding);
  pushKeyValue(status, "expected.bit_depth", std::to_string(config_.expected_bit_depth));
  pushKeyValue(status, "expected.layout", config_.expected_layout);
  pushKeyValue(status, "frames.in", std::to_string(frames_in_.load()));
  pushKeyValue(status, "frames.out", std::to_string(frames_out_.load()));
  pushKeyValue(status, "frames.drop", std::to_string(frames_dropped_.load()));

  array_msg.status.push_back(status);
  diag_pub_->publish(array_msg);
}

}  // namespace fa_upmix

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<fa_upmix::FaUpmixNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return EXIT_SUCCESS;
  } catch (const std::exception & e) {
    RCLCPP_FATAL(rclcpp::get_logger("fa_upmix"), "Exception: %s", e.what());
    rclcpp::shutdown();
    return EXIT_FAILURE;
  }
}

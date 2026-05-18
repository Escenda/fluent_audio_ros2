#include "fa_pan/fa_pan_node.hpp"

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

namespace fa_pan
{

namespace
{
constexpr const char * kEncodingFloat32 = "FLOAT32LE";
constexpr const char * kInterleavedLayout = "interleaved";
constexpr double kPi = 3.14159265358979323846;
constexpr float kMinNormalizedSample = -1.0F;
constexpr float kMaxNormalizedSample = 1.0F;

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

double readRequiredDouble(const rclcpp::Node & node, const std::string & name)
{
  const rclcpp::Parameter parameter = getRequiredParameter(node, name);
  if (parameter.get_type() != rclcpp::ParameterType::PARAMETER_DOUBLE) {
    throw std::runtime_error(name + " must be a double");
  }
  return parameter.as_double();
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
}  // namespace

FaPanNode::FaPanNode()
: rclcpp::Node("fa_pan")
{
  RCLCPP_INFO(this->get_logger(), "Starting FA Pan node");
  loadParameters();
  configurePan();
  setupInterfaces();
}

void FaPanNode::loadParameters()
{
  this->declare_parameter<std::string>("input_topic");
  this->declare_parameter<std::string>("output_topic");
  this->declare_parameter<double>("pan.position");
  this->declare_parameter<int>("expected.sample_rate");
  this->declare_parameter<int>("expected.channels");
  this->declare_parameter<std::string>("expected.encoding");
  this->declare_parameter<int>("expected.bit_depth");
  this->declare_parameter<std::string>("expected.layout");
  this->declare_parameter<int>("qos.depth");
  this->declare_parameter<bool>("qos.reliable");
  this->declare_parameter<int>("diagnostics.publish_period_ms");
  this->declare_parameter<int>("diagnostics.qos.depth");
  this->declare_parameter<bool>("diagnostics.qos.reliable");

  config_.input_topic = readRequiredString(*this, "input_topic");
  config_.output_topic = readRequiredString(*this, "output_topic");
  config_.pan_position = readRequiredDouble(*this, "pan.position");
  config_.expected_sample_rate = readRequiredInt(*this, "expected.sample_rate");
  config_.expected_channels = readRequiredInt(*this, "expected.channels");
  config_.expected_encoding = readRequiredString(*this, "expected.encoding");
  config_.expected_bit_depth = readRequiredInt(*this, "expected.bit_depth");
  config_.expected_layout = readRequiredString(*this, "expected.layout");
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
  if (!isFinite(config_.pan_position) || config_.pan_position < -1.0 || config_.pan_position > 1.0) {
    throw std::runtime_error("pan.position must be finite and in [-1.0, 1.0]");
  }
  if (config_.expected_sample_rate <= 0) {
    throw std::runtime_error("expected.sample_rate must be > 0");
  }
  if (config_.expected_channels != 2) {
    throw std::runtime_error("fa_pan requires expected.channels=2");
  }
  if (config_.expected_encoding != kEncodingFloat32) {
    throw std::runtime_error("fa_pan requires expected.encoding=FLOAT32LE");
  }
  if (config_.expected_bit_depth != 32) {
    throw std::runtime_error("fa_pan requires expected.bit_depth=32");
  }
  if (config_.expected_layout != kInterleavedLayout) {
    throw std::runtime_error("fa_pan requires expected.layout=interleaved");
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
}

void FaPanNode::configurePan()
{
  const double angle = (config_.pan_position + 1.0) * kPi / 4.0;
  left_gain_ = std::cos(angle);
  right_gain_ = std::sin(angle);

  if (!isFinite(left_gain_) || !isFinite(right_gain_)) {
    throw std::runtime_error("pan gain calculation produced non-finite coefficients");
  }

  RCLCPP_INFO(
    this->get_logger(),
    "Pan config: input=%s output=%s position=%f left_gain=%f right_gain=%f expected=%dHz/%d/%s/%d/%s "
    "qos_depth=%d reliable=%s diag=%dms",
    config_.input_topic.c_str(),
    config_.output_topic.c_str(),
    config_.pan_position,
    left_gain_,
    right_gain_,
    config_.expected_sample_rate,
    config_.expected_channels,
    config_.expected_encoding.c_str(),
    config_.expected_bit_depth,
    config_.expected_layout.c_str(),
    config_.qos_depth,
    config_.qos_reliable ? "true" : "false",
    config_.diagnostics_publish_period_ms);
}

void FaPanNode::setupInterfaces()
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
    std::bind(&FaPanNode::handleFrame, this, std::placeholders::_1));

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
    std::bind(&FaPanNode::publishDiagnostics, this));
}

void FaPanNode::handleFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg)
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
  if (!applyPan(*msg, out)) {
    frames_dropped_.fetch_add(1);
    return;
  }

  audio_pub_->publish(out);
  frames_out_.fetch_add(1);
}

bool FaPanNode::validateFrame(const fa_interfaces::msg::AudioFrame & msg)
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

  const size_t bytes_per_frame = static_cast<size_t>(config_.expected_channels) * sizeof(float);
  if (msg.data.empty() || (msg.data.size() % bytes_per_frame) != 0) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "AudioFrame data size is invalid for stereo FLOAT32LE interleaved samples");
    return false;
  }
  return true;
}

bool FaPanNode::applyPan(
  const fa_interfaces::msg::AudioFrame & in,
  fa_interfaces::msg::AudioFrame & out)
{
  std::vector<uint8_t> output_data;
  output_data.reserve(in.data.size());

  const size_t sample_count = in.data.size() / sizeof(float);
  for (size_t sample_index = 0; sample_index < sample_count; sample_index += 2U) {
    const float left = readFloat32Le(in.data, sample_index);
    const float right = readFloat32Le(in.data, sample_index + 1U);
    if (!isNormalizedFinite(left) || !isNormalizedFinite(right)) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 3000,
        "Dropping frame because input sample is outside normalized FLOAT32LE range");
      return false;
    }

    const double panned_left = static_cast<double>(left) * left_gain_;
    const double panned_right = static_cast<double>(right) * right_gain_;
    if (!isFinite(panned_left) || !isFinite(panned_right) ||
        panned_left < kMinNormalizedSample || panned_left > kMaxNormalizedSample ||
        panned_right < kMinNormalizedSample || panned_right > kMaxNormalizedSample)
    {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 3000,
        "Dropping frame because pan output is outside normalized FLOAT32LE range");
      return false;
    }

    appendFloat32Le(static_cast<float>(panned_left), output_data);
    appendFloat32Le(static_cast<float>(panned_right), output_data);
  }

  out = in;
  out.stream_id = config_.output_topic;
  out.data = output_data;
  return true;
}

float FaPanNode::readFloat32Le(const std::vector<uint8_t> & bytes, size_t sample_index)
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

void FaPanNode::appendFloat32Le(float sample, std::vector<uint8_t> & out_bytes)
{
  uint32_t raw = 0;
  std::memcpy(&raw, &sample, sizeof(float));
  out_bytes.push_back(static_cast<uint8_t>(raw & 0xFFU));
  out_bytes.push_back(static_cast<uint8_t>((raw >> 8U) & 0xFFU));
  out_bytes.push_back(static_cast<uint8_t>((raw >> 16U) & 0xFFU));
  out_bytes.push_back(static_cast<uint8_t>((raw >> 24U) & 0xFFU));
}

bool FaPanNode::isNormalizedFinite(float sample)
{
  return std::isfinite(sample) && sample >= kMinNormalizedSample && sample <= kMaxNormalizedSample;
}

void FaPanNode::publishDiagnostics()
{
  diagnostic_msgs::msg::DiagnosticArray array_msg;
  array_msg.header.stamp = this->now();

  diagnostic_msgs::msg::DiagnosticStatus status;
  status.name = "fa_pan";
  status.hardware_id = "dsp";
  status.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
  status.message = "running";

  status.values.reserve(12);
  pushKeyValue(status, "input_topic", config_.input_topic);
  pushKeyValue(status, "output_topic", config_.output_topic);
  pushKeyValue(status, "pan.position", std::to_string(config_.pan_position));
  pushKeyValue(status, "pan.left_gain", std::to_string(left_gain_));
  pushKeyValue(status, "pan.right_gain", std::to_string(right_gain_));
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

}  // namespace fa_pan

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<fa_pan::FaPanNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return EXIT_SUCCESS;
  } catch (const std::exception & e) {
    RCLCPP_FATAL(rclcpp::get_logger("fa_pan"), "Exception: %s", e.what());
    rclcpp::shutdown();
    return EXIT_FAILURE;
  }
}

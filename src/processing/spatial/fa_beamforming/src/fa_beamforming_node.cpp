#include "fa_beamforming/fa_beamforming_node.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "diagnostic_msgs/msg/diagnostic_status.hpp"
#include "diagnostic_msgs/msg/key_value.hpp"

namespace fa_beamforming
{

namespace
{
constexpr const char * kEncodingFloat32 = "FLOAT32LE";
constexpr const char * kInterleavedLayout = "interleaved";
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

FaBeamformingNode::FaBeamformingNode()
: rclcpp::Node("fa_beamforming")
{
  RCLCPP_INFO(this->get_logger(), "Starting FA Beamforming node");
  loadParameters();
  setupInterfaces();
}

void FaBeamformingNode::loadParameters()
{
  declareRequiredParameter("input_topic");
  declareRequiredParameter("output_topic");
  declareRequiredParameter("beamforming.weights");
  declareRequiredParameter("output.channels");
  declareRequiredParameter("expected.sample_rate");
  declareRequiredParameter("expected.channels");
  declareRequiredParameter("expected.encoding");
  declareRequiredParameter("expected.bit_depth");
  declareRequiredParameter("expected.layout");
  declareRequiredParameter("qos.depth");
  declareRequiredParameter("qos.reliable");
  declareRequiredParameter("diagnostics.publish_period_ms");
  declareRequiredParameter("diagnostics.qos.depth");
  declareRequiredParameter("diagnostics.qos.reliable");

  config_.input_topic = requireStringParameter("input_topic");
  config_.output_topic = requireStringParameter("output_topic");
  config_.weights = requireDoubleArrayParameter("beamforming.weights");
  config_.output_channels = requireIntegerParameter("output.channels");
  config_.expected_sample_rate = requireIntegerParameter("expected.sample_rate");
  config_.expected_channels = requireIntegerParameter("expected.channels");
  config_.expected_encoding = requireStringParameter("expected.encoding");
  config_.expected_bit_depth = requireIntegerParameter("expected.bit_depth");
  config_.expected_layout = requireStringParameter("expected.layout");
  config_.qos_depth = requireIntegerParameter("qos.depth");
  config_.qos_reliable = requireBoolParameter("qos.reliable");
  config_.diagnostics_publish_period_ms =
    requireIntegerParameter("diagnostics.publish_period_ms");
  config_.diagnostics_qos_depth = requireIntegerParameter("diagnostics.qos.depth");
  config_.diagnostics_qos_reliable = requireBoolParameter("diagnostics.qos.reliable");

  if (config_.input_topic.empty()) {
    throw std::runtime_error("input_topic is required");
  }
  if (config_.output_topic.empty()) {
    throw std::runtime_error("output_topic is required");
  }
  if (config_.output_channels != 1) {
    throw std::runtime_error("fa_beamforming requires output.channels=1");
  }
  if (config_.expected_sample_rate <= 0) {
    throw std::runtime_error("expected.sample_rate must be > 0");
  }
  if (config_.expected_channels <= 0) {
    throw std::runtime_error("expected.channels must be > 0");
  }
  if (config_.expected_encoding != kEncodingFloat32) {
    throw std::runtime_error("fa_beamforming requires expected.encoding=FLOAT32LE");
  }
  if (config_.expected_bit_depth != 32) {
    throw std::runtime_error("fa_beamforming requires expected.bit_depth=32");
  }
  if (config_.expected_layout != kInterleavedLayout) {
    throw std::runtime_error("fa_beamforming requires expected.layout=interleaved");
  }
  if (config_.weights.size() != static_cast<size_t>(config_.expected_channels)) {
    throw std::runtime_error("beamforming.weights length must equal expected.channels");
  }

  weights_sum_abs_ = 0.0;
  for (const double weight : config_.weights) {
    if (!std::isfinite(weight)) {
      throw std::runtime_error("beamforming.weights must contain only finite values");
    }
    weights_sum_abs_ += std::abs(weight);
  }
  if (weights_sum_abs_ <= 0.0) {
    throw std::runtime_error("beamforming.weights absolute sum must be nonzero");
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

  const std::string weights_text = formatWeights(config_.weights);
  RCLCPP_INFO(
    this->get_logger(),
    "Beamforming config: input=%s output=%s weights=[%s] expected=%dHz/%dch/%s/%d/%s "
    "output_channels=%d qos_depth=%d reliable=%s diag=%dms",
    config_.input_topic.c_str(),
    config_.output_topic.c_str(),
    weights_text.c_str(),
    config_.expected_sample_rate,
    config_.expected_channels,
    config_.expected_encoding.c_str(),
    config_.expected_bit_depth,
    config_.expected_layout.c_str(),
    config_.output_channels,
    config_.qos_depth,
    config_.qos_reliable ? "true" : "false",
    config_.diagnostics_publish_period_ms);
}

void FaBeamformingNode::declareRequiredParameter(const std::string & name)
{
  this->declare_parameter(name, rclcpp::ParameterValue{});
}

std::string FaBeamformingNode::requireStringParameter(const std::string & name)
{
  rclcpp::Parameter parameter;
  if (!this->get_parameter(name, parameter)) {
    throw std::runtime_error(name + " is required");
  }
  try {
    return parameter.as_string();
  } catch (const std::exception &) {
    throw std::runtime_error(name + " must be a string parameter");
  }
}

int FaBeamformingNode::requireIntegerParameter(const std::string & name)
{
  rclcpp::Parameter parameter;
  if (!this->get_parameter(name, parameter)) {
    throw std::runtime_error(name + " is required");
  }

  int64_t value = 0;
  try {
    value = parameter.as_int();
  } catch (const std::exception &) {
    throw std::runtime_error(name + " must be an integer parameter");
  }

  if (value < static_cast<int64_t>(std::numeric_limits<int>::min()) ||
      value > static_cast<int64_t>(std::numeric_limits<int>::max()))
  {
    throw std::runtime_error(name + " is outside the supported integer range");
  }
  return static_cast<int>(value);
}

bool FaBeamformingNode::requireBoolParameter(const std::string & name)
{
  rclcpp::Parameter parameter;
  if (!this->get_parameter(name, parameter)) {
    throw std::runtime_error(name + " is required");
  }
  try {
    return parameter.as_bool();
  } catch (const std::exception &) {
    throw std::runtime_error(name + " must be a bool parameter");
  }
}

std::vector<double> FaBeamformingNode::requireDoubleArrayParameter(const std::string & name)
{
  rclcpp::Parameter parameter;
  if (!this->get_parameter(name, parameter)) {
    throw std::runtime_error(name + " is required");
  }
  try {
    return parameter.as_double_array();
  } catch (const std::exception &) {
    throw std::runtime_error(name + " must be a double array parameter");
  }
}

void FaBeamformingNode::setupInterfaces()
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
    std::bind(&FaBeamformingNode::handleFrame, this, std::placeholders::_1));

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
    std::bind(&FaBeamformingNode::publishDiagnostics, this));
}

void FaBeamformingNode::handleFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg)
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
  if (!beamformFrame(*msg, out)) {
    frames_dropped_.fetch_add(1);
    return;
  }

  audio_pub_->publish(out);
  frames_out_.fetch_add(1);
}

bool FaBeamformingNode::validateFrame(const fa_interfaces::msg::AudioFrame & msg)
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
      "AudioFrame data size is invalid for FLOAT32LE interleaved input channels");
    return false;
  }
  return true;
}

bool FaBeamformingNode::beamformFrame(
  const fa_interfaces::msg::AudioFrame & in,
  fa_interfaces::msg::AudioFrame & out)
{
  std::vector<uint8_t> output_data;
  const size_t input_channels = static_cast<size_t>(config_.expected_channels);
  const size_t frame_count = in.data.size() / (input_channels * sizeof(float));
  output_data.reserve(frame_count * sizeof(float));

  for (size_t frame_index = 0; frame_index < frame_count; ++frame_index) {
    double weighted_sum = 0.0;
    const size_t frame_sample_offset = frame_index * input_channels;
    for (size_t channel_index = 0; channel_index < input_channels; ++channel_index) {
      const float sample = readFloat32Le(in.data, frame_sample_offset + channel_index);
      if (!isNormalizedFinite(sample)) {
        RCLCPP_WARN_THROTTLE(
          this->get_logger(), *this->get_clock(), 3000,
          "Dropping frame because input sample is outside normalized FLOAT32LE range");
        return false;
      }
      weighted_sum += static_cast<double>(sample) * config_.weights.at(channel_index);
    }

    if (!isNormalizedFinite(weighted_sum)) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 3000,
        "Dropping frame because beamforming output is outside normalized FLOAT32LE range");
      return false;
    }
    appendFloat32Le(static_cast<float>(weighted_sum), output_data);
  }

  out = in;
  out.stream_id = config_.output_topic;
  out.channels = static_cast<uint32_t>(config_.output_channels);
  out.encoding = kEncodingFloat32;
  out.bit_depth = 32;
  out.layout = kInterleavedLayout;
  out.data = output_data;
  return true;
}

float FaBeamformingNode::readFloat32Le(const std::vector<uint8_t> & bytes, size_t sample_index)
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

void FaBeamformingNode::appendFloat32Le(float sample, std::vector<uint8_t> & out_bytes)
{
  uint32_t raw = 0;
  std::memcpy(&raw, &sample, sizeof(float));
  out_bytes.push_back(static_cast<uint8_t>(raw & 0xFFU));
  out_bytes.push_back(static_cast<uint8_t>((raw >> 8U) & 0xFFU));
  out_bytes.push_back(static_cast<uint8_t>((raw >> 16U) & 0xFFU));
  out_bytes.push_back(static_cast<uint8_t>((raw >> 24U) & 0xFFU));
}

bool FaBeamformingNode::isNormalizedFinite(double sample)
{
  return std::isfinite(sample) && sample >= kMinNormalizedSample && sample <= kMaxNormalizedSample;
}

std::string FaBeamformingNode::formatWeights(const std::vector<double> & weights)
{
  std::ostringstream stream;
  for (size_t index = 0; index < weights.size(); ++index) {
    if (index > 0U) {
      stream << ",";
    }
    stream << weights.at(index);
  }
  return stream.str();
}

void FaBeamformingNode::publishDiagnostics()
{
  diagnostic_msgs::msg::DiagnosticArray array_msg;
  array_msg.header.stamp = this->now();

  diagnostic_msgs::msg::DiagnosticStatus status;
  status.name = "fa_beamforming";
  status.hardware_id = "dsp";
  status.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
  status.message = "running";

  status.values.reserve(14);
  pushKeyValue(status, "input_topic", config_.input_topic);
  pushKeyValue(status, "output_topic", config_.output_topic);
  pushKeyValue(status, "beamforming.weights", formatWeights(config_.weights));
  pushKeyValue(status, "beamforming.weights_sum_abs", std::to_string(weights_sum_abs_));
  pushKeyValue(status, "expected.sample_rate", std::to_string(config_.expected_sample_rate));
  pushKeyValue(status, "expected.channels", std::to_string(config_.expected_channels));
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

}  // namespace fa_beamforming

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<fa_beamforming::FaBeamformingNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return EXIT_SUCCESS;
  } catch (const std::exception & e) {
    RCLCPP_FATAL(rclcpp::get_logger("fa_beamforming"), "Exception: %s", e.what());
    rclcpp::shutdown();
    return EXIT_FAILURE;
  }
}

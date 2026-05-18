#include "fa_aec_nn/fa_aec_nn_node.hpp"

#include <chrono>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "diagnostic_msgs/msg/diagnostic_status.hpp"
#include "diagnostic_msgs/msg/key_value.hpp"
#include "fa_aec_nn/backends/aec_nn_backend.hpp"
#include "fa_aec_nn/backends/passthrough_backend.hpp"

namespace fa_aec_nn
{

namespace
{
constexpr int kRequiredSampleRate = 16000;
constexpr const char * kEncodingPcm16 = "PCM16LE";
constexpr const char * kEncodingFloat32 = "FLOAT32LE";
constexpr const char * kInterleavedLayout = "interleaved";

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
    throw std::runtime_error(name + " must be a string parameter");
  }
  return parameter.as_string();
}

int readRequiredInt(const rclcpp::Node & node, const std::string & name)
{
  const rclcpp::Parameter parameter = getRequiredParameter(node, name);
  if (parameter.get_type() != rclcpp::ParameterType::PARAMETER_INTEGER) {
    throw std::runtime_error(name + " must be an integer parameter");
  }
  const int64_t value = parameter.as_int();
  if (value < static_cast<int64_t>(std::numeric_limits<int>::min()) ||
      value > static_cast<int64_t>(std::numeric_limits<int>::max()))
  {
    throw std::runtime_error(name + " must fit in a 32-bit signed integer");
  }
  return static_cast<int>(value);
}

bool readRequiredBool(const rclcpp::Node & node, const std::string & name)
{
  const rclcpp::Parameter parameter = getRequiredParameter(node, name);
  if (parameter.get_type() != rclcpp::ParameterType::PARAMETER_BOOL) {
    throw std::runtime_error(name + " must be a bool parameter");
  }
  return parameter.as_bool();
}

bool isSupportedAudioFormatPair(const std::string & encoding, int bit_depth)
{
  return (encoding == kEncodingPcm16 && bit_depth == 16) ||
         (encoding == kEncodingFloat32 && bit_depth == 32);
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

FaAecNnNode::FaAecNnNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("fa_aec_nn", options)
{
  RCLCPP_INFO(this->get_logger(), "Starting FA AEC NN node");
  loadParameters();
  initializeBackend();
  setupInterfaces();
}

FaAecNnNode::~FaAecNnNode() = default;

void FaAecNnNode::loadParameters()
{
  this->declare_parameter<bool>("enabled");
  this->declare_parameter<std::string>("backend.name");
  this->declare_parameter<std::string>("input_topic");
  this->declare_parameter<std::string>("output_topic");
  this->declare_parameter<std::string>("input_stream_id");
  this->declare_parameter<std::string>("output.stream_id");
  this->declare_parameter<int>("expected_sample_rate");
  this->declare_parameter<int>("expected_channels");
  this->declare_parameter<std::string>("expected.encoding");
  this->declare_parameter<int>("expected.bit_depth");
  this->declare_parameter<int>("qos.depth");
  this->declare_parameter<bool>("qos.reliable");
  this->declare_parameter<int>("diagnostics.publish_period_ms");
  this->declare_parameter<int>("diagnostics.qos.depth");
  this->declare_parameter<bool>("diagnostics.qos.reliable");

  config_.enabled = readRequiredBool(*this, "enabled");
  config_.backend_name = readRequiredString(*this, "backend.name");
  config_.input_topic = readRequiredString(*this, "input_topic");
  config_.output_topic = readRequiredString(*this, "output_topic");
  config_.input_stream_id = readRequiredString(*this, "input_stream_id");
  config_.output_stream_id = readRequiredString(*this, "output.stream_id");
  config_.expected_sample_rate = readRequiredInt(*this, "expected_sample_rate");
  config_.expected_channels = readRequiredInt(*this, "expected_channels");
  config_.expected_encoding = readRequiredString(*this, "expected.encoding");
  config_.expected_bit_depth = readRequiredInt(*this, "expected.bit_depth");
  config_.qos_depth = readRequiredInt(*this, "qos.depth");
  config_.qos_reliable = readRequiredBool(*this, "qos.reliable");
  config_.diagnostics_publish_period_ms = readRequiredInt(
    *this,
    "diagnostics.publish_period_ms");
  config_.diagnostics_qos_depth = readRequiredInt(*this, "diagnostics.qos.depth");
  config_.diagnostics_qos_reliable = readRequiredBool(*this, "diagnostics.qos.reliable");

  if (config_.input_topic.empty()) {
    throw std::runtime_error("input_topic is required (set via YAML)");
  }
  if (config_.output_topic.empty()) {
    throw std::runtime_error("output_topic is required (set via YAML)");
  }
  if (config_.input_stream_id.empty()) {
    throw std::runtime_error("input_stream_id is required (set via YAML)");
  }
  if (config_.output_stream_id.empty()) {
    throw std::runtime_error("output.stream_id is required (set via YAML)");
  }
  config_.resolved_input_topic =
    this->get_node_topics_interface()->resolve_topic_name(config_.input_topic);
  config_.resolved_output_topic =
    this->get_node_topics_interface()->resolve_topic_name(config_.output_topic);
  if (config_.resolved_input_topic == config_.resolved_output_topic) {
    throw std::runtime_error("resolved input_topic and output_topic must be distinct");
  }
  if (sameIdentityString(config_.input_stream_id, config_.input_topic) ||
      sameIdentityString(config_.input_stream_id, config_.output_topic) ||
      sameIdentityString(config_.input_stream_id, config_.resolved_input_topic) ||
      sameIdentityString(config_.input_stream_id, config_.resolved_output_topic))
  {
    throw std::runtime_error("input_stream_id must be distinct from ROS topics");
  }
  if (sameIdentityString(config_.output_stream_id, config_.input_topic) ||
      sameIdentityString(config_.output_stream_id, config_.output_topic) ||
      sameIdentityString(config_.output_stream_id, config_.resolved_input_topic) ||
      sameIdentityString(config_.output_stream_id, config_.resolved_output_topic))
  {
    throw std::runtime_error("output.stream_id must be distinct from ROS topics");
  }
  if (sameIdentityString(config_.input_stream_id, config_.output_stream_id)) {
    throw std::runtime_error("input_stream_id and output.stream_id must be distinct");
  }
  if (config_.expected_sample_rate != kRequiredSampleRate) {
    throw std::runtime_error(
            "fa_aec_nn requires expected_sample_rate=16000 by design (got " +
            std::to_string(config_.expected_sample_rate) + ")");
  }
  if (config_.expected_channels <= 0) {
    throw std::runtime_error("expected_channels must be > 0");
  }
  if (!isSupportedAudioFormatPair(config_.expected_encoding, config_.expected_bit_depth)) {
    throw std::runtime_error("expected encoding/bit_depth must be PCM16LE/16 or FLOAT32LE/32");
  }
  if (config_.qos_depth <= 0) {
    throw std::runtime_error("qos.depth must be > 0 (set via YAML)");
  }
  if (config_.diagnostics_publish_period_ms <= 0) {
    throw std::runtime_error("diagnostics.publish_period_ms must be > 0 (set via YAML)");
  }
  if (config_.diagnostics_qos_depth <= 0) {
    throw std::runtime_error("diagnostics.qos.depth must be > 0");
  }
  if (config_.backend_name.empty()) {
    throw std::runtime_error("backend.name is required (set via YAML)");
  }
  if (config_.backend_name != "passthrough") {
    throw std::runtime_error("backend.name must be passthrough");
  }

  RCLCPP_INFO(this->get_logger(),
    "AEC NN config: enabled=%s backend.name=%s input=%s/%s output=%s/%s expected_sr=%d expected_ch=%d expected=%s/%d qos_depth=%d reliable=%s",
    config_.enabled ? "true" : "false",
    config_.backend_name.c_str(),
    config_.input_topic.c_str(),
    config_.input_stream_id.c_str(),
    config_.output_topic.c_str(),
    config_.output_stream_id.c_str(),
    config_.expected_sample_rate,
    config_.expected_channels,
    config_.expected_encoding.c_str(),
    config_.expected_bit_depth,
    config_.qos_depth,
    config_.qos_reliable ? "true" : "false");
}

void FaAecNnNode::initializeBackend()
{
  if (config_.backend_name == "passthrough") {
    backend_ = std::make_unique<backends::PassthroughBackend>();
    return;
  }
  throw std::runtime_error("unsupported fa_aec_nn backend.name: " + config_.backend_name);
}

void FaAecNnNode::setupInterfaces()
{
  rclcpp::QoS qos(static_cast<size_t>(config_.qos_depth));
  if (config_.qos_reliable) {
    qos.reliable();
  } else {
    qos.best_effort();
  }

  pub_ = this->create_publisher<fa_interfaces::msg::AudioFrame>(config_.output_topic, qos);
  sub_ = this->create_subscription<fa_interfaces::msg::AudioFrame>(
    config_.input_topic, qos,
    std::bind(&FaAecNnNode::onAudioFrame, this, std::placeholders::_1));

  rclcpp::QoS diagnostics_qos(static_cast<size_t>(config_.diagnostics_qos_depth));
  if (config_.diagnostics_qos_reliable) {
    diagnostics_qos.reliable();
  } else {
    diagnostics_qos.best_effort();
  }

  diag_pub_ = this->create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
    "diagnostics", diagnostics_qos);
  diag_timer_ = this->create_wall_timer(
    std::chrono::milliseconds(config_.diagnostics_publish_period_ms),
    std::bind(&FaAecNnNode::publishDiagnostics, this));
}

bool FaAecNnNode::validateFrame(const fa_interfaces::msg::AudioFrame & msg) const
{
  if (msg.sample_rate != static_cast<uint32_t>(config_.expected_sample_rate)) {
    return false;
  }
  if (msg.channels != static_cast<uint32_t>(config_.expected_channels)) {
    return false;
  }
  if (msg.channels == 0 || msg.sample_rate == 0) {
    return false;
  }
  if (msg.encoding != config_.expected_encoding ||
      msg.bit_depth != static_cast<uint32_t>(config_.expected_bit_depth))
  {
    return false;
  }
  if (msg.source_id.empty()) {
    return false;
  }
  if (msg.stream_id != config_.input_stream_id) {
    return false;
  }
  if (msg.layout != kInterleavedLayout) {
    return false;
  }
  if (msg.data.empty()) {
    return false;
  }
  const size_t bytes_per_sample = static_cast<size_t>(msg.bit_depth / 8);
  const size_t bytes_per_frame = static_cast<size_t>(msg.channels) * bytes_per_sample;
  if (bytes_per_frame == 0 || (msg.data.size() % bytes_per_frame) != 0) {
    return false;
  }
  return true;
}

void FaAecNnNode::onAudioFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg)
{
  in_.fetch_add(1);
  if (!msg) {
    throw std::logic_error("fa_aec_nn received a null AudioFrame pointer");
  }
  if (!pub_) {
    throw std::logic_error("fa_aec_nn publisher is not initialized");
  }

  if (!config_.enabled) {
    drop_.fetch_add(1);
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping frame because fa_aec_nn is disabled; disable the system node instead");
    return;
  }

  if (!validateFrame(*msg)) {
    drop_.fetch_add(1);
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping invalid frame: sr=%u ch=%u bits=%u bytes=%zu (expected sr=%d ch=%d)",
      msg->sample_rate, msg->channels, msg->bit_depth, msg->data.size(),
      config_.expected_sample_rate, config_.expected_channels);
    return;
  }

  if (!backend_) {
    throw std::logic_error("fa_aec_nn backend is not initialized");
  }

  backends::AudioChunk chunk;
  chunk.sample_rate = static_cast<int>(msg->sample_rate);
  chunk.channels = static_cast<int>(msg->channels);
  chunk.bit_depth = static_cast<int>(msg->bit_depth);
  chunk.encoding = msg->encoding;
  chunk.layout = msg->layout;
  chunk.data = msg->data.data();
  chunk.data_size = msg->data.size();

  backends::ProcessedAudioChunk processed_chunk;
  try {
    processed_chunk = backend_->process(chunk);
  } catch (const std::exception & e) {
    drop_.fetch_add(1);
    RCLCPP_ERROR_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping frame because fa_aec_nn backend failed: %s", e.what());
    return;
  }

  const auto processed_contract_error =
    backends::validateProcessedAudioChunk(chunk, processed_chunk);
  if (!processed_contract_error.empty()) {
    drop_.fetch_add(1);
    RCLCPP_ERROR_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping frame because fa_aec_nn backend violated output contract: %s",
      processed_contract_error.c_str());
    return;
  }

  auto out_msg = *msg;
  out_msg.sample_rate = static_cast<uint32_t>(processed_chunk.sample_rate);
  out_msg.channels = static_cast<uint32_t>(processed_chunk.channels);
  out_msg.encoding = processed_chunk.encoding;
  out_msg.bit_depth = static_cast<uint32_t>(processed_chunk.bit_depth);
  out_msg.data = std::move(processed_chunk.data);
  out_msg.stream_id = config_.output_stream_id;
  out_msg.layout = processed_chunk.layout;
  pub_->publish(out_msg);
  out_.fetch_add(1);
}

void FaAecNnNode::publishDiagnostics()
{
  diagnostic_msgs::msg::DiagnosticArray array_msg;
  array_msg.header.stamp = this->now();

  diagnostic_msgs::msg::DiagnosticStatus status;
  status.name = "fa_aec_nn";
  status.hardware_id = "dsp";
  status.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
  status.message = "running";

  auto push_kv = [&status](const std::string & key, const std::string & value) {
      diagnostic_msgs::msg::KeyValue kv;
      kv.key = key;
      kv.value = value;
      status.values.push_back(kv);
    };

  status.values.reserve(14);
  push_kv("enabled", config_.enabled ? "true" : "false");
  push_kv("backend.name", config_.backend_name);
  push_kv("input_topic", config_.input_topic);
  push_kv("output_topic", config_.output_topic);
  push_kv("input_stream_id", config_.input_stream_id);
  push_kv("output_stream_id", config_.output_stream_id);
  push_kv("resolved_input_topic", config_.resolved_input_topic);
  push_kv("resolved_output_topic", config_.resolved_output_topic);
  push_kv("expected_sample_rate", std::to_string(config_.expected_sample_rate));
  push_kv("expected_channels", std::to_string(config_.expected_channels));
  push_kv("frames.in", std::to_string(in_.load()));
  push_kv("frames.out", std::to_string(out_.load()));
  push_kv("frames.drop", std::to_string(drop_.load()));

  array_msg.status.push_back(status);
  diag_pub_->publish(array_msg);
}

}  // namespace fa_aec_nn

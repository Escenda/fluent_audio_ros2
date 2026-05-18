#include "fa_sidechain/fa_sidechain_node.hpp"

#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "diagnostic_msgs/msg/diagnostic_status.hpp"
#include "diagnostic_msgs/msg/key_value.hpp"

namespace fa_sidechain
{

namespace
{
constexpr const char * kEncodingFloat32 = "FLOAT32LE";
constexpr const char * kInterleavedLayout = "interleaved";
constexpr int kMaxExpectedSampleRate = 384000;
constexpr int kMaxExpectedChannels = 64;

std::string removeLeadingSlashes(std::string value)
{
  while (!value.empty() && value.front() == '/') {
    value.erase(value.begin());
  }
  return value;
}

bool sameIdentityString(const std::string & left, const std::string & right)
{
  return left == right || removeLeadingSlashes(left) == removeLeadingSlashes(right);
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
  if (value < static_cast<int64_t>(std::numeric_limits<int>::min()) ||
      value > static_cast<int64_t>(std::numeric_limits<int>::max()))
  {
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

bool streamMatchesTopic(
  const std::string & stream_id,
  const std::string & raw_sidechain,
  const std::string & raw_control,
  const std::string & resolved_sidechain,
  const std::string & resolved_control)
{
  return sameIdentityString(stream_id, raw_sidechain) ||
         sameIdentityString(stream_id, raw_control) ||
         sameIdentityString(stream_id, resolved_sidechain) ||
         sameIdentityString(stream_id, resolved_control);
}
}  // namespace

FaSidechainNode::FaSidechainNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("fa_sidechain", options)
{
  RCLCPP_INFO(this->get_logger(), "Starting FA Sidechain node");
  loadParameters();
  configureBackend();
  setupInterfaces();
}

void FaSidechainNode::loadParameters()
{
  this->declare_parameter<std::string>("sidechain_topic");
  this->declare_parameter<std::string>("control_topic");
  this->declare_parameter<std::string>("sidechain_stream_id");
  this->declare_parameter<std::string>("control.stream_id");
  this->declare_parameter<double>("detector.threshold_rms");
  this->declare_parameter<double>("detector.active_gain_db");
  this->declare_parameter<double>("detector.inactive_gain_db");
  this->declare_parameter<int>("control.sample_rate");
  this->declare_parameter<int>("expected.sample_rate");
  this->declare_parameter<int>("expected.channels");
  this->declare_parameter<std::string>("expected.encoding");
  this->declare_parameter<int>("expected.bit_depth");
  this->declare_parameter<std::string>("expected.layout");
  this->declare_parameter<int>("qos.depth");
  this->declare_parameter<bool>("qos.reliable");
  this->declare_parameter<int>("diagnostics.qos.depth");
  this->declare_parameter<bool>("diagnostics.qos.reliable");
  this->declare_parameter<int>("diagnostics.publish_period_ms");

  config_.sidechain_topic = readRequiredString(*this, "sidechain_topic");
  config_.control_topic = readRequiredString(*this, "control_topic");
  config_.sidechain_stream_id = readRequiredString(*this, "sidechain_stream_id");
  config_.control_stream_id = readRequiredString(*this, "control.stream_id");
  config_.threshold_rms = readRequiredDouble(*this, "detector.threshold_rms");
  config_.active_gain_db = readRequiredDouble(*this, "detector.active_gain_db");
  config_.inactive_gain_db = readRequiredDouble(*this, "detector.inactive_gain_db");
  config_.control_sample_rate = readRequiredInt(*this, "control.sample_rate");
  config_.expected_sample_rate = readRequiredInt(*this, "expected.sample_rate");
  config_.expected_channels = readRequiredInt(*this, "expected.channels");
  config_.expected_encoding = readRequiredString(*this, "expected.encoding");
  config_.expected_bit_depth = readRequiredInt(*this, "expected.bit_depth");
  config_.expected_layout = readRequiredString(*this, "expected.layout");
  config_.qos_depth = readRequiredInt(*this, "qos.depth");
  config_.qos_reliable = readRequiredBool(*this, "qos.reliable");
  config_.diagnostics_qos_depth = readRequiredInt(*this, "diagnostics.qos.depth");
  config_.diagnostics_qos_reliable = readRequiredBool(*this, "diagnostics.qos.reliable");
  config_.diagnostics_publish_period_ms = readRequiredInt(*this, "diagnostics.publish_period_ms");

  if (config_.sidechain_topic.empty()) {
    throw std::runtime_error("sidechain_topic is required");
  }
  if (config_.control_topic.empty()) {
    throw std::runtime_error("control_topic is required");
  }
  const std::string resolved_sidechain =
    this->get_node_topics_interface()->resolve_topic_name(config_.sidechain_topic);
  const std::string resolved_control =
    this->get_node_topics_interface()->resolve_topic_name(config_.control_topic);
  if (resolved_sidechain == resolved_control) {
    throw std::runtime_error("control_topic must differ from sidechain_topic after resolution");
  }
  if (config_.sidechain_stream_id.empty()) {
    throw std::runtime_error("sidechain_stream_id is required");
  }
  if (config_.control_stream_id.empty()) {
    throw std::runtime_error("control.stream_id is required");
  }
  if (config_.sidechain_stream_id == config_.control_stream_id) {
    throw std::runtime_error("sidechain_stream_id and control.stream_id must be distinct");
  }
  if (streamMatchesTopic(
      config_.sidechain_stream_id,
      config_.sidechain_topic,
      config_.control_topic,
      resolved_sidechain,
      resolved_control))
  {
    throw std::runtime_error("sidechain_stream_id must not match raw or resolved topic names");
  }
  if (streamMatchesTopic(
      config_.control_stream_id,
      config_.sidechain_topic,
      config_.control_topic,
      resolved_sidechain,
      resolved_control))
  {
    throw std::runtime_error("control.stream_id must not match raw or resolved topic names");
  }
  if (!isFinite(config_.threshold_rms) || config_.threshold_rms <= 0.0 || config_.threshold_rms > 1.0) {
    throw std::runtime_error("detector.threshold_rms must be finite and in (0.0, 1.0]");
  }
  if (!isFinite(config_.active_gain_db)) {
    throw std::runtime_error("detector.active_gain_db must be finite");
  }
  if (!isFinite(config_.inactive_gain_db)) {
    throw std::runtime_error("detector.inactive_gain_db must be finite");
  }
  if (config_.control_sample_rate <= 0) {
    throw std::runtime_error("control.sample_rate must be > 0");
  }
  if (config_.expected_sample_rate <= 0 || config_.expected_sample_rate > kMaxExpectedSampleRate) {
    throw std::runtime_error("expected.sample_rate must be in (0, 384000]");
  }
  if (config_.expected_channels <= 0 || config_.expected_channels > kMaxExpectedChannels) {
    throw std::runtime_error("expected.channels must be in (0, 64]");
  }
  if (config_.expected_encoding != kEncodingFloat32) {
    throw std::runtime_error("fa_sidechain requires expected.encoding=FLOAT32LE");
  }
  if (config_.expected_bit_depth != 32) {
    throw std::runtime_error("fa_sidechain requires expected.bit_depth=32");
  }
  if (config_.expected_layout != kInterleavedLayout) {
    throw std::runtime_error("fa_sidechain requires expected.layout=interleaved");
  }
  if (config_.qos_depth <= 0) {
    throw std::runtime_error("qos.depth must be > 0");
  }
  if (config_.diagnostics_qos_depth <= 0) {
    throw std::runtime_error("diagnostics.qos.depth must be > 0");
  }
  if (config_.diagnostics_publish_period_ms <= 0) {
    throw std::runtime_error("diagnostics.publish_period_ms must be > 0");
  }

  RCLCPP_INFO(
    this->get_logger(),
    "Sidechain config: sidechain_topic=%s control_topic=%s sidechain_stream=%s control_stream=%s threshold=%f active=%f inactive=%f control_rate=%d expected=%dHz/%d/%s/%d/%s qos=%d reliable=%s diag_qos=%d diag_reliable=%s diag=%dms",
    config_.sidechain_topic.c_str(),
    config_.control_topic.c_str(),
    config_.sidechain_stream_id.c_str(),
    config_.control_stream_id.c_str(),
    config_.threshold_rms,
    config_.active_gain_db,
    config_.inactive_gain_db,
    config_.control_sample_rate,
    config_.expected_sample_rate,
    config_.expected_channels,
    config_.expected_encoding.c_str(),
    config_.expected_bit_depth,
    config_.expected_layout.c_str(),
    config_.qos_depth,
    config_.qos_reliable ? "true" : "false",
    config_.diagnostics_qos_depth,
    config_.diagnostics_qos_reliable ? "true" : "false",
    config_.diagnostics_publish_period_ms);
}

void FaSidechainNode::configureBackend()
{
  backend_ = std::make_unique<backends::InternalSidechainDetectorBackend>(
    backends::InternalSidechainDetectorConfig{
      config_.expected_channels,
      config_.threshold_rms,
      config_.active_gain_db,
      config_.inactive_gain_db});
}

void FaSidechainNode::setupInterfaces()
{
  rclcpp::QoS qos(static_cast<size_t>(config_.qos_depth));
  if (config_.qos_reliable) {
    qos.reliable();
  } else {
    qos.best_effort();
  }

  rclcpp::QoS diagnostics_qos(static_cast<size_t>(config_.diagnostics_qos_depth));
  if (config_.diagnostics_qos_reliable) {
    diagnostics_qos.reliable();
  } else {
    diagnostics_qos.best_effort();
  }

  control_pub_ = this->create_publisher<fa_interfaces::msg::AudioFrame>(config_.control_topic, qos);
  sidechain_sub_ = this->create_subscription<fa_interfaces::msg::AudioFrame>(
    config_.sidechain_topic,
    qos,
    std::bind(&FaSidechainNode::handleSidechainFrame, this, std::placeholders::_1));

  diag_pub_ = this->create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
    "diagnostics",
    diagnostics_qos);
  diag_timer_ = this->create_wall_timer(
    std::chrono::milliseconds(config_.diagnostics_publish_period_ms),
    std::bind(&FaSidechainNode::publishDiagnostics, this));
}

void FaSidechainNode::handleSidechainFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg)
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

  if (!backend_) {
    throw std::logic_error("FaSidechainNode backend is not initialized");
  }

  std::vector<uint8_t> control_data;
  const auto result = backend_->detect(msg->data, control_data);
  if (result.status != backends::ProcessStatus::kOk) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping sidechain frame because backend rejected input: %s",
      backends::processStatusMessage(result.status));
    frames_dropped_.fetch_add(1);
    return;
  }

  fa_interfaces::msg::AudioFrame control_frame;
  if (!buildControlFrame(*msg, control_data, control_frame)) {
    frames_dropped_.fetch_add(1);
    return;
  }

  control_pub_->publish(control_frame);
  frames_out_.fetch_add(1);
  if (result.active) {
    active_frames_.fetch_add(1);
  } else {
    inactive_frames_.fetch_add(1);
  }
}

bool FaSidechainNode::validateFrame(const fa_interfaces::msg::AudioFrame & msg)
{
  if (msg.source_id.empty() || msg.stream_id.empty()) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping sidechain frame because source_id and stream_id are required");
    return false;
  }
  if (msg.stream_id != config_.sidechain_stream_id) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping sidechain frame because stream_id mismatch: %s != %s",
      msg.stream_id.c_str(),
      config_.sidechain_stream_id.c_str());
    return false;
  }
  if (msg.sample_rate != static_cast<uint32_t>(config_.expected_sample_rate) ||
      msg.channels != static_cast<uint32_t>(config_.expected_channels))
  {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping sidechain frame because format mismatch: frame=%uHz/%u config=%dHz/%d",
      msg.sample_rate,
      msg.channels,
      config_.expected_sample_rate,
      config_.expected_channels);
    return false;
  }
  if (msg.encoding != config_.expected_encoding ||
      msg.bit_depth != static_cast<uint32_t>(config_.expected_bit_depth))
  {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping sidechain frame because encoding mismatch: %s/%u != %s/%d",
      msg.encoding.c_str(),
      msg.bit_depth,
      config_.expected_encoding.c_str(),
      config_.expected_bit_depth);
    return false;
  }
  if (msg.layout != config_.expected_layout) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping sidechain frame because layout mismatch: %s != %s",
      msg.layout.c_str(),
      config_.expected_layout.c_str());
    return false;
  }
  if (msg.data.empty() || (msg.data.size() % bytesPerFrame()) != 0) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping sidechain frame because data size is invalid for FLOAT32LE interleaved samples");
    return false;
  }
  return true;
}

bool FaSidechainNode::buildControlFrame(
  const fa_interfaces::msg::AudioFrame & input,
  const std::vector<uint8_t> & control_data,
  fa_interfaces::msg::AudioFrame & output)
{
  if (control_data.size() != sizeof(float)) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping sidechain frame because backend control payload size is invalid");
    return false;
  }

  output.header = input.header;
  output.source_id = input.source_id;
  output.stream_id = config_.control_stream_id;
  output.sample_rate = static_cast<uint32_t>(config_.control_sample_rate);
  output.channels = 1;
  output.encoding = kEncodingFloat32;
  output.bit_depth = 32;
  output.layout = kInterleavedLayout;
  output.epoch = input.epoch;
  output.data = control_data;
  return true;
}

size_t FaSidechainNode::bytesPerFrame() const
{
  return static_cast<size_t>(config_.expected_channels) * sizeof(float);
}

void FaSidechainNode::publishDiagnostics()
{
  if (!backend_) {
    throw std::logic_error("FaSidechainNode backend is not initialized");
  }

  diagnostic_msgs::msg::DiagnosticArray array_msg;
  array_msg.header.stamp = this->now();

  diagnostic_msgs::msg::DiagnosticStatus status;
  status.name = "fa_sidechain";
  status.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
  status.message = "running";
  status.values.reserve(32);
  pushKeyValue(status, "sidechain_topic", config_.sidechain_topic);
  pushKeyValue(status, "control_topic", config_.control_topic);
  pushKeyValue(status, "sidechain_stream_id", config_.sidechain_stream_id);
  pushKeyValue(status, "control_stream_id", config_.control_stream_id);
  pushKeyValue(status, "threshold_rms", std::to_string(backend_->thresholdRms()));
  pushKeyValue(status, "active_gain_db", std::to_string(backend_->activeGainDb()));
  pushKeyValue(status, "active_gain_linear", std::to_string(backend_->activeGainLinear()));
  pushKeyValue(status, "inactive_gain_db", std::to_string(backend_->inactiveGainDb()));
  pushKeyValue(status, "inactive_gain_linear", std::to_string(backend_->inactiveGainLinear()));
  pushKeyValue(status, "control.sample_rate", std::to_string(config_.control_sample_rate));
  pushKeyValue(status, "expected.sample_rate", std::to_string(config_.expected_sample_rate));
  pushKeyValue(status, "expected.channels", std::to_string(config_.expected_channels));
  pushKeyValue(status, "expected.encoding", config_.expected_encoding);
  pushKeyValue(status, "expected.bit_depth", std::to_string(config_.expected_bit_depth));
  pushKeyValue(status, "expected.layout", config_.expected_layout);
  pushKeyValue(status, "diagnostics_qos_depth", std::to_string(config_.diagnostics_qos_depth));
  pushKeyValue(status, "diagnostics_qos_reliable", config_.diagnostics_qos_reliable ? "true" : "false");
  pushKeyValue(status, "frames_in", std::to_string(frames_in_.load()));
  pushKeyValue(status, "frames_out", std::to_string(frames_out_.load()));
  pushKeyValue(status, "frames_dropped", std::to_string(frames_dropped_.load()));
  pushKeyValue(status, "last_rms", std::to_string(backend_->lastRms()));
  pushKeyValue(status, "last_gain_linear", std::to_string(backend_->lastGainLinear()));
  pushKeyValue(status, "last_active", backend_->lastActive() ? "true" : "false");
  pushKeyValue(status, "active_frames", std::to_string(active_frames_.load()));
  pushKeyValue(status, "inactive_frames", std::to_string(inactive_frames_.load()));

  array_msg.status.push_back(status);
  diag_pub_->publish(array_msg);
}

}  // namespace fa_sidechain

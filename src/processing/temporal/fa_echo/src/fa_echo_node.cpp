#include "fa_echo/fa_echo_node.hpp"

#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

#include "diagnostic_msgs/msg/diagnostic_status.hpp"
#include "diagnostic_msgs/msg/key_value.hpp"
#include "fa_echo/backends/internal_feedback_echo.hpp"

namespace fa_echo
{

namespace
{
constexpr const char * kEncodingFloat32 = "FLOAT32LE";
constexpr const char * kInterleavedLayout = "interleaved";
constexpr int kMaxExpectedSampleRate = 384000;
constexpr int kMaxExpectedChannels = 64;

bool isFinite(double value)
{
  return std::isfinite(value);
}

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

FaEchoNode::FaEchoNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("fa_echo", options)
{
  RCLCPP_INFO(this->get_logger(), "Starting FA Echo node");
  loadParameters();
  configureBackend();
  setupInterfaces();
}

FaEchoNode::~FaEchoNode() = default;

void FaEchoNode::loadParameters()
{
  this->declare_parameter<std::string>("input_topic");
  this->declare_parameter<std::string>("output_topic");
  this->declare_parameter<std::string>("input_stream_id");
  this->declare_parameter<std::string>("output.stream_id");
  this->declare_parameter<double>("echo.delay_ms");
  this->declare_parameter<double>("echo.feedback_gain");
  this->declare_parameter<double>("echo.wet_gain");
  this->declare_parameter<double>("echo.dry_gain");
  this->declare_parameter<int>("expected.sample_rate");
  this->declare_parameter<int>("expected.channels");
  this->declare_parameter<std::string>("expected.encoding");
  this->declare_parameter<int>("expected.bit_depth");
  this->declare_parameter<std::string>("expected.layout");
  this->declare_parameter<int>("qos.depth");
  this->declare_parameter<bool>("qos.reliable");
  this->declare_parameter<int>("diagnostics.publish_period_ms");

  config_.input_topic = readRequiredString(*this, "input_topic");
  config_.output_topic = readRequiredString(*this, "output_topic");
  config_.input_stream_id = readRequiredString(*this, "input_stream_id");
  config_.output_stream_id = readRequiredString(*this, "output.stream_id");
  config_.delay_ms = readRequiredDouble(*this, "echo.delay_ms");
  config_.feedback_gain = readRequiredDouble(*this, "echo.feedback_gain");
  config_.wet_gain = readRequiredDouble(*this, "echo.wet_gain");
  config_.dry_gain = readRequiredDouble(*this, "echo.dry_gain");
  config_.expected_sample_rate = readRequiredInt(*this, "expected.sample_rate");
  config_.expected_channels = readRequiredInt(*this, "expected.channels");
  config_.expected_encoding = readRequiredString(*this, "expected.encoding");
  config_.expected_bit_depth = readRequiredInt(*this, "expected.bit_depth");
  config_.expected_layout = readRequiredString(*this, "expected.layout");
  config_.qos_depth = readRequiredInt(*this, "qos.depth");
  config_.qos_reliable = readRequiredBool(*this, "qos.reliable");
  config_.diagnostics_publish_period_ms = readRequiredInt(*this, "diagnostics.publish_period_ms");

  if (config_.input_topic.empty()) {
    throw std::runtime_error("input_topic is required");
  }
  if (config_.output_topic.empty()) {
    throw std::runtime_error("output_topic is required");
  }
  if (config_.input_topic == config_.output_topic) {
    throw std::runtime_error("input_topic and output_topic must be distinct");
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
  if (resolved_input_topic == resolved_output_topic) {
    throw std::runtime_error("input_topic and output_topic must resolve to distinct ROS topics");
  }
  if (sameIdentityString(config_.input_stream_id, config_.input_topic) ||
      sameIdentityString(config_.input_stream_id, config_.output_topic) ||
      sameIdentityString(config_.input_stream_id, resolved_input_topic) ||
      sameIdentityString(config_.input_stream_id, resolved_output_topic))
  {
    throw std::runtime_error("input_stream_id must be distinct from ROS topics");
  }
  if (sameIdentityString(config_.output_stream_id, config_.input_topic) ||
      sameIdentityString(config_.output_stream_id, config_.output_topic) ||
      sameIdentityString(config_.output_stream_id, resolved_input_topic) ||
      sameIdentityString(config_.output_stream_id, resolved_output_topic))
  {
    throw std::runtime_error("output.stream_id must be distinct from ROS topics");
  }
  if (config_.input_stream_id == config_.output_stream_id) {
    throw std::runtime_error("input_stream_id and output.stream_id must be distinct");
  }
  if (!isFinite(config_.delay_ms) || config_.delay_ms <= 0.0) {
    throw std::runtime_error("echo.delay_ms must be finite and > 0.0");
  }
  if (!isFinite(config_.feedback_gain) || std::abs(config_.feedback_gain) >= 1.0) {
    throw std::runtime_error("echo.feedback_gain must be finite and satisfy abs(value) < 1.0");
  }
  if (!isFinite(config_.wet_gain)) {
    throw std::runtime_error("echo.wet_gain must be finite");
  }
  if (!isFinite(config_.dry_gain)) {
    throw std::runtime_error("echo.dry_gain must be finite");
  }
  if (config_.expected_sample_rate <= 0 || config_.expected_sample_rate > kMaxExpectedSampleRate) {
    throw std::runtime_error("expected.sample_rate must satisfy 0 < value <= 384000");
  }
  if (config_.expected_channels <= 0 || config_.expected_channels > kMaxExpectedChannels) {
    throw std::runtime_error("expected.channels must satisfy 0 < value <= 64");
  }
  if (config_.expected_encoding != kEncodingFloat32) {
    throw std::runtime_error("fa_echo requires expected.encoding=FLOAT32LE");
  }
  if (config_.expected_bit_depth != 32) {
    throw std::runtime_error("fa_echo requires expected.bit_depth=32");
  }
  if (config_.expected_layout != kInterleavedLayout) {
    throw std::runtime_error("fa_echo requires expected.layout=interleaved");
  }
  if (config_.qos_depth <= 0) {
    throw std::runtime_error("qos.depth must be > 0");
  }
  if (config_.diagnostics_publish_period_ms <= 0) {
    throw std::runtime_error("diagnostics.publish_period_ms must be > 0");
  }

  const double raw_delay_samples =
    config_.delay_ms * static_cast<double>(config_.expected_sample_rate) / 1000.0;
  if (!isFinite(raw_delay_samples) ||
      raw_delay_samples > static_cast<double>(std::numeric_limits<long long>::max()))
  {
    throw std::runtime_error("echo.delay_ms converts to an unsupported sample count");
  }
  const size_t delay_samples = static_cast<size_t>(std::llround(raw_delay_samples));
  if (delay_samples == 0) {
    throw std::runtime_error("echo.delay_ms must convert to at least 1 sample");
  }

  RCLCPP_INFO(
    this->get_logger(),
    "Echo config: input_topic=%s output_topic=%s input_stream_id=%s output_stream_id=%s delay=%.3fms delay_samples=%zu "
    "feedback=%.6f wet=%.6f dry=%.6f expected=%dHz/%d/%s/%d/%s qos_depth=%d reliable=%s diag=%dms",
    config_.input_topic.c_str(),
    config_.output_topic.c_str(),
    config_.input_stream_id.c_str(),
    config_.output_stream_id.c_str(),
    config_.delay_ms,
    delay_samples,
    config_.feedback_gain,
    config_.wet_gain,
    config_.dry_gain,
    config_.expected_sample_rate,
    config_.expected_channels,
    config_.expected_encoding.c_str(),
    config_.expected_bit_depth,
    config_.expected_layout.c_str(),
    config_.qos_depth,
    config_.qos_reliable ? "true" : "false",
    config_.diagnostics_publish_period_ms);
}

void FaEchoNode::configureBackend()
{
  const double raw_delay_samples =
    config_.delay_ms * static_cast<double>(config_.expected_sample_rate) / 1000.0;
  const size_t delay_samples = static_cast<size_t>(std::llround(raw_delay_samples));
  backend_ = std::make_unique<backends::InternalFeedbackEchoBackend>(
    backends::InternalFeedbackEchoConfig{
      config_.expected_channels,
      delay_samples,
      config_.feedback_gain,
      config_.wet_gain,
      config_.dry_gain});
}

void FaEchoNode::setupInterfaces()
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
    std::bind(&FaEchoNode::handleFrame, this, std::placeholders::_1));

  diag_pub_ = this->create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
    "diagnostics",
    rclcpp::SystemDefaultsQoS());
  diag_timer_ = this->create_wall_timer(
    std::chrono::milliseconds(config_.diagnostics_publish_period_ms),
    std::bind(&FaEchoNode::publishDiagnostics, this));
}

void FaEchoNode::handleFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg)
{
  messages_in_.fetch_add(1);

  if (!msg) {
    throw std::logic_error("FaEchoNode received null AudioFrame pointer");
  }
  if (!validateFrame(*msg)) {
    messages_dropped_.fetch_add(1);
    return;
  }

  fa_interfaces::msg::AudioFrame out;
  if (!applyEcho(*msg, out)) {
    messages_dropped_.fetch_add(1);
    return;
  }

  if (!audio_pub_) {
    throw std::logic_error("FaEchoNode publisher is not initialized");
  }
  audio_pub_->publish(out);
  messages_out_.fetch_add(1);
}

bool FaEchoNode::validateFrame(const fa_interfaces::msg::AudioFrame & msg)
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

bool FaEchoNode::applyEcho(
  const fa_interfaces::msg::AudioFrame & in,
  fa_interfaces::msg::AudioFrame & out)
{
  if (!backend_) {
    throw std::logic_error("FaEchoNode backend is not initialized");
  }

  out = in;
  out.stream_id = config_.output_stream_id;
  const backends::ProcessResult result = backend_->process(in.source_id, in.data, out.data);
  if (result.status != backends::ProcessStatus::kOk) {
    const char * status_message = backends::processStatusMessage(result.status);
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping frame because echo backend rejected input or output: %s",
      status_message);
    return false;
  }

  if (result.source_reset) {
    source_resets_.fetch_add(1);
    RCLCPP_WARN(
      this->get_logger(),
      "AudioFrame source_id changed; resetting echo delay state for source %s",
      in.source_id.c_str());
  }
  return true;
}

size_t FaEchoNode::bytesPerFrame() const
{
  return static_cast<size_t>(config_.expected_channels) * sizeof(float);
}

void FaEchoNode::publishDiagnostics()
{
  diagnostic_msgs::msg::DiagnosticArray array_msg;
  array_msg.header.stamp = this->now();

  diagnostic_msgs::msg::DiagnosticStatus status;
  status.name = "fa_echo";
  status.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
  status.message = "running";
  status.values.reserve(14);
  pushKeyValue(status, "delay_ms", std::to_string(config_.delay_ms));
  pushKeyValue(status, "delay_samples", std::to_string(backend_->delaySamples()));
  pushKeyValue(status, "feedback_gain", std::to_string(backend_->feedbackGain()));
  pushKeyValue(status, "wet_gain", std::to_string(backend_->wetGain()));
  pushKeyValue(status, "dry_gain", std::to_string(backend_->dryGain()));
  pushKeyValue(status, "current_source_id", backend_->currentSourceId());
  pushKeyValue(status, "input_topic", config_.input_topic);
  pushKeyValue(status, "output_topic", config_.output_topic);
  pushKeyValue(status, "input_stream_id", config_.input_stream_id);
  pushKeyValue(status, "output_stream_id", config_.output_stream_id);
  pushKeyValue(status, "messages_in", std::to_string(messages_in_.load()));
  pushKeyValue(status, "messages_out", std::to_string(messages_out_.load()));
  pushKeyValue(status, "messages_dropped", std::to_string(messages_dropped_.load()));
  pushKeyValue(status, "source_resets", std::to_string(source_resets_.load()));
  pushKeyValue(status, "expected_channels", std::to_string(config_.expected_channels));

  array_msg.status.push_back(status);
  diag_pub_->publish(array_msg);
}

}  // namespace fa_echo

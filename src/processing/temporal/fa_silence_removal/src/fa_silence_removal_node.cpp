#include "fa_silence_removal/fa_silence_removal_node.hpp"

#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

#include "diagnostic_msgs/msg/diagnostic_status.hpp"
#include "diagnostic_msgs/msg/key_value.hpp"
#include "fa_silence_removal/backends/internal_rms_silence_removal.hpp"

namespace fa_silence_removal
{

namespace
{
constexpr const char * kEncodingFloat32 = "FLOAT32LE";
constexpr const char * kInterleavedLayout = "interleaved";
constexpr double kMillisecondsPerSecond = 1000.0;
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

FaSilenceRemovalNode::FaSilenceRemovalNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("fa_silence_removal", options)
{
  RCLCPP_INFO(this->get_logger(), "Starting FA Silence Removal node");
  loadParameters();
  configureBackend();
  setupInterfaces();
}

FaSilenceRemovalNode::~FaSilenceRemovalNode() = default;

void FaSilenceRemovalNode::loadParameters()
{
  this->declare_parameter<std::string>("input_topic");
  this->declare_parameter<std::string>("output_topic");
  this->declare_parameter<std::string>("input_stream_id");
  this->declare_parameter<std::string>("output.stream_id");
  this->declare_parameter<double>("threshold.rms");
  this->declare_parameter<double>("hangover_ms");
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
  config_.input_stream_id = readRequiredString(*this, "input_stream_id");
  config_.output_stream_id = readRequiredString(*this, "output.stream_id");
  config_.threshold_rms = readRequiredDouble(*this, "threshold.rms");
  config_.hangover_ms = readRequiredDouble(*this, "hangover_ms");
  config_.expected_sample_rate = readRequiredInt(*this, "expected.sample_rate");
  config_.expected_channels = readRequiredInt(*this, "expected.channels");
  config_.expected_encoding = readRequiredString(*this, "expected.encoding");
  config_.expected_bit_depth = readRequiredInt(*this, "expected.bit_depth");
  config_.expected_layout = readRequiredString(*this, "expected.layout");
  config_.qos_depth = readRequiredInt(*this, "qos.depth");
  config_.qos_reliable = readRequiredBool(*this, "qos.reliable");
  config_.diagnostics_publish_period_ms = readRequiredInt(*this, "diagnostics.publish_period_ms");
  config_.diagnostics_qos_depth = readRequiredInt(*this, "diagnostics.qos.depth");
  config_.diagnostics_qos_reliable = readRequiredBool(*this, "diagnostics.qos.reliable");

  if (config_.input_topic.empty()) {
    throw std::runtime_error("input_topic is required");
  }
  if (config_.output_topic.empty()) {
    throw std::runtime_error("output_topic is required");
  }
  const std::string resolved_input_topic =
    this->get_node_topics_interface()->resolve_topic_name(config_.input_topic);
  const std::string resolved_output_topic =
    this->get_node_topics_interface()->resolve_topic_name(config_.output_topic);
  if (resolved_input_topic == resolved_output_topic) {
    throw std::runtime_error("input_topic and output_topic must resolve to distinct ROS topics");
  }
  if (config_.input_stream_id.empty()) {
    throw std::runtime_error("input_stream_id is required");
  }
  if (config_.output_stream_id.empty()) {
    throw std::runtime_error("output.stream_id is required");
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
  if (!isFinite(config_.threshold_rms) ||
      config_.threshold_rms < 0.0 ||
      config_.threshold_rms > 1.0)
  {
    throw std::runtime_error("threshold.rms must be finite and in [0.0, 1.0]");
  }
  if (!isFinite(config_.hangover_ms) || config_.hangover_ms < 0.0) {
    throw std::runtime_error("hangover_ms must be finite and >= 0");
  }
  if (config_.expected_sample_rate <= 0 || config_.expected_sample_rate > kMaxExpectedSampleRate) {
    throw std::runtime_error("expected.sample_rate must satisfy 0 < value <= 384000");
  }
  if (config_.expected_channels <= 0 || config_.expected_channels > kMaxExpectedChannels) {
    throw std::runtime_error("expected.channels must satisfy 0 < value <= 64");
  }
  if (config_.expected_encoding != kEncodingFloat32) {
    throw std::runtime_error("fa_silence_removal requires expected.encoding=FLOAT32LE");
  }
  if (config_.expected_bit_depth != 32) {
    throw std::runtime_error("fa_silence_removal requires expected.bit_depth=32");
  }
  if (config_.expected_layout != kInterleavedLayout) {
    throw std::runtime_error("fa_silence_removal requires expected.layout=interleaved");
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

  const double raw_hangover_samples =
    config_.hangover_ms * static_cast<double>(config_.expected_sample_rate) / kMillisecondsPerSecond;
  if (!isFinite(raw_hangover_samples) ||
      raw_hangover_samples > static_cast<double>(std::numeric_limits<size_t>::max()))
  {
    throw std::runtime_error("hangover_ms converts to an unsupported sample count");
  }
  hangover_samples_ = static_cast<size_t>(std::ceil(raw_hangover_samples));

  RCLCPP_INFO(
    this->get_logger(),
    "Silence removal config: input_topic=%s output_topic=%s input_stream_id=%s "
    "output_stream_id=%s threshold_rms=%f hangover=%.3fms hangover_samples=%zu "
    "expected=%dHz/%d/%s/%d/%s qos_depth=%d reliable=%s diag=%dms",
    config_.input_topic.c_str(),
    config_.output_topic.c_str(),
    config_.input_stream_id.c_str(),
    config_.output_stream_id.c_str(),
    config_.threshold_rms,
    config_.hangover_ms,
    hangover_samples_,
    config_.expected_sample_rate,
    config_.expected_channels,
    config_.expected_encoding.c_str(),
    config_.expected_bit_depth,
    config_.expected_layout.c_str(),
    config_.qos_depth,
    config_.qos_reliable ? "true" : "false",
    config_.diagnostics_publish_period_ms);
}

void FaSilenceRemovalNode::configureBackend()
{
  backend_ = std::make_unique<backends::InternalRmsSilenceRemovalBackend>(
    backends::InternalRmsSilenceRemovalConfig{
      config_.expected_channels,
      config_.threshold_rms,
      hangover_samples_});
}

void FaSilenceRemovalNode::setupInterfaces()
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
    std::bind(&FaSilenceRemovalNode::handleFrame, this, std::placeholders::_1));

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
    std::bind(&FaSilenceRemovalNode::publishDiagnostics, this));
}

void FaSilenceRemovalNode::handleFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg)
{
  messages_in_.fetch_add(1);

  if (!msg) {
    throw std::logic_error("FaSilenceRemovalNode received null AudioFrame pointer");
  }
  if (!validateFrame(*msg)) {
    invalid_frames_dropped_.fetch_add(1);
    messages_dropped_.fetch_add(1);
    return;
  }
  if (!backend_) {
    throw std::logic_error("FaSilenceRemovalNode backend is not initialized");
  }

  const backends::ProcessResult result = backend_->process(msg->data);
  if (result.status != backends::ProcessStatus::kOk) {
    const char * status_message = backends::processStatusMessage(result.status);
    invalid_frames_dropped_.fetch_add(1);
    messages_dropped_.fetch_add(1);
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping frame because silence removal backend rejected input: %s",
      status_message);
    return;
  }

  const char * decision_name = backends::decisionName(result.decision);
  switch (result.decision) {
    case backends::Decision::kAcceptedActive:
      active_frames_.fetch_add(1);
      publishAcceptedFrame(*msg);
      return;
    case backends::Decision::kAcceptedHangover:
      hangover_frames_.fetch_add(1);
      publishAcceptedFrame(*msg);
      return;
    case backends::Decision::kDroppedSilent:
      silent_frames_dropped_.fetch_add(1);
      messages_dropped_.fetch_add(1);
      RCLCPP_DEBUG(
        this->get_logger(),
        "Dropping silent frame: decision=%s rms=%.6f",
        decision_name,
        result.rms);
      return;
  }
  throw std::logic_error("unhandled silence removal backend decision");
}

bool FaSilenceRemovalNode::validateFrame(const fa_interfaces::msg::AudioFrame & msg)
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

void FaSilenceRemovalNode::publishAcceptedFrame(const fa_interfaces::msg::AudioFrame & msg)
{
  if (!audio_pub_) {
    throw std::logic_error("FaSilenceRemovalNode publisher is not initialized");
  }
  fa_interfaces::msg::AudioFrame out = msg;
  out.stream_id = config_.output_stream_id;
  audio_pub_->publish(out);
  messages_out_.fetch_add(1);
}

size_t FaSilenceRemovalNode::bytesPerFrame() const
{
  return static_cast<size_t>(config_.expected_channels) * sizeof(float);
}

void FaSilenceRemovalNode::publishDiagnostics()
{
  if (!backend_) {
    throw std::logic_error("FaSilenceRemovalNode backend is not initialized");
  }

  diagnostic_msgs::msg::DiagnosticArray array_msg;
  array_msg.header.stamp = this->now();

  diagnostic_msgs::msg::DiagnosticStatus status;
  status.name = "fa_silence_removal";
  status.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
  status.message = "running";
  status.values.reserve(18);
  pushKeyValue(status, "threshold_rms", std::to_string(config_.threshold_rms));
  pushKeyValue(status, "hangover_ms", std::to_string(config_.hangover_ms));
  pushKeyValue(status, "hangover_samples", std::to_string(backend_->hangoverSamples()));
  pushKeyValue(
    status,
    "hangover_samples_remaining",
    std::to_string(backend_->hangoverSamplesRemaining()));
  pushKeyValue(status, "last_rms", std::to_string(backend_->lastRms()));
  pushKeyValue(status, "input_topic", config_.input_topic);
  pushKeyValue(status, "output_topic", config_.output_topic);
  pushKeyValue(status, "input_stream_id", config_.input_stream_id);
  pushKeyValue(status, "output_stream_id", config_.output_stream_id);
  pushKeyValue(status, "expected_sample_rate", std::to_string(config_.expected_sample_rate));
  pushKeyValue(status, "expected_channels", std::to_string(config_.expected_channels));
  pushKeyValue(status, "messages_in", std::to_string(messages_in_.load()));
  pushKeyValue(status, "messages_out", std::to_string(messages_out_.load()));
  pushKeyValue(status, "messages_dropped", std::to_string(messages_dropped_.load()));
  pushKeyValue(status, "invalid_frames_dropped", std::to_string(invalid_frames_dropped_.load()));
  pushKeyValue(status, "silent_frames_dropped", std::to_string(silent_frames_dropped_.load()));
  pushKeyValue(status, "hangover_frames", std::to_string(hangover_frames_.load()));
  pushKeyValue(status, "active_frames", std::to_string(active_frames_.load()));

  array_msg.status.push_back(status);
  diag_pub_->publish(array_msg);
}

}  // namespace fa_silence_removal

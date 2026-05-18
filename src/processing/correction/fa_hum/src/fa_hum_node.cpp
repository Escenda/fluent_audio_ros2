#include "fa_hum/fa_hum_node.hpp"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "diagnostic_msgs/msg/diagnostic_status.hpp"
#include "diagnostic_msgs/msg/key_value.hpp"
#include "fa_hum/backends/internal_notch_cascade.hpp"

namespace fa_hum
{

namespace
{
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

double readRequiredDouble(const rclcpp::Node & node, const std::string & name)
{
  const rclcpp::Parameter parameter = getRequiredParameter(node, name);
  if (parameter.get_type() != rclcpp::ParameterType::PARAMETER_DOUBLE) {
    throw std::runtime_error(name + " must be a double parameter");
  }
  return parameter.as_double();
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

FaHumNode::FaHumNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("fa_hum", options)
{
  RCLCPP_INFO(this->get_logger(), "Starting FA Hum Removal node");
  loadParameters();
  configureBackend();
  setupInterfaces();
}

FaHumNode::~FaHumNode() = default;

void FaHumNode::loadParameters()
{
  this->declare_parameter<std::string>("input_topic");
  this->declare_parameter<std::string>("output_topic");
  this->declare_parameter<std::string>("input_stream_id");
  this->declare_parameter<std::string>("output.stream_id");
  this->declare_parameter<double>("hum.frequency_hz");
  this->declare_parameter<int>("hum.harmonics");
  this->declare_parameter<double>("hum.q");
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
  config_.frequency_hz = readRequiredDouble(*this, "hum.frequency_hz");
  config_.harmonics = readRequiredInt(*this, "hum.harmonics");
  config_.q = readRequiredDouble(*this, "hum.q");
  config_.expected_sample_rate = readRequiredInt(*this, "expected.sample_rate");
  config_.expected_channels = readRequiredInt(*this, "expected.channels");
  config_.expected_encoding = readRequiredString(*this, "expected.encoding");
  config_.expected_bit_depth = readRequiredInt(*this, "expected.bit_depth");
  config_.expected_layout = readRequiredString(*this, "expected.layout");
  config_.qos_depth = readRequiredInt(*this, "qos.depth");
  config_.qos_reliable = readRequiredBool(*this, "qos.reliable");
  config_.diagnostics_publish_period_ms = readRequiredInt(
    *this,
    "diagnostics.publish_period_ms");
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
  if (config_.input_topic == config_.output_topic) {
    throw std::runtime_error("input_topic and output_topic must be distinct");
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
  if (config_.expected_sample_rate <= 0) {
    throw std::runtime_error("expected.sample_rate must be > 0");
  }
  const double nyquist_hz = static_cast<double>(config_.expected_sample_rate) / 2.0;
  if (!isFinite(config_.frequency_hz) || config_.frequency_hz <= 0.0) {
    throw std::runtime_error("hum.frequency_hz must be finite and > 0.0");
  }
  if (config_.frequency_hz >= nyquist_hz) {
    throw std::runtime_error("hum.frequency_hz must produce at least one harmonic below Nyquist");
  }
  if (config_.harmonics < 1) {
    throw std::runtime_error("hum.harmonics must be >= 1");
  }
  if (!isFinite(config_.q) || config_.q <= 0.0) {
    throw std::runtime_error("hum.q must be finite and > 0.0");
  }
  if (config_.expected_channels <= 0) {
    throw std::runtime_error("expected.channels must be > 0");
  }
  if (config_.expected_encoding != kEncodingFloat32) {
    throw std::runtime_error("fa_hum requires expected.encoding=FLOAT32LE");
  }
  if (config_.expected_bit_depth != 32) {
    throw std::runtime_error("fa_hum requires expected.bit_depth=32");
  }
  if (config_.expected_layout != kInterleavedLayout) {
    throw std::runtime_error("fa_hum requires expected.layout=interleaved");
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
    "Hum config: input=%s output=%s frequency=%fHz harmonics=%d q=%f "
    "streams=%s->%s expected=%dHz/%d/%s/%d/%s qos_depth=%d reliable=%s diag=%dms",
    config_.input_topic.c_str(),
    config_.output_topic.c_str(),
    config_.frequency_hz,
    config_.harmonics,
    config_.q,
    config_.input_stream_id.c_str(),
    config_.output_stream_id.c_str(),
    config_.expected_sample_rate,
    config_.expected_channels,
    config_.expected_encoding.c_str(),
    config_.expected_bit_depth,
    config_.expected_layout.c_str(),
    config_.qos_depth,
    config_.qos_reliable ? "true" : "false",
    config_.diagnostics_publish_period_ms);
}

void FaHumNode::configureBackend()
{
  backend_ = std::make_unique<backends::InternalNotchCascadeBackend>(
    backends::InternalNotchCascadeConfig{
      config_.expected_sample_rate,
      config_.expected_channels,
      config_.frequency_hz,
      config_.harmonics,
      config_.q});
}

void FaHumNode::setupInterfaces()
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
    std::bind(&FaHumNode::handleFrame, this, std::placeholders::_1));

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
    std::bind(&FaHumNode::publishDiagnostics, this));
}

void FaHumNode::handleFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg)
{
  frames_in_.fetch_add(1);

  if (!msg) {
    throw std::logic_error("received null AudioFrame pointer");
  }

  if (!validateFrame(*msg)) {
    frames_dropped_.fetch_add(1);
    return;
  }

  fa_interfaces::msg::AudioFrame out;
  if (!applyHumRemoval(*msg, out)) {
    frames_dropped_.fetch_add(1);
    return;
  }

  if (!audio_pub_) {
    throw std::logic_error("audio publisher is not initialized");
  }
  rememberAcceptedFrame(*msg);
  audio_pub_->publish(out);
  frames_out_.fetch_add(1);
}

bool FaHumNode::validateFrame(const fa_interfaces::msg::AudioFrame & msg)
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
  if (msg.encoding != config_.expected_encoding || msg.bit_depth != static_cast<uint32_t>(config_.expected_bit_depth)) {
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
      "AudioFrame data size is invalid for FLOAT32LE interleaved samples");
    return false;
  }
  if (isStaleFrame(msg)) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "AudioFrame header.stamp is older than the last accepted frame for the same source_id and epoch");
    return false;
  }
  return true;
}

bool FaHumNode::isStaleFrame(const fa_interfaces::msg::AudioFrame & msg) const
{
  if (!has_last_accepted_frame_) {
    return false;
  }
  if (msg.source_id != last_source_id_ || msg.epoch != last_epoch_) {
    return false;
  }
  if (msg.header.stamp.sec < last_stamp_sec_) {
    return true;
  }
  if (msg.header.stamp.sec == last_stamp_sec_ &&
      msg.header.stamp.nanosec < last_stamp_nanosec_)
  {
    return true;
  }
  return false;
}

void FaHumNode::rememberAcceptedFrame(const fa_interfaces::msg::AudioFrame & msg)
{
  has_last_accepted_frame_ = true;
  last_source_id_ = msg.source_id;
  last_epoch_ = msg.epoch;
  last_stamp_sec_ = msg.header.stamp.sec;
  last_stamp_nanosec_ = msg.header.stamp.nanosec;
}

bool FaHumNode::applyHumRemoval(
  const fa_interfaces::msg::AudioFrame & in,
  fa_interfaces::msg::AudioFrame & out)
{
  if (!backend_) {
    throw std::logic_error("hum backend is not initialized");
  }

  const std::string previous_source_id = backend_->activeSourceId();
  const uint32_t previous_epoch = backend_->activeEpoch();
  std::vector<uint8_t> processed_data;
  const backends::ProcessResult result = backend_->process(
    in.source_id,
    in.epoch,
    in.data,
    processed_data);
  if (result.status != backends::ProcessStatus::kOk) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping frame because hum backend rejected input or output: %s",
      backends::processStatusMessage(result.status));
    return false;
  }

  if (result.source_reset) {
    RCLCPP_INFO(
      this->get_logger(),
      "Resetting hum filter state because source_id changed: %s -> %s",
      previous_source_id.c_str(),
      in.source_id.c_str());
    source_resets_.fetch_add(1);
  }
  if (result.epoch_reset) {
    RCLCPP_INFO(
      this->get_logger(),
      "Resetting hum filter state because epoch changed for source_id=%s: %u -> %u",
      in.source_id.c_str(),
      previous_epoch,
      in.epoch);
    epoch_resets_.fetch_add(1);
  }

  out = in;
  out.stream_id = config_.output_stream_id;
  out.data = std::move(processed_data);
  return true;
}

void FaHumNode::publishDiagnostics()
{
  if (!diag_pub_) {
    throw std::logic_error("diagnostics publisher is not initialized");
  }
  if (!backend_) {
    throw std::logic_error("hum backend is not initialized");
  }

  diagnostic_msgs::msg::DiagnosticArray array_msg;
  array_msg.header.stamp = this->now();

  diagnostic_msgs::msg::DiagnosticStatus status;
  status.name = "fa_hum";
  status.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
  status.message = "running";
  const std::vector<double> center_frequencies = backend_->centerFrequenciesHz();
  status.values.reserve(14 + center_frequencies.size());
  pushKeyValue(status, "hum_frequency_hz", std::to_string(config_.frequency_hz));
  pushKeyValue(status, "hum_harmonics_requested", std::to_string(config_.harmonics));
  pushKeyValue(status, "hum_q", std::to_string(config_.q));
  pushKeyValue(status, "notch_stage_count", std::to_string(backend_->stageCount()));
  for (size_t stage = 0; stage < center_frequencies.size(); ++stage) {
    pushKeyValue(
      status,
      "notch_stage_" + std::to_string(stage + 1) + "_center_hz",
      std::to_string(center_frequencies.at(stage)));
  }
  pushKeyValue(status, "frames_in", std::to_string(frames_in_.load()));
  pushKeyValue(status, "frames_out", std::to_string(frames_out_.load()));
  pushKeyValue(status, "frames_dropped", std::to_string(frames_dropped_.load()));
  pushKeyValue(status, "source_resets", std::to_string(source_resets_.load()));
  pushKeyValue(status, "epoch_resets", std::to_string(epoch_resets_.load()));
  pushKeyValue(status, "active_source_id", backend_->activeSourceId());
  pushKeyValue(status, "active_epoch", std::to_string(backend_->activeEpoch()));
  pushKeyValue(status, "input_topic", config_.input_topic);
  pushKeyValue(status, "output_topic", config_.output_topic);
  pushKeyValue(status, "input_stream_id", config_.input_stream_id);
  pushKeyValue(status, "output_stream_id", config_.output_stream_id);
  pushKeyValue(status, "resolved_input_topic", config_.resolved_input_topic);
  pushKeyValue(status, "resolved_output_topic", config_.resolved_output_topic);
  pushKeyValue(status, "backend.name", backends::InternalNotchCascadeBackend::kName);

  array_msg.status.push_back(status);
  diag_pub_->publish(array_msg);
}

}  // namespace fa_hum

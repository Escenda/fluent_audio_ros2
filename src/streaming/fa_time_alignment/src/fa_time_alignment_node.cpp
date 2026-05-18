#include "fa_time_alignment/fa_time_alignment_node.hpp"

#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

#include "diagnostic_msgs/msg/diagnostic_status.hpp"
#include "diagnostic_msgs/msg/key_value.hpp"

namespace fa_time_alignment
{

namespace
{
constexpr int64_t kNanosecondsPerSecond = 1000000000LL;
constexpr int64_t kMaxBuiltinTimeNanoseconds = (2147483647LL * kNanosecondsPerSecond) + 999999999LL;
constexpr long double kNanosecondsPerMillisecond = 1000000.0L;

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

long double stampToNanoseconds(const builtin_interfaces::msg::Time & stamp)
{
  return (static_cast<long double>(stamp.sec) * static_cast<long double>(kNanosecondsPerSecond)) +
         static_cast<long double>(stamp.nanosec);
}

builtin_interfaces::msg::Time nanosecondsToStamp(const int64_t nanoseconds)
{
  builtin_interfaces::msg::Time stamp;
  stamp.sec = static_cast<int32_t>(nanoseconds / kNanosecondsPerSecond);
  stamp.nanosec = static_cast<uint32_t>(nanoseconds % kNanosecondsPerSecond);
  return stamp;
}
}  // namespace

FaTimeAlignmentNode::FaTimeAlignmentNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("fa_time_alignment", options)
{
  RCLCPP_INFO(this->get_logger(), "Starting FA Time Alignment node");
  loadParameters();
  setupInterfaces();
}

void FaTimeAlignmentNode::loadParameters()
{
  this->declare_parameter<std::string>("input_topic");
  this->declare_parameter<std::string>("output_topic");
  this->declare_parameter<int>("expected.sample_rate");
  this->declare_parameter<int>("expected.channels");
  this->declare_parameter<std::string>("expected.encoding");
  this->declare_parameter<int>("expected.bit_depth");
  this->declare_parameter<std::string>("expected.layout");
  this->declare_parameter<double>("alignment.period_ms");
  this->declare_parameter<double>("alignment.phase_ms");
  this->declare_parameter<double>("alignment.max_adjust_ms");
  this->declare_parameter<int>("qos.depth");
  this->declare_parameter<bool>("qos.reliable");
  this->declare_parameter<int>("diagnostics.publish_period_ms");

  config_.input_topic = readRequiredString(*this, "input_topic");
  config_.output_topic = readRequiredString(*this, "output_topic");
  config_.expected_sample_rate = readRequiredInt(*this, "expected.sample_rate");
  config_.expected_channels = readRequiredInt(*this, "expected.channels");
  config_.expected_encoding = readRequiredString(*this, "expected.encoding");
  config_.expected_bit_depth = readRequiredInt(*this, "expected.bit_depth");
  config_.expected_layout = readRequiredString(*this, "expected.layout");
  config_.alignment_period_ms = readRequiredDouble(*this, "alignment.period_ms");
  config_.alignment_phase_ms = readRequiredDouble(*this, "alignment.phase_ms");
  config_.alignment_max_adjust_ms = readRequiredDouble(*this, "alignment.max_adjust_ms");
  config_.qos_depth = readRequiredInt(*this, "qos.depth");
  config_.qos_reliable = readRequiredBool(*this, "qos.reliable");
  config_.diagnostics_publish_period_ms = readRequiredInt(
    *this,
    "diagnostics.publish_period_ms");

  if (config_.input_topic.empty()) {
    throw std::runtime_error("input_topic is required");
  }
  if (config_.output_topic.empty()) {
    throw std::runtime_error("output_topic is required");
  }
  if (config_.expected_sample_rate <= 0) {
    throw std::runtime_error("expected.sample_rate must be > 0");
  }
  if (config_.expected_channels <= 0) {
    throw std::runtime_error("expected.channels must be > 0");
  }
  if (config_.expected_encoding.empty()) {
    throw std::runtime_error("expected.encoding is required");
  }
  if (config_.expected_bit_depth <= 0) {
    throw std::runtime_error("expected.bit_depth must be > 0");
  }
  if ((config_.expected_bit_depth % 8) != 0) {
    throw std::runtime_error("expected.bit_depth must be byte-aligned");
  }
  if (config_.expected_layout.empty()) {
    throw std::runtime_error("expected.layout is required");
  }
  if (!std::isfinite(config_.alignment_period_ms) || config_.alignment_period_ms <= 0.0) {
    throw std::runtime_error("alignment.period_ms must be finite and > 0");
  }
  if (!std::isfinite(config_.alignment_phase_ms) || config_.alignment_phase_ms < 0.0 ||
      config_.alignment_phase_ms >= config_.alignment_period_ms)
  {
    throw std::runtime_error("alignment.phase_ms must be finite, >= 0, and < alignment.period_ms");
  }
  if (!std::isfinite(config_.alignment_max_adjust_ms) || config_.alignment_max_adjust_ms < 0.0) {
    throw std::runtime_error("alignment.max_adjust_ms must be finite and >= 0");
  }
  if (config_.qos_depth <= 0) {
    throw std::runtime_error("qos.depth must be > 0");
  }
  if (config_.diagnostics_publish_period_ms <= 0) {
    throw std::runtime_error("diagnostics.publish_period_ms must be > 0");
  }

  RCLCPP_INFO(
    this->get_logger(),
    "Time alignment config: input=%s output=%s expected=%dHz/%d/%s/%d/%s period=%.6fms "
    "phase=%.6fms max_adjust=%.6fms qos_depth=%d reliable=%s diag=%dms",
    config_.input_topic.c_str(),
    config_.output_topic.c_str(),
    config_.expected_sample_rate,
    config_.expected_channels,
    config_.expected_encoding.c_str(),
    config_.expected_bit_depth,
    config_.expected_layout.c_str(),
    config_.alignment_period_ms,
    config_.alignment_phase_ms,
    config_.alignment_max_adjust_ms,
    config_.qos_depth,
    config_.qos_reliable ? "true" : "false",
    config_.diagnostics_publish_period_ms);
}

void FaTimeAlignmentNode::setupInterfaces()
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
    std::bind(&FaTimeAlignmentNode::handleFrame, this, std::placeholders::_1));

  diag_pub_ = this->create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
    "diagnostics",
    rclcpp::SystemDefaultsQoS());
  diag_timer_ = this->create_wall_timer(
    std::chrono::milliseconds(config_.diagnostics_publish_period_ms),
    std::bind(&FaTimeAlignmentNode::publishDiagnostics, this));
}

void FaTimeAlignmentNode::handleFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg)
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
  if (!alignFrame(*msg, out)) {
    frames_dropped_.fetch_add(1);
    return;
  }

  audio_pub_->publish(out);
  frames_out_.fetch_add(1);
}

bool FaTimeAlignmentNode::validateFrame(const fa_interfaces::msg::AudioFrame & msg)
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

  const size_t bytes_per_sample = static_cast<size_t>(config_.expected_bit_depth) / 8U;
  const size_t bytes_per_frame = static_cast<size_t>(config_.expected_channels) * bytes_per_sample;
  if (bytes_per_sample == 0U || bytes_per_frame == 0U ||
      msg.data.empty() || (msg.data.size() % bytes_per_frame) != 0U)
  {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "AudioFrame data size is invalid for expected frame byte alignment");
    return false;
  }
  return true;
}

bool FaTimeAlignmentNode::alignFrame(
  const fa_interfaces::msg::AudioFrame & in,
  fa_interfaces::msg::AudioFrame & out)
{
  const long double input_ns = stampToNanoseconds(in.header.stamp);
  const long double period_ns =
    static_cast<long double>(config_.alignment_period_ms) * kNanosecondsPerMillisecond;
  const long double phase_ns =
    static_cast<long double>(config_.alignment_phase_ms) * kNanosecondsPerMillisecond;
  const long double max_adjust_ns =
    static_cast<long double>(config_.alignment_max_adjust_ms) * kNanosecondsPerMillisecond;
  const long double grid_index = std::round((input_ns - phase_ns) / period_ns);
  const long double aligned_ns_decimal = phase_ns + (grid_index * period_ns);

  if (!std::isfinite(static_cast<double>(aligned_ns_decimal))) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping frame because computed aligned timestamp is not finite");
    return false;
  }
  if (aligned_ns_decimal < 0.0L) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping frame because computed aligned timestamp is negative");
    return false;
  }
  if (aligned_ns_decimal > static_cast<long double>(kMaxBuiltinTimeNanoseconds)) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping frame because computed aligned timestamp exceeds builtin_interfaces/Time range");
    return false;
  }

  const long double adjustment_ns = aligned_ns_decimal - input_ns;
  if (std::fabs(adjustment_ns) > max_adjust_ns) {
    frames_excess_adjust_.fetch_add(1);
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping frame because timestamp adjustment %.3Lf ns exceeds max %.3Lf ns",
      adjustment_ns,
      max_adjust_ns);
    return false;
  }

  const int64_t aligned_ns = static_cast<int64_t>(std::llround(aligned_ns_decimal));
  out = in;
  out.header.stamp = nanosecondsToStamp(aligned_ns);
  out.stream_id = config_.output_topic;
  frames_aligned_.fetch_add(1);
  return true;
}

void FaTimeAlignmentNode::publishDiagnostics()
{
  diagnostic_msgs::msg::DiagnosticArray array_msg;
  array_msg.header.stamp = this->now();

  diagnostic_msgs::msg::DiagnosticStatus status;
  status.name = "fa_time_alignment";
  status.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
  status.message = "running";
  status.values.reserve(12);
  pushKeyValue(status, "input_topic", config_.input_topic);
  pushKeyValue(status, "output_topic", config_.output_topic);
  pushKeyValue(status, "alignment_period_ms", std::to_string(config_.alignment_period_ms));
  pushKeyValue(status, "alignment_phase_ms", std::to_string(config_.alignment_phase_ms));
  pushKeyValue(status, "alignment_max_adjust_ms", std::to_string(config_.alignment_max_adjust_ms));
  pushKeyValue(status, "frames_in", std::to_string(frames_in_.load()));
  pushKeyValue(status, "frames_out", std::to_string(frames_out_.load()));
  pushKeyValue(status, "frames_dropped", std::to_string(frames_dropped_.load()));
  pushKeyValue(status, "frames_aligned", std::to_string(frames_aligned_.load()));
  pushKeyValue(status, "frames_excess_adjust", std::to_string(frames_excess_adjust_.load()));
  pushKeyValue(status, "backend.name", "no_runtime_backend");

  array_msg.status.push_back(status);
  diag_pub_->publish(array_msg);
}

}  // namespace fa_time_alignment

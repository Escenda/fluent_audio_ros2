#include "fa_eq/fa_eq_node.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

#include "diagnostic_msgs/msg/diagnostic_status.hpp"
#include "diagnostic_msgs/msg/key_value.hpp"
#include "fa_eq/backends/internal_three_band_eq.hpp"

namespace fa_eq
{

namespace
{
constexpr const char * kEncodingFloat32 = "FLOAT32LE";
constexpr const char * kInterleavedLayout = "interleaved";

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

FaEqNode::FaEqNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("fa_eq", options)
{
  RCLCPP_INFO(this->get_logger(), "Starting FA EQ node");
  loadParameters();
  configureBackend();
  setupInterfaces();
}

FaEqNode::~FaEqNode() = default;

void FaEqNode::loadParameters()
{
  this->declare_parameter<std::string>("input_topic");
  this->declare_parameter<std::string>("output_topic");
  this->declare_parameter<double>("low.cutoff_hz");
  this->declare_parameter<double>("high.cutoff_hz");
  this->declare_parameter<double>("gains.low_db");
  this->declare_parameter<double>("gains.mid_db");
  this->declare_parameter<double>("gains.high_db");
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
  config_.low_cutoff_hz = readRequiredDouble(*this, "low.cutoff_hz");
  config_.high_cutoff_hz = readRequiredDouble(*this, "high.cutoff_hz");
  config_.gain_low_db = readRequiredDouble(*this, "gains.low_db");
  config_.gain_mid_db = readRequiredDouble(*this, "gains.mid_db");
  config_.gain_high_db = readRequiredDouble(*this, "gains.high_db");
  config_.expected_sample_rate = readRequiredInt(*this, "expected.sample_rate");
  config_.expected_channels = readRequiredInt(*this, "expected.channels");
  config_.expected_encoding = readRequiredString(*this, "expected.encoding");
  config_.expected_bit_depth = readRequiredInt(*this, "expected.bit_depth");
  config_.expected_layout = readRequiredString(*this, "expected.layout");
  config_.qos_depth = readRequiredInt(*this, "qos.depth");
  config_.qos_reliable = readRequiredBool(*this, "qos.reliable");
  config_.diagnostics_publish_period_ms = readRequiredInt(
    *this, "diagnostics.publish_period_ms");

  if (config_.input_topic.empty()) {
    throw std::runtime_error("input_topic is required");
  }
  if (config_.output_topic.empty()) {
    throw std::runtime_error("output_topic is required");
  }
  if (config_.expected_sample_rate <= 0) {
    throw std::runtime_error("expected.sample_rate must be > 0");
  }
  const double nyquist_hz = static_cast<double>(config_.expected_sample_rate) / 2.0;
  if (!isFinite(config_.low_cutoff_hz) || config_.low_cutoff_hz <= 0.0) {
    throw std::runtime_error("low.cutoff_hz must be finite and > 0.0");
  }
  if (!isFinite(config_.high_cutoff_hz) ||
      config_.high_cutoff_hz <= config_.low_cutoff_hz ||
      config_.high_cutoff_hz >= nyquist_hz)
  {
    throw std::runtime_error(
      "high.cutoff_hz must be finite, > low.cutoff_hz, and < expected.sample_rate / 2.0");
  }
  if (config_.expected_channels <= 0) {
    throw std::runtime_error("expected.channels must be > 0");
  }
  if (config_.expected_encoding != kEncodingFloat32) {
    throw std::runtime_error("fa_eq requires expected.encoding=FLOAT32LE");
  }
  if (config_.expected_bit_depth != 32) {
    throw std::runtime_error("fa_eq requires expected.bit_depth=32");
  }
  if (config_.expected_layout != kInterleavedLayout) {
    throw std::runtime_error("fa_eq requires expected.layout=interleaved");
  }
  if (config_.qos_depth <= 0) {
    throw std::runtime_error("qos.depth must be > 0");
  }
  if (config_.diagnostics_publish_period_ms <= 0) {
    throw std::runtime_error("diagnostics.publish_period_ms must be > 0");
  }
  if (!isFinite(config_.gain_low_db) ||
      !isFinite(config_.gain_mid_db) ||
      !isFinite(config_.gain_high_db))
  {
    throw std::runtime_error("gains.*_db must be finite");
  }

  RCLCPP_INFO(
    this->get_logger(),
    "EQ config: input=%s output=%s low_cutoff=%fHz high_cutoff=%fHz gains_db=%f/%f/%f expected=%dHz/%d/%s/%d qos_depth=%d reliable=%s diag=%dms",
    config_.input_topic.c_str(),
    config_.output_topic.c_str(),
    config_.low_cutoff_hz,
    config_.high_cutoff_hz,
    config_.gain_low_db,
    config_.gain_mid_db,
    config_.gain_high_db,
    config_.expected_sample_rate,
    config_.expected_channels,
    config_.expected_encoding.c_str(),
    config_.expected_bit_depth,
    config_.qos_depth,
    config_.qos_reliable ? "true" : "false",
    config_.diagnostics_publish_period_ms);
}

void FaEqNode::configureBackend()
{
  backend_ = createBackend();
}

std::unique_ptr<backends::InternalThreeBandEqBackend> FaEqNode::createBackend() const
{
  return std::make_unique<backends::InternalThreeBandEqBackend>(
    backends::InternalThreeBandEqConfig{
      config_.expected_sample_rate,
      config_.expected_channels,
      config_.low_cutoff_hz,
      config_.high_cutoff_hz,
      config_.gain_low_db,
      config_.gain_mid_db,
      config_.gain_high_db});
}

void FaEqNode::setupInterfaces()
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
    std::bind(&FaEqNode::handleFrame, this, std::placeholders::_1));

  diag_pub_ = this->create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
    "diagnostics",
    rclcpp::SystemDefaultsQoS());
  diag_timer_ = this->create_wall_timer(
    std::chrono::milliseconds(config_.diagnostics_publish_period_ms),
    std::bind(&FaEqNode::publishDiagnostics, this));
}

void FaEqNode::handleFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg)
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
  if (!applyEq(*msg, out)) {
    frames_dropped_.fetch_add(1);
    return;
  }

  audio_pub_->publish(out);
  frames_out_.fetch_add(1);
}

bool FaEqNode::validateFrame(const fa_interfaces::msg::AudioFrame & msg)
{
  if (msg.source_id.empty() || msg.stream_id.empty()) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "AudioFrame source_id and stream_id are required");
    return false;
  }
  if (!active_source_id_.empty() && msg.source_id != active_source_id_) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "AudioFrame source_id mismatch: %s != %s",
      msg.source_id.c_str(),
      active_source_id_.c_str());
    return false;
  }
  if (last_epoch_.has_value() && msg.epoch <= *last_epoch_) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping non-monotonic AudioFrame epoch %u after epoch %u",
      msg.epoch,
      *last_epoch_);
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
      "AudioFrame data size is invalid for FLOAT32LE interleaved samples");
    return false;
  }
  return true;
}

bool FaEqNode::applyEq(
  const fa_interfaces::msg::AudioFrame & in,
  fa_interfaces::msg::AudioFrame & out)
{
  const bool should_bind_source = active_source_id_.empty();
  const bool should_reset_state =
    last_epoch_.has_value() && in.epoch != (*last_epoch_ + 1U);

  out = in;
  out.stream_id = config_.output_topic;
  const backends::ProcessStatus status = backend_->process(in.data, out.data, should_reset_state);
  if (status != backends::ProcessStatus::kOk) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping frame because EQ backend rejected input or output: %s",
      backends::processStatusMessage(status));
    return false;
  }

  if (should_bind_source) {
    active_source_id_ = in.source_id;
  }
  if (should_reset_state) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "AudioFrame epoch jumped from %u to %u; processing with fresh EQ state",
      *last_epoch_,
      in.epoch);
    state_resets_.fetch_add(1);
  }
  last_epoch_ = in.epoch;
  return true;
}

void FaEqNode::publishDiagnostics()
{
  diagnostic_msgs::msg::DiagnosticArray array_msg;
  array_msg.header.stamp = this->now();

  diagnostic_msgs::msg::DiagnosticStatus status;
  status.name = "fa_eq";
  status.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
  status.message = "running";
  status.values.reserve(15);
  pushKeyValue(status, "low_cutoff_hz", std::to_string(config_.low_cutoff_hz));
  pushKeyValue(status, "high_cutoff_hz", std::to_string(config_.high_cutoff_hz));
  pushKeyValue(status, "low_alpha", std::to_string(backend_->lowAlpha()));
  pushKeyValue(status, "high_alpha", std::to_string(backend_->highAlpha()));
  pushKeyValue(status, "gain_low_db", std::to_string(config_.gain_low_db));
  pushKeyValue(status, "gain_mid_db", std::to_string(config_.gain_mid_db));
  pushKeyValue(status, "gain_high_db", std::to_string(config_.gain_high_db));
  pushKeyValue(status, "gain_low_linear", std::to_string(backend_->gainLowLinear()));
  pushKeyValue(status, "gain_mid_linear", std::to_string(backend_->gainMidLinear()));
  pushKeyValue(status, "gain_high_linear", std::to_string(backend_->gainHighLinear()));
  pushKeyValue(status, "state_source_id", active_source_id_);
  pushKeyValue(status, "state_resets", std::to_string(state_resets_.load()));
  pushKeyValue(status, "frames_in", std::to_string(frames_in_.load()));
  pushKeyValue(status, "frames_out", std::to_string(frames_out_.load()));
  pushKeyValue(status, "frames_dropped", std::to_string(frames_dropped_.load()));
  pushKeyValue(status, "output_topic", config_.output_topic);

  array_msg.status.push_back(status);
  diag_pub_->publish(array_msg);
}

}  // namespace fa_eq

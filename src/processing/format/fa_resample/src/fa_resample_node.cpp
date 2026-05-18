#include "fa_resample/fa_resample_node.hpp"

#include "fa_resample/backends/internal_linear_resampler.hpp"

#include <chrono>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "diagnostic_msgs/msg/diagnostic_status.hpp"
#include "diagnostic_msgs/msg/key_value.hpp"

namespace fa_resample
{

namespace
{
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

FaResampleNode::FaResampleNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("fa_resample", options)
{
  RCLCPP_INFO(this->get_logger(), "Starting FA Resample node");
  loadParameters();
  configureBackend();
  setupInterfaces();
}

FaResampleNode::~FaResampleNode() = default;

void FaResampleNode::loadParameters()
{
  this->declare_parameter<int>("target_sample_rate");
  this->declare_parameter<std::string>("input.encoding");
  this->declare_parameter<int>("input.bit_depth");
  this->declare_parameter<std::string>("input.layout");
  this->declare_parameter<std::string>("output.encoding");
  this->declare_parameter<int>("output.bit_depth");

  this->declare_parameter<bool>("mic.enabled");
  this->declare_parameter<std::string>("mic.input_topic");
  this->declare_parameter<std::string>("mic.output_topic");
  this->declare_parameter<std::string>("mic.input_stream_id");
  this->declare_parameter<std::string>("mic.output.stream_id");

  this->declare_parameter<bool>("ref.enabled");
  this->declare_parameter<std::string>("ref.input_topic");
  this->declare_parameter<std::string>("ref.output_topic");
  this->declare_parameter<std::string>("ref.input_stream_id");
  this->declare_parameter<std::string>("ref.output.stream_id");

  this->declare_parameter<int>("qos.depth");
  this->declare_parameter<bool>("qos.reliable");

  this->declare_parameter<int>("diagnostics.publish_period_ms");
  this->declare_parameter<int>("diagnostics.qos.depth");
  this->declare_parameter<bool>("diagnostics.qos.reliable");

  config_.target_sample_rate = readRequiredInt(*this, "target_sample_rate");
  config_.input_encoding = readRequiredString(*this, "input.encoding");
  config_.input_bit_depth = readRequiredInt(*this, "input.bit_depth");
  config_.input_layout = readRequiredString(*this, "input.layout");
  config_.output_encoding = readRequiredString(*this, "output.encoding");
  config_.output_bit_depth = readRequiredInt(*this, "output.bit_depth");

  config_.mic_enabled = readRequiredBool(*this, "mic.enabled");
  config_.mic_input_topic = readRequiredString(*this, "mic.input_topic");
  config_.mic_output_topic = readRequiredString(*this, "mic.output_topic");
  config_.mic_input_stream_id = readRequiredString(*this, "mic.input_stream_id");
  config_.mic_output_stream_id = readRequiredString(*this, "mic.output.stream_id");

  config_.ref_enabled = readRequiredBool(*this, "ref.enabled");
  config_.ref_input_topic = readRequiredString(*this, "ref.input_topic");
  config_.ref_output_topic = readRequiredString(*this, "ref.output_topic");
  config_.ref_input_stream_id = readRequiredString(*this, "ref.input_stream_id");
  config_.ref_output_stream_id = readRequiredString(*this, "ref.output.stream_id");

  config_.qos_depth = readRequiredInt(*this, "qos.depth");
  config_.qos_reliable = readRequiredBool(*this, "qos.reliable");

  config_.diagnostics_publish_period_ms = readRequiredInt(
    *this, "diagnostics.publish_period_ms");
  config_.diagnostics_qos_depth = readRequiredInt(*this, "diagnostics.qos.depth");
  config_.diagnostics_qos_reliable = readRequiredBool(*this, "diagnostics.qos.reliable");

  if (config_.target_sample_rate <= 0) {
    throw std::runtime_error("target_sample_rate must be > 0 (set via YAML)");
  }
  if (config_.input_encoding != backends::kEncodingFloat32Le) {
    throw std::runtime_error("fa_resample input.encoding must be FLOAT32LE");
  }
  if (config_.input_bit_depth != 32) {
    throw std::runtime_error("fa_resample input.bit_depth must be 32");
  }
  if (config_.input_layout != backends::kInterleavedLayout) {
    throw std::runtime_error("fa_resample input.layout must be interleaved");
  }
  if (config_.output_encoding != backends::kEncodingFloat32Le) {
    throw std::runtime_error("fa_resample output.encoding must be FLOAT32LE");
  }
  if (config_.output_bit_depth != 32) {
    throw std::runtime_error("fa_resample output.bit_depth must be 32");
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
  if (config_.mic_enabled) {
    if (config_.mic_input_topic.empty()) {
      throw std::runtime_error("mic.input_topic is required when mic.enabled=true");
    }
    if (config_.mic_output_topic.empty()) {
      throw std::runtime_error("mic.output_topic is required when mic.enabled=true");
    }
    if (config_.mic_input_stream_id.empty()) {
      throw std::runtime_error("mic.input_stream_id is required when mic.enabled=true");
    }
    if (config_.mic_output_stream_id.empty()) {
      throw std::runtime_error("mic.output.stream_id is required when mic.enabled=true");
    }
    const std::string resolved_input_topic =
      this->get_node_topics_interface()->resolve_topic_name(config_.mic_input_topic);
    const std::string resolved_output_topic =
      this->get_node_topics_interface()->resolve_topic_name(config_.mic_output_topic);
    if (sameIdentityString(config_.mic_input_stream_id, config_.mic_input_topic) ||
        sameIdentityString(config_.mic_input_stream_id, config_.mic_output_topic) ||
        sameIdentityString(config_.mic_input_stream_id, resolved_input_topic) ||
        sameIdentityString(config_.mic_input_stream_id, resolved_output_topic)) {
      throw std::runtime_error("mic.input_stream_id must be distinct from ROS topics");
    }
    if (sameIdentityString(config_.mic_output_stream_id, config_.mic_input_topic) ||
        sameIdentityString(config_.mic_output_stream_id, config_.mic_output_topic) ||
        sameIdentityString(config_.mic_output_stream_id, resolved_input_topic) ||
        sameIdentityString(config_.mic_output_stream_id, resolved_output_topic)) {
      throw std::runtime_error("mic.output.stream_id must be distinct from ROS topics");
    }
    if (sameIdentityString(config_.mic_input_stream_id, config_.mic_output_stream_id)) {
      throw std::runtime_error("mic.input_stream_id and mic.output.stream_id must be distinct");
    }
  }
  if (config_.ref_enabled) {
    if (config_.ref_input_topic.empty()) {
      throw std::runtime_error("ref.input_topic is required when ref.enabled=true");
    }
    if (config_.ref_output_topic.empty()) {
      throw std::runtime_error("ref.output_topic is required when ref.enabled=true");
    }
    if (config_.ref_input_stream_id.empty()) {
      throw std::runtime_error("ref.input_stream_id is required when ref.enabled=true");
    }
    if (config_.ref_output_stream_id.empty()) {
      throw std::runtime_error("ref.output.stream_id is required when ref.enabled=true");
    }
    const std::string resolved_input_topic =
      this->get_node_topics_interface()->resolve_topic_name(config_.ref_input_topic);
    const std::string resolved_output_topic =
      this->get_node_topics_interface()->resolve_topic_name(config_.ref_output_topic);
    if (sameIdentityString(config_.ref_input_stream_id, config_.ref_input_topic) ||
        sameIdentityString(config_.ref_input_stream_id, config_.ref_output_topic) ||
        sameIdentityString(config_.ref_input_stream_id, resolved_input_topic) ||
        sameIdentityString(config_.ref_input_stream_id, resolved_output_topic)) {
      throw std::runtime_error("ref.input_stream_id must be distinct from ROS topics");
    }
    if (sameIdentityString(config_.ref_output_stream_id, config_.ref_input_topic) ||
        sameIdentityString(config_.ref_output_stream_id, config_.ref_output_topic) ||
        sameIdentityString(config_.ref_output_stream_id, resolved_input_topic) ||
        sameIdentityString(config_.ref_output_stream_id, resolved_output_topic)) {
      throw std::runtime_error("ref.output.stream_id must be distinct from ROS topics");
    }
    if (sameIdentityString(config_.ref_input_stream_id, config_.ref_output_stream_id)) {
      throw std::runtime_error("ref.input_stream_id and ref.output.stream_id must be distinct");
    }
  }

  RCLCPP_INFO(this->get_logger(),
    "Resample config: target_sr=%dHz input=%s/%d/%s output=%s/%d qos_depth=%d reliable=%s "
    "mic=%s (%s/%s -> %s/%s) ref=%s (%s/%s -> %s/%s) diag=%dms",
    config_.target_sample_rate,
    config_.input_encoding.c_str(), config_.input_bit_depth, config_.input_layout.c_str(),
    config_.output_encoding.c_str(), config_.output_bit_depth,
    config_.qos_depth, config_.qos_reliable ? "true" : "false",
    config_.mic_enabled ? "on" : "off",
    config_.mic_input_topic.c_str(), config_.mic_input_stream_id.c_str(),
    config_.mic_output_topic.c_str(), config_.mic_output_stream_id.c_str(),
    config_.ref_enabled ? "on" : "off",
    config_.ref_input_topic.c_str(), config_.ref_input_stream_id.c_str(),
    config_.ref_output_topic.c_str(), config_.ref_output_stream_id.c_str(),
    config_.diagnostics_publish_period_ms);
}

void FaResampleNode::configureBackend()
{
  backend_ = std::make_unique<backends::InternalLinearResamplerBackend>(
    backends::InternalLinearResamplerConfig{config_.target_sample_rate});
}

void FaResampleNode::setupInterfaces()
{
  rclcpp::QoS qos(static_cast<size_t>(config_.qos_depth));
  if (config_.qos_reliable) {
    qos.reliable();
  } else {
    qos.best_effort();
  }

  if (config_.mic_enabled) {
    mic_pub_ = this->create_publisher<fa_interfaces::msg::AudioFrame>(config_.mic_output_topic, qos);
    mic_sub_ = this->create_subscription<fa_interfaces::msg::AudioFrame>(
      config_.mic_input_topic, qos,
      std::bind(&FaResampleNode::handleMicFrame, this, std::placeholders::_1));
  }
  if (config_.ref_enabled) {
    ref_pub_ = this->create_publisher<fa_interfaces::msg::AudioFrame>(config_.ref_output_topic, qos);
    ref_sub_ = this->create_subscription<fa_interfaces::msg::AudioFrame>(
      config_.ref_input_topic, qos,
      std::bind(&FaResampleNode::handleRefFrame, this, std::placeholders::_1));
  }

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
    std::bind(&FaResampleNode::publishDiagnostics, this));
}

void FaResampleNode::handleMicFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg)
{
  mic_in_.fetch_add(1);
  if (!msg || !mic_pub_) {
    mic_drop_.fetch_add(1);
    return;
  }
  processAndPublish(
    *msg,
    mic_pub_,
    config_.mic_input_stream_id,
    config_.mic_output_stream_id,
    mic_out_,
    mic_drop_);
}

void FaResampleNode::handleRefFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg)
{
  ref_in_.fetch_add(1);
  if (!msg || !ref_pub_) {
    ref_drop_.fetch_add(1);
    return;
  }
  processAndPublish(
    *msg,
    ref_pub_,
    config_.ref_input_stream_id,
    config_.ref_output_stream_id,
    ref_out_,
    ref_drop_);
}

bool FaResampleNode::processAndPublish(
  const fa_interfaces::msg::AudioFrame & in,
  const rclcpp::Publisher<fa_interfaces::msg::AudioFrame>::SharedPtr & pub,
  const std::string & expected_input_stream_id,
  const std::string & output_stream_id,
  std::atomic<uint64_t> & out_counter,
  std::atomic<uint64_t> & drop_counter)
{
  if (!pub) {
    drop_counter.fetch_add(1);
    return false;
  }

  if (in.source_id.empty() || in.stream_id.empty()) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Invalid frame (%s): source_id and stream_id are required", output_stream_id.c_str());
    drop_counter.fetch_add(1);
    return false;
  }
  if (in.stream_id != expected_input_stream_id) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Invalid frame (%s): stream_id mismatch %s != %s",
      output_stream_id.c_str(),
      in.stream_id.c_str(),
      expected_input_stream_id.c_str());
    drop_counter.fetch_add(1);
    return false;
  }

  const backends::FrameContract frame_contract{
    in.encoding,
    in.sample_rate,
    in.channels,
    in.bit_depth,
    in.layout,
    in.data.size()};
  std::vector<uint8_t> out_bytes;
  const backends::ProcessResult result = backend_->process(in.data, frame_contract, out_bytes);
  if (result.status != backends::ProcessStatus::kOk) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Resample backend rejected frame (%s): %s (%s)",
      output_stream_id.c_str(),
      backends::processStatusMessage(result.status),
      backends::frameContractStatusName(result.frame_contract_status));
    drop_counter.fetch_add(1);
    return false;
  }

  fa_interfaces::msg::AudioFrame out;
  out.header = in.header;
  out.source_id = in.source_id;
  out.stream_id = output_stream_id;
  out.encoding = config_.output_encoding;
  out.sample_rate = static_cast<uint32_t>(backend_->targetSampleRate());
  out.channels = in.channels;
  out.bit_depth = static_cast<uint32_t>(config_.output_bit_depth);
  out.layout = backends::kInterleavedLayout;
  out.data = out_bytes;
  out.epoch = in.epoch;

  pub->publish(out);
  out_counter.fetch_add(1);
  return true;
}

void FaResampleNode::publishDiagnostics()
{
  diagnostic_msgs::msg::DiagnosticArray array_msg;
  array_msg.header.stamp = this->now();

  diagnostic_msgs::msg::DiagnosticStatus status;
  status.name = "fa_resample";
  status.hardware_id = "dsp";
  status.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
  status.message = "running";

  auto push_kv = [&status](const std::string & key, const std::string & value) {
      diagnostic_msgs::msg::KeyValue kv;
      kv.key = key;
      kv.value = value;
      status.values.push_back(kv);
    };

  status.values.reserve(16);
  push_kv("target_sample_rate", std::to_string(backend_->targetSampleRate()));
  push_kv("output.encoding", config_.output_encoding);
  push_kv("output.bit_depth", std::to_string(config_.output_bit_depth));
  push_kv("mic.input_stream_id", config_.mic_input_stream_id);
  push_kv("mic.output_stream_id", config_.mic_output_stream_id);
  push_kv("ref.input_stream_id", config_.ref_input_stream_id);
  push_kv("ref.output_stream_id", config_.ref_output_stream_id);
  push_kv("mic.in", std::to_string(mic_in_.load()));
  push_kv("mic.out", std::to_string(mic_out_.load()));
  push_kv("mic.drop", std::to_string(mic_drop_.load()));
  push_kv("ref.in", std::to_string(ref_in_.load()));
  push_kv("ref.out", std::to_string(ref_out_.load()));
  push_kv("ref.drop", std::to_string(ref_drop_.load()));

  array_msg.status.push_back(status);
  diag_pub_->publish(array_msg);
}

}  // namespace fa_resample

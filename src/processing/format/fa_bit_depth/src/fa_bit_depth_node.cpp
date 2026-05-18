#include "fa_bit_depth/fa_bit_depth_node.hpp"

#include <chrono>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "diagnostic_msgs/msg/diagnostic_status.hpp"
#include "diagnostic_msgs/msg/key_value.hpp"

namespace fa_bit_depth
{

namespace
{
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

backends::FrameContract frameContractFrom(const fa_interfaces::msg::AudioFrame & msg)
{
  return backends::FrameContract{
    msg.encoding,
    msg.bit_depth,
    msg.sample_rate,
    msg.channels,
    msg.layout,
    msg.data.size()};
}
}  // namespace

FaBitDepthNode::FaBitDepthNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("fa_bit_depth", options)
{
  RCLCPP_INFO(this->get_logger(), "Starting FA Bit Depth node");
  loadParameters();
  setupInterfaces();
}

void FaBitDepthNode::loadParameters()
{
  this->declare_parameter("input_topic", config_.input_topic);
  this->declare_parameter("output_topic", config_.output_topic);
  this->declare_parameter("input.encoding", config_.input_encoding);
  this->declare_parameter<int>("input.bit_depth", config_.input_bit_depth);
  this->declare_parameter("output.encoding", config_.output_encoding);
  this->declare_parameter<int>("output.bit_depth", config_.output_bit_depth);
  this->declare_parameter<int>("expected.sample_rate", config_.expected_sample_rate);
  this->declare_parameter<int>("expected.channels", config_.expected_channels);
  this->declare_parameter("expected.layout", config_.expected_layout);
  this->declare_parameter<int>("qos.depth", config_.qos_depth);
  this->declare_parameter<bool>("qos.reliable", config_.qos_reliable);
  this->declare_parameter<int>(
    "diagnostics.publish_period_ms",
    config_.diagnostics_publish_period_ms);

  config_.input_topic = this->get_parameter("input_topic").as_string();
  config_.output_topic = this->get_parameter("output_topic").as_string();
  config_.input_encoding = this->get_parameter("input.encoding").as_string();
  config_.input_bit_depth = this->get_parameter("input.bit_depth").as_int();
  config_.output_encoding = this->get_parameter("output.encoding").as_string();
  config_.output_bit_depth = this->get_parameter("output.bit_depth").as_int();
  config_.expected_sample_rate = this->get_parameter("expected.sample_rate").as_int();
  config_.expected_channels = this->get_parameter("expected.channels").as_int();
  config_.expected_layout = this->get_parameter("expected.layout").as_string();
  config_.qos_depth = this->get_parameter("qos.depth").as_int();
  config_.qos_reliable = this->get_parameter("qos.reliable").as_bool();
  config_.diagnostics_publish_period_ms =
    this->get_parameter("diagnostics.publish_period_ms").as_int();

  if (config_.input_topic.empty()) {
    throw std::runtime_error("input_topic is required");
  }
  if (config_.output_topic.empty()) {
    throw std::runtime_error("output_topic is required");
  }
  if (config_.qos_depth <= 0) {
    throw std::runtime_error("qos.depth must be > 0");
  }
  if (config_.diagnostics_publish_period_ms <= 0) {
    throw std::runtime_error("diagnostics.publish_period_ms must be > 0");
  }

  backend_ = std::make_unique<backends::InternalIntegerBitDepthBackend>(
    backends::InternalIntegerBitDepthConfig{
      config_.input_encoding,
      config_.input_bit_depth,
      config_.output_encoding,
      config_.output_bit_depth,
      config_.expected_sample_rate,
      config_.expected_channels,
      config_.expected_layout});

  RCLCPP_INFO(
    this->get_logger(),
    "Bit depth config: input=%s output=%s conversion=%s/%d -> %s/%d expected=%dHz/%d/%s "
    "qos_depth=%d reliable=%s diag=%dms",
    config_.input_topic.c_str(),
    config_.output_topic.c_str(),
    config_.input_encoding.c_str(),
    config_.input_bit_depth,
    config_.output_encoding.c_str(),
    config_.output_bit_depth,
    config_.expected_sample_rate,
    config_.expected_channels,
    config_.expected_layout.c_str(),
    config_.qos_depth,
    config_.qos_reliable ? "true" : "false",
    config_.diagnostics_publish_period_ms);
}

void FaBitDepthNode::setupInterfaces()
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
    std::bind(&FaBitDepthNode::handleFrame, this, std::placeholders::_1));

  diag_pub_ = this->create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
    "diagnostics",
    rclcpp::SystemDefaultsQoS());
  diag_timer_ = this->create_wall_timer(
    std::chrono::milliseconds(config_.diagnostics_publish_period_ms),
    std::bind(&FaBitDepthNode::publishDiagnostics, this));
}

void FaBitDepthNode::handleFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg)
{
  frames_in_.fetch_add(1);
  if (!msg) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping null AudioFrame");
    frames_dropped_.fetch_add(1);
    return;
  }

  if (!validateFrame(*msg)) {
    frames_dropped_.fetch_add(1);
    return;
  }

  fa_interfaces::msg::AudioFrame out;
  if (!convertFrame(*msg, out)) {
    frames_dropped_.fetch_add(1);
    return;
  }

  audio_pub_->publish(out);
  frames_out_.fetch_add(1);
}

bool FaBitDepthNode::validateFrame(const fa_interfaces::msg::AudioFrame & msg)
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

  const backends::FrameContractStatus contract_status =
    backend_->validateContract(frameContractFrom(msg));
  if (contract_status != backends::FrameContractStatus::kOk) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "AudioFrame bit-depth contract mismatch: %s",
      backends::frameContractStatusName(contract_status));
    return false;
  }

  return true;
}

bool FaBitDepthNode::convertFrame(
  const fa_interfaces::msg::AudioFrame & in,
  fa_interfaces::msg::AudioFrame & out)
{
  std::vector<uint8_t> output_data;
  const backends::ProcessResult result =
    backend_->process(in.data, frameContractFrom(in), output_data);
  if (result.status != backends::ProcessStatus::kOk) {
    RCLCPP_ERROR_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping frame because bit-depth backend failed: %s (%s)",
      backends::processStatusMessage(result.status),
      backends::frameContractStatusName(result.frame_contract_status));
    return false;
  }

  out = in;
  out.stream_id = config_.output_topic;
  out.encoding = backend_->outputEncoding();
  out.bit_depth = static_cast<uint32_t>(backend_->outputBitDepth());
  out.sample_rate = in.sample_rate;
  out.channels = in.channels;
  out.layout = in.layout;
  out.data = output_data;
  return true;
}

void FaBitDepthNode::publishDiagnostics()
{
  diagnostic_msgs::msg::DiagnosticArray array_msg;
  array_msg.header.stamp = this->now();

  diagnostic_msgs::msg::DiagnosticStatus status;
  status.name = "fa_bit_depth";
  status.hardware_id = "dsp";
  status.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
  status.message = "running";

  status.values.reserve(10);
  pushKeyValue(status, "input_topic", config_.input_topic);
  pushKeyValue(status, "output_topic", config_.output_topic);
  pushKeyValue(status, "input.encoding", config_.input_encoding);
  pushKeyValue(status, "input.bit_depth", std::to_string(config_.input_bit_depth));
  pushKeyValue(status, "output.encoding", config_.output_encoding);
  pushKeyValue(status, "output.bit_depth", std::to_string(config_.output_bit_depth));
  pushKeyValue(status, "frames.in", std::to_string(frames_in_.load()));
  pushKeyValue(status, "frames.out", std::to_string(frames_out_.load()));
  pushKeyValue(status, "frames.drop", std::to_string(frames_dropped_.load()));

  array_msg.status.push_back(status);
  diag_pub_->publish(array_msg);
}

}  // namespace fa_bit_depth

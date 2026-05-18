#include "fa_resample/fa_resample_node.hpp"

#include "fa_resample/backends/internal_linear_resampler.hpp"

#include <chrono>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "diagnostic_msgs/msg/diagnostic_status.hpp"
#include "diagnostic_msgs/msg/key_value.hpp"

namespace fa_resample
{

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
  this->declare_parameter<int>("target_sample_rate", config_.target_sample_rate);
  this->declare_parameter("input.encoding", config_.input_encoding);
  this->declare_parameter<int>("input.bit_depth", config_.input_bit_depth);
  this->declare_parameter("input.layout", config_.input_layout);
  this->declare_parameter("output.encoding", config_.output_encoding);
  this->declare_parameter<int>("output.bit_depth", config_.output_bit_depth);

  this->declare_parameter<bool>("mic.enabled", config_.mic_enabled);
  this->declare_parameter("mic.input_topic", config_.mic_input_topic);
  this->declare_parameter("mic.output_topic", config_.mic_output_topic);

  this->declare_parameter<bool>("ref.enabled", config_.ref_enabled);
  this->declare_parameter("ref.input_topic", config_.ref_input_topic);
  this->declare_parameter("ref.output_topic", config_.ref_output_topic);

  this->declare_parameter<int>("qos.depth", config_.qos_depth);
  this->declare_parameter<bool>("qos.reliable", config_.qos_reliable);

  this->declare_parameter<int>(
    "diagnostics.publish_period_ms",
    config_.diagnostics_publish_period_ms);

  config_.target_sample_rate = this->get_parameter("target_sample_rate").as_int();
  config_.input_encoding = this->get_parameter("input.encoding").as_string();
  config_.input_bit_depth = this->get_parameter("input.bit_depth").as_int();
  config_.input_layout = this->get_parameter("input.layout").as_string();
  config_.output_encoding = this->get_parameter("output.encoding").as_string();
  config_.output_bit_depth = this->get_parameter("output.bit_depth").as_int();

  config_.mic_enabled = this->get_parameter("mic.enabled").as_bool();
  config_.mic_input_topic = this->get_parameter("mic.input_topic").as_string();
  config_.mic_output_topic = this->get_parameter("mic.output_topic").as_string();

  config_.ref_enabled = this->get_parameter("ref.enabled").as_bool();
  config_.ref_input_topic = this->get_parameter("ref.input_topic").as_string();
  config_.ref_output_topic = this->get_parameter("ref.output_topic").as_string();

  config_.qos_depth = this->get_parameter("qos.depth").as_int();
  config_.qos_reliable = this->get_parameter("qos.reliable").as_bool();

  config_.diagnostics_publish_period_ms =
    this->get_parameter("diagnostics.publish_period_ms").as_int();

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
  if (config_.mic_enabled) {
    if (config_.mic_input_topic.empty()) {
      throw std::runtime_error("mic.input_topic is required when mic.enabled=true");
    }
    if (config_.mic_output_topic.empty()) {
      throw std::runtime_error("mic.output_topic is required when mic.enabled=true");
    }
  }
  if (config_.ref_enabled) {
    if (config_.ref_input_topic.empty()) {
      throw std::runtime_error("ref.input_topic is required when ref.enabled=true");
    }
    if (config_.ref_output_topic.empty()) {
      throw std::runtime_error("ref.output_topic is required when ref.enabled=true");
    }
  }

  RCLCPP_INFO(this->get_logger(),
    "Resample config: target_sr=%dHz input=%s/%d/%s output=%s/%d qos_depth=%d reliable=%s "
    "mic=%s (%s -> %s) ref=%s (%s -> %s) diag=%dms",
    config_.target_sample_rate,
    config_.input_encoding.c_str(), config_.input_bit_depth, config_.input_layout.c_str(),
    config_.output_encoding.c_str(), config_.output_bit_depth,
    config_.qos_depth, config_.qos_reliable ? "true" : "false",
    config_.mic_enabled ? "on" : "off",
    config_.mic_input_topic.c_str(), config_.mic_output_topic.c_str(),
    config_.ref_enabled ? "on" : "off",
    config_.ref_input_topic.c_str(), config_.ref_output_topic.c_str(),
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

  diag_pub_ = this->create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
    "diagnostics", rclcpp::SystemDefaultsQoS());
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
    config_.mic_input_topic,
    config_.mic_output_topic,
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
    config_.ref_input_topic,
    config_.ref_output_topic,
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

  status.values.reserve(12);
  push_kv("target_sample_rate", std::to_string(backend_->targetSampleRate()));
  push_kv("output.encoding", config_.output_encoding);
  push_kv("output.bit_depth", std::to_string(config_.output_bit_depth));
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

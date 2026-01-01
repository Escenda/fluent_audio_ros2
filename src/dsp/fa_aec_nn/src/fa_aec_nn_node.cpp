#include "fa_aec_nn/fa_aec_nn_node.hpp"

#include <algorithm>
#include <chrono>
#include <memory>
#include <stdexcept>
#include <string>

#include "diagnostic_msgs/msg/diagnostic_status.hpp"
#include "diagnostic_msgs/msg/key_value.hpp"

namespace fa_aec_nn
{

namespace
{
constexpr int kRequiredSampleRate = 16000;
}

FaAecNnNode::FaAecNnNode()
: rclcpp::Node("fa_aec_nn")
{
  RCLCPP_INFO(this->get_logger(), "Starting FA AEC NN node");
  loadParameters();
  setupInterfaces();
}

void FaAecNnNode::loadParameters()
{
  this->declare_parameter<bool>("enabled", config_.enabled);
  this->declare_parameter("backend", config_.backend);
  this->declare_parameter("input_topic", config_.input_topic);
  this->declare_parameter("output_topic", config_.output_topic);
  this->declare_parameter<int>("expected_sample_rate", config_.expected_sample_rate);
  this->declare_parameter<int>("expected_channels", config_.expected_channels);
  this->declare_parameter("onnx.model_path", config_.onnx_model_path);
  this->declare_parameter<int>("qos.depth", config_.qos_depth);
  this->declare_parameter<bool>("qos.reliable", config_.qos_reliable);
  this->declare_parameter<int>(
    "diagnostics.publish_period_ms",
    config_.diagnostics_publish_period_ms);

  config_.enabled = this->get_parameter("enabled").as_bool();
  config_.backend = this->get_parameter("backend").as_string();
  config_.input_topic = this->get_parameter("input_topic").as_string();
  config_.output_topic = this->get_parameter("output_topic").as_string();
  config_.expected_sample_rate = this->get_parameter("expected_sample_rate").as_int();
  config_.expected_channels = this->get_parameter("expected_channels").as_int();
  config_.onnx_model_path = this->get_parameter("onnx.model_path").as_string();
  config_.qos_depth = this->get_parameter("qos.depth").as_int();
  config_.qos_reliable = this->get_parameter("qos.reliable").as_bool();
  config_.diagnostics_publish_period_ms =
    this->get_parameter("diagnostics.publish_period_ms").as_int();

  if (config_.input_topic.empty()) {
    throw std::runtime_error("input_topic is required (set via YAML)");
  }
  if (config_.output_topic.empty()) {
    throw std::runtime_error("output_topic is required (set via YAML)");
  }
  if (config_.expected_sample_rate != kRequiredSampleRate) {
    throw std::runtime_error(
            "fa_aec_nn requires expected_sample_rate=16000 by design (got " +
            std::to_string(config_.expected_sample_rate) + ")");
  }
  if (config_.qos_depth <= 0) {
    throw std::runtime_error("qos.depth must be > 0 (set via YAML)");
  }
  if (config_.diagnostics_publish_period_ms <= 0) {
    throw std::runtime_error("diagnostics.publish_period_ms must be > 0 (set via YAML)");
  }
  if (config_.backend.empty()) {
    throw std::runtime_error("backend is required (set via YAML)");
  }
  if (config_.backend != "passthrough" && config_.backend != "onnxruntime") {
    throw std::runtime_error("backend must be passthrough or onnxruntime");
  }
  if (config_.backend == "onnxruntime") {
    RCLCPP_WARN(this->get_logger(), "backend=onnxruntime is not implemented yet; passthrough behavior");
  }

  RCLCPP_INFO(this->get_logger(),
    "AEC NN config: enabled=%s backend=%s input=%s output=%s expected_sr=%d expected_ch=%d qos_depth=%d reliable=%s",
    config_.enabled ? "true" : "false",
    config_.backend.c_str(),
    config_.input_topic.c_str(),
    config_.output_topic.c_str(),
    config_.expected_sample_rate,
    config_.expected_channels,
    config_.qos_depth,
    config_.qos_reliable ? "true" : "false");
}

void FaAecNnNode::setupInterfaces()
{
  rclcpp::QoS qos(std::max<int>(1, config_.qos_depth));
  if (config_.qos_reliable) {
    qos.reliable();
  } else {
    qos.best_effort();
  }

  pub_ = this->create_publisher<fa_interfaces::msg::AudioFrame>(config_.output_topic, qos);
  sub_ = this->create_subscription<fa_interfaces::msg::AudioFrame>(
    config_.input_topic, qos,
    std::bind(&FaAecNnNode::onAudioFrame, this, std::placeholders::_1));

  diag_pub_ = this->create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
    "diagnostics", rclcpp::SystemDefaultsQoS());
  diag_timer_ = this->create_wall_timer(
    std::chrono::milliseconds(config_.diagnostics_publish_period_ms),
    std::bind(&FaAecNnNode::publishDiagnostics, this));
}

bool FaAecNnNode::validateFrame(const fa_interfaces::msg::AudioFrame & msg) const
{
  if (msg.sample_rate != static_cast<uint32_t>(config_.expected_sample_rate)) {
    return false;
  }
  if (config_.expected_channels > 0 && msg.channels != static_cast<uint32_t>(config_.expected_channels)) {
    return false;
  }
  if (msg.channels == 0 || msg.sample_rate == 0) {
    return false;
  }
  if (msg.bit_depth != 16 && msg.bit_depth != 32) {
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
  if (!msg || !pub_) {
    drop_.fetch_add(1);
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

  pub_->publish(*msg);
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

  status.values.reserve(10);
  push_kv("enabled", config_.enabled ? "true" : "false");
  push_kv("backend", config_.backend);
  push_kv("input_topic", config_.input_topic);
  push_kv("output_topic", config_.output_topic);
  push_kv("expected_sample_rate", std::to_string(config_.expected_sample_rate));
  push_kv("expected_channels", std::to_string(config_.expected_channels));
  push_kv("frames.in", std::to_string(in_.load()));
  push_kv("frames.out", std::to_string(out_.load()));
  push_kv("frames.drop", std::to_string(drop_.load()));

  array_msg.status.push_back(status);
  diag_pub_->publish(array_msg);
}

}  // namespace fa_aec_nn

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<fa_aec_nn::FaAecNnNode>();
    rclcpp::spin(node);
  } catch (const std::exception & e) {
    RCLCPP_FATAL(rclcpp::get_logger("fa_aec_nn"), "Exception: %s", e.what());
    return EXIT_FAILURE;
  }
  rclcpp::shutdown();
  return EXIT_SUCCESS;
}


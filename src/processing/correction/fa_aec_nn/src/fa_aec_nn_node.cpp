#include "fa_aec_nn/fa_aec_nn_node.hpp"

#include <algorithm>
#include <chrono>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "diagnostic_msgs/msg/diagnostic_status.hpp"
#include "diagnostic_msgs/msg/key_value.hpp"
#include "fa_aec_nn/backends/aec_nn_backend.hpp"
#include "fa_aec_nn/backends/passthrough_backend.hpp"

namespace fa_aec_nn
{

namespace
{
constexpr int kRequiredSampleRate = 16000;
constexpr const char * kInterleavedLayout = "interleaved";
}

FaAecNnNode::FaAecNnNode()
: rclcpp::Node("fa_aec_nn")
{
  RCLCPP_INFO(this->get_logger(), "Starting FA AEC NN node");
  loadParameters();
  initializeBackend();
  setupInterfaces();
}

void FaAecNnNode::loadParameters()
{
  this->declare_parameter<bool>("enabled", config_.enabled);
  this->declare_parameter("backend.name", config_.backend_name);
  this->declare_parameter("input_topic", config_.input_topic);
  this->declare_parameter("output_topic", config_.output_topic);
  this->declare_parameter<int>("expected_sample_rate", config_.expected_sample_rate);
  this->declare_parameter<int>("expected_channels", config_.expected_channels);
  this->declare_parameter<int>("qos.depth", config_.qos_depth);
  this->declare_parameter<bool>("qos.reliable", config_.qos_reliable);
  this->declare_parameter<int>(
    "diagnostics.publish_period_ms",
    config_.diagnostics_publish_period_ms);

  config_.enabled = this->get_parameter("enabled").as_bool();
  config_.backend_name = this->get_parameter("backend.name").as_string();
  config_.input_topic = this->get_parameter("input_topic").as_string();
  config_.output_topic = this->get_parameter("output_topic").as_string();
  config_.expected_sample_rate = this->get_parameter("expected_sample_rate").as_int();
  config_.expected_channels = this->get_parameter("expected_channels").as_int();
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
  if (config_.backend_name.empty()) {
    throw std::runtime_error("backend.name is required (set via YAML)");
  }
  if (config_.backend_name != "passthrough") {
    throw std::runtime_error("backend.name must be passthrough");
  }

  RCLCPP_INFO(this->get_logger(),
    "AEC NN config: enabled=%s backend.name=%s input=%s output=%s expected_sr=%d expected_ch=%d qos_depth=%d reliable=%s",
    config_.enabled ? "true" : "false",
    config_.backend_name.c_str(),
    config_.input_topic.c_str(),
    config_.output_topic.c_str(),
    config_.expected_sample_rate,
    config_.expected_channels,
    config_.qos_depth,
    config_.qos_reliable ? "true" : "false");
}

void FaAecNnNode::initializeBackend()
{
  if (config_.backend_name == "passthrough") {
    backend_ = std::make_unique<backends::PassthroughBackend>();
    return;
  }
  throw std::runtime_error("unsupported fa_aec_nn backend.name: " + config_.backend_name);
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
  if (msg.source_id.empty() || msg.stream_id.empty()) {
    return false;
  }
  if (msg.layout != kInterleavedLayout) {
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

  if (!config_.enabled) {
    drop_.fetch_add(1);
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping frame because fa_aec_nn is disabled; disable the system node instead");
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

  if (!backend_) {
    drop_.fetch_add(1);
    RCLCPP_ERROR_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping frame because fa_aec_nn backend is not initialized");
    return;
  }

  backends::AudioChunk chunk;
  chunk.sample_rate = static_cast<int>(msg->sample_rate);
  chunk.channels = static_cast<int>(msg->channels);
  chunk.bit_depth = static_cast<int>(msg->bit_depth);
  chunk.layout = msg->layout;
  chunk.data = msg->data.data();
  chunk.data_size = msg->data.size();

  std::vector<uint8_t> processed_data;
  try {
    processed_data = backend_->process(chunk);
  } catch (const std::exception & e) {
    drop_.fetch_add(1);
    RCLCPP_ERROR_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping frame because fa_aec_nn backend failed: %s", e.what());
    return;
  }

  if (processed_data.empty()) {
    drop_.fetch_add(1);
    RCLCPP_ERROR_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping frame because fa_aec_nn backend returned empty audio data");
    return;
  }

  auto out_msg = *msg;
  out_msg.data = std::move(processed_data);
  out_msg.stream_id = config_.output_topic;
  out_msg.layout = kInterleavedLayout;
  pub_->publish(out_msg);
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
  push_kv("backend.name", config_.backend_name);
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

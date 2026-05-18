#include "fa_agc/fa_agc_node.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>

#include "diagnostic_msgs/msg/diagnostic_status.hpp"
#include "diagnostic_msgs/msg/key_value.hpp"
#include "fa_agc/backends/internal_rms_agc.hpp"

namespace fa_agc
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
}  // namespace

FaAgcNode::FaAgcNode()
: rclcpp::Node("fa_agc")
{
  RCLCPP_INFO(this->get_logger(), "Starting FA AGC node");
  loadParameters();
  configureBackend();
  setupInterfaces();
}

FaAgcNode::~FaAgcNode() = default;

void FaAgcNode::loadParameters()
{
  this->declare_parameter("input_topic", config_.input_topic);
  this->declare_parameter("output_topic", config_.output_topic);
  this->declare_parameter<double>("agc.target_rms", config_.target_rms);
  this->declare_parameter<double>("agc.min_gain", config_.min_gain);
  this->declare_parameter<double>("agc.max_gain", config_.max_gain);
  this->declare_parameter<double>("agc.attack_ms", config_.attack_ms);
  this->declare_parameter<double>("agc.release_ms", config_.release_ms);
  this->declare_parameter<int>("expected.sample_rate", config_.expected_sample_rate);
  this->declare_parameter<int>("expected.channels", config_.expected_channels);
  this->declare_parameter("expected.encoding", config_.expected_encoding);
  this->declare_parameter<int>("expected.bit_depth", config_.expected_bit_depth);
  this->declare_parameter("expected.layout", config_.expected_layout);
  this->declare_parameter<int>("qos.depth", config_.qos_depth);
  this->declare_parameter<bool>("qos.reliable", config_.qos_reliable);
  this->declare_parameter<int>(
    "diagnostics.publish_period_ms",
    config_.diagnostics_publish_period_ms);

  config_.input_topic = this->get_parameter("input_topic").as_string();
  config_.output_topic = this->get_parameter("output_topic").as_string();
  config_.target_rms = this->get_parameter("agc.target_rms").as_double();
  config_.min_gain = this->get_parameter("agc.min_gain").as_double();
  config_.max_gain = this->get_parameter("agc.max_gain").as_double();
  config_.attack_ms = this->get_parameter("agc.attack_ms").as_double();
  config_.release_ms = this->get_parameter("agc.release_ms").as_double();
  config_.expected_sample_rate = this->get_parameter("expected.sample_rate").as_int();
  config_.expected_channels = this->get_parameter("expected.channels").as_int();
  config_.expected_encoding = this->get_parameter("expected.encoding").as_string();
  config_.expected_bit_depth = this->get_parameter("expected.bit_depth").as_int();
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
  if (!isFinite(config_.target_rms) ||
      config_.target_rms <= 0.0 ||
      config_.target_rms > 1.0)
  {
    throw std::runtime_error("agc.target_rms must be finite and in (0.0, 1.0]");
  }
  if (!isFinite(config_.min_gain) || config_.min_gain <= 0.0) {
    throw std::runtime_error("agc.min_gain must be finite and > 0.0");
  }
  if (!isFinite(config_.max_gain) || config_.max_gain < config_.min_gain) {
    throw std::runtime_error("agc.max_gain must be finite and >= agc.min_gain");
  }
  if (config_.min_gain > 1.0 || config_.max_gain < 1.0) {
    throw std::runtime_error("agc.min_gain <= 1.0 <= agc.max_gain is required for initial gain");
  }
  if (!isFinite(config_.attack_ms) || config_.attack_ms <= 0.0) {
    throw std::runtime_error("agc.attack_ms must be finite and > 0.0");
  }
  if (!isFinite(config_.release_ms) || config_.release_ms <= 0.0) {
    throw std::runtime_error("agc.release_ms must be finite and > 0.0");
  }
  if (config_.expected_sample_rate <= 0) {
    throw std::runtime_error("expected.sample_rate must be > 0");
  }
  if (config_.expected_channels <= 0) {
    throw std::runtime_error("expected.channels must be > 0");
  }
  if (config_.expected_encoding != kEncodingFloat32) {
    throw std::runtime_error("fa_agc requires expected.encoding=FLOAT32LE");
  }
  if (config_.expected_bit_depth != 32) {
    throw std::runtime_error("fa_agc requires expected.bit_depth=32");
  }
  if (config_.expected_layout != kInterleavedLayout) {
    throw std::runtime_error("fa_agc requires expected.layout=interleaved");
  }
  if (config_.qos_depth <= 0) {
    throw std::runtime_error("qos.depth must be > 0");
  }
  if (config_.diagnostics_publish_period_ms <= 0) {
    throw std::runtime_error("diagnostics.publish_period_ms must be > 0");
  }

  RCLCPP_INFO(
    this->get_logger(),
    "AGC config: input=%s output=%s target_rms=%f min_gain=%f max_gain=%f attack=%fms release=%fms expected=%dHz/%d/%s/%d qos_depth=%d reliable=%s diag=%dms",
    config_.input_topic.c_str(),
    config_.output_topic.c_str(),
    config_.target_rms,
    config_.min_gain,
    config_.max_gain,
    config_.attack_ms,
    config_.release_ms,
    config_.expected_sample_rate,
    config_.expected_channels,
    config_.expected_encoding.c_str(),
    config_.expected_bit_depth,
    config_.qos_depth,
    config_.qos_reliable ? "true" : "false",
    config_.diagnostics_publish_period_ms);
}

void FaAgcNode::configureBackend()
{
  backend_ = std::make_unique<backends::InternalRmsAgcBackend>(
    backends::InternalRmsAgcConfig{
      config_.expected_sample_rate,
      config_.expected_channels,
      config_.target_rms,
      config_.min_gain,
      config_.max_gain,
      config_.attack_ms,
      config_.release_ms});
}

void FaAgcNode::setupInterfaces()
{
  rclcpp::QoS qos(std::max<int>(1, config_.qos_depth));
  if (config_.qos_reliable) {
    qos.reliable();
  } else {
    qos.best_effort();
  }

  audio_pub_ = this->create_publisher<fa_interfaces::msg::AudioFrame>(config_.output_topic, qos);
  audio_sub_ = this->create_subscription<fa_interfaces::msg::AudioFrame>(
    config_.input_topic,
    qos,
    std::bind(&FaAgcNode::handleFrame, this, std::placeholders::_1));

  diag_pub_ = this->create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
    "diagnostics",
    rclcpp::SystemDefaultsQoS());
  diag_timer_ = this->create_wall_timer(
    std::chrono::milliseconds(config_.diagnostics_publish_period_ms),
    std::bind(&FaAgcNode::publishDiagnostics, this));
}

void FaAgcNode::handleFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg)
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
  if (!applyAgc(*msg, out)) {
    frames_dropped_.fetch_add(1);
    return;
  }

  audio_pub_->publish(out);
  frames_out_.fetch_add(1);
}

bool FaAgcNode::validateFrame(const fa_interfaces::msg::AudioFrame & msg)
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
  return true;
}

bool FaAgcNode::applyAgc(
  const fa_interfaces::msg::AudioFrame & in,
  fa_interfaces::msg::AudioFrame & out)
{
  out = in;
  out.stream_id = config_.output_topic;
  const backends::ProcessResult result = backend_->process(in.data, out.data);
  if (result.status != backends::ProcessStatus::kOk) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping frame because AGC backend rejected input or output: %s",
      backends::processStatusMessage(result.status));
    return false;
  }

  if (result.gain_direction == backends::GainDirection::kReduction) {
    gain_reductions_.fetch_add(1);
  } else if (result.gain_direction == backends::GainDirection::kIncrease) {
    gain_increases_.fetch_add(1);
  }
  return true;
}

void FaAgcNode::publishDiagnostics()
{
  diagnostic_msgs::msg::DiagnosticArray array_msg;
  array_msg.header.stamp = this->now();

  diagnostic_msgs::msg::DiagnosticStatus status;
  status.name = "fa_agc";
  status.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
  status.message = "running";
  status.values.reserve(14);
  pushKeyValue(status, "target_rms", std::to_string(backend_->targetRms()));
  pushKeyValue(status, "min_gain", std::to_string(backend_->minGain()));
  pushKeyValue(status, "max_gain", std::to_string(backend_->maxGain()));
  pushKeyValue(status, "attack_ms", std::to_string(backend_->attackMs()));
  pushKeyValue(status, "release_ms", std::to_string(backend_->releaseMs()));
  pushKeyValue(status, "current_gain", std::to_string(backend_->currentGain()));
  pushKeyValue(status, "last_frame_rms", std::to_string(backend_->lastFrameRms()));
  pushKeyValue(status, "last_target_gain", std::to_string(backend_->lastTargetGain()));
  pushKeyValue(status, "frames_in", std::to_string(frames_in_.load()));
  pushKeyValue(status, "frames_out", std::to_string(frames_out_.load()));
  pushKeyValue(status, "frames_dropped", std::to_string(frames_dropped_.load()));
  pushKeyValue(status, "gain_reductions", std::to_string(gain_reductions_.load()));
  pushKeyValue(status, "gain_increases", std::to_string(gain_increases_.load()));
  pushKeyValue(status, "output_topic", config_.output_topic);

  array_msg.status.push_back(status);
  diag_pub_->publish(array_msg);
}

}  // namespace fa_agc

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<fa_agc::FaAgcNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return EXIT_SUCCESS;
  } catch (const std::exception & e) {
    RCLCPP_FATAL(rclcpp::get_logger("fa_agc"), "Exception: %s", e.what());
    rclcpp::shutdown();
    return EXIT_FAILURE;
  }
}

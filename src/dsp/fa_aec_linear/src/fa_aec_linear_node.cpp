#include "fa_aec_linear/fa_aec_linear_node.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "diagnostic_msgs/msg/diagnostic_status.hpp"
#include "diagnostic_msgs/msg/key_value.hpp"

namespace fa_aec_linear
{

namespace
{
constexpr int kRequiredSampleRate = 16000;
}

FaAecLinearNode::FaAecLinearNode()
: rclcpp::Node("fa_aec_linear")
{
  RCLCPP_INFO(this->get_logger(), "Starting FA AEC Linear node");
  loadParameters();
  setupInterfaces();
}

void FaAecLinearNode::loadParameters()
{
  this->declare_parameter<bool>("enabled", config_.enabled);
  this->declare_parameter("mic_topic", config_.mic_topic);
  this->declare_parameter("ref_topic", config_.ref_topic);
  this->declare_parameter("output_topic", config_.output_topic);
  this->declare_parameter<int>("expected_sample_rate", config_.expected_sample_rate);
  this->declare_parameter<int>("expected_channels", config_.expected_channels);
  this->declare_parameter<int>("ref_timeout_ms", config_.ref_timeout_ms);
  this->declare_parameter<double>("cancel_gain", config_.cancel_gain);
  this->declare_parameter<int>("qos.depth", config_.qos_depth);
  this->declare_parameter<bool>("qos.reliable", config_.qos_reliable);
  this->declare_parameter<int>(
    "diagnostics.publish_period_ms",
    config_.diagnostics_publish_period_ms);

  config_.enabled = this->get_parameter("enabled").as_bool();
  config_.mic_topic = this->get_parameter("mic_topic").as_string();
  config_.ref_topic = this->get_parameter("ref_topic").as_string();
  config_.output_topic = this->get_parameter("output_topic").as_string();
  config_.expected_sample_rate = this->get_parameter("expected_sample_rate").as_int();
  config_.expected_channels = this->get_parameter("expected_channels").as_int();
  config_.ref_timeout_ms = this->get_parameter("ref_timeout_ms").as_int();
  config_.cancel_gain = this->get_parameter("cancel_gain").as_double();
  config_.qos_depth = this->get_parameter("qos.depth").as_int();
  config_.qos_reliable = this->get_parameter("qos.reliable").as_bool();
  config_.diagnostics_publish_period_ms =
    this->get_parameter("diagnostics.publish_period_ms").as_int();

  if (config_.mic_topic.empty()) {
    throw std::runtime_error("mic_topic is required (set via YAML)");
  }
  if (config_.ref_topic.empty()) {
    throw std::runtime_error("ref_topic is required (set via YAML)");
  }
  if (config_.output_topic.empty()) {
    throw std::runtime_error("output_topic is required (set via YAML)");
  }
  if (config_.expected_sample_rate != kRequiredSampleRate) {
    throw std::runtime_error(
            "fa_aec_linear requires expected_sample_rate=16000 by design (got " +
            std::to_string(config_.expected_sample_rate) + ")");
  }
  if (config_.qos_depth <= 0) {
    throw std::runtime_error("qos.depth must be > 0 (set via YAML)");
  }
  if (config_.diagnostics_publish_period_ms <= 0) {
    throw std::runtime_error("diagnostics.publish_period_ms must be > 0 (set via YAML)");
  }
  if (config_.ref_timeout_ms <= 0) {
    throw std::runtime_error("ref_timeout_ms must be > 0 (set via YAML)");
  }

  RCLCPP_INFO(this->get_logger(),
    "AEC Linear config: enabled=%s mic=%s ref=%s output=%s expected_sr=%d expected_ch=%d "
    "ref_timeout=%dms cancel_gain=%.3f qos_depth=%d reliable=%s",
    config_.enabled ? "true" : "false",
    config_.mic_topic.c_str(),
    config_.ref_topic.c_str(),
    config_.output_topic.c_str(),
    config_.expected_sample_rate,
    config_.expected_channels,
    config_.ref_timeout_ms,
    config_.cancel_gain,
    config_.qos_depth,
    config_.qos_reliable ? "true" : "false");
}

void FaAecLinearNode::setupInterfaces()
{
  rclcpp::QoS qos(std::max<int>(1, config_.qos_depth));
  if (config_.qos_reliable) {
    qos.reliable();
  } else {
    qos.best_effort();
  }

  out_pub_ = this->create_publisher<fa_interfaces::msg::AudioFrame>(config_.output_topic, qos);
  mic_sub_ = this->create_subscription<fa_interfaces::msg::AudioFrame>(
    config_.mic_topic, qos,
    std::bind(&FaAecLinearNode::onMicFrame, this, std::placeholders::_1));
  ref_sub_ = this->create_subscription<fa_interfaces::msg::AudioFrame>(
    config_.ref_topic, qos,
    std::bind(&FaAecLinearNode::onRefFrame, this, std::placeholders::_1));

  diag_pub_ = this->create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
    "diagnostics", rclcpp::SystemDefaultsQoS());
  diag_timer_ = this->create_wall_timer(
    std::chrono::milliseconds(config_.diagnostics_publish_period_ms),
    std::bind(&FaAecLinearNode::publishDiagnostics, this));
}

bool FaAecLinearNode::validateFrame(const fa_interfaces::msg::AudioFrame & msg) const
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

bool FaAecLinearNode::decodeToFloat(const fa_interfaces::msg::AudioFrame & msg, std::vector<float> & out_samples)
{
  out_samples.clear();
  if (msg.channels == 0) {
    return false;
  }
  if (msg.bit_depth != 16 && msg.bit_depth != 32) {
    return false;
  }
  const size_t bytes_per_sample = static_cast<size_t>(msg.bit_depth / 8);
  const size_t bytes_per_frame = static_cast<size_t>(msg.channels) * bytes_per_sample;
  if (bytes_per_frame == 0 || (msg.data.size() % bytes_per_frame) != 0) {
    return false;
  }

  const size_t frames = msg.data.size() / bytes_per_frame;
  const size_t sample_count = frames * msg.channels;
  if (sample_count == 0) {
    return false;
  }
  out_samples.resize(sample_count);

  if (msg.bit_depth == 16) {
    std::vector<int16_t> tmp(sample_count);
    std::memcpy(tmp.data(), msg.data.data(), msg.data.size());
    for (size_t i = 0; i < sample_count; ++i) {
      out_samples[i] = static_cast<float>(tmp[i]) / 32768.0f;
    }
  } else {
    std::memcpy(out_samples.data(), msg.data.data(), msg.data.size());
  }
  return true;
}

void FaAecLinearNode::encodeFromFloat(const std::vector<float> & samples, uint32_t bit_depth, std::vector<uint8_t> & out_bytes)
{
  out_bytes.clear();
  if (samples.empty()) {
    return;
  }
  if (bit_depth == 16) {
    std::vector<int16_t> pcm(samples.size());
    for (size_t i = 0; i < samples.size(); ++i) {
      float s = std::clamp(samples[i], -1.0f, 1.0f);
      const int32_t scaled = static_cast<int32_t>(std::lround(static_cast<double>(s) * 32767.0));
      pcm[i] = static_cast<int16_t>(std::clamp<int32_t>(scaled, -32768, 32767));
    }
    out_bytes.resize(pcm.size() * sizeof(int16_t));
    std::memcpy(out_bytes.data(), pcm.data(), out_bytes.size());
    return;
  }
  if (bit_depth == 32) {
    out_bytes.resize(samples.size() * sizeof(float));
    std::memcpy(out_bytes.data(), samples.data(), out_bytes.size());
  }
}

void FaAecLinearNode::computeRmsPeak(const std::vector<float> & interleaved, float & out_rms, float & out_peak)
{
  out_rms = 0.0f;
  out_peak = 0.0f;
  if (interleaved.empty()) {
    return;
  }
  double accum = 0.0;
  double peak = 0.0;
  for (float v : interleaved) {
    const double dv = static_cast<double>(v);
    accum += dv * dv;
    peak = std::max(peak, std::abs(dv));
  }
  out_rms = static_cast<float>(std::sqrt(accum / static_cast<double>(interleaved.size())));
  out_peak = static_cast<float>(peak);
}

void FaAecLinearNode::onRefFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg)
{
  ref_in_.fetch_add(1);
  if (!msg) {
    ref_drop_.fetch_add(1);
    return;
  }
  if (!validateFrame(*msg)) {
    ref_drop_.fetch_add(1);
    return;
  }

  std::lock_guard<std::mutex> lock(ref_mutex_);
  last_ref_ = msg;
  last_ref_stamp_ = this->now();
}

void FaAecLinearNode::onMicFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg)
{
  mic_in_.fetch_add(1);
  if (!msg || !out_pub_) {
    mic_drop_.fetch_add(1);
    return;
  }
  if (!validateFrame(*msg)) {
    mic_drop_.fetch_add(1);
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping invalid mic frame: sr=%u ch=%u bits=%u bytes=%zu (expected sr=%d ch=%d)",
      msg->sample_rate, msg->channels, msg->bit_depth, msg->data.size(),
      config_.expected_sample_rate, config_.expected_channels);
    return;
  }

  if (!config_.enabled) {
    out_pub_->publish(*msg);
    mic_out_.fetch_add(1);
    return;
  }

  fa_interfaces::msg::AudioFrame::SharedPtr ref;
  rclcpp::Time ref_stamp{0, 0, RCL_ROS_TIME};
  {
    std::lock_guard<std::mutex> lock(ref_mutex_);
    ref = last_ref_;
    ref_stamp = last_ref_stamp_;
  }

  const rclcpp::Time now = this->now();
  const int64_t ref_age_ms = (now - ref_stamp).nanoseconds() / 1000000;
  const bool has_ref = ref && (ref_age_ms >= 0) && (ref_age_ms <= config_.ref_timeout_ms);

  if (!has_ref) {
    out_pub_->publish(*msg);
    mic_out_.fetch_add(1);
    return;
  }

  if (ref->sample_rate != msg->sample_rate || ref->channels != msg->channels || ref->bit_depth != msg->bit_depth) {
    out_pub_->publish(*msg);
    mic_out_.fetch_add(1);
    return;
  }

  std::vector<float> mic_f32;
  std::vector<float> ref_f32;
  if (!decodeToFloat(*msg, mic_f32) || !decodeToFloat(*ref, ref_f32)) {
    out_pub_->publish(*msg);
    mic_out_.fetch_add(1);
    return;
  }

  const size_t sample_count = std::min(mic_f32.size(), ref_f32.size());
  if (sample_count == 0) {
    out_pub_->publish(*msg);
    mic_out_.fetch_add(1);
    return;
  }

  std::vector<float> out_f32 = mic_f32;
  const float gain = static_cast<float>(config_.cancel_gain);
  for (size_t i = 0; i < sample_count; ++i) {
    out_f32[i] = mic_f32[i] - gain * ref_f32[i];
  }

  std::vector<uint8_t> out_bytes;
  encodeFromFloat(out_f32, msg->bit_depth, out_bytes);
  if (out_bytes.empty()) {
    mic_drop_.fetch_add(1);
    return;
  }

  float out_rms = 0.0f;
  float out_peak = 0.0f;
  computeRmsPeak(out_f32, out_rms, out_peak);

  fa_interfaces::msg::AudioFrame out_msg;
  out_msg.header = msg->header;
  out_msg.encoding = msg->encoding;
  out_msg.sample_rate = msg->sample_rate;
  out_msg.channels = msg->channels;
  out_msg.bit_depth = msg->bit_depth;
  out_msg.rms = out_rms;
  out_msg.peak = out_peak;
  out_msg.vad = msg->vad;
  out_msg.data = std::move(out_bytes);
  out_msg.epoch = msg->epoch;

  out_pub_->publish(out_msg);
  mic_out_.fetch_add(1);
}

void FaAecLinearNode::publishDiagnostics()
{
  diagnostic_msgs::msg::DiagnosticArray array_msg;
  array_msg.header.stamp = this->now();

  diagnostic_msgs::msg::DiagnosticStatus status;
  status.name = "fa_aec_linear";
  status.hardware_id = "dsp";
  status.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
  status.message = "running";

  rclcpp::Time ref_stamp{0, 0, RCL_ROS_TIME};
  {
    std::lock_guard<std::mutex> lock(ref_mutex_);
    ref_stamp = last_ref_stamp_;
  }
  const int64_t ref_age_ms = (this->now() - ref_stamp).nanoseconds() / 1000000;

  auto push_kv = [&status](const std::string & key, const std::string & value) {
      diagnostic_msgs::msg::KeyValue kv;
      kv.key = key;
      kv.value = value;
      status.values.push_back(kv);
    };

  status.values.reserve(16);
  push_kv("enabled", config_.enabled ? "true" : "false");
  push_kv("mic_topic", config_.mic_topic);
  push_kv("ref_topic", config_.ref_topic);
  push_kv("output_topic", config_.output_topic);
  push_kv("expected_sample_rate", std::to_string(config_.expected_sample_rate));
  push_kv("expected_channels", std::to_string(config_.expected_channels));
  push_kv("ref_timeout_ms", std::to_string(config_.ref_timeout_ms));
  push_kv("cancel_gain", std::to_string(config_.cancel_gain));
  push_kv("ref_age_ms", std::to_string(ref_age_ms));
  push_kv("mic.in", std::to_string(mic_in_.load()));
  push_kv("mic.out", std::to_string(mic_out_.load()));
  push_kv("mic.drop", std::to_string(mic_drop_.load()));
  push_kv("ref.in", std::to_string(ref_in_.load()));
  push_kv("ref.drop", std::to_string(ref_drop_.load()));

  array_msg.status.push_back(status);
  diag_pub_->publish(array_msg);
}

}  // namespace fa_aec_linear

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<fa_aec_linear::FaAecLinearNode>();
    rclcpp::spin(node);
  } catch (const std::exception & e) {
    RCLCPP_FATAL(rclcpp::get_logger("fa_aec_linear"), "Exception: %s", e.what());
    return EXIT_FAILURE;
  }
  rclcpp::shutdown();
  return EXIT_SUCCESS;
}


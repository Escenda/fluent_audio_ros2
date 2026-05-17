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
constexpr const char * kEncodingPcm16 = "PCM16LE";
constexpr const char * kEncodingFloat32 = "FLOAT32LE";
constexpr const char * kInterleavedLayout = "interleaved";

bool isSupportedAudioFormatPair(const std::string & encoding, uint32_t bit_depth)
{
  return (encoding == kEncodingPcm16 && bit_depth == 16) ||
         (encoding == kEncodingFloat32 && bit_depth == 32);
}
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
  this->declare_parameter("reference_failure_policy", config_.reference_failure_policy);
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
  config_.reference_failure_policy = this->get_parameter("reference_failure_policy").as_string();
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
  if (config_.reference_failure_policy != "drop") {
    throw std::runtime_error("reference_failure_policy must be drop");
  }
  if (!std::isfinite(config_.cancel_gain)) {
    throw std::runtime_error("cancel_gain must be finite");
  }

  RCLCPP_INFO(this->get_logger(),
    "AEC Linear config: enabled=%s mic=%s ref=%s output=%s expected_sr=%d expected_ch=%d "
    "ref_timeout=%dms reference_failure_policy=%s cancel_gain=%.3f qos_depth=%d reliable=%s",
    config_.enabled ? "true" : "false",
    config_.mic_topic.c_str(),
    config_.ref_topic.c_str(),
    config_.output_topic.c_str(),
    config_.expected_sample_rate,
    config_.expected_channels,
    config_.ref_timeout_ms,
    config_.reference_failure_policy.c_str(),
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
  if (!isSupportedAudioFormatPair(msg.encoding, msg.bit_depth)) {
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

bool FaAecLinearNode::decodeToFloat(const fa_interfaces::msg::AudioFrame & msg, std::vector<float> & out_samples)
{
  out_samples.clear();
  if (msg.channels == 0) {
    return false;
  }
  if (!isSupportedAudioFormatPair(msg.encoding, msg.bit_depth)) {
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

  if (msg.encoding == kEncodingPcm16 && msg.bit_depth == 16) {
    std::vector<int16_t> tmp(sample_count);
    std::memcpy(tmp.data(), msg.data.data(), msg.data.size());
    for (size_t i = 0; i < sample_count; ++i) {
      out_samples[i] = static_cast<float>(tmp[i]) / 32768.0f;
    }
    return true;
  }

  if (msg.encoding != kEncodingFloat32 || msg.bit_depth != 32) {
    return false;
  }
  std::memcpy(out_samples.data(), msg.data.data(), msg.data.size());
  for (const float sample : out_samples) {
    if (!std::isfinite(sample) || sample < -1.0f || sample > 1.0f) {
      return false;
    }
  }
  return true;
}

bool FaAecLinearNode::encodeFromFloat(
  const std::vector<float> & samples,
  const std::string & encoding,
  uint32_t bit_depth,
  std::vector<uint8_t> & out_bytes,
  std::string & error_message)
{
  out_bytes.clear();
  error_message.clear();
  if (samples.empty()) {
    error_message = "AEC linear output sample buffer is empty";
    return false;
  }
  if (!isSupportedAudioFormatPair(encoding, bit_depth)) {
    error_message = "AEC linear output format must be PCM16LE/16 or FLOAT32LE/32";
    return false;
  }
  for (const float sample : samples) {
    if (!std::isfinite(sample)) {
      error_message = "AEC linear output sample is not finite";
      return false;
    }
    if (sample < -1.0f || sample > 1.0f) {
      error_message = "AEC linear output sample out of normalized range";
      return false;
    }
  }
  if (encoding == kEncodingPcm16 && bit_depth == 16) {
    std::vector<int16_t> pcm(samples.size());
    for (size_t i = 0; i < samples.size(); ++i) {
      const double scaled = samples[i] < 0.0f ?
        static_cast<double>(samples[i]) * 32768.0 :
        static_cast<double>(samples[i]) * 32767.0;
      const int32_t rounded = static_cast<int32_t>(std::lround(scaled));
      if (rounded < -32768 || rounded > 32767) {
        error_message = "AEC linear output sample does not fit PCM16 after scaling";
        return false;
      }
      pcm[i] = static_cast<int16_t>(rounded);
    }
    out_bytes.resize(pcm.size() * sizeof(int16_t));
    std::memcpy(out_bytes.data(), pcm.data(), out_bytes.size());
    return true;
  }
  if (encoding == kEncodingFloat32 && bit_depth == 32) {
    out_bytes.resize(samples.size() * sizeof(float));
    std::memcpy(out_bytes.data(), samples.data(), out_bytes.size());
    return true;
  }
  error_message = "AEC linear output format must be PCM16LE/16 or FLOAT32LE/32";
  return false;
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
    mic_drop_.fetch_add(1);
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping mic frame because fa_aec_linear is disabled; disable the system node instead");
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
    mic_drop_.fetch_add(1);
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping mic frame because reference is missing or stale: ref_age_ms=%ld timeout_ms=%d",
      static_cast<long>(ref_age_ms), config_.ref_timeout_ms);
    return;
  }

  if (ref->sample_rate != msg->sample_rate || ref->channels != msg->channels ||
    ref->encoding != msg->encoding || ref->bit_depth != msg->bit_depth)
  {
    mic_drop_.fetch_add(1);
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping mic frame because reference format does not match: "
      "mic=%uHz/%uch/%s/%ubit ref=%uHz/%uch/%s/%ubit",
      msg->sample_rate, msg->channels, msg->encoding.c_str(), msg->bit_depth,
      ref->sample_rate, ref->channels, ref->encoding.c_str(), ref->bit_depth);
    return;
  }

  std::vector<float> mic_f32;
  std::vector<float> ref_f32;
  if (!decodeToFloat(*msg, mic_f32) || !decodeToFloat(*ref, ref_f32)) {
    mic_drop_.fetch_add(1);
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping mic frame because mic/reference decode failed");
    return;
  }

  if (mic_f32.empty() || ref_f32.empty()) {
    mic_drop_.fetch_add(1);
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping mic frame because no aligned mic/reference samples are available");
    return;
  }
  if (mic_f32.size() != ref_f32.size()) {
    mic_drop_.fetch_add(1);
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping mic frame because mic/reference sample counts differ: mic=%zu ref=%zu",
      mic_f32.size(), ref_f32.size());
    return;
  }

  std::vector<float> out_f32 = mic_f32;
  const float gain = static_cast<float>(config_.cancel_gain);
  for (size_t i = 0; i < mic_f32.size(); ++i) {
    out_f32[i] = mic_f32[i] - gain * ref_f32[i];
  }

  std::vector<uint8_t> out_bytes;
  std::string encode_error;
  if (!encodeFromFloat(out_f32, msg->encoding, msg->bit_depth, out_bytes, encode_error)) {
    mic_drop_.fetch_add(1);
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping AEC linear output frame: %s. Add an explicit dynamics/limiter node if range control is required.",
      encode_error.c_str());
    return;
  }

  fa_interfaces::msg::AudioFrame out_msg;
  out_msg.header = msg->header;
  out_msg.source_id = msg->source_id;
  out_msg.stream_id = config_.output_topic;
  out_msg.encoding = msg->encoding;
  out_msg.sample_rate = msg->sample_rate;
  out_msg.channels = msg->channels;
  out_msg.bit_depth = msg->bit_depth;
  out_msg.layout = kInterleavedLayout;
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
  push_kv("reference_failure_policy", config_.reference_failure_policy);
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

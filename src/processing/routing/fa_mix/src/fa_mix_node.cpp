#include "fa_mix/fa_mix_node.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "diagnostic_msgs/msg/diagnostic_status.hpp"
#include "diagnostic_msgs/msg/key_value.hpp"

namespace fa_mix
{

namespace
{
constexpr const char * kEncodingPcm16Le = "PCM16LE";
constexpr const char * kInterleavedLayout = "interleaved";
constexpr uint32_t kNanosecondsPerSecond = 1000000000U;

double dbToLinear(double db)
{
  return std::pow(10.0, db / 20.0);
}
bool isFinite(double value)
{
  return std::isfinite(value);
}

bool hasValidFrameStamp(const fa_interfaces::msg::AudioFrame & msg)
{
  if (msg.header.stamp.sec < 0) {
    return false;
  }
  if (msg.header.stamp.nanosec >= kNanosecondsPerSecond) {
    return false;
  }
  return msg.header.stamp.sec != 0 || msg.header.stamp.nanosec != 0;
}

rclcpp::Time frameStamp(const fa_interfaces::msg::AudioFrame & msg)
{
  return rclcpp::Time(msg.header.stamp, RCL_ROS_TIME);
}
}  // namespace

FaMixNode::FaMixNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("fa_mix", options)
{
  RCLCPP_INFO(this->get_logger(), "Starting FA Mix node");
  loadParameters();
  setupInterfaces();
}

void FaMixNode::loadParameters()
{
  this->declare_parameter<std::vector<std::string>>("input_topics", config_.input_topics);
  this->declare_parameter<std::vector<double>>("input_gains_db", config_.input_gains_db);
  this->declare_parameter<int>("master_index", config_.master_index);
  this->declare_parameter("output_topic", config_.output_topic);

  this->declare_parameter<int>("expected.sample_rate", config_.expected_sample_rate);
  this->declare_parameter<int>("expected.channels", config_.expected_channels);
  this->declare_parameter<int>("expected.bit_depth", config_.expected_bit_depth);
  this->declare_parameter("expected.encoding", config_.expected_encoding);

  this->declare_parameter<int>("max_frame_age_ms", config_.max_frame_age_ms);

  this->declare_parameter<int>("qos.depth", config_.qos_depth);
  this->declare_parameter<bool>("qos.reliable", config_.qos_reliable);

  this->declare_parameter<int>(
    "diagnostics.publish_period_ms",
    config_.diagnostics_publish_period_ms);

  config_.input_topics = this->get_parameter("input_topics").as_string_array();
  config_.input_gains_db = this->get_parameter("input_gains_db").as_double_array();
  config_.master_index = this->get_parameter("master_index").as_int();
  config_.output_topic = this->get_parameter("output_topic").as_string();

  config_.expected_sample_rate = this->get_parameter("expected.sample_rate").as_int();
  config_.expected_channels = this->get_parameter("expected.channels").as_int();
  config_.expected_bit_depth = this->get_parameter("expected.bit_depth").as_int();
  config_.expected_encoding = this->get_parameter("expected.encoding").as_string();

  config_.max_frame_age_ms = this->get_parameter("max_frame_age_ms").as_int();

  config_.qos_depth = this->get_parameter("qos.depth").as_int();
  config_.qos_reliable = this->get_parameter("qos.reliable").as_bool();

  config_.diagnostics_publish_period_ms =
    this->get_parameter("diagnostics.publish_period_ms").as_int();

  if (config_.input_topics.empty()) {
    throw std::runtime_error("input_topics must have at least 1 topic (set via YAML)");
  }
  for (const auto & topic : config_.input_topics) {
    if (topic.empty()) {
      throw std::runtime_error("input_topics must not contain an empty topic");
    }
  }
  if (config_.output_topic.empty()) {
    throw std::runtime_error("output_topic is required (set via YAML)");
  }
  if (config_.master_index < 0 || config_.master_index >= static_cast<int>(config_.input_topics.size())) {
    throw std::runtime_error("master_index out of range");
  }
  if (config_.input_gains_db.empty()) {
    throw std::runtime_error("input_gains_db must be size 1 or match input_topics length");
  }
  if (config_.input_gains_db.size() != 1 &&
    config_.input_gains_db.size() != config_.input_topics.size()) {
    throw std::runtime_error("input_gains_db must be size 1 or match input_topics length");
  }
  if (config_.expected_sample_rate <= 0 || config_.expected_channels <= 0 || config_.expected_bit_depth <= 0) {
    throw std::runtime_error("expected.* must be set (sample_rate/channels/bit_depth)");
  }
  if (config_.expected_bit_depth != 16) {
    throw std::runtime_error("fa_mix currently supports expected.bit_depth=16 only");
  }
  if (config_.expected_encoding != kEncodingPcm16Le) {
    throw std::runtime_error("fa_mix currently supports expected.encoding=PCM16LE only");
  }
  if (config_.max_frame_age_ms <= 0) {
    throw std::runtime_error("max_frame_age_ms must be > 0");
  }
  if (config_.qos_depth <= 0) {
    throw std::runtime_error("qos.depth must be > 0");
  }
  if (config_.diagnostics_publish_period_ms <= 0) {
    throw std::runtime_error("diagnostics.publish_period_ms must be > 0");
  }

  config_.input_gains_linear.clear();
  config_.input_gains_linear.reserve(config_.input_topics.size());
  for (size_t i = 0; i < config_.input_topics.size(); ++i) {
    const double gain_db = config_.input_gains_db.size() == 1 ?
      config_.input_gains_db[0] :
      config_.input_gains_db[i];
    const double gain_linear = dbToLinear(gain_db);
    if (!isFinite(gain_db) || !isFinite(gain_linear)) {
      throw std::runtime_error("input_gains_db must resolve to finite linear gains");
    }
    config_.input_gains_linear.push_back(gain_linear);
  }

  RCLCPP_INFO(this->get_logger(),
    "Mix config: inputs=%zu master=%d output=%s expected=%dHz/%dch/%dbits enc=%s max_age=%dms qos_depth=%d reliable=%s",
    config_.input_topics.size(),
    config_.master_index,
    config_.output_topic.c_str(),
    config_.expected_sample_rate,
    config_.expected_channels,
    config_.expected_bit_depth,
    config_.expected_encoding.c_str(),
    config_.max_frame_age_ms,
    config_.qos_depth,
    config_.qos_reliable ? "true" : "false");
}

void FaMixNode::setupInterfaces()
{
  rclcpp::QoS qos(static_cast<size_t>(config_.qos_depth));
  if (config_.qos_reliable) {
    qos.reliable();
  } else {
    qos.best_effort();
  }

  pub_ = this->create_publisher<fa_interfaces::msg::AudioFrame>(config_.output_topic, qos);

  subs_.resize(config_.input_topics.size());
  latest_frames_.resize(config_.input_topics.size());

  for (size_t i = 0; i < config_.input_topics.size(); ++i) {
    const std::string topic = config_.input_topics[i];
    subs_[i] = this->create_subscription<fa_interfaces::msg::AudioFrame>(
      topic, qos,
      [this, i](const fa_interfaces::msg::AudioFrame::SharedPtr msg) { this->onInputFrame(i, msg); });
  }

  diag_pub_ = this->create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
    "diagnostics", rclcpp::SystemDefaultsQoS());
  diag_timer_ = this->create_wall_timer(
    std::chrono::milliseconds(config_.diagnostics_publish_period_ms),
    std::bind(&FaMixNode::publishDiagnostics, this));
}

bool FaMixNode::validateFrame(
  const fa_interfaces::msg::AudioFrame & msg,
  const std::string & expected_stream_id) const
{
  if (msg.sample_rate != static_cast<uint32_t>(config_.expected_sample_rate)) {
    return false;
  }
  if (msg.channels != static_cast<uint32_t>(config_.expected_channels)) {
    return false;
  }
  if (msg.bit_depth != static_cast<uint32_t>(config_.expected_bit_depth)) {
    return false;
  }
  if (msg.encoding != config_.expected_encoding) {
    return false;
  }
  if (msg.source_id.empty()) {
    return false;
  }
  if (msg.stream_id != expected_stream_id) {
    return false;
  }
  if (!hasValidFrameStamp(msg)) {
    return false;
  }
  if (msg.layout != kInterleavedLayout) {
    return false;
  }
  if (msg.data.empty()) {
    return false;
  }
  const size_t bytes_per_frame = static_cast<size_t>(msg.channels) * (msg.bit_depth / 8);
  if (bytes_per_frame == 0 || (msg.data.size() % bytes_per_frame) != 0) {
    return false;
  }
  return true;
}

bool FaMixNode::decodePcm16ToFloat(const fa_interfaces::msg::AudioFrame & msg, std::vector<float> & out_samples)
{
  out_samples.clear();
  if (msg.bit_depth != 16 || msg.channels == 0) {
    return false;
  }
  const size_t bytes_per_frame = static_cast<size_t>(msg.channels) * 2;
  if (bytes_per_frame == 0 || (msg.data.size() % bytes_per_frame) != 0) {
    return false;
  }
  const size_t frames = msg.data.size() / bytes_per_frame;
  const size_t sample_count = frames * msg.channels;
  if (sample_count == 0) {
    return false;
  }
  out_samples.resize(sample_count);
  std::vector<int16_t> tmp(sample_count);
  std::memcpy(tmp.data(), msg.data.data(), msg.data.size());
  for (size_t i = 0; i < sample_count; ++i) {
    out_samples[i] = static_cast<float>(tmp[i]) / 32768.0f;
  }
  return true;
}

bool FaMixNode::encodeFloatToPcm16(
  const std::vector<float> & samples,
  std::vector<uint8_t> & out_bytes,
  std::string & error_message)
{
  out_bytes.clear();
  error_message.clear();
  if (samples.empty()) {
    error_message = "mixed sample buffer is empty";
    return false;
  }
  std::vector<int16_t> pcm(samples.size());
  for (size_t i = 0; i < samples.size(); ++i) {
    const float sample = samples[i];
    if (!std::isfinite(sample)) {
      error_message = "mixed sample is not finite";
      return false;
    }
    if (sample < -1.0f || sample > 1.0f) {
      error_message = "mixed sample out of normalized PCM16 range";
      return false;
    }

    const double scaled = sample < 0.0f ?
      static_cast<double>(sample) * 32768.0 :
      static_cast<double>(sample) * 32767.0;
    const int32_t rounded = static_cast<int32_t>(std::lround(scaled));
    if (rounded < -32768 || rounded > 32767) {
      error_message = "mixed sample does not fit PCM16 after scaling";
      return false;
    }
    pcm[i] = static_cast<int16_t>(rounded);
  }
  out_bytes.resize(pcm.size() * sizeof(int16_t));
  std::memcpy(out_bytes.data(), pcm.data(), out_bytes.size());
  return true;
}

void FaMixNode::onInputFrame(size_t index, const fa_interfaces::msg::AudioFrame::SharedPtr msg)
{
  in_.fetch_add(1);
  if (!msg) {
    drop_.fetch_add(1);
    return;
  }
  if (!validateFrame(*msg, config_.input_topics[index])) {
    drop_.fetch_add(1);
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping invalid frame[%zu]: sr=%u ch=%u bits=%u enc=%s bytes=%zu (expected %dHz/%d/%d enc=%s)",
      index, msg->sample_rate, msg->channels, msg->bit_depth, msg->encoding.c_str(), msg->data.size(),
      config_.expected_sample_rate, config_.expected_channels, config_.expected_bit_depth, config_.expected_encoding.c_str());
    return;
  }

  {
    std::lock_guard<std::mutex> lock(frames_mutex_);
    latest_frames_[index] = msg;
  }

  if (static_cast<int>(index) != config_.master_index) {
    return;
  }
  mixAndPublish(*msg);
}

void FaMixNode::mixAndPublish(const fa_interfaces::msg::AudioFrame & base)
{
  if (!pub_) {
    drop_.fetch_add(1);
    return;
  }

  std::vector<float> mixed;
  if (!decodePcm16ToFloat(base, mixed)) {
    drop_.fetch_add(1);
    return;
  }

  const rclcpp::Time base_time = frameStamp(base);
  uint32_t epoch = base.epoch;
  const size_t expected_sample_count = mixed.size();

  for (size_t i = 0; i < config_.input_topics.size(); ++i) {
    if (static_cast<int>(i) == config_.master_index) {
      const float g = static_cast<float>(config_.input_gains_linear[i]);
      for (float & v : mixed) {
        v *= g;
      }
      continue;
    }

    fa_interfaces::msg::AudioFrame::SharedPtr other;
    {
      std::lock_guard<std::mutex> lock(frames_mutex_);
      other = latest_frames_[i];
    }
    if (!other) {
      drop_.fetch_add(1);
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 3000,
        "Dropping mix because input %zu has no valid frame", i);
      return;
    }
    const rclcpp::Time other_time = frameStamp(*other);
    const int64_t age_ms = (base_time - other_time).nanoseconds() / 1000000;
    if (age_ms < 0 || age_ms > config_.max_frame_age_ms) {
      drop_.fetch_add(1);
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 3000,
        "Dropping mix because input %zu is stale: age_ms=%ld max_age_ms=%d",
        i,
        static_cast<long>(age_ms),
        config_.max_frame_age_ms);
      return;
    }

    std::vector<float> other_f32;
    if (!decodePcm16ToFloat(*other, other_f32)) {
      drop_.fetch_add(1);
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 3000,
        "Dropping mix because input %zu could not be decoded", i);
      return;
    }
    if (other_f32.size() != expected_sample_count) {
      drop_.fetch_add(1);
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 3000,
        "Dropping mix because input %zu sample count differs from master", i);
      return;
    }

    const float g = static_cast<float>(config_.input_gains_linear[i]);
    for (size_t s = 0; s < expected_sample_count; ++s) {
      mixed[s] += other_f32[s] * g;
    }

    epoch = std::max<uint32_t>(epoch, other->epoch);
  }

  std::vector<uint8_t> out_bytes;
  std::string encode_error;
  if (!encodeFloatToPcm16(mixed, out_bytes, encode_error)) {
    drop_.fetch_add(1);
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping mixed frame: %s. Insert an explicit dynamics/limiter node before fa_mix or lower input_gains_db.",
      encode_error.c_str());
    return;
  }

  fa_interfaces::msg::AudioFrame out;
  out.header = base.header;
  out.source_id = base.source_id;
  out.stream_id = config_.output_topic;
  out.encoding = config_.expected_encoding;
  out.sample_rate = static_cast<uint32_t>(config_.expected_sample_rate);
  out.channels = static_cast<uint32_t>(config_.expected_channels);
  out.bit_depth = static_cast<uint32_t>(config_.expected_bit_depth);
  out.layout = kInterleavedLayout;
  out.data = std::move(out_bytes);
  out.epoch = epoch;

  pub_->publish(out);
  out_.fetch_add(1);
}

void FaMixNode::publishDiagnostics()
{
  diagnostic_msgs::msg::DiagnosticArray array_msg;
  array_msg.header.stamp = this->now();

  diagnostic_msgs::msg::DiagnosticStatus status;
  status.name = "fa_mix";
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
  push_kv("inputs", std::to_string(config_.input_topics.size()));
  push_kv("master_index", std::to_string(config_.master_index));
  push_kv("output_topic", config_.output_topic);
  push_kv("expected.sample_rate", std::to_string(config_.expected_sample_rate));
  push_kv("expected.channels", std::to_string(config_.expected_channels));
  push_kv("expected.bit_depth", std::to_string(config_.expected_bit_depth));
  push_kv("expected.encoding", config_.expected_encoding);
  push_kv("frames.in", std::to_string(in_.load()));
  push_kv("frames.out", std::to_string(out_.load()));
  push_kv("frames.drop", std::to_string(drop_.load()));

  array_msg.status.push_back(status);
  diag_pub_->publish(array_msg);
}

}  // namespace fa_mix

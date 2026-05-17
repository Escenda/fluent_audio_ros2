#include "fa_jitter_buffer/fa_jitter_buffer_node.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>

#include "diagnostic_msgs/msg/diagnostic_status.hpp"
#include "diagnostic_msgs/msg/key_value.hpp"

namespace fa_jitter_buffer
{

namespace
{
constexpr const char * kEncodingFloat32 = "FLOAT32LE";
constexpr const char * kInterleavedLayout = "interleaved";

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

FaJitterBufferNode::FaJitterBufferNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("fa_jitter_buffer_node", options)
{
  RCLCPP_INFO(this->get_logger(), "Starting FA Jitter Buffer node");
  loadParameters();
  setupInterfaces();
}

void FaJitterBufferNode::loadParameters()
{
  this->declare_parameter<std::string>("input_topic");
  this->declare_parameter<std::string>("output_topic");
  this->declare_parameter<int>("expected.sample_rate");
  this->declare_parameter<int>("expected.channels");
  this->declare_parameter<std::string>("expected.encoding");
  this->declare_parameter<int>("expected.bit_depth");
  this->declare_parameter<std::string>("expected.layout");
  this->declare_parameter<int>("jitter.target_depth_frames");
  this->declare_parameter<int>("jitter.max_depth_frames");
  this->declare_parameter<bool>("jitter.reset_on_epoch_regression");
  this->declare_parameter<int>("qos.depth");
  this->declare_parameter<bool>("qos.reliable");
  this->declare_parameter<int>("diagnostics.publish_period_ms");

  config_.input_topic = this->get_parameter("input_topic").as_string();
  config_.output_topic = this->get_parameter("output_topic").as_string();
  config_.expected_sample_rate = this->get_parameter("expected.sample_rate").as_int();
  config_.expected_channels = this->get_parameter("expected.channels").as_int();
  config_.expected_encoding = this->get_parameter("expected.encoding").as_string();
  config_.expected_bit_depth = this->get_parameter("expected.bit_depth").as_int();
  config_.expected_layout = this->get_parameter("expected.layout").as_string();
  config_.target_depth_frames = this->get_parameter("jitter.target_depth_frames").as_int();
  config_.max_depth_frames = this->get_parameter("jitter.max_depth_frames").as_int();
  config_.reset_on_epoch_regression =
    this->get_parameter("jitter.reset_on_epoch_regression").as_bool();
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
  if (config_.expected_sample_rate <= 0) {
    throw std::runtime_error("expected.sample_rate must be > 0");
  }
  if (config_.expected_channels <= 0) {
    throw std::runtime_error("expected.channels must be > 0");
  }
  if (config_.expected_encoding.empty()) {
    throw std::runtime_error("expected.encoding is required");
  }
  if (config_.expected_bit_depth <= 0) {
    throw std::runtime_error("expected.bit_depth must be > 0");
  }
  if ((config_.expected_bit_depth % 8) != 0) {
    throw std::runtime_error("expected.bit_depth must be byte-aligned");
  }
  if (config_.expected_encoding == kEncodingFloat32 && config_.expected_bit_depth != 32) {
    throw std::runtime_error("FLOAT32LE requires expected.bit_depth=32");
  }
  if (config_.expected_layout.empty()) {
    throw std::runtime_error("expected.layout is required");
  }
  if (config_.target_depth_frames < 0) {
    throw std::runtime_error("jitter.target_depth_frames must be >= 0");
  }
  if (config_.max_depth_frames <= 0) {
    throw std::runtime_error("jitter.max_depth_frames must be > 0");
  }
  if (config_.max_depth_frames <= config_.target_depth_frames) {
    throw std::runtime_error("jitter.max_depth_frames must be > jitter.target_depth_frames");
  }
  if (config_.qos_depth <= 0) {
    throw std::runtime_error("qos.depth must be > 0");
  }
  if (config_.diagnostics_publish_period_ms <= 0) {
    throw std::runtime_error("diagnostics.publish_period_ms must be > 0");
  }

  RCLCPP_INFO(
    this->get_logger(),
    "Jitter buffer config: input=%s output=%s expected=%dHz/%d/%s/%d/%s target=%d max=%d "
    "reset_on_epoch_regression=%s qos_depth=%d reliable=%s diag=%dms",
    config_.input_topic.c_str(),
    config_.output_topic.c_str(),
    config_.expected_sample_rate,
    config_.expected_channels,
    config_.expected_encoding.c_str(),
    config_.expected_bit_depth,
    config_.expected_layout.c_str(),
    config_.target_depth_frames,
    config_.max_depth_frames,
    config_.reset_on_epoch_regression ? "true" : "false",
    config_.qos_depth,
    config_.qos_reliable ? "true" : "false",
    config_.diagnostics_publish_period_ms);
}

void FaJitterBufferNode::setupInterfaces()
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
    std::bind(&FaJitterBufferNode::handleFrame, this, std::placeholders::_1));

  diag_pub_ = this->create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
    "diagnostics",
    rclcpp::SystemDefaultsQoS());
  diag_timer_ = this->create_wall_timer(
    std::chrono::milliseconds(config_.diagnostics_publish_period_ms),
    std::bind(&FaJitterBufferNode::publishDiagnostics, this));
}

void FaJitterBufferNode::handleFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg)
{
  frames_in_.fetch_add(1);
  if (!msg) {
    frames_dropped_.fetch_add(1);
    return;
  }

  if (!validateFrame(*msg)) {
    frames_dropped_.fetch_add(1);
    return;
  }

  if (hasDifferentContract(*msg)) {
    RCLCPP_WARN(
      this->get_logger(),
      "AudioFrame source or format contract changed; clearing jitter buffer");
    resetBuffer();
    resets_.fetch_add(1);
  }

  if (!active_stream_.has_value()) {
    activateStream(*msg);
  }

  if (isDuplicateEpoch(*msg)) {
    duplicate_drops_.fetch_add(1);
    frames_dropped_.fetch_add(1);
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping duplicate AudioFrame epoch %u",
      msg->epoch);
    return;
  }

  if (isLateEpoch(*msg)) {
    if (!config_.reset_on_epoch_regression) {
      late_drops_.fetch_add(1);
      frames_dropped_.fetch_add(1);
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 3000,
        "Dropping late AudioFrame epoch %u after published epoch %u",
        msg->epoch,
        *last_published_epoch_);
      return;
    }

    RCLCPP_WARN(
      this->get_logger(),
      "AudioFrame epoch regressed from published epoch %u to %u; resetting jitter buffer",
      *last_published_epoch_,
      msg->epoch);
    resetBuffer();
    resets_.fetch_add(1);
    activateStream(*msg);
  }

  if (buffered_frames_.size() >= static_cast<size_t>(config_.max_depth_frames)) {
    RCLCPP_WARN(
      this->get_logger(),
      "Jitter buffer reached jitter.max_depth_frames=%d; resetting before accepting epoch %u",
      config_.max_depth_frames,
      msg->epoch);
    resetBuffer();
    max_depth_resets_.fetch_add(1);
    resets_.fetch_add(1);
    frames_dropped_.fetch_add(1);
    return;
  }

  insertFrame(*msg);
  publishReadyFrames();
}

bool FaJitterBufferNode::validateFrame(const fa_interfaces::msg::AudioFrame & msg)
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
  if (msg.encoding != config_.expected_encoding ||
      msg.bit_depth != static_cast<uint32_t>(config_.expected_bit_depth) ||
      msg.layout != config_.expected_layout)
  {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "AudioFrame encoding/layout mismatch: %s/%u/%s != %s/%d/%s",
      msg.encoding.c_str(),
      msg.bit_depth,
      msg.layout.c_str(),
      config_.expected_encoding.c_str(),
      config_.expected_bit_depth,
      config_.expected_layout.c_str());
    return false;
  }
  if (msg.data.empty() || (msg.data.size() % bytesPerSampleFrame()) != 0U) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "AudioFrame data size is empty or not aligned to configured sample frames");
    return false;
  }
  if (msg.encoding == kEncodingFloat32 && msg.layout == kInterleavedLayout &&
      !validateFloat32InterleavedSamples(msg.data))
  {
    return false;
  }
  return true;
}

bool FaJitterBufferNode::validateFloat32InterleavedSamples(
  const std::vector<uint8_t> & data)
{
  for (size_t offset = 0; offset < data.size(); offset += sizeof(float)) {
    const float sample = readFloat32LeSample(data, offset);
    if (!std::isfinite(sample) || sample < -1.0F || sample > 1.0F) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 3000,
        "AudioFrame contains non-finite or non-normalized FLOAT32LE sample");
      return false;
    }
  }
  return true;
}

bool FaJitterBufferNode::hasDifferentContract(const fa_interfaces::msg::AudioFrame & msg) const
{
  if (!active_stream_.has_value()) {
    return false;
  }
  return msg.source_id != active_stream_->source_id ||
         msg.stream_id != active_stream_->stream_id ||
         msg.sample_rate != active_stream_->sample_rate ||
         msg.channels != active_stream_->channels ||
         msg.encoding != active_stream_->encoding ||
         msg.bit_depth != active_stream_->bit_depth ||
         msg.layout != active_stream_->layout;
}

bool FaJitterBufferNode::isDuplicateEpoch(const fa_interfaces::msg::AudioFrame & msg) const
{
  return buffered_frames_.find(msg.epoch) != buffered_frames_.end() ||
         (last_published_epoch_.has_value() && msg.epoch == *last_published_epoch_);
}

bool FaJitterBufferNode::isLateEpoch(const fa_interfaces::msg::AudioFrame & msg) const
{
  return last_published_epoch_.has_value() && msg.epoch < *last_published_epoch_;
}

void FaJitterBufferNode::activateStream(const fa_interfaces::msg::AudioFrame & msg)
{
  ActiveStreamIdentity identity;
  identity.source_id = msg.source_id;
  identity.stream_id = msg.stream_id;
  identity.sample_rate = msg.sample_rate;
  identity.channels = msg.channels;
  identity.encoding = msg.encoding;
  identity.bit_depth = msg.bit_depth;
  identity.layout = msg.layout;
  active_stream_ = identity;
}

void FaJitterBufferNode::resetBuffer()
{
  buffered_frames_.clear();
  active_stream_.reset();
  last_published_epoch_.reset();
  buffered_frame_count_.store(0);
}

void FaJitterBufferNode::insertFrame(const fa_interfaces::msg::AudioFrame & msg)
{
  const auto inserted = buffered_frames_.emplace(msg.epoch, msg);
  if (!inserted.second) {
    duplicate_drops_.fetch_add(1);
    frames_dropped_.fetch_add(1);
    return;
  }
  buffered_frame_count_.store(buffered_frames_.size());
}

void FaJitterBufferNode::publishReadyFrames()
{
  while (buffered_frames_.size() > static_cast<size_t>(config_.target_depth_frames)) {
    auto oldest = buffered_frames_.begin();
    fa_interfaces::msg::AudioFrame out = oldest->second;
    out.stream_id = config_.output_topic;
    audio_pub_->publish(out);
    last_published_epoch_ = oldest->first;
    buffered_frames_.erase(oldest);
    frames_out_.fetch_add(1);
  }
  buffered_frame_count_.store(buffered_frames_.size());
}

float FaJitterBufferNode::readFloat32LeSample(
  const std::vector<uint8_t> & data,
  const size_t byte_offset) const
{
  const uint32_t raw =
    static_cast<uint32_t>(data[byte_offset]) |
    (static_cast<uint32_t>(data[byte_offset + 1U]) << 8U) |
    (static_cast<uint32_t>(data[byte_offset + 2U]) << 16U) |
    (static_cast<uint32_t>(data[byte_offset + 3U]) << 24U);
  float sample = 0.0F;
  std::memcpy(&sample, &raw, sizeof(float));
  return sample;
}

size_t FaJitterBufferNode::bytesPerSampleFrame() const
{
  return static_cast<size_t>(config_.expected_channels) *
         (static_cast<size_t>(config_.expected_bit_depth) / 8U);
}

void FaJitterBufferNode::publishDiagnostics()
{
  diagnostic_msgs::msg::DiagnosticArray array_msg;
  array_msg.header.stamp = this->now();

  diagnostic_msgs::msg::DiagnosticStatus status;
  status.name = "fa_jitter_buffer";
  status.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
  status.message = "running";
  status.values.reserve(17);
  pushKeyValue(status, "input_topic", config_.input_topic);
  pushKeyValue(status, "output_topic", config_.output_topic);
  pushKeyValue(status, "expected_sample_rate", std::to_string(config_.expected_sample_rate));
  pushKeyValue(status, "expected_channels", std::to_string(config_.expected_channels));
  pushKeyValue(status, "expected_encoding", config_.expected_encoding);
  pushKeyValue(status, "expected_bit_depth", std::to_string(config_.expected_bit_depth));
  pushKeyValue(status, "expected_layout", config_.expected_layout);
  pushKeyValue(status, "target_depth_frames", std::to_string(config_.target_depth_frames));
  pushKeyValue(status, "max_depth_frames", std::to_string(config_.max_depth_frames));
  pushKeyValue(
    status,
    "reset_on_epoch_regression",
    config_.reset_on_epoch_regression ? "true" : "false");
  pushKeyValue(status, "buffered_frames", std::to_string(buffered_frame_count_.load()));
  pushKeyValue(status, "frames_in", std::to_string(frames_in_.load()));
  pushKeyValue(status, "frames_out", std::to_string(frames_out_.load()));
  pushKeyValue(status, "frames_dropped", std::to_string(frames_dropped_.load()));
  pushKeyValue(status, "duplicate_drops", std::to_string(duplicate_drops_.load()));
  pushKeyValue(status, "late_drops", std::to_string(late_drops_.load()));
  pushKeyValue(status, "max_depth_resets", std::to_string(max_depth_resets_.load()));
  pushKeyValue(status, "resets", std::to_string(resets_.load()));

  array_msg.status.push_back(status);
  diag_pub_->publish(array_msg);
}

}  // namespace fa_jitter_buffer

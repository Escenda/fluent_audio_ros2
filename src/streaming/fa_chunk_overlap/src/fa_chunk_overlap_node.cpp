#include "fa_chunk_overlap/fa_chunk_overlap_node.hpp"

#include <chrono>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

#include "builtin_interfaces/msg/time.hpp"
#include "diagnostic_msgs/msg/diagnostic_status.hpp"
#include "diagnostic_msgs/msg/key_value.hpp"

namespace fa_chunk_overlap
{

namespace
{
constexpr const char * kEncodingFloat32 = "FLOAT32LE";
constexpr const char * kInterleavedLayout = "interleaved";
constexpr int64_t kNanosecondsPerSecond = 1000000000LL;
constexpr int64_t kMaxBuiltinTimeNanoseconds = (2147483647LL * kNanosecondsPerSecond) + 999999999LL;
constexpr long double kNanosecondsPerSampleRateUnit = 1000000000.0L;

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
    throw std::runtime_error(name + " must be a string parameter");
  }
  return parameter.as_string();
}

int readRequiredInt(const rclcpp::Node & node, const std::string & name)
{
  const rclcpp::Parameter parameter = getRequiredParameter(node, name);
  if (parameter.get_type() != rclcpp::ParameterType::PARAMETER_INTEGER) {
    throw std::runtime_error(name + " must be an integer parameter");
  }
  const int64_t value = parameter.as_int();
  if (value < static_cast<int64_t>(std::numeric_limits<int>::min()) ||
      value > static_cast<int64_t>(std::numeric_limits<int>::max()))
  {
    throw std::runtime_error(name + " must fit in a 32-bit signed integer");
  }
  return static_cast<int>(value);
}

bool readRequiredBool(const rclcpp::Node & node, const std::string & name)
{
  const rclcpp::Parameter parameter = getRequiredParameter(node, name);
  if (parameter.get_type() != rclcpp::ParameterType::PARAMETER_BOOL) {
    throw std::runtime_error(name + " must be a bool parameter");
  }
  return parameter.as_bool();
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

int64_t stampToNanoseconds(const builtin_interfaces::msg::Time & stamp)
{
  return (static_cast<int64_t>(stamp.sec) * kNanosecondsPerSecond) +
         static_cast<int64_t>(stamp.nanosec);
}

builtin_interfaces::msg::Time nanosecondsToStamp(const int64_t nanoseconds)
{
  builtin_interfaces::msg::Time stamp;
  stamp.sec = static_cast<int32_t>(nanoseconds / kNanosecondsPerSecond);
  stamp.nanosec = static_cast<uint32_t>(nanoseconds % kNanosecondsPerSecond);
  return stamp;
}
}  // namespace

FaChunkOverlapNode::FaChunkOverlapNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("fa_chunk_overlap", options)
{
  RCLCPP_INFO(this->get_logger(), "Starting FA Chunk Overlap node");
  loadParameters();
  setupInterfaces();
}

void FaChunkOverlapNode::loadParameters()
{
  this->declare_parameter<std::string>("input_topic");
  this->declare_parameter<std::string>("output_topic");
  this->declare_parameter<int>("expected.sample_rate");
  this->declare_parameter<int>("expected.channels");
  this->declare_parameter<std::string>("expected.encoding");
  this->declare_parameter<int>("expected.bit_depth");
  this->declare_parameter<std::string>("expected.layout");
  this->declare_parameter<int>("window.frame_samples");
  this->declare_parameter<int>("window.hop_samples");
  this->declare_parameter<int>("qos.depth");
  this->declare_parameter<bool>("qos.reliable");
  this->declare_parameter<int>("diagnostics.publish_period_ms");
  this->declare_parameter<int>("diagnostics.qos.depth");
  this->declare_parameter<bool>("diagnostics.qos.reliable");

  config_.input_topic = readRequiredString(*this, "input_topic");
  config_.output_topic = readRequiredString(*this, "output_topic");
  config_.expected_sample_rate = readRequiredInt(*this, "expected.sample_rate");
  config_.expected_channels = readRequiredInt(*this, "expected.channels");
  config_.expected_encoding = readRequiredString(*this, "expected.encoding");
  config_.expected_bit_depth = readRequiredInt(*this, "expected.bit_depth");
  config_.expected_layout = readRequiredString(*this, "expected.layout");
  config_.frame_samples = readRequiredInt(*this, "window.frame_samples");
  config_.hop_samples = readRequiredInt(*this, "window.hop_samples");
  config_.qos_depth = readRequiredInt(*this, "qos.depth");
  config_.qos_reliable = readRequiredBool(*this, "qos.reliable");
  config_.diagnostics_publish_period_ms = readRequiredInt(
    *this,
    "diagnostics.publish_period_ms");
  config_.diagnostics_qos_depth = readRequiredInt(*this, "diagnostics.qos.depth");
  config_.diagnostics_qos_reliable = readRequiredBool(*this, "diagnostics.qos.reliable");

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
  if (config_.expected_encoding != kEncodingFloat32) {
    throw std::runtime_error("fa_chunk_overlap requires expected.encoding=FLOAT32LE");
  }
  if (config_.expected_bit_depth != 32) {
    throw std::runtime_error("fa_chunk_overlap requires expected.bit_depth=32");
  }
  if (config_.expected_layout != kInterleavedLayout) {
    throw std::runtime_error("fa_chunk_overlap requires expected.layout=interleaved");
  }
  if (config_.frame_samples <= 0) {
    throw std::runtime_error("window.frame_samples must be > 0");
  }
  if (config_.hop_samples <= 0) {
    throw std::runtime_error("window.hop_samples must be > 0");
  }
  if (config_.hop_samples > config_.frame_samples) {
    throw std::runtime_error("window.hop_samples must be <= window.frame_samples");
  }
  if (config_.qos_depth <= 0) {
    throw std::runtime_error("qos.depth must be > 0");
  }
  if (config_.diagnostics_publish_period_ms <= 0) {
    throw std::runtime_error("diagnostics.publish_period_ms must be > 0");
  }
  if (config_.diagnostics_qos_depth <= 0) {
    throw std::runtime_error("diagnostics.qos.depth must be > 0");
  }

  RCLCPP_INFO(
    this->get_logger(),
    "Chunk overlap config: input=%s output=%s expected=%dHz/%d/%s/%d/%s "
    "frame_samples=%d hop_samples=%d qos_depth=%d reliable=%s diag=%dms",
    config_.input_topic.c_str(),
    config_.output_topic.c_str(),
    config_.expected_sample_rate,
    config_.expected_channels,
    config_.expected_encoding.c_str(),
    config_.expected_bit_depth,
    config_.expected_layout.c_str(),
    config_.frame_samples,
    config_.hop_samples,
    config_.qos_depth,
    config_.qos_reliable ? "true" : "false",
    config_.diagnostics_publish_period_ms);
}

void FaChunkOverlapNode::setupInterfaces()
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
    std::bind(&FaChunkOverlapNode::handleFrame, this, std::placeholders::_1));

  rclcpp::QoS diagnostics_qos(static_cast<size_t>(config_.diagnostics_qos_depth));
  if (config_.diagnostics_qos_reliable) {
    diagnostics_qos.reliable();
  } else {
    diagnostics_qos.best_effort();
  }

  diag_pub_ = this->create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
    "diagnostics",
    diagnostics_qos);
  diag_timer_ = this->create_wall_timer(
    std::chrono::milliseconds(config_.diagnostics_publish_period_ms),
    std::bind(&FaChunkOverlapNode::publishDiagnostics, this));
}

void FaChunkOverlapNode::handleFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg)
{
  frames_in_.fetch_add(1);
  if (!msg) {
    input_frames_dropped_.fetch_add(1);
    return;
  }

  if (!validateFrame(*msg)) {
    input_frames_dropped_.fetch_add(1);
    return;
  }

  if (hasDifferentSource(*msg)) {
    RCLCPP_WARN(
      this->get_logger(),
      "Audio source_id changed from %s to %s; clearing buffered overlap state",
      active_stream_->source_id.c_str(),
      msg->source_id.c_str());
    resetActiveBuffer();
    source_resets_.fetch_add(1);
  }

  if (!active_stream_.has_value()) {
    activateStream(*msg);
  }

  appendFrame(*msg);
  input_frames_accepted_.fetch_add(1);
  publishAvailableChunks();
}

bool FaChunkOverlapNode::validateFrame(const fa_interfaces::msg::AudioFrame & msg)
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
  if (msg.data.empty() || (msg.data.size() % bytesPerSampleFrame()) != 0) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "AudioFrame data size is empty or not aligned to FLOAT32LE interleaved sample frames");
    return false;
  }

  for (size_t offset = 0; offset < msg.data.size(); offset += sizeof(float)) {
    const float sample = readFloat32LeSample(msg.data, offset);
    if (!std::isfinite(sample) || sample < -1.0F || sample > 1.0F) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 3000,
        "AudioFrame contains non-finite or non-normalized FLOAT32LE sample");
      return false;
    }
  }

  return true;
}

bool FaChunkOverlapNode::hasDifferentSource(const fa_interfaces::msg::AudioFrame & msg) const
{
  return active_stream_.has_value() && msg.source_id != active_stream_->source_id;
}

void FaChunkOverlapNode::activateStream(const fa_interfaces::msg::AudioFrame & msg)
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

void FaChunkOverlapNode::appendFrame(const fa_interfaces::msg::AudioFrame & msg)
{
  buffered_segments_.push_back(BufferedSegment{msg.header, msg.data.size()});
  buffer_.insert(buffer_.end(), msg.data.begin(), msg.data.end());
  buffered_sample_frames_.store(buffer_.size() / bytesPerSampleFrame());
}

void FaChunkOverlapNode::publishAvailableChunks()
{
  const size_t bytes_per_window = windowBytes();
  const size_t bytes_per_hop = hopBytes();
  while (
    active_stream_.has_value() &&
    !buffered_segments_.empty() &&
    buffer_.size() >= bytes_per_window)
  {
    fa_interfaces::msg::AudioFrame out;
    out.header = buffered_segments_.front().header;
    out.source_id = active_stream_->source_id;
    out.stream_id = config_.output_topic;
    out.encoding = active_stream_->encoding;
    out.sample_rate = active_stream_->sample_rate;
    out.channels = active_stream_->channels;
    out.bit_depth = active_stream_->bit_depth;
    out.layout = active_stream_->layout;
    out.epoch = next_output_epoch_;
    out.data.assign(
      buffer_.begin(),
      buffer_.begin() + static_cast<std::ptrdiff_t>(bytes_per_window));

    audio_pub_->publish(out);
    next_output_epoch_ += 1U;
    chunks_out_.fetch_add(1);
    sample_frames_out_.fetch_add(static_cast<uint64_t>(config_.frame_samples));
    buffer_.erase(buffer_.begin(), buffer_.begin() + static_cast<std::ptrdiff_t>(bytes_per_hop));
    if (!consumeBufferedBytes(bytes_per_hop)) {
      resetActiveBuffer();
      return;
    }
  }
  buffered_sample_frames_.store(buffer_.size() / bytesPerSampleFrame());
}

void FaChunkOverlapNode::resetActiveBuffer()
{
  buffer_.clear();
  buffered_segments_.clear();
  active_stream_.reset();
  buffered_sample_frames_.store(0);
}

bool FaChunkOverlapNode::advanceSegmentHeader(BufferedSegment & segment, const size_t consumed_bytes)
{
  if (consumed_bytes == 0U) {
    return true;
  }
  const size_t bytes_per_sample_frame = bytesPerSampleFrame();
  if (bytes_per_sample_frame == 0U || (consumed_bytes % bytes_per_sample_frame) != 0U) {
    RCLCPP_WARN(
      this->get_logger(),
      "Internal overlap buffer consumed bytes are not aligned to sample frames; clearing buffer");
    return false;
  }
  const size_t consumed_sample_frames = consumed_bytes / bytes_per_sample_frame;
  const long double advance_ns_decimal =
    (static_cast<long double>(consumed_sample_frames) * kNanosecondsPerSampleRateUnit) /
    static_cast<long double>(config_.expected_sample_rate);
  if (!std::isfinite(static_cast<double>(advance_ns_decimal)) ||
      advance_ns_decimal < 0.0L ||
      advance_ns_decimal > static_cast<long double>(std::numeric_limits<int64_t>::max()))
  {
    RCLCPP_WARN(
      this->get_logger(),
      "Internal overlap buffer timestamp advance is outside int64 nanosecond range; clearing buffer");
    return false;
  }

  const int64_t advance_ns = static_cast<int64_t>(std::llround(advance_ns_decimal));
  const int64_t current_ns = stampToNanoseconds(segment.header.stamp);
  if (current_ns < 0 || advance_ns <= 0 || current_ns > kMaxBuiltinTimeNanoseconds - advance_ns) {
    RCLCPP_WARN(
      this->get_logger(),
      "Internal overlap buffer timestamp advance exceeds builtin_interfaces/Time range; clearing buffer");
    return false;
  }

  const int64_t advanced_ns = current_ns + advance_ns;
  segment.header.stamp = nanosecondsToStamp(advanced_ns);
  return true;
}

bool FaChunkOverlapNode::consumeBufferedBytes(size_t byte_count)
{
  size_t remaining = byte_count;
  while (remaining > 0 && !buffered_segments_.empty()) {
    BufferedSegment & segment = buffered_segments_.front();
    if (remaining < segment.byte_count) {
      if (!advanceSegmentHeader(segment, remaining)) {
        return false;
      }
      segment.byte_count -= remaining;
      remaining = 0;
    } else {
      remaining -= segment.byte_count;
      buffered_segments_.pop_front();
    }
  }
  return true;
}

float FaChunkOverlapNode::readFloat32LeSample(
  const std::vector<uint8_t> & data,
  size_t byte_offset) const
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

size_t FaChunkOverlapNode::bytesPerSampleFrame() const
{
  return static_cast<size_t>(config_.expected_channels) * sizeof(float);
}

size_t FaChunkOverlapNode::windowBytes() const
{
  return static_cast<size_t>(config_.frame_samples) * bytesPerSampleFrame();
}

size_t FaChunkOverlapNode::hopBytes() const
{
  return static_cast<size_t>(config_.hop_samples) * bytesPerSampleFrame();
}

void FaChunkOverlapNode::publishDiagnostics()
{
  diagnostic_msgs::msg::DiagnosticArray array_msg;
  array_msg.header.stamp = this->now();

  diagnostic_msgs::msg::DiagnosticStatus status;
  status.name = "fa_chunk_overlap";
  status.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
  status.message = "running";
  status.values.reserve(14);
  pushKeyValue(status, "input_topic", config_.input_topic);
  pushKeyValue(status, "output_topic", config_.output_topic);
  pushKeyValue(status, "frame_samples", std::to_string(config_.frame_samples));
  pushKeyValue(status, "hop_samples", std::to_string(config_.hop_samples));
  pushKeyValue(
    status,
    "overlap_samples",
    std::to_string(config_.frame_samples - config_.hop_samples));
  pushKeyValue(status, "window_bytes", std::to_string(windowBytes()));
  pushKeyValue(status, "hop_bytes", std::to_string(hopBytes()));
  pushKeyValue(status, "buffered_bytes", std::to_string(buffer_.size()));
  pushKeyValue(status, "buffered_sample_frames", std::to_string(buffered_sample_frames_.load()));
  pushKeyValue(status, "frames_in", std::to_string(frames_in_.load()));
  pushKeyValue(status, "input_frames_accepted", std::to_string(input_frames_accepted_.load()));
  pushKeyValue(status, "input_frames_dropped", std::to_string(input_frames_dropped_.load()));
  pushKeyValue(status, "chunks_out", std::to_string(chunks_out_.load()));
  pushKeyValue(status, "sample_frames_out", std::to_string(sample_frames_out_.load()));
  pushKeyValue(status, "source_resets", std::to_string(source_resets_.load()));

  array_msg.status.push_back(status);
  diag_pub_->publish(array_msg);
}

}  // namespace fa_chunk_overlap

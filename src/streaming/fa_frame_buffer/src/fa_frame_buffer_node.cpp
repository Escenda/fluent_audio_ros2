#include "fa_frame_buffer/fa_frame_buffer_node.hpp"

#include <algorithm>
#include <chrono>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>

#include "diagnostic_msgs/msg/diagnostic_status.hpp"
#include "diagnostic_msgs/msg/key_value.hpp"

namespace fa_frame_buffer
{

namespace
{
constexpr const char * kEncodingFloat32 = "FLOAT32LE";
constexpr const char * kInterleavedLayout = "interleaved";

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

std::string identityWithoutLeadingSlash(const std::string & value)
{
  if (!value.empty() && value.front() == '/') {
    return value.substr(1);
  }
  return value;
}

bool sameIdentityString(const std::string & left, const std::string & right)
{
  return identityWithoutLeadingSlash(left) == identityWithoutLeadingSlash(right);
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

FaFrameBufferNode::FaFrameBufferNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("fa_frame_buffer", options)
{
  RCLCPP_INFO(this->get_logger(), "Starting FA Frame Buffer node");
  loadParameters();
  setupInterfaces();
}

void FaFrameBufferNode::loadParameters()
{
  this->declare_parameter<std::string>("input_topic");
  this->declare_parameter<std::string>("output_topic");
  this->declare_parameter<std::string>("input_stream_id");
  this->declare_parameter<std::string>("output.stream_id");
  this->declare_parameter<int>("expected.sample_rate");
  this->declare_parameter<int>("expected.channels");
  this->declare_parameter<std::string>("expected.encoding");
  this->declare_parameter<int>("expected.bit_depth");
  this->declare_parameter<std::string>("expected.layout");
  this->declare_parameter<int>("buffering.frames_per_chunk");
  this->declare_parameter<int>("buffering.max_buffered_chunks");
  this->declare_parameter<int>("qos.depth");
  this->declare_parameter<bool>("qos.reliable");
  this->declare_parameter<int>("diagnostics.publish_period_ms");
  this->declare_parameter<int>("diagnostics.qos.depth");
  this->declare_parameter<bool>("diagnostics.qos.reliable");

  config_.input_topic = readRequiredString(*this, "input_topic");
  config_.output_topic = readRequiredString(*this, "output_topic");
  config_.input_stream_id = readRequiredString(*this, "input_stream_id");
  config_.output_stream_id = readRequiredString(*this, "output.stream_id");
  config_.expected_sample_rate = readRequiredInt(*this, "expected.sample_rate");
  config_.expected_channels = readRequiredInt(*this, "expected.channels");
  config_.expected_encoding = readRequiredString(*this, "expected.encoding");
  config_.expected_bit_depth = readRequiredInt(*this, "expected.bit_depth");
  config_.expected_layout = readRequiredString(*this, "expected.layout");
  config_.frames_per_chunk = readRequiredInt(*this, "buffering.frames_per_chunk");
  config_.max_buffered_chunks = readRequiredInt(*this, "buffering.max_buffered_chunks");
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
  if (config_.input_stream_id.empty()) {
    throw std::runtime_error("input_stream_id is required");
  }
  if (config_.output_stream_id.empty()) {
    throw std::runtime_error("output.stream_id is required");
  }
  const std::string resolved_input_topic =
    this->get_node_topics_interface()->resolve_topic_name(config_.input_topic);
  const std::string resolved_output_topic =
    this->get_node_topics_interface()->resolve_topic_name(config_.output_topic);
  if (sameIdentityString(config_.input_stream_id, config_.input_topic) ||
      sameIdentityString(config_.input_stream_id, config_.output_topic) ||
      sameIdentityString(config_.input_stream_id, resolved_input_topic) ||
      sameIdentityString(config_.input_stream_id, resolved_output_topic)) {
    throw std::runtime_error("input_stream_id must be distinct from ROS topics");
  }
  if (sameIdentityString(config_.output_stream_id, config_.input_topic) ||
      sameIdentityString(config_.output_stream_id, config_.output_topic) ||
      sameIdentityString(config_.output_stream_id, resolved_input_topic) ||
      sameIdentityString(config_.output_stream_id, resolved_output_topic)) {
    throw std::runtime_error("output.stream_id must be distinct from ROS topics");
  }
  if (sameIdentityString(config_.input_stream_id, config_.output_stream_id)) {
    throw std::runtime_error("input_stream_id and output.stream_id must be distinct");
  }
  if (config_.expected_sample_rate <= 0) {
    throw std::runtime_error("expected.sample_rate must be > 0");
  }
  if (config_.expected_channels <= 0) {
    throw std::runtime_error("expected.channels must be > 0");
  }
  if (config_.expected_encoding != kEncodingFloat32) {
    throw std::runtime_error("fa_frame_buffer requires expected.encoding=FLOAT32LE");
  }
  if (config_.expected_bit_depth != 32) {
    throw std::runtime_error("fa_frame_buffer requires expected.bit_depth=32");
  }
  if (config_.expected_layout != kInterleavedLayout) {
    throw std::runtime_error("fa_frame_buffer requires expected.layout=interleaved");
  }
  if (config_.frames_per_chunk <= 0) {
    throw std::runtime_error("buffering.frames_per_chunk must be > 0");
  }
  if (config_.max_buffered_chunks <= 0) {
    throw std::runtime_error("buffering.max_buffered_chunks must be > 0");
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
    "Frame buffer config: input=%s/%s output=%s/%s expected=%dHz/%d/%s/%d/%s chunk_frames=%d "
    "max_chunks=%d qos_depth=%d reliable=%s diag=%dms",
    config_.input_topic.c_str(),
    config_.input_stream_id.c_str(),
    config_.output_topic.c_str(),
    config_.output_stream_id.c_str(),
    config_.expected_sample_rate,
    config_.expected_channels,
    config_.expected_encoding.c_str(),
    config_.expected_bit_depth,
    config_.expected_layout.c_str(),
    config_.frames_per_chunk,
    config_.max_buffered_chunks,
    config_.qos_depth,
    config_.qos_reliable ? "true" : "false",
    config_.diagnostics_publish_period_ms);
}

void FaFrameBufferNode::setupInterfaces()
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
    std::bind(&FaFrameBufferNode::handleFrame, this, std::placeholders::_1));

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
    std::bind(&FaFrameBufferNode::publishDiagnostics, this));
}

void FaFrameBufferNode::handleFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg)
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

  if (!isCompatibleWithBufferedStream(*msg)) {
    RCLCPP_WARN(
      this->get_logger(),
      "Dropping buffered audio because incoming stream identity or contract changed while partial data exists");
    clearBufferedStream();
    buffer_resets_.fetch_add(1);
  }

  appendFrame(*msg);
  publishAvailableChunks();
}

bool FaFrameBufferNode::validateFrame(const fa_interfaces::msg::AudioFrame & msg)
{
  if (msg.source_id.empty() || msg.stream_id.empty()) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "AudioFrame source_id and stream_id are required");
    return false;
  }
  if (msg.stream_id != config_.input_stream_id) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "AudioFrame stream_id mismatch: %s != %s",
      msg.stream_id.c_str(),
      config_.input_stream_id.c_str());
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
  if (msg.encoding != config_.expected_encoding ||
      msg.bit_depth != static_cast<uint32_t>(config_.expected_bit_depth))
  {
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
  if (msg.data.empty() || (msg.data.size() % bytesPerFrame()) != 0) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "AudioFrame data size is invalid for FLOAT32LE interleaved samples");
    return false;
  }
  return true;
}

bool FaFrameBufferNode::isCompatibleWithBufferedStream(const fa_interfaces::msg::AudioFrame & msg) const
{
  if (buffer_.empty() || buffered_segments_.empty()) {
    return true;
  }

  const BufferedFrameIdentity & identity = buffered_segments_.front().identity;
  return msg.source_id == identity.source_id &&
         msg.sample_rate == identity.sample_rate &&
         msg.channels == identity.channels &&
         msg.encoding == identity.encoding &&
         msg.bit_depth == identity.bit_depth &&
         msg.layout == identity.layout;
}

void FaFrameBufferNode::appendFrame(const fa_interfaces::msg::AudioFrame & msg)
{
  buffered_segments_.push_back(BufferedSegment{identityFromFrame(msg), msg.data.size()});
  buffer_.insert(buffer_.end(), msg.data.begin(), msg.data.end());
  while (buffer_.size() > maxBufferedBytes()) {
    dropOldestChunkForOverflow();
  }
  partial_frames_buffered_.store(buffer_.size() / bytesPerFrame());
}

void FaFrameBufferNode::publishAvailableChunks()
{
  const size_t bytes_per_chunk = chunkBytes();
  while (buffer_.size() >= bytes_per_chunk) {
    const BufferedFrameIdentity & identity = buffered_segments_.front().identity;
    fa_interfaces::msg::AudioFrame out;
    out.header = identity.header;
    out.source_id = identity.source_id;
    out.stream_id = config_.output_stream_id;
    out.encoding = identity.encoding;
    out.sample_rate = identity.sample_rate;
    out.channels = identity.channels;
    out.bit_depth = identity.bit_depth;
    out.layout = identity.layout;
    out.epoch = identity.epoch;
    out.data.assign(buffer_.begin(), buffer_.begin() + static_cast<std::ptrdiff_t>(bytes_per_chunk));

    audio_pub_->publish(out);
    frames_out_.fetch_add(static_cast<uint64_t>(config_.frames_per_chunk));
    chunks_out_.fetch_add(1);
    buffer_.erase(buffer_.begin(), buffer_.begin() + static_cast<std::ptrdiff_t>(bytes_per_chunk));
    consumeBufferedBytes(bytes_per_chunk);
  }
  partial_frames_buffered_.store(buffer_.size() / bytesPerFrame());
}

void FaFrameBufferNode::dropOldestChunkForOverflow()
{
  const size_t bytes_to_drop = std::min(chunkBytes(), buffer_.size());
  buffer_.erase(buffer_.begin(), buffer_.begin() + static_cast<std::ptrdiff_t>(bytes_to_drop));
  overflow_count_.fetch_add(1);
  frames_dropped_.fetch_add(static_cast<uint64_t>(bytes_to_drop / bytesPerFrame()));
  RCLCPP_WARN(
    this->get_logger(),
    "Frame buffer overflow: dropped oldest %zu bytes of buffered audio", bytes_to_drop);
  consumeBufferedBytes(bytes_to_drop);
}

void FaFrameBufferNode::clearBufferedStream()
{
  buffer_.clear();
  buffered_segments_.clear();
  partial_frames_buffered_.store(0);
}

void FaFrameBufferNode::consumeBufferedBytes(size_t byte_count)
{
  size_t remaining = byte_count;
  while (remaining > 0 && !buffered_segments_.empty()) {
    BufferedSegment & segment = buffered_segments_.front();
    if (remaining < segment.byte_count) {
      segment.byte_count -= remaining;
      remaining = 0;
    } else {
      remaining -= segment.byte_count;
      buffered_segments_.pop_front();
    }
  }
}

BufferedFrameIdentity FaFrameBufferNode::identityFromFrame(const fa_interfaces::msg::AudioFrame & msg) const
{
  BufferedFrameIdentity identity;
  identity.header = msg.header;
  identity.source_id = msg.source_id;
  identity.sample_rate = msg.sample_rate;
  identity.channels = msg.channels;
  identity.encoding = msg.encoding;
  identity.bit_depth = msg.bit_depth;
  identity.layout = msg.layout;
  identity.epoch = msg.epoch;
  return identity;
}

size_t FaFrameBufferNode::bytesPerFrame() const
{
  return static_cast<size_t>(config_.expected_channels) * sizeof(float);
}

size_t FaFrameBufferNode::chunkBytes() const
{
  return static_cast<size_t>(config_.frames_per_chunk) * bytesPerFrame();
}

size_t FaFrameBufferNode::maxBufferedBytes() const
{
  return static_cast<size_t>(config_.max_buffered_chunks) * chunkBytes();
}

void FaFrameBufferNode::publishDiagnostics()
{
  diagnostic_msgs::msg::DiagnosticArray array_msg;
  array_msg.header.stamp = this->now();

  diagnostic_msgs::msg::DiagnosticStatus status;
  status.name = "fa_frame_buffer";
  status.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
  status.message = "running";
  status.values.reserve(15);
  pushKeyValue(status, "input_topic", config_.input_topic);
  pushKeyValue(status, "output_topic", config_.output_topic);
  pushKeyValue(status, "input_stream_id", config_.input_stream_id);
  pushKeyValue(status, "output_stream_id", config_.output_stream_id);
  pushKeyValue(status, "frames_per_chunk", std::to_string(config_.frames_per_chunk));
  pushKeyValue(status, "max_buffered_chunks", std::to_string(config_.max_buffered_chunks));
  pushKeyValue(status, "chunk_bytes", std::to_string(chunkBytes()));
  pushKeyValue(status, "buffered_bytes", std::to_string(buffer_.size()));
  pushKeyValue(status, "partial_frames_buffered", std::to_string(partial_frames_buffered_.load()));
  pushKeyValue(status, "frames_in", std::to_string(frames_in_.load()));
  pushKeyValue(status, "frames_out", std::to_string(frames_out_.load()));
  pushKeyValue(status, "frames_dropped", std::to_string(frames_dropped_.load()));
  pushKeyValue(status, "chunks_out", std::to_string(chunks_out_.load()));
  pushKeyValue(status, "overflow_count", std::to_string(overflow_count_.load()));
  pushKeyValue(status, "buffer_resets", std::to_string(buffer_resets_.load()));

  array_msg.status.push_back(status);
  diag_pub_->publish(array_msg);
}

}  // namespace fa_frame_buffer

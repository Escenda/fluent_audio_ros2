#include "fa_packet_loss_concealment/fa_packet_loss_concealment_node.hpp"

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

namespace fa_packet_loss_concealment
{

namespace
{
constexpr const char * kEncodingFloat32 = "FLOAT32LE";
constexpr const char * kInterleavedLayout = "interleaved";
constexpr const char * kBackendName = "repeat_attenuation_plc";
constexpr int64_t kNanosecondsPerSecond = 1000000000LL;
constexpr int64_t kMaxBuiltinTimeNanoseconds =
  (2147483647LL * kNanosecondsPerSecond) + 999999999LL;
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

double readRequiredDouble(const rclcpp::Node & node, const std::string & name)
{
  const rclcpp::Parameter parameter = getRequiredParameter(node, name);
  if (parameter.get_type() != rclcpp::ParameterType::PARAMETER_DOUBLE) {
    throw std::runtime_error(name + " must be a double parameter");
  }
  return parameter.as_double();
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

FaPacketLossConcealmentNode::FaPacketLossConcealmentNode(
  const rclcpp::NodeOptions & options)
: rclcpp::Node("fa_packet_loss_concealment_node", options)
{
  RCLCPP_INFO(this->get_logger(), "Starting FA Packet Loss Concealment node");
  loadParameters();
  setupInterfaces();
}

void FaPacketLossConcealmentNode::loadParameters()
{
  this->declare_parameter<std::string>("input_topic");
  this->declare_parameter<std::string>("output_topic");
  this->declare_parameter<int>("expected.sample_rate");
  this->declare_parameter<int>("expected.channels");
  this->declare_parameter<std::string>("expected.encoding");
  this->declare_parameter<int>("expected.bit_depth");
  this->declare_parameter<std::string>("expected.layout");
  this->declare_parameter<int>("plc.max_gap_frames");
  this->declare_parameter<double>("plc.attenuation_per_gap");
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
  config_.max_gap_frames = readRequiredInt(*this, "plc.max_gap_frames");
  config_.attenuation_per_gap = readRequiredDouble(*this, "plc.attenuation_per_gap");
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
    throw std::runtime_error(
      "fa_packet_loss_concealment requires expected.encoding=FLOAT32LE");
  }
  if (config_.expected_bit_depth != 32) {
    throw std::runtime_error("fa_packet_loss_concealment requires expected.bit_depth=32");
  }
  if (config_.expected_layout != kInterleavedLayout) {
    throw std::runtime_error(
      "fa_packet_loss_concealment requires expected.layout=interleaved");
  }
  if (config_.max_gap_frames < 0) {
    throw std::runtime_error("plc.max_gap_frames must be >= 0");
  }
  if (!std::isfinite(config_.attenuation_per_gap) ||
      config_.attenuation_per_gap <= 0.0 ||
      config_.attenuation_per_gap > 1.0)
  {
    throw std::runtime_error(
      "plc.attenuation_per_gap must be finite and satisfy 0.0 < value <= 1.0");
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
    "Packet loss concealment config: input=%s output=%s expected=%dHz/%d/%s/%d/%s "
    "max_gap_frames=%d attenuation_per_gap=%.6f qos_depth=%d reliable=%s diag=%dms",
    config_.input_topic.c_str(),
    config_.output_topic.c_str(),
    config_.expected_sample_rate,
    config_.expected_channels,
    config_.expected_encoding.c_str(),
    config_.expected_bit_depth,
    config_.expected_layout.c_str(),
    config_.max_gap_frames,
    config_.attenuation_per_gap,
    config_.qos_depth,
    config_.qos_reliable ? "true" : "false",
    config_.diagnostics_publish_period_ms);
}

void FaPacketLossConcealmentNode::setupInterfaces()
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
    std::bind(&FaPacketLossConcealmentNode::handleFrame, this, std::placeholders::_1));

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
    std::bind(&FaPacketLossConcealmentNode::publishDiagnostics, this));
}

void FaPacketLossConcealmentNode::handleFrame(
  const fa_interfaces::msg::AudioFrame::SharedPtr msg)
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

  if (sourceChanged(*msg)) {
    RCLCPP_WARN(
      this->get_logger(),
      "Audio source_id changed from %s to %s; discarding PLC baseline",
      previous_frame_.source_id.c_str(),
      msg->source_id.c_str());
    resetPreviousFrame();
    gap_resets_.fetch_add(1);
  }

  if (has_previous_frame_) {
    if (msg->epoch <= previous_frame_.epoch) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 3000,
        "Dropping duplicate or regressing AudioFrame epoch: frame=%u previous=%u",
        msg->epoch,
        previous_frame_.epoch);
      duplicate_drops_.fetch_add(1);
      frames_dropped_.fetch_add(1);
      return;
    }

    const uint32_t missing_frame_count = msg->epoch - previous_frame_.epoch - 1U;
    if (missing_frame_count > 0U) {
      if (missing_frame_count > static_cast<uint32_t>(config_.max_gap_frames)) {
        RCLCPP_WARN(
          this->get_logger(),
          "PLC gap of %u frames exceeds plc.max_gap_frames=%d; publishing current frame only",
          missing_frame_count,
          config_.max_gap_frames);
        resetPreviousFrame();
        gap_resets_.fetch_add(1);
      } else if (!publishConcealedFrames(missing_frame_count)) {
        resetPreviousFrame();
        gap_resets_.fetch_add(1);
      }
    }
  }

  publishCurrentFrame(*msg);
  updatePreviousFrame(*msg);
}

bool FaPacketLossConcealmentNode::validateFrame(const fa_interfaces::msg::AudioFrame & msg)
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

bool FaPacketLossConcealmentNode::sourceChanged(
  const fa_interfaces::msg::AudioFrame & msg) const
{
  return has_previous_frame_ && msg.source_id != previous_frame_.source_id;
}

bool FaPacketLossConcealmentNode::publishConcealedFrames(
  const uint32_t missing_frame_count)
{
  // Explicit PLC, not hidden fallback:
  // - meaning: only missing epochs are synthesized by repeating the last valid frame.
  // - bound: at most plc.max_gap_frames and attenuation_per_gap^gap_index.
  // - expiry: gaps larger than the configured bound discard the old baseline.
  // - visibility: each synthesized frame and reset is counted in diagnostics.
  // - safety: no device I/O, resampling, padding, or arbitrary data generation is performed.
  for (uint32_t gap_index = 1U; gap_index <= missing_frame_count; ++gap_index) {
    fa_interfaces::msg::AudioFrame out;
    if (!buildConcealedFrame(gap_index, out)) {
      return false;
    }
    audio_pub_->publish(out);
    frames_out_.fetch_add(1);
    concealed_frames_.fetch_add(1);
  }
  return true;
}

bool FaPacketLossConcealmentNode::buildConcealedFrame(
  const uint32_t gap_index,
  fa_interfaces::msg::AudioFrame & out) const
{
  if (!has_previous_frame_) {
    return false;
  }

  out.header = previous_frame_.header;
  if (!timestampForGap(gap_index, out.header)) {
    return false;
  }
  out.source_id = previous_frame_.source_id;
  out.stream_id = config_.output_topic;
  out.encoding = previous_frame_.encoding;
  out.sample_rate = previous_frame_.sample_rate;
  out.channels = previous_frame_.channels;
  out.bit_depth = previous_frame_.bit_depth;
  out.layout = previous_frame_.layout;
  out.epoch = previous_frame_.epoch + gap_index;
  out.data.resize(previous_frame_.data.size());

  const double attenuation =
    std::pow(config_.attenuation_per_gap, static_cast<double>(gap_index));
  for (size_t offset = 0; offset < previous_frame_.data.size(); offset += sizeof(float)) {
    const float sample = readFloat32LeSample(previous_frame_.data, offset);
    writeFloat32LeSample(out.data, offset, static_cast<float>(static_cast<double>(sample) * attenuation));
  }
  return true;
}

bool FaPacketLossConcealmentNode::timestampForGap(
  const uint32_t gap_index,
  std_msgs::msg::Header & header) const
{
  const size_t sample_frames =
    previous_frame_.data.size() / static_cast<size_t>(config_.expected_channels * sizeof(float));
  const long double frame_duration_ns_decimal =
    (static_cast<long double>(sample_frames) * kNanosecondsPerSampleRateUnit) /
    static_cast<long double>(config_.expected_sample_rate);
  const long double advance_ns_decimal =
    frame_duration_ns_decimal * static_cast<long double>(gap_index);
  if (!std::isfinite(static_cast<double>(advance_ns_decimal)) ||
      advance_ns_decimal < 0.0L ||
      advance_ns_decimal > static_cast<long double>(std::numeric_limits<int64_t>::max()))
  {
    RCLCPP_WARN(
      this->get_logger(),
      "PLC timestamp advance is outside int64 nanosecond range");
    return false;
  }

  const int64_t advance_ns = static_cast<int64_t>(std::llround(advance_ns_decimal));
  const int64_t previous_ns = stampToNanoseconds(previous_frame_.header.stamp);
  if (previous_ns < 0 || previous_ns > kMaxBuiltinTimeNanoseconds - advance_ns) {
    RCLCPP_WARN(
      this->get_logger(),
      "PLC timestamp advance exceeds builtin_interfaces/Time range");
    return false;
  }

  header.stamp = nanosecondsToStamp(previous_ns + advance_ns);
  return true;
}

void FaPacketLossConcealmentNode::publishCurrentFrame(
  const fa_interfaces::msg::AudioFrame & msg)
{
  fa_interfaces::msg::AudioFrame out = msg;
  out.stream_id = config_.output_topic;
  audio_pub_->publish(out);
  frames_out_.fetch_add(1);
}

void FaPacketLossConcealmentNode::updatePreviousFrame(
  const fa_interfaces::msg::AudioFrame & msg)
{
  previous_frame_.header = msg.header;
  previous_frame_.source_id = msg.source_id;
  previous_frame_.encoding = msg.encoding;
  previous_frame_.sample_rate = msg.sample_rate;
  previous_frame_.channels = msg.channels;
  previous_frame_.bit_depth = msg.bit_depth;
  previous_frame_.layout = msg.layout;
  previous_frame_.data = msg.data;
  previous_frame_.epoch = msg.epoch;
  has_previous_frame_ = true;
}

void FaPacketLossConcealmentNode::resetPreviousFrame()
{
  previous_frame_ = PreviousValidFrame{};
  has_previous_frame_ = false;
}

float FaPacketLossConcealmentNode::readFloat32LeSample(
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

void FaPacketLossConcealmentNode::writeFloat32LeSample(
  std::vector<uint8_t> & data,
  const size_t byte_offset,
  const float sample) const
{
  uint32_t raw = 0U;
  std::memcpy(&raw, &sample, sizeof(float));
  data[byte_offset] = static_cast<uint8_t>(raw & 0xFFU);
  data[byte_offset + 1U] = static_cast<uint8_t>((raw >> 8U) & 0xFFU);
  data[byte_offset + 2U] = static_cast<uint8_t>((raw >> 16U) & 0xFFU);
  data[byte_offset + 3U] = static_cast<uint8_t>((raw >> 24U) & 0xFFU);
}

size_t FaPacketLossConcealmentNode::bytesPerSampleFrame() const
{
  return static_cast<size_t>(config_.expected_channels) * sizeof(float);
}

void FaPacketLossConcealmentNode::publishDiagnostics()
{
  diagnostic_msgs::msg::DiagnosticArray array_msg;
  array_msg.header.stamp = this->now();

  diagnostic_msgs::msg::DiagnosticStatus status;
  status.name = "fa_packet_loss_concealment_node";
  status.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
  status.message = "running";
  status.values.reserve(18);
  pushKeyValue(status, "backend.name", kBackendName);
  pushKeyValue(status, "input_topic", config_.input_topic);
  pushKeyValue(status, "output_topic", config_.output_topic);
  pushKeyValue(status, "expected_sample_rate", std::to_string(config_.expected_sample_rate));
  pushKeyValue(status, "expected_channels", std::to_string(config_.expected_channels));
  pushKeyValue(status, "expected_encoding", config_.expected_encoding);
  pushKeyValue(status, "expected_bit_depth", std::to_string(config_.expected_bit_depth));
  pushKeyValue(status, "expected_layout", config_.expected_layout);
  pushKeyValue(status, "plc_max_gap_frames", std::to_string(config_.max_gap_frames));
  pushKeyValue(status, "plc_attenuation_per_gap", std::to_string(config_.attenuation_per_gap));
  pushKeyValue(status, "qos_depth", std::to_string(config_.qos_depth));
  pushKeyValue(status, "qos_reliable", config_.qos_reliable ? "true" : "false");
  pushKeyValue(status, "diagnostics_publish_period_ms",
    std::to_string(config_.diagnostics_publish_period_ms));
  pushKeyValue(status, "frames_in", std::to_string(frames_in_.load()));
  pushKeyValue(status, "frames_out", std::to_string(frames_out_.load()));
  pushKeyValue(status, "frames_dropped", std::to_string(frames_dropped_.load()));
  pushKeyValue(status, "concealed_frames", std::to_string(concealed_frames_.load()));
  pushKeyValue(status, "duplicate_drops", std::to_string(duplicate_drops_.load()));
  pushKeyValue(status, "gap_resets", std::to_string(gap_resets_.load()));

  array_msg.status.push_back(status);
  diag_pub_->publish(array_msg);
}

}  // namespace fa_packet_loss_concealment

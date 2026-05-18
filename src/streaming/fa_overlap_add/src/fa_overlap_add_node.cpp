#include "fa_overlap_add/fa_overlap_add_node.hpp"

#include <chrono>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

#include "diagnostic_msgs/msg/diagnostic_status.hpp"
#include "diagnostic_msgs/msg/key_value.hpp"

namespace fa_overlap_add
{

namespace
{
constexpr const char * kEncodingFloat32 = "FLOAT32LE";
constexpr const char * kInterleavedLayout = "interleaved";
constexpr const char * kWindowRectangular = "rectangular";
constexpr const char * kWindowHann = "hann";
constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kMinNormalizationWeight = 1.0e-12;
constexpr float kMinNormalizedSample = -1.0F;
constexpr float kMaxNormalizedSample = 1.0F;
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

FaOverlapAddNode::FaOverlapAddNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("fa_overlap_add_node", options)
{
  RCLCPP_INFO(this->get_logger(), "Starting FA Overlap Add node");
  loadParameters();
  buildSynthesisWindow();
  setupInterfaces();
}

void FaOverlapAddNode::loadParameters()
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
  this->declare_parameter<int>("window.frame_samples");
  this->declare_parameter<int>("window.hop_samples");
  this->declare_parameter<std::string>("window.type");
  this->declare_parameter<int>("overlap.max_buffered_chunks");
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
  config_.frame_samples = readRequiredInt(*this, "window.frame_samples");
  config_.hop_samples = readRequiredInt(*this, "window.hop_samples");
  config_.window_type = readRequiredString(*this, "window.type");
  config_.max_buffered_chunks = readRequiredInt(*this, "overlap.max_buffered_chunks");
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
    throw std::runtime_error("fa_overlap_add requires expected.encoding=FLOAT32LE");
  }
  if (config_.expected_bit_depth != 32) {
    throw std::runtime_error("fa_overlap_add requires expected.bit_depth=32");
  }
  if (config_.expected_layout != kInterleavedLayout) {
    throw std::runtime_error("fa_overlap_add requires expected.layout=interleaved");
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
  if (config_.window_type != kWindowRectangular && config_.window_type != kWindowHann) {
    throw std::runtime_error("window.type must be rectangular or hann");
  }
  if (config_.max_buffered_chunks <= 0) {
    throw std::runtime_error("overlap.max_buffered_chunks must be > 0");
  }
  if (
    static_cast<size_t>(config_.max_buffered_chunks) >
    (std::numeric_limits<size_t>::max() / static_cast<size_t>(config_.frame_samples)))
  {
    throw std::runtime_error("overlap.max_buffered_chunks * window.frame_samples exceeds size_t range");
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
    "Overlap-add config: input=%s/%s output=%s/%s expected=%dHz/%d/%s/%d/%s "
    "frame_samples=%d hop_samples=%d window=%s max_buffered_chunks=%d qos_depth=%d "
    "reliable=%s diag=%dms",
    config_.input_topic.c_str(),
    config_.input_stream_id.c_str(),
    config_.output_topic.c_str(),
    config_.output_stream_id.c_str(),
    config_.expected_sample_rate,
    config_.expected_channels,
    config_.expected_encoding.c_str(),
    config_.expected_bit_depth,
    config_.expected_layout.c_str(),
    config_.frame_samples,
    config_.hop_samples,
    config_.window_type.c_str(),
    config_.max_buffered_chunks,
    config_.qos_depth,
    config_.qos_reliable ? "true" : "false",
    config_.diagnostics_publish_period_ms);
}

void FaOverlapAddNode::buildSynthesisWindow()
{
  synthesis_window_.assign(static_cast<size_t>(config_.frame_samples), 1.0);
  if (config_.window_type == kWindowRectangular) {
    return;
  }

  for (size_t index = 0; index < synthesis_window_.size(); ++index) {
    const double centered_phase =
      (2.0 * kPi * (static_cast<double>(index) + 0.5)) /
      static_cast<double>(config_.frame_samples);
    synthesis_window_[index] = 0.5 - (0.5 * std::cos(centered_phase));
    if (!std::isfinite(synthesis_window_[index]) ||
        synthesis_window_[index] <= kMinNormalizationWeight)
    {
      throw std::runtime_error("window.type=hann produced a non-positive synthesis weight");
    }
  }
}

void FaOverlapAddNode::setupInterfaces()
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
    std::bind(&FaOverlapAddNode::handleFrame, this, std::placeholders::_1));

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
    std::bind(&FaOverlapAddNode::publishDiagnostics, this));
}

void FaOverlapAddNode::handleFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg)
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

  if (hasInputEpochRegression(*msg)) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping duplicate or regressing overlap-add input epoch %u; expected next epoch %u",
      msg->epoch,
      next_expected_input_epoch_.value());
    epoch_regression_drops_.fetch_add(1);
    frames_dropped_.fetch_add(1);
    return;
  }

  if (requiresStreamReset(*msg)) {
    RCLCPP_WARN(
      this->get_logger(),
      "Audio stream identity, format, or future epoch gap changed; clearing overlap-add state before accepting new chunk");
    resetOverlapState();
    resets_.fetch_add(1);
  }

  if (!active_stream_.has_value()) {
    activateStream(*msg);
  }

  if (!canAccumulateChunk()) {
    RCLCPP_WARN(
      this->get_logger(),
      "Dropping chunk because overlap-add buffer would exceed overlap.max_buffered_chunks");
    resetOverlapState();
    frames_dropped_.fetch_add(1);
    resets_.fetch_add(1);
    return;
  }

  accumulateChunk(*msg);
  publishAvailableFrames();
}

bool FaOverlapAddNode::validateFrame(const fa_interfaces::msg::AudioFrame & msg)
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
  if (msg.data.size() != chunkBytes()) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "AudioFrame chunk data size must equal window.frame_samples * channels * sizeof(float)");
    return false;
  }

  for (size_t offset = 0; offset < msg.data.size(); offset += sizeof(float)) {
    const float sample = readFloat32LeSample(msg.data, offset);
    if (!std::isfinite(sample) || sample < kMinNormalizedSample || sample > kMaxNormalizedSample) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 3000,
        "AudioFrame contains non-finite or non-normalized FLOAT32LE sample");
      return false;
    }
  }

  return true;
}

bool FaOverlapAddNode::hasInputEpochRegression(const fa_interfaces::msg::AudioFrame & msg) const
{
  if (!active_stream_.has_value() || !next_expected_input_epoch_.has_value()) {
    return false;
  }
  if (msg.source_id != active_stream_->source_id || hasFormatChange(msg)) {
    return false;
  }
  return msg.epoch < next_expected_input_epoch_.value();
}

bool FaOverlapAddNode::requiresStreamReset(const fa_interfaces::msg::AudioFrame & msg) const
{
  if (!active_stream_.has_value()) {
    return false;
  }
  if (msg.source_id != active_stream_->source_id || hasFormatChange(msg)) {
    return true;
  }
  return next_expected_input_epoch_.has_value() && msg.epoch > next_expected_input_epoch_.value();
}

bool FaOverlapAddNode::hasFormatChange(const fa_interfaces::msg::AudioFrame & msg) const
{
  return active_stream_.has_value() &&
         (msg.sample_rate != active_stream_->sample_rate ||
          msg.channels != active_stream_->channels ||
          msg.encoding != active_stream_->encoding ||
          msg.bit_depth != active_stream_->bit_depth ||
          msg.layout != active_stream_->layout);
}

void FaOverlapAddNode::activateStream(const fa_interfaces::msg::AudioFrame & msg)
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
  next_output_stamp_ = msg.header.stamp;
}

bool FaOverlapAddNode::canAccumulateChunk() const
{
  const size_t required_sample_frames =
    next_chunk_start_sample_frames_ + static_cast<size_t>(config_.frame_samples);
  return required_sample_frames <= maxBufferedSampleFrames();
}

void FaOverlapAddNode::accumulateChunk(const fa_interfaces::msg::AudioFrame & msg)
{
  const size_t channels = static_cast<size_t>(config_.expected_channels);
  const size_t chunk_start = next_chunk_start_sample_frames_;
  const size_t required_sample_frames = chunk_start + static_cast<size_t>(config_.frame_samples);
  const size_t required_samples = required_sample_frames * channels;

  if (sample_sums_.size() < required_samples) {
    sample_sums_.resize(required_samples, 0.0);
  }
  if (weight_sums_.size() < required_sample_frames) {
    weight_sums_.resize(required_sample_frames, 0.0);
  }

  for (size_t frame_index = 0; frame_index < static_cast<size_t>(config_.frame_samples); ++frame_index) {
    const double weight = synthesis_window_[frame_index];
    const size_t input_sample_offset = frame_index * channels;
    const size_t output_sample_offset = (chunk_start + frame_index) * channels;

    for (size_t channel = 0; channel < channels; ++channel) {
      const size_t input_index = input_sample_offset + channel;
      const float sample = readFloat32LeSample(msg.data, input_index * sizeof(float));
      sample_sums_[output_sample_offset + channel] += static_cast<double>(sample) * weight;
    }
    weight_sums_[chunk_start + frame_index] += weight;
  }

  next_chunk_start_sample_frames_ += static_cast<size_t>(config_.hop_samples);
  next_expected_input_epoch_ = msg.epoch + 1U;
  chunks_accumulated_.fetch_add(1);
  buffered_sample_frames_.store(weight_sums_.size());
}

void FaOverlapAddNode::publishAvailableFrames()
{
  const size_t hop_sample_frames = static_cast<size_t>(config_.hop_samples);
  while (
    active_stream_.has_value() &&
    next_chunk_start_sample_frames_ >= hop_sample_frames &&
    weight_sums_.size() >= hop_sample_frames &&
    sample_sums_.size() >= hop_sample_frames * static_cast<size_t>(config_.expected_channels))
  {
    fa_interfaces::msg::AudioFrame out;
    if (!buildOutputFrame(out)) {
      RCLCPP_WARN(
        this->get_logger(),
        "Dropping buffered overlap-add state because normalized output would violate FLOAT32LE contract");
      resetOverlapState();
      frames_dropped_.fetch_add(1);
      resets_.fetch_add(1);
      return;
    }

    audio_pub_->publish(out);
    frames_out_.fetch_add(1);
    ++next_output_epoch_;
    consumePublishedHop();

    if (!advanceNextOutputStamp()) {
      RCLCPP_WARN(
        this->get_logger(),
        "Resetting overlap-add state because output timestamp advance exceeded valid range");
      resetOverlapState();
      resets_.fetch_add(1);
      return;
    }
  }
  buffered_sample_frames_.store(weight_sums_.size());
}

bool FaOverlapAddNode::buildOutputFrame(fa_interfaces::msg::AudioFrame & out) const
{
  if (!active_stream_.has_value() || !next_output_stamp_.has_value()) {
    return false;
  }

  const size_t channels = static_cast<size_t>(config_.expected_channels);
  const size_t hop_sample_frames = static_cast<size_t>(config_.hop_samples);

  out.header.stamp = next_output_stamp_.value();
  out.source_id = active_stream_->source_id;
  out.stream_id = config_.output_stream_id;
  out.encoding = active_stream_->encoding;
  out.sample_rate = active_stream_->sample_rate;
  out.channels = active_stream_->channels;
  out.bit_depth = active_stream_->bit_depth;
  out.layout = active_stream_->layout;
  out.epoch = next_output_epoch_;
  out.data.resize(hop_sample_frames * channels * sizeof(float));

  for (size_t frame_index = 0; frame_index < hop_sample_frames; ++frame_index) {
    const double weight = weight_sums_[frame_index];
    if (!std::isfinite(weight) || weight <= kMinNormalizationWeight) {
      return false;
    }

    for (size_t channel = 0; channel < channels; ++channel) {
      const size_t sample_index = (frame_index * channels) + channel;
      const double normalized = sample_sums_[sample_index] / weight;
      if (!std::isfinite(normalized) ||
          normalized < static_cast<double>(kMinNormalizedSample) ||
          normalized > static_cast<double>(kMaxNormalizedSample))
      {
        return false;
      }
      writeFloat32LeSample(
        out.data,
        sample_index * sizeof(float),
        static_cast<float>(normalized));
    }
  }

  return true;
}

bool FaOverlapAddNode::advanceNextOutputStamp()
{
  if (!next_output_stamp_.has_value()) {
    return false;
  }

  const long double advance_ns_decimal =
    (static_cast<long double>(config_.hop_samples) * kNanosecondsPerSampleRateUnit) /
    static_cast<long double>(config_.expected_sample_rate);
  if (!std::isfinite(static_cast<double>(advance_ns_decimal)) ||
      advance_ns_decimal <= 0.0L ||
      advance_ns_decimal > static_cast<long double>(std::numeric_limits<int64_t>::max()))
  {
    return false;
  }

  const int64_t advance_ns = static_cast<int64_t>(std::llround(advance_ns_decimal));
  const int64_t current_ns = stampToNanoseconds(next_output_stamp_.value());
  if (current_ns < 0 || current_ns > kMaxBuiltinTimeNanoseconds - advance_ns) {
    return false;
  }

  next_output_stamp_ = nanosecondsToStamp(current_ns + advance_ns);
  return true;
}

void FaOverlapAddNode::consumePublishedHop()
{
  const size_t channels = static_cast<size_t>(config_.expected_channels);
  const size_t hop_sample_frames = static_cast<size_t>(config_.hop_samples);
  const size_t consumed_samples = hop_sample_frames * channels;

  sample_sums_.erase(
    sample_sums_.begin(),
    sample_sums_.begin() + static_cast<std::ptrdiff_t>(consumed_samples));
  weight_sums_.erase(
    weight_sums_.begin(),
    weight_sums_.begin() + static_cast<std::ptrdiff_t>(hop_sample_frames));

  next_chunk_start_sample_frames_ -= hop_sample_frames;
  buffered_sample_frames_.store(weight_sums_.size());
}

void FaOverlapAddNode::resetOverlapState()
{
  sample_sums_.clear();
  weight_sums_.clear();
  active_stream_.reset();
  next_expected_input_epoch_.reset();
  next_output_stamp_.reset();
  next_chunk_start_sample_frames_ = 0U;
  buffered_sample_frames_.store(0);
}

float FaOverlapAddNode::readFloat32LeSample(
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

void FaOverlapAddNode::writeFloat32LeSample(
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

size_t FaOverlapAddNode::bytesPerSampleFrame() const
{
  return static_cast<size_t>(config_.expected_channels) * sizeof(float);
}

size_t FaOverlapAddNode::chunkBytes() const
{
  return static_cast<size_t>(config_.frame_samples) * bytesPerSampleFrame();
}

size_t FaOverlapAddNode::maxBufferedSampleFrames() const
{
  return static_cast<size_t>(config_.max_buffered_chunks) *
         static_cast<size_t>(config_.frame_samples);
}

void FaOverlapAddNode::publishDiagnostics()
{
  diagnostic_msgs::msg::DiagnosticArray array_msg;
  array_msg.header.stamp = this->now();

  diagnostic_msgs::msg::DiagnosticStatus status;
  status.name = "fa_overlap_add";
  status.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
  status.message = "running";
  status.values.reserve(20);
  pushKeyValue(status, "input_topic", config_.input_topic);
  pushKeyValue(status, "output_topic", config_.output_topic);
  pushKeyValue(status, "input_stream_id", config_.input_stream_id);
  pushKeyValue(status, "output_stream_id", config_.output_stream_id);
  pushKeyValue(status, "expected_sample_rate", std::to_string(config_.expected_sample_rate));
  pushKeyValue(status, "expected_channels", std::to_string(config_.expected_channels));
  pushKeyValue(status, "expected_encoding", config_.expected_encoding);
  pushKeyValue(status, "expected_bit_depth", std::to_string(config_.expected_bit_depth));
  pushKeyValue(status, "expected_layout", config_.expected_layout);
  pushKeyValue(status, "frame_samples", std::to_string(config_.frame_samples));
  pushKeyValue(status, "hop_samples", std::to_string(config_.hop_samples));
  pushKeyValue(status, "window_type", config_.window_type);
  pushKeyValue(status, "max_buffered_chunks", std::to_string(config_.max_buffered_chunks));
  pushKeyValue(status, "buffered_sample_frames", std::to_string(buffered_sample_frames_.load()));
  pushKeyValue(status, "frames_in", std::to_string(frames_in_.load()));
  pushKeyValue(status, "frames_out", std::to_string(frames_out_.load()));
  pushKeyValue(status, "frames_dropped", std::to_string(frames_dropped_.load()));
  pushKeyValue(status, "epoch_regression_drops", std::to_string(epoch_regression_drops_.load()));
  pushKeyValue(status, "chunks_accumulated", std::to_string(chunks_accumulated_.load()));
  pushKeyValue(status, "resets", std::to_string(resets_.load()));
  pushKeyValue(status, "backend.name", "internal_overlap_add");

  array_msg.status.push_back(status);
  diag_pub_->publish(array_msg);
}

}  // namespace fa_overlap_add

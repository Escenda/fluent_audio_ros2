#include "fa_crossfade/fa_crossfade_node.hpp"

#include <chrono>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "diagnostic_msgs/msg/diagnostic_status.hpp"
#include "diagnostic_msgs/msg/key_value.hpp"
#include "fa_crossfade/backends/internal_crossfade.hpp"

namespace fa_crossfade
{

namespace
{
constexpr const char * kEncodingFloat32 = "FLOAT32LE";
constexpr const char * kInterleavedLayout = "interleaved";
constexpr int kMaxExpectedSampleRate = 384000;
constexpr int kMaxExpectedChannels = 64;

std::string removeLeadingSlashes(std::string value)
{
  while (!value.empty() && value.front() == '/') {
    value.erase(value.begin());
  }
  return value;
}

bool sameIdentityString(const std::string & left, const std::string & right)
{
  return left == right || removeLeadingSlashes(left) == removeLeadingSlashes(right);
}

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
    throw std::runtime_error(name + " must be a string");
  }
  return parameter.as_string();
}

int readRequiredInt(const rclcpp::Node & node, const std::string & name)
{
  const rclcpp::Parameter parameter = getRequiredParameter(node, name);
  if (parameter.get_type() != rclcpp::ParameterType::PARAMETER_INTEGER) {
    throw std::runtime_error(name + " must be an integer");
  }
  const int64_t value = parameter.as_int();
  if (value < static_cast<int64_t>(std::numeric_limits<int>::min()) ||
      value > static_cast<int64_t>(std::numeric_limits<int>::max()))
  {
    throw std::runtime_error(name + " is outside supported integer range");
  }
  return static_cast<int>(value);
}

bool readRequiredBool(const rclcpp::Node & node, const std::string & name)
{
  const rclcpp::Parameter parameter = getRequiredParameter(node, name);
  if (parameter.get_type() != rclcpp::ParameterType::PARAMETER_BOOL) {
    throw std::runtime_error(name + " must be a bool");
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

bool streamMatchesTopic(
  const std::string & stream_id,
  const std::string & raw_a,
  const std::string & raw_b,
  const std::string & raw_output,
  const std::string & resolved_a,
  const std::string & resolved_b,
  const std::string & resolved_output)
{
  return sameIdentityString(stream_id, raw_a) ||
         sameIdentityString(stream_id, raw_b) ||
         sameIdentityString(stream_id, raw_output) ||
         sameIdentityString(stream_id, resolved_a) ||
         sameIdentityString(stream_id, resolved_b) ||
         sameIdentityString(stream_id, resolved_output);
}
}  // namespace

FaCrossfadeNode::FaCrossfadeNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("fa_crossfade", options)
{
  RCLCPP_INFO(this->get_logger(), "Starting FA Crossfade node");
  loadParameters();
  configureBackend();
  setupInterfaces();
}

FaCrossfadeNode::~FaCrossfadeNode() = default;

void FaCrossfadeNode::loadParameters()
{
  this->declare_parameter<std::string>("input_a_topic");
  this->declare_parameter<std::string>("input_b_topic");
  this->declare_parameter<std::string>("output_topic");
  this->declare_parameter<std::string>("input_a_stream_id");
  this->declare_parameter<std::string>("input_b_stream_id");
  this->declare_parameter<std::string>("output.stream_id");
  this->declare_parameter<int>("crossfade.overlap_frames");
  this->declare_parameter<std::string>("crossfade.curve");
  this->declare_parameter<int>("expected.sample_rate");
  this->declare_parameter<int>("expected.channels");
  this->declare_parameter<std::string>("expected.encoding");
  this->declare_parameter<int>("expected.bit_depth");
  this->declare_parameter<std::string>("expected.layout");
  this->declare_parameter<int>("qos.depth");
  this->declare_parameter<bool>("qos.reliable");
  this->declare_parameter<int>("diagnostics.qos.depth");
  this->declare_parameter<bool>("diagnostics.qos.reliable");
  this->declare_parameter<int>("diagnostics.publish_period_ms");

  config_.input_a_topic = readRequiredString(*this, "input_a_topic");
  config_.input_b_topic = readRequiredString(*this, "input_b_topic");
  config_.output_topic = readRequiredString(*this, "output_topic");
  config_.input_a_stream_id = readRequiredString(*this, "input_a_stream_id");
  config_.input_b_stream_id = readRequiredString(*this, "input_b_stream_id");
  config_.output_stream_id = readRequiredString(*this, "output.stream_id");
  config_.overlap_frames = readRequiredInt(*this, "crossfade.overlap_frames");
  config_.curve = readRequiredString(*this, "crossfade.curve");
  config_.expected_sample_rate = readRequiredInt(*this, "expected.sample_rate");
  config_.expected_channels = readRequiredInt(*this, "expected.channels");
  config_.expected_encoding = readRequiredString(*this, "expected.encoding");
  config_.expected_bit_depth = readRequiredInt(*this, "expected.bit_depth");
  config_.expected_layout = readRequiredString(*this, "expected.layout");
  config_.qos_depth = readRequiredInt(*this, "qos.depth");
  config_.qos_reliable = readRequiredBool(*this, "qos.reliable");
  config_.diagnostics_qos_depth = readRequiredInt(*this, "diagnostics.qos.depth");
  config_.diagnostics_qos_reliable = readRequiredBool(*this, "diagnostics.qos.reliable");
  config_.diagnostics_publish_period_ms = readRequiredInt(*this, "diagnostics.publish_period_ms");

  if (config_.input_a_topic.empty()) {
    throw std::runtime_error("input_a_topic is required");
  }
  if (config_.input_b_topic.empty()) {
    throw std::runtime_error("input_b_topic is required");
  }
  if (config_.output_topic.empty()) {
    throw std::runtime_error("output_topic is required");
  }
  const std::string resolved_input_a =
    this->get_node_topics_interface()->resolve_topic_name(config_.input_a_topic);
  const std::string resolved_input_b =
    this->get_node_topics_interface()->resolve_topic_name(config_.input_b_topic);
  const std::string resolved_output =
    this->get_node_topics_interface()->resolve_topic_name(config_.output_topic);
  if (resolved_input_a == resolved_input_b ||
      resolved_input_a == resolved_output ||
      resolved_input_b == resolved_output)
  {
    throw std::runtime_error("input_a_topic, input_b_topic, and output_topic must resolve to distinct ROS topics");
  }
  if (config_.input_a_stream_id.empty()) {
    throw std::runtime_error("input_a_stream_id is required");
  }
  if (config_.input_b_stream_id.empty()) {
    throw std::runtime_error("input_b_stream_id is required");
  }
  if (config_.output_stream_id.empty()) {
    throw std::runtime_error("output.stream_id is required");
  }
  if (config_.input_a_stream_id == config_.input_b_stream_id ||
      config_.input_a_stream_id == config_.output_stream_id ||
      config_.input_b_stream_id == config_.output_stream_id)
  {
    throw std::runtime_error("input and output stream ids must be distinct");
  }
  if (streamMatchesTopic(
      config_.input_a_stream_id,
      config_.input_a_topic,
      config_.input_b_topic,
      config_.output_topic,
      resolved_input_a,
      resolved_input_b,
      resolved_output))
  {
    throw std::runtime_error("input_a_stream_id must be distinct from ROS topics");
  }
  if (streamMatchesTopic(
      config_.input_b_stream_id,
      config_.input_a_topic,
      config_.input_b_topic,
      config_.output_topic,
      resolved_input_a,
      resolved_input_b,
      resolved_output))
  {
    throw std::runtime_error("input_b_stream_id must be distinct from ROS topics");
  }
  if (streamMatchesTopic(
      config_.output_stream_id,
      config_.input_a_topic,
      config_.input_b_topic,
      config_.output_topic,
      resolved_input_a,
      resolved_input_b,
      resolved_output))
  {
    throw std::runtime_error("output.stream_id must be distinct from ROS topics");
  }
  if (config_.overlap_frames <= 0) {
    throw std::runtime_error("crossfade.overlap_frames must be > 0");
  }
  static_cast<void>(backends::fadeCurveFromName(config_.curve));
  if (config_.expected_sample_rate <= 0 || config_.expected_sample_rate > kMaxExpectedSampleRate) {
    throw std::runtime_error("expected.sample_rate must satisfy 0 < value <= 384000");
  }
  if (config_.expected_channels <= 0 || config_.expected_channels > kMaxExpectedChannels) {
    throw std::runtime_error("expected.channels must satisfy 0 < value <= 64");
  }
  if (config_.expected_encoding != kEncodingFloat32) {
    throw std::runtime_error("fa_crossfade requires expected.encoding=FLOAT32LE");
  }
  if (config_.expected_bit_depth != 32) {
    throw std::runtime_error("fa_crossfade requires expected.bit_depth=32");
  }
  if (config_.expected_layout != kInterleavedLayout) {
    throw std::runtime_error("fa_crossfade requires expected.layout=interleaved");
  }
  if (config_.qos_depth <= 0) {
    throw std::runtime_error("qos.depth must be > 0");
  }
  if (config_.diagnostics_qos_depth <= 0) {
    throw std::runtime_error("diagnostics.qos.depth must be > 0");
  }
  if (config_.diagnostics_publish_period_ms <= 0) {
    throw std::runtime_error("diagnostics.publish_period_ms must be > 0");
  }

  RCLCPP_INFO(
    this->get_logger(),
    "Crossfade config: input_a_topic=%s input_b_topic=%s output_topic=%s "
    "input_a_stream_id=%s input_b_stream_id=%s output_stream_id=%s overlap_frames=%d "
    "curve=%s expected=%dHz/%d/%s/%d/%s qos_depth=%d reliable=%s "
    "diag_qos_depth=%d diag_reliable=%s diag=%dms",
    config_.input_a_topic.c_str(),
    config_.input_b_topic.c_str(),
    config_.output_topic.c_str(),
    config_.input_a_stream_id.c_str(),
    config_.input_b_stream_id.c_str(),
    config_.output_stream_id.c_str(),
    config_.overlap_frames,
    config_.curve.c_str(),
    config_.expected_sample_rate,
    config_.expected_channels,
    config_.expected_encoding.c_str(),
    config_.expected_bit_depth,
    config_.expected_layout.c_str(),
    config_.qos_depth,
    config_.qos_reliable ? "true" : "false",
    config_.diagnostics_qos_depth,
    config_.diagnostics_qos_reliable ? "true" : "false",
    config_.diagnostics_publish_period_ms);
}

void FaCrossfadeNode::configureBackend()
{
  backend_ = std::make_unique<backends::InternalCrossfadeBackend>(
    backends::InternalCrossfadeConfig{
      config_.expected_channels,
      static_cast<size_t>(config_.overlap_frames),
      backends::fadeCurveFromName(config_.curve)});
}

void FaCrossfadeNode::setupInterfaces()
{
  rclcpp::QoS qos(static_cast<size_t>(config_.qos_depth));
  if (config_.qos_reliable) {
    qos.reliable();
  } else {
    qos.best_effort();
  }

  audio_pub_ = this->create_publisher<fa_interfaces::msg::AudioFrame>(config_.output_topic, qos);
  input_a_sub_ = this->create_subscription<fa_interfaces::msg::AudioFrame>(
    config_.input_a_topic,
    qos,
    std::bind(&FaCrossfadeNode::handleInputA, this, std::placeholders::_1));
  input_b_sub_ = this->create_subscription<fa_interfaces::msg::AudioFrame>(
    config_.input_b_topic,
    qos,
    std::bind(&FaCrossfadeNode::handleInputB, this, std::placeholders::_1));

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
    std::bind(&FaCrossfadeNode::publishDiagnostics, this));
}

void FaCrossfadeNode::handleInputA(const fa_interfaces::msg::AudioFrame::SharedPtr msg)
{
  handleFrame(msg, true);
}

void FaCrossfadeNode::handleInputB(const fa_interfaces::msg::AudioFrame::SharedPtr msg)
{
  handleFrame(msg, false);
}

void FaCrossfadeNode::handleFrame(
  const fa_interfaces::msg::AudioFrame::SharedPtr msg,
  bool is_input_a)
{
  if (is_input_a) {
    input_a_frames_in_.fetch_add(1);
  } else {
    input_b_frames_in_.fetch_add(1);
  }

  if (!msg) {
    throw std::logic_error("FaCrossfadeNode received null AudioFrame pointer");
  }

  const std::string & expected_stream_id =
    is_input_a ? config_.input_a_stream_id : config_.input_b_stream_id;
  const char * input_name = is_input_a ? "input_a" : "input_b";
  if (!validateFrame(*msg, expected_stream_id, input_name)) {
    invalid_frames_dropped_.fetch_add(1);
    frames_dropped_.fetch_add(1);
    return;
  }

  std::lock_guard<std::mutex> lock(pending_mutex_);
  if (is_input_a) {
    if (has_pending_a_) {
      frames_dropped_.fetch_add(1);
    }
    pending_a_ = *msg;
    has_pending_a_ = true;
  } else {
    if (has_pending_b_) {
      frames_dropped_.fetch_add(1);
    }
    pending_b_ = *msg;
    has_pending_b_ = true;
  }
  tryPublishPairLocked();
}

void FaCrossfadeNode::tryPublishPairLocked()
{
  if (!has_pending_a_ || !has_pending_b_) {
    return;
  }
  if (!backend_) {
    throw std::logic_error("FaCrossfadeNode backend is not initialized");
  }
  if (!audio_pub_) {
    throw std::logic_error("FaCrossfadeNode publisher is not initialized");
  }

  if (pending_a_.epoch != pending_b_.epoch) {
    epoch_mismatches_.fetch_add(1);
    frames_dropped_.fetch_add(2);
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping crossfade pair because epoch mismatch: %u != %u",
      pending_a_.epoch,
      pending_b_.epoch);
    clearPendingLocked();
    return;
  }
  if (!pairMetadataMatches(pending_a_, pending_b_)) {
    metadata_mismatches_.fetch_add(1);
    frames_dropped_.fetch_add(2);
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping crossfade pair because adjacent segment metadata does not match");
    clearPendingLocked();
    return;
  }

  std::vector<uint8_t> output_data;
  const backends::ProcessResult result =
    backend_->process(pending_a_.data, pending_b_.data, output_data);
  if (result.status != backends::ProcessStatus::kOk) {
    const char * status_message = backends::processStatusMessage(result.status);
    backend_rejections_.fetch_add(1);
    frames_dropped_.fetch_add(2);
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping crossfade pair because backend rejected input: %s",
      status_message);
    clearPendingLocked();
    return;
  }

  fa_interfaces::msg::AudioFrame out = pending_a_;
  out.stream_id = config_.output_stream_id;
  out.data = std::move(output_data);
  audio_pub_->publish(out);
  pairs_out_.fetch_add(1);
  clearPendingLocked();
}

void FaCrossfadeNode::clearPendingLocked()
{
  has_pending_a_ = false;
  has_pending_b_ = false;
  pending_a_ = fa_interfaces::msg::AudioFrame{};
  pending_b_ = fa_interfaces::msg::AudioFrame{};
}

bool FaCrossfadeNode::validateFrame(
  const fa_interfaces::msg::AudioFrame & msg,
  const std::string & expected_stream_id,
  const char * input_name)
{
  if (msg.source_id.empty() || msg.stream_id.empty()) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping %s segment because source_id and stream_id are required",
      input_name);
    return false;
  }
  if (msg.stream_id != expected_stream_id) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping %s segment because stream_id mismatch: %s != %s",
      input_name,
      msg.stream_id.c_str(),
      expected_stream_id.c_str());
    return false;
  }
  if (msg.layout != config_.expected_layout) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping %s segment because layout mismatch: %s != %s",
      input_name,
      msg.layout.c_str(),
      config_.expected_layout.c_str());
    return false;
  }
  if (msg.encoding != config_.expected_encoding ||
      msg.bit_depth != static_cast<uint32_t>(config_.expected_bit_depth))
  {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping %s segment because encoding mismatch: %s/%u != %s/%d",
      input_name,
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
      "Dropping %s segment because format mismatch: frame=%uHz/%u config=%dHz/%d",
      input_name,
      msg.sample_rate,
      msg.channels,
      config_.expected_sample_rate,
      config_.expected_channels);
    return false;
  }
  if (msg.data.empty() || (msg.data.size() % bytesPerFrame()) != 0) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping %s segment because data size is invalid for FLOAT32LE interleaved samples",
      input_name);
    return false;
  }
  return true;
}

bool FaCrossfadeNode::pairMetadataMatches(
  const fa_interfaces::msg::AudioFrame & segment_a,
  const fa_interfaces::msg::AudioFrame & segment_b) const
{
  return segment_a.source_id == segment_b.source_id &&
         segment_a.sample_rate == segment_b.sample_rate &&
         segment_a.channels == segment_b.channels &&
         segment_a.encoding == segment_b.encoding &&
         segment_a.bit_depth == segment_b.bit_depth &&
         segment_a.layout == segment_b.layout;
}

size_t FaCrossfadeNode::bytesPerFrame() const
{
  return static_cast<size_t>(config_.expected_channels) * sizeof(float);
}

void FaCrossfadeNode::publishDiagnostics()
{
  if (!backend_) {
    throw std::logic_error("FaCrossfadeNode backend is not initialized");
  }

  bool has_a = false;
  bool has_b = false;
  uint32_t pending_a_epoch = 0U;
  uint32_t pending_b_epoch = 0U;
  {
    std::lock_guard<std::mutex> lock(pending_mutex_);
    has_a = has_pending_a_;
    has_b = has_pending_b_;
    pending_a_epoch = pending_a_.epoch;
    pending_b_epoch = pending_b_.epoch;
  }

  diagnostic_msgs::msg::DiagnosticArray array_msg;
  array_msg.header.stamp = this->now();

  diagnostic_msgs::msg::DiagnosticStatus status;
  status.name = "fa_crossfade";
  status.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
  status.message = "running";
  status.values.reserve(24);
  pushKeyValue(status, "overlap_frames", std::to_string(backend_->overlapFrames()));
  pushKeyValue(status, "fade_curve", backends::fadeCurveName(backend_->fadeCurve()));
  pushKeyValue(status, "input_a_topic", config_.input_a_topic);
  pushKeyValue(status, "input_b_topic", config_.input_b_topic);
  pushKeyValue(status, "output_topic", config_.output_topic);
  pushKeyValue(status, "input_a_stream_id", config_.input_a_stream_id);
  pushKeyValue(status, "input_b_stream_id", config_.input_b_stream_id);
  pushKeyValue(status, "output_stream_id", config_.output_stream_id);
  pushKeyValue(status, "diagnostics_qos_depth", std::to_string(config_.diagnostics_qos_depth));
  pushKeyValue(status, "diagnostics_qos_reliable", config_.diagnostics_qos_reliable ? "true" : "false");
  pushKeyValue(status, "expected_sample_rate", std::to_string(config_.expected_sample_rate));
  pushKeyValue(status, "expected_channels", std::to_string(config_.expected_channels));
  pushKeyValue(status, "has_pending_a", has_a ? "true" : "false");
  pushKeyValue(status, "has_pending_b", has_b ? "true" : "false");
  pushKeyValue(status, "pending_a_epoch", std::to_string(pending_a_epoch));
  pushKeyValue(status, "pending_b_epoch", std::to_string(pending_b_epoch));
  pushKeyValue(status, "input_a_frames_in", std::to_string(input_a_frames_in_.load()));
  pushKeyValue(status, "input_b_frames_in", std::to_string(input_b_frames_in_.load()));
  pushKeyValue(status, "pairs_out", std::to_string(pairs_out_.load()));
  pushKeyValue(status, "frames_dropped", std::to_string(frames_dropped_.load()));
  pushKeyValue(status, "invalid_frames_dropped", std::to_string(invalid_frames_dropped_.load()));
  pushKeyValue(status, "epoch_mismatches", std::to_string(epoch_mismatches_.load()));
  pushKeyValue(status, "metadata_mismatches", std::to_string(metadata_mismatches_.load()));
  pushKeyValue(status, "backend_rejections", std::to_string(backend_rejections_.load()));

  array_msg.status.push_back(status);
  diag_pub_->publish(array_msg);
}

}  // namespace fa_crossfade

#include "fa_aec_linear/fa_aec_linear_node.hpp"

#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "builtin_interfaces/msg/time.hpp"
#include "diagnostic_msgs/msg/diagnostic_status.hpp"
#include "diagnostic_msgs/msg/key_value.hpp"
#include "fa_aec_linear/backends/baseline_linear.hpp"

namespace fa_aec_linear
{

namespace
{
constexpr int kRequiredSampleRate = 16000;
constexpr uint32_t kNanosecondsPerSecond = 1000000000U;
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

double readRequiredDouble(const rclcpp::Node & node, const std::string & name)
{
  const rclcpp::Parameter parameter = getRequiredParameter(node, name);
  if (parameter.get_type() != rclcpp::ParameterType::PARAMETER_DOUBLE) {
    throw std::runtime_error(name + " must be a double parameter");
  }
  return parameter.as_double();
}

bool hasValidStamp(const builtin_interfaces::msg::Time & stamp)
{
  if (stamp.sec < 0) {
    return false;
  }
  if (stamp.nanosec >= kNanosecondsPerSecond) {
    return false;
  }
  return stamp.sec != 0 || stamp.nanosec != 0U;
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

const char * frameValidationStatusMessage(const FrameValidationStatus status)
{
  switch (status) {
    case FrameValidationStatus::kOk:
      return "ok";
    case FrameValidationStatus::kMissingSourceId:
      return "source_id is empty";
    case FrameValidationStatus::kStreamIdMismatch:
      return "stream_id does not match expected stream identity";
    case FrameValidationStatus::kInvalidTimestamp:
      return "header.stamp is missing or invalid";
    case FrameValidationStatus::kSampleRateMismatch:
      return "sample_rate does not match expected_sample_rate";
    case FrameValidationStatus::kChannelsMismatch:
      return "channels does not match expected_channels";
    case FrameValidationStatus::kFormatMismatch:
      return "encoding/bit_depth does not match expected format";
    case FrameValidationStatus::kLayoutMismatch:
      return "layout is not interleaved";
    case FrameValidationStatus::kEmptyData:
      return "data is empty";
    case FrameValidationStatus::kMisalignedData:
      return "data byte length is not aligned to frame size";
  }
  return "unknown frame validation status";
}
}  // namespace

FaAecLinearNode::FaAecLinearNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("fa_aec_linear", options)
{
  RCLCPP_INFO(this->get_logger(), "Starting FA AEC Linear node");
  loadParameters();
  configureBackend();
  setupInterfaces();
}

FaAecLinearNode::~FaAecLinearNode() = default;

void FaAecLinearNode::loadParameters()
{
  this->declare_parameter<bool>("enabled");
  this->declare_parameter<std::string>("mic_topic");
  this->declare_parameter<std::string>("ref_topic");
  this->declare_parameter<std::string>("output_topic");
  this->declare_parameter<std::string>("mic_stream_id");
  this->declare_parameter<std::string>("ref_stream_id");
  this->declare_parameter<std::string>("output.stream_id");
  this->declare_parameter<int>("expected_sample_rate");
  this->declare_parameter<int>("expected_channels");
  this->declare_parameter<std::string>("expected.encoding");
  this->declare_parameter<int>("expected.bit_depth");
  this->declare_parameter<int>("ref_timeout_ms");
  this->declare_parameter<std::string>("reference_failure_policy");
  this->declare_parameter<double>("cancel_gain");
  this->declare_parameter<int>("qos.depth");
  this->declare_parameter<bool>("qos.reliable");
  this->declare_parameter<int>("diagnostics.publish_period_ms");
  this->declare_parameter<int>("diagnostics.qos.depth");
  this->declare_parameter<bool>("diagnostics.qos.reliable");

  config_.enabled = readRequiredBool(*this, "enabled");
  config_.mic_topic = readRequiredString(*this, "mic_topic");
  config_.ref_topic = readRequiredString(*this, "ref_topic");
  config_.output_topic = readRequiredString(*this, "output_topic");
  config_.mic_stream_id = readRequiredString(*this, "mic_stream_id");
  config_.ref_stream_id = readRequiredString(*this, "ref_stream_id");
  config_.output_stream_id = readRequiredString(*this, "output.stream_id");
  config_.expected_sample_rate = readRequiredInt(*this, "expected_sample_rate");
  config_.expected_channels = readRequiredInt(*this, "expected_channels");
  config_.expected_encoding = readRequiredString(*this, "expected.encoding");
  config_.expected_bit_depth = readRequiredInt(*this, "expected.bit_depth");
  config_.ref_timeout_ms = readRequiredInt(*this, "ref_timeout_ms");
  config_.reference_failure_policy = readRequiredString(*this, "reference_failure_policy");
  config_.cancel_gain = readRequiredDouble(*this, "cancel_gain");
  config_.qos_depth = readRequiredInt(*this, "qos.depth");
  config_.qos_reliable = readRequiredBool(*this, "qos.reliable");
  config_.diagnostics_publish_period_ms = readRequiredInt(
    *this,
    "diagnostics.publish_period_ms");
  config_.diagnostics_qos_depth = readRequiredInt(*this, "diagnostics.qos.depth");
  config_.diagnostics_qos_reliable = readRequiredBool(*this, "diagnostics.qos.reliable");

  if (config_.mic_topic.empty()) {
    throw std::runtime_error("mic_topic is required (set via YAML)");
  }
  if (config_.ref_topic.empty()) {
    throw std::runtime_error("ref_topic is required (set via YAML)");
  }
  if (config_.output_topic.empty()) {
    throw std::runtime_error("output_topic is required (set via YAML)");
  }
  if (config_.mic_stream_id.empty()) {
    throw std::runtime_error("mic_stream_id is required (set via YAML)");
  }
  if (config_.ref_stream_id.empty()) {
    throw std::runtime_error("ref_stream_id is required (set via YAML)");
  }
  if (config_.output_stream_id.empty()) {
    throw std::runtime_error("output.stream_id is required (set via YAML)");
  }
  config_.resolved_mic_topic =
    this->get_node_topics_interface()->resolve_topic_name(config_.mic_topic);
  config_.resolved_ref_topic =
    this->get_node_topics_interface()->resolve_topic_name(config_.ref_topic);
  config_.resolved_output_topic =
    this->get_node_topics_interface()->resolve_topic_name(config_.output_topic);
  if (config_.resolved_mic_topic == config_.resolved_ref_topic) {
    throw std::runtime_error("resolved mic_topic and ref_topic must be distinct");
  }
  if (config_.resolved_mic_topic == config_.resolved_output_topic) {
    throw std::runtime_error("resolved mic_topic and output_topic must be distinct");
  }
  if (config_.resolved_ref_topic == config_.resolved_output_topic) {
    throw std::runtime_error("resolved ref_topic and output_topic must be distinct");
  }
  if (sameIdentityString(config_.mic_topic, config_.ref_topic) ||
      sameIdentityString(config_.mic_topic, config_.output_topic) ||
      sameIdentityString(config_.ref_topic, config_.output_topic))
  {
    throw std::runtime_error("mic_topic, ref_topic, and output_topic must be distinct");
  }

  const std::vector<std::string> topic_identities = {
    config_.mic_topic,
    config_.ref_topic,
    config_.output_topic,
    config_.resolved_mic_topic,
    config_.resolved_ref_topic,
    config_.resolved_output_topic};
  const std::vector<std::string> stream_identities = {
    config_.mic_stream_id,
    config_.ref_stream_id,
    config_.output_stream_id};
  std::set<std::string> unique_stream_identities;
  for (const std::string & stream_id : stream_identities) {
    if (!unique_stream_identities.insert(identityWithoutLeadingSlash(stream_id)).second) {
      throw std::runtime_error(
              "mic_stream_id, ref_stream_id, and output.stream_id must be distinct");
    }
    for (const std::string & topic_identity : topic_identities) {
      if (sameIdentityString(stream_id, topic_identity)) {
        throw std::runtime_error(
                "mic_stream_id, ref_stream_id, and output.stream_id must be distinct from ROS topics");
      }
    }
  }
  if (config_.expected_sample_rate != kRequiredSampleRate) {
    throw std::runtime_error(
            "fa_aec_linear requires expected_sample_rate=16000 by design (got " +
            std::to_string(config_.expected_sample_rate) + ")");
  }
  if (config_.expected_channels <= 0) {
    throw std::runtime_error("expected_channels must be > 0");
  }
  if (!backends::isSupportedAudioFormatPair(
        config_.expected_encoding,
        config_.expected_bit_depth))
  {
    throw std::runtime_error("expected encoding/bit_depth must be PCM16LE/16 or FLOAT32LE/32");
  }
  if (config_.qos_depth <= 0) {
    throw std::runtime_error("qos.depth must be > 0 (set via YAML)");
  }
  if (config_.diagnostics_publish_period_ms <= 0) {
    throw std::runtime_error("diagnostics.publish_period_ms must be > 0 (set via YAML)");
  }
  if (config_.diagnostics_qos_depth <= 0) {
    throw std::runtime_error("diagnostics.qos.depth must be > 0");
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
    "AEC Linear config: enabled=%s mic=%s mic_stream=%s ref=%s ref_stream=%s output=%s "
    "output_stream=%s expected_sr=%d expected_ch=%d "
    "expected=%s/%d ref_timeout=%dms reference_failure_policy=%s cancel_gain=%.3f qos_depth=%d reliable=%s",
    config_.enabled ? "true" : "false",
    config_.mic_topic.c_str(),
    config_.mic_stream_id.c_str(),
    config_.ref_topic.c_str(),
    config_.ref_stream_id.c_str(),
    config_.output_topic.c_str(),
    config_.output_stream_id.c_str(),
    config_.expected_sample_rate,
    config_.expected_channels,
    config_.expected_encoding.c_str(),
    config_.expected_bit_depth,
    config_.ref_timeout_ms,
    config_.reference_failure_policy.c_str(),
    config_.cancel_gain,
    config_.qos_depth,
    config_.qos_reliable ? "true" : "false");
}

void FaAecLinearNode::configureBackend()
{
  backend_ = std::make_unique<backends::BaselineLinearBackend>(
    backends::BaselineLinearConfig{
      config_.expected_channels,
      config_.expected_encoding,
      config_.expected_bit_depth,
      config_.cancel_gain});
}

void FaAecLinearNode::setupInterfaces()
{
  rclcpp::QoS qos(static_cast<size_t>(config_.qos_depth));
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

  rclcpp::QoS diagnostics_qos(static_cast<size_t>(config_.diagnostics_qos_depth));
  if (config_.diagnostics_qos_reliable) {
    diagnostics_qos.reliable();
  } else {
    diagnostics_qos.best_effort();
  }

  diag_pub_ = this->create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
    "diagnostics", diagnostics_qos);
  diag_timer_ = this->create_wall_timer(
    std::chrono::milliseconds(config_.diagnostics_publish_period_ms),
    std::bind(&FaAecLinearNode::publishDiagnostics, this));
}

FrameValidationStatus FaAecLinearNode::validateFrame(
  const fa_interfaces::msg::AudioFrame & msg,
  const std::string & expected_stream_id) const
{
  if (msg.sample_rate != static_cast<uint32_t>(config_.expected_sample_rate)) {
    return FrameValidationStatus::kSampleRateMismatch;
  }
  if (msg.channels != static_cast<uint32_t>(config_.expected_channels)) {
    return FrameValidationStatus::kChannelsMismatch;
  }
  if (msg.channels == 0 || msg.sample_rate == 0) {
    return FrameValidationStatus::kSampleRateMismatch;
  }
  if (msg.encoding != config_.expected_encoding ||
      msg.bit_depth != static_cast<uint32_t>(config_.expected_bit_depth))
  {
    return FrameValidationStatus::kFormatMismatch;
  }
  if (msg.source_id.empty()) {
    return FrameValidationStatus::kMissingSourceId;
  }
  if (msg.stream_id != expected_stream_id) {
    return FrameValidationStatus::kStreamIdMismatch;
  }
  if (!hasValidStamp(msg.header.stamp)) {
    return FrameValidationStatus::kInvalidTimestamp;
  }
  if (msg.layout != kInterleavedLayout) {
    return FrameValidationStatus::kLayoutMismatch;
  }
  if (msg.data.empty()) {
    return FrameValidationStatus::kEmptyData;
  }
  if (!backends::isSupportedAudioFormatPair(msg.encoding, static_cast<int>(msg.bit_depth))) {
    return FrameValidationStatus::kFormatMismatch;
  }
  const size_t bytes_per_sample = static_cast<size_t>(msg.bit_depth / 8);
  const size_t bytes_per_frame = static_cast<size_t>(msg.channels) * bytes_per_sample;
  if (bytes_per_frame == 0 || (msg.data.size() % bytes_per_frame) != 0) {
    return FrameValidationStatus::kMisalignedData;
  }
  return FrameValidationStatus::kOk;
}

void FaAecLinearNode::onRefFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg)
{
  ref_in_.fetch_add(1);
  if (!msg) {
    throw std::logic_error("fa_aec_linear received a null reference AudioFrame pointer");
  }
  const FrameValidationStatus validation_status = validateFrame(*msg, config_.ref_stream_id);
  if (validation_status != FrameValidationStatus::kOk) {
    ref_drop_.fetch_add(1);
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping invalid reference frame: %s",
      frameValidationStatusMessage(validation_status));
    return;
  }

  std::lock_guard<std::mutex> lock(ref_mutex_);
  last_ref_ = msg;
  last_ref_stamp_ = rclcpp::Time(msg->header.stamp, RCL_ROS_TIME);
}

void FaAecLinearNode::onMicFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg)
{
  mic_in_.fetch_add(1);
  if (!msg) {
    throw std::logic_error("fa_aec_linear received a null mic AudioFrame pointer");
  }
  if (!out_pub_) {
    throw std::logic_error("fa_aec_linear publisher is not initialized");
  }

  if (!config_.enabled) {
    mic_drop_.fetch_add(1);
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping mic frame because fa_aec_linear is disabled; disable the system node instead");
    return;
  }

  const FrameValidationStatus validation_status = validateFrame(*msg, config_.mic_stream_id);
  if (validation_status != FrameValidationStatus::kOk) {
    mic_drop_.fetch_add(1);
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping invalid mic frame: %s sr=%u ch=%u bits=%u bytes=%zu (expected sr=%d ch=%d)",
      frameValidationStatusMessage(validation_status),
      msg->sample_rate, msg->channels, msg->bit_depth, msg->data.size(),
      config_.expected_sample_rate, config_.expected_channels);
    return;
  }

  fa_interfaces::msg::AudioFrame::SharedPtr ref;
  rclcpp::Time ref_stamp{0, 0, RCL_ROS_TIME};
  {
    std::lock_guard<std::mutex> lock(ref_mutex_);
    ref = last_ref_;
    ref_stamp = last_ref_stamp_;
  }

  const rclcpp::Time mic_stamp(msg->header.stamp, RCL_ROS_TIME);
  const int64_t ref_skew_ms = (mic_stamp - ref_stamp).nanoseconds() / 1000000;
  const bool has_ref = ref && (ref_skew_ms >= 0) && (ref_skew_ms <= config_.ref_timeout_ms);

  if (!has_ref) {
    mic_drop_.fetch_add(1);
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping mic frame because reference is missing or outside timestamp contract: "
      "mic_ref_skew_ms=%ld timeout_ms=%d",
      static_cast<long>(ref_skew_ms), config_.ref_timeout_ms);
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

  std::vector<uint8_t> out_bytes;
  if (!backend_) {
    throw std::logic_error("fa_aec_linear backend is not initialized");
  }
  const backends::ProcessResult process_result =
    backend_->process(msg->data, ref->data, out_bytes);
  if (process_result.status != backends::ProcessStatus::kOk) {
    mic_drop_.fetch_add(1);
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping AEC linear output frame because baseline_linear backend rejected input or output: %s. "
      "Add an explicit dynamics/limiter node if range control is required.",
      backends::processStatusMessage(process_result.status));
    return;
  }

  fa_interfaces::msg::AudioFrame out_msg;
  out_msg.header = msg->header;
  out_msg.source_id = msg->source_id;
  out_msg.stream_id = config_.output_stream_id;
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
  if (!diag_pub_) {
    throw std::logic_error("fa_aec_linear diagnostics publisher is not initialized");
  }
  if (!backend_) {
    throw std::logic_error("fa_aec_linear backend is not initialized");
  }

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

  status.values.reserve(20);
  push_kv("enabled", config_.enabled ? "true" : "false");
  push_kv("mic_topic", config_.mic_topic);
  push_kv("ref_topic", config_.ref_topic);
  push_kv("output_topic", config_.output_topic);
  push_kv("mic_stream_id", config_.mic_stream_id);
  push_kv("ref_stream_id", config_.ref_stream_id);
  push_kv("output_stream_id", config_.output_stream_id);
  push_kv("resolved_mic_topic", config_.resolved_mic_topic);
  push_kv("resolved_ref_topic", config_.resolved_ref_topic);
  push_kv("resolved_output_topic", config_.resolved_output_topic);
  push_kv("expected_sample_rate", std::to_string(config_.expected_sample_rate));
  push_kv("expected_channels", std::to_string(config_.expected_channels));
  push_kv("expected.encoding", config_.expected_encoding);
  push_kv("expected.bit_depth", std::to_string(config_.expected_bit_depth));
  push_kv("ref_timeout_ms", std::to_string(config_.ref_timeout_ms));
  push_kv("reference_failure_policy", config_.reference_failure_policy);
  push_kv("cancel_gain", std::to_string(config_.cancel_gain));
  push_kv("ref_age_ms", std::to_string(ref_age_ms));
  push_kv("mic.in", std::to_string(mic_in_.load()));
  push_kv("mic.out", std::to_string(mic_out_.load()));
  push_kv("mic.drop", std::to_string(mic_drop_.load()));
  push_kv("ref.in", std::to_string(ref_in_.load()));
  push_kv("ref.drop", std::to_string(ref_drop_.load()));
  push_kv("backend.name", backends::BaselineLinearBackend::kName);

  array_msg.status.push_back(status);
  diag_pub_->publish(array_msg);
}

}  // namespace fa_aec_linear

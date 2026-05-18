#include "fa_mix/fa_mix_node.hpp"

#include <algorithm>
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

#include "diagnostic_msgs/msg/diagnostic_status.hpp"
#include "diagnostic_msgs/msg/key_value.hpp"

namespace fa_mix
{

namespace
{
constexpr const char * kEncodingPcm16Le = "PCM16LE";
constexpr const char * kInterleavedLayout = "interleaved";
constexpr uint32_t kNanosecondsPerSecond = 1000000000U;
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

std::vector<std::string> readRequiredStringArray(const rclcpp::Node & node, const std::string & name)
{
  const rclcpp::Parameter parameter = getRequiredParameter(node, name);
  if (parameter.get_type() != rclcpp::ParameterType::PARAMETER_STRING_ARRAY) {
    throw std::runtime_error(name + " must be a string array");
  }
  return parameter.as_string_array();
}

std::vector<double> readRequiredDoubleArray(const rclcpp::Node & node, const std::string & name)
{
  const rclcpp::Parameter parameter = getRequiredParameter(node, name);
  if (parameter.get_type() != rclcpp::ParameterType::PARAMETER_DOUBLE_ARRAY) {
    throw std::runtime_error(name + " must be a double array");
  }
  return parameter.as_double_array();
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

void ensureUniqueIdentities(const std::vector<std::string> & values, const std::string & label)
{
  std::set<std::string> unique;
  for (const std::string & value : values) {
    const std::string normalized = removeLeadingSlashes(value);
    if (!unique.insert(normalized).second) {
      throw std::runtime_error(label + " must be unique");
    }
  }
}

void ensureStreamDoesNotMatchTopic(
  const std::string & stream_id,
  const std::vector<std::string> & raw_topics,
  const std::vector<std::string> & resolved_topics,
  const std::string & label)
{
  for (const std::string & topic : raw_topics) {
    if (sameIdentityString(stream_id, topic)) {
      throw std::runtime_error(label + " must not match raw topic names");
    }
  }
  for (const std::string & topic : resolved_topics) {
    if (sameIdentityString(stream_id, topic)) {
      throw std::runtime_error(label + " must not match resolved topic names");
    }
  }
}
}  // namespace

FaMixNode::FaMixNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("fa_mix", options)
{
  RCLCPP_INFO(this->get_logger(), "Starting FA Mix node");
  loadParameters();
  configureBackend();
  setupInterfaces();
}

void FaMixNode::loadParameters()
{
  this->declare_parameter<std::vector<std::string>>("input_topics");
  this->declare_parameter<std::vector<std::string>>("input_stream_ids");
  this->declare_parameter<std::vector<double>>("input_gains_db");
  this->declare_parameter<int>("master_index");
  this->declare_parameter<std::string>("output_topic");
  this->declare_parameter<std::string>("output.stream_id");
  this->declare_parameter<int>("expected.sample_rate");
  this->declare_parameter<int>("expected.channels");
  this->declare_parameter<int>("expected.bit_depth");
  this->declare_parameter<std::string>("expected.encoding");
  this->declare_parameter<std::string>("expected.layout");
  this->declare_parameter<int>("max_frame_age_ms");
  this->declare_parameter<int>("qos.depth");
  this->declare_parameter<bool>("qos.reliable");
  this->declare_parameter<int>("diagnostics.qos.depth");
  this->declare_parameter<bool>("diagnostics.qos.reliable");
  this->declare_parameter<int>("diagnostics.publish_period_ms");

  config_.input_topics = readRequiredStringArray(*this, "input_topics");
  config_.input_stream_ids = readRequiredStringArray(*this, "input_stream_ids");
  config_.input_gains_db = readRequiredDoubleArray(*this, "input_gains_db");
  config_.master_index = readRequiredInt(*this, "master_index");
  config_.output_topic = readRequiredString(*this, "output_topic");
  config_.output_stream_id = readRequiredString(*this, "output.stream_id");
  config_.expected_sample_rate = readRequiredInt(*this, "expected.sample_rate");
  config_.expected_channels = readRequiredInt(*this, "expected.channels");
  config_.expected_bit_depth = readRequiredInt(*this, "expected.bit_depth");
  config_.expected_encoding = readRequiredString(*this, "expected.encoding");
  config_.expected_layout = readRequiredString(*this, "expected.layout");
  config_.max_frame_age_ms = readRequiredInt(*this, "max_frame_age_ms");
  config_.qos_depth = readRequiredInt(*this, "qos.depth");
  config_.qos_reliable = readRequiredBool(*this, "qos.reliable");
  config_.diagnostics_qos_depth = readRequiredInt(*this, "diagnostics.qos.depth");
  config_.diagnostics_qos_reliable = readRequiredBool(*this, "diagnostics.qos.reliable");
  config_.diagnostics_publish_period_ms = readRequiredInt(*this, "diagnostics.publish_period_ms");

  if (config_.input_topics.empty()) {
    throw std::runtime_error("input_topics must have at least 1 topic");
  }
  if (config_.input_stream_ids.size() != config_.input_topics.size()) {
    throw std::runtime_error("input_stream_ids must match input_topics length");
  }
  if (config_.input_gains_db.empty()) {
    throw std::runtime_error("input_gains_db must be size 1 or match input_topics length");
  }
  if (config_.input_gains_db.size() != 1U && config_.input_gains_db.size() != config_.input_topics.size()) {
    throw std::runtime_error("input_gains_db must be size 1 or match input_topics length");
  }
  if (config_.master_index < 0 || config_.master_index >= static_cast<int>(config_.input_topics.size())) {
    throw std::runtime_error("master_index out of range");
  }
  if (config_.output_topic.empty()) {
    throw std::runtime_error("output_topic is required");
  }
  if (config_.output_stream_id.empty()) {
    throw std::runtime_error("output.stream_id is required");
  }

  std::vector<std::string> raw_topics = config_.input_topics;
  raw_topics.push_back(config_.output_topic);
  std::vector<std::string> resolved_topics;
  resolved_topics.reserve(raw_topics.size());
  for (const std::string & topic : raw_topics) {
    if (topic.empty()) {
      throw std::runtime_error("input_topics and output_topic must not be empty");
    }
    resolved_topics.push_back(this->get_node_topics_interface()->resolve_topic_name(topic));
  }
  ensureUniqueIdentities(raw_topics, "raw topics");
  ensureUniqueIdentities(resolved_topics, "resolved topics");

  std::vector<std::string> stream_ids = config_.input_stream_ids;
  stream_ids.push_back(config_.output_stream_id);
  for (const std::string & stream_id : stream_ids) {
    if (stream_id.empty()) {
      throw std::runtime_error("input_stream_ids and output.stream_id must not be empty");
    }
  }
  ensureUniqueIdentities(stream_ids, "stream IDs");
  for (size_t index = 0; index < stream_ids.size(); ++index) {
    ensureStreamDoesNotMatchTopic(stream_ids[index], raw_topics, resolved_topics, "stream ID");
  }

  if (config_.expected_sample_rate <= 0 || config_.expected_sample_rate > kMaxExpectedSampleRate) {
    throw std::runtime_error("expected.sample_rate must be in (0, 384000]");
  }
  if (config_.expected_channels <= 0 || config_.expected_channels > kMaxExpectedChannels) {
    throw std::runtime_error("expected.channels must be in (0, 64]");
  }
  if (config_.expected_bit_depth != 16) {
    throw std::runtime_error("fa_mix currently supports expected.bit_depth=16 only");
  }
  if (config_.expected_encoding != kEncodingPcm16Le) {
    throw std::runtime_error("fa_mix currently supports expected.encoding=PCM16LE only");
  }
  if (config_.expected_layout != kInterleavedLayout) {
    throw std::runtime_error("fa_mix requires expected.layout=interleaved");
  }
  if (config_.max_frame_age_ms <= 0) {
    throw std::runtime_error("max_frame_age_ms must be > 0");
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
  for (const double gain_db : config_.input_gains_db) {
    if (!isFinite(gain_db)) {
      throw std::runtime_error("input_gains_db must be finite");
    }
  }

  RCLCPP_INFO(
    this->get_logger(),
    "Mix config: inputs=%zu master=%d output_topic=%s output_stream=%s expected=%dHz/%dch/%dbits enc=%s layout=%s max_age=%dms qos_depth=%d reliable=%s diag_qos_depth=%d diag_reliable=%s diag=%dms",
    config_.input_topics.size(),
    config_.master_index,
    config_.output_topic.c_str(),
    config_.output_stream_id.c_str(),
    config_.expected_sample_rate,
    config_.expected_channels,
    config_.expected_bit_depth,
    config_.expected_encoding.c_str(),
    config_.expected_layout.c_str(),
    config_.max_frame_age_ms,
    config_.qos_depth,
    config_.qos_reliable ? "true" : "false",
    config_.diagnostics_qos_depth,
    config_.diagnostics_qos_reliable ? "true" : "false",
    config_.diagnostics_publish_period_ms);
}

void FaMixNode::configureBackend()
{
  std::vector<double> expanded_gains;
  expanded_gains.reserve(config_.input_topics.size());
  for (size_t i = 0; i < config_.input_topics.size(); ++i) {
    expanded_gains.push_back(
      config_.input_gains_db.size() == 1U ? config_.input_gains_db[0] : config_.input_gains_db[i]);
  }

  backend_ = std::make_unique<backends::InternalPcm16MixerBackend>(
    backends::InternalPcm16MixerConfig{config_.expected_channels, expanded_gains});
}

void FaMixNode::setupInterfaces()
{
  rclcpp::QoS qos(static_cast<size_t>(config_.qos_depth));
  if (config_.qos_reliable) {
    qos.reliable();
  } else {
    qos.best_effort();
  }

  rclcpp::QoS diagnostics_qos(static_cast<size_t>(config_.diagnostics_qos_depth));
  if (config_.diagnostics_qos_reliable) {
    diagnostics_qos.reliable();
  } else {
    diagnostics_qos.best_effort();
  }

  pub_ = this->create_publisher<fa_interfaces::msg::AudioFrame>(config_.output_topic, qos);

  subs_.resize(config_.input_topics.size());
  latest_frames_.resize(config_.input_topics.size());
  for (size_t i = 0; i < config_.input_topics.size(); ++i) {
    const std::string topic = config_.input_topics[i];
    subs_[i] = this->create_subscription<fa_interfaces::msg::AudioFrame>(
      topic,
      qos,
      [this, i](const fa_interfaces::msg::AudioFrame::SharedPtr msg) {
        this->onInputFrame(i, msg);
      });
  }

  diag_pub_ = this->create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
    "diagnostics",
    diagnostics_qos);
  diag_timer_ = this->create_wall_timer(
    std::chrono::milliseconds(config_.diagnostics_publish_period_ms),
    std::bind(&FaMixNode::publishDiagnostics, this));
}

bool FaMixNode::validateFrame(
  const fa_interfaces::msg::AudioFrame & msg,
  const std::string & expected_stream_id) const
{
  if (msg.source_id.empty() || msg.stream_id.empty()) {
    return false;
  }
  if (msg.stream_id != expected_stream_id) {
    return false;
  }
  if (!hasValidFrameStamp(msg)) {
    return false;
  }
  if (msg.layout != config_.expected_layout) {
    return false;
  }
  if (msg.encoding != config_.expected_encoding ||
      msg.bit_depth != static_cast<uint32_t>(config_.expected_bit_depth))
  {
    return false;
  }
  if (msg.sample_rate != static_cast<uint32_t>(config_.expected_sample_rate) ||
      msg.channels != static_cast<uint32_t>(config_.expected_channels))
  {
    return false;
  }
  return true;
}

void FaMixNode::onInputFrame(size_t index, const fa_interfaces::msg::AudioFrame::SharedPtr msg)
{
  in_.fetch_add(1);
  if (!msg) {
    drop_.fetch_add(1);
    return;
  }
  if (!validateFrame(*msg, config_.input_stream_ids[index])) {
    drop_.fetch_add(1);
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping invalid mix input[%zu]: stream=%s sr=%u ch=%u bits=%u enc=%s bytes=%zu",
      index,
      msg->stream_id.c_str(),
      msg->sample_rate,
      msg->channels,
      msg->bit_depth,
      msg->encoding.c_str(),
      msg->data.size());
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
  if (!backend_) {
    throw std::logic_error("FaMixNode backend is not initialized");
  }

  const rclcpp::Time base_time = frameStamp(base);
  uint32_t epoch = base.epoch;
  std::vector<std::vector<uint8_t>> input_data;
  input_data.reserve(config_.input_topics.size());

  for (size_t i = 0; i < config_.input_topics.size(); ++i) {
    fa_interfaces::msg::AudioFrame::SharedPtr frame;
    if (static_cast<int>(i) == config_.master_index) {
      frame = std::make_shared<fa_interfaces::msg::AudioFrame>(base);
    } else {
      std::lock_guard<std::mutex> lock(frames_mutex_);
      frame = latest_frames_[i];
    }
    if (!frame) {
      drop_.fetch_add(1);
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 3000,
        "Dropping mix because input %zu has no valid frame",
        i);
      return;
    }

    const rclcpp::Time other_time = frameStamp(*frame);
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

    input_data.push_back(frame->data);
    epoch = std::max<uint32_t>(epoch, frame->epoch);
  }

  std::vector<uint8_t> out_bytes;
  const auto result = backend_->mix(input_data, out_bytes);
  if (result.status != backends::MixStatus::kOk) {
    drop_.fetch_add(1);
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping mixed frame: %s. Insert an explicit dynamics/limiter node before fa_mix or lower input_gains_db.",
      backends::mixStatusMessage(result.status));
    return;
  }

  fa_interfaces::msg::AudioFrame out;
  out.header = base.header;
  out.source_id = base.source_id;
  out.stream_id = config_.output_stream_id;
  out.encoding = config_.expected_encoding;
  out.sample_rate = static_cast<uint32_t>(config_.expected_sample_rate);
  out.channels = static_cast<uint32_t>(config_.expected_channels);
  out.bit_depth = static_cast<uint32_t>(config_.expected_bit_depth);
  out.layout = config_.expected_layout;
  out.data = std::move(out_bytes);
  out.epoch = epoch;

  pub_->publish(out);
  out_.fetch_add(1);
}

void FaMixNode::publishDiagnostics()
{
  if (!backend_) {
    throw std::logic_error("FaMixNode backend is not initialized");
  }

  diagnostic_msgs::msg::DiagnosticArray array_msg;
  array_msg.header.stamp = this->now();

  diagnostic_msgs::msg::DiagnosticStatus status;
  status.name = "fa_mix";
  status.hardware_id = "dsp";
  status.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
  status.message = "running";
  status.values.reserve(20);

  pushKeyValue(status, "inputs", std::to_string(config_.input_topics.size()));
  pushKeyValue(status, "input_stream_ids", std::to_string(config_.input_stream_ids.size()));
  pushKeyValue(status, "master_index", std::to_string(config_.master_index));
  pushKeyValue(status, "output_topic", config_.output_topic);
  pushKeyValue(status, "output_stream_id", config_.output_stream_id);
  pushKeyValue(status, "backend", backends::InternalPcm16MixerBackend::kName);
  pushKeyValue(status, "backend_inputs", std::to_string(backend_->inputCount()));
  pushKeyValue(status, "backend_last_sample_count", std::to_string(backend_->lastSampleCount()));
  pushKeyValue(status, "expected.sample_rate", std::to_string(config_.expected_sample_rate));
  pushKeyValue(status, "expected.channels", std::to_string(config_.expected_channels));
  pushKeyValue(status, "expected.bit_depth", std::to_string(config_.expected_bit_depth));
  pushKeyValue(status, "expected.encoding", config_.expected_encoding);
  pushKeyValue(status, "expected.layout", config_.expected_layout);
  pushKeyValue(status, "diagnostics_qos_depth", std::to_string(config_.diagnostics_qos_depth));
  pushKeyValue(status, "diagnostics_qos_reliable", config_.diagnostics_qos_reliable ? "true" : "false");
  pushKeyValue(status, "frames.in", std::to_string(in_.load()));
  pushKeyValue(status, "frames.out", std::to_string(out_.load()));
  pushKeyValue(status, "frames.drop", std::to_string(drop_.load()));

  array_msg.status.push_back(status);
  diag_pub_->publish(array_msg);
}

}  // namespace fa_mix

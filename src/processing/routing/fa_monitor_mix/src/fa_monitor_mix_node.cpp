#include "fa_monitor_mix/fa_monitor_mix_node.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "diagnostic_msgs/msg/diagnostic_status.hpp"
#include "diagnostic_msgs/msg/key_value.hpp"
#include "fa_monitor_mix/backends/internal_monitor_mix.hpp"

namespace fa_monitor_mix
{

namespace
{
constexpr const char * kEncodingFloat32 = "FLOAT32LE";
constexpr const char * kInterleavedLayout = "interleaved";
constexpr size_t kFloat32Bytes = sizeof(float);

bool isFinite(double value)
{
  return std::isfinite(value);
}

double dbToLinear(double db)
{
  return std::pow(10.0, db / 20.0);
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

std::vector<std::string> readRequiredStringArray(
  const rclcpp::Node & node,
  const std::string & name)
{
  const rclcpp::Parameter parameter = getRequiredParameter(node, name);
  if (parameter.get_type() != rclcpp::ParameterType::PARAMETER_STRING_ARRAY) {
    throw std::runtime_error(name + " must be a string array");
  }
  return parameter.as_string_array();
}

std::vector<double> readRequiredDoubleArray(
  const rclcpp::Node & node,
  const std::string & name)
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
}  // namespace

FaMonitorMixNode::FaMonitorMixNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("fa_monitor_mix", options)
{
  RCLCPP_INFO(this->get_logger(), "Starting FA Monitor Mix node");
  loadParameters();
  configureBackend();
  setupInterfaces();
}

FaMonitorMixNode::~FaMonitorMixNode() = default;

void FaMonitorMixNode::loadParameters()
{
  this->declare_parameter<std::vector<std::string>>("input_topics");
  this->declare_parameter<std::vector<std::string>>("input_stream_ids");
  this->declare_parameter<std::vector<double>>("input_gains_db");
  this->declare_parameter<int>("master_index");
  this->declare_parameter<std::string>("output_topic");
  this->declare_parameter<std::string>("output.stream_id");
  this->declare_parameter<std::string>("output.source_id");
  this->declare_parameter<int>("expected.sample_rate");
  this->declare_parameter<int>("expected.channels");
  this->declare_parameter<std::string>("expected.encoding");
  this->declare_parameter<int>("expected.bit_depth");
  this->declare_parameter<std::string>("expected.layout");
  this->declare_parameter<int>("max_frame_age_ms");
  this->declare_parameter<int>("qos.depth");
  this->declare_parameter<bool>("qos.reliable");
  this->declare_parameter<int>("diagnostics.publish_period_ms");
  this->declare_parameter<int>("diagnostics.qos.depth");
  this->declare_parameter<bool>("diagnostics.qos.reliable");

  config_.input_topics = readRequiredStringArray(*this, "input_topics");
  config_.input_stream_ids = readRequiredStringArray(*this, "input_stream_ids");
  config_.input_gains_db = readRequiredDoubleArray(*this, "input_gains_db");
  config_.master_index = readRequiredInt(*this, "master_index");
  config_.output_topic = readRequiredString(*this, "output_topic");
  config_.output_stream_id = readRequiredString(*this, "output.stream_id");
  config_.output_source_id = readRequiredString(*this, "output.source_id");
  config_.expected_sample_rate = readRequiredInt(*this, "expected.sample_rate");
  config_.expected_channels = readRequiredInt(*this, "expected.channels");
  config_.expected_encoding = readRequiredString(*this, "expected.encoding");
  config_.expected_bit_depth = readRequiredInt(*this, "expected.bit_depth");
  config_.expected_layout = readRequiredString(*this, "expected.layout");
  config_.max_frame_age_ms = readRequiredInt(*this, "max_frame_age_ms");
  config_.qos_depth = readRequiredInt(*this, "qos.depth");
  config_.qos_reliable = readRequiredBool(*this, "qos.reliable");
  config_.diagnostics_publish_period_ms = readRequiredInt(
    *this, "diagnostics.publish_period_ms");
  config_.diagnostics_qos_depth = readRequiredInt(*this, "diagnostics.qos.depth");
  config_.diagnostics_qos_reliable = readRequiredBool(*this, "diagnostics.qos.reliable");

  if (config_.input_topics.empty()) {
    throw std::runtime_error("input_topics must contain at least one topic");
  }
  for (const auto & topic : config_.input_topics) {
    if (topic.empty()) {
      throw std::runtime_error("input_topics must not contain an empty topic");
    }
  }
  if (config_.input_stream_ids.size() != config_.input_topics.size()) {
    throw std::runtime_error("input_stream_ids must match input_topics length");
  }
  for (const auto & stream_id : config_.input_stream_ids) {
    if (stream_id.empty()) {
      throw std::runtime_error("input_stream_ids must not contain an empty stream_id");
    }
  }
  for (size_t i = 0; i < config_.input_topics.size(); ++i) {
    const std::string resolved_input =
      this->get_node_topics_interface()->resolve_topic_name(config_.input_topics[i]);
    for (size_t j = i + 1; j < config_.input_topics.size(); ++j) {
      if (resolved_input == this->get_node_topics_interface()->resolve_topic_name(config_.input_topics[j])) {
        throw std::runtime_error("resolved input_topics must be unique");
      }
    }
  }
  if (config_.input_gains_db.size() != 1 && config_.input_gains_db.size() != config_.input_topics.size()) {
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
  if (config_.output_source_id.empty()) {
    throw std::runtime_error("output.source_id is required");
  }
  const std::string resolved_output =
    this->get_node_topics_interface()->resolve_topic_name(config_.output_topic);
  for (const auto & input_topic : config_.input_topics) {
    if (resolved_output == this->get_node_topics_interface()->resolve_topic_name(input_topic)) {
      throw std::runtime_error("resolved output_topic must be distinct from input_topics");
    }
  }
  if (config_.expected_sample_rate <= 0) {
    throw std::runtime_error("expected.sample_rate must be > 0");
  }
  if (config_.expected_channels <= 0) {
    throw std::runtime_error("expected.channels must be > 0");
  }
  if (config_.expected_encoding != kEncodingFloat32) {
    throw std::runtime_error("fa_monitor_mix requires expected.encoding=FLOAT32LE");
  }
  if (config_.expected_bit_depth != 32) {
    throw std::runtime_error("fa_monitor_mix requires expected.bit_depth=32");
  }
  if (config_.expected_layout != kInterleavedLayout) {
    throw std::runtime_error("fa_monitor_mix requires expected.layout=interleaved");
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
  if (config_.diagnostics_qos_depth <= 0) {
    throw std::runtime_error("diagnostics.qos.depth must be > 0");
  }

  config_.input_gains_linear.clear();
  config_.input_gains_linear.reserve(config_.input_topics.size());
  for (size_t i = 0; i < config_.input_topics.size(); ++i) {
    const double gain_db = gainDbForIndex(i);
    const double gain_linear = dbToLinear(gain_db);
    if (!isFinite(gain_db) || !isFinite(gain_linear)) {
      throw std::runtime_error("input_gains_db must resolve to finite linear gains");
    }
    config_.input_gains_linear.push_back(gain_linear);
  }

  RCLCPP_INFO(
    this->get_logger(),
    "Monitor mix config: inputs=%zu master=%d output=%s stream_id=%s source_id=%s expected=%dHz/%d/%s/%d/%s max_age=%dms qos_depth=%d reliable=%s diag=%dms",
    config_.input_topics.size(),
    config_.master_index,
    config_.output_topic.c_str(),
    config_.output_stream_id.c_str(),
    config_.output_source_id.c_str(),
    config_.expected_sample_rate,
    config_.expected_channels,
    config_.expected_encoding.c_str(),
    config_.expected_bit_depth,
    config_.expected_layout.c_str(),
    config_.max_frame_age_ms,
    config_.qos_depth,
    config_.qos_reliable ? "true" : "false",
    config_.diagnostics_publish_period_ms);
}

void FaMonitorMixNode::configureBackend()
{
  backends::InternalMonitorMixConfig backend_config;
  backend_config.input_count = config_.input_topics.size();
  backend_config.master_index = static_cast<size_t>(config_.master_index);
  backend_config.channels = static_cast<size_t>(config_.expected_channels);
  backend_config.gains_linear = config_.input_gains_linear;
  backend_ = std::make_unique<backends::InternalMonitorMixBackend>(backend_config);
}

void FaMonitorMixNode::setupInterfaces()
{
  rclcpp::QoS qos(static_cast<size_t>(config_.qos_depth));
  if (config_.qos_reliable) {
    qos.reliable();
  } else {
    qos.best_effort();
  }

  audio_pub_ = this->create_publisher<fa_interfaces::msg::AudioFrame>(config_.output_topic, qos);
  input_subs_.resize(config_.input_topics.size());
  latest_frames_.resize(config_.input_topics.size());
  latest_frame_received_at_.resize(config_.input_topics.size(), rclcpp::Time(0, 0, RCL_ROS_TIME));

  for (size_t i = 0; i < config_.input_topics.size(); ++i) {
    input_subs_[i] = this->create_subscription<fa_interfaces::msg::AudioFrame>(
      config_.input_topics[i],
      qos,
      [this, i](const fa_interfaces::msg::AudioFrame::SharedPtr msg) {
        this->handleInputFrame(i, msg);
      });
  }

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
    std::bind(&FaMonitorMixNode::publishDiagnostics, this));
}

void FaMonitorMixNode::handleInputFrame(
  size_t index,
  const fa_interfaces::msg::AudioFrame::SharedPtr msg)
{
  frames_in_.fetch_add(1);

  if (!msg) {
    throw std::logic_error("fa_monitor_mix received a null AudioFrame pointer");
  }

  if (!validateFrame(*msg, config_.input_stream_ids[index])) {
    frames_dropped_.fetch_add(1);
    return;
  }

  {
    std::lock_guard<std::mutex> lock(frames_mutex_);
    latest_frames_[index] = msg;
    latest_frame_received_at_[index] = this->now();
  }
  frames_valid_.fetch_add(1);

  if (static_cast<int>(index) != config_.master_index) {
    return;
  }

  if (!mixAndPublish(*msg)) {
    mix_frames_dropped_.fetch_add(1);
  }
}

bool FaMonitorMixNode::validateFrame(
  const fa_interfaces::msg::AudioFrame & msg,
  const std::string & expected_stream_id)
{
  if (msg.source_id.empty()) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping monitor input frame because source_id is required");
    return false;
  }
  if (msg.stream_id != expected_stream_id) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping monitor input frame because stream_id mismatch: %s != %s",
      msg.stream_id.c_str(),
      expected_stream_id.c_str());
    return false;
  }
  if (msg.sample_rate != static_cast<uint32_t>(config_.expected_sample_rate) ||
      msg.channels != static_cast<uint32_t>(config_.expected_channels))
  {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping monitor input frame because format mismatch: frame=%uHz/%u config=%dHz/%d",
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
      "Dropping monitor input frame because encoding contract mismatch");
    return false;
  }

  const size_t bytes_per_frame = static_cast<size_t>(msg.channels) * kFloat32Bytes;
  if (msg.data.empty() || (msg.data.size() % bytes_per_frame) != 0) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping monitor input frame because data is empty or not aligned: bytes=%zu frame_bytes=%zu",
      msg.data.size(),
      bytes_per_frame);
    return false;
  }

  if (!backend_) {
    throw std::logic_error("monitor mix backend is not initialized");
  }
  const backends::ProcessStatus status = backend_->validateFrameBytes(msg.data);
  if (status != backends::ProcessStatus::kOk) {
    if (status == backends::ProcessStatus::kOutOfRangeInput ||
        status == backends::ProcessStatus::kNonFiniteInput)
    {
      range_drops_.fetch_add(1);
    }
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping monitor input frame because backend rejected bytes: %s",
      backends::processStatusMessage(status));
    return false;
  }
  return true;
}

bool FaMonitorMixNode::mixAndPublish(const fa_interfaces::msg::AudioFrame & master_frame)
{
  if (!audio_pub_) {
    throw std::logic_error("monitor mix publisher is not initialized");
  }
  if (!backend_) {
    throw std::logic_error("monitor mix backend is not initialized");
  }

  const rclcpp::Time now = this->now();

  std::vector<fa_interfaces::msg::AudioFrame::SharedPtr> frames;
  std::vector<rclcpp::Time> received_at;
  {
    std::lock_guard<std::mutex> lock(frames_mutex_);
    frames = latest_frames_;
    received_at = latest_frame_received_at_;
  }

  std::vector<std::vector<uint8_t>> input_bytes(frames.size());
  input_bytes[static_cast<size_t>(config_.master_index)] = master_frame.data;
  for (size_t i = 0; i < frames.size(); ++i) {
    if (static_cast<int>(i) == config_.master_index) {
      continue;
    }
    if (!frames[i]) {
      missing_frame_drops_.fetch_add(1);
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 3000,
        "Dropping monitor mix because input %zu has no valid frame", i);
      return false;
    }
    const int64_t age_ms = (now - received_at[i]).nanoseconds() / 1000000;
    if (age_ms < 0 || age_ms > config_.max_frame_age_ms) {
      stale_frame_drops_.fetch_add(1);
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 3000,
        "Dropping monitor mix because input %zu is stale: age_ms=%ld max_age_ms=%d",
        i,
        static_cast<long>(age_ms),
        config_.max_frame_age_ms);
      return false;
    }
    input_bytes[i] = frames[i]->data;
  }

  const backends::ProcessResult result = backend_->mix(input_bytes);
  if (result.status != backends::ProcessStatus::kOk) {
    if (result.status == backends::ProcessStatus::kByteLengthMismatch) {
      missing_frame_drops_.fetch_add(1);
    }
    if (result.status == backends::ProcessStatus::kOutOfRangeInput ||
        result.status == backends::ProcessStatus::kNonFiniteInput ||
        result.status == backends::ProcessStatus::kOutOfRangeOutput ||
        result.status == backends::ProcessStatus::kNonFiniteOutput)
    {
      range_drops_.fetch_add(1);
    }
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping monitor mix because backend rejected input or output: %s",
      backends::processStatusMessage(result.status));
    return false;
  }

  fa_interfaces::msg::AudioFrame out;
  out.header = master_frame.header;
  out.source_id = config_.output_source_id;
  out.stream_id = config_.output_stream_id;
  out.encoding = config_.expected_encoding;
  out.sample_rate = static_cast<uint32_t>(config_.expected_sample_rate);
  out.channels = static_cast<uint32_t>(config_.expected_channels);
  out.bit_depth = static_cast<uint32_t>(config_.expected_bit_depth);
  out.layout = config_.expected_layout;
  out.data = result.output;
  out.epoch = master_frame.epoch;

  audio_pub_->publish(std::move(out));
  mix_frames_out_.fetch_add(1);
  return true;
}

double FaMonitorMixNode::gainDbForIndex(size_t index) const
{
  return config_.input_gains_db.size() == 1 ? config_.input_gains_db[0] : config_.input_gains_db[index];
}

void FaMonitorMixNode::publishDiagnostics()
{
  if (!diag_pub_) {
    throw std::logic_error("monitor mix diagnostics publisher is not initialized");
  }

  diagnostic_msgs::msg::DiagnosticArray array_msg;
  array_msg.header.stamp = this->now();

  diagnostic_msgs::msg::DiagnosticStatus status;
  status.name = "fa_monitor_mix";
  status.hardware_id = "routing";
  status.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
  status.message = "running";

  status.values.reserve(18);
  pushKeyValue(status, "inputs", std::to_string(config_.input_topics.size()));
  pushKeyValue(status, "master_index", std::to_string(config_.master_index));
  pushKeyValue(status, "output_topic", config_.output_topic);
  pushKeyValue(status, "output.stream_id", config_.output_stream_id);
  pushKeyValue(status, "output.source_id", config_.output_source_id);
  pushKeyValue(status, "expected.sample_rate", std::to_string(config_.expected_sample_rate));
  pushKeyValue(status, "expected.channels", std::to_string(config_.expected_channels));
  pushKeyValue(status, "expected.encoding", config_.expected_encoding);
  pushKeyValue(status, "expected.bit_depth", std::to_string(config_.expected_bit_depth));
  pushKeyValue(status, "expected.layout", config_.expected_layout);
  pushKeyValue(status, "backend.name", backends::InternalMonitorMixBackend::kName);
  pushKeyValue(status, "frames_in", std::to_string(frames_in_.load()));
  pushKeyValue(status, "frames_valid", std::to_string(frames_valid_.load()));
  pushKeyValue(status, "frames_dropped", std::to_string(frames_dropped_.load()));
  pushKeyValue(status, "mix_frames_out", std::to_string(mix_frames_out_.load()));
  pushKeyValue(status, "mix_frames_dropped", std::to_string(mix_frames_dropped_.load()));
  pushKeyValue(status, "stale_frame_drops", std::to_string(stale_frame_drops_.load()));
  pushKeyValue(status, "missing_frame_drops", std::to_string(missing_frame_drops_.load()));
  pushKeyValue(status, "range_drops", std::to_string(range_drops_.load()));

  array_msg.status.push_back(status);
  diag_pub_->publish(array_msg);
}

}  // namespace fa_monitor_mix

#include "fa_ducking/fa_ducking_node.hpp"

#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "diagnostic_msgs/msg/diagnostic_status.hpp"
#include "diagnostic_msgs/msg/key_value.hpp"
#include "fa_ducking/backends/internal_sidechain_ducking.hpp"

namespace fa_ducking
{

namespace
{
constexpr const char * kEncodingFloat32 = "FLOAT32LE";
constexpr const char * kInterleavedLayout = "interleaved";
constexpr int kMaxExpectedSampleRate = 384000;
constexpr int kMaxExpectedChannels = 64;
constexpr int64_t kNanosecondsPerMillisecond = 1000000;

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

double readRequiredDouble(const rclcpp::Node & node, const std::string & name)
{
  const rclcpp::Parameter parameter = getRequiredParameter(node, name);
  if (parameter.get_type() != rclcpp::ParameterType::PARAMETER_DOUBLE) {
    throw std::runtime_error(name + " must be a double");
  }
  return parameter.as_double();
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

std::string nanosecondsToMillisecondsString(int64_t nanoseconds)
{
  if (nanoseconds < 0) {
    return "-1";
  }
  const double milliseconds =
    static_cast<double>(nanoseconds) / static_cast<double>(kNanosecondsPerMillisecond);
  return std::to_string(milliseconds);
}

bool streamMatchesTopic(
  const std::string & stream_id,
  const std::string & raw_program,
  const std::string & raw_sidechain,
  const std::string & raw_output,
  const std::string & resolved_program,
  const std::string & resolved_sidechain,
  const std::string & resolved_output)
{
  return sameIdentityString(stream_id, raw_program) ||
         sameIdentityString(stream_id, raw_sidechain) ||
         sameIdentityString(stream_id, raw_output) ||
         sameIdentityString(stream_id, resolved_program) ||
         sameIdentityString(stream_id, resolved_sidechain) ||
         sameIdentityString(stream_id, resolved_output);
}
}  // namespace

FaDuckingNode::FaDuckingNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("fa_ducking", options)
{
  RCLCPP_INFO(this->get_logger(), "Starting FA Ducking node");
  loadParameters();
  configureBackend();
  setupInterfaces();
}

FaDuckingNode::~FaDuckingNode() = default;

void FaDuckingNode::loadParameters()
{
  this->declare_parameter<std::string>("program_topic");
  this->declare_parameter<std::string>("sidechain_topic");
  this->declare_parameter<std::string>("output_topic");
  this->declare_parameter<std::string>("program_stream_id");
  this->declare_parameter<std::string>("sidechain_stream_id");
  this->declare_parameter<std::string>("output.stream_id");
  this->declare_parameter<double>("sidechain.threshold_rms");
  this->declare_parameter<int>("sidechain.max_age_ms");
  this->declare_parameter<double>("ducking.gain_db");
  this->declare_parameter<double>("ducking.attack_ms");
  this->declare_parameter<double>("ducking.release_ms");
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

  config_.program_topic = readRequiredString(*this, "program_topic");
  config_.sidechain_topic = readRequiredString(*this, "sidechain_topic");
  config_.output_topic = readRequiredString(*this, "output_topic");
  config_.program_stream_id = readRequiredString(*this, "program_stream_id");
  config_.sidechain_stream_id = readRequiredString(*this, "sidechain_stream_id");
  config_.output_stream_id = readRequiredString(*this, "output.stream_id");
  config_.sidechain_threshold_rms = readRequiredDouble(*this, "sidechain.threshold_rms");
  config_.sidechain_max_age_ms = readRequiredInt(*this, "sidechain.max_age_ms");
  config_.ducking_gain_db = readRequiredDouble(*this, "ducking.gain_db");
  config_.attack_ms = readRequiredDouble(*this, "ducking.attack_ms");
  config_.release_ms = readRequiredDouble(*this, "ducking.release_ms");
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

  if (config_.program_topic.empty()) {
    throw std::runtime_error("program_topic is required");
  }
  if (config_.sidechain_topic.empty()) {
    throw std::runtime_error("sidechain_topic is required");
  }
  if (config_.output_topic.empty()) {
    throw std::runtime_error("output_topic is required");
  }
  const std::string resolved_program =
    this->get_node_topics_interface()->resolve_topic_name(config_.program_topic);
  const std::string resolved_sidechain =
    this->get_node_topics_interface()->resolve_topic_name(config_.sidechain_topic);
  const std::string resolved_output =
    this->get_node_topics_interface()->resolve_topic_name(config_.output_topic);
  if (resolved_program == resolved_sidechain ||
      resolved_program == resolved_output ||
      resolved_sidechain == resolved_output)
  {
    throw std::runtime_error("program_topic, sidechain_topic, and output_topic must resolve to distinct ROS topics");
  }
  if (config_.program_stream_id.empty()) {
    throw std::runtime_error("program_stream_id is required");
  }
  if (config_.sidechain_stream_id.empty()) {
    throw std::runtime_error("sidechain_stream_id is required");
  }
  if (config_.output_stream_id.empty()) {
    throw std::runtime_error("output.stream_id is required");
  }
  if (config_.program_stream_id == config_.sidechain_stream_id ||
      config_.program_stream_id == config_.output_stream_id ||
      config_.sidechain_stream_id == config_.output_stream_id)
  {
    throw std::runtime_error("program, sidechain, and output stream ids must be distinct");
  }
  if (streamMatchesTopic(
      config_.program_stream_id,
      config_.program_topic,
      config_.sidechain_topic,
      config_.output_topic,
      resolved_program,
      resolved_sidechain,
      resolved_output))
  {
    throw std::runtime_error("program_stream_id must be distinct from ROS topics");
  }
  if (streamMatchesTopic(
      config_.sidechain_stream_id,
      config_.program_topic,
      config_.sidechain_topic,
      config_.output_topic,
      resolved_program,
      resolved_sidechain,
      resolved_output))
  {
    throw std::runtime_error("sidechain_stream_id must be distinct from ROS topics");
  }
  if (streamMatchesTopic(
      config_.output_stream_id,
      config_.program_topic,
      config_.sidechain_topic,
      config_.output_topic,
      resolved_program,
      resolved_sidechain,
      resolved_output))
  {
    throw std::runtime_error("output.stream_id must be distinct from ROS topics");
  }
  if (!isFinite(config_.sidechain_threshold_rms) ||
      config_.sidechain_threshold_rms <= 0.0 ||
      config_.sidechain_threshold_rms > 1.0)
  {
    throw std::runtime_error("sidechain.threshold_rms must be finite and in (0.0, 1.0]");
  }
  if (config_.sidechain_max_age_ms <= 0) {
    throw std::runtime_error("sidechain.max_age_ms must be > 0");
  }
  if (!isFinite(config_.ducking_gain_db) ||
      config_.ducking_gain_db < -96.0 ||
      config_.ducking_gain_db >= 0.0)
  {
    throw std::runtime_error("ducking.gain_db must be finite and in [-96.0, 0.0)");
  }
  if (!isFinite(config_.attack_ms) || config_.attack_ms <= 0.0) {
    throw std::runtime_error("ducking.attack_ms must be finite and > 0.0");
  }
  if (!isFinite(config_.release_ms) || config_.release_ms <= 0.0) {
    throw std::runtime_error("ducking.release_ms must be finite and > 0.0");
  }
  if (config_.expected_sample_rate <= 0 || config_.expected_sample_rate > kMaxExpectedSampleRate) {
    throw std::runtime_error("expected.sample_rate must satisfy 0 < value <= 384000");
  }
  if (config_.expected_channels <= 0 || config_.expected_channels > kMaxExpectedChannels) {
    throw std::runtime_error("expected.channels must satisfy 0 < value <= 64");
  }
  if (config_.expected_encoding != kEncodingFloat32) {
    throw std::runtime_error("fa_ducking requires expected.encoding=FLOAT32LE");
  }
  if (config_.expected_bit_depth != 32) {
    throw std::runtime_error("fa_ducking requires expected.bit_depth=32");
  }
  if (config_.expected_layout != kInterleavedLayout) {
    throw std::runtime_error("fa_ducking requires expected.layout=interleaved");
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
    "Ducking config: program_topic=%s sidechain_topic=%s output_topic=%s "
    "program_stream_id=%s sidechain_stream_id=%s output_stream_id=%s "
    "threshold_rms=%f gain_db=%f attack=%fms release=%fms sidechain_max_age=%dms "
    "expected=%dHz/%d/%s/%d/%s qos_depth=%d reliable=%s diag_qos_depth=%d "
    "diag_reliable=%s diag=%dms",
    config_.program_topic.c_str(),
    config_.sidechain_topic.c_str(),
    config_.output_topic.c_str(),
    config_.program_stream_id.c_str(),
    config_.sidechain_stream_id.c_str(),
    config_.output_stream_id.c_str(),
    config_.sidechain_threshold_rms,
    config_.ducking_gain_db,
    config_.attack_ms,
    config_.release_ms,
    config_.sidechain_max_age_ms,
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

void FaDuckingNode::configureBackend()
{
  const int64_t max_age_ns =
    static_cast<int64_t>(config_.sidechain_max_age_ms) * kNanosecondsPerMillisecond;
  backend_ = std::make_unique<backends::InternalSidechainDuckingBackend>(
    backends::InternalSidechainDuckingConfig{
      config_.expected_channels,
      config_.expected_sample_rate,
      config_.sidechain_threshold_rms,
      max_age_ns,
      config_.ducking_gain_db,
      config_.attack_ms,
      config_.release_ms});
}

void FaDuckingNode::setupInterfaces()
{
  rclcpp::QoS qos(static_cast<size_t>(config_.qos_depth));
  if (config_.qos_reliable) {
    qos.reliable();
  } else {
    qos.best_effort();
  }

  audio_pub_ = this->create_publisher<fa_interfaces::msg::AudioFrame>(config_.output_topic, qos);
  program_sub_ = this->create_subscription<fa_interfaces::msg::AudioFrame>(
    config_.program_topic,
    qos,
    std::bind(&FaDuckingNode::handleProgramFrame, this, std::placeholders::_1));
  sidechain_sub_ = this->create_subscription<fa_interfaces::msg::AudioFrame>(
    config_.sidechain_topic,
    qos,
    std::bind(&FaDuckingNode::handleSidechainFrame, this, std::placeholders::_1));

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
    std::bind(&FaDuckingNode::publishDiagnostics, this));
}

void FaDuckingNode::handleProgramFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg)
{
  program_frames_in_.fetch_add(1);

  if (!msg) {
    throw std::logic_error("FaDuckingNode received null program AudioFrame pointer");
  }
  if (!validateFrame(*msg, config_.program_stream_id, "program")) {
    program_frames_dropped_.fetch_add(1);
    return;
  }
  if (!backend_) {
    throw std::logic_error("FaDuckingNode backend is not initialized");
  }

  std::vector<uint8_t> output_data;
  const backends::ProgramResult result =
    backend_->processProgram(msg->data, nowNanoseconds(), output_data);
  if (result.status != backends::ProcessStatus::kOk) {
    const char * status_message = backends::processStatusMessage(result.status);
    program_frames_dropped_.fetch_add(1);
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping program frame because ducking backend rejected input: %s",
      status_message);
    return;
  }

  fa_interfaces::msg::AudioFrame out = *msg;
  out.stream_id = config_.output_stream_id;
  out.data = std::move(output_data);
  audio_pub_->publish(out);
  program_frames_out_.fetch_add(1);
  if (result.sidechain_stale) {
    stale_sidechain_checks_.fetch_add(1);
  }
  if (result.sidechain_active) {
    ducked_program_frames_.fetch_add(1);
  } else {
    released_program_frames_.fetch_add(1);
  }
}

void FaDuckingNode::handleSidechainFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg)
{
  sidechain_frames_in_.fetch_add(1);

  if (!msg) {
    throw std::logic_error("FaDuckingNode received null sidechain AudioFrame pointer");
  }
  if (!validateFrame(*msg, config_.sidechain_stream_id, "sidechain")) {
    sidechain_frames_dropped_.fetch_add(1);
    if (!backend_) {
      throw std::logic_error("FaDuckingNode backend is not initialized");
    }
    backend_->invalidateSidechain();
    sidechain_state_invalidations_.fetch_add(1);
    return;
  }
  if (!backend_) {
    throw std::logic_error("FaDuckingNode backend is not initialized");
  }

  const backends::SidechainResult result = backend_->observeSidechain(msg->data, nowNanoseconds());
  if (result.status != backends::ProcessStatus::kOk) {
    const char * status_message = backends::processStatusMessage(result.status);
    sidechain_frames_dropped_.fetch_add(1);
    backend_->invalidateSidechain();
    sidechain_state_invalidations_.fetch_add(1);
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping sidechain frame because ducking backend rejected input: %s",
      status_message);
    return;
  }

  sidechain_frames_valid_.fetch_add(1);
}

bool FaDuckingNode::validateFrame(
  const fa_interfaces::msg::AudioFrame & msg,
  const std::string & expected_stream_id,
  const char * input_name)
{
  if (msg.source_id.empty() || msg.stream_id.empty()) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping %s frame because source_id and stream_id are required",
      input_name);
    return false;
  }
  if (msg.stream_id != expected_stream_id) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping %s frame because stream_id mismatch: %s != %s",
      input_name,
      msg.stream_id.c_str(),
      expected_stream_id.c_str());
    return false;
  }
  if (msg.layout != config_.expected_layout) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping %s frame because layout mismatch: %s != %s",
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
      "Dropping %s frame because encoding mismatch: %s/%u != %s/%d",
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
      "Dropping %s frame because format mismatch: frame=%uHz/%u config=%dHz/%d",
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
      "Dropping %s frame because data size is invalid for FLOAT32LE interleaved samples",
      input_name);
    return false;
  }
  return true;
}

size_t FaDuckingNode::bytesPerFrame() const
{
  return static_cast<size_t>(config_.expected_channels) * sizeof(float);
}

int64_t FaDuckingNode::nowNanoseconds() const
{
  return this->now().nanoseconds();
}

void FaDuckingNode::publishDiagnostics()
{
  if (!backend_) {
    throw std::logic_error("FaDuckingNode backend is not initialized");
  }

  diagnostic_msgs::msg::DiagnosticArray array_msg;
  array_msg.header.stamp = this->now();

  diagnostic_msgs::msg::DiagnosticStatus status;
  status.name = "fa_ducking";
  status.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
  status.message = "running";
  status.values.reserve(32);
  pushKeyValue(status, "program_topic", config_.program_topic);
  pushKeyValue(status, "sidechain_topic", config_.sidechain_topic);
  pushKeyValue(status, "output_topic", config_.output_topic);
  pushKeyValue(status, "program_stream_id", config_.program_stream_id);
  pushKeyValue(status, "sidechain_stream_id", config_.sidechain_stream_id);
  pushKeyValue(status, "output_stream_id", config_.output_stream_id);
  pushKeyValue(status, "sidechain_threshold_rms", std::to_string(backend_->sidechainThresholdRms()));
  pushKeyValue(status, "sidechain_max_age_ms", nanosecondsToMillisecondsString(backend_->sidechainMaxAgeNs()));
  pushKeyValue(status, "ducking_gain_db", std::to_string(backend_->duckingGainDb()));
  pushKeyValue(status, "ducking_gain_linear", std::to_string(backend_->duckingGainLinear()));
  pushKeyValue(status, "attack_ms", std::to_string(backend_->attackMs()));
  pushKeyValue(status, "release_ms", std::to_string(backend_->releaseMs()));
  pushKeyValue(status, "current_gain", std::to_string(backend_->currentGain()));
  pushKeyValue(status, "last_target_gain", std::to_string(backend_->lastTargetGain()));
  pushKeyValue(status, "last_sidechain_rms", std::to_string(backend_->lastSidechainRms()));
  pushKeyValue(status, "last_sidechain_age_ms", nanosecondsToMillisecondsString(backend_->lastSidechainAgeNs()));
  pushKeyValue(status, "last_sidechain_active", backend_->lastSidechainActive() ? "true" : "false");
  pushKeyValue(status, "has_sidechain", backend_->hasSidechain() ? "true" : "false");
  pushKeyValue(status, "diagnostics_qos_depth", std::to_string(config_.diagnostics_qos_depth));
  pushKeyValue(status, "diagnostics_qos_reliable", config_.diagnostics_qos_reliable ? "true" : "false");
  pushKeyValue(status, "program_frames_in", std::to_string(program_frames_in_.load()));
  pushKeyValue(status, "program_frames_out", std::to_string(program_frames_out_.load()));
  pushKeyValue(status, "program_frames_dropped", std::to_string(program_frames_dropped_.load()));
  pushKeyValue(status, "sidechain_frames_in", std::to_string(sidechain_frames_in_.load()));
  pushKeyValue(status, "sidechain_frames_valid", std::to_string(sidechain_frames_valid_.load()));
  pushKeyValue(status, "sidechain_frames_dropped", std::to_string(sidechain_frames_dropped_.load()));
  pushKeyValue(status, "sidechain_state_invalidations", std::to_string(sidechain_state_invalidations_.load()));
  pushKeyValue(status, "ducked_program_frames", std::to_string(ducked_program_frames_.load()));
  pushKeyValue(status, "released_program_frames", std::to_string(released_program_frames_.load()));
  pushKeyValue(status, "stale_sidechain_checks", std::to_string(stale_sidechain_checks_.load()));

  array_msg.status.push_back(status);
  diag_pub_->publish(array_msg);
}

}  // namespace fa_ducking

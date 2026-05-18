#include "fa_file_out/fa_file_out_node.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <stdexcept>

#include "diagnostic_msgs/msg/diagnostic_status.hpp"
#include "diagnostic_msgs/msg/key_value.hpp"

namespace fa_file_out
{

namespace
{
constexpr const char * kBackendPcmFileWriter = "pcm_file_writer";
constexpr const char * kEncodingPcm16 = "PCM16LE";
constexpr const char * kEncodingPcm32 = "PCM32LE";
constexpr const char * kEncodingFloat32 = "FLOAT32LE";
constexpr const char * kInterleavedLayout = "interleaved";

bool isSupportedEncodingPair(const std::string & encoding, const int bit_depth)
{
  return (encoding == kEncodingPcm16 && bit_depth == 16) ||
         (encoding == kEncodingPcm32 && bit_depth == 32) ||
         (encoding == kEncodingFloat32 && bit_depth == 32);
}

void requirePositive(const std::string & name, const int value)
{
  if (value <= 0) {
    throw std::runtime_error(name + " must be > 0");
  }
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

FaFileOutNode::FaFileOutNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("fa_file_out", options)
{
  RCLCPP_INFO(this->get_logger(), "Starting FA File Out node");
  loadParameters();
  validateConfig();
  openFile();
  setupInterfaces();
}

FaFileOutNode::~FaFileOutNode()
{
  if (backend_) {
    backend_->close();
  }
}

bool FaFileOutNode::hasFatalError() const
{
  return fatal_error_.load();
}

void FaFileOutNode::loadParameters()
{
  this->declare_parameter<std::string>("backend.name");
  this->declare_parameter<std::string>("file.path");
  this->declare_parameter<std::string>("input_topic");
  this->declare_parameter<int>("expected.sample_rate");
  this->declare_parameter<int>("expected.channels");
  this->declare_parameter<std::string>("expected.encoding");
  this->declare_parameter<int>("expected.bit_depth");
  this->declare_parameter<std::string>("expected.layout");
  this->declare_parameter<bool>("overwrite.enabled");
  this->declare_parameter<int>("qos.depth");
  this->declare_parameter<bool>("qos.reliable");
  this->declare_parameter<int>("diagnostics.publish_period_ms");

  config_.backend_name = readRequiredString(*this, "backend.name");
  config_.file_path = readRequiredString(*this, "file.path");
  config_.input_topic = readRequiredString(*this, "input_topic");
  config_.expected_sample_rate = readRequiredInt(*this, "expected.sample_rate");
  config_.expected_channels = readRequiredInt(*this, "expected.channels");
  config_.expected_encoding = readRequiredString(*this, "expected.encoding");
  config_.expected_bit_depth = readRequiredInt(*this, "expected.bit_depth");
  config_.expected_layout = readRequiredString(*this, "expected.layout");
  config_.overwrite_enabled = readRequiredBool(*this, "overwrite.enabled");
  config_.qos_depth = readRequiredInt(*this, "qos.depth");
  config_.qos_reliable = readRequiredBool(*this, "qos.reliable");
  config_.diagnostics_publish_period_ms = readRequiredInt(
    *this, "diagnostics.publish_period_ms");
}

void FaFileOutNode::validateConfig() const
{
  if (config_.backend_name.empty()) {
    throw std::runtime_error("backend.name is required");
  }
  if (config_.backend_name != kBackendPcmFileWriter) {
    throw std::runtime_error("unsupported fa_file_out backend.name: " + config_.backend_name);
  }
  if (config_.file_path.empty()) {
    throw std::runtime_error("file.path is required");
  }
  if (config_.input_topic.empty()) {
    throw std::runtime_error("input_topic is required");
  }

  requirePositive("expected.sample_rate", config_.expected_sample_rate);
  requirePositive("expected.channels", config_.expected_channels);
  requirePositive("expected.bit_depth", config_.expected_bit_depth);
  requirePositive("qos.depth", config_.qos_depth);
  requirePositive("diagnostics.publish_period_ms", config_.diagnostics_publish_period_ms);

  if (config_.expected_layout != kInterleavedLayout) {
    throw std::runtime_error("expected.layout must be interleaved");
  }
  if (!isSupportedEncodingPair(config_.expected_encoding, config_.expected_bit_depth)) {
    throw std::runtime_error(
      "expected.encoding/expected.bit_depth must be one of PCM16LE/16, PCM32LE/32, FLOAT32LE/32");
  }
}

void FaFileOutNode::openFile()
{
  backend_ = std::make_unique<backends::PcmFileWriterBackend>();
  try {
    backend_->open(config_.file_path, config_.overwrite_enabled);
  } catch (const backends::BackendError & e) {
    throw std::runtime_error(e.what());
  }

  RCLCPP_INFO(
    this->get_logger(),
    "File sink config: path=%s input=%s expected=%dHz/%d/%s/%d/%s overwrite=%s",
    config_.file_path.c_str(),
    config_.input_topic.c_str(),
    config_.expected_sample_rate,
    config_.expected_channels,
    config_.expected_encoding.c_str(),
    config_.expected_bit_depth,
    config_.expected_layout.c_str(),
    config_.overwrite_enabled ? "true" : "false");
}

void FaFileOutNode::setupInterfaces()
{
  rclcpp::QoS qos(std::max<int>(1, config_.qos_depth));
  if (config_.qos_reliable) {
    qos.reliable();
  } else {
    qos.best_effort();
  }

  audio_sub_ = this->create_subscription<fa_interfaces::msg::AudioFrame>(
    config_.input_topic,
    qos,
    std::bind(&FaFileOutNode::handleFrame, this, std::placeholders::_1));
  diag_pub_ = this->create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
    "diagnostics",
    rclcpp::SystemDefaultsQoS());
  diag_timer_ = this->create_wall_timer(
    std::chrono::milliseconds(config_.diagnostics_publish_period_ms),
    std::bind(&FaFileOutNode::publishDiagnostics, this));
}

void FaFileOutNode::handleFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg)
{
  if (fatal_error_.load()) {
    return;
  }
  if (!msg) {
    frames_rejected_.fetch_add(1);
    failClosed("received null AudioFrame");
    return;
  }
  if (!validateFrame(*msg)) {
    frames_rejected_.fetch_add(1);
    failClosed("incoming AudioFrame does not match expected raw PCM file sink contract");
    return;
  }
  if (!backend_) {
    failClosed("pcm_file_writer backend is required");
    return;
  }

  try {
    backend_->write(msg->data.data(), msg->data.size());
  } catch (const backends::BackendError & e) {
    failClosed(e.what());
    return;
  }

  frames_written_.fetch_add(1);
  bytes_written_.fetch_add(msg->data.size());
}

bool FaFileOutNode::validateFrame(const fa_interfaces::msg::AudioFrame & msg)
{
  if (msg.source_id.empty() || msg.stream_id.empty()) {
    RCLCPP_ERROR(this->get_logger(), "AudioFrame source_id and stream_id are required");
    return false;
  }
  if (msg.sample_rate != static_cast<uint32_t>(config_.expected_sample_rate) ||
      msg.channels != static_cast<uint32_t>(config_.expected_channels))
  {
    RCLCPP_ERROR(
      this->get_logger(),
      "AudioFrame sample rate/channels mismatch: %uHz/%u != %dHz/%d",
      msg.sample_rate,
      msg.channels,
      config_.expected_sample_rate,
      config_.expected_channels);
    return false;
  }
  if (msg.encoding != config_.expected_encoding ||
      msg.bit_depth != static_cast<uint32_t>(config_.expected_bit_depth))
  {
    RCLCPP_ERROR(
      this->get_logger(),
      "AudioFrame encoding mismatch: %s/%u != %s/%d",
      msg.encoding.c_str(),
      msg.bit_depth,
      config_.expected_encoding.c_str(),
      config_.expected_bit_depth);
    return false;
  }
  if (msg.layout != config_.expected_layout) {
    RCLCPP_ERROR(
      this->get_logger(),
      "AudioFrame layout mismatch: %s != %s",
      msg.layout.c_str(),
      config_.expected_layout.c_str());
    return false;
  }
  if (msg.data.empty() || (msg.data.size() % bytesPerFrame()) != 0) {
    RCLCPP_ERROR(
      this->get_logger(),
      "AudioFrame payload byte size must be non-empty and divisible by expected frame byte size");
    return false;
  }
  return true;
}

void FaFileOutNode::publishDiagnostics()
{
  diagnostic_msgs::msg::DiagnosticArray array_msg;
  array_msg.header.stamp = this->now();

  diagnostic_msgs::msg::DiagnosticStatus status;
  status.name = "fa_file_out";
  status.hardware_id = config_.file_path;
  status.level = fatal_error_.load() ? diagnostic_msgs::msg::DiagnosticStatus::ERROR
                                     : diagnostic_msgs::msg::DiagnosticStatus::OK;
  status.message = fatal_error_.load() ? "failed" : "running";

  status.values.reserve(6);
  pushKeyValue(status, "backend.name", config_.backend_name);
  pushKeyValue(status, "input_topic", config_.input_topic);
  pushKeyValue(status, "frames_written", std::to_string(frames_written_.load()));
  pushKeyValue(status, "frames_rejected", std::to_string(frames_rejected_.load()));
  pushKeyValue(status, "bytes_written", std::to_string(bytes_written_.load()));
  if (backend_) {
    pushKeyValue(status, "backend_bytes_written", std::to_string(backend_->bytesWritten()));
  }

  array_msg.status.push_back(status);
  diag_pub_->publish(array_msg);
}

void FaFileOutNode::failClosed(const std::string & reason)
{
  if (fatal_error_.exchange(true)) {
    return;
  }
  RCLCPP_FATAL(this->get_logger(), "Failing closed: %s", reason.c_str());
  if (backend_) {
    backend_->close();
  }
  rclcpp::shutdown();
}

size_t FaFileOutNode::bytesPerFrame() const
{
  return static_cast<size_t>(config_.expected_channels) *
         (static_cast<size_t>(config_.expected_bit_depth) / 8U);
}

}  // namespace fa_file_out

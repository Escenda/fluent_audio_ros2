#include "fa_file_in/fa_file_in_node.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <stdexcept>
#include <vector>

#include "diagnostic_msgs/msg/diagnostic_status.hpp"
#include "diagnostic_msgs/msg/key_value.hpp"

namespace fa_file_in
{

namespace
{
constexpr const char * kBackendPcmFileReader = "pcm_file_reader";
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

uint32_t requirePositiveUint32(const std::string & name, const int value)
{
  if (value <= 0) {
    throw std::runtime_error(name + " must be > 0");
  }
  return static_cast<uint32_t>(value);
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

FaFileInNode::FaFileInNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("fa_file_in", options)
{
  RCLCPP_INFO(this->get_logger(), "Starting FA File In node");
  loadParameters();
  validateConfig();
  openFile();
  setupInterfaces();
}

FaFileInNode::~FaFileInNode()
{
  if (backend_) {
    backend_->close();
  }
}

bool FaFileInNode::hasFatalError() const
{
  return fatal_error_.load();
}

void FaFileInNode::loadParameters()
{
  this->declare_parameter<std::string>("backend.name");
  this->declare_parameter<std::string>("file.path");
  this->declare_parameter<std::string>("output_topic");
  this->declare_parameter<std::string>("audio.source_id");
  this->declare_parameter<std::string>("audio.stream_id");
  this->declare_parameter<int>("expected.sample_rate");
  this->declare_parameter<int>("expected.channels");
  this->declare_parameter<std::string>("expected.encoding");
  this->declare_parameter<int>("expected.bit_depth");
  this->declare_parameter<std::string>("expected.layout");
  this->declare_parameter<int>("audio.frames_per_chunk");
  this->declare_parameter<bool>("playback.loop");
  this->declare_parameter<int>("playback.publish_period_ms");
  this->declare_parameter<int>("qos.depth");
  this->declare_parameter<bool>("qos.reliable");
  this->declare_parameter<int>("diagnostics.publish_period_ms");

  config_.backend_name = readRequiredString(*this, "backend.name");
  config_.file_path = readRequiredString(*this, "file.path");
  config_.output_topic = readRequiredString(*this, "output_topic");
  config_.source_id = readRequiredString(*this, "audio.source_id");
  config_.stream_id = readRequiredString(*this, "audio.stream_id");
  config_.expected_sample_rate = readRequiredInt(*this, "expected.sample_rate");
  config_.expected_channels = readRequiredInt(*this, "expected.channels");
  config_.expected_encoding = readRequiredString(*this, "expected.encoding");
  config_.expected_bit_depth = readRequiredInt(*this, "expected.bit_depth");
  config_.expected_layout = readRequiredString(*this, "expected.layout");
  config_.frames_per_chunk = readRequiredInt(*this, "audio.frames_per_chunk");
  config_.playback_loop = readRequiredBool(*this, "playback.loop");
  config_.playback_publish_period_ms = readRequiredInt(*this, "playback.publish_period_ms");
  config_.qos_depth = readRequiredInt(*this, "qos.depth");
  config_.qos_reliable = readRequiredBool(*this, "qos.reliable");
  config_.diagnostics_publish_period_ms = readRequiredInt(
    *this, "diagnostics.publish_period_ms");
}

void FaFileInNode::validateConfig() const
{
  if (config_.backend_name.empty()) {
    throw std::runtime_error("backend.name is required");
  }
  if (config_.backend_name != kBackendPcmFileReader) {
    throw std::runtime_error("unsupported fa_file_in backend.name: " + config_.backend_name);
  }
  if (config_.file_path.empty()) {
    throw std::runtime_error("file.path is required");
  }
  if (config_.output_topic.empty()) {
    throw std::runtime_error("output_topic is required");
  }
  if (config_.source_id.empty()) {
    throw std::runtime_error("audio.source_id is required");
  }
  if (config_.stream_id.empty()) {
    throw std::runtime_error("audio.stream_id is required");
  }

  requirePositiveUint32("expected.sample_rate", config_.expected_sample_rate);
  requirePositiveUint32("expected.channels", config_.expected_channels);
  requirePositiveUint32("expected.bit_depth", config_.expected_bit_depth);
  requirePositiveUint32("audio.frames_per_chunk", config_.frames_per_chunk);
  requirePositiveUint32("playback.publish_period_ms", config_.playback_publish_period_ms);
  requirePositiveUint32("qos.depth", config_.qos_depth);
  requirePositiveUint32("diagnostics.publish_period_ms", config_.diagnostics_publish_period_ms);

  if (config_.expected_layout != kInterleavedLayout) {
    throw std::runtime_error("expected.layout must be interleaved");
  }
  if (!isSupportedEncodingPair(config_.expected_encoding, config_.expected_bit_depth)) {
    throw std::runtime_error(
      "expected.encoding/expected.bit_depth must be one of PCM16LE/16, PCM32LE/32, FLOAT32LE/32");
  }
}

void FaFileInNode::openFile()
{
  backend_ = std::make_unique<backends::PcmFileReaderBackend>();
  try {
    backend_->open(config_.file_path);
  } catch (const backends::BackendError & e) {
    throw std::runtime_error(e.what());
  }

  if ((backend_->fileSizeBytes() % bytesPerFrame()) != 0) {
    throw std::runtime_error("file payload byte size must be divisible by expected frame byte size");
  }

  RCLCPP_INFO(
    this->get_logger(),
    "File source config: path=%s output=%s expected=%dHz/%d/%s/%d/%s chunk_frames=%d period=%dms loop=%s",
    config_.file_path.c_str(),
    config_.output_topic.c_str(),
    config_.expected_sample_rate,
    config_.expected_channels,
    config_.expected_encoding.c_str(),
    config_.expected_bit_depth,
    config_.expected_layout.c_str(),
    config_.frames_per_chunk,
    config_.playback_publish_period_ms,
    config_.playback_loop ? "true" : "false");
}

void FaFileInNode::setupInterfaces()
{
  rclcpp::QoS qos(std::max<int>(1, config_.qos_depth));
  if (config_.qos_reliable) {
    qos.reliable();
  } else {
    qos.best_effort();
  }

  audio_pub_ = this->create_publisher<fa_interfaces::msg::AudioFrame>(config_.output_topic, qos);
  diag_pub_ = this->create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
    "diagnostics",
    rclcpp::SystemDefaultsQoS());

  publish_timer_ = this->create_wall_timer(
    std::chrono::milliseconds(config_.playback_publish_period_ms),
    std::bind(&FaFileInNode::publishNextChunk, this));
  diag_timer_ = this->create_wall_timer(
    std::chrono::milliseconds(config_.diagnostics_publish_period_ms),
    std::bind(&FaFileInNode::publishDiagnostics, this));
}

void FaFileInNode::publishNextChunk()
{
  if (completed_.load() || fatal_error_.load()) {
    return;
  }
  if (!backend_) {
    failClosed("pcm_file_reader backend is required");
    return;
  }

  std::vector<uint8_t> payload(bytesPerChunk());
  size_t read_bytes = 0;
  try {
    read_bytes = backend_->read(payload.data(), payload.size());
  } catch (const backends::BackendError & e) {
    failClosed(e.what());
    return;
  }

  if (read_bytes == 0) {
    loops_completed_.fetch_add(1);
    if (config_.playback_loop) {
      try {
        backend_->reset();
      } catch (const backends::BackendError & e) {
        failClosed(e.what());
      }
      return;
    }
    completed_.store(true);
    if (publish_timer_) {
      publish_timer_->cancel();
    }
    return;
  }

  if ((read_bytes % bytesPerFrame()) != 0) {
    failClosed("file payload chunk byte size is not divisible by expected frame byte size");
    return;
  }
  if (read_bytes < payload.size()) {
    short_chunks_published_.fetch_add(1);
  }

  payload.resize(read_bytes);
  auto frame = buildFrame(payload.data(), payload.size());
  audio_pub_->publish(frame);
  frames_published_.fetch_add(1);
  bytes_published_.fetch_add(read_bytes);
}

fa_interfaces::msg::AudioFrame FaFileInNode::buildFrame(const uint8_t * data, const size_t byte_count)
{
  fa_interfaces::msg::AudioFrame frame_msg;
  frame_msg.header.stamp = this->now();
  frame_msg.header.frame_id = config_.source_id;
  frame_msg.source_id = config_.source_id;
  frame_msg.stream_id = config_.stream_id;
  frame_msg.encoding = config_.expected_encoding;
  frame_msg.sample_rate = static_cast<uint32_t>(config_.expected_sample_rate);
  frame_msg.channels = static_cast<uint32_t>(config_.expected_channels);
  frame_msg.bit_depth = static_cast<uint32_t>(config_.expected_bit_depth);
  frame_msg.layout = config_.expected_layout;
  frame_msg.epoch = static_cast<uint32_t>(loops_completed_.load());
  frame_msg.data.assign(data, data + byte_count);
  return frame_msg;
}

void FaFileInNode::publishDiagnostics()
{
  diagnostic_msgs::msg::DiagnosticArray array_msg;
  array_msg.header.stamp = this->now();

  diagnostic_msgs::msg::DiagnosticStatus status;
  status.name = "fa_file_in";
  status.hardware_id = config_.file_path;
  status.level = fatal_error_.load() ? diagnostic_msgs::msg::DiagnosticStatus::ERROR
                : completed_.load() ? diagnostic_msgs::msg::DiagnosticStatus::WARN
                                    : diagnostic_msgs::msg::DiagnosticStatus::OK;
  status.message = fatal_error_.load() ? "failed" : completed_.load() ? "complete" : "running";

  status.values.reserve(7);
  pushKeyValue(status, "backend.name", config_.backend_name);
  pushKeyValue(status, "source_id", config_.source_id);
  pushKeyValue(status, "stream_id", config_.stream_id);
  pushKeyValue(status, "frames_published", std::to_string(frames_published_.load()));
  pushKeyValue(status, "bytes_published", std::to_string(bytes_published_.load()));
  pushKeyValue(status, "loops_completed", std::to_string(loops_completed_.load()));
  pushKeyValue(status, "short_chunks_published", std::to_string(short_chunks_published_.load()));

  array_msg.status.push_back(status);
  diag_pub_->publish(array_msg);
}

void FaFileInNode::failClosed(const std::string & reason)
{
  if (fatal_error_.exchange(true)) {
    return;
  }
  RCLCPP_FATAL(this->get_logger(), "Failing closed: %s", reason.c_str());
  if (publish_timer_) {
    publish_timer_->cancel();
  }
  if (backend_) {
    backend_->close();
  }
  rclcpp::shutdown();
}

size_t FaFileInNode::bytesPerFrame() const
{
  return static_cast<size_t>(config_.expected_channels) *
         (static_cast<size_t>(config_.expected_bit_depth) / 8U);
}

size_t FaFileInNode::bytesPerChunk() const
{
  return bytesPerFrame() * static_cast<size_t>(config_.frames_per_chunk);
}

}  // namespace fa_file_in

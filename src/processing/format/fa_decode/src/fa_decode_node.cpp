#include "fa_decode/fa_decode_node.hpp"

#include <chrono>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>

#include "diagnostic_msgs/msg/diagnostic_status.hpp"
#include "diagnostic_msgs/msg/key_value.hpp"

namespace fa_decode
{

namespace
{
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

void requirePositive(const std::string & name, const int value)
{
  if (value <= 0) {
    throw std::runtime_error(name + " must be > 0");
  }
}

backends::EncodedChunkContract chunkContractFrom(
  const fa_interfaces::msg::EncodedAudioChunk & msg)
{
  return backends::EncodedChunkContract{
    msg.codec,
    msg.container,
    msg.payload_format,
    msg.sample_rate,
    msg.channels,
    msg.duration_ns,
    msg.data.size()};
}
}  // namespace

FaDecodeNode::FaDecodeNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("fa_decode", options)
{
  RCLCPP_INFO(this->get_logger(), "Starting FA Decode node");
  loadParameters();
  setupBackend();
  setupInterfaces();
}

void FaDecodeNode::loadParameters()
{
  this->declare_parameter<std::string>("backend.name");
  this->declare_parameter<std::string>("backend.command.executable");
  this->declare_parameter<std::vector<std::string>>(
    "backend.command.arguments",
    config_.command_arguments);
  this->declare_parameter<int>("backend.command.timeout_ms");
  this->declare_parameter<int>("backend.command.max_output_bytes");
  this->declare_parameter("input_topic", config_.input_topic);
  this->declare_parameter("output_topic", config_.output_topic);
  this->declare_parameter<std::string>("input.codec");
  this->declare_parameter<std::string>("input.container");
  this->declare_parameter<std::string>("input.payload_format");
  this->declare_parameter<int>("input.sample_rate");
  this->declare_parameter<int>("input.channels");
  this->declare_parameter<int>("output.sample_rate");
  this->declare_parameter<int>("output.channels");
  this->declare_parameter<std::string>("output.encoding");
  this->declare_parameter<int>("output.bit_depth");
  this->declare_parameter<std::string>("output.layout");
  this->declare_parameter<int>("qos.depth", config_.qos_depth);
  this->declare_parameter<bool>("qos.reliable", config_.qos_reliable);
  this->declare_parameter<int>("diagnostics.publish_period_ms");

  config_.backend_name = this->get_parameter("backend.name").as_string();
  config_.command_executable =
    this->get_parameter("backend.command.executable").as_string();
  config_.command_arguments =
    this->get_parameter("backend.command.arguments").as_string_array();
  config_.command_timeout_ms = this->get_parameter("backend.command.timeout_ms").as_int();
  config_.command_max_output_bytes =
    this->get_parameter("backend.command.max_output_bytes").as_int();
  config_.input_topic = this->get_parameter("input_topic").as_string();
  config_.output_topic = this->get_parameter("output_topic").as_string();
  config_.input_codec = this->get_parameter("input.codec").as_string();
  config_.input_container = this->get_parameter("input.container").as_string();
  config_.input_payload_format = this->get_parameter("input.payload_format").as_string();
  config_.input_sample_rate = this->get_parameter("input.sample_rate").as_int();
  config_.input_channels = this->get_parameter("input.channels").as_int();
  config_.output_sample_rate = this->get_parameter("output.sample_rate").as_int();
  config_.output_channels = this->get_parameter("output.channels").as_int();
  config_.output_encoding = this->get_parameter("output.encoding").as_string();
  config_.output_bit_depth = this->get_parameter("output.bit_depth").as_int();
  config_.output_layout = this->get_parameter("output.layout").as_string();
  config_.qos_depth = this->get_parameter("qos.depth").as_int();
  config_.qos_reliable = this->get_parameter("qos.reliable").as_bool();
  config_.diagnostics_publish_period_ms =
    this->get_parameter("diagnostics.publish_period_ms").as_int();

  if (config_.backend_name.empty()) {
    throw std::runtime_error("backend.name is required");
  }
  if (config_.backend_name != backends::kBackendName) {
    throw std::runtime_error("unsupported fa_decode backend.name: " + config_.backend_name);
  }
  if (config_.input_topic.empty()) {
    throw std::runtime_error("input_topic is required");
  }
  if (config_.output_topic.empty()) {
    throw std::runtime_error("output_topic is required");
  }
  requirePositive("qos.depth", config_.qos_depth);
  requirePositive("diagnostics.publish_period_ms", config_.diagnostics_publish_period_ms);
}

void FaDecodeNode::setupBackend()
{
  backend_ = std::make_unique<backends::ExternalCodecDecoderBackend>(
    backends::ExternalCodecDecoderConfig{
      config_.command_executable,
      config_.command_arguments,
      config_.command_timeout_ms,
      config_.command_max_output_bytes,
      config_.input_codec,
      config_.input_container,
      config_.input_payload_format,
      config_.input_sample_rate,
      config_.input_channels,
      config_.output_sample_rate,
      config_.output_channels,
      config_.output_encoding,
      config_.output_bit_depth,
      config_.output_layout});

  RCLCPP_INFO(
    this->get_logger(),
    "Decode config: input=%s output=%s backend=%s command=%s encoded=%s/%s/%s/%dHz/%d "
    "pcm=%dHz/%d/%s/%d/%s qos_depth=%d reliable=%s diag=%dms",
    config_.input_topic.c_str(),
    config_.output_topic.c_str(),
    config_.backend_name.c_str(),
    config_.command_executable.c_str(),
    config_.input_codec.c_str(),
    config_.input_container.c_str(),
    config_.input_payload_format.c_str(),
    config_.input_sample_rate,
    config_.input_channels,
    config_.output_sample_rate,
    config_.output_channels,
    config_.output_encoding.c_str(),
    config_.output_bit_depth,
    config_.output_layout.c_str(),
    config_.qos_depth,
    config_.qos_reliable ? "true" : "false",
    config_.diagnostics_publish_period_ms);
}

void FaDecodeNode::setupInterfaces()
{
  rclcpp::QoS qos(static_cast<size_t>(config_.qos_depth));
  if (config_.qos_reliable) {
    qos.reliable();
  } else {
    qos.best_effort();
  }

  audio_pub_ =
    this->create_publisher<fa_interfaces::msg::AudioFrame>(config_.output_topic, qos);
  encoded_sub_ = this->create_subscription<fa_interfaces::msg::EncodedAudioChunk>(
    config_.input_topic,
    qos,
    std::bind(&FaDecodeNode::handleChunk, this, std::placeholders::_1));

  diag_pub_ = this->create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
    "diagnostics",
    rclcpp::SystemDefaultsQoS());
  diag_timer_ = this->create_wall_timer(
    std::chrono::milliseconds(config_.diagnostics_publish_period_ms),
    std::bind(&FaDecodeNode::publishDiagnostics, this));
}

void FaDecodeNode::handleChunk(const fa_interfaces::msg::EncodedAudioChunk::SharedPtr msg)
{
  chunks_in_.fetch_add(1);
  if (!msg) {
    chunks_dropped_.fetch_add(1);
    return;
  }
  if (!validateChunk(*msg)) {
    chunks_dropped_.fetch_add(1);
    return;
  }
  if (!backend_) {
    chunks_dropped_.fetch_add(1);
    RCLCPP_ERROR(this->get_logger(), "external_codec_decoder backend is required");
    return;
  }

  const backends::DecodeResult result = backend_->decode(msg->data, chunkContractFrom(*msg));
  if (result.status != backends::DecodeStatus::kOk) {
    chunks_dropped_.fetch_add(1);
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping chunk because codec decoder failed: %s (%s exit=%d)",
      backends::decodeStatusMessage(result.status),
      backends::encodedChunkContractStatusName(result.chunk_contract_status),
      result.exit_code);
    return;
  }

  fa_interfaces::msg::AudioFrame out;
  if (!buildFrame(*msg, result, out)) {
    chunks_dropped_.fetch_add(1);
    return;
  }

  audio_pub_->publish(out);
  frames_out_.fetch_add(1);
  decoded_bytes_out_.fetch_add(out.data.size());
}

bool FaDecodeNode::validateChunk(const fa_interfaces::msg::EncodedAudioChunk & msg)
{
  if (msg.source_id.empty() || msg.stream_id.empty()) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "EncodedAudioChunk source_id and stream_id are required");
    return false;
  }
  if (msg.stream_id != config_.input_topic) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "EncodedAudioChunk stream_id mismatch: %s != %s",
      msg.stream_id.c_str(),
      config_.input_topic.c_str());
    return false;
  }
  if (msg.header.stamp.sec == 0 && msg.header.stamp.nanosec == 0) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "EncodedAudioChunk header.stamp is required for decoded media timeline");
    return false;
  }

  const backends::EncodedChunkContractStatus contract_status =
    backend_->validateContract(chunkContractFrom(msg));
  if (contract_status != backends::EncodedChunkContractStatus::kOk) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "EncodedAudioChunk decode contract mismatch: %s",
      backends::encodedChunkContractStatusName(contract_status));
    return false;
  }
  return true;
}

bool FaDecodeNode::buildFrame(
  const fa_interfaces::msg::EncodedAudioChunk & in,
  const backends::DecodeResult & result,
  fa_interfaces::msg::AudioFrame & out)
{
  if (result.encoding != config_.output_encoding ||
      result.bit_depth != static_cast<uint32_t>(config_.output_bit_depth) ||
      result.sample_rate != static_cast<uint32_t>(config_.output_sample_rate) ||
      result.channels != static_cast<uint32_t>(config_.output_channels) ||
      result.layout != config_.output_layout)
  {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Decoded output contract mismatch: %s/%u/%u/%u/%s != %s/%d/%d/%d/%s",
      result.encoding.c_str(),
      result.bit_depth,
      result.sample_rate,
      result.channels,
      result.layout.c_str(),
      config_.output_encoding.c_str(),
      config_.output_bit_depth,
      config_.output_sample_rate,
      config_.output_channels,
      config_.output_layout.c_str());
    return false;
  }
  if (result.data.empty()) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Decoded AudioFrame data must not be empty");
    return false;
  }

  out.header = in.header;
  out.source_id = in.source_id;
  out.stream_id = config_.output_topic;
  out.encoding = result.encoding;
  out.sample_rate = result.sample_rate;
  out.channels = result.channels;
  out.bit_depth = result.bit_depth;
  out.layout = result.layout;
  out.data = result.data;
  out.epoch = in.epoch;
  return true;
}

void FaDecodeNode::publishDiagnostics()
{
  diagnostic_msgs::msg::DiagnosticArray array_msg;
  array_msg.header.stamp = this->now();

  diagnostic_msgs::msg::DiagnosticStatus status;
  status.name = "fa_decode";
  status.hardware_id = config_.backend_name;
  status.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
  status.message = "running";

  status.values.reserve(11);
  pushKeyValue(status, "backend.name", config_.backend_name);
  pushKeyValue(status, "input_topic", config_.input_topic);
  pushKeyValue(status, "output_topic", config_.output_topic);
  pushKeyValue(status, "input.codec", config_.input_codec);
  pushKeyValue(status, "input.container", config_.input_container);
  pushKeyValue(status, "output.encoding", config_.output_encoding);
  pushKeyValue(status, "output.bit_depth", std::to_string(config_.output_bit_depth));
  pushKeyValue(status, "chunks.in", std::to_string(chunks_in_.load()));
  pushKeyValue(status, "frames.out", std::to_string(frames_out_.load()));
  pushKeyValue(status, "chunks.drop", std::to_string(chunks_dropped_.load()));
  pushKeyValue(status, "decoded.bytes.out", std::to_string(decoded_bytes_out_.load()));

  array_msg.status.push_back(status);
  diag_pub_->publish(array_msg);
}

}  // namespace fa_decode

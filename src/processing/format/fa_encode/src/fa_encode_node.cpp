#include "fa_encode/fa_encode_node.hpp"

#include <chrono>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>

#include "diagnostic_msgs/msg/diagnostic_status.hpp"
#include "diagnostic_msgs/msg/key_value.hpp"

namespace fa_encode
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

backends::PcmFrameContract frameContractFrom(const fa_interfaces::msg::AudioFrame & msg)
{
  return backends::PcmFrameContract{
    msg.encoding,
    msg.bit_depth,
    msg.sample_rate,
    msg.channels,
    msg.layout,
    msg.data.size()};
}
}  // namespace

FaEncodeNode::FaEncodeNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("fa_encode", options)
{
  RCLCPP_INFO(this->get_logger(), "Starting FA Encode node");
  loadParameters();
  setupBackend();
  setupInterfaces();
}

void FaEncodeNode::loadParameters()
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
  this->declare_parameter<int>("input.sample_rate");
  this->declare_parameter<int>("input.channels");
  this->declare_parameter<std::string>("input.encoding");
  this->declare_parameter<int>("input.bit_depth");
  this->declare_parameter<std::string>("input.layout");
  this->declare_parameter<std::string>("output.codec");
  this->declare_parameter<std::string>("output.container");
  this->declare_parameter<std::string>("output.payload_format");
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
  config_.input_sample_rate = this->get_parameter("input.sample_rate").as_int();
  config_.input_channels = this->get_parameter("input.channels").as_int();
  config_.input_encoding = this->get_parameter("input.encoding").as_string();
  config_.input_bit_depth = this->get_parameter("input.bit_depth").as_int();
  config_.input_layout = this->get_parameter("input.layout").as_string();
  config_.output_codec = this->get_parameter("output.codec").as_string();
  config_.output_container = this->get_parameter("output.container").as_string();
  config_.output_payload_format = this->get_parameter("output.payload_format").as_string();
  config_.qos_depth = this->get_parameter("qos.depth").as_int();
  config_.qos_reliable = this->get_parameter("qos.reliable").as_bool();
  config_.diagnostics_publish_period_ms =
    this->get_parameter("diagnostics.publish_period_ms").as_int();

  if (config_.backend_name.empty()) {
    throw std::runtime_error("backend.name is required");
  }
  if (config_.backend_name != backends::kBackendName) {
    throw std::runtime_error("unsupported fa_encode backend.name: " + config_.backend_name);
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

void FaEncodeNode::setupBackend()
{
  backend_ = std::make_unique<backends::ExternalCodecEncoderBackend>(
    backends::ExternalCodecEncoderConfig{
      config_.command_executable,
      config_.command_arguments,
      config_.command_timeout_ms,
      config_.command_max_output_bytes,
      config_.input_sample_rate,
      config_.input_channels,
      config_.input_encoding,
      config_.input_bit_depth,
      config_.input_layout,
      config_.output_codec,
      config_.output_container,
      config_.output_payload_format});

  RCLCPP_INFO(
    this->get_logger(),
    "Encode config: input=%s output=%s backend=%s command=%s expected=%dHz/%d/%s/%d/%s "
    "encoded=%s/%s/%s qos_depth=%d reliable=%s diag=%dms",
    config_.input_topic.c_str(),
    config_.output_topic.c_str(),
    config_.backend_name.c_str(),
    config_.command_executable.c_str(),
    config_.input_sample_rate,
    config_.input_channels,
    config_.input_encoding.c_str(),
    config_.input_bit_depth,
    config_.input_layout.c_str(),
    config_.output_codec.c_str(),
    config_.output_container.c_str(),
    config_.output_payload_format.c_str(),
    config_.qos_depth,
    config_.qos_reliable ? "true" : "false",
    config_.diagnostics_publish_period_ms);
}

void FaEncodeNode::setupInterfaces()
{
  rclcpp::QoS qos(static_cast<size_t>(config_.qos_depth));
  if (config_.qos_reliable) {
    qos.reliable();
  } else {
    qos.best_effort();
  }

  encoded_pub_ =
    this->create_publisher<fa_interfaces::msg::EncodedAudioChunk>(config_.output_topic, qos);
  audio_sub_ = this->create_subscription<fa_interfaces::msg::AudioFrame>(
    config_.input_topic,
    qos,
    std::bind(&FaEncodeNode::handleFrame, this, std::placeholders::_1));

  diag_pub_ = this->create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
    "diagnostics",
    rclcpp::SystemDefaultsQoS());
  diag_timer_ = this->create_wall_timer(
    std::chrono::milliseconds(config_.diagnostics_publish_period_ms),
    std::bind(&FaEncodeNode::publishDiagnostics, this));
}

void FaEncodeNode::handleFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg)
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
  if (!backend_) {
    frames_dropped_.fetch_add(1);
    RCLCPP_ERROR(this->get_logger(), "external_codec_encoder backend is required");
    return;
  }

  const backends::EncodeResult result = backend_->encode(msg->data, frameContractFrom(*msg));
  if (result.status != backends::EncodeStatus::kOk) {
    frames_dropped_.fetch_add(1);
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping frame because codec encoder failed: %s (%s exit=%d)",
      backends::encodeStatusMessage(result.status),
      backends::frameContractStatusName(result.frame_contract_status),
      result.exit_code);
    return;
  }

  fa_interfaces::msg::EncodedAudioChunk out;
  if (!buildChunk(*msg, result, out)) {
    frames_dropped_.fetch_add(1);
    return;
  }

  encoded_pub_->publish(out);
  chunks_out_.fetch_add(1);
  encoded_bytes_out_.fetch_add(out.data.size());
}

bool FaEncodeNode::validateFrame(const fa_interfaces::msg::AudioFrame & msg)
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
  if (msg.header.stamp.sec == 0 && msg.header.stamp.nanosec == 0) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "AudioFrame header.stamp is required for encoded media timeline");
    return false;
  }

  const backends::FrameContractStatus contract_status =
    backend_->validateContract(frameContractFrom(msg));
  if (contract_status != backends::FrameContractStatus::kOk) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "AudioFrame encode contract mismatch: %s",
      backends::frameContractStatusName(contract_status));
    return false;
  }
  return true;
}

bool FaEncodeNode::buildChunk(
  const fa_interfaces::msg::AudioFrame & in,
  const backends::EncodeResult & result,
  fa_interfaces::msg::EncodedAudioChunk & out)
{
  if (result.codec != config_.output_codec ||
      result.container != config_.output_container ||
      result.payload_format != config_.output_payload_format)
  {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Encoded metadata mismatch: %s/%s/%s != %s/%s/%s",
      result.codec.c_str(),
      result.container.c_str(),
      result.payload_format.c_str(),
      config_.output_codec.c_str(),
      config_.output_container.c_str(),
      config_.output_payload_format.c_str());
    return false;
  }
  if (result.sample_rate != static_cast<uint32_t>(config_.input_sample_rate) ||
      result.channels != static_cast<uint32_t>(config_.input_channels))
  {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Encoded sample rate/channels mismatch");
    return false;
  }
  if (result.data.empty()) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Encoded payload must not be empty");
    return false;
  }

  const uint64_t duration_ns = durationNsFromFrame(in);
  if (duration_ns == 0) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Encoded chunk duration must be > 0");
    return false;
  }

  if (!have_epoch_ || active_epoch_ != in.epoch) {
    have_epoch_ = true;
    active_epoch_ = in.epoch;
    next_sequence_ = 0;
    next_media_time_ns_ = 0;
  }

  out.header = in.header;
  out.source_id = in.source_id;
  out.stream_id = config_.output_topic;
  out.codec = result.codec;
  out.container = result.container;
  out.payload_format = result.payload_format;
  out.sample_rate = result.sample_rate;
  out.channels = result.channels;
  out.sequence = next_sequence_;
  out.media_time_ns = next_media_time_ns_;
  out.duration_ns = duration_ns;
  out.epoch = in.epoch;
  out.data = result.data;

  next_sequence_ += 1;
  next_media_time_ns_ += duration_ns;
  return true;
}

uint64_t FaEncodeNode::durationNsFromFrame(const fa_interfaces::msg::AudioFrame & msg) const
{
  const size_t bytes_per_frame = bytesPerFrame();
  if (bytes_per_frame == 0 || msg.data.empty() || (msg.data.size() % bytes_per_frame) != 0) {
    return 0;
  }
  const size_t sample_frames = msg.data.size() / bytes_per_frame;
  const long double duration_ns =
    (static_cast<long double>(sample_frames) * 1000000000.0L) /
    static_cast<long double>(config_.input_sample_rate);
  return static_cast<uint64_t>(duration_ns);
}

size_t FaEncodeNode::bytesPerFrame() const
{
  return static_cast<size_t>(config_.input_channels) *
         backends::bytesPerSample(config_.input_bit_depth);
}

void FaEncodeNode::publishDiagnostics()
{
  diagnostic_msgs::msg::DiagnosticArray array_msg;
  array_msg.header.stamp = this->now();

  diagnostic_msgs::msg::DiagnosticStatus status;
  status.name = "fa_encode";
  status.hardware_id = config_.backend_name;
  status.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
  status.message = "running";

  status.values.reserve(11);
  pushKeyValue(status, "backend.name", config_.backend_name);
  pushKeyValue(status, "input_topic", config_.input_topic);
  pushKeyValue(status, "output_topic", config_.output_topic);
  pushKeyValue(status, "input.encoding", config_.input_encoding);
  pushKeyValue(status, "input.bit_depth", std::to_string(config_.input_bit_depth));
  pushKeyValue(status, "output.codec", config_.output_codec);
  pushKeyValue(status, "output.container", config_.output_container);
  pushKeyValue(status, "frames.in", std::to_string(frames_in_.load()));
  pushKeyValue(status, "chunks.out", std::to_string(chunks_out_.load()));
  pushKeyValue(status, "frames.drop", std::to_string(frames_dropped_.load()));
  pushKeyValue(status, "encoded.bytes.out", std::to_string(encoded_bytes_out_.load()));

  array_msg.status.push_back(status);
  diag_pub_->publish(array_msg);
}

}  // namespace fa_encode

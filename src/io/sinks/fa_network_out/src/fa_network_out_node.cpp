#include "fa_network_out/fa_network_out_node.hpp"

#include <algorithm>
#include <chrono>
#include <stdexcept>

#include "diagnostic_msgs/msg/diagnostic_status.hpp"
#include "diagnostic_msgs/msg/key_value.hpp"

namespace fa_network_out
{

namespace
{
constexpr const char * kBackendNetworkPcmSender = "network_pcm_sender";
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
}  // namespace

FaNetworkOutNode::FaNetworkOutNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("fa_network_out", options)
{
  RCLCPP_INFO(this->get_logger(), "Starting FA Network Out node");
  loadParameters();
  validateConfig();
  openEndpoint();
  setupInterfaces();
}

FaNetworkOutNode::~FaNetworkOutNode()
{
  if (backend_) {
    backend_->close();
  }
}

bool FaNetworkOutNode::hasFatalError() const
{
  return fatal_error_.load();
}

void FaNetworkOutNode::loadParameters()
{
  this->declare_parameter<std::string>("backend.name");
  this->declare_parameter<std::string>("endpoint.uri");
  this->declare_parameter<std::string>("transport.identity");
  this->declare_parameter("input_topic", config_.input_topic);
  this->declare_parameter<int>("expected.sample_rate");
  this->declare_parameter<int>("expected.channels");
  this->declare_parameter<std::string>("expected.encoding");
  this->declare_parameter<int>("expected.bit_depth");
  this->declare_parameter<std::string>("expected.layout");
  this->declare_parameter<int>("qos.depth", config_.qos_depth);
  this->declare_parameter<bool>("qos.reliable", config_.qos_reliable);
  this->declare_parameter<int>("diagnostics.publish_period_ms");

  config_.backend_name = this->get_parameter("backend.name").as_string();
  config_.endpoint_uri = this->get_parameter("endpoint.uri").as_string();
  config_.transport_identity = this->get_parameter("transport.identity").as_string();
  config_.input_topic = this->get_parameter("input_topic").as_string();
  config_.expected_sample_rate = this->get_parameter("expected.sample_rate").as_int();
  config_.expected_channels = this->get_parameter("expected.channels").as_int();
  config_.expected_encoding = this->get_parameter("expected.encoding").as_string();
  config_.expected_bit_depth = this->get_parameter("expected.bit_depth").as_int();
  config_.expected_layout = this->get_parameter("expected.layout").as_string();
  config_.qos_depth = this->get_parameter("qos.depth").as_int();
  config_.qos_reliable = this->get_parameter("qos.reliable").as_bool();
  config_.diagnostics_publish_period_ms =
    this->get_parameter("diagnostics.publish_period_ms").as_int();
}

void FaNetworkOutNode::validateConfig() const
{
  if (config_.backend_name.empty()) {
    throw std::runtime_error("backend.name is required");
  }
  if (config_.backend_name != kBackendNetworkPcmSender) {
    throw std::runtime_error("unsupported fa_network_out backend.name: " + config_.backend_name);
  }
  if (config_.endpoint_uri.empty()) {
    throw std::runtime_error("endpoint.uri is required");
  }
  if (config_.transport_identity.empty()) {
    throw std::runtime_error("transport.identity is required");
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

void FaNetworkOutNode::openEndpoint()
{
  backend_ = std::make_unique<backends::NetworkPcmSenderBackend>();
  try {
    backend_->open(config_.endpoint_uri);
  } catch (const backends::BackendError & e) {
    throw std::runtime_error(e.what());
  }

  RCLCPP_INFO(
    this->get_logger(),
    "Network sink config: endpoint=%s input=%s identity=%s expected=%dHz/%d/%s/%d/%s",
    config_.endpoint_uri.c_str(),
    config_.input_topic.c_str(),
    config_.transport_identity.c_str(),
    config_.expected_sample_rate,
    config_.expected_channels,
    config_.expected_encoding.c_str(),
    config_.expected_bit_depth,
    config_.expected_layout.c_str());
}

void FaNetworkOutNode::setupInterfaces()
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
    std::bind(&FaNetworkOutNode::handleFrame, this, std::placeholders::_1));
  diag_pub_ = this->create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
    "diagnostics",
    rclcpp::SystemDefaultsQoS());
  diag_timer_ = this->create_wall_timer(
    std::chrono::milliseconds(config_.diagnostics_publish_period_ms),
    std::bind(&FaNetworkOutNode::publishDiagnostics, this));
}

void FaNetworkOutNode::handleFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg)
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
    failClosed("incoming AudioFrame does not match expected raw PCM network sink contract");
    return;
  }
  if (!backend_) {
    failClosed("network_pcm_sender backend is required");
    return;
  }

  try {
    backend_->send(msg->data.data(), msg->data.size());
  } catch (const backends::BackendError & e) {
    failClosed(e.what());
    return;
  }

  frames_sent_.fetch_add(1);
  bytes_sent_.fetch_add(msg->data.size());
}

bool FaNetworkOutNode::validateFrame(const fa_interfaces::msg::AudioFrame & msg)
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

void FaNetworkOutNode::publishDiagnostics()
{
  diagnostic_msgs::msg::DiagnosticArray array_msg;
  array_msg.header.stamp = this->now();

  diagnostic_msgs::msg::DiagnosticStatus status;
  status.name = "fa_network_out";
  status.hardware_id = config_.endpoint_uri;
  status.level = fatal_error_.load() ? diagnostic_msgs::msg::DiagnosticStatus::ERROR
                                     : diagnostic_msgs::msg::DiagnosticStatus::OK;
  status.message = fatal_error_.load() ? "failed" : "running";

  status.values.reserve(7);
  pushKeyValue(status, "backend.name", config_.backend_name);
  pushKeyValue(status, "endpoint.uri", config_.endpoint_uri);
  pushKeyValue(status, "transport.identity", config_.transport_identity);
  pushKeyValue(status, "frames_sent", std::to_string(frames_sent_.load()));
  pushKeyValue(status, "frames_rejected", std::to_string(frames_rejected_.load()));
  pushKeyValue(status, "bytes_sent", std::to_string(bytes_sent_.load()));
  if (backend_) {
    pushKeyValue(status, "backend_packets_sent", std::to_string(backend_->packetsSent()));
  }

  array_msg.status.push_back(status);
  diag_pub_->publish(array_msg);
}

void FaNetworkOutNode::failClosed(const std::string & reason)
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

size_t FaNetworkOutNode::bytesPerFrame() const
{
  return static_cast<size_t>(config_.expected_channels) *
         (static_cast<size_t>(config_.expected_bit_depth) / 8U);
}

}  // namespace fa_network_out

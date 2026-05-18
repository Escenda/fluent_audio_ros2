#include "fa_network_in/fa_network_in_node.hpp"

#include <chrono>
#include <limits>
#include <stdexcept>
#include <vector>

#include "diagnostic_msgs/msg/diagnostic_status.hpp"
#include "diagnostic_msgs/msg/key_value.hpp"

namespace fa_network_in
{

namespace
{
constexpr const char * kBackendNetworkPcmReceiver = "network_pcm_receiver";
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
}  // namespace

FaNetworkInNode::FaNetworkInNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("fa_network_in", options)
{
  RCLCPP_INFO(this->get_logger(), "Starting FA Network In node");
  loadParameters();
  validateConfig();
  openEndpoint();
  setupInterfaces();
}

FaNetworkInNode::~FaNetworkInNode()
{
  if (backend_) {
    backend_->close();
  }
}

bool FaNetworkInNode::hasFatalError() const
{
  return fatal_error_.load();
}

void FaNetworkInNode::loadParameters()
{
  this->declare_parameter<std::string>("backend.name");
  this->declare_parameter<std::string>("endpoint.uri");
  this->declare_parameter<std::string>("transport.identity");
  this->declare_parameter<std::string>("output_topic");
  this->declare_parameter<std::string>("audio.source_id");
  this->declare_parameter<std::string>("audio.stream_id");
  this->declare_parameter<int>("expected.sample_rate");
  this->declare_parameter<int>("expected.channels");
  this->declare_parameter<std::string>("expected.encoding");
  this->declare_parameter<int>("expected.bit_depth");
  this->declare_parameter<std::string>("expected.layout");
  this->declare_parameter<int>("network.max_packet_bytes");
  this->declare_parameter<int>("polling.period_ms");
  this->declare_parameter<int>("qos.depth");
  this->declare_parameter<bool>("qos.reliable");
  this->declare_parameter<int>("diagnostics.publish_period_ms");
  this->declare_parameter<int>("diagnostics.qos.depth");
  this->declare_parameter<bool>("diagnostics.qos.reliable");

  config_.backend_name = readRequiredString(*this, "backend.name");
  config_.endpoint_uri = readRequiredString(*this, "endpoint.uri");
  config_.transport_identity = readRequiredString(*this, "transport.identity");
  config_.output_topic = readRequiredString(*this, "output_topic");
  config_.source_id = readRequiredString(*this, "audio.source_id");
  config_.stream_id = readRequiredString(*this, "audio.stream_id");
  config_.expected_sample_rate = readRequiredInt(*this, "expected.sample_rate");
  config_.expected_channels = readRequiredInt(*this, "expected.channels");
  config_.expected_encoding = readRequiredString(*this, "expected.encoding");
  config_.expected_bit_depth = readRequiredInt(*this, "expected.bit_depth");
  config_.expected_layout = readRequiredString(*this, "expected.layout");
  config_.max_packet_bytes = readRequiredInt(*this, "network.max_packet_bytes");
  config_.polling_period_ms = readRequiredInt(*this, "polling.period_ms");
  config_.qos_depth = readRequiredInt(*this, "qos.depth");
  config_.qos_reliable = readRequiredBool(*this, "qos.reliable");
  config_.diagnostics_publish_period_ms = readRequiredInt(
    *this, "diagnostics.publish_period_ms");
  config_.diagnostics_qos_depth = readRequiredInt(*this, "diagnostics.qos.depth");
  config_.diagnostics_qos_reliable = readRequiredBool(*this, "diagnostics.qos.reliable");
}

void FaNetworkInNode::validateConfig()
{
  if (config_.backend_name.empty()) {
    throw std::runtime_error("backend.name is required");
  }
  if (config_.backend_name != kBackendNetworkPcmReceiver) {
    throw std::runtime_error("unsupported fa_network_in backend.name: " + config_.backend_name);
  }
  if (config_.endpoint_uri.empty()) {
    throw std::runtime_error("endpoint.uri is required");
  }
  if (config_.transport_identity.empty()) {
    throw std::runtime_error("transport.identity is required");
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
  const std::string resolved_output_topic =
    this->get_node_topics_interface()->resolve_topic_name(config_.output_topic);
  if (sameIdentityString(config_.stream_id, config_.output_topic) ||
      sameIdentityString(config_.stream_id, resolved_output_topic)) {
    throw std::runtime_error("audio.stream_id must be distinct from output_topic");
  }

  requirePositive("expected.sample_rate", config_.expected_sample_rate);
  requirePositive("expected.channels", config_.expected_channels);
  requirePositive("expected.bit_depth", config_.expected_bit_depth);
  requirePositive("network.max_packet_bytes", config_.max_packet_bytes);
  requirePositive("polling.period_ms", config_.polling_period_ms);
  requirePositive("qos.depth", config_.qos_depth);
  requirePositive("diagnostics.publish_period_ms", config_.diagnostics_publish_period_ms);
  requirePositive("diagnostics.qos.depth", config_.diagnostics_qos_depth);

  if (config_.expected_layout != kInterleavedLayout) {
    throw std::runtime_error("expected.layout must be interleaved");
  }
  if (!isSupportedEncodingPair(config_.expected_encoding, config_.expected_bit_depth)) {
    throw std::runtime_error(
      "expected.encoding/expected.bit_depth must be one of PCM16LE/16, PCM32LE/32, FLOAT32LE/32");
  }
  if ((static_cast<size_t>(config_.max_packet_bytes) % bytesPerFrame()) != 0) {
    throw std::runtime_error("network.max_packet_bytes must be divisible by expected frame byte size");
  }
}

void FaNetworkInNode::openEndpoint()
{
  backend_ = std::make_unique<backends::NetworkPcmReceiverBackend>();
  try {
    backend_->open(config_.endpoint_uri);
  } catch (const backends::BackendError & e) {
    throw std::runtime_error(e.what());
  }

  RCLCPP_INFO(
    this->get_logger(),
    "Network source config: endpoint=%s output=%s source=%s stream=%s expected=%dHz/%d/%s/%d/%s max_packet=%d poll=%dms",
    config_.endpoint_uri.c_str(),
    config_.output_topic.c_str(),
    config_.source_id.c_str(),
    config_.stream_id.c_str(),
    config_.expected_sample_rate,
    config_.expected_channels,
    config_.expected_encoding.c_str(),
    config_.expected_bit_depth,
    config_.expected_layout.c_str(),
    config_.max_packet_bytes,
    config_.polling_period_ms);
}

void FaNetworkInNode::setupInterfaces()
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

  audio_pub_ = this->create_publisher<fa_interfaces::msg::AudioFrame>(config_.output_topic, qos);
  diag_pub_ = this->create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
    "diagnostics",
    diagnostics_qos);
  poll_timer_ = this->create_wall_timer(
    std::chrono::milliseconds(config_.polling_period_ms),
    std::bind(&FaNetworkInNode::pollEndpoint, this));
  diag_timer_ = this->create_wall_timer(
    std::chrono::milliseconds(config_.diagnostics_publish_period_ms),
    std::bind(&FaNetworkInNode::publishDiagnostics, this));
}

void FaNetworkInNode::pollEndpoint()
{
  if (fatal_error_.load()) {
    return;
  }
  if (!backend_) {
    failClosed("network_pcm_receiver backend is required");
    return;
  }

  std::vector<uint8_t> packet(static_cast<size_t>(config_.max_packet_bytes));
  backends::ReceiveResult result;
  try {
    result = backend_->receive(packet.data(), packet.size());
  } catch (const backends::BackendError & e) {
    failClosed(e.what());
    return;
  }
  if (!result.has_packet) {
    return;
  }
  if ((result.byte_count % bytesPerFrame()) != 0) {
    packets_rejected_.fetch_add(1);
    failClosed("received UDP packet byte size is not divisible by expected frame byte size");
    return;
  }

  auto frame = buildFrame(packet.data(), result.byte_count);
  audio_pub_->publish(frame);
  packets_published_.fetch_add(1);
  bytes_published_.fetch_add(result.byte_count);
}

fa_interfaces::msg::AudioFrame FaNetworkInNode::buildFrame(const uint8_t * data, const size_t byte_count)
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
  frame_msg.data.assign(data, data + byte_count);
  return frame_msg;
}

void FaNetworkInNode::publishDiagnostics()
{
  diagnostic_msgs::msg::DiagnosticArray array_msg;
  array_msg.header.stamp = this->now();

  diagnostic_msgs::msg::DiagnosticStatus status;
  status.name = "fa_network_in";
  status.hardware_id = config_.endpoint_uri;
  status.level = fatal_error_.load() ? diagnostic_msgs::msg::DiagnosticStatus::ERROR
                                     : diagnostic_msgs::msg::DiagnosticStatus::OK;
  status.message = fatal_error_.load() ? "failed" : "running";

  status.values.reserve(7);
  pushKeyValue(status, "backend.name", config_.backend_name);
  pushKeyValue(status, "endpoint.uri", config_.endpoint_uri);
  pushKeyValue(status, "transport.identity", config_.transport_identity);
  pushKeyValue(status, "source_id", config_.source_id);
  pushKeyValue(status, "stream_id", config_.stream_id);
  pushKeyValue(status, "packets_published", std::to_string(packets_published_.load()));
  pushKeyValue(status, "packets_rejected", std::to_string(packets_rejected_.load()));
  pushKeyValue(status, "bytes_published", std::to_string(bytes_published_.load()));
  if (backend_) {
    pushKeyValue(status, "backend_packets_received", std::to_string(backend_->packetsReceived()));
  }

  array_msg.status.push_back(status);
  diag_pub_->publish(array_msg);
}

void FaNetworkInNode::failClosed(const std::string & reason)
{
  if (fatal_error_.exchange(true)) {
    return;
  }
  RCLCPP_FATAL(this->get_logger(), "Failing closed: %s", reason.c_str());
  if (poll_timer_) {
    poll_timer_->cancel();
  }
  if (backend_) {
    backend_->close();
  }
  rclcpp::shutdown();
}

size_t FaNetworkInNode::bytesPerFrame() const
{
  return static_cast<size_t>(config_.expected_channels) *
         (static_cast<size_t>(config_.expected_bit_depth) / 8U);
}

}  // namespace fa_network_in

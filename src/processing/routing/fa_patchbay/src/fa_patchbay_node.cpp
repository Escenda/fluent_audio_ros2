#include "fa_patchbay/fa_patchbay_node.hpp"

#include <chrono>
#include <cstdlib>
#include <functional>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

#include "diagnostic_msgs/msg/diagnostic_status.hpp"
#include "diagnostic_msgs/msg/key_value.hpp"

namespace fa_patchbay
{

namespace
{
constexpr const char * kEncodingPcm16 = "PCM16LE";
constexpr const char * kEncodingPcm32 = "PCM32LE";
constexpr const char * kEncodingFloat32 = "FLOAT32LE";
constexpr const char * kInterleavedLayout = "interleaved";

bool isSupportedEncodingBitDepth(const std::string & encoding, int bit_depth)
{
  return (encoding == kEncodingPcm16 && bit_depth == 16) ||
         (encoding == kEncodingPcm32 && bit_depth == 32) ||
         (encoding == kEncodingFloat32 && bit_depth == 32);
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

FaPatchbayNode::FaPatchbayNode()
: rclcpp::Node("fa_patchbay")
{
  RCLCPP_INFO(this->get_logger(), "Starting FA Patchbay node");
  loadParameters();
  setupInterfaces();
}

void FaPatchbayNode::loadParameters()
{
  this->declare_parameter<std::vector<std::string>>("input_topics");
  this->declare_parameter<std::vector<std::string>>("output_topics");
  this->declare_parameter<int>("expected.sample_rate");
  this->declare_parameter<int>("expected.channels");
  this->declare_parameter<std::string>("expected.encoding");
  this->declare_parameter<int>("expected.bit_depth");
  this->declare_parameter<std::string>("expected.layout");
  this->declare_parameter<int>("qos.depth");
  this->declare_parameter<bool>("qos.reliable");
  this->declare_parameter<int>("diagnostics.publish_period_ms");

  config_.input_topics = readRequiredStringArray(*this, "input_topics");
  config_.output_topics = readRequiredStringArray(*this, "output_topics");
  config_.expected_sample_rate = readRequiredInt(*this, "expected.sample_rate");
  config_.expected_channels = readRequiredInt(*this, "expected.channels");
  config_.expected_encoding = readRequiredString(*this, "expected.encoding");
  config_.expected_bit_depth = readRequiredInt(*this, "expected.bit_depth");
  config_.expected_layout = readRequiredString(*this, "expected.layout");
  config_.qos_depth = readRequiredInt(*this, "qos.depth");
  config_.qos_reliable = readRequiredBool(*this, "qos.reliable");
  config_.diagnostics_publish_period_ms = readRequiredInt(
    *this, "diagnostics.publish_period_ms");

  if (config_.input_topics.empty()) {
    throw std::runtime_error("input_topics must contain at least one topic");
  }
  if (config_.output_topics.empty()) {
    throw std::runtime_error("output_topics must contain at least one topic");
  }
  if (config_.input_topics.size() != config_.output_topics.size()) {
    throw std::runtime_error("input_topics and output_topics must have the same length");
  }

  std::set<std::pair<std::string, std::string>> unique_routes;
  std::set<std::string> unique_inputs;
  routes_.reserve(config_.input_topics.size());
  for (size_t index = 0; index < config_.input_topics.size(); ++index) {
    const std::string & input_topic = config_.input_topics[index];
    const std::string & output_topic = config_.output_topics[index];
    if (input_topic.empty()) {
      throw std::runtime_error("input_topics must not contain empty topic");
    }
    if (output_topic.empty()) {
      throw std::runtime_error("output_topics must not contain empty topic");
    }
    if (input_topic == output_topic) {
      throw std::runtime_error("route output topic must not equal its input topic");
    }
    if (!unique_routes.insert(std::make_pair(input_topic, output_topic)).second) {
      throw std::runtime_error("route pairs must be unique");
    }
    routes_.push_back(PatchbayRoute{input_topic, output_topic, nullptr});
    route_indices_by_input_[input_topic].push_back(index);
    if (unique_inputs.insert(input_topic).second) {
      unique_input_topics_.push_back(input_topic);
    }
  }

  if (config_.expected_sample_rate <= 0) {
    throw std::runtime_error("expected.sample_rate must be > 0");
  }
  if (config_.expected_channels <= 0) {
    throw std::runtime_error("expected.channels must be > 0");
  }
  if (config_.expected_encoding.empty()) {
    throw std::runtime_error("expected.encoding is required");
  }
  if (config_.expected_bit_depth <= 0) {
    throw std::runtime_error("expected.bit_depth must be > 0");
  }
  if ((config_.expected_bit_depth % 8) != 0) {
    throw std::runtime_error("expected.bit_depth must be byte-aligned");
  }
  if (!isSupportedEncodingBitDepth(config_.expected_encoding, config_.expected_bit_depth)) {
    throw std::runtime_error(
      "expected.encoding/expected.bit_depth must be one of PCM16LE/16, PCM32LE/32, FLOAT32LE/32");
  }
  if (config_.expected_layout != kInterleavedLayout) {
    throw std::runtime_error("fa_patchbay requires expected.layout=interleaved");
  }
  if (config_.qos_depth <= 0) {
    throw std::runtime_error("qos.depth must be > 0");
  }
  if (config_.diagnostics_publish_period_ms <= 0) {
    throw std::runtime_error("diagnostics.publish_period_ms must be > 0");
  }

  RCLCPP_INFO(
    this->get_logger(),
    "Patchbay config: routes=%zu unique_inputs=%zu expected=%dHz/%d/%s/%d/%s qos_depth=%d reliable=%s diag=%dms",
    routes_.size(),
    unique_input_topics_.size(),
    config_.expected_sample_rate,
    config_.expected_channels,
    config_.expected_encoding.c_str(),
    config_.expected_bit_depth,
    config_.expected_layout.c_str(),
    config_.qos_depth,
    config_.qos_reliable ? "true" : "false",
    config_.diagnostics_publish_period_ms);
}

void FaPatchbayNode::setupInterfaces()
{
  rclcpp::QoS qos(static_cast<size_t>(config_.qos_depth));
  if (config_.qos_reliable) {
    qos.reliable();
  } else {
    qos.best_effort();
  }

  for (PatchbayRoute & route : routes_) {
    route.publisher = this->create_publisher<fa_interfaces::msg::AudioFrame>(
      route.output_topic,
      qos);
  }

  audio_subs_.reserve(unique_input_topics_.size());
  for (const std::string & input_topic : unique_input_topics_) {
    audio_subs_.push_back(this->create_subscription<fa_interfaces::msg::AudioFrame>(
      input_topic,
      qos,
      [this, input_topic](const fa_interfaces::msg::AudioFrame::SharedPtr msg) {
        this->handleFrame(input_topic, msg);
      }));
  }

  diag_pub_ = this->create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
    "diagnostics",
    rclcpp::SystemDefaultsQoS());
  diag_timer_ = this->create_wall_timer(
    std::chrono::milliseconds(config_.diagnostics_publish_period_ms),
    std::bind(&FaPatchbayNode::publishDiagnostics, this));
}

void FaPatchbayNode::handleFrame(
  const std::string & input_topic,
  const fa_interfaces::msg::AudioFrame::SharedPtr msg)
{
  frames_in_.fetch_add(1);
  if (!msg) {
    frames_dropped_.fetch_add(1);
    return;
  }

  if (!validateFrame(input_topic, *msg)) {
    frames_dropped_.fetch_add(1);
    return;
  }

  publishCopies(input_topic, *msg);
}

bool FaPatchbayNode::validateFrame(
  const std::string & input_topic,
  const fa_interfaces::msg::AudioFrame & msg)
{
  if (msg.source_id.empty() || msg.stream_id.empty()) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "AudioFrame source_id and stream_id are required");
    return false;
  }
  if (msg.stream_id != input_topic) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "AudioFrame stream_id mismatch: %s != %s",
      msg.stream_id.c_str(),
      input_topic.c_str());
    return false;
  }
  if (msg.layout != config_.expected_layout) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "AudioFrame layout mismatch: %s != %s",
      msg.layout.c_str(),
      config_.expected_layout.c_str());
    return false;
  }
  if (msg.encoding != config_.expected_encoding ||
      msg.bit_depth != static_cast<uint32_t>(config_.expected_bit_depth))
  {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "AudioFrame encoding mismatch: %s/%u != %s/%d",
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
      "AudioFrame format mismatch: frame=%uHz/%u config=%dHz/%d",
      msg.sample_rate,
      msg.channels,
      config_.expected_sample_rate,
      config_.expected_channels);
    return false;
  }
  if (msg.data.empty() || (msg.data.size() % bytesPerFrame()) != 0) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "AudioFrame data size is invalid for configured interleaved samples");
    return false;
  }
  return true;
}

void FaPatchbayNode::publishCopies(
  const std::string & input_topic,
  const fa_interfaces::msg::AudioFrame & msg)
{
  const std::vector<size_t> & route_indices = route_indices_by_input_.at(input_topic);
  for (const size_t route_index : route_indices) {
    const PatchbayRoute & route = routes_[route_index];
    fa_interfaces::msg::AudioFrame out = msg;
    out.stream_id = route.output_topic;
    route.publisher->publish(out);
    copies_out_.fetch_add(1);
  }
}

size_t FaPatchbayNode::bytesPerFrame() const
{
  return static_cast<size_t>(config_.expected_channels) *
         static_cast<size_t>(config_.expected_bit_depth / 8);
}

void FaPatchbayNode::publishDiagnostics()
{
  diagnostic_msgs::msg::DiagnosticArray array_msg;
  array_msg.header.stamp = this->now();

  diagnostic_msgs::msg::DiagnosticStatus status;
  status.name = "fa_patchbay";
  status.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
  status.message = "running";
  status.values.reserve(14 + (routes_.size() * 2));
  pushKeyValue(status, "route_count", std::to_string(routes_.size()));
  pushKeyValue(status, "unique_input_count", std::to_string(unique_input_topics_.size()));
  for (size_t index = 0; index < routes_.size(); ++index) {
    pushKeyValue(status, "input_topic." + std::to_string(index), routes_[index].input_topic);
    pushKeyValue(status, "output_topic." + std::to_string(index), routes_[index].output_topic);
  }
  pushKeyValue(status, "expected.sample_rate", std::to_string(config_.expected_sample_rate));
  pushKeyValue(status, "expected.channels", std::to_string(config_.expected_channels));
  pushKeyValue(status, "expected.encoding", config_.expected_encoding);
  pushKeyValue(status, "expected.bit_depth", std::to_string(config_.expected_bit_depth));
  pushKeyValue(status, "expected.layout", config_.expected_layout);
  pushKeyValue(status, "qos.depth", std::to_string(config_.qos_depth));
  pushKeyValue(status, "qos.reliable", config_.qos_reliable ? "true" : "false");
  pushKeyValue(status, "frames_in", std::to_string(frames_in_.load()));
  pushKeyValue(status, "frames_dropped", std::to_string(frames_dropped_.load()));
  pushKeyValue(status, "copies_out", std::to_string(copies_out_.load()));

  array_msg.status.push_back(status);
  diag_pub_->publish(array_msg);
}

}  // namespace fa_patchbay

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<fa_patchbay::FaPatchbayNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return EXIT_SUCCESS;
  } catch (const std::exception & e) {
    RCLCPP_FATAL(rclcpp::get_logger("fa_patchbay"), "Exception: %s", e.what());
    rclcpp::shutdown();
    return EXIT_FAILURE;
  }
}

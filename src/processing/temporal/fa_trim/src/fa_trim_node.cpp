#include "fa_trim/fa_trim_node.hpp"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "diagnostic_msgs/msg/diagnostic_status.hpp"
#include "diagnostic_msgs/msg/key_value.hpp"
#include "rclcpp/exceptions.hpp"

namespace fa_trim
{

namespace
{
constexpr const char * kEncodingFloat32 = "FLOAT32LE";
constexpr const char * kInterleavedLayout = "interleaved";
constexpr float kMinNormalizedSample = -1.0F;
constexpr float kMaxNormalizedSample = 1.0F;

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

template<typename ParameterT>
ParameterT declareRequiredParameter(rclcpp::Node & node, const std::string & name)
{
  try {
    return node.declare_parameter<ParameterT>(name);
  } catch (const rclcpp::exceptions::ParameterUninitializedException &) {
    throw std::runtime_error(name + " is required");
  }
}

float readFloatSample(const std::vector<uint8_t> & data, size_t sample_index)
{
  float sample = 0.0F;
  std::memcpy(&sample, data.data() + (sample_index * sizeof(float)), sizeof(float));
  return sample;
}
}  // namespace

FaTrimNode::FaTrimNode()
: rclcpp::Node("fa_trim")
{
  RCLCPP_INFO(this->get_logger(), "Starting FA Trim node");
  loadParameters();
  setupInterfaces();
}

void FaTrimNode::loadParameters()
{
  config_.input_topic = declareRequiredParameter<std::string>(*this, "input_topic");
  config_.output_topic = declareRequiredParameter<std::string>(*this, "output_topic");
  config_.leading_frames = declareRequiredParameter<int>(*this, "trim.leading_frames");
  config_.trailing_frames = declareRequiredParameter<int>(*this, "trim.trailing_frames");
  config_.expected_sample_rate = declareRequiredParameter<int>(*this, "expected.sample_rate");
  config_.expected_channels = declareRequiredParameter<int>(*this, "expected.channels");
  config_.expected_encoding = declareRequiredParameter<std::string>(*this, "expected.encoding");
  config_.expected_bit_depth = declareRequiredParameter<int>(*this, "expected.bit_depth");
  config_.expected_layout = declareRequiredParameter<std::string>(*this, "expected.layout");
  config_.qos_depth = declareRequiredParameter<int>(*this, "qos.depth");
  config_.qos_reliable = declareRequiredParameter<bool>(*this, "qos.reliable");
  config_.diagnostics_publish_period_ms =
    declareRequiredParameter<int>(*this, "diagnostics.publish_period_ms");

  if (config_.input_topic.empty()) {
    throw std::runtime_error("input_topic is required");
  }
  if (config_.output_topic.empty()) {
    throw std::runtime_error("output_topic is required");
  }
  if (config_.leading_frames < 0) {
    throw std::runtime_error("trim.leading_frames must be >= 0");
  }
  if (config_.trailing_frames < 0) {
    throw std::runtime_error("trim.trailing_frames must be >= 0");
  }
  if (config_.leading_frames == 0 && config_.trailing_frames == 0) {
    throw std::runtime_error(
      "at least one of trim.leading_frames or trim.trailing_frames must be > 0");
  }
  if (config_.expected_sample_rate <= 0) {
    throw std::runtime_error("expected.sample_rate must be > 0");
  }
  if (config_.expected_channels <= 0) {
    throw std::runtime_error("expected.channels must be > 0");
  }
  if (config_.expected_encoding != kEncodingFloat32) {
    throw std::runtime_error("fa_trim requires expected.encoding=FLOAT32LE");
  }
  if (config_.expected_bit_depth != 32) {
    throw std::runtime_error("fa_trim requires expected.bit_depth=32");
  }
  if (config_.expected_layout != kInterleavedLayout) {
    throw std::runtime_error("fa_trim requires expected.layout=interleaved");
  }
  if (config_.qos_depth <= 0) {
    throw std::runtime_error("qos.depth must be > 0");
  }
  if (config_.diagnostics_publish_period_ms <= 0) {
    throw std::runtime_error("diagnostics.publish_period_ms must be > 0");
  }

  RCLCPP_INFO(
    this->get_logger(),
    "Trim config: input=%s output=%s leading_frames=%d trailing_frames=%d "
    "expected=%dHz/%d/%s/%d/%s qos_depth=%d reliable=%s diag=%dms",
    config_.input_topic.c_str(),
    config_.output_topic.c_str(),
    config_.leading_frames,
    config_.trailing_frames,
    config_.expected_sample_rate,
    config_.expected_channels,
    config_.expected_encoding.c_str(),
    config_.expected_bit_depth,
    config_.expected_layout.c_str(),
    config_.qos_depth,
    config_.qos_reliable ? "true" : "false",
    config_.diagnostics_publish_period_ms);
}

void FaTrimNode::setupInterfaces()
{
  rclcpp::QoS qos(static_cast<size_t>(config_.qos_depth));
  if (config_.qos_reliable) {
    qos.reliable();
  } else {
    qos.best_effort();
  }

  audio_pub_ = this->create_publisher<fa_interfaces::msg::AudioFrame>(config_.output_topic, qos);
  audio_sub_ = this->create_subscription<fa_interfaces::msg::AudioFrame>(
    config_.input_topic,
    qos,
    std::bind(&FaTrimNode::handleFrame, this, std::placeholders::_1));

  diag_pub_ = this->create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
    "diagnostics",
    rclcpp::SystemDefaultsQoS());
  diag_timer_ = this->create_wall_timer(
    std::chrono::milliseconds(config_.diagnostics_publish_period_ms),
    std::bind(&FaTrimNode::publishDiagnostics, this));
}

void FaTrimNode::handleFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg)
{
  frames_in_.fetch_add(1);

  if (!msg) {
    contract_drops_.fetch_add(1);
    frames_dropped_.fetch_add(1);
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Received null AudioFrame pointer");
    return;
  }
  if (!validateFrame(*msg)) {
    frames_dropped_.fetch_add(1);
    return;
  }
  if (!validateSamples(*msg)) {
    frames_dropped_.fetch_add(1);
    return;
  }

  fa_interfaces::msg::AudioFrame out;
  if (!trimFrame(*msg, out)) {
    frames_dropped_.fetch_add(1);
    return;
  }

  audio_pub_->publish(out);
  frames_out_.fetch_add(1);
}

bool FaTrimNode::validateFrame(const fa_interfaces::msg::AudioFrame & msg)
{
  if (msg.source_id.empty() || msg.stream_id.empty()) {
    contract_drops_.fetch_add(1);
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "AudioFrame source_id and stream_id are required");
    return false;
  }
  if (msg.stream_id != config_.input_topic) {
    contract_drops_.fetch_add(1);
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "AudioFrame stream_id mismatch: %s != %s",
      msg.stream_id.c_str(),
      config_.input_topic.c_str());
    return false;
  }
  if (msg.layout != config_.expected_layout) {
    contract_drops_.fetch_add(1);
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
    contract_drops_.fetch_add(1);
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
    contract_drops_.fetch_add(1);
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
    contract_drops_.fetch_add(1);
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "AudioFrame data size is invalid for FLOAT32LE interleaved sample frames");
    return false;
  }

  return true;
}

bool FaTrimNode::validateSamples(const fa_interfaces::msg::AudioFrame & msg)
{
  const size_t sample_count = msg.data.size() / sizeof(float);
  for (size_t sample_index = 0; sample_index < sample_count; ++sample_index) {
    const float sample = readFloatSample(msg.data, sample_index);
    if (!std::isfinite(sample) ||
        sample < kMinNormalizedSample ||
        sample > kMaxNormalizedSample)
    {
      invalid_sample_drops_.fetch_add(1);
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 3000,
        "Dropping frame because input sample is outside normalized FLOAT32LE range");
      return false;
    }
  }
  return true;
}

bool FaTrimNode::trimFrame(
  const fa_interfaces::msg::AudioFrame & in,
  fa_interfaces::msg::AudioFrame & out)
{
  const size_t frame_count = in.data.size() / bytesPerFrame();
  last_input_frame_count_.store(static_cast<uint64_t>(frame_count));

  const size_t leading_frames = static_cast<size_t>(config_.leading_frames);
  const size_t trailing_frames = static_cast<size_t>(config_.trailing_frames);
  if (leading_frames >= frame_count || trailing_frames >= (frame_count - leading_frames)) {
    trim_exhausted_drops_.fetch_add(1);
    last_output_frame_count_.store(0);
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping frame because trim removes all sample frames: input=%zu leading=%zu trailing=%zu",
      frame_count,
      leading_frames,
      trailing_frames);
    return false;
  }

  if (in.epoch == std::numeric_limits<uint32_t>::max()) {
    epoch_overflow_drops_.fetch_add(1);
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping frame because epoch increment would wrap uint32");
    return false;
  }

  const size_t output_frame_count = frame_count - leading_frames - trailing_frames;
  const size_t start_byte = leading_frames * bytesPerFrame();
  const size_t byte_count = output_frame_count * bytesPerFrame();

  out = in;
  out.stream_id = config_.output_topic;
  out.epoch = in.epoch + 1U;
  out.data.assign(
    in.data.begin() + static_cast<std::ptrdiff_t>(start_byte),
    in.data.begin() + static_cast<std::ptrdiff_t>(start_byte + byte_count));

  last_output_frame_count_.store(static_cast<uint64_t>(output_frame_count));
  return true;
}

size_t FaTrimNode::bytesPerFrame() const
{
  return static_cast<size_t>(config_.expected_channels) * sizeof(float);
}

void FaTrimNode::publishDiagnostics()
{
  diagnostic_msgs::msg::DiagnosticArray array_msg;
  array_msg.header.stamp = this->now();

  diagnostic_msgs::msg::DiagnosticStatus status;
  status.name = "fa_trim";
  status.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
  status.message = "running";
  status.values.reserve(15);
  pushKeyValue(status, "input_topic", config_.input_topic);
  pushKeyValue(status, "output_topic", config_.output_topic);
  pushKeyValue(status, "leading_frames", std::to_string(config_.leading_frames));
  pushKeyValue(status, "trailing_frames", std::to_string(config_.trailing_frames));
  pushKeyValue(status, "expected_channels", std::to_string(config_.expected_channels));
  pushKeyValue(status, "last_input_frame_count", std::to_string(last_input_frame_count_.load()));
  pushKeyValue(status, "last_output_frame_count", std::to_string(last_output_frame_count_.load()));
  pushKeyValue(status, "frames_in", std::to_string(frames_in_.load()));
  pushKeyValue(status, "frames_out", std::to_string(frames_out_.load()));
  pushKeyValue(status, "frames_dropped", std::to_string(frames_dropped_.load()));
  pushKeyValue(status, "contract_drops", std::to_string(contract_drops_.load()));
  pushKeyValue(status, "invalid_sample_drops", std::to_string(invalid_sample_drops_.load()));
  pushKeyValue(status, "trim_exhausted_drops", std::to_string(trim_exhausted_drops_.load()));
  pushKeyValue(status, "epoch_overflow_drops", std::to_string(epoch_overflow_drops_.load()));

  array_msg.status.push_back(status);
  diag_pub_->publish(array_msg);
}

}  // namespace fa_trim

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<fa_trim::FaTrimNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return EXIT_SUCCESS;
  } catch (const std::exception & e) {
    RCLCPP_FATAL(rclcpp::get_logger("fa_trim"), "Exception: %s", e.what());
    rclcpp::shutdown();
    return EXIT_FAILURE;
  }
}

#include "fa_sidechain/fa_sidechain_node.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>

#include "diagnostic_msgs/msg/diagnostic_status.hpp"
#include "diagnostic_msgs/msg/key_value.hpp"

namespace fa_sidechain
{

namespace
{
constexpr const char * kEncodingFloat32 = "FLOAT32LE";
constexpr const char * kInterleavedLayout = "interleaved";
constexpr float kMinNormalizedSample = -1.0F;
constexpr float kMaxNormalizedSample = 1.0F;
constexpr double kMinControlGain = 0.0;
constexpr double kMaxControlGain = 4.0;

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
}  // namespace

FaSidechainNode::FaSidechainNode()
: rclcpp::Node("fa_sidechain")
{
  RCLCPP_INFO(this->get_logger(), "Starting FA Sidechain node");
  loadParameters();
  setupInterfaces();
}

void FaSidechainNode::loadParameters()
{
  this->declare_parameter("sidechain_topic", config_.sidechain_topic);
  this->declare_parameter("control_topic", config_.control_topic);
  this->declare_parameter<double>("detector.threshold_rms", config_.threshold_rms);
  this->declare_parameter<double>("detector.active_gain_db", config_.active_gain_db);
  this->declare_parameter<double>("detector.inactive_gain_db", config_.inactive_gain_db);
  this->declare_parameter<int>("control.sample_rate", config_.control_sample_rate);
  this->declare_parameter<int>("expected.sample_rate", config_.expected_sample_rate);
  this->declare_parameter<int>("expected.channels", config_.expected_channels);
  this->declare_parameter("expected.encoding", config_.expected_encoding);
  this->declare_parameter<int>("expected.bit_depth", config_.expected_bit_depth);
  this->declare_parameter("expected.layout", config_.expected_layout);
  this->declare_parameter<int>("qos.depth", config_.qos_depth);
  this->declare_parameter<bool>("qos.reliable", config_.qos_reliable);
  this->declare_parameter<int>(
    "diagnostics.publish_period_ms",
    config_.diagnostics_publish_period_ms);

  config_.sidechain_topic = this->get_parameter("sidechain_topic").as_string();
  config_.control_topic = this->get_parameter("control_topic").as_string();
  config_.threshold_rms = this->get_parameter("detector.threshold_rms").as_double();
  config_.active_gain_db = this->get_parameter("detector.active_gain_db").as_double();
  config_.inactive_gain_db = this->get_parameter("detector.inactive_gain_db").as_double();
  config_.control_sample_rate = this->get_parameter("control.sample_rate").as_int();
  config_.expected_sample_rate = this->get_parameter("expected.sample_rate").as_int();
  config_.expected_channels = this->get_parameter("expected.channels").as_int();
  config_.expected_encoding = this->get_parameter("expected.encoding").as_string();
  config_.expected_bit_depth = this->get_parameter("expected.bit_depth").as_int();
  config_.expected_layout = this->get_parameter("expected.layout").as_string();
  config_.qos_depth = this->get_parameter("qos.depth").as_int();
  config_.qos_reliable = this->get_parameter("qos.reliable").as_bool();
  config_.diagnostics_publish_period_ms =
    this->get_parameter("diagnostics.publish_period_ms").as_int();

  if (config_.sidechain_topic.empty()) {
    throw std::runtime_error("sidechain_topic is required");
  }
  if (config_.control_topic.empty()) {
    throw std::runtime_error("control_topic is required");
  }
  if (config_.control_topic == config_.sidechain_topic) {
    throw std::runtime_error("control_topic must differ from sidechain_topic");
  }
  if (!isFinite(config_.threshold_rms) || config_.threshold_rms <= 0.0 || config_.threshold_rms > 1.0) {
    throw std::runtime_error("detector.threshold_rms must be finite and in (0.0, 1.0]");
  }
  if (!isFinite(config_.active_gain_db)) {
    throw std::runtime_error("detector.active_gain_db must be finite");
  }
  if (!isFinite(config_.inactive_gain_db)) {
    throw std::runtime_error("detector.inactive_gain_db must be finite");
  }
  config_.active_gain_linear = dbToLinear(config_.active_gain_db);
  config_.inactive_gain_linear = dbToLinear(config_.inactive_gain_db);
  if (!isFinite(config_.active_gain_linear) ||
      config_.active_gain_linear < kMinControlGain ||
      config_.active_gain_linear > kMaxControlGain)
  {
    throw std::runtime_error("detector.active_gain_db must resolve to finite linear gain in [0.0, 4.0]");
  }
  if (!isFinite(config_.inactive_gain_linear) ||
      config_.inactive_gain_linear < kMinControlGain ||
      config_.inactive_gain_linear > kMaxControlGain)
  {
    throw std::runtime_error("detector.inactive_gain_db must resolve to finite linear gain in [0.0, 4.0]");
  }
  if (config_.control_sample_rate <= 0) {
    throw std::runtime_error("control.sample_rate must be > 0");
  }
  if (config_.expected_sample_rate <= 0) {
    throw std::runtime_error("expected.sample_rate must be > 0");
  }
  if (config_.expected_channels <= 0) {
    throw std::runtime_error("expected.channels must be > 0");
  }
  if (config_.expected_encoding != kEncodingFloat32) {
    throw std::runtime_error("fa_sidechain requires expected.encoding=FLOAT32LE");
  }
  if (config_.expected_bit_depth != 32) {
    throw std::runtime_error("fa_sidechain requires expected.bit_depth=32");
  }
  if (config_.expected_layout != kInterleavedLayout) {
    throw std::runtime_error("fa_sidechain requires expected.layout=interleaved");
  }
  if (config_.qos_depth <= 0) {
    throw std::runtime_error("qos.depth must be > 0");
  }
  if (config_.diagnostics_publish_period_ms <= 0) {
    throw std::runtime_error("diagnostics.publish_period_ms must be > 0");
  }

  last_gain_linear_.store(config_.inactive_gain_linear);

  RCLCPP_INFO(
    this->get_logger(),
    "Sidechain config: sidechain=%s control=%s threshold_rms=%f active_gain_db=%f active_gain=%f inactive_gain_db=%f inactive_gain=%f control_rate=%d expected=%dHz/%d/%s/%d/%s qos_depth=%d reliable=%s diag=%dms",
    config_.sidechain_topic.c_str(),
    config_.control_topic.c_str(),
    config_.threshold_rms,
    config_.active_gain_db,
    config_.active_gain_linear,
    config_.inactive_gain_db,
    config_.inactive_gain_linear,
    config_.control_sample_rate,
    config_.expected_sample_rate,
    config_.expected_channels,
    config_.expected_encoding.c_str(),
    config_.expected_bit_depth,
    config_.expected_layout.c_str(),
    config_.qos_depth,
    config_.qos_reliable ? "true" : "false",
    config_.diagnostics_publish_period_ms);
}

void FaSidechainNode::setupInterfaces()
{
  rclcpp::QoS qos(std::max<int>(1, config_.qos_depth));
  if (config_.qos_reliable) {
    qos.reliable();
  } else {
    qos.best_effort();
  }

  control_pub_ = this->create_publisher<fa_interfaces::msg::AudioFrame>(config_.control_topic, qos);
  sidechain_sub_ = this->create_subscription<fa_interfaces::msg::AudioFrame>(
    config_.sidechain_topic,
    qos,
    std::bind(&FaSidechainNode::handleSidechainFrame, this, std::placeholders::_1));

  diag_pub_ = this->create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
    "diagnostics",
    rclcpp::SystemDefaultsQoS());
  diag_timer_ = this->create_wall_timer(
    std::chrono::milliseconds(config_.diagnostics_publish_period_ms),
    std::bind(&FaSidechainNode::publishDiagnostics, this));
}

void FaSidechainNode::handleSidechainFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg)
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

  std::vector<float> samples;
  if (!readSamples(*msg, samples)) {
    frames_dropped_.fetch_add(1);
    return;
  }

  const double rms = calculateFrameRms(samples);
  if (!isFinite(rms) || rms < 0.0 || rms > 1.0) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping sidechain frame because RMS is outside normalized range");
    frames_dropped_.fetch_add(1);
    return;
  }

  fa_interfaces::msg::AudioFrame control_frame;
  if (!buildControlFrame(*msg, rms, control_frame)) {
    frames_dropped_.fetch_add(1);
    return;
  }

  control_pub_->publish(control_frame);
  frames_out_.fetch_add(1);
}

bool FaSidechainNode::validateFrame(const fa_interfaces::msg::AudioFrame & msg)
{
  if (msg.source_id.empty()) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping sidechain frame because source_id is required");
    return false;
  }
  if (msg.stream_id != config_.sidechain_topic) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping sidechain frame because stream_id mismatch: %s != %s",
      msg.stream_id.c_str(),
      config_.sidechain_topic.c_str());
    return false;
  }
  if (msg.sample_rate != static_cast<uint32_t>(config_.expected_sample_rate) ||
      msg.channels != static_cast<uint32_t>(config_.expected_channels))
  {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping sidechain frame because format mismatch: frame=%uHz/%u config=%dHz/%d",
      msg.sample_rate,
      msg.channels,
      config_.expected_sample_rate,
      config_.expected_channels);
    return false;
  }
  if (msg.encoding != config_.expected_encoding || msg.bit_depth != static_cast<uint32_t>(config_.expected_bit_depth)) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping sidechain frame because encoding mismatch: %s/%u != %s/%d",
      msg.encoding.c_str(),
      msg.bit_depth,
      config_.expected_encoding.c_str(),
      config_.expected_bit_depth);
    return false;
  }
  if (msg.layout != config_.expected_layout) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping sidechain frame because layout mismatch: %s != %s",
      msg.layout.c_str(),
      config_.expected_layout.c_str());
    return false;
  }

  const size_t bytes_per_frame = static_cast<size_t>(config_.expected_channels) * sizeof(float);
  if (msg.data.empty() || (msg.data.size() % bytes_per_frame) != 0) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping sidechain frame because data size is invalid for FLOAT32LE interleaved samples");
    return false;
  }
  return true;
}

bool FaSidechainNode::readSamples(const fa_interfaces::msg::AudioFrame & msg, std::vector<float> & samples)
{
  const size_t sample_count = msg.data.size() / sizeof(float);
  samples.resize(sample_count);

  for (size_t i = 0; i < sample_count; ++i) {
    float sample = 0.0F;
    std::memcpy(&sample, msg.data.data() + (i * sizeof(float)), sizeof(float));
    if (!std::isfinite(sample) || sample < kMinNormalizedSample || sample > kMaxNormalizedSample) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 3000,
        "Dropping sidechain frame because input sample is outside normalized FLOAT32LE range");
      return false;
    }
    samples[i] = sample;
  }

  return true;
}

double FaSidechainNode::calculateFrameRms(const std::vector<float> & samples) const
{
  double square_sum = 0.0;
  for (const float sample : samples) {
    const double value = static_cast<double>(sample);
    square_sum += value * value;
  }

  const double mean_square = square_sum / static_cast<double>(samples.size());
  return std::sqrt(mean_square);
}

double FaSidechainNode::targetGainForRms(double rms) const
{
  return rms >= config_.threshold_rms ? config_.active_gain_linear : config_.inactive_gain_linear;
}

bool FaSidechainNode::buildControlFrame(
  const fa_interfaces::msg::AudioFrame & input,
  double rms,
  fa_interfaces::msg::AudioFrame & output)
{
  const double gain_linear = targetGainForRms(rms);
  if (!isFinite(gain_linear) || gain_linear < kMinControlGain || gain_linear > kMaxControlGain) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping sidechain frame because derived control gain is outside [0.0, 4.0]");
    return false;
  }

  const float gain_sample = static_cast<float>(gain_linear);
  if (!std::isfinite(gain_sample) ||
      gain_sample < static_cast<float>(kMinControlGain) ||
      gain_sample > static_cast<float>(kMaxControlGain))
  {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping sidechain frame because control gain cannot be represented as FLOAT32LE");
    return false;
  }

  output.header = input.header;
  output.source_id = input.source_id;
  output.stream_id = config_.control_topic;
  output.sample_rate = static_cast<uint32_t>(config_.control_sample_rate);
  output.channels = 1;
  output.encoding = kEncodingFloat32;
  output.bit_depth = 32;
  output.layout = kInterleavedLayout;
  output.epoch = input.epoch;
  output.data.resize(sizeof(float));
  std::memcpy(output.data.data(), &gain_sample, sizeof(float));

  last_rms_.store(rms);
  last_gain_linear_.store(gain_linear);
  if (rms >= config_.threshold_rms) {
    active_frames_.fetch_add(1);
  } else {
    inactive_frames_.fetch_add(1);
  }

  return true;
}

void FaSidechainNode::publishDiagnostics()
{
  diagnostic_msgs::msg::DiagnosticArray array_msg;
  array_msg.header.stamp = this->now();

  diagnostic_msgs::msg::DiagnosticStatus status;
  status.name = "fa_sidechain";
  status.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
  status.message = "running";
  status.values.reserve(20);
  pushKeyValue(status, "sidechain_topic", config_.sidechain_topic);
  pushKeyValue(status, "control_topic", config_.control_topic);
  pushKeyValue(status, "threshold_rms", std::to_string(config_.threshold_rms));
  pushKeyValue(status, "active_gain_db", std::to_string(config_.active_gain_db));
  pushKeyValue(status, "active_gain_linear", std::to_string(config_.active_gain_linear));
  pushKeyValue(status, "inactive_gain_db", std::to_string(config_.inactive_gain_db));
  pushKeyValue(status, "inactive_gain_linear", std::to_string(config_.inactive_gain_linear));
  pushKeyValue(status, "control.sample_rate", std::to_string(config_.control_sample_rate));
  pushKeyValue(status, "expected.sample_rate", std::to_string(config_.expected_sample_rate));
  pushKeyValue(status, "expected.channels", std::to_string(config_.expected_channels));
  pushKeyValue(status, "expected.encoding", config_.expected_encoding);
  pushKeyValue(status, "expected.bit_depth", std::to_string(config_.expected_bit_depth));
  pushKeyValue(status, "expected.layout", config_.expected_layout);
  pushKeyValue(status, "frames_in", std::to_string(frames_in_.load()));
  pushKeyValue(status, "frames_out", std::to_string(frames_out_.load()));
  pushKeyValue(status, "frames_dropped", std::to_string(frames_dropped_.load()));
  pushKeyValue(status, "last_rms", std::to_string(last_rms_.load()));
  pushKeyValue(status, "last_gain_linear", std::to_string(last_gain_linear_.load()));
  pushKeyValue(status, "active_frames", std::to_string(active_frames_.load()));
  pushKeyValue(status, "inactive_frames", std::to_string(inactive_frames_.load()));

  array_msg.status.push_back(status);
  diag_pub_->publish(array_msg);
}

}  // namespace fa_sidechain

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<fa_sidechain::FaSidechainNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return EXIT_SUCCESS;
  } catch (const std::exception & e) {
    RCLCPP_FATAL(rclcpp::get_logger("fa_sidechain"), "Exception: %s", e.what());
    rclcpp::shutdown();
    return EXIT_FAILURE;
  }
}

#include "fa_eq/fa_eq_node.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>

#include "diagnostic_msgs/msg/diagnostic_status.hpp"
#include "diagnostic_msgs/msg/key_value.hpp"
#include "fa_eq/backends/internal_three_band_eq.hpp"

namespace fa_eq
{

namespace
{
constexpr const char * kEncodingFloat32 = "FLOAT32LE";
constexpr const char * kInterleavedLayout = "interleaved";

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
}  // namespace

FaEqNode::FaEqNode()
: rclcpp::Node("fa_eq")
{
  RCLCPP_INFO(this->get_logger(), "Starting FA EQ node");
  loadParameters();
  configureBackend();
  setupInterfaces();
}

FaEqNode::~FaEqNode() = default;

void FaEqNode::loadParameters()
{
  this->declare_parameter("input_topic", config_.input_topic);
  this->declare_parameter("output_topic", config_.output_topic);
  this->declare_parameter<double>("low.cutoff_hz", config_.low_cutoff_hz);
  this->declare_parameter<double>("high.cutoff_hz", config_.high_cutoff_hz);
  this->declare_parameter<double>("gains.low_db", config_.gain_low_db);
  this->declare_parameter<double>("gains.mid_db", config_.gain_mid_db);
  this->declare_parameter<double>("gains.high_db", config_.gain_high_db);
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

  config_.input_topic = this->get_parameter("input_topic").as_string();
  config_.output_topic = this->get_parameter("output_topic").as_string();
  config_.low_cutoff_hz = this->get_parameter("low.cutoff_hz").as_double();
  config_.high_cutoff_hz = this->get_parameter("high.cutoff_hz").as_double();
  config_.gain_low_db = this->get_parameter("gains.low_db").as_double();
  config_.gain_mid_db = this->get_parameter("gains.mid_db").as_double();
  config_.gain_high_db = this->get_parameter("gains.high_db").as_double();
  config_.expected_sample_rate = this->get_parameter("expected.sample_rate").as_int();
  config_.expected_channels = this->get_parameter("expected.channels").as_int();
  config_.expected_encoding = this->get_parameter("expected.encoding").as_string();
  config_.expected_bit_depth = this->get_parameter("expected.bit_depth").as_int();
  config_.expected_layout = this->get_parameter("expected.layout").as_string();
  config_.qos_depth = this->get_parameter("qos.depth").as_int();
  config_.qos_reliable = this->get_parameter("qos.reliable").as_bool();
  config_.diagnostics_publish_period_ms =
    this->get_parameter("diagnostics.publish_period_ms").as_int();

  if (config_.input_topic.empty()) {
    throw std::runtime_error("input_topic is required");
  }
  if (config_.output_topic.empty()) {
    throw std::runtime_error("output_topic is required");
  }
  if (config_.expected_sample_rate <= 0) {
    throw std::runtime_error("expected.sample_rate must be > 0");
  }
  const double nyquist_hz = static_cast<double>(config_.expected_sample_rate) / 2.0;
  if (!isFinite(config_.low_cutoff_hz) || config_.low_cutoff_hz <= 0.0) {
    throw std::runtime_error("low.cutoff_hz must be finite and > 0.0");
  }
  if (!isFinite(config_.high_cutoff_hz) ||
      config_.high_cutoff_hz <= config_.low_cutoff_hz ||
      config_.high_cutoff_hz >= nyquist_hz)
  {
    throw std::runtime_error(
      "high.cutoff_hz must be finite, > low.cutoff_hz, and < expected.sample_rate / 2.0");
  }
  if (config_.expected_channels <= 0) {
    throw std::runtime_error("expected.channels must be > 0");
  }
  if (config_.expected_encoding != kEncodingFloat32) {
    throw std::runtime_error("fa_eq requires expected.encoding=FLOAT32LE");
  }
  if (config_.expected_bit_depth != 32) {
    throw std::runtime_error("fa_eq requires expected.bit_depth=32");
  }
  if (config_.expected_layout != kInterleavedLayout) {
    throw std::runtime_error("fa_eq requires expected.layout=interleaved");
  }
  if (config_.qos_depth <= 0) {
    throw std::runtime_error("qos.depth must be > 0");
  }
  if (config_.diagnostics_publish_period_ms <= 0) {
    throw std::runtime_error("diagnostics.publish_period_ms must be > 0");
  }
  if (!isFinite(config_.gain_low_db) ||
      !isFinite(config_.gain_mid_db) ||
      !isFinite(config_.gain_high_db))
  {
    throw std::runtime_error("gains.*_db must be finite");
  }

  RCLCPP_INFO(
    this->get_logger(),
    "EQ config: input=%s output=%s low_cutoff=%fHz high_cutoff=%fHz gains_db=%f/%f/%f expected=%dHz/%d/%s/%d qos_depth=%d reliable=%s diag=%dms",
    config_.input_topic.c_str(),
    config_.output_topic.c_str(),
    config_.low_cutoff_hz,
    config_.high_cutoff_hz,
    config_.gain_low_db,
    config_.gain_mid_db,
    config_.gain_high_db,
    config_.expected_sample_rate,
    config_.expected_channels,
    config_.expected_encoding.c_str(),
    config_.expected_bit_depth,
    config_.qos_depth,
    config_.qos_reliable ? "true" : "false",
    config_.diagnostics_publish_period_ms);
}

void FaEqNode::configureBackend()
{
  backend_ = std::make_unique<backends::InternalThreeBandEqBackend>(
    backends::InternalThreeBandEqConfig{
      config_.expected_sample_rate,
      config_.expected_channels,
      config_.low_cutoff_hz,
      config_.high_cutoff_hz,
      config_.gain_low_db,
      config_.gain_mid_db,
      config_.gain_high_db});
}

void FaEqNode::setupInterfaces()
{
  rclcpp::QoS qos(std::max<int>(1, config_.qos_depth));
  if (config_.qos_reliable) {
    qos.reliable();
  } else {
    qos.best_effort();
  }

  audio_pub_ = this->create_publisher<fa_interfaces::msg::AudioFrame>(config_.output_topic, qos);
  audio_sub_ = this->create_subscription<fa_interfaces::msg::AudioFrame>(
    config_.input_topic,
    qos,
    std::bind(&FaEqNode::handleFrame, this, std::placeholders::_1));

  diag_pub_ = this->create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
    "diagnostics",
    rclcpp::SystemDefaultsQoS());
  diag_timer_ = this->create_wall_timer(
    std::chrono::milliseconds(config_.diagnostics_publish_period_ms),
    std::bind(&FaEqNode::publishDiagnostics, this));
}

void FaEqNode::handleFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg)
{
  frames_in_.fetch_add(1);

  if (!validateFrame(*msg)) {
    frames_dropped_.fetch_add(1);
    return;
  }

  fa_interfaces::msg::AudioFrame out;
  if (!applyEq(*msg, out)) {
    frames_dropped_.fetch_add(1);
    return;
  }

  audio_pub_->publish(out);
  frames_out_.fetch_add(1);
}

bool FaEqNode::validateFrame(const fa_interfaces::msg::AudioFrame & msg)
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
  const size_t bytes_per_frame = static_cast<size_t>(config_.expected_channels) * sizeof(float);
  if (msg.data.empty() || (msg.data.size() % bytes_per_frame) != 0) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "AudioFrame data size is invalid for FLOAT32LE interleaved samples");
    return false;
  }
  return true;
}

bool FaEqNode::applyEq(
  const fa_interfaces::msg::AudioFrame & in,
  fa_interfaces::msg::AudioFrame & out)
{
  const bool source_changed = !active_source_id_.empty() && in.source_id != active_source_id_;

  out = in;
  out.stream_id = config_.output_topic;
  const backends::ProcessStatus status = backend_->process(in.data, out.data, source_changed);
  if (status != backends::ProcessStatus::kOk) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping frame because EQ backend rejected input or output: %s",
      backends::processStatusMessage(status));
    return false;
  }

  active_source_id_ = in.source_id;
  if (source_changed) {
    source_resets_.fetch_add(1);
  }
  return true;
}

void FaEqNode::publishDiagnostics()
{
  diagnostic_msgs::msg::DiagnosticArray array_msg;
  array_msg.header.stamp = this->now();

  diagnostic_msgs::msg::DiagnosticStatus status;
  status.name = "fa_eq";
  status.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
  status.message = "running";
  status.values.reserve(15);
  pushKeyValue(status, "low_cutoff_hz", std::to_string(config_.low_cutoff_hz));
  pushKeyValue(status, "high_cutoff_hz", std::to_string(config_.high_cutoff_hz));
  pushKeyValue(status, "low_alpha", std::to_string(backend_->lowAlpha()));
  pushKeyValue(status, "high_alpha", std::to_string(backend_->highAlpha()));
  pushKeyValue(status, "gain_low_db", std::to_string(config_.gain_low_db));
  pushKeyValue(status, "gain_mid_db", std::to_string(config_.gain_mid_db));
  pushKeyValue(status, "gain_high_db", std::to_string(config_.gain_high_db));
  pushKeyValue(status, "gain_low_linear", std::to_string(backend_->gainLowLinear()));
  pushKeyValue(status, "gain_mid_linear", std::to_string(backend_->gainMidLinear()));
  pushKeyValue(status, "gain_high_linear", std::to_string(backend_->gainHighLinear()));
  pushKeyValue(status, "state_source_id", active_source_id_);
  pushKeyValue(status, "source_resets", std::to_string(source_resets_.load()));
  pushKeyValue(status, "frames_in", std::to_string(frames_in_.load()));
  pushKeyValue(status, "frames_out", std::to_string(frames_out_.load()));
  pushKeyValue(status, "frames_dropped", std::to_string(frames_dropped_.load()));
  pushKeyValue(status, "output_topic", config_.output_topic);

  array_msg.status.push_back(status);
  diag_pub_->publish(array_msg);
}

}  // namespace fa_eq

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<fa_eq::FaEqNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return EXIT_SUCCESS;
  } catch (const std::exception & e) {
    RCLCPP_FATAL(rclcpp::get_logger("fa_eq"), "Exception: %s", e.what());
    rclcpp::shutdown();
    return EXIT_FAILURE;
  }
}

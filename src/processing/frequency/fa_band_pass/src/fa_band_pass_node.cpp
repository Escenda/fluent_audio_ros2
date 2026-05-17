#include "fa_band_pass/fa_band_pass_node.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "diagnostic_msgs/msg/diagnostic_status.hpp"
#include "diagnostic_msgs/msg/key_value.hpp"

namespace fa_band_pass
{

namespace
{
constexpr const char * kEncodingFloat32 = "FLOAT32LE";
constexpr const char * kInterleavedLayout = "interleaved";
constexpr double kPi = 3.14159265358979323846;
constexpr float kNormalizedMin = -1.0F;
constexpr float kNormalizedMax = 1.0F;

bool isFinite(double value)
{
  return std::isfinite(value);
}

bool isNormalizedSample(float value)
{
  return std::isfinite(value) && value >= kNormalizedMin && value <= kNormalizedMax;
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

FaBandPassNode::FaBandPassNode()
: rclcpp::Node("fa_band_pass")
{
  RCLCPP_INFO(this->get_logger(), "Starting FA Band Pass node");
  loadParameters();
  configureFilterState();
  setupInterfaces();
}

void FaBandPassNode::loadParameters()
{
  this->declare_parameter("input_topic", config_.input_topic);
  this->declare_parameter("output_topic", config_.output_topic);
  this->declare_parameter<double>("filter.low_cut_hz", config_.low_cut_hz);
  this->declare_parameter<double>("filter.high_cut_hz", config_.high_cut_hz);
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
  config_.low_cut_hz = this->get_parameter("filter.low_cut_hz").as_double();
  config_.high_cut_hz = this->get_parameter("filter.high_cut_hz").as_double();
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
  if (!isFinite(config_.low_cut_hz) || config_.low_cut_hz <= 0.0) {
    throw std::runtime_error("filter.low_cut_hz must be finite and > 0.0");
  }
  if (!isFinite(config_.high_cut_hz) ||
      config_.high_cut_hz <= config_.low_cut_hz ||
      config_.high_cut_hz >= nyquist_hz)
  {
    throw std::runtime_error(
      "filter.high_cut_hz must be finite, > filter.low_cut_hz, and < expected.sample_rate / 2.0");
  }
  if (config_.expected_channels <= 0) {
    throw std::runtime_error("expected.channels must be > 0");
  }
  if (config_.expected_encoding != kEncodingFloat32) {
    throw std::runtime_error("fa_band_pass requires expected.encoding=FLOAT32LE");
  }
  if (config_.expected_bit_depth != 32) {
    throw std::runtime_error("fa_band_pass requires expected.bit_depth=32");
  }
  if (config_.expected_layout != kInterleavedLayout) {
    throw std::runtime_error("fa_band_pass requires expected.layout=interleaved");
  }
  if (config_.qos_depth <= 0) {
    throw std::runtime_error("qos.depth must be > 0");
  }
  if (config_.diagnostics_publish_period_ms <= 0) {
    throw std::runtime_error("diagnostics.publish_period_ms must be > 0");
  }

  const double dt = 1.0 / static_cast<double>(config_.expected_sample_rate);
  const double rc_hp = 1.0 / (2.0 * kPi * config_.low_cut_hz);
  const double rc_lp = 1.0 / (2.0 * kPi * config_.high_cut_hz);
  hp_alpha_ = rc_hp / (rc_hp + dt);
  lp_alpha_ = dt / (rc_lp + dt);
  if (!isFinite(hp_alpha_) || hp_alpha_ <= 0.0 || hp_alpha_ >= 1.0) {
    throw std::runtime_error("high-pass coefficient alpha must be finite and in (0.0, 1.0)");
  }
  if (!isFinite(lp_alpha_) || lp_alpha_ <= 0.0 || lp_alpha_ >= 1.0) {
    throw std::runtime_error("low-pass coefficient alpha must be finite and in (0.0, 1.0)");
  }

  RCLCPP_INFO(
    this->get_logger(),
    "Band-pass config: input=%s output=%s low_cut=%fHz high_cut=%fHz hp_alpha=%f lp_alpha=%f expected=%dHz/%d/%s/%d qos_depth=%d reliable=%s diag=%dms",
    config_.input_topic.c_str(),
    config_.output_topic.c_str(),
    config_.low_cut_hz,
    config_.high_cut_hz,
    hp_alpha_,
    lp_alpha_,
    config_.expected_sample_rate,
    config_.expected_channels,
    config_.expected_encoding.c_str(),
    config_.expected_bit_depth,
    config_.qos_depth,
    config_.qos_reliable ? "true" : "false",
    config_.diagnostics_publish_period_ms);
}

void FaBandPassNode::configureFilterState()
{
  channel_states_.assign(static_cast<size_t>(config_.expected_channels), ChannelFilterState{});
}

void FaBandPassNode::setupInterfaces()
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
    std::bind(&FaBandPassNode::handleFrame, this, std::placeholders::_1));

  diag_pub_ = this->create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
    "diagnostics",
    rclcpp::SystemDefaultsQoS());
  diag_timer_ = this->create_wall_timer(
    std::chrono::milliseconds(config_.diagnostics_publish_period_ms),
    std::bind(&FaBandPassNode::publishDiagnostics, this));
}

void FaBandPassNode::handleFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg)
{
  frames_in_.fetch_add(1);

  if (!validateFrame(*msg)) {
    frames_dropped_.fetch_add(1);
    return;
  }

  fa_interfaces::msg::AudioFrame out;
  if (!applyBandPass(*msg, out)) {
    frames_dropped_.fetch_add(1);
    return;
  }

  audio_pub_->publish(out);
  frames_out_.fetch_add(1);
}

bool FaBandPassNode::validateFrame(const fa_interfaces::msg::AudioFrame & msg)
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

bool FaBandPassNode::applyBandPass(
  const fa_interfaces::msg::AudioFrame & in,
  fa_interfaces::msg::AudioFrame & out)
{
  std::vector<ChannelFilterState> next_states = channel_states_;
  bool source_changed = false;
  if (active_source_id_.empty() || in.source_id != active_source_id_) {
    source_changed = !active_source_id_.empty();
    next_states.assign(static_cast<size_t>(config_.expected_channels), ChannelFilterState{});
  }

  out = in;
  out.stream_id = config_.output_topic;
  out.data.resize(in.data.size());

  const size_t channel_count = static_cast<size_t>(config_.expected_channels);
  const size_t sample_count = in.data.size() / sizeof(float);
  for (size_t i = 0; i < sample_count; ++i) {
    float sample = 0.0F;
    std::memcpy(&sample, in.data.data() + (i * sizeof(float)), sizeof(float));
    if (!isNormalizedSample(sample)) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 3000,
        "Dropping frame because input sample is not finite normalized FLOAT32LE");
      return false;
    }

    ChannelFilterState & state = next_states.at(i % channel_count);
    float hp_sample = 0.0F;
    float out_sample = 0.0F;
    if (state.initialized) {
      const double high_passed =
        hp_alpha_ * (static_cast<double>(state.previous_hp_output) +
        static_cast<double>(sample) - static_cast<double>(state.previous_hp_input));
      if (!isFinite(high_passed)) {
        RCLCPP_WARN_THROTTLE(
          this->get_logger(), *this->get_clock(), 3000,
          "Dropping frame because high-pass output is not finite");
        return false;
      }
      hp_sample = static_cast<float>(high_passed);

      const double low_passed =
        static_cast<double>(state.previous_lp_output) +
        lp_alpha_ * (static_cast<double>(hp_sample) - static_cast<double>(state.previous_lp_output));
      if (!isFinite(low_passed)) {
        RCLCPP_WARN_THROTTLE(
          this->get_logger(), *this->get_clock(), 3000,
          "Dropping frame because low-pass output is not finite");
        return false;
      }
      out_sample = static_cast<float>(low_passed);
    }

    if (!isNormalizedSample(out_sample)) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 3000,
        "Dropping frame because band-pass output is outside normalized FLOAT32LE range");
      return false;
    }

    state.previous_hp_input = sample;
    state.previous_hp_output = hp_sample;
    state.previous_lp_output = out_sample;
    state.initialized = true;
    std::memcpy(out.data.data() + (i * sizeof(float)), &out_sample, sizeof(float));
  }

  active_source_id_ = in.source_id;
  channel_states_ = next_states;
  if (source_changed) {
    source_resets_.fetch_add(1);
  }
  return true;
}

void FaBandPassNode::publishDiagnostics()
{
  diagnostic_msgs::msg::DiagnosticArray array_msg;
  array_msg.header.stamp = this->now();

  diagnostic_msgs::msg::DiagnosticStatus status;
  status.name = "fa_band_pass";
  status.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
  status.message = "running";
  status.values.reserve(10);
  pushKeyValue(status, "filter_low_cut_hz", std::to_string(config_.low_cut_hz));
  pushKeyValue(status, "filter_high_cut_hz", std::to_string(config_.high_cut_hz));
  pushKeyValue(status, "hp_alpha", std::to_string(hp_alpha_));
  pushKeyValue(status, "lp_alpha", std::to_string(lp_alpha_));
  pushKeyValue(status, "state_source_id", active_source_id_);
  pushKeyValue(status, "source_resets", std::to_string(source_resets_.load()));
  pushKeyValue(status, "frames_in", std::to_string(frames_in_.load()));
  pushKeyValue(status, "frames_out", std::to_string(frames_out_.load()));
  pushKeyValue(status, "frames_dropped", std::to_string(frames_dropped_.load()));
  pushKeyValue(status, "output_topic", config_.output_topic);

  array_msg.status.push_back(status);
  diag_pub_->publish(array_msg);
}

}  // namespace fa_band_pass

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<fa_band_pass::FaBandPassNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return EXIT_SUCCESS;
  } catch (const std::exception & e) {
    RCLCPP_FATAL(rclcpp::get_logger("fa_band_pass"), "Exception: %s", e.what());
    rclcpp::shutdown();
    return EXIT_FAILURE;
  }
}

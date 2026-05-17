#include "fa_hum/fa_hum_node.hpp"

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

namespace fa_hum
{

namespace
{
constexpr const char * kEncodingFloat32 = "FLOAT32LE";
constexpr const char * kInterleavedLayout = "interleaved";
constexpr double kPi = 3.14159265358979323846;
constexpr double kNormalizedMin = -1.0;
constexpr double kNormalizedMax = 1.0;

bool isFinite(double value)
{
  return std::isfinite(value);
}

bool isNormalized(double value)
{
  return value >= kNormalizedMin && value <= kNormalizedMax;
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

FaHumNode::FaHumNode()
: rclcpp::Node("fa_hum")
{
  RCLCPP_INFO(this->get_logger(), "Starting FA Hum Removal node");
  loadParameters();
  configureCascade();
  setupInterfaces();
}

void FaHumNode::loadParameters()
{
  this->declare_parameter("input_topic", config_.input_topic);
  this->declare_parameter("output_topic", config_.output_topic);
  this->declare_parameter<double>("hum.frequency_hz", config_.frequency_hz);
  this->declare_parameter<int>("hum.harmonics", config_.harmonics);
  this->declare_parameter<double>("hum.q", config_.q);
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
  config_.frequency_hz = this->get_parameter("hum.frequency_hz").as_double();
  config_.harmonics = this->get_parameter("hum.harmonics").as_int();
  config_.q = this->get_parameter("hum.q").as_double();
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
  if (!isFinite(config_.frequency_hz) || config_.frequency_hz <= 0.0) {
    throw std::runtime_error("hum.frequency_hz must be finite and > 0.0");
  }
  if (config_.frequency_hz >= nyquist_hz) {
    throw std::runtime_error("hum.frequency_hz must produce at least one harmonic below Nyquist");
  }
  if (config_.harmonics < 1) {
    throw std::runtime_error("hum.harmonics must be >= 1");
  }
  if (!isFinite(config_.q) || config_.q <= 0.0) {
    throw std::runtime_error("hum.q must be finite and > 0.0");
  }
  if (config_.expected_channels <= 0) {
    throw std::runtime_error("expected.channels must be > 0");
  }
  if (config_.expected_encoding != kEncodingFloat32) {
    throw std::runtime_error("fa_hum requires expected.encoding=FLOAT32LE");
  }
  if (config_.expected_bit_depth != 32) {
    throw std::runtime_error("fa_hum requires expected.bit_depth=32");
  }
  if (config_.expected_layout != kInterleavedLayout) {
    throw std::runtime_error("fa_hum requires expected.layout=interleaved");
  }
  if (config_.qos_depth <= 0) {
    throw std::runtime_error("qos.depth must be > 0");
  }
  if (config_.diagnostics_publish_period_ms <= 0) {
    throw std::runtime_error("diagnostics.publish_period_ms must be > 0");
  }

  RCLCPP_INFO(
    this->get_logger(),
    "Hum config: input=%s output=%s frequency=%fHz harmonics=%d q=%f "
    "expected=%dHz/%d/%s/%d/%s qos_depth=%d reliable=%s diag=%dms",
    config_.input_topic.c_str(),
    config_.output_topic.c_str(),
    config_.frequency_hz,
    config_.harmonics,
    config_.q,
    config_.expected_sample_rate,
    config_.expected_channels,
    config_.expected_encoding.c_str(),
    config_.expected_bit_depth,
    config_.expected_layout.c_str(),
    config_.qos_depth,
    config_.qos_reliable ? "true" : "false",
    config_.diagnostics_publish_period_ms);
}

void FaHumNode::configureCascade()
{
  cascade_coefficients_.clear();
  const double nyquist_hz = static_cast<double>(config_.expected_sample_rate) / 2.0;

  for (int harmonic = 1; harmonic <= config_.harmonics; ++harmonic) {
    const double center_hz = config_.frequency_hz * static_cast<double>(harmonic);
    if (center_hz >= nyquist_hz) {
      break;
    }

    const double omega = 2.0 * kPi * center_hz / static_cast<double>(config_.expected_sample_rate);
    const double alpha = std::sin(omega) / (2.0 * config_.q);
    const double cos_omega = std::cos(omega);
    const double a0 = 1.0 + alpha;
    if (!isFinite(center_hz) || !isFinite(a0) || a0 == 0.0) {
      throw std::runtime_error("hum notch coefficient normalization failed because a0 is invalid");
    }

    BiquadCoefficients coefficients;
    coefficients.center_hz = center_hz;
    coefficients.b0 = 1.0 / a0;
    coefficients.b1 = (-2.0 * cos_omega) / a0;
    coefficients.b2 = 1.0 / a0;
    coefficients.a1 = (-2.0 * cos_omega) / a0;
    coefficients.a2 = (1.0 - alpha) / a0;

    if (!isFinite(coefficients.b0) || !isFinite(coefficients.b1) || !isFinite(coefficients.b2) ||
        !isFinite(coefficients.a1) || !isFinite(coefficients.a2))
    {
      throw std::runtime_error("hum notch coefficient normalization produced non-finite coefficients");
    }
    cascade_coefficients_.push_back(coefficients);
  }

  if (cascade_coefficients_.empty()) {
    throw std::runtime_error("hum configuration produced no notch stages below Nyquist");
  }

  channel_states_.assign(
    static_cast<size_t>(config_.expected_channels),
    ChannelCascadeState(cascade_coefficients_.size(), BiquadState{}));
}

void FaHumNode::setupInterfaces()
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
    std::bind(&FaHumNode::handleFrame, this, std::placeholders::_1));

  diag_pub_ = this->create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
    "diagnostics",
    rclcpp::SystemDefaultsQoS());
  diag_timer_ = this->create_wall_timer(
    std::chrono::milliseconds(config_.diagnostics_publish_period_ms),
    std::bind(&FaHumNode::publishDiagnostics, this));
}

void FaHumNode::handleFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg)
{
  frames_in_.fetch_add(1);

  if (!msg) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping null AudioFrame pointer");
    frames_dropped_.fetch_add(1);
    return;
  }

  if (!validateFrame(*msg)) {
    frames_dropped_.fetch_add(1);
    return;
  }

  if (active_source_id_ != msg->source_id) {
    resetFilterStateForSource(msg->source_id);
  }

  fa_interfaces::msg::AudioFrame out;
  if (!applyHumRemoval(*msg, out)) {
    frames_dropped_.fetch_add(1);
    return;
  }

  audio_pub_->publish(out);
  frames_out_.fetch_add(1);
}

void FaHumNode::resetFilterStateForSource(const std::string & source_id)
{
  if (!active_source_id_.empty()) {
    RCLCPP_INFO(
      this->get_logger(),
      "Resetting hum filter state because source_id changed: %s -> %s",
      active_source_id_.c_str(),
      source_id.c_str());
    source_resets_.fetch_add(1);
  }
  active_source_id_ = source_id;
  channel_states_.assign(
    static_cast<size_t>(config_.expected_channels),
    ChannelCascadeState(cascade_coefficients_.size(), BiquadState{}));
}

bool FaHumNode::validateFrame(const fa_interfaces::msg::AudioFrame & msg)
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
  if (msg.encoding != config_.expected_encoding || msg.bit_depth != static_cast<uint32_t>(config_.expected_bit_depth)) {
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

bool FaHumNode::applyHumRemoval(
  const fa_interfaces::msg::AudioFrame & in,
  fa_interfaces::msg::AudioFrame & out)
{
  std::vector<ChannelCascadeState> next_states = channel_states_;

  out = in;
  out.stream_id = config_.output_topic;
  out.data.resize(in.data.size());

  const size_t channel_count = static_cast<size_t>(config_.expected_channels);
  const size_t sample_count = in.data.size() / sizeof(float);
  for (size_t i = 0; i < sample_count; ++i) {
    float sample = 0.0F;
    std::memcpy(&sample, in.data.data() + (i * sizeof(float)), sizeof(float));
    if (!std::isfinite(sample) || !isNormalized(static_cast<double>(sample))) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 3000,
        "Dropping frame because input sample is not finite or outside normalized FLOAT32LE range [-1, 1]");
      return false;
    }

    double filtered = static_cast<double>(sample);
    ChannelCascadeState & channel_state = next_states.at(i % channel_count);
    for (size_t stage = 0; stage < cascade_coefficients_.size(); ++stage) {
      const BiquadCoefficients & coefficients = cascade_coefficients_.at(stage);
      BiquadState & state = channel_state.at(stage);
      const double stage_output =
        coefficients.b0 * filtered +
        coefficients.b1 * state.previous_input_1 +
        coefficients.b2 * state.previous_input_2 -
        coefficients.a1 * state.previous_output_1 -
        coefficients.a2 * state.previous_output_2;
      if (!isFinite(stage_output)) {
        RCLCPP_WARN_THROTTLE(
          this->get_logger(), *this->get_clock(), 3000,
          "Dropping frame because hum notch output is not finite");
        return false;
      }

      state.previous_input_2 = state.previous_input_1;
      state.previous_input_1 = filtered;
      state.previous_output_2 = state.previous_output_1;
      state.previous_output_1 = stage_output;
      filtered = stage_output;
    }

    if (!isNormalized(filtered)) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 3000,
        "Dropping frame because hum removal output is outside normalized FLOAT32LE range [-1, 1]");
      return false;
    }

    const float out_sample = static_cast<float>(filtered);
    if (!std::isfinite(out_sample) || !isNormalized(static_cast<double>(out_sample))) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 3000,
        "Dropping frame because hum removal output cannot be represented as normalized FLOAT32LE");
      return false;
    }
    std::memcpy(out.data.data() + (i * sizeof(float)), &out_sample, sizeof(float));
  }

  channel_states_ = next_states;
  return true;
}

void FaHumNode::publishDiagnostics()
{
  diagnostic_msgs::msg::DiagnosticArray array_msg;
  array_msg.header.stamp = this->now();

  diagnostic_msgs::msg::DiagnosticStatus status;
  status.name = "fa_hum";
  status.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
  status.message = "running";
  status.values.reserve(12 + cascade_coefficients_.size());
  pushKeyValue(status, "hum_frequency_hz", std::to_string(config_.frequency_hz));
  pushKeyValue(status, "hum_harmonics_requested", std::to_string(config_.harmonics));
  pushKeyValue(status, "hum_q", std::to_string(config_.q));
  pushKeyValue(status, "notch_stage_count", std::to_string(cascade_coefficients_.size()));
  for (size_t stage = 0; stage < cascade_coefficients_.size(); ++stage) {
    pushKeyValue(
      status,
      "notch_stage_" + std::to_string(stage + 1) + "_center_hz",
      std::to_string(cascade_coefficients_.at(stage).center_hz));
  }
  pushKeyValue(status, "frames_in", std::to_string(frames_in_.load()));
  pushKeyValue(status, "frames_out", std::to_string(frames_out_.load()));
  pushKeyValue(status, "frames_dropped", std::to_string(frames_dropped_.load()));
  pushKeyValue(status, "source_resets", std::to_string(source_resets_.load()));
  pushKeyValue(status, "active_source_id", active_source_id_);
  pushKeyValue(status, "output_topic", config_.output_topic);
  pushKeyValue(status, "backend.name", "internal_notch_cascade");

  array_msg.status.push_back(status);
  diag_pub_->publish(array_msg);
}

}  // namespace fa_hum

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<fa_hum::FaHumNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return EXIT_SUCCESS;
  } catch (const std::exception & e) {
    RCLCPP_FATAL(rclcpp::get_logger("fa_hum"), "Exception: %s", e.what());
    rclcpp::shutdown();
    return EXIT_FAILURE;
  }
}

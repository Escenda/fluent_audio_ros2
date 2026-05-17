#include "fa_deesser/fa_deesser_node.hpp"

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

namespace fa_deesser
{

namespace
{
constexpr const char * kEncodingFloat32 = "FLOAT32LE";
constexpr const char * kInterleavedLayout = "interleaved";
constexpr float kMinNormalizedSample = -1.0F;
constexpr float kMaxNormalizedSample = 1.0F;
constexpr double kPi = 3.14159265358979323846;
constexpr double kMinimumAttenuationDb = -120.0;

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

FaDeesserNode::FaDeesserNode()
: rclcpp::Node("fa_deesser")
{
  RCLCPP_INFO(this->get_logger(), "Starting FA De-esser node");
  loadParameters();
  low_band_state_.assign(static_cast<size_t>(config_.expected_channels), 0.0);
  setupInterfaces();
}

void FaDeesserNode::loadParameters()
{
  this->declare_parameter("input_topic", config_.input_topic);
  this->declare_parameter("output_topic", config_.output_topic);
  this->declare_parameter<double>("detector.cutoff_hz", config_.cutoff_hz);
  this->declare_parameter<double>("detector.threshold", config_.threshold);
  this->declare_parameter<double>("detector.attenuation_db", config_.attenuation_db);
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
  config_.cutoff_hz = this->get_parameter("detector.cutoff_hz").as_double();
  config_.threshold = this->get_parameter("detector.threshold").as_double();
  config_.attenuation_db = this->get_parameter("detector.attenuation_db").as_double();
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
  const double nyquist_hz = static_cast<double>(config_.expected_sample_rate) * 0.5;
  if (!isFinite(config_.cutoff_hz) || config_.cutoff_hz <= 0.0 || config_.cutoff_hz >= nyquist_hz) {
    throw std::runtime_error("detector.cutoff_hz must be finite and in (0.0, Nyquist)");
  }
  if (!isFinite(config_.threshold) || config_.threshold < 0.0 || config_.threshold > 1.0) {
    throw std::runtime_error("detector.threshold must be finite and in [0.0, 1.0]");
  }
  if (!isFinite(config_.attenuation_db) ||
      config_.attenuation_db < kMinimumAttenuationDb ||
      config_.attenuation_db > 0.0)
  {
    throw std::runtime_error("detector.attenuation_db must be finite and in [-120.0, 0.0]");
  }
  if (config_.expected_channels <= 0) {
    throw std::runtime_error("expected.channels must be > 0");
  }
  if (config_.expected_encoding != kEncodingFloat32) {
    throw std::runtime_error("fa_deesser requires expected.encoding=FLOAT32LE");
  }
  if (config_.expected_bit_depth != 32) {
    throw std::runtime_error("fa_deesser requires expected.bit_depth=32");
  }
  if (config_.expected_layout != kInterleavedLayout) {
    throw std::runtime_error("fa_deesser requires expected.layout=interleaved");
  }
  if (config_.qos_depth <= 0) {
    throw std::runtime_error("qos.depth must be > 0");
  }
  if (config_.diagnostics_publish_period_ms <= 0) {
    throw std::runtime_error("diagnostics.publish_period_ms must be > 0");
  }

  config_.attenuation_gain = std::pow(10.0, config_.attenuation_db / 20.0);
  if (!isFinite(config_.attenuation_gain) ||
      config_.attenuation_gain < 0.0 ||
      config_.attenuation_gain > 1.0)
  {
    throw std::runtime_error("detector.attenuation_db produced an invalid linear gain");
  }

  RCLCPP_INFO(
    this->get_logger(),
    "De-esser config: input=%s output=%s cutoff=%fHz threshold=%f attenuation=%fdB gain=%f expected=%dHz/%d/%s/%d qos_depth=%d reliable=%s diag=%dms",
    config_.input_topic.c_str(),
    config_.output_topic.c_str(),
    config_.cutoff_hz,
    config_.threshold,
    config_.attenuation_db,
    config_.attenuation_gain,
    config_.expected_sample_rate,
    config_.expected_channels,
    config_.expected_encoding.c_str(),
    config_.expected_bit_depth,
    config_.qos_depth,
    config_.qos_reliable ? "true" : "false",
    config_.diagnostics_publish_period_ms);
}

void FaDeesserNode::setupInterfaces()
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
    std::bind(&FaDeesserNode::handleFrame, this, std::placeholders::_1));

  diag_pub_ = this->create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
    "diagnostics",
    rclcpp::SystemDefaultsQoS());
  diag_timer_ = this->create_wall_timer(
    std::chrono::milliseconds(config_.diagnostics_publish_period_ms),
    std::bind(&FaDeesserNode::publishDiagnostics, this));
}

void FaDeesserNode::handleFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg)
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

  fa_interfaces::msg::AudioFrame out;
  if (!applyDeesser(*msg, out)) {
    frames_dropped_.fetch_add(1);
    return;
  }

  audio_pub_->publish(out);
  frames_out_.fetch_add(1);
}

bool FaDeesserNode::validateFrame(const fa_interfaces::msg::AudioFrame & msg)
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

bool FaDeesserNode::applyDeesser(
  const fa_interfaces::msg::AudioFrame & in,
  fa_interfaces::msg::AudioFrame & out)
{
  out = in;
  out.stream_id = config_.output_topic;
  out.data.resize(in.data.size());

  const double alpha = 1.0 - std::exp((-2.0 * kPi * config_.cutoff_hz) / config_.expected_sample_rate);
  if (!isFinite(alpha) || alpha <= 0.0 || alpha >= 1.0) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping frame because de-esser filter coefficient is invalid");
    return false;
  }

  std::vector<double> next_low_band_state = low_band_state_;
  const bool source_changed = in.source_id != active_source_id_;
  if (source_changed) {
    std::fill(next_low_band_state.begin(), next_low_band_state.end(), 0.0);
  }

  const size_t channels = static_cast<size_t>(config_.expected_channels);
  const size_t sample_count = in.data.size() / sizeof(float);
  uint64_t attenuated_in_frame = 0;
  for (size_t i = 0; i < sample_count; ++i) {
    float sample = 0.0F;
    std::memcpy(&sample, in.data.data() + (i * sizeof(float)), sizeof(float));
    if (!std::isfinite(sample) || sample < kMinNormalizedSample || sample > kMaxNormalizedSample) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 3000,
        "Dropping frame because input sample is outside normalized FLOAT32LE range");
      return false;
    }

    const size_t channel_index = i % channels;
    const double input = static_cast<double>(sample);
    const double low_band = next_low_band_state[channel_index] +
      (alpha * (input - next_low_band_state[channel_index]));
    const double high_band = input - low_band;
    next_low_band_state[channel_index] = low_band;

    double processed_high_band = high_band;
    if (std::abs(high_band) >= config_.threshold) {
      processed_high_band = high_band * config_.attenuation_gain;
      ++attenuated_in_frame;
    }

    const double output = low_band + processed_high_band;
    if (!isFinite(output) || output < kMinNormalizedSample || output > kMaxNormalizedSample) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 3000,
        "Dropping frame because de-esser output is outside normalized FLOAT32LE range");
      return false;
    }

    const float out_sample = static_cast<float>(output);
    if (!std::isfinite(out_sample) ||
        out_sample < kMinNormalizedSample ||
        out_sample > kMaxNormalizedSample)
    {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 3000,
        "Dropping frame because de-esser output cannot be represented as normalized FLOAT32LE");
      return false;
    }
    std::memcpy(out.data.data() + (i * sizeof(float)), &out_sample, sizeof(float));
  }

  low_band_state_ = next_low_band_state;
  if (source_changed) {
    active_source_id_ = in.source_id;
    filter_resets_.fetch_add(1);
  }
  samples_attenuated_.fetch_add(attenuated_in_frame);
  return true;
}

void FaDeesserNode::publishDiagnostics()
{
  diagnostic_msgs::msg::DiagnosticArray array_msg;
  array_msg.header.stamp = this->now();

  diagnostic_msgs::msg::DiagnosticStatus status;
  status.name = "fa_deesser";
  status.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
  status.message = "running";
  status.values.reserve(12);
  pushKeyValue(status, "detector_cutoff_hz", std::to_string(config_.cutoff_hz));
  pushKeyValue(status, "detector_threshold", std::to_string(config_.threshold));
  pushKeyValue(status, "detector_attenuation_db", std::to_string(config_.attenuation_db));
  pushKeyValue(status, "detector_attenuation_gain", std::to_string(config_.attenuation_gain));
  pushKeyValue(status, "expected_sample_rate", std::to_string(config_.expected_sample_rate));
  pushKeyValue(status, "expected_channels", std::to_string(config_.expected_channels));
  pushKeyValue(status, "frames_in", std::to_string(frames_in_.load()));
  pushKeyValue(status, "frames_out", std::to_string(frames_out_.load()));
  pushKeyValue(status, "frames_dropped", std::to_string(frames_dropped_.load()));
  pushKeyValue(status, "samples_attenuated", std::to_string(samples_attenuated_.load()));
  pushKeyValue(status, "filter_resets", std::to_string(filter_resets_.load()));
  pushKeyValue(status, "output_topic", config_.output_topic);

  array_msg.status.push_back(status);
  diag_pub_->publish(array_msg);
}

}  // namespace fa_deesser

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<fa_deesser::FaDeesserNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return EXIT_SUCCESS;
  } catch (const std::exception & e) {
    RCLCPP_FATAL(rclcpp::get_logger("fa_deesser"), "Exception: %s", e.what());
    rclcpp::shutdown();
    return EXIT_FAILURE;
  }
}

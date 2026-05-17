#include "fa_declick/fa_declick_node.hpp"

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

namespace fa_declick
{

namespace
{
constexpr const char * kEncodingFloat32 = "FLOAT32LE";
constexpr const char * kInterleavedLayout = "interleaved";
constexpr float kMinNormalizedSample = -1.0F;
constexpr float kMaxNormalizedSample = 1.0F;

bool isFinite(double value)
{
  return std::isfinite(value);
}

bool isNormalizedSample(double value)
{
  return isFinite(value) &&
    value >= static_cast<double>(kMinNormalizedSample) &&
    value <= static_cast<double>(kMaxNormalizedSample);
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

size_t sampleIndex(size_t frame_index, size_t channel_index, size_t channel_count)
{
  return (frame_index * channel_count) + channel_index;
}

float readFloatSample(const std::vector<uint8_t> & data, size_t sample_index)
{
  float sample = 0.0F;
  std::memcpy(&sample, data.data() + (sample_index * sizeof(float)), sizeof(float));
  return sample;
}

void writeFloatSample(std::vector<uint8_t> & data, size_t sample_index, float sample)
{
  std::memcpy(data.data() + (sample_index * sizeof(float)), &sample, sizeof(float));
}
}  // namespace

FaDeclickNode::FaDeclickNode()
: rclcpp::Node("fa_declick")
{
  RCLCPP_INFO(this->get_logger(), "Starting FA Declick node");
  loadParameters();
  setupInterfaces();
}

void FaDeclickNode::loadParameters()
{
  this->declare_parameter("input_topic", config_.input_topic);
  this->declare_parameter("output_topic", config_.output_topic);
  this->declare_parameter<double>("threshold.delta", config_.threshold_delta);
  this->declare_parameter<int>("window.max_samples", config_.window_max_samples);
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
  config_.threshold_delta = this->get_parameter("threshold.delta").as_double();
  config_.window_max_samples = this->get_parameter("window.max_samples").as_int();
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
  if (!isFinite(config_.threshold_delta) ||
      config_.threshold_delta <= 0.0 ||
      config_.threshold_delta > 2.0)
  {
    throw std::runtime_error("threshold.delta must be finite and in (0.0, 2.0]");
  }
  if (config_.window_max_samples <= 0) {
    throw std::runtime_error("window.max_samples must be > 0");
  }
  if (config_.expected_sample_rate <= 0) {
    throw std::runtime_error("expected.sample_rate must be > 0");
  }
  if (config_.expected_channels <= 0) {
    throw std::runtime_error("expected.channels must be > 0");
  }
  if (config_.expected_encoding != kEncodingFloat32) {
    throw std::runtime_error("fa_declick requires expected.encoding=FLOAT32LE");
  }
  if (config_.expected_bit_depth != 32) {
    throw std::runtime_error("fa_declick requires expected.bit_depth=32");
  }
  if (config_.expected_layout != kInterleavedLayout) {
    throw std::runtime_error("fa_declick requires expected.layout=interleaved");
  }
  if (config_.qos_depth <= 0) {
    throw std::runtime_error("qos.depth must be > 0");
  }
  if (config_.diagnostics_publish_period_ms <= 0) {
    throw std::runtime_error("diagnostics.publish_period_ms must be > 0");
  }

  max_click_samples_ = static_cast<size_t>(config_.window_max_samples);

  RCLCPP_INFO(
    this->get_logger(),
    "Declick config: input=%s output=%s threshold_delta=%f window_max_samples=%d "
    "expected=%dHz/%d/%s/%d/%s qos_depth=%d reliable=%s diag=%dms",
    config_.input_topic.c_str(),
    config_.output_topic.c_str(),
    config_.threshold_delta,
    config_.window_max_samples,
    config_.expected_sample_rate,
    config_.expected_channels,
    config_.expected_encoding.c_str(),
    config_.expected_bit_depth,
    config_.expected_layout.c_str(),
    config_.qos_depth,
    config_.qos_reliable ? "true" : "false",
    config_.diagnostics_publish_period_ms);
}

void FaDeclickNode::setupInterfaces()
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
    std::bind(&FaDeclickNode::handleFrame, this, std::placeholders::_1));

  diag_pub_ = this->create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
    "diagnostics",
    rclcpp::SystemDefaultsQoS());
  diag_timer_ = this->create_wall_timer(
    std::chrono::milliseconds(config_.diagnostics_publish_period_ms),
    std::bind(&FaDeclickNode::publishDiagnostics, this));
}

void FaDeclickNode::handleFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg)
{
  frames_in_.fetch_add(1);

  if (!msg) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Received null AudioFrame pointer");
    frames_dropped_.fetch_add(1);
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
  if (!applyDeclick(*msg, out)) {
    frames_dropped_.fetch_add(1);
    return;
  }

  audio_pub_->publish(out);
  frames_out_.fetch_add(1);
}

bool FaDeclickNode::validateFrame(const fa_interfaces::msg::AudioFrame & msg)
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
  if (msg.data.empty() || (msg.data.size() % bytesPerFrame()) != 0) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "AudioFrame data size is invalid for FLOAT32LE interleaved samples");
    return false;
  }
  return true;
}

bool FaDeclickNode::validateSamples(const fa_interfaces::msg::AudioFrame & msg)
{
  const size_t sample_count = msg.data.size() / sizeof(float);
  for (size_t sample_index = 0; sample_index < sample_count; ++sample_index) {
    const float sample = readFloatSample(msg.data, sample_index);
    if (!isNormalizedSample(static_cast<double>(sample))) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 3000,
        "Dropping frame because input sample is outside normalized FLOAT32LE range");
      return false;
    }
  }
  return true;
}

bool FaDeclickNode::applyDeclick(
  const fa_interfaces::msg::AudioFrame & in,
  fa_interfaces::msg::AudioFrame & out)
{
  const size_t channel_count = static_cast<size_t>(config_.expected_channels);
  const size_t sample_count = in.data.size() / sizeof(float);
  const size_t frame_count = in.data.size() / bytesPerFrame();
  std::vector<float> input_samples(sample_count, 0.0F);
  std::vector<float> output_samples(sample_count, 0.0F);

  for (size_t sample_index = 0; sample_index < sample_count; ++sample_index) {
    const float sample = readFloatSample(in.data, sample_index);
    input_samples.at(sample_index) = sample;
    output_samples.at(sample_index) = sample;
  }

  uint64_t corrected_samples = 0;
  uint64_t corrected_runs = 0;
  if (frame_count >= 3) {
    for (size_t channel_index = 0; channel_index < channel_count; ++channel_index) {
      size_t frame_index = 1;
      while (frame_index + 1 < frame_count) {
        const size_t run_length = detectClickRun(
          input_samples, frame_index, channel_index, frame_count, channel_count);
        if (run_length == 0) {
          ++frame_index;
          continue;
        }

        const float previous = input_samples.at(
          sampleIndex(frame_index - 1, channel_index, channel_count));
        const float next = input_samples.at(
          sampleIndex(frame_index + run_length, channel_index, channel_count));
        const double corrected = (static_cast<double>(previous) + static_cast<double>(next)) / 2.0;
        if (!isNormalizedSample(corrected)) {
          RCLCPP_WARN_THROTTLE(
            this->get_logger(), *this->get_clock(), 3000,
            "Dropping frame because declick output is outside normalized FLOAT32LE range");
          return false;
        }

        const float corrected_sample = static_cast<float>(corrected);
        if (!isNormalizedSample(static_cast<double>(corrected_sample))) {
          RCLCPP_WARN_THROTTLE(
            this->get_logger(), *this->get_clock(), 3000,
            "Dropping frame because declick output cannot be represented as normalized FLOAT32LE");
          return false;
        }

        for (size_t offset = 0; offset < run_length; ++offset) {
          output_samples.at(sampleIndex(frame_index + offset, channel_index, channel_count)) =
            corrected_sample;
        }
        corrected_samples += run_length;
        ++corrected_runs;
        frame_index += run_length;
      }
    }
  }

  out = in;
  out.stream_id = config_.output_topic;
  out.data.resize(in.data.size());
  for (size_t sample_index = 0; sample_index < sample_count; ++sample_index) {
    writeFloatSample(out.data, sample_index, output_samples.at(sample_index));
  }

  samples_corrected_.fetch_add(corrected_samples);
  click_runs_corrected_.fetch_add(corrected_runs);
  return true;
}

size_t FaDeclickNode::detectClickRun(
  const std::vector<float> & samples,
  size_t frame_index,
  size_t channel_index,
  size_t frame_count,
  size_t channel_count) const
{
  if (frame_index == 0 || frame_index + 1 >= frame_count) {
    return 0;
  }

  const double delta = config_.threshold_delta;
  const size_t max_window = std::min(max_click_samples_, frame_count - frame_index - 1);
  const float previous = samples.at(sampleIndex(frame_index - 1, channel_index, channel_count));
  for (size_t run_length = 1; run_length <= max_window; ++run_length) {
    const float next = samples.at(sampleIndex(frame_index + run_length, channel_index, channel_count));
    if (std::abs(static_cast<double>(previous) - static_cast<double>(next)) > delta) {
      continue;
    }

    bool all_samples_are_clicks = true;
    for (size_t offset = 0; offset < run_length; ++offset) {
      const float current = samples.at(sampleIndex(frame_index + offset, channel_index, channel_count));
      const bool current_differs_from_previous =
        std::abs(static_cast<double>(current) - static_cast<double>(previous)) > delta;
      const bool current_differs_from_next =
        std::abs(static_cast<double>(current) - static_cast<double>(next)) > delta;
      if (!current_differs_from_previous || !current_differs_from_next) {
        all_samples_are_clicks = false;
        break;
      }
    }

    if (all_samples_are_clicks) {
      return run_length;
    }
  }

  return 0;
}

size_t FaDeclickNode::bytesPerFrame() const
{
  return static_cast<size_t>(config_.expected_channels) * sizeof(float);
}

void FaDeclickNode::publishDiagnostics()
{
  diagnostic_msgs::msg::DiagnosticArray array_msg;
  array_msg.header.stamp = this->now();

  diagnostic_msgs::msg::DiagnosticStatus status;
  status.name = "fa_declick";
  status.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
  status.message = "running";
  status.values.reserve(11);
  pushKeyValue(status, "threshold_delta", std::to_string(config_.threshold_delta));
  pushKeyValue(status, "window_max_samples", std::to_string(config_.window_max_samples));
  pushKeyValue(status, "input_topic", config_.input_topic);
  pushKeyValue(status, "output_topic", config_.output_topic);
  pushKeyValue(status, "expected_sample_rate", std::to_string(config_.expected_sample_rate));
  pushKeyValue(status, "expected_channels", std::to_string(config_.expected_channels));
  pushKeyValue(status, "frames_in", std::to_string(frames_in_.load()));
  pushKeyValue(status, "frames_out", std::to_string(frames_out_.load()));
  pushKeyValue(status, "frames_dropped", std::to_string(frames_dropped_.load()));
  pushKeyValue(status, "samples_corrected", std::to_string(samples_corrected_.load()));
  pushKeyValue(status, "click_runs_corrected", std::to_string(click_runs_corrected_.load()));

  array_msg.status.push_back(status);
  diag_pub_->publish(array_msg);
}

}  // namespace fa_declick

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<fa_declick::FaDeclickNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return EXIT_SUCCESS;
  } catch (const std::exception & e) {
    RCLCPP_FATAL(rclcpp::get_logger("fa_declick"), "Exception: %s", e.what());
    rclcpp::shutdown();
    return EXIT_FAILURE;
  }
}

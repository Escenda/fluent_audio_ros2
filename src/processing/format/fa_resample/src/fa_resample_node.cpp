#include "fa_resample/fa_resample_node.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "diagnostic_msgs/msg/diagnostic_status.hpp"
#include "diagnostic_msgs/msg/key_value.hpp"

namespace fa_resample
{

namespace
{
constexpr int kRequiredTargetSampleRate = 16000;

bool isFiniteFloat(float v)
{
  return std::isfinite(static_cast<double>(v));
}
}  // namespace

FaResampleNode::FaResampleNode()
: rclcpp::Node("fa_resample")
{
  RCLCPP_INFO(this->get_logger(), "Starting FA Resample node");
  loadParameters();
  setupInterfaces();
}

void FaResampleNode::loadParameters()
{
  this->declare_parameter<int>("target_sample_rate", config_.target_sample_rate);
  this->declare_parameter("output.encoding", config_.output_encoding);
  this->declare_parameter<int>("output.bit_depth", config_.output_bit_depth);

  this->declare_parameter<bool>("mic.enabled", config_.mic_enabled);
  this->declare_parameter("mic.input_topic", config_.mic_input_topic);
  this->declare_parameter("mic.output_topic", config_.mic_output_topic);

  this->declare_parameter<bool>("ref.enabled", config_.ref_enabled);
  this->declare_parameter("ref.input_topic", config_.ref_input_topic);
  this->declare_parameter("ref.output_topic", config_.ref_output_topic);

  this->declare_parameter<int>("qos.depth", config_.qos_depth);
  this->declare_parameter<bool>("qos.reliable", config_.qos_reliable);

  this->declare_parameter<int>(
    "diagnostics.publish_period_ms",
    config_.diagnostics_publish_period_ms);

  config_.target_sample_rate = this->get_parameter("target_sample_rate").as_int();
  config_.output_encoding = this->get_parameter("output.encoding").as_string();
  config_.output_bit_depth = this->get_parameter("output.bit_depth").as_int();

  config_.mic_enabled = this->get_parameter("mic.enabled").as_bool();
  config_.mic_input_topic = this->get_parameter("mic.input_topic").as_string();
  config_.mic_output_topic = this->get_parameter("mic.output_topic").as_string();

  config_.ref_enabled = this->get_parameter("ref.enabled").as_bool();
  config_.ref_input_topic = this->get_parameter("ref.input_topic").as_string();
  config_.ref_output_topic = this->get_parameter("ref.output_topic").as_string();

  config_.qos_depth = this->get_parameter("qos.depth").as_int();
  config_.qos_reliable = this->get_parameter("qos.reliable").as_bool();

  config_.diagnostics_publish_period_ms =
    this->get_parameter("diagnostics.publish_period_ms").as_int();

  if (config_.target_sample_rate != kRequiredTargetSampleRate) {
    throw std::runtime_error(
            "fa_resample requires target_sample_rate=16000 by design (got " +
            std::to_string(config_.target_sample_rate) + ")");
  }
  if (config_.output_encoding.empty()) {
    throw std::runtime_error("output.encoding is required (set via YAML)");
  }
  if (config_.output_bit_depth != 16 && config_.output_bit_depth != 32) {
    throw std::runtime_error("output.bit_depth must be 16 or 32 (set via YAML)");
  }
  if (config_.qos_depth <= 0) {
    throw std::runtime_error("qos.depth must be > 0 (set via YAML)");
  }
  if (config_.diagnostics_publish_period_ms <= 0) {
    throw std::runtime_error("diagnostics.publish_period_ms must be > 0 (set via YAML)");
  }
  if (config_.mic_enabled) {
    if (config_.mic_input_topic.empty()) {
      throw std::runtime_error("mic.input_topic is required when mic.enabled=true");
    }
    if (config_.mic_output_topic.empty()) {
      throw std::runtime_error("mic.output_topic is required when mic.enabled=true");
    }
  }
  if (config_.ref_enabled) {
    if (config_.ref_input_topic.empty()) {
      throw std::runtime_error("ref.input_topic is required when ref.enabled=true");
    }
    if (config_.ref_output_topic.empty()) {
      throw std::runtime_error("ref.output_topic is required when ref.enabled=true");
    }
  }

  RCLCPP_INFO(this->get_logger(),
    "Resample config: target_sr=%dHz out_encoding=%s out_bits=%d qos_depth=%d reliable=%s "
    "mic=%s (%s -> %s) ref=%s (%s -> %s) diag=%dms",
    config_.target_sample_rate, config_.output_encoding.c_str(), config_.output_bit_depth,
    config_.qos_depth, config_.qos_reliable ? "true" : "false",
    config_.mic_enabled ? "on" : "off",
    config_.mic_input_topic.c_str(), config_.mic_output_topic.c_str(),
    config_.ref_enabled ? "on" : "off",
    config_.ref_input_topic.c_str(), config_.ref_output_topic.c_str(),
    config_.diagnostics_publish_period_ms);
}

void FaResampleNode::setupInterfaces()
{
  rclcpp::QoS qos(std::max<int>(1, config_.qos_depth));
  if (config_.qos_reliable) {
    qos.reliable();
  } else {
    qos.best_effort();
  }

  if (config_.mic_enabled) {
    mic_pub_ = this->create_publisher<fa_interfaces::msg::AudioFrame>(config_.mic_output_topic, qos);
    mic_sub_ = this->create_subscription<fa_interfaces::msg::AudioFrame>(
      config_.mic_input_topic, qos,
      std::bind(&FaResampleNode::handleMicFrame, this, std::placeholders::_1));
  }
  if (config_.ref_enabled) {
    ref_pub_ = this->create_publisher<fa_interfaces::msg::AudioFrame>(config_.ref_output_topic, qos);
    ref_sub_ = this->create_subscription<fa_interfaces::msg::AudioFrame>(
      config_.ref_input_topic, qos,
      std::bind(&FaResampleNode::handleRefFrame, this, std::placeholders::_1));
  }

  diag_pub_ = this->create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
    "diagnostics", rclcpp::SystemDefaultsQoS());
  diag_timer_ = this->create_wall_timer(
    std::chrono::milliseconds(config_.diagnostics_publish_period_ms),
    std::bind(&FaResampleNode::publishDiagnostics, this));
}

void FaResampleNode::handleMicFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg)
{
  mic_in_.fetch_add(1);
  if (!msg || !mic_pub_) {
    mic_drop_.fetch_add(1);
    return;
  }
  processAndPublish(*msg, mic_pub_, "mic", mic_out_, mic_drop_);
}

void FaResampleNode::handleRefFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg)
{
  ref_in_.fetch_add(1);
  if (!msg || !ref_pub_) {
    ref_drop_.fetch_add(1);
    return;
  }
  processAndPublish(*msg, ref_pub_, "ref", ref_out_, ref_drop_);
}

bool FaResampleNode::processAndPublish(
  const fa_interfaces::msg::AudioFrame & in,
  const rclcpp::Publisher<fa_interfaces::msg::AudioFrame>::SharedPtr & pub,
  const std::string & stream_name,
  std::atomic<uint64_t> & out_counter,
  std::atomic<uint64_t> & drop_counter)
{
  if (!pub) {
    drop_counter.fetch_add(1);
    return false;
  }

  if (in.sample_rate == 0 || in.channels == 0) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Invalid frame (%s): sample_rate=%u channels=%u", stream_name.c_str(), in.sample_rate, in.channels);
    drop_counter.fetch_add(1);
    return false;
  }
  if (in.data.empty()) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Empty frame data (%s), dropping", stream_name.c_str());
    drop_counter.fetch_add(1);
    return false;
  }

  std::vector<float> in_f32;
  uint32_t in_frames = 0;
  uint32_t in_channels = 0;
  if (!decodeToFloat(in, in_f32, in_frames, in_channels)) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Failed to decode input frame (%s): bit_depth=%u data=%zu",
      stream_name.c_str(), in.bit_depth, in.data.size());
    drop_counter.fetch_add(1);
    return false;
  }
  if (in_channels != in.channels) {
    drop_counter.fetch_add(1);
    return false;
  }

  uint32_t out_frames = 0;
  std::vector<float> out_f32 = resampleLinear(
    in_f32,
    in.sample_rate,
    static_cast<uint32_t>(config_.target_sample_rate),
    in_channels,
    in_frames,
    out_frames);

  if (out_f32.empty() || out_frames == 0) {
    drop_counter.fetch_add(1);
    return false;
  }

  for (float v : out_f32) {
    if (!isFiniteFloat(v)) {
      drop_counter.fetch_add(1);
      return false;
    }
  }

  std::vector<uint8_t> out_bytes;
  encodeFromFloat(out_f32, config_.output_bit_depth, out_bytes);
  if (out_bytes.empty()) {
    drop_counter.fetch_add(1);
    return false;
  }

  float out_rms = 0.0f;
  float out_peak = 0.0f;
  computeRmsPeak(out_f32, out_rms, out_peak);

  fa_interfaces::msg::AudioFrame out;
  out.header = in.header;
  out.encoding = config_.output_encoding;
  out.sample_rate = static_cast<uint32_t>(config_.target_sample_rate);
  out.channels = in.channels;
  out.bit_depth = static_cast<uint32_t>(config_.output_bit_depth);
  out.rms = out_rms;
  out.peak = out_peak;
  out.vad = in.vad;
  out.data = out_bytes;
  out.epoch = in.epoch;

  pub->publish(out);
  out_counter.fetch_add(1);
  return true;
}

bool FaResampleNode::decodeToFloat(
  const fa_interfaces::msg::AudioFrame & msg,
  std::vector<float> & out_interleaved,
  uint32_t & out_frames,
  uint32_t & out_channels)
{
  out_interleaved.clear();
  out_frames = 0;
  out_channels = 0;

  if (msg.channels == 0) {
    return false;
  }
  const uint32_t channels = msg.channels;
  const uint32_t bit_depth = msg.bit_depth;
  if (bit_depth != 16 && bit_depth != 32) {
    return false;
  }
  const size_t bytes_per_sample = static_cast<size_t>(bit_depth / 8);
  const size_t bytes_per_frame = static_cast<size_t>(channels) * bytes_per_sample;
  if (bytes_per_frame == 0 || (msg.data.size() % bytes_per_frame) != 0) {
    return false;
  }

  const uint32_t frames = static_cast<uint32_t>(msg.data.size() / bytes_per_frame);
  const size_t sample_count = static_cast<size_t>(frames) * channels;
  if (sample_count == 0) {
    return false;
  }

  out_interleaved.resize(sample_count);

  if (bit_depth == 16) {
    std::vector<int16_t> tmp(sample_count);
    std::memcpy(tmp.data(), msg.data.data(), msg.data.size());
    for (size_t i = 0; i < sample_count; ++i) {
      out_interleaved[i] = static_cast<float>(tmp[i]) / 32768.0f;
    }
  } else {
    std::memcpy(out_interleaved.data(), msg.data.data(), msg.data.size());
  }

  out_frames = frames;
  out_channels = channels;
  return true;
}

std::vector<float> FaResampleNode::resampleLinear(
  const std::vector<float> & interleaved,
  uint32_t in_rate,
  uint32_t out_rate,
  uint32_t channels,
  uint32_t in_frames,
  uint32_t & out_frames)
{
  out_frames = 0;
  if (in_rate == 0 || out_rate == 0 || channels == 0 || in_frames == 0 || interleaved.empty()) {
    return {};
  }
  if (interleaved.size() != static_cast<size_t>(in_frames) * channels) {
    return {};
  }

  if (in_rate == out_rate) {
    out_frames = in_frames;
    return interleaved;
  }

  const double ratio = static_cast<double>(out_rate) / static_cast<double>(in_rate);
  const double out_frames_f = static_cast<double>(in_frames) * ratio;
  const uint32_t frames = static_cast<uint32_t>(std::max<double>(1.0, std::lround(out_frames_f)));

  std::vector<float> out;
  out.resize(static_cast<size_t>(frames) * channels);

  const double step = static_cast<double>(in_rate) / static_cast<double>(out_rate);
  for (uint32_t i = 0; i < frames; ++i) {
    const double src_pos = static_cast<double>(i) * step;
    uint32_t idx0 = static_cast<uint32_t>(std::floor(src_pos));
    const double frac = src_pos - static_cast<double>(idx0);

    if (idx0 >= in_frames) {
      idx0 = in_frames - 1;
    }
    const uint32_t idx1 = std::min<uint32_t>(idx0 + 1, in_frames - 1);

    for (uint32_t ch = 0; ch < channels; ++ch) {
      const float s0 = interleaved[static_cast<size_t>(idx0) * channels + ch];
      const float s1 = interleaved[static_cast<size_t>(idx1) * channels + ch];
      const float v = static_cast<float>((1.0 - frac) * static_cast<double>(s0) + frac * static_cast<double>(s1));
      out[static_cast<size_t>(i) * channels + ch] = v;
    }
  }

  out_frames = frames;
  return out;
}

void FaResampleNode::encodeFromFloat(
  const std::vector<float> & interleaved,
  int bit_depth,
  std::vector<uint8_t> & out_bytes)
{
  out_bytes.clear();
  if (interleaved.empty()) {
    return;
  }

  if (bit_depth == 16) {
    std::vector<int16_t> pcm(interleaved.size());
    for (size_t i = 0; i < interleaved.size(); ++i) {
      float s = std::clamp(interleaved[i], -1.0f, 1.0f);
      const int32_t scaled = static_cast<int32_t>(std::lround(static_cast<double>(s) * 32767.0));
      pcm[i] = static_cast<int16_t>(std::clamp<int32_t>(scaled, -32768, 32767));
    }
    out_bytes.resize(pcm.size() * sizeof(int16_t));
    std::memcpy(out_bytes.data(), pcm.data(), out_bytes.size());
    return;
  }

  if (bit_depth == 32) {
    out_bytes.resize(interleaved.size() * sizeof(float));
    std::memcpy(out_bytes.data(), interleaved.data(), out_bytes.size());
  }
}

void FaResampleNode::computeRmsPeak(const std::vector<float> & interleaved, float & out_rms, float & out_peak)
{
  out_rms = 0.0f;
  out_peak = 0.0f;
  if (interleaved.empty()) {
    return;
  }

  double accum = 0.0;
  double peak = 0.0;
  for (float v : interleaved) {
    const double dv = static_cast<double>(v);
    accum += dv * dv;
    peak = std::max(peak, std::abs(dv));
  }
  out_rms = static_cast<float>(std::sqrt(accum / static_cast<double>(interleaved.size())));
  out_peak = static_cast<float>(std::min<double>(peak, static_cast<double>(std::numeric_limits<float>::max())));
}

void FaResampleNode::publishDiagnostics()
{
  diagnostic_msgs::msg::DiagnosticArray array_msg;
  array_msg.header.stamp = this->now();

  diagnostic_msgs::msg::DiagnosticStatus status;
  status.name = "fa_resample";
  status.hardware_id = "dsp";
  status.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
  status.message = "running";

  auto push_kv = [&status](const std::string & key, const std::string & value) {
      diagnostic_msgs::msg::KeyValue kv;
      kv.key = key;
      kv.value = value;
      status.values.push_back(kv);
    };

  status.values.reserve(12);
  push_kv("target_sample_rate", std::to_string(config_.target_sample_rate));
  push_kv("output.encoding", config_.output_encoding);
  push_kv("output.bit_depth", std::to_string(config_.output_bit_depth));
  push_kv("mic.in", std::to_string(mic_in_.load()));
  push_kv("mic.out", std::to_string(mic_out_.load()));
  push_kv("mic.drop", std::to_string(mic_drop_.load()));
  push_kv("ref.in", std::to_string(ref_in_.load()));
  push_kv("ref.out", std::to_string(ref_out_.load()));
  push_kv("ref.drop", std::to_string(ref_drop_.load()));

  array_msg.status.push_back(status);
  diag_pub_->publish(array_msg);
}

}  // namespace fa_resample

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<fa_resample::FaResampleNode>();
    rclcpp::spin(node);
  } catch (const std::exception & e) {
    RCLCPP_FATAL(rclcpp::get_logger("fa_resample"), "Exception: %s", e.what());
    return EXIT_FAILURE;
  }
  rclcpp::shutdown();
  return EXIT_SUCCESS;
}


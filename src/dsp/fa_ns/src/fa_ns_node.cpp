#include "fa_ns/fa_ns_node.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "ament_index_cpp/get_package_share_directory.hpp"
#include "diagnostic_msgs/msg/diagnostic_status.hpp"
#include "diagnostic_msgs/msg/key_value.hpp"

#ifdef FA_NS_WITH_ONNXRUNTIME
#include "fa_ns/dtln_onnx_engine.hpp"
#endif

namespace fa_ns
{

namespace
{
constexpr int kRequiredSampleRate = 16000;

std::string resolveModelPathOrThrow(const std::string & path_or_empty, const std::string & file_name)
{
  std::filesystem::path path(path_or_empty);
  if (path_or_empty.empty()) {
    const std::filesystem::path share_dir(ament_index_cpp::get_package_share_directory("fa_ns"));
    path = share_dir / "models" / file_name;
  }
  std::error_code ec;
  if (!std::filesystem::exists(path, ec) || ec) {
    throw std::runtime_error("Model file not found: " + path.string());
  }
  return path.string();
}

uint64_t nanosSince(const std::chrono::steady_clock::time_point & start)
{
  return static_cast<uint64_t>(
    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count());
}

void updateMaxAtomic(std::atomic<uint64_t> & target, uint64_t value)
{
  uint64_t current = target.load();
  while (value > current && !target.compare_exchange_weak(current, value)) {
    // retry
  }
}
}

FaNsNode::FaNsNode()
: rclcpp::Node("fa_ns")
{
  RCLCPP_INFO(this->get_logger(), "Starting FA NS node");
  loadParameters();
  setupInterfaces();
}

FaNsNode::~FaNsNode() = default;

void FaNsNode::loadParameters()
{
  this->declare_parameter<bool>("enabled", config_.enabled);
  this->declare_parameter("backend", config_.backend);
  this->declare_parameter("input_topic", config_.input_topic);
  this->declare_parameter("output_topic", config_.output_topic);
  this->declare_parameter<int>("expected_sample_rate", config_.expected_sample_rate);
  this->declare_parameter<int>("expected_channels", config_.expected_channels);
  this->declare_parameter("output.encoding", config_.output_encoding);
  this->declare_parameter<int>("output.bit_depth", config_.output_bit_depth);

  this->declare_parameter<int>("dtln.block_len", config_.dtln_block_len);
  this->declare_parameter<int>("dtln.block_shift", config_.dtln_block_shift);
  this->declare_parameter("dtln.model_1_path", config_.dtln_model_1_path);
  this->declare_parameter("dtln.model_2_path", config_.dtln_model_2_path);
  this->declare_parameter<int>("dtln.intra_op_num_threads", config_.dtln_intra_op_num_threads);
  this->declare_parameter<int>("dtln.inter_op_num_threads", config_.dtln_inter_op_num_threads);
  this->declare_parameter<bool>(
    "dtln.enable_ort_optimizations",
    config_.dtln_enable_ort_optimizations);
  this->declare_parameter<int>("qos.depth", config_.qos_depth);
  this->declare_parameter<bool>("qos.reliable", config_.qos_reliable);
  this->declare_parameter<int>(
    "diagnostics.publish_period_ms",
    config_.diagnostics_publish_period_ms);

  config_.enabled = this->get_parameter("enabled").as_bool();
  config_.backend = this->get_parameter("backend").as_string();
  config_.input_topic = this->get_parameter("input_topic").as_string();
  config_.output_topic = this->get_parameter("output_topic").as_string();
  config_.expected_sample_rate = this->get_parameter("expected_sample_rate").as_int();
  config_.expected_channels = this->get_parameter("expected_channels").as_int();
  config_.output_encoding = this->get_parameter("output.encoding").as_string();
  config_.output_bit_depth = this->get_parameter("output.bit_depth").as_int();

  config_.dtln_block_len = this->get_parameter("dtln.block_len").as_int();
  config_.dtln_block_shift = this->get_parameter("dtln.block_shift").as_int();
  config_.dtln_model_1_path = this->get_parameter("dtln.model_1_path").as_string();
  config_.dtln_model_2_path = this->get_parameter("dtln.model_2_path").as_string();
  config_.dtln_intra_op_num_threads = this->get_parameter("dtln.intra_op_num_threads").as_int();
  config_.dtln_inter_op_num_threads = this->get_parameter("dtln.inter_op_num_threads").as_int();
  config_.dtln_enable_ort_optimizations = this->get_parameter("dtln.enable_ort_optimizations").as_bool();
  config_.qos_depth = this->get_parameter("qos.depth").as_int();
  config_.qos_reliable = this->get_parameter("qos.reliable").as_bool();
  config_.diagnostics_publish_period_ms =
    this->get_parameter("diagnostics.publish_period_ms").as_int();

  if (config_.input_topic.empty()) {
    throw std::runtime_error("input_topic is required (set via YAML)");
  }
  if (config_.output_topic.empty()) {
    throw std::runtime_error("output_topic is required (set via YAML)");
  }
  if (config_.expected_sample_rate != kRequiredSampleRate) {
    throw std::runtime_error(
            "fa_ns requires expected_sample_rate=16000 by design (got " +
            std::to_string(config_.expected_sample_rate) + ")");
  }
  if (config_.qos_depth <= 0) {
    throw std::runtime_error("qos.depth must be > 0 (set via YAML)");
  }
  if (config_.diagnostics_publish_period_ms <= 0) {
    throw std::runtime_error("diagnostics.publish_period_ms must be > 0 (set via YAML)");
  }
  if (config_.output_encoding.empty()) {
    throw std::runtime_error("output.encoding is required (set via YAML)");
  }
  if (config_.output_bit_depth != 16 && config_.output_bit_depth != 32) {
    throw std::runtime_error("output.bit_depth must be 16 or 32 (set via YAML)");
  }
  if (config_.backend.empty()) {
    throw std::runtime_error("backend is required (set via YAML)");
  }

  if (config_.backend == "onnxruntime") {
    RCLCPP_WARN(this->get_logger(), "backend=onnxruntime is deprecated; use backend=dtln_onnx");
    config_.backend = "dtln_onnx";
  }

  if (config_.backend != "passthrough" && config_.backend != "dtln_onnx") {
    throw std::runtime_error("backend must be passthrough or dtln_onnx");
  }

  if (config_.backend == "dtln_onnx") {
    if (config_.dtln_block_len <= 0 || config_.dtln_block_shift <= 0) {
      throw std::runtime_error("dtln.block_len and dtln.block_shift must be > 0 (set via YAML)");
    }
    if (config_.dtln_block_shift > config_.dtln_block_len) {
      throw std::runtime_error("dtln.block_shift must be <= dtln.block_len");
    }

#ifdef FA_NS_WITH_ONNXRUNTIME
    DtlnOnnxConfig dtln_cfg;
    dtln_cfg.block_len = config_.dtln_block_len;
    dtln_cfg.block_shift = config_.dtln_block_shift;
    dtln_cfg.model_1_path = resolveModelPathOrThrow(config_.dtln_model_1_path, "model_1.onnx");
    dtln_cfg.model_2_path = resolveModelPathOrThrow(config_.dtln_model_2_path, "model_2.onnx");
    dtln_cfg.intra_op_num_threads = std::max<int>(1, config_.dtln_intra_op_num_threads);
    dtln_cfg.inter_op_num_threads = std::max<int>(1, config_.dtln_inter_op_num_threads);
    dtln_cfg.enable_ort_optimizations = config_.dtln_enable_ort_optimizations;
    dtln_ = std::make_unique<DtlnOnnxEngine>(dtln_cfg);
#else
    throw std::runtime_error("fa_ns was built without ONNX Runtime support (FA_NS_WITH_ONNXRUNTIME=0)");
#endif
  }

  RCLCPP_INFO(this->get_logger(),
    "NS config: enabled=%s backend=%s input=%s output=%s expected_sr=%d expected_ch=%d "
    "out_enc=%s out_bits=%d qos_depth=%d reliable=%s",
    config_.enabled ? "true" : "false",
    config_.backend.c_str(),
    config_.input_topic.c_str(),
    config_.output_topic.c_str(),
    config_.expected_sample_rate,
    config_.expected_channels,
    config_.output_encoding.c_str(),
    config_.output_bit_depth,
    config_.qos_depth,
    config_.qos_reliable ? "true" : "false");
}

void FaNsNode::setupInterfaces()
{
  rclcpp::QoS qos(std::max<int>(1, config_.qos_depth));
  if (config_.qos_reliable) {
    qos.reliable();
  } else {
    qos.best_effort();
  }

  pub_ = this->create_publisher<fa_interfaces::msg::AudioFrame>(config_.output_topic, qos);
  sub_ = this->create_subscription<fa_interfaces::msg::AudioFrame>(
    config_.input_topic, qos,
    std::bind(&FaNsNode::onAudioFrame, this, std::placeholders::_1));

  diag_pub_ = this->create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
    "diagnostics", rclcpp::SystemDefaultsQoS());
  diag_timer_ = this->create_wall_timer(
    std::chrono::milliseconds(config_.diagnostics_publish_period_ms),
    std::bind(&FaNsNode::publishDiagnostics, this));
}

bool FaNsNode::validateFrame(const fa_interfaces::msg::AudioFrame & msg) const
{
  if (msg.sample_rate != static_cast<uint32_t>(config_.expected_sample_rate)) {
    return false;
  }
  if (config_.expected_channels > 0 && msg.channels != static_cast<uint32_t>(config_.expected_channels)) {
    return false;
  }
  if (msg.channels == 0 || msg.sample_rate == 0) {
    return false;
  }
  if (msg.bit_depth != 16 && msg.bit_depth != 32) {
    return false;
  }
  if (msg.data.empty()) {
    return false;
  }
  const size_t bytes_per_sample = static_cast<size_t>(msg.bit_depth / 8);
  const size_t bytes_per_frame = static_cast<size_t>(msg.channels) * bytes_per_sample;
  if (bytes_per_frame == 0 || (msg.data.size() % bytes_per_frame) != 0) {
    return false;
  }
  return true;
}

bool FaNsNode::decodeToFloat(const fa_interfaces::msg::AudioFrame & msg, std::vector<float> & out)
{
  out.clear();
  if (msg.channels == 0) {
    return false;
  }
  if (msg.bit_depth != 16 && msg.bit_depth != 32) {
    return false;
  }
  const size_t bytes_per_sample = static_cast<size_t>(msg.bit_depth / 8);
  const size_t bytes_per_frame = static_cast<size_t>(msg.channels) * bytes_per_sample;
  if (bytes_per_frame == 0 || (msg.data.size() % bytes_per_frame) != 0) {
    return false;
  }

  const size_t frames = msg.data.size() / bytes_per_frame;
  const size_t sample_count = frames * msg.channels;
  if (sample_count == 0) {
    return false;
  }

  out.resize(sample_count);
  if (msg.bit_depth == 16) {
    std::vector<int16_t> tmp(sample_count);
    std::memcpy(tmp.data(), msg.data.data(), msg.data.size());
    for (size_t i = 0; i < sample_count; ++i) {
      out[i] = static_cast<float>(tmp[i]) / 32768.0f;
    }
    return true;
  }

  std::memcpy(out.data(), msg.data.data(), msg.data.size());
  return true;
}

void FaNsNode::encodeFromFloat(
  const std::vector<float> & samples,
  int bit_depth,
  std::vector<uint8_t> & out_bytes)
{
  out_bytes.clear();
  if (samples.empty()) {
    return;
  }
  if (bit_depth == 16) {
    std::vector<int16_t> pcm(samples.size());
    for (size_t i = 0; i < samples.size(); ++i) {
      const float s = std::clamp(samples[i], -1.0f, 1.0f);
      const int32_t scaled = static_cast<int32_t>(std::lround(static_cast<double>(s) * 32767.0));
      pcm[i] = static_cast<int16_t>(std::clamp<int32_t>(scaled, -32768, 32767));
    }
    out_bytes.resize(pcm.size() * sizeof(int16_t));
    std::memcpy(out_bytes.data(), pcm.data(), out_bytes.size());
    return;
  }
  if (bit_depth == 32) {
    out_bytes.resize(samples.size() * sizeof(float));
    std::memcpy(out_bytes.data(), samples.data(), out_bytes.size());
  }
}

void FaNsNode::computeRmsPeak(const std::vector<float> & interleaved, float & out_rms, float & out_peak)
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
  out_peak = static_cast<float>(peak);
}

void FaNsNode::onAudioFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg)
{
  in_.fetch_add(1);
  if (!msg || !pub_) {
    drop_.fetch_add(1);
    return;
  }

  if (!validateFrame(*msg)) {
    drop_.fetch_add(1);
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping invalid frame: sr=%u ch=%u bits=%u bytes=%zu (expected sr=%d ch=%d)",
      msg->sample_rate, msg->channels, msg->bit_depth, msg->data.size(),
      config_.expected_sample_rate, config_.expected_channels);
    return;
  }

  const auto start = std::chrono::steady_clock::now();

  if (!config_.enabled || config_.backend == "passthrough") {
    pub_->publish(*msg);
    out_.fetch_add(1);
    const uint64_t elapsed_ns = nanosSince(start);
    process_ns_sum_.fetch_add(elapsed_ns);
    process_count_.fetch_add(1);
    updateMaxAtomic(process_ns_max_, elapsed_ns);
    return;
  }

  if (config_.backend != "dtln_onnx") {
    drop_.fetch_add(1);
    return;
  }

#ifdef FA_NS_WITH_ONNXRUNTIME
  if (!dtln_) {
    drop_.fetch_add(1);
    return;
  }

  std::vector<float> in_f32;
  if (!decodeToFloat(*msg, in_f32)) {
    drop_.fetch_add(1);
    return;
  }

  // DTLN is mono-only.
  if (msg->channels != 1) {
    drop_.fetch_add(1);
    return;
  }

  std::vector<float> out_f32;
  try {
    out_f32 = dtln_->process(in_f32.data(), in_f32.size());
  } catch (const std::exception & e) {
    drop_.fetch_add(1);
    RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 3000, "DTLN process failed: %s", e.what());
    return;
  }

  if (out_f32.empty()) {
    const uint64_t elapsed_ns = nanosSince(start);
    process_ns_sum_.fetch_add(elapsed_ns);
    process_count_.fetch_add(1);
    updateMaxAtomic(process_ns_max_, elapsed_ns);
    return;
  }

  std::vector<uint8_t> out_bytes;
  encodeFromFloat(out_f32, config_.output_bit_depth, out_bytes);
  if (out_bytes.empty()) {
    drop_.fetch_add(1);
    return;
  }

  float out_rms = 0.0f;
  float out_peak = 0.0f;
  computeRmsPeak(out_f32, out_rms, out_peak);

  fa_interfaces::msg::AudioFrame out_msg;
  out_msg.header = msg->header;
  out_msg.encoding = config_.output_encoding;
  out_msg.sample_rate = msg->sample_rate;
  out_msg.channels = msg->channels;
  out_msg.bit_depth = static_cast<uint32_t>(config_.output_bit_depth);
  out_msg.rms = out_rms;
  out_msg.peak = out_peak;
  out_msg.vad = msg->vad;
  out_msg.data = std::move(out_bytes);
  out_msg.epoch = msg->epoch;

  pub_->publish(out_msg);
  out_.fetch_add(1);
#else
  drop_.fetch_add(1);
  return;
#endif

  const uint64_t elapsed_ns = nanosSince(start);
  process_ns_sum_.fetch_add(elapsed_ns);
  process_count_.fetch_add(1);
  updateMaxAtomic(process_ns_max_, elapsed_ns);
}

void FaNsNode::publishDiagnostics()
{
  diagnostic_msgs::msg::DiagnosticArray array_msg;
  array_msg.header.stamp = this->now();

  diagnostic_msgs::msg::DiagnosticStatus status;
  status.name = "fa_ns";
  status.hardware_id = "dsp";
  status.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
  status.message = "running";

  auto push_kv = [&status](const std::string & key, const std::string & value) {
      diagnostic_msgs::msg::KeyValue kv;
      kv.key = key;
      kv.value = value;
      status.values.push_back(kv);
    };

  status.values.reserve(10);
  push_kv("enabled", config_.enabled ? "true" : "false");
  push_kv("backend", config_.backend);
  push_kv("input_topic", config_.input_topic);
  push_kv("output_topic", config_.output_topic);
  push_kv("expected_sample_rate", std::to_string(config_.expected_sample_rate));
  push_kv("expected_channels", std::to_string(config_.expected_channels));
  push_kv("output.encoding", config_.output_encoding);
  push_kv("output.bit_depth", std::to_string(config_.output_bit_depth));
  push_kv("frames.in", std::to_string(in_.load()));
  push_kv("frames.out", std::to_string(out_.load()));
  push_kv("frames.drop", std::to_string(drop_.load()));

  const uint64_t count = process_count_.load();
  const uint64_t sum_ns = process_ns_sum_.load();
  const uint64_t max_ns = process_ns_max_.load();
  if (count > 0) {
    const double mean_ms = (static_cast<double>(sum_ns) / static_cast<double>(count)) / 1e6;
    const double max_ms = static_cast<double>(max_ns) / 1e6;
    push_kv("process.mean_ms", std::to_string(mean_ms));
    push_kv("process.max_ms", std::to_string(max_ms));
  }

#ifdef FA_NS_WITH_ONNXRUNTIME
  if (dtln_) {
    push_kv("dtln.pending_samples", std::to_string(dtln_->pendingInputSamples()));
  }
#endif

  array_msg.status.push_back(status);
  diag_pub_->publish(array_msg);
}

}  // namespace fa_ns

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<fa_ns::FaNsNode>();
    rclcpp::spin(node);
  } catch (const std::exception & e) {
    RCLCPP_FATAL(rclcpp::get_logger("fa_ns"), "Exception: %s", e.what());
    return EXIT_FAILURE;
  }
  rclcpp::shutdown();
  return EXIT_SUCCESS;
}

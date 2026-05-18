#include "fa_denoise/fa_denoise_node.hpp"

#include <chrono>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "builtin_interfaces/msg/time.hpp"
#include "diagnostic_msgs/msg/diagnostic_status.hpp"
#include "diagnostic_msgs/msg/key_value.hpp"
#include "fa_denoise/backends/denoise_backend.hpp"
#include "fa_denoise/backends/passthrough_backend.hpp"

#ifdef FA_DENOISE_WITH_ONNXRUNTIME
#include "fa_denoise/backends/dtln_onnx_backend.hpp"
#endif

namespace fa_denoise
{

namespace
{
constexpr int kRequiredSampleRate = 16000;
constexpr const char * kInterleavedLayout = "interleaved";

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
    throw std::runtime_error(name + " must be a string parameter");
  }
  return parameter.as_string();
}

int readRequiredInt(const rclcpp::Node & node, const std::string & name)
{
  const rclcpp::Parameter parameter = getRequiredParameter(node, name);
  if (parameter.get_type() != rclcpp::ParameterType::PARAMETER_INTEGER) {
    throw std::runtime_error(name + " must be an integer parameter");
  }
  const int64_t value = parameter.as_int();
  if (value < static_cast<int64_t>(std::numeric_limits<int>::min()) ||
      value > static_cast<int64_t>(std::numeric_limits<int>::max()))
  {
    throw std::runtime_error(name + " must fit in a 32-bit signed integer");
  }
  return static_cast<int>(value);
}

bool readRequiredBool(const rclcpp::Node & node, const std::string & name)
{
  const rclcpp::Parameter parameter = getRequiredParameter(node, name);
  if (parameter.get_type() != rclcpp::ParameterType::PARAMETER_BOOL) {
    throw std::runtime_error(name + " must be a bool parameter");
  }
  return parameter.as_bool();
}

bool hasValidStamp(const builtin_interfaces::msg::Time & stamp)
{
  return stamp.sec != 0 || stamp.nanosec != 0U;
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

FaDenoiseNode::FaDenoiseNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("fa_denoise", options)
{
  RCLCPP_INFO(this->get_logger(), "Starting FA Denoise node");
  loadParameters();
  configureBackend();
  setupInterfaces();
}

FaDenoiseNode::~FaDenoiseNode() = default;

void FaDenoiseNode::loadParameters()
{
  this->declare_parameter<bool>("enabled");
  this->declare_parameter<std::string>("backend.name");
  this->declare_parameter<std::string>("input_topic");
  this->declare_parameter<std::string>("output_topic");
  this->declare_parameter<int>("expected_sample_rate");
  this->declare_parameter<int>("expected_channels");
  this->declare_parameter<std::string>("expected.encoding");
  this->declare_parameter<int>("expected.bit_depth");
  this->declare_parameter<std::string>("output.encoding");
  this->declare_parameter<int>("output.bit_depth");

  this->declare_parameter<int>("dtln.block_len");
  this->declare_parameter<int>("dtln.block_shift");
  this->declare_parameter<std::string>("dtln.model_1_path");
  this->declare_parameter<std::string>("dtln.model_2_path");
  this->declare_parameter<int>("dtln.intra_op_num_threads");
  this->declare_parameter<int>("dtln.inter_op_num_threads");
  this->declare_parameter<bool>("dtln.enable_ort_optimizations");
  this->declare_parameter<int>("qos.depth");
  this->declare_parameter<bool>("qos.reliable");
  this->declare_parameter<int>("diagnostics.publish_period_ms");
  this->declare_parameter<int>("diagnostics.qos.depth");
  this->declare_parameter<bool>("diagnostics.qos.reliable");

  config_.enabled = readRequiredBool(*this, "enabled");
  config_.backend_name = readRequiredString(*this, "backend.name");
  config_.input_topic = readRequiredString(*this, "input_topic");
  config_.output_topic = readRequiredString(*this, "output_topic");
  config_.expected_sample_rate = readRequiredInt(*this, "expected_sample_rate");
  config_.expected_channels = readRequiredInt(*this, "expected_channels");
  config_.expected_encoding = readRequiredString(*this, "expected.encoding");
  config_.expected_bit_depth = readRequiredInt(*this, "expected.bit_depth");
  config_.output_encoding = readRequiredString(*this, "output.encoding");
  config_.output_bit_depth = readRequiredInt(*this, "output.bit_depth");

  config_.dtln_block_len = readRequiredInt(*this, "dtln.block_len");
  config_.dtln_block_shift = readRequiredInt(*this, "dtln.block_shift");
  config_.dtln_model_1_path = readRequiredString(*this, "dtln.model_1_path");
  config_.dtln_model_2_path = readRequiredString(*this, "dtln.model_2_path");
  config_.dtln_intra_op_num_threads = readRequiredInt(*this, "dtln.intra_op_num_threads");
  config_.dtln_inter_op_num_threads = readRequiredInt(*this, "dtln.inter_op_num_threads");
  config_.dtln_enable_ort_optimizations = readRequiredBool(
    *this,
    "dtln.enable_ort_optimizations");
  config_.qos_depth = readRequiredInt(*this, "qos.depth");
  config_.qos_reliable = readRequiredBool(*this, "qos.reliable");
  config_.diagnostics_publish_period_ms = readRequiredInt(
    *this,
    "diagnostics.publish_period_ms");
  config_.diagnostics_qos_depth = readRequiredInt(*this, "diagnostics.qos.depth");
  config_.diagnostics_qos_reliable = readRequiredBool(*this, "diagnostics.qos.reliable");

  if (config_.input_topic.empty()) {
    throw std::runtime_error("input_topic is required (set via YAML)");
  }
  if (config_.output_topic.empty()) {
    throw std::runtime_error("output_topic is required (set via YAML)");
  }
  config_.resolved_input_topic =
    this->get_node_topics_interface()->resolve_topic_name(config_.input_topic);
  config_.resolved_output_topic =
    this->get_node_topics_interface()->resolve_topic_name(config_.output_topic);
  if (config_.resolved_input_topic == config_.resolved_output_topic) {
    throw std::runtime_error("resolved input_topic and output_topic must be distinct");
  }
  if (config_.expected_sample_rate != kRequiredSampleRate) {
    throw std::runtime_error(
            "fa_denoise requires expected_sample_rate=16000 by design (got " +
            std::to_string(config_.expected_sample_rate) + ")");
  }
  if (config_.expected_channels <= 0) {
    throw std::runtime_error("expected_channels must be > 0 (set via YAML)");
  }
  if (!backends::isSupportedAudioFormatPair(config_.expected_encoding, config_.expected_bit_depth)) {
    throw std::runtime_error(
      "expected.encoding/expected.bit_depth must be PCM16LE/16 or FLOAT32LE/32");
  }
  if (config_.qos_depth <= 0) {
    throw std::runtime_error("qos.depth must be > 0 (set via YAML)");
  }
  if (config_.diagnostics_publish_period_ms <= 0) {
    throw std::runtime_error("diagnostics.publish_period_ms must be > 0 (set via YAML)");
  }
  if (config_.diagnostics_qos_depth <= 0) {
    throw std::runtime_error("diagnostics.qos.depth must be > 0");
  }
  if (!backends::isSupportedAudioFormatPair(config_.output_encoding, config_.output_bit_depth)) {
    throw std::runtime_error(
      "output.encoding/output.bit_depth must be PCM16LE/16 or FLOAT32LE/32");
  }
  if (config_.backend_name.empty()) {
    throw std::runtime_error("backend.name is required (set via YAML)");
  }

  if (config_.backend_name != "passthrough" && config_.backend_name != "dtln_onnx") {
    throw std::runtime_error("backend.name must be passthrough or dtln_onnx");
  }

  if (config_.backend_name == "passthrough" &&
      (config_.output_encoding != config_.expected_encoding ||
       config_.output_bit_depth != config_.expected_bit_depth))
  {
    throw std::runtime_error(
      "fa_denoise passthrough requires output format to match expected input format");
  }

  if (config_.backend_name == "dtln_onnx") {
    if (config_.expected_channels != 1) {
      throw std::runtime_error("fa_denoise dtln_onnx requires expected_channels=1");
    }
    if (config_.dtln_block_len <= 0 || config_.dtln_block_shift <= 0) {
      throw std::runtime_error("dtln.block_len and dtln.block_shift must be > 0 (set via YAML)");
    }
    if (config_.dtln_block_shift > config_.dtln_block_len) {
      throw std::runtime_error("dtln.block_shift must be <= dtln.block_len");
    }
    if (config_.dtln_model_1_path.empty()) {
      throw std::runtime_error("dtln.model_1_path is required for dtln_onnx backend");
    }
    if (config_.dtln_model_2_path.empty()) {
      throw std::runtime_error("dtln.model_2_path is required for dtln_onnx backend");
    }
    if (config_.dtln_intra_op_num_threads <= 0 || config_.dtln_inter_op_num_threads <= 0) {
      throw std::runtime_error("dtln intra/inter op thread counts must be > 0 (set via YAML)");
    }
  }

  RCLCPP_INFO(this->get_logger(),
    "NS config: enabled=%s backend.name=%s input=%s output=%s expected_sr=%d expected_ch=%d "
    "expected_enc=%s expected_bits=%d out_enc=%s out_bits=%d qos_depth=%d reliable=%s",
    config_.enabled ? "true" : "false",
    config_.backend_name.c_str(),
    config_.input_topic.c_str(),
    config_.output_topic.c_str(),
    config_.expected_sample_rate,
    config_.expected_channels,
    config_.expected_encoding.c_str(),
    config_.expected_bit_depth,
    config_.output_encoding.c_str(),
    config_.output_bit_depth,
    config_.qos_depth,
    config_.qos_reliable ? "true" : "false");
}

void FaDenoiseNode::configureBackend()
{
  backends::AudioFormat expected_format;
  expected_format.sample_rate = config_.expected_sample_rate;
  expected_format.channels = config_.expected_channels;
  expected_format.encoding = config_.expected_encoding;
  expected_format.bit_depth = config_.expected_bit_depth;

  backends::AudioFormat output_format;
  output_format.sample_rate = config_.expected_sample_rate;
  output_format.channels = config_.expected_channels;
  output_format.encoding = config_.output_encoding;
  output_format.bit_depth = config_.output_bit_depth;

  if (config_.backend_name == backends::PassthroughBackend::kName) {
    backend_ = std::make_unique<backends::PassthroughBackend>(expected_format, output_format);
    return;
  }

  if (config_.backend_name != "dtln_onnx") {
    throw std::logic_error("fa_denoise backend invariant violated after startup validation");
  }

#ifdef FA_DENOISE_WITH_ONNXRUNTIME
  backends::DtlnOnnxConfig dtln_cfg;
  dtln_cfg.block_len = config_.dtln_block_len;
  dtln_cfg.block_shift = config_.dtln_block_shift;
  dtln_cfg.model_1_path = config_.dtln_model_1_path;
  dtln_cfg.model_2_path = config_.dtln_model_2_path;
  dtln_cfg.intra_op_num_threads = config_.dtln_intra_op_num_threads;
  dtln_cfg.inter_op_num_threads = config_.dtln_inter_op_num_threads;
  dtln_cfg.enable_ort_optimizations = config_.dtln_enable_ort_optimizations;

  backends::DtlnOnnxBackendConfig backend_config;
  backend_config.expected_format = expected_format;
  backend_config.output_format = output_format;
  backend_config.engine_config = std::move(dtln_cfg);
  backend_ = std::make_unique<backends::DtlnOnnxBackend>(backend_config);
#else
  throw std::runtime_error("fa_denoise was built without ONNX Runtime support (FA_DENOISE_WITH_ONNXRUNTIME=0)");
#endif
}

void FaDenoiseNode::setupInterfaces()
{
  rclcpp::QoS qos(static_cast<size_t>(config_.qos_depth));
  if (config_.qos_reliable) {
    qos.reliable();
  } else {
    qos.best_effort();
  }

  pub_ = this->create_publisher<fa_interfaces::msg::AudioFrame>(config_.output_topic, qos);
  sub_ = this->create_subscription<fa_interfaces::msg::AudioFrame>(
    config_.input_topic, qos,
    std::bind(&FaDenoiseNode::onAudioFrame, this, std::placeholders::_1));

  rclcpp::QoS diagnostics_qos(static_cast<size_t>(config_.diagnostics_qos_depth));
  if (config_.diagnostics_qos_reliable) {
    diagnostics_qos.reliable();
  } else {
    diagnostics_qos.best_effort();
  }

  diag_pub_ = this->create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
    "diagnostics", diagnostics_qos);
  diag_timer_ = this->create_wall_timer(
    std::chrono::milliseconds(config_.diagnostics_publish_period_ms),
    std::bind(&FaDenoiseNode::publishDiagnostics, this));
}

bool FaDenoiseNode::validateFrame(const fa_interfaces::msg::AudioFrame & msg) const
{
  if (!hasValidStamp(msg.header.stamp)) {
    return false;
  }
  if (msg.sample_rate != static_cast<uint32_t>(config_.expected_sample_rate)) {
    return false;
  }
  if (msg.channels != static_cast<uint32_t>(config_.expected_channels)) {
    return false;
  }
  if (msg.channels == 0 || msg.sample_rate == 0) {
    return false;
  }
  if (msg.encoding != config_.expected_encoding) {
    return false;
  }
  if (msg.bit_depth != static_cast<uint32_t>(config_.expected_bit_depth)) {
    return false;
  }
  if (msg.source_id.empty() || msg.stream_id.empty()) {
    return false;
  }
  if (msg.stream_id != config_.input_topic) {
    return false;
  }
  if (msg.layout != kInterleavedLayout) {
    return false;
  }
  if (msg.data.empty()) {
    return false;
  }
  if (!backends::isSupportedAudioFormatPair(msg.encoding, static_cast<int>(msg.bit_depth))) {
    return false;
  }
  const size_t bytes_per_sample = static_cast<size_t>(msg.bit_depth / 8);
  const size_t bytes_per_frame = static_cast<size_t>(msg.channels) * bytes_per_sample;
  if (bytes_per_frame == 0 || (msg.data.size() % bytes_per_frame) != 0) {
    return false;
  }
  return true;
}

void FaDenoiseNode::onAudioFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg)
{
  in_.fetch_add(1);
  if (!msg) {
    throw std::logic_error("fa_denoise received a null AudioFrame pointer");
  }
  if (!pub_) {
    throw std::logic_error("fa_denoise publisher is not initialized");
  }
  if (!backend_) {
    throw std::logic_error("fa_denoise backend is not initialized");
  }

  if (!config_.enabled) {
    drop_.fetch_add(1);
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping frame because fa_denoise is disabled; disable the system node instead");
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

  backends::AudioBuffer backend_input;
  backend_input.format.sample_rate = static_cast<int>(msg->sample_rate);
  backend_input.format.channels = static_cast<int>(msg->channels);
  backend_input.format.encoding = msg->encoding;
  backend_input.format.bit_depth = static_cast<int>(msg->bit_depth);
  backend_input.data = msg->data;

  backends::ProcessResult backend_result = backend_->process(backend_input);
  if (backend_result.status != backends::ProcessStatus::kOk) {
    drop_.fetch_add(1);
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping denoise frame because backend %s rejected input or output: %s",
      backend_->name(),
      backends::processStatusMessage(backend_result.status));
    return;
  }

  fa_interfaces::msg::AudioFrame out_msg;
  out_msg.header = msg->header;
  out_msg.source_id = msg->source_id;
  out_msg.stream_id = config_.output_topic;
  out_msg.encoding = backend_result.output.format.encoding;
  out_msg.sample_rate = static_cast<uint32_t>(backend_result.output.format.sample_rate);
  out_msg.channels = static_cast<uint32_t>(backend_result.output.format.channels);
  out_msg.bit_depth = static_cast<uint32_t>(backend_result.output.format.bit_depth);
  out_msg.layout = kInterleavedLayout;
  out_msg.data = std::move(backend_result.output.data);
  out_msg.epoch = msg->epoch;

  pub_->publish(out_msg);
  out_.fetch_add(1);

  const uint64_t elapsed_ns = nanosSince(start);
  process_ns_sum_.fetch_add(elapsed_ns);
  process_count_.fetch_add(1);
  updateMaxAtomic(process_ns_max_, elapsed_ns);
}

void FaDenoiseNode::publishDiagnostics()
{
  if (!diag_pub_) {
    throw std::logic_error("fa_denoise diagnostics publisher is not initialized");
  }
  if (!backend_) {
    throw std::logic_error("fa_denoise backend is not initialized");
  }

  diagnostic_msgs::msg::DiagnosticArray array_msg;
  array_msg.header.stamp = this->now();

  diagnostic_msgs::msg::DiagnosticStatus status;
  status.name = "fa_denoise";
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
  push_kv("backend.name", backend_->name());
  push_kv("input_topic", config_.input_topic);
  push_kv("output_topic", config_.output_topic);
  push_kv("expected_sample_rate", std::to_string(config_.expected_sample_rate));
  push_kv("expected_channels", std::to_string(config_.expected_channels));
  push_kv("expected.encoding", config_.expected_encoding);
  push_kv("expected.bit_depth", std::to_string(config_.expected_bit_depth));
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

  push_kv("backend.pending_samples", std::to_string(backend_->pendingInputSamples()));

  array_msg.status.push_back(status);
  diag_pub_->publish(array_msg);
}

}  // namespace fa_denoise

#include "fa_ducking/fa_ducking_node.hpp"

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

namespace fa_ducking
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

FaDuckingNode::FaDuckingNode()
: rclcpp::Node("fa_ducking")
{
  RCLCPP_INFO(this->get_logger(), "Starting FA Ducking node");
  loadParameters();
  setupInterfaces();
}

void FaDuckingNode::loadParameters()
{
  this->declare_parameter("program_topic", config_.program_topic);
  this->declare_parameter("sidechain_topic", config_.sidechain_topic);
  this->declare_parameter("output_topic", config_.output_topic);
  this->declare_parameter<double>("sidechain.threshold_rms", config_.sidechain_threshold_rms);
  this->declare_parameter<int>("sidechain.max_age_ms", config_.sidechain_max_age_ms);
  this->declare_parameter<double>("ducking.gain_db", config_.ducking_gain_db);
  this->declare_parameter<double>("ducking.attack_ms", config_.attack_ms);
  this->declare_parameter<double>("ducking.release_ms", config_.release_ms);
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

  config_.program_topic = this->get_parameter("program_topic").as_string();
  config_.sidechain_topic = this->get_parameter("sidechain_topic").as_string();
  config_.output_topic = this->get_parameter("output_topic").as_string();
  config_.sidechain_threshold_rms = this->get_parameter("sidechain.threshold_rms").as_double();
  config_.sidechain_max_age_ms = this->get_parameter("sidechain.max_age_ms").as_int();
  config_.ducking_gain_db = this->get_parameter("ducking.gain_db").as_double();
  config_.attack_ms = this->get_parameter("ducking.attack_ms").as_double();
  config_.release_ms = this->get_parameter("ducking.release_ms").as_double();
  config_.expected_sample_rate = this->get_parameter("expected.sample_rate").as_int();
  config_.expected_channels = this->get_parameter("expected.channels").as_int();
  config_.expected_encoding = this->get_parameter("expected.encoding").as_string();
  config_.expected_bit_depth = this->get_parameter("expected.bit_depth").as_int();
  config_.expected_layout = this->get_parameter("expected.layout").as_string();
  config_.qos_depth = this->get_parameter("qos.depth").as_int();
  config_.qos_reliable = this->get_parameter("qos.reliable").as_bool();
  config_.diagnostics_publish_period_ms =
    this->get_parameter("diagnostics.publish_period_ms").as_int();

  if (config_.program_topic.empty()) {
    throw std::runtime_error("program_topic is required");
  }
  if (config_.sidechain_topic.empty()) {
    throw std::runtime_error("sidechain_topic is required");
  }
  if (config_.output_topic.empty()) {
    throw std::runtime_error("output_topic is required");
  }
  if (!isFinite(config_.sidechain_threshold_rms) ||
      config_.sidechain_threshold_rms <= 0.0 ||
      config_.sidechain_threshold_rms > 1.0)
  {
    throw std::runtime_error("sidechain.threshold_rms must be finite and in (0.0, 1.0]");
  }
  if (config_.sidechain_max_age_ms <= 0) {
    throw std::runtime_error("sidechain.max_age_ms must be > 0");
  }
  if (!isFinite(config_.ducking_gain_db) ||
      config_.ducking_gain_db < -96.0 ||
      config_.ducking_gain_db >= 0.0)
  {
    throw std::runtime_error("ducking.gain_db must be finite and in [-96.0, 0.0)");
  }
  config_.ducking_gain_linear = dbToLinear(config_.ducking_gain_db);
  if (!isFinite(config_.ducking_gain_linear) ||
      config_.ducking_gain_linear <= 0.0 ||
      config_.ducking_gain_linear >= 1.0)
  {
    throw std::runtime_error("ducking.gain_db must resolve to a finite attenuation gain in (0.0, 1.0)");
  }
  if (!isFinite(config_.attack_ms) || config_.attack_ms <= 0.0) {
    throw std::runtime_error("ducking.attack_ms must be finite and > 0.0");
  }
  if (!isFinite(config_.release_ms) || config_.release_ms <= 0.0) {
    throw std::runtime_error("ducking.release_ms must be finite and > 0.0");
  }
  if (config_.expected_sample_rate <= 0) {
    throw std::runtime_error("expected.sample_rate must be > 0");
  }
  if (config_.expected_channels <= 0) {
    throw std::runtime_error("expected.channels must be > 0");
  }
  if (config_.expected_encoding != kEncodingFloat32) {
    throw std::runtime_error("fa_ducking requires expected.encoding=FLOAT32LE");
  }
  if (config_.expected_bit_depth != 32) {
    throw std::runtime_error("fa_ducking requires expected.bit_depth=32");
  }
  if (config_.expected_layout != kInterleavedLayout) {
    throw std::runtime_error("fa_ducking requires expected.layout=interleaved");
  }
  if (config_.qos_depth <= 0) {
    throw std::runtime_error("qos.depth must be > 0");
  }
  if (config_.diagnostics_publish_period_ms <= 0) {
    throw std::runtime_error("diagnostics.publish_period_ms must be > 0");
  }

  current_gain_.store(1.0);
  last_target_gain_.store(1.0);

  RCLCPP_INFO(
    this->get_logger(),
    "Ducking config: program=%s sidechain=%s output=%s threshold_rms=%f duck_gain_db=%f duck_gain=%f attack=%fms release=%fms sidechain_max_age=%dms expected=%dHz/%d/%s/%d qos_depth=%d reliable=%s diag=%dms",
    config_.program_topic.c_str(),
    config_.sidechain_topic.c_str(),
    config_.output_topic.c_str(),
    config_.sidechain_threshold_rms,
    config_.ducking_gain_db,
    config_.ducking_gain_linear,
    config_.attack_ms,
    config_.release_ms,
    config_.sidechain_max_age_ms,
    config_.expected_sample_rate,
    config_.expected_channels,
    config_.expected_encoding.c_str(),
    config_.expected_bit_depth,
    config_.qos_depth,
    config_.qos_reliable ? "true" : "false",
    config_.diagnostics_publish_period_ms);
}

void FaDuckingNode::setupInterfaces()
{
  rclcpp::QoS qos(std::max<int>(1, config_.qos_depth));
  if (config_.qos_reliable) {
    qos.reliable();
  } else {
    qos.best_effort();
  }

  audio_pub_ = this->create_publisher<fa_interfaces::msg::AudioFrame>(config_.output_topic, qos);
  program_sub_ = this->create_subscription<fa_interfaces::msg::AudioFrame>(
    config_.program_topic,
    qos,
    std::bind(&FaDuckingNode::handleProgramFrame, this, std::placeholders::_1));
  sidechain_sub_ = this->create_subscription<fa_interfaces::msg::AudioFrame>(
    config_.sidechain_topic,
    qos,
    std::bind(&FaDuckingNode::handleSidechainFrame, this, std::placeholders::_1));

  diag_pub_ = this->create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
    "diagnostics",
    rclcpp::SystemDefaultsQoS());
  diag_timer_ = this->create_wall_timer(
    std::chrono::milliseconds(config_.diagnostics_publish_period_ms),
    std::bind(&FaDuckingNode::publishDiagnostics, this));
}

void FaDuckingNode::handleProgramFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg)
{
  program_frames_in_.fetch_add(1);

  if (!msg) {
    program_frames_dropped_.fetch_add(1);
    return;
  }

  if (!validateFrame(*msg, config_.program_topic, "program")) {
    program_frames_dropped_.fetch_add(1);
    return;
  }

  fa_interfaces::msg::AudioFrame out;
  if (!applyDucking(*msg, out)) {
    program_frames_dropped_.fetch_add(1);
    return;
  }

  audio_pub_->publish(out);
  program_frames_out_.fetch_add(1);
}

void FaDuckingNode::handleSidechainFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg)
{
  sidechain_frames_in_.fetch_add(1);

  if (!msg) {
    sidechain_frames_dropped_.fetch_add(1);
    invalidateSidechainState();
    return;
  }

  if (!validateFrame(*msg, config_.sidechain_topic, "sidechain")) {
    sidechain_frames_dropped_.fetch_add(1);
    invalidateSidechainState();
    return;
  }

  std::vector<float> samples;
  if (!readSamples(*msg, samples, "sidechain")) {
    sidechain_frames_dropped_.fetch_add(1);
    invalidateSidechainState();
    return;
  }

  const double rms = calculateFrameRms(samples);
  if (!isFinite(rms) || rms < 0.0 || rms > 1.0) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping sidechain frame because RMS is outside normalized range");
    sidechain_frames_dropped_.fetch_add(1);
    invalidateSidechainState();
    return;
  }

  {
    std::lock_guard<std::mutex> lock(sidechain_mutex_);
    has_sidechain_ = true;
    latest_sidechain_rms_ = rms;
    latest_sidechain_received_at_ = this->now();
  }

  last_sidechain_rms_.store(rms);
  sidechain_frames_valid_.fetch_add(1);
}

bool FaDuckingNode::validateFrame(
  const fa_interfaces::msg::AudioFrame & msg,
  const std::string & expected_stream_id,
  const char * input_name)
{
  if (msg.source_id.empty() || msg.stream_id.empty()) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping %s frame because source_id and stream_id are required",
      input_name);
    return false;
  }
  if (msg.stream_id != expected_stream_id) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping %s frame because stream_id mismatch: %s != %s",
      input_name,
      msg.stream_id.c_str(),
      expected_stream_id.c_str());
    return false;
  }
  if (msg.layout != config_.expected_layout) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping %s frame because layout mismatch: %s != %s",
      input_name,
      msg.layout.c_str(),
      config_.expected_layout.c_str());
    return false;
  }
  if (msg.encoding != config_.expected_encoding || msg.bit_depth != static_cast<uint32_t>(config_.expected_bit_depth)) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping %s frame because encoding mismatch: %s/%u != %s/%d",
      input_name,
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
      "Dropping %s frame because format mismatch: frame=%uHz/%u config=%dHz/%d",
      input_name,
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
      "Dropping %s frame because data size is invalid for FLOAT32LE interleaved samples",
      input_name);
    return false;
  }
  return true;
}

bool FaDuckingNode::readSamples(
  const fa_interfaces::msg::AudioFrame & msg,
  std::vector<float> & samples,
  const char * input_name)
{
  const size_t sample_count = msg.data.size() / sizeof(float);
  samples.resize(sample_count);

  for (size_t i = 0; i < sample_count; ++i) {
    float sample = 0.0F;
    std::memcpy(&sample, msg.data.data() + (i * sizeof(float)), sizeof(float));
    if (!std::isfinite(sample) || sample < kMinNormalizedSample || sample > kMaxNormalizedSample) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 3000,
        "Dropping %s frame because input sample is outside normalized FLOAT32LE range",
        input_name);
      return false;
    }
    samples[i] = sample;
  }

  return true;
}

double FaDuckingNode::calculateFrameRms(const std::vector<float> & samples) const
{
  double square_sum = 0.0;
  for (const float sample : samples) {
    const double value = static_cast<double>(sample);
    square_sum += value * value;
  }

  const double mean_square = square_sum / static_cast<double>(samples.size());
  return std::sqrt(mean_square);
}

void FaDuckingNode::invalidateSidechainState()
{
  {
    std::lock_guard<std::mutex> lock(sidechain_mutex_);
    has_sidechain_ = false;
    latest_sidechain_rms_ = 0.0;
    latest_sidechain_received_at_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
  }

  last_sidechain_rms_.store(0.0);
  last_sidechain_age_ms_.store(-1);
  last_sidechain_active_.store(false);
  sidechain_state_invalidations_.fetch_add(1);
}

SidechainSnapshot FaDuckingNode::sidechainSnapshot() const
{
  std::lock_guard<std::mutex> lock(sidechain_mutex_);
  SidechainSnapshot snapshot;
  snapshot.available = has_sidechain_;
  snapshot.rms = latest_sidechain_rms_;
  snapshot.received_at = latest_sidechain_received_at_;
  return snapshot;
}

bool FaDuckingNode::sidechainIsActive(const rclcpp::Time & now)
{
  const SidechainSnapshot snapshot = sidechainSnapshot();
  if (!snapshot.available) {
    last_sidechain_age_ms_.store(-1);
    last_sidechain_active_.store(false);
    return false;
  }

  const int64_t age_ms = (now - snapshot.received_at).nanoseconds() / 1000000;
  last_sidechain_age_ms_.store(age_ms);
  if (age_ms < 0 || age_ms > config_.sidechain_max_age_ms) {
    stale_sidechain_checks_.fetch_add(1);
    last_sidechain_active_.store(false);
    return false;
  }

  const bool active = snapshot.rms >= config_.sidechain_threshold_rms;
  last_sidechain_active_.store(active);
  return active;
}

double FaDuckingNode::smoothingAlpha(double time_constant_ms, size_t sample_count) const
{
  const double frame_count =
    static_cast<double>(sample_count) / static_cast<double>(config_.expected_channels);
  const double frame_seconds = frame_count / static_cast<double>(config_.expected_sample_rate);
  const double time_constant_seconds = time_constant_ms / 1000.0;
  return 1.0 - std::exp(-frame_seconds / time_constant_seconds);
}

double FaDuckingNode::smoothedGain(double target_gain, size_t sample_count) const
{
  const double current_gain = current_gain_.load();
  const double time_constant_ms = target_gain < current_gain ? config_.attack_ms : config_.release_ms;
  const double alpha = smoothingAlpha(time_constant_ms, sample_count);
  const double next_gain = current_gain + (alpha * (target_gain - current_gain));

  if (next_gain < config_.ducking_gain_linear) {
    return config_.ducking_gain_linear;
  }
  if (next_gain > 1.0) {
    return 1.0;
  }
  return next_gain;
}

bool FaDuckingNode::applyDucking(
  const fa_interfaces::msg::AudioFrame & in,
  fa_interfaces::msg::AudioFrame & out)
{
  std::vector<float> samples;
  if (!readSamples(in, samples, "program")) {
    return false;
  }

  const bool active_sidechain = sidechainIsActive(this->now());
  const double target_gain = active_sidechain ? config_.ducking_gain_linear : 1.0;
  const double candidate_gain = smoothedGain(target_gain, samples.size());
  if (!isFinite(target_gain) || !isFinite(candidate_gain)) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Dropping program frame because ducking gain is not finite");
    return false;
  }

  out = in;
  out.stream_id = config_.output_topic;
  out.data.resize(in.data.size());

  for (size_t i = 0; i < samples.size(); ++i) {
    const double output = static_cast<double>(samples[i]) * candidate_gain;
    if (!isFinite(output) ||
        output < kMinNormalizedSample ||
        output > kMaxNormalizedSample)
    {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 3000,
        "Dropping program frame because ducking output is outside normalized FLOAT32LE range");
      return false;
    }

    const float out_sample = static_cast<float>(output);
    if (!std::isfinite(out_sample) ||
        out_sample < kMinNormalizedSample ||
        out_sample > kMaxNormalizedSample)
    {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 3000,
        "Dropping program frame because ducking output cannot be represented as normalized FLOAT32LE");
      return false;
    }
    std::memcpy(out.data.data() + (i * sizeof(float)), &out_sample, sizeof(float));
  }

  current_gain_.store(candidate_gain);
  last_target_gain_.store(target_gain);
  if (active_sidechain) {
    ducked_program_frames_.fetch_add(1);
  } else {
    released_program_frames_.fetch_add(1);
  }
  return true;
}

void FaDuckingNode::publishDiagnostics()
{
  diagnostic_msgs::msg::DiagnosticArray array_msg;
  array_msg.header.stamp = this->now();

  diagnostic_msgs::msg::DiagnosticStatus status;
  status.name = "fa_ducking";
  status.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
  status.message = "running";
  status.values.reserve(24);
  pushKeyValue(status, "program_topic", config_.program_topic);
  pushKeyValue(status, "sidechain_topic", config_.sidechain_topic);
  pushKeyValue(status, "output_topic", config_.output_topic);
  pushKeyValue(status, "sidechain_threshold_rms", std::to_string(config_.sidechain_threshold_rms));
  pushKeyValue(status, "sidechain_max_age_ms", std::to_string(config_.sidechain_max_age_ms));
  pushKeyValue(status, "ducking_gain_db", std::to_string(config_.ducking_gain_db));
  pushKeyValue(status, "ducking_gain_linear", std::to_string(config_.ducking_gain_linear));
  pushKeyValue(status, "attack_ms", std::to_string(config_.attack_ms));
  pushKeyValue(status, "release_ms", std::to_string(config_.release_ms));
  pushKeyValue(status, "current_gain", std::to_string(current_gain_.load()));
  pushKeyValue(status, "last_target_gain", std::to_string(last_target_gain_.load()));
  pushKeyValue(status, "last_sidechain_rms", std::to_string(last_sidechain_rms_.load()));
  pushKeyValue(status, "last_sidechain_age_ms", std::to_string(last_sidechain_age_ms_.load()));
  pushKeyValue(status, "last_sidechain_active", last_sidechain_active_.load() ? "true" : "false");
  pushKeyValue(status, "program_frames_in", std::to_string(program_frames_in_.load()));
  pushKeyValue(status, "program_frames_out", std::to_string(program_frames_out_.load()));
  pushKeyValue(status, "program_frames_dropped", std::to_string(program_frames_dropped_.load()));
  pushKeyValue(status, "sidechain_frames_in", std::to_string(sidechain_frames_in_.load()));
  pushKeyValue(status, "sidechain_frames_valid", std::to_string(sidechain_frames_valid_.load()));
  pushKeyValue(status, "sidechain_frames_dropped", std::to_string(sidechain_frames_dropped_.load()));
  pushKeyValue(status, "sidechain_state_invalidations", std::to_string(sidechain_state_invalidations_.load()));
  pushKeyValue(status, "ducked_program_frames", std::to_string(ducked_program_frames_.load()));
  pushKeyValue(status, "released_program_frames", std::to_string(released_program_frames_.load()));
  pushKeyValue(status, "stale_sidechain_checks", std::to_string(stale_sidechain_checks_.load()));

  array_msg.status.push_back(status);
  diag_pub_->publish(array_msg);
}

}  // namespace fa_ducking

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<fa_ducking::FaDuckingNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return EXIT_SUCCESS;
  } catch (const std::exception & e) {
    RCLCPP_FATAL(rclcpp::get_logger("fa_ducking"), "Exception: %s", e.what());
    rclcpp::shutdown();
    return EXIT_FAILURE;
  }
}

#include "fa_out/fa_out_node.hpp"

#include "fa_out/audio_config_validation.hpp"
#include "fa_out/backends/alsa_playback_backend.hpp"
#include "fa_out/backends/pcm_file_writer_backend.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <utility>

#include <rcl_interfaces/msg/parameter_descriptor.hpp>

namespace fa_out
{

namespace
{
constexpr const char * kEncodingPcm16 = "PCM16LE";
constexpr const char * kEncodingPcm32 = "PCM32LE";
constexpr const char * kEncodingFloat32 = "FLOAT32LE";
constexpr const char * kInterleavedLayout = "interleaved";
constexpr const char * kPlaybackCommandStop = "stop";
constexpr const char * kPlaybackCommandPause = "pause";
constexpr const char * kPlaybackCommandResume = "resume";

FaOutNode::BackendFactory defaultBackendFactory()
{
  return [](const backends::AlsaPlaybackConfig & backend_config) {
    return std::make_unique<backends::AlsaPlaybackBackend>(backend_config);
  };
}

bool isSupportedFileEncodingPair(const std::string & encoding, const uint32_t bit_depth)
{
  return (encoding == kEncodingPcm16 && bit_depth == 16u) ||
         (encoding == kEncodingPcm32 && bit_depth == 32u) ||
         (encoding == kEncodingFloat32 && bit_depth == 32u);
}

std::string removeLeadingSlashes(std::string value)
{
  while (!value.empty() && value.front() == '/') {
    value.erase(value.begin());
  }
  return value;
}

bool sameIdentityString(const std::string & left, const std::string & right)
{
  return left == right || removeLeadingSlashes(left) == removeLeadingSlashes(right);
}

bool isRequiredParameterSet(const rclcpp::Parameter & parameter)
{
  return parameter.get_type() != rclcpp::ParameterType::PARAMETER_NOT_SET;
}

rclcpp::Parameter getRequiredParameter(const rclcpp::Node & node, const std::string & name)
{
  rclcpp::Parameter parameter;
  if (!node.get_parameter(name, parameter) || !isRequiredParameterSet(parameter)) {
    throw std::invalid_argument(name + " is required");
  }
  return parameter;
}

std::string readRequiredString(const rclcpp::Node & node, const std::string & name)
{
  const rclcpp::Parameter parameter = getRequiredParameter(node, name);
  if (parameter.get_type() != rclcpp::ParameterType::PARAMETER_STRING) {
    throw std::invalid_argument(name + " must be a string");
  }
  return parameter.as_string();
}

int readRequiredInt(const rclcpp::Node & node, const std::string & name)
{
  const rclcpp::Parameter parameter = getRequiredParameter(node, name);
  if (parameter.get_type() != rclcpp::ParameterType::PARAMETER_INTEGER) {
    throw std::invalid_argument(name + " must be an integer");
  }
  const int64_t value = parameter.as_int();
  if (value < static_cast<int64_t>(std::numeric_limits<int>::min()) ||
      value > static_cast<int64_t>(std::numeric_limits<int>::max()))
  {
    throw std::invalid_argument(name + " is outside supported integer range");
  }
  return static_cast<int>(value);
}

bool readRequiredBool(const rclcpp::Node & node, const std::string & name)
{
  const rclcpp::Parameter parameter = getRequiredParameter(node, name);
  if (parameter.get_type() != rclcpp::ParameterType::PARAMETER_BOOL) {
    throw std::invalid_argument(name + " must be a bool");
  }
  return parameter.as_bool();
}

rcl_interfaces::msg::ParameterDescriptor dynamicParameterDescriptor()
{
  rcl_interfaces::msg::ParameterDescriptor descriptor;
  descriptor.dynamic_typing = true;
  return descriptor;
}
}

FaOutNode::FaOutNode(const rclcpp::NodeOptions & options)
: FaOutNode(options, defaultBackendFactory())
{
}

FaOutNode::FaOutNode(const rclcpp::NodeOptions & options, BackendFactory backend_factory)
: rclcpp::Node("fa_out", options), backend_factory_(std::move(backend_factory))
{
  RCLCPP_INFO(this->get_logger(), "Starting FA Out node");
  if (!backend_factory_) {
    throw std::invalid_argument("fa_out backend factory is required");
  }
  loadParameters();

  bytes_per_frame_ = validation::bytesPerFrame(config_.channels, config_.bit_depth);
  if (bytes_per_frame_ == 0) {
    throw std::runtime_error("Invalid audio configuration: bytes_per_frame is zero");
  }
  config_.playback_chunk_bytes = validation::bytesForFrames(
    "audio.chunk_duration_ms", config_.playback_chunk_frames, bytes_per_frame_);

  openBackend();

  // QoS は fa_tts 側と合わせ、reliable/best_effort をパラメータで切り替える。
  rclcpp::QoS audio_qos(config_.qos_depth);
  if (config_.qos_reliable) {
    audio_qos.reliable();
  } else {
    audio_qos.best_effort();
  }
  audio_sub_ = this->create_subscription<fa_interfaces::msg::AudioFrame>(
    config_.input_topic, audio_qos,
    std::bind(&FaOutNode::handleFrame, this, std::placeholders::_1));

  rclcpp::QoS lifecycle_qos(config_.lifecycle_qos_depth);
  if (config_.lifecycle_qos_reliable) {
    lifecycle_qos.reliable();
  } else {
    lifecycle_qos.best_effort();
  }
  playback_done_pub_ = this->create_publisher<fa_interfaces::msg::PlaybackDone>(
    config_.playback_done_topic, lifecycle_qos);

  playback_control_srv_ = this->create_service<fa_interfaces::srv::PlaybackControl>(
    config_.playback_control_service,
    std::bind(
      &FaOutNode::handlePlaybackControl, this,
      std::placeholders::_1, std::placeholders::_2));

  running_.store(true);
  playback_thread_ = std::thread(&FaOutNode::playbackThread, this);
}

FaOutNode::~FaOutNode()
{
  running_.store(false);
  queue_cv_.notify_all();
  if (playback_thread_.joinable()) {
    playback_thread_.join();
  }
  closeBackend();
}

bool FaOutNode::hasFatalError() const
{
  return fatal_error_.load();
}

void FaOutNode::loadParameters()
{
  this->declare_parameter<std::string>("backend.name");
  this->declare_parameter<std::string>("input_topic");
  this->declare_parameter<std::string>("input_stream_id");
  this->declare_parameter<std::string>("playback_done_topic");
  this->declare_parameter<std::string>("playback_control_service");
  const auto dynamic_parameter = dynamicParameterDescriptor();
  this->declare_parameter("audio.device_id", rclcpp::ParameterValue{}, dynamic_parameter);
  this->declare_parameter("file.path", rclcpp::ParameterValue{}, dynamic_parameter);
  this->declare_parameter("overwrite.enabled", rclcpp::ParameterValue{}, dynamic_parameter);
  this->declare_parameter("audio.alsa.buffer_frames", rclcpp::ParameterValue{}, dynamic_parameter);
  this->declare_parameter("audio.alsa.period_frames", rclcpp::ParameterValue{}, dynamic_parameter);
  this->declare_parameter<std::string>("audio.encoding");
  this->declare_parameter<int>("audio.sample_rate");
  this->declare_parameter<int>("audio.channels");
  this->declare_parameter<int>("audio.bit_depth");
  this->declare_parameter<int>("queue.max_frames");
  this->declare_parameter<int>("audio.chunk_duration_ms");
  this->declare_parameter<int>("audio.qos.depth");
  this->declare_parameter<bool>("audio.qos.reliable");
  this->declare_parameter<int>("lifecycle.qos.depth");
  this->declare_parameter<bool>("lifecycle.qos.reliable");

  config_.backend_name = readRequiredString(*this, "backend.name");
  config_.input_topic = readRequiredString(*this, "input_topic");
  config_.input_stream_id = readRequiredString(*this, "input_stream_id");
  config_.playback_done_topic = readRequiredString(*this, "playback_done_topic");
  config_.playback_control_service = readRequiredString(*this, "playback_control_service");
  config_.encoding = readRequiredString(*this, "audio.encoding");
  if (config_.backend_name.empty()) {
    throw std::invalid_argument("backend.name is required");
  }
  if (config_.backend_name != "alsa_playback" && config_.backend_name != "pcm_file_writer") {
    throw std::invalid_argument("unsupported fa_out backend.name: " + config_.backend_name);
  }
  if (config_.input_topic.empty()) {
    throw std::invalid_argument("input_topic is required");
  }
  if (config_.input_stream_id.empty()) {
    throw std::invalid_argument("input_stream_id is required");
  }
  if (config_.playback_done_topic.empty()) {
    throw std::invalid_argument("playback_done_topic is required");
  }
  if (config_.playback_control_service.empty()) {
    throw std::invalid_argument("playback_control_service is required");
  }
  const std::string resolved_input_topic =
    this->get_node_topics_interface()->resolve_topic_name(config_.input_topic);
  if (sameIdentityString(config_.input_stream_id, config_.input_topic) ||
      sameIdentityString(config_.input_stream_id, resolved_input_topic))
  {
    throw std::invalid_argument("input_stream_id must be distinct from ROS input_topic");
  }
  const std::string resolved_playback_done_topic =
    this->get_node_topics_interface()->resolve_topic_name(config_.playback_done_topic);
  if (sameIdentityString(config_.playback_done_topic, config_.input_topic) ||
      sameIdentityString(resolved_playback_done_topic, resolved_input_topic))
  {
    throw std::invalid_argument("playback_done_topic must be distinct from ROS input_topic");
  }
  config_.sample_rate = validation::requirePositiveUint32(
    "audio.sample_rate", readRequiredInt(*this, "audio.sample_rate"));
  config_.channels = validation::requirePositiveUint32(
    "audio.channels", readRequiredInt(*this, "audio.channels"));
  config_.bit_depth = validation::requirePositiveUint32(
    "audio.bit_depth", readRequiredInt(*this, "audio.bit_depth"));

  if (config_.backend_name == "alsa_playback") {
    config_.device_id = readRequiredString(*this, "audio.device_id");
    if (config_.device_id.empty()) {
      throw std::invalid_argument("audio.device_id is required for backend.name=alsa_playback");
    }
    if (config_.encoding != kEncodingPcm16) {
      throw std::invalid_argument("audio.encoding must be PCM16LE for backend.name=alsa_playback");
    }
    if (config_.bit_depth != 16) {
      throw std::invalid_argument("audio.bit_depth must be 16 for PCM16LE playback");
    }
    validation::requireRawAlsaHardwareSink(config_.device_id);
    config_.alsa_buffer_frames = validation::requirePositiveSize(
      "audio.alsa.buffer_frames", readRequiredInt(*this, "audio.alsa.buffer_frames"));
    config_.alsa_period_frames = validation::requirePositiveSize(
      "audio.alsa.period_frames", readRequiredInt(*this, "audio.alsa.period_frames"));
    if (config_.alsa_period_frames > config_.alsa_buffer_frames) {
      throw std::invalid_argument("audio.alsa.period_frames must be <= audio.alsa.buffer_frames");
    }
  } else if (config_.backend_name == "pcm_file_writer") {
    config_.file_path = readRequiredString(*this, "file.path");
    config_.overwrite_enabled = readRequiredBool(*this, "overwrite.enabled");
    if (config_.file_path.empty()) {
      throw std::invalid_argument("file.path is required for backend.name=pcm_file_writer");
    }
    if (!isSupportedFileEncodingPair(config_.encoding, config_.bit_depth)) {
      throw std::invalid_argument(
        "audio.encoding/audio.bit_depth must be one of PCM16LE/16, PCM32LE/32, FLOAT32LE/32");
    }
  }
  config_.max_queue_frames = validation::requirePositiveSize(
    "queue.max_frames", readRequiredInt(*this, "queue.max_frames"));

  const int64_t chunk_duration_ms_param =
    readRequiredInt(*this, "audio.chunk_duration_ms");
  config_.chunk_duration_ms = validation::requirePositiveUint32(
    "audio.chunk_duration_ms", chunk_duration_ms_param);
  if (chunk_duration_ms_param > 1000) {
    throw std::invalid_argument("audio.chunk_duration_ms must be <= 1000");
  }
  config_.playback_chunk_frames = validation::playbackChunkFrames(
    config_.sample_rate, config_.chunk_duration_ms);

  config_.qos_depth = validation::requirePositiveSize(
    "audio.qos.depth", readRequiredInt(*this, "audio.qos.depth"));
  config_.qos_reliable = readRequiredBool(*this, "audio.qos.reliable");
  config_.lifecycle_qos_depth = validation::requirePositiveSize(
    "lifecycle.qos.depth", readRequiredInt(*this, "lifecycle.qos.depth"));
  config_.lifecycle_qos_reliable = readRequiredBool(*this, "lifecycle.qos.reliable");

  RCLCPP_INFO(this->get_logger(),
    "Output config: backend.name=%s input_topic=%s input_stream_id=%s playback_done_topic=%s "
    "playback_control_service=%s sink=%s encoding=%s rate=%uHz channels=%u bits=%u queue=%zu "
    "chunk=%ums chunk_frames=%zu qos_depth=%zu reliable=%s lifecycle_qos_depth=%zu "
    "lifecycle_reliable=%s",
    config_.backend_name.c_str(), config_.input_topic.c_str(), config_.input_stream_id.c_str(),
    config_.playback_done_topic.c_str(), config_.playback_control_service.c_str(),
    configuredSinkLabel().c_str(), config_.encoding.c_str(), config_.sample_rate,
    config_.channels, config_.bit_depth, config_.max_queue_frames, config_.chunk_duration_ms,
    config_.playback_chunk_frames, config_.qos_depth, config_.qos_reliable ? "true" : "false",
    config_.lifecycle_qos_depth, config_.lifecycle_qos_reliable ? "true" : "false");
}

void FaOutNode::openBackend()
{
  closeBackend();

  std::unique_ptr<backends::SinkBackend> backend;
  if (config_.backend_name == "alsa_playback") {
    backends::AlsaPlaybackConfig backend_config;
    backend_config.device_id = config_.device_id;
    backend_config.encoding = config_.encoding;
    backend_config.sample_rate = config_.sample_rate;
    backend_config.channels = config_.channels;
    backend_config.bit_depth = config_.bit_depth;
    backend_config.buffer_frames = config_.alsa_buffer_frames;
    backend_config.period_frames = config_.alsa_period_frames;
    backend = backend_factory_(backend_config);
  } else if (config_.backend_name == "pcm_file_writer") {
    backends::PcmFileWriterConfig backend_config;
    backend_config.file_path = config_.file_path;
    backend_config.encoding = config_.encoding;
    backend_config.channels = config_.channels;
    backend_config.bit_depth = config_.bit_depth;
    backend_config.overwrite_enabled = config_.overwrite_enabled;
    backend = std::make_unique<backends::PcmFileWriterBackend>(backend_config);
  } else {
    throw std::runtime_error("unsupported fa_out backend.name: " + config_.backend_name);
  }
  if (!backend) {
    throw std::runtime_error("fa_out backend factory returned null backend");
  }
  try {
    const backends::SinkOpenInfo open_info = backend->open();
    for (const auto & info : open_info.info_messages) {
      RCLCPP_INFO(this->get_logger(), "%s", info.c_str());
    }
  } catch (const std::exception & e) {
    RCLCPP_ERROR(this->get_logger(), "Failed to open configured sink: %s", e.what());
    throw std::runtime_error(e.what());
  }
  std::lock_guard<std::mutex> lock(backend_mutex_);
  sink_backend_ = std::move(backend);
}

void FaOutNode::closeBackend()
{
  std::lock_guard<std::mutex> lock(backend_mutex_);
  if (sink_backend_) {
    sink_backend_->close();
    sink_backend_.reset();
  }
}

size_t FaOutNode::writeBackendFrames(const uint8_t * data, size_t frame_count)
{
  std::lock_guard<std::mutex> lock(backend_mutex_);
  if (!sink_backend_ || !sink_backend_->isOpen()) {
    throw std::runtime_error("configured sink closed while fa_out is running: " + configuredSinkLabel());
  }
  return sink_backend_->writeFrames(data, frame_count);
}

std::string FaOutNode::configuredSinkLabel() const
{
  if (config_.backend_name == "pcm_file_writer") {
    return config_.file_path;
  }
  return config_.device_id;
}

bool FaOutNode::isBackendRunning()
{
  std::string error_message;
  bool running = false;
  {
    std::lock_guard<std::mutex> lock(backend_mutex_);
    if (!sink_backend_) {
      error_message = "required sink backend missing while fa_out is running";
    } else {
      try {
        running = sink_backend_->isRunning();
      } catch (const std::exception & e) {
        error_message = e.what();
      }
    }
  }
  if (!error_message.empty()) {
    failClosed(error_message);
    return false;
  }
  return running;
}

void FaOutNode::failClosed(const std::string &reason)
{
  if (fatal_error_.exchange(true)) {
    return;
  }

  RCLCPP_FATAL(this->get_logger(), "Failing closed: %s", reason.c_str());
  running_.store(false);
  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    frame_queue_.clear();
  }
  closeBackend();
  queue_cv_.notify_all();
  rclcpp::shutdown();
}

void FaOutNode::handleFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg)
{
  if (fatal_error_.load()) {
    RCLCPP_ERROR_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Rejecting playback frame because fa_out is failing closed");
    return;
  }

  RCLCPP_DEBUG_THROTTLE(
    this->get_logger(), *this->get_clock(), 2000,
    "Frame received: %zu bytes, %uHz, %u ch, encoding=%s",
    msg->data.size(), msg->sample_rate, msg->channels, msg->encoding.c_str());

  if (!validateFrame(*msg)) {
    return;
  }

  QueuedFrame queued_frame;
  queued_frame.header = msg->header;
  queued_frame.request_id = msg->header.frame_id;
  queued_frame.epoch = msg->epoch;
  queued_frame.data = std::vector<uint8_t>(msg->data.begin(), msg->data.end());
  bool queue_overflow = false;
  bool stale_epoch = false;
  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    if (queued_frame.epoch < active_epoch_.load()) {
      stale_epoch = true;
    } else if (frame_queue_.size() >= config_.max_queue_frames) {
      queue_overflow = true;
    } else {
      frame_queue_.emplace_back(std::move(queued_frame));
    }
  }
  if (stale_epoch) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Rejecting stale playback frame epoch=%u active_epoch=%u",
      msg->epoch, active_epoch_.load());
    return;
  }
  if (queue_overflow) {
    failClosed(
      "playback queue exceeded queue.max_frames=" + std::to_string(config_.max_queue_frames) +
      " for required sink " + configuredSinkLabel());
    return;
  }
  queue_cv_.notify_one();
  RCLCPP_DEBUG_THROTTLE(
    this->get_logger(), *this->get_clock(), 2000,
    "Frame queued for playback");
}

bool FaOutNode::validateFrame(const fa_interfaces::msg::AudioFrame & msg)
{
  if (msg.source_id.empty() || msg.stream_id.empty()) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "AudioFrame source_id and stream_id are required");
    return false;
  }
  if (msg.stream_id != config_.input_stream_id) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "AudioFrame stream_id mismatch: %s != %s",
      msg.stream_id.c_str(),
      config_.input_stream_id.c_str());
    return false;
  }
  if (msg.layout != kInterleavedLayout) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Unsupported audio layout %s, expected %s", msg.layout.c_str(), kInterleavedLayout);
    return false;
  }
  if (msg.encoding != config_.encoding) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Unsupported encoding %s, expected %s", msg.encoding.c_str(), config_.encoding.c_str());
    return false;
  }
  if (msg.sample_rate != config_.sample_rate || msg.channels != config_.channels ||
    msg.bit_depth != config_.bit_depth) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Frame format mismatch: frame=%uHz/%u/%u config=%uHz/%u/%u",
      msg.sample_rate, msg.channels, msg.bit_depth,
      config_.sample_rate, config_.channels, config_.bit_depth);
    return false;
  }
  if (msg.data.empty()) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Empty audio frame data, dropping");
    return false;
  }
  if (bytes_per_frame_ > 0 && (msg.data.size() % bytes_per_frame_) != 0) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Invalid audio frame size: %zu (bytes_per_frame=%zu)",
      msg.data.size(), bytes_per_frame_);
    return false;
  }
  return true;
}

void FaOutNode::handlePlaybackControl(
  const std::shared_ptr<fa_interfaces::srv::PlaybackControl::Request> request,
  std::shared_ptr<fa_interfaces::srv::PlaybackControl::Response> response)
{
  if (fatal_error_.load()) {
    response->accepted = false;
    response->message = "fa_out is failing closed";
    response->active_epoch = active_epoch_.load();
    response->paused = paused_.load();
    return;
  }

  if (request->command == kPlaybackCommandStop) {
    uint32_t next_epoch = 0;
    bool epoch_overflow = false;
    {
      std::lock_guard<std::mutex> lock(queue_mutex_);
      const uint32_t current_epoch = active_epoch_.load();
      if (current_epoch == std::numeric_limits<uint32_t>::max()) {
        response->accepted = false;
        response->message = "playback epoch overflow";
        response->active_epoch = current_epoch;
        response->paused = paused_.load();
        epoch_overflow = true;
      } else {
        next_epoch = current_epoch + 1u;
        active_epoch_.store(next_epoch);
        paused_.store(false);
        frame_queue_.clear();
      }
    }
    if (epoch_overflow) {
      failClosed(response->message);
      queue_cv_.notify_all();
      return;
    }
    queue_cv_.notify_all();
    response->accepted = true;
    response->message = "playback stopped";
    response->active_epoch = next_epoch;
    response->paused = false;
    return;
  }

  if (request->command == kPlaybackCommandPause) {
    paused_.store(true);
    queue_cv_.notify_all();
    response->accepted = true;
    response->message = "playback paused";
    response->active_epoch = active_epoch_.load();
    response->paused = true;
    return;
  }

  if (request->command == kPlaybackCommandResume) {
    paused_.store(false);
    queue_cv_.notify_all();
    response->accepted = true;
    response->message = "playback resumed";
    response->active_epoch = active_epoch_.load();
    response->paused = false;
    return;
  }

  response->accepted = false;
  response->message = "unsupported playback control command: " + request->command;
  response->active_epoch = active_epoch_.load();
  response->paused = paused_.load();
}

void FaOutNode::publishPlaybackDone(const QueuedFrame & queued_frame)
{
  fa_interfaces::msg::PlaybackDone done_msg;
  done_msg.header = queued_frame.header;
  done_msg.request_id = queued_frame.request_id;
  done_msg.epoch = queued_frame.epoch;
  playback_done_pub_->publish(done_msg);
}

void FaOutNode::playbackThread()
{
  const size_t chunk_bytes = config_.playback_chunk_bytes;

  rclcpp::Rate idle_rate(50);
  while (running_.load()) {
    QueuedFrame queued_frame;
    bool has_frame = false;
    {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      queue_cv_.wait_for(lock, std::chrono::milliseconds(100), [this]() {
        return (!frame_queue_.empty() && !paused_.load()) || !running_.load();
      });
      if (!paused_.load() && !frame_queue_.empty()) {
        queued_frame = std::move(frame_queue_.front());
        frame_queue_.pop_front();
        has_frame = true;
      }
    }

    if (has_frame && !queued_frame.data.empty()) {
      size_t total_bytes = queued_frame.data.size();
      size_t bytes_written = 0;
      uint8_t * data_ptr = queued_frame.data.data();
      bool stale_epoch = false;

      while (bytes_written < total_bytes && running_.load()) {
        {
          std::unique_lock<std::mutex> lock(queue_mutex_);
          queue_cv_.wait(lock, [this, &queued_frame]() {
            return !running_.load() || !paused_.load() ||
                   queued_frame.epoch < active_epoch_.load();
          });
          if (queued_frame.epoch < active_epoch_.load()) {
            stale_epoch = true;
            break;
          }
        }
        if (!running_.load()) {
          break;
        }
        size_t write_bytes = std::min(chunk_bytes, total_bytes - bytes_written);
        size_t write_frames = write_bytes / bytes_per_frame_;

        try {
          const size_t frames_written = writeBackendFrames(
            data_ptr + bytes_written,
            write_frames);
          bytes_written += frames_written * bytes_per_frame_;
        } catch (const std::exception & e) {
          failClosed(e.what());
          break;
        }
      }

      if (fatal_error_.load()) {
        continue;
      }
      if (!stale_epoch && bytes_written == total_bytes && running_.load()) {
        publishPlaybackDone(queued_frame);
      }
    } else {
      if (isBackendRunning()) {
        idle_rate.sleep();
        continue;
      }
      idle_rate.sleep();
    }
  }
}

}  // namespace fa_out

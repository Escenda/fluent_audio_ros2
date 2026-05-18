#include "fa_out/fa_out_node.hpp"

#include "fa_out/audio_config_validation.hpp"
#include "fa_out/backends/alsa_playback_backend.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <stdexcept>
#include <utility>

namespace fa_out
{

namespace
{
constexpr const char * kInputTopic = "audio/output/frame";
constexpr const char * kEncodingPcm16 = "PCM16LE";
constexpr const char * kInterleavedLayout = "interleaved";

FaOutNode::BackendFactory defaultBackendFactory()
{
  return [](const backends::AlsaPlaybackConfig & backend_config) {
    return std::make_unique<backends::AlsaPlaybackBackend>(backend_config);
  };
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
    kInputTopic, audio_qos,
    std::bind(&FaOutNode::handleFrame, this, std::placeholders::_1));

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
  this->declare_parameter<std::string>("audio.device_id");
  this->declare_parameter<std::string>("audio.encoding");
  this->declare_parameter<int>("audio.sample_rate");
  this->declare_parameter<int>("audio.channels");
  this->declare_parameter<int>("audio.bit_depth");
  this->declare_parameter<int>("audio.alsa.buffer_frames");
  this->declare_parameter<int>("audio.alsa.period_frames");
  this->declare_parameter<int>("queue.max_frames");
  this->declare_parameter<int>("audio.chunk_duration_ms");
  this->declare_parameter<int>("audio.qos.depth");
  this->declare_parameter<bool>("audio.qos.reliable");

  config_.backend_name = this->get_parameter("backend.name").as_string();
  config_.device_id = this->get_parameter("audio.device_id").as_string();
  config_.encoding = this->get_parameter("audio.encoding").as_string();
  if (config_.backend_name.empty()) {
    throw std::invalid_argument("backend.name is required");
  }
  if (config_.backend_name != "alsa_playback") {
    throw std::invalid_argument("unsupported fa_out backend.name: " + config_.backend_name);
  }
  if (config_.device_id.empty()) {
    throw std::invalid_argument("audio.device_id is required for backend.name=alsa_playback");
  }
  if (config_.encoding != kEncodingPcm16) {
    throw std::invalid_argument("audio.encoding must be PCM16LE for backend.name=alsa_playback");
  }
  validation::requireRawAlsaHardwareSink(config_.device_id);
  config_.sample_rate = validation::requirePositiveUint32(
    "audio.sample_rate", this->get_parameter("audio.sample_rate").as_int());
  config_.channels = validation::requirePositiveUint32(
    "audio.channels", this->get_parameter("audio.channels").as_int());
  config_.bit_depth = validation::requirePositiveUint32(
    "audio.bit_depth", this->get_parameter("audio.bit_depth").as_int());
  if (config_.bit_depth != 16) {
    throw std::invalid_argument("audio.bit_depth must be 16 for PCM16LE playback");
  }
  config_.alsa_buffer_frames = validation::requirePositiveSize(
    "audio.alsa.buffer_frames", this->get_parameter("audio.alsa.buffer_frames").as_int());
  config_.alsa_period_frames = validation::requirePositiveSize(
    "audio.alsa.period_frames", this->get_parameter("audio.alsa.period_frames").as_int());
  if (config_.alsa_period_frames > config_.alsa_buffer_frames) {
    throw std::invalid_argument("audio.alsa.period_frames must be <= audio.alsa.buffer_frames");
  }
  config_.max_queue_frames = validation::requirePositiveSize(
    "queue.max_frames", this->get_parameter("queue.max_frames").as_int());

  const int64_t chunk_duration_ms_param =
    this->get_parameter("audio.chunk_duration_ms").as_int();
  config_.chunk_duration_ms = validation::requirePositiveUint32(
    "audio.chunk_duration_ms", chunk_duration_ms_param);
  if (chunk_duration_ms_param > 1000) {
    throw std::invalid_argument("audio.chunk_duration_ms must be <= 1000");
  }
  config_.playback_chunk_frames = validation::playbackChunkFrames(
    config_.sample_rate, config_.chunk_duration_ms);

  config_.qos_depth = validation::requirePositiveSize(
    "audio.qos.depth", this->get_parameter("audio.qos.depth").as_int());
  config_.qos_reliable = this->get_parameter("audio.qos.reliable").as_bool();

  RCLCPP_INFO(this->get_logger(),
    "Output config: backend.name=%s device=%s encoding=%s rate=%uHz channels=%u bits=%u queue=%zu "
    "chunk=%ums chunk_frames=%zu alsa_buffer=%zu alsa_period=%zu qos_depth=%zu reliable=%s",
    config_.backend_name.c_str(), config_.device_id.c_str(), config_.encoding.c_str(),
    config_.sample_rate, config_.channels, config_.bit_depth, config_.max_queue_frames,
    config_.chunk_duration_ms, config_.playback_chunk_frames, config_.alsa_buffer_frames,
    config_.alsa_period_frames, config_.qos_depth, config_.qos_reliable ? "true" : "false");
}

void FaOutNode::openBackend()
{
  closeBackend();

  backends::AlsaPlaybackConfig backend_config;
  backend_config.device_id = config_.device_id;
  backend_config.encoding = config_.encoding;
  backend_config.sample_rate = config_.sample_rate;
  backend_config.channels = config_.channels;
  backend_config.bit_depth = config_.bit_depth;
  backend_config.buffer_frames = config_.alsa_buffer_frames;
  backend_config.period_frames = config_.alsa_period_frames;

  auto backend = backend_factory_(backend_config);
  if (!backend) {
    throw std::runtime_error("fa_out backend factory returned null backend");
  }
  try {
    const backends::SinkOpenInfo open_info = backend->open();
    for (const auto & info : open_info.info_messages) {
      RCLCPP_INFO(this->get_logger(), "%s", info.c_str());
    }
  } catch (const std::exception & e) {
    RCLCPP_ERROR(this->get_logger(), "Failed to open ALSA playback device: %s", e.what());
    throw std::runtime_error("Failed to open ALSA playback device");
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
    throw std::runtime_error("ALSA playback handle closed while the configured sink is required");
  }
  return sink_backend_->writeFrames(data, frame_count);
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
  queued_frame.data = std::vector<uint8_t>(msg->data.begin(), msg->data.end());
  bool queue_overflow = false;
  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    if (frame_queue_.size() >= config_.max_queue_frames) {
      queue_overflow = true;
    } else {
      frame_queue_.emplace_back(std::move(queued_frame));
    }
  }
  if (queue_overflow) {
    failClosed(
      "playback queue exceeded queue.max_frames=" + std::to_string(config_.max_queue_frames) +
      " for required sink " + config_.device_id);
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
  if (msg.stream_id != kInputTopic) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "AudioFrame stream_id must match audio/output/frame");
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
        return !frame_queue_.empty() || !running_.load();
      });
      if (!frame_queue_.empty()) {
        queued_frame = std::move(frame_queue_.front());
        frame_queue_.pop_front();
        has_frame = true;
      }
    }

    if (has_frame && !queued_frame.data.empty()) {
      size_t total_bytes = queued_frame.data.size();
      size_t bytes_written = 0;
      uint8_t * data_ptr = queued_frame.data.data();

      while (bytes_written < total_bytes && running_.load()) {
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

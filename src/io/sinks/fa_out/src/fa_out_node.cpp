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
constexpr const char * kEncodingPcm16 = "PCM16LE";
constexpr const char * kInterleavedLayout = "interleaved";
}

FaOutNode::FaOutNode()
: rclcpp::Node("fa_out")
{
  RCLCPP_INFO(this->get_logger(), "Starting FA Out node");
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
    "audio/output/frame", audio_qos,
    std::bind(&FaOutNode::handleFrame, this, std::placeholders::_1));

  // 停止リクエスト subscription
  stop_sub_ = this->create_subscription<std_msgs::msg::Empty>(
    "audio/output/stop", 10,
    std::bind(&FaOutNode::handleStop, this, std::placeholders::_1));

  // 一時停止リクエスト subscription
  pause_sub_ = this->create_subscription<std_msgs::msg::Empty>(
    "audio/output/pause", 10,
    std::bind(&FaOutNode::handlePause, this, std::placeholders::_1));

  // 再開リクエスト subscription
  resume_sub_ = this->create_subscription<std_msgs::msg::Empty>(
    "audio/output/resume", 10,
    std::bind(&FaOutNode::handleResume, this, std::placeholders::_1));

  // 再生完了通知 publisher
  playback_done_pub_ = this->create_publisher<fa_interfaces::msg::PlaybackDone>(
    "audio/output/playback_done", 10);

  // 一時停止完了通知 publisher
  paused_pub_ = this->create_publisher<std_msgs::msg::Empty>(
    "audio/output/paused", 10);

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
  if (!backends::AlsaPlaybackBackend::isRawHardwareDevice(config_.device_id)) {
    throw std::invalid_argument(
      "audio.device_id must be an ALSA raw hardware id starting with hw: for backend.name=alsa_playback");
  }
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
    "Output config: backend=%s device=%s encoding=%s rate=%uHz channels=%u bits=%u queue=%zu "
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

  auto backend = std::make_unique<backends::AlsaPlaybackBackend>(backend_config);
  try {
    const backends::SinkOpenInfo open_info = backend->open();
    for (const auto & info : open_info.info_messages) {
      RCLCPP_INFO(this->get_logger(), "%s", info.c_str());
    }
    for (const auto & warning : open_info.warnings) {
      RCLCPP_WARN(this->get_logger(), "%s", warning.c_str());
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

bool FaOutNode::discardBackendBuffer(const char *operation)
{
  std::string error_message;
  {
    std::lock_guard<std::mutex> lock(backend_mutex_);
    if (!sink_backend_ || !sink_backend_->isOpen()) {
      error_message = std::string(operation) + " requested without an open ALSA playback device";
    } else {
      try {
        sink_backend_->discardBuffer(operation);
      } catch (const std::exception & e) {
        error_message = e.what();
      }
    }
  }
  if (!error_message.empty()) {
    failClosed(error_message);
    return false;
  }
  return true;
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

  // Barge-in 後の古いフレームをフィルタリング
  {
    std::lock_guard<std::mutex> lock(stop_time_mutex_);
    rclcpp::Time frame_time(msg->header.stamp);
    if (last_stop_time_.nanoseconds() > 0 && frame_time < last_stop_time_) {
      RCLCPP_WARN(this->get_logger(),
        "Dropping stale frame (frame_time < last_stop_time): %.3f < %.3f",
        frame_time.seconds(), last_stop_time_.seconds());
      return;
    }
  }

  // stop 後に遅れて到着した「旧リクエスト由来のフレーム」を破棄する。
  // publish 時刻 stamp では判別できないため epoch を使う。
  const uint32_t frame_epoch = msg->epoch;
  const uint32_t current_epoch = current_epoch_.load();
  if (frame_epoch != 0 && frame_epoch < current_epoch) {
    RCLCPP_WARN(this->get_logger(),
      "Dropping stale frame (epoch < current_epoch): %u < %u",
      static_cast<unsigned int>(frame_epoch),
      static_cast<unsigned int>(current_epoch));
    return;
  }

  QueuedFrame queued_frame;
  queued_frame.data = std::vector<uint8_t>(msg->data.begin(), msg->data.end());
  queued_frame.header = msg->header;
  queued_frame.epoch = msg->epoch;
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

void FaOutNode::handleStop(const std_msgs::msg::Empty::SharedPtr /*msg*/)
{
  RCLCPP_INFO(this->get_logger(), "Stop requested");

  const auto new_epoch = current_epoch_.fetch_add(1) + 1;
  RCLCPP_INFO(this->get_logger(), "Playback epoch advanced: %u", static_cast<unsigned int>(new_epoch));

  // 停止時刻を記録（この時刻より前のフレームは無視される）
  {
    std::lock_guard<std::mutex> lock(stop_time_mutex_);
    last_stop_time_ = this->now();
    RCLCPP_INFO(this->get_logger(), "Recording stop time: %.3f", last_stop_time_.seconds());
  }

  // 停止フラグを立てる
  stop_requested_.store(true);

  // キューをクリア
  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    frame_queue_.clear();
  }

  // ALSA バッファをドロップ（即座に停止）
  if (!discardBackendBuffer("stop")) {
    return;
  }

  // 停止フラグをリセット
  stop_requested_.store(false);

  // 一時停止も解除
  is_paused_.store(false);
  pause_requested_.store(false);

  RCLCPP_INFO(this->get_logger(), "Playback stopped");
}

void FaOutNode::handlePause(const std_msgs::msg::Empty::SharedPtr /*msg*/)
{
  if (is_paused_.load()) {
    RCLCPP_INFO(this->get_logger(), "Already paused, ignoring pause request");
    return;
  }

  RCLCPP_INFO(this->get_logger(), "Pause requested");
  pause_requested_.store(true);
  queue_cv_.notify_one();  // playbackThreadを起こす
}

void FaOutNode::handleResume(const std_msgs::msg::Empty::SharedPtr /*msg*/)
{
  if (!is_paused_.load()) {
    RCLCPP_DEBUG(this->get_logger(), "Not paused, ignoring resume request");
    return;
  }

  RCLCPP_INFO(this->get_logger(), "Resume requested");
  is_paused_.store(false);
  queue_cv_.notify_all();  // playbackThreadを起こす
}

bool FaOutNode::validateFrame(const fa_interfaces::msg::AudioFrame & msg)
{
  if (msg.source_id.empty() || msg.stream_id.empty()) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "AudioFrame source_id and stream_id are required");
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
  // チャンク単位で書くことで stop / pause の応答性を固定する。
  const size_t chunk_bytes = config_.playback_chunk_bytes;

  rclcpp::Rate idle_rate(50);
  while (running_.load()) {
    // 一時停止中なら待機
    if (is_paused_.load()) {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      queue_cv_.wait_for(lock, std::chrono::milliseconds(100), [this]() {
        return !is_paused_.load() || !running_.load() || stop_requested_.load();
      });
      continue;
    }

    // 一時停止リクエストがあれば即座に停止
    if (pause_requested_.load()) {
      // ALSAバッファをドロップして即座に停止
      if (!discardBackendBuffer("pause")) {
        continue;
      }
      is_paused_.store(true);
      pause_requested_.store(false);
      RCLCPP_INFO(this->get_logger(), "Playback paused immediately");
      paused_pub_->publish(std_msgs::msg::Empty());
      continue;
    }

    QueuedFrame queued_frame;
    bool has_frame = false;
    {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      queue_cv_.wait_for(lock, std::chrono::milliseconds(100), [this]() {
        return !frame_queue_.empty() || !running_.load() || pause_requested_.load();
      });
      if (!frame_queue_.empty() && !pause_requested_.load()) {
        queued_frame = std::move(frame_queue_.front());
        frame_queue_.pop_front();
        has_frame = true;
      }
    }

    if (has_frame && !queued_frame.data.empty()) {
      size_t total_bytes = queued_frame.data.size();
      size_t bytes_written = 0;
      uint8_t * data_ptr = queued_frame.data.data();
      bool interrupted = false;

      // チャンク単位で書き込み（停止/一時停止の応答性向上）
      while (bytes_written < total_bytes && running_.load()) {
        // 停止/一時停止チェック
        if (stop_requested_.load() || pause_requested_.load()) {
          interrupted = true;
          break;
        }

        // 今回書き込むバイト数（最大chunk_bytes）
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

      if (interrupted) {
        // 中断時: ALSAバッファをドロップして即停止
        if (!discardBackendBuffer("interruption")) {
          continue;
        }
        RCLCPP_INFO(this->get_logger(), "Playback interrupted, dropped remaining audio");
      } else if (bytes_written > 0) {
        // 正常完了: drain せずに即座に playback_done を送信
        // （次のフレームがあれば連続再生される）
        RCLCPP_INFO(this->get_logger(), "Playback done, publishing notification");
        fa_interfaces::msg::PlaybackDone done_msg;
        done_msg.header = queued_frame.header;
        done_msg.request_id = queued_frame.header.frame_id;
        done_msg.epoch = queued_frame.epoch;
        playback_done_pub_->publish(done_msg);
      }
    } else {
      // キューが空の場合のみ、残りのALSAバッファを再生完了待ち
      if (isBackendRunning()) {
        // まだ再生中なら少し待つ
        idle_rate.sleep();
        continue;
      }
      idle_rate.sleep();
    }
  }
}

}  // namespace fa_out

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<fa_out::FaOutNode>();
    rclcpp::spin(node);
    const bool fatal_error = node->hasFatalError();
    rclcpp::shutdown();
    return fatal_error ? EXIT_FAILURE : EXIT_SUCCESS;
  } catch (const std::exception & e) {
    RCLCPP_FATAL(rclcpp::get_logger("fa_out"), "Exception: %s", e.what());
    rclcpp::shutdown();
    return EXIT_FAILURE;
  }
}

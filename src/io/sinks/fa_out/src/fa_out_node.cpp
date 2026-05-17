#include "fa_out/fa_out_node.hpp"

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

bool isRawAlsaPlaybackDevice(const std::string & device_id)
{
  return device_id.rfind("hw:", 0) == 0;
}
}

FaOutNode::FaOutNode()
: rclcpp::Node("fa_out")
{
  RCLCPP_INFO(this->get_logger(), "Starting FA Out node");
  loadParameters();

  bytes_per_frame_ = config_.channels * (config_.bit_depth / 8);
  if (bytes_per_frame_ == 0) {
    throw std::runtime_error("Invalid audio configuration: bytes_per_frame is zero");
  }

  if (!openDevice()) {
    throw std::runtime_error("Failed to open ALSA playback device");
  }

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
  closeDevice();
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
  if (!isRawAlsaPlaybackDevice(config_.device_id)) {
    throw std::invalid_argument(
      "audio.device_id must be an ALSA raw hardware id starting with hw: for backend.name=alsa_playback");
  }
  config_.sample_rate = this->get_parameter("audio.sample_rate").as_int();
  config_.channels = this->get_parameter("audio.channels").as_int();
  config_.bit_depth = this->get_parameter("audio.bit_depth").as_int();
  if (config_.sample_rate == 0) {
    throw std::invalid_argument("audio.sample_rate must be > 0");
  }
  if (config_.channels == 0) {
    throw std::invalid_argument("audio.channels must be > 0");
  }
  if (config_.bit_depth != 16) {
    throw std::invalid_argument("audio.bit_depth must be 16 for PCM16LE playback");
  }
  const int64_t max_frames_param = this->get_parameter("queue.max_frames").as_int();
  if (max_frames_param <= 0) {
    throw std::invalid_argument("queue.max_frames must be > 0");
  }
  config_.max_queue_frames = static_cast<size_t>(max_frames_param);

  const int64_t chunk_duration_ms_param = this->get_parameter("audio.chunk_duration_ms").as_int();
  if (chunk_duration_ms_param <= 0) {
    throw std::invalid_argument("audio.chunk_duration_ms must be > 0");
  }
  if (chunk_duration_ms_param > 1000) {
    throw std::invalid_argument("audio.chunk_duration_ms must be <= 1000");
  }
  config_.chunk_duration_ms = static_cast<uint32_t>(chunk_duration_ms_param);

  const int64_t qos_depth_param = this->get_parameter("audio.qos.depth").as_int();
  if (qos_depth_param <= 0) {
    throw std::invalid_argument("audio.qos.depth must be > 0");
  }
  config_.qos_depth = static_cast<size_t>(qos_depth_param);
  config_.qos_reliable = this->get_parameter("audio.qos.reliable").as_bool();

  RCLCPP_INFO(this->get_logger(),
    "Output config: backend=%s device=%s encoding=%s rate=%uHz channels=%u bits=%u queue=%zu "
    "chunk=%ums qos_depth=%zu reliable=%s",
    config_.backend_name.c_str(), config_.device_id.c_str(), config_.encoding.c_str(),
    config_.sample_rate, config_.channels, config_.bit_depth, config_.max_queue_frames,
    config_.chunk_duration_ms, config_.qos_depth, config_.qos_reliable ? "true" : "false");
}

bool FaOutNode::openDevice()
{
  closeDevice();

  snd_pcm_t * handle = nullptr;
  int err = snd_pcm_open(&handle, config_.device_id.c_str(), SND_PCM_STREAM_PLAYBACK, 0);
  if (err < 0) {
    RCLCPP_ERROR(this->get_logger(), "snd_pcm_open failed: %s", snd_strerror(err));
    return false;
  }

  snd_pcm_format_t format = SND_PCM_FORMAT_S16_LE;

  // Use hardware parameters for more control over buffer settings
  snd_pcm_hw_params_t *hw_params;
  snd_pcm_hw_params_alloca(&hw_params);

  err = snd_pcm_hw_params_any(handle, hw_params);
  if (err < 0) {
    RCLCPP_ERROR(this->get_logger(), "snd_pcm_hw_params_any failed: %s", snd_strerror(err));
    snd_pcm_close(handle);
    return false;
  }

  err = snd_pcm_hw_params_set_rate_resample(handle, hw_params, 0);
  if (err < 0) {
    RCLCPP_ERROR(this->get_logger(), "Failed to disable ALSA software resampling: %s", snd_strerror(err));
    snd_pcm_close(handle);
    return false;
  }

  err = snd_pcm_hw_params_set_access(handle, hw_params, SND_PCM_ACCESS_RW_INTERLEAVED);
  if (err < 0) {
    RCLCPP_ERROR(this->get_logger(), "snd_pcm_hw_params_set_access failed: %s", snd_strerror(err));
    snd_pcm_close(handle);
    return false;
  }

  err = snd_pcm_hw_params_set_format(handle, hw_params, format);
  if (err < 0) {
    RCLCPP_ERROR(this->get_logger(), "snd_pcm_hw_params_set_format failed: %s", snd_strerror(err));
    snd_pcm_close(handle);
    return false;
  }

  err = snd_pcm_hw_params_set_channels(handle, hw_params, config_.channels);
  if (err < 0) {
    RCLCPP_ERROR(this->get_logger(), "snd_pcm_hw_params_set_channels failed: %s", snd_strerror(err));
    snd_pcm_close(handle);
    return false;
  }

  unsigned int rate = config_.sample_rate;
  err = snd_pcm_hw_params_set_rate(handle, hw_params, rate, 0);
  if (err < 0) {
    RCLCPP_ERROR(this->get_logger(), "snd_pcm_hw_params_set_rate failed: %s", snd_strerror(err));
    snd_pcm_close(handle);
    return false;
  }

  // Set larger buffer size to prevent XRUNs: 16384 frames = ~341ms at 48kHz
  snd_pcm_uframes_t buffer_size = 16384;
  err = snd_pcm_hw_params_set_buffer_size_near(handle, hw_params, &buffer_size);
  if (err < 0) {
    RCLCPP_ERROR(this->get_logger(), "snd_pcm_hw_params_set_buffer_size_near failed: %s", snd_strerror(err));
    snd_pcm_close(handle);
    return false;
  }

  // Set period size: buffer_size / 4
  snd_pcm_uframes_t period_size = buffer_size / 4;
  err = snd_pcm_hw_params_set_period_size_near(handle, hw_params, &period_size, 0);
  if (err < 0) {
    RCLCPP_ERROR(this->get_logger(), "snd_pcm_hw_params_set_period_size_near failed: %s", snd_strerror(err));
    snd_pcm_close(handle);
    return false;
  }

  err = snd_pcm_hw_params(handle, hw_params);
  if (err < 0) {
    RCLCPP_ERROR(this->get_logger(), "snd_pcm_hw_params failed: %s", snd_strerror(err));
    snd_pcm_close(handle);
    return false;
  }

  RCLCPP_INFO(this->get_logger(), "ALSA buffer: %lu frames, period: %lu frames",
              buffer_size, period_size);

  // Configure software parameters for better playback behavior
  snd_pcm_sw_params_t *sw_params;
  snd_pcm_sw_params_alloca(&sw_params);

  err = snd_pcm_sw_params_current(handle, sw_params);
  if (err < 0) {
    RCLCPP_WARN(this->get_logger(), "snd_pcm_sw_params_current failed: %s", snd_strerror(err));
  } else {
    // Start playback automatically when buffer is at least half full
    err = snd_pcm_sw_params_set_start_threshold(handle, sw_params, buffer_size / 2);
    if (err < 0) {
      RCLCPP_WARN(this->get_logger(), "snd_pcm_sw_params_set_start_threshold failed: %s", snd_strerror(err));
    }

    // Wake up the writer when at least one period of space is available
    err = snd_pcm_sw_params_set_avail_min(handle, sw_params, period_size);
    if (err < 0) {
      RCLCPP_WARN(this->get_logger(), "snd_pcm_sw_params_set_avail_min failed: %s", snd_strerror(err));
    }

    err = snd_pcm_sw_params(handle, sw_params);
    if (err < 0) {
      RCLCPP_WARN(this->get_logger(), "snd_pcm_sw_params failed: %s", snd_strerror(err));
    } else {
      RCLCPP_INFO(this->get_logger(), "ALSA software params: start_threshold=%lu frames", buffer_size / 2);
    }
  }

  pcm_handle_ = handle;
  return true;
}

void FaOutNode::closeDevice()
{
  if (pcm_handle_) {
    snd_pcm_close(pcm_handle_);
    pcm_handle_ = nullptr;
  }
}

bool FaOutNode::discardDeviceBuffer(const char *operation)
{
  if (!pcm_handle_) {
    failClosed(std::string(operation) + " requested without an open ALSA playback device");
    return false;
  }

  int err = snd_pcm_drop(pcm_handle_);
  if (err < 0) {
    failClosed(std::string("snd_pcm_drop failed during ") + operation + ": " + snd_strerror(err));
    return false;
  }

  err = snd_pcm_prepare(pcm_handle_);
  if (err < 0) {
    failClosed(std::string("snd_pcm_prepare failed during ") + operation + ": " + snd_strerror(err));
    return false;
  }
  return true;
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
  closeDevice();
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
  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    if (frame_queue_.size() >= config_.max_queue_frames) {
      frame_queue_.pop_front();
      auto & clk = *this->get_clock();
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), clk, 2000,
        "Playback queue overflow, dropping oldest frame");
    }
    frame_queue_.emplace_back(std::move(queued_frame));
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
  if (pcm_handle_) {
    if (!discardDeviceBuffer("stop")) {
      return;
    }
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
  // チャンクのサンプル数を計算（停止/一時停止の応答性向上用）
  const uint64_t chunk_samples_u64 =
    (static_cast<uint64_t>(config_.sample_rate) * config_.chunk_duration_ms) / 1000;
  const size_t chunk_samples = std::max<size_t>(1, static_cast<size_t>(chunk_samples_u64));
  const size_t chunk_bytes = chunk_samples * bytes_per_frame_;

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
      if (pcm_handle_) {
        if (!discardDeviceBuffer("pause")) {
          continue;
        }
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
      if (!pcm_handle_) {
        failClosed("ALSA playback handle closed while the configured sink is required");
        continue;
      }

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

        snd_pcm_sframes_t result = snd_pcm_writei(
          pcm_handle_,
          data_ptr + bytes_written,
          write_frames);

        if (result == -EPIPE) {
          failClosed("ALSA playback XRUN on required sink " + config_.device_id);
          break;
        } else if (result == -EAGAIN) {
          failClosed("snd_pcm_writei returned EAGAIN on required sink " + config_.device_id);
          break;
        } else if (result < 0) {
          failClosed(
            "snd_pcm_writei failed on required sink " + config_.device_id + ": " +
            snd_strerror(static_cast<int>(result)));
          break;
        } else if (result == 0) {
          failClosed("snd_pcm_writei wrote zero frames on required sink " + config_.device_id);
          break;
        }

        bytes_written += static_cast<size_t>(result) * bytes_per_frame_;
      }

      if (fatal_error_.load()) {
        continue;
      }

      if (interrupted) {
        // 中断時: ALSAバッファをドロップして即停止
        if (pcm_handle_) {
          if (!discardDeviceBuffer("interruption")) {
            continue;
          }
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
      if (pcm_handle_) {
        snd_pcm_state_t state = snd_pcm_state(pcm_handle_);
        if (state == SND_PCM_STATE_RUNNING) {
          // まだ再生中なら少し待つ
          idle_rate.sleep();
          continue;
        }
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

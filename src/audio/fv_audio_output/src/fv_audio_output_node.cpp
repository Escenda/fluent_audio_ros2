#include "fv_audio_output/fv_audio_output_node.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <utility>

namespace fv_audio_output
{

namespace
{
constexpr const char * kEncodingPcm16 = "PCM16LE";
}

FvAudioOutputNode::FvAudioOutputNode()
: rclcpp::Node("fv_audio_output")
{
  RCLCPP_INFO(this->get_logger(), "Starting FV Audio Output node");
  loadParameters();

  bytes_per_frame_ = config_.channels * (config_.bit_depth / 8);
  if (bytes_per_frame_ == 0) {
    throw std::runtime_error("Invalid audio configuration: bytes_per_frame is zero");
  }

  if (!openDevice()) {
    throw std::runtime_error("Failed to open ALSA playback device");
  }

  audio_sub_ = this->create_subscription<fv_audio::msg::AudioFrame>(
    "audio/output/frame", rclcpp::SensorDataQoS(),
    std::bind(&FvAudioOutputNode::handleFrame, this, std::placeholders::_1));

  running_.store(true);
  playback_thread_ = std::thread(&FvAudioOutputNode::playbackThread, this);
}

FvAudioOutputNode::~FvAudioOutputNode()
{
  running_.store(false);
  queue_cv_.notify_all();
  if (playback_thread_.joinable()) {
    playback_thread_.join();
  }
  closeDevice();
}

void FvAudioOutputNode::loadParameters()
{
  this->declare_parameter("audio.device_id", config_.device_id);
  this->declare_parameter<int>("audio.sample_rate", static_cast<int>(config_.sample_rate));
  this->declare_parameter<int>("audio.channels", static_cast<int>(config_.channels));
  this->declare_parameter<int>("audio.bit_depth", static_cast<int>(config_.bit_depth));
  this->declare_parameter<int>("queue.max_frames", static_cast<int>(config_.max_queue_frames));

  config_.device_id = this->get_parameter("audio.device_id").as_string();
  config_.sample_rate = this->get_parameter("audio.sample_rate").as_int();
  config_.channels = this->get_parameter("audio.channels").as_int();
  config_.bit_depth = this->get_parameter("audio.bit_depth").as_int();
  const int64_t max_frames_param = this->get_parameter("queue.max_frames").as_int();
  config_.max_queue_frames = static_cast<size_t>(std::max<int64_t>(1, max_frames_param));

  RCLCPP_INFO(this->get_logger(),
    "Output config: device=%s rate=%uHz channels=%u bits=%u queue=%zu",
    config_.device_id.c_str(), config_.sample_rate, config_.channels, config_.bit_depth,
    config_.max_queue_frames);
}

bool FvAudioOutputNode::openDevice()
{
  closeDevice();

  snd_pcm_t * handle = nullptr;
  int err = snd_pcm_open(&handle, config_.device_id.c_str(), SND_PCM_STREAM_PLAYBACK, 0);
  if (err < 0) {
    RCLCPP_ERROR(this->get_logger(), "snd_pcm_open failed: %s", snd_strerror(err));
    return false;
  }

  snd_pcm_format_t format = SND_PCM_FORMAT_UNKNOWN;
  if (config_.bit_depth == 16) {
    format = SND_PCM_FORMAT_S16_LE;
  } else if (config_.bit_depth == 32) {
    format = SND_PCM_FORMAT_S32_LE;
  }

  if (format == SND_PCM_FORMAT_UNKNOWN) {
    RCLCPP_ERROR(this->get_logger(), "Unsupported bit depth: %u", config_.bit_depth);
    snd_pcm_close(handle);
    return false;
  }

  err = snd_pcm_set_params(
    handle,
    format,
    SND_PCM_ACCESS_RW_INTERLEAVED,
    static_cast<unsigned int>(config_.channels),
    static_cast<unsigned int>(config_.sample_rate),
    1,
    200000);  // latency in usec

  if (err < 0) {
    RCLCPP_ERROR(this->get_logger(), "snd_pcm_set_params failed: %s", snd_strerror(err));
    snd_pcm_close(handle);
    return false;
  }

  pcm_handle_ = handle;
  return true;
}

void FvAudioOutputNode::closeDevice()
{
  if (pcm_handle_) {
    snd_pcm_drop(pcm_handle_);
    snd_pcm_close(pcm_handle_);
    pcm_handle_ = nullptr;
  }
}

void FvAudioOutputNode::handleFrame(const fv_audio::msg::AudioFrame::SharedPtr msg)
{
  if (!validateFrame(*msg)) {
    return;
  }

  std::vector<uint8_t> buffer(msg->data.begin(), msg->data.end());
  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    if (frame_queue_.size() >= config_.max_queue_frames) {
      frame_queue_.pop_front();
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 2000,
        "Playback queue overflow, dropping oldest frame");
    }
    frame_queue_.emplace_back(std::move(buffer));
  }
  queue_cv_.notify_one();
}

bool FvAudioOutputNode::validateFrame(const fv_audio::msg::AudioFrame & msg) const
{
  if (msg.encoding != kEncodingPcm16) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Unsupported encoding %s, expected %s", msg.encoding.c_str(), kEncodingPcm16);
    return false;
  }
  if (msg.sample_rate != config_.sample_rate || msg.channels != config_.channels ||
    msg.bit_depth != config_.bit_depth)
  {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Frame format mismatch: frame=%uHz/%u/%u config=%uHz/%u/%u",
      msg.sample_rate, msg.channels, msg.bit_depth,
      config_.sample_rate, config_.channels, config_.bit_depth);
    return false;
  }
  return true;
}

void FvAudioOutputNode::playbackThread()
{
  rclcpp::Rate idle_rate(50);
  while (running_.load()) {
    std::vector<uint8_t> frame;
    {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      queue_cv_.wait_for(lock, std::chrono::milliseconds(100), [this]() {
        return !frame_queue_.empty() || !running_.load();
      });
      if (!frame_queue_.empty()) {
        frame = std::move(frame_queue_.front());
        frame_queue_.pop_front();
      }
    }

    if (!frame.empty()) {
      if (!pcm_handle_) {
        if (!openDevice()) {
          RCLCPP_ERROR_THROTTLE(
            this->get_logger(), *this->get_clock(), 2000,
            "ALSA device unavailable, dropping frame");
          continue;
        }
      }

      size_t frames_total = frame.size() / bytes_per_frame_;
      size_t frames_written = 0;
      uint8_t * data_ptr = frame.data();

      while (frames_written < frames_total && running_.load()) {
        snd_pcm_sframes_t result = snd_pcm_writei(
          pcm_handle_,
          data_ptr + frames_written * bytes_per_frame_,
          frames_total - frames_written);

        if (result == -EPIPE) {
          RCLCPP_WARN(this->get_logger(), "XRUN detected, preparing device");
          snd_pcm_prepare(pcm_handle_);
          continue;
        } else if (result == -EAGAIN) {
          continue;
        } else if (result < 0) {
          RCLCPP_ERROR(this->get_logger(), "snd_pcm_writei failed: %s", snd_strerror(result));
          snd_pcm_prepare(pcm_handle_);
          break;
        }

        frames_written += static_cast<size_t>(result);
      }
    } else {
      idle_rate.sleep();
    }
  }
}

}  // namespace fv_audio_output

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<fv_audio_output::FvAudioOutputNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}

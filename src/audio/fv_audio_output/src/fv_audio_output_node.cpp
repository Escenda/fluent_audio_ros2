#include "fv_audio_output/fv_audio_output_node.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
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

  play_file_srv_ = this->create_service<fv_audio_output::srv::PlayFile>(
    "audio/output/play_file",
    std::bind(&FvAudioOutputNode::handlePlayFile, this,
      std::placeholders::_1, std::placeholders::_2));

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

  // Use hardware parameters for more control over buffer settings
  snd_pcm_hw_params_t *hw_params;
  snd_pcm_hw_params_alloca(&hw_params);

  err = snd_pcm_hw_params_any(handle, hw_params);
  if (err < 0) {
    RCLCPP_ERROR(this->get_logger(), "snd_pcm_hw_params_any failed: %s", snd_strerror(err));
    snd_pcm_close(handle);
    return false;
  }

  // Enable software resampling for better stability
  err = snd_pcm_hw_params_set_rate_resample(handle, hw_params, 1);
  if (err < 0) {
    RCLCPP_WARN(this->get_logger(), "snd_pcm_hw_params_set_rate_resample failed: %s", snd_strerror(err));
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
  err = snd_pcm_hw_params_set_rate_near(handle, hw_params, &rate, 0);
  if (err < 0) {
    RCLCPP_ERROR(this->get_logger(), "snd_pcm_hw_params_set_rate_near failed: %s", snd_strerror(err));
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

void FvAudioOutputNode::handlePlayFile(
  const std::shared_ptr<fv_audio_output::srv::PlayFile::Request> request,
  std::shared_ptr<fv_audio_output::srv::PlayFile::Response> response)
{
  std::vector<uint8_t> wav_data;
  uint32_t sample_rate, channels, bit_depth;

  if (!loadWavFile(request->file_path, wav_data, sample_rate, channels, bit_depth)) {
    response->success = false;
    response->message = "Failed to load WAV file: " + request->file_path;
    RCLCPP_ERROR(this->get_logger(), "%s", response->message.c_str());
    return;
  }

  if (sample_rate != config_.sample_rate || channels != config_.channels ||
    bit_depth != config_.bit_depth)
  {
    response->success = false;
    response->message = "WAV format mismatch: file=" +
      std::to_string(sample_rate) + "Hz/" + std::to_string(channels) + "ch/" +
      std::to_string(bit_depth) + "bit, expected=" +
      std::to_string(config_.sample_rate) + "Hz/" + std::to_string(config_.channels) + "ch/" +
      std::to_string(config_.bit_depth) + "bit";
    RCLCPP_ERROR(this->get_logger(), "%s", response->message.c_str());
    return;
  }

  if (request->volume_scale > 0.0f && request->volume_scale != 1.0f) {
    applyVolumeScale(wav_data, request->volume_scale);
  }

  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    if (request->interrupt) {
      frame_queue_.clear();
    }
    if (frame_queue_.size() >= config_.max_queue_frames) {
      frame_queue_.pop_front();
      RCLCPP_WARN(this->get_logger(), "Queue overflow, dropping oldest frame for file playback");
    }
    frame_queue_.emplace_back(std::move(wav_data));
  }
  queue_cv_.notify_one();

  response->success = true;
  response->message = "File queued for playback";
  RCLCPP_INFO(this->get_logger(), "Queued audio file: %s", request->file_path.c_str());
}

bool FvAudioOutputNode::loadWavFile(const std::string & file_path, std::vector<uint8_t> & out_data,
  uint32_t & out_sample_rate, uint32_t & out_channels, uint32_t & out_bit_depth)
{
  std::ifstream file(file_path, std::ios::binary);
  if (!file) {
    RCLCPP_ERROR(this->get_logger(), "Cannot open file: %s", file_path.c_str());
    return false;
  }

  // Read RIFF header
  char riff[4];
  file.read(riff, 4);
  if (std::strncmp(riff, "RIFF", 4) != 0) {
    RCLCPP_ERROR(this->get_logger(), "Not a valid RIFF file");
    return false;
  }

  uint32_t file_size;
  file.read(reinterpret_cast<char*>(&file_size), 4);

  char wave[4];
  file.read(wave, 4);
  if (std::strncmp(wave, "WAVE", 4) != 0) {
    RCLCPP_ERROR(this->get_logger(), "Not a valid WAVE file");
    return false;
  }

  // Find fmt chunk
  bool found_fmt = false;
  uint16_t audio_format = 0;
  uint32_t byte_rate = 0, block_align = 0;

  while (file) {
    char chunk_id[4];
    file.read(chunk_id, 4);
    if (!file) break;

    uint32_t chunk_size;
    file.read(reinterpret_cast<char*>(&chunk_size), 4);

    if (std::strncmp(chunk_id, "fmt ", 4) == 0) {
      file.read(reinterpret_cast<char*>(&audio_format), 2);
      file.read(reinterpret_cast<char*>(&out_channels), 2);
      file.read(reinterpret_cast<char*>(&out_sample_rate), 4);
      file.read(reinterpret_cast<char*>(&byte_rate), 4);
      file.read(reinterpret_cast<char*>(&block_align), 2);
      file.read(reinterpret_cast<char*>(&out_bit_depth), 2);

      // Skip remaining fmt chunk
      file.seekg(chunk_size - 16, std::ios::cur);
      found_fmt = true;
    } else if (std::strncmp(chunk_id, "data", 4) == 0) {
      if (!found_fmt) {
        RCLCPP_ERROR(this->get_logger(), "data chunk before fmt chunk");
        return false;
      }

      out_data.resize(chunk_size);
      file.read(reinterpret_cast<char*>(out_data.data()), chunk_size);

      if (audio_format != 1) {  // PCM
        RCLCPP_ERROR(this->get_logger(), "Unsupported audio format: %u (expected PCM=1)", audio_format);
        return false;
      }

      return true;
    } else {
      // Skip unknown chunk
      file.seekg(chunk_size, std::ios::cur);
    }
  }

  RCLCPP_ERROR(this->get_logger(), "No data chunk found in WAV file");
  return false;
}

void FvAudioOutputNode::applyVolumeScale(std::vector<uint8_t> & data, float volume_scale)
{
  if (config_.bit_depth == 16) {
    int16_t * samples = reinterpret_cast<int16_t*>(data.data());
    size_t num_samples = data.size() / 2;
    for (size_t i = 0; i < num_samples; ++i) {
      int32_t scaled = static_cast<int32_t>(samples[i] * volume_scale);
      samples[i] = static_cast<int16_t>(std::clamp(scaled, -32768, 32767));
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

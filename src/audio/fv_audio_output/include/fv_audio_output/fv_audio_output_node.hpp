#pragma once

#include <alsa/asoundlib.h>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include "fv_audio/msg/audio_frame.hpp"

namespace fv_audio_output
{

struct OutputConfig
{
  std::string device_id = "default";
  uint32_t sample_rate = 48000;
  uint32_t channels = 1;
  uint32_t bit_depth = 16;
  size_t max_queue_frames = 8;
};

class FvAudioOutputNode : public rclcpp::Node
{
public:
  FvAudioOutputNode();
  ~FvAudioOutputNode() override;

private:
  void loadParameters();
  bool openDevice();
  void closeDevice();
  void playbackThread();
  void handleFrame(const fv_audio::msg::AudioFrame::SharedPtr msg);
  bool validateFrame(const fv_audio::msg::AudioFrame & msg) const;

  OutputConfig config_;
  snd_pcm_t * pcm_handle_ = nullptr;
  size_t bytes_per_frame_ = 0;

  std::mutex queue_mutex_;
  std::condition_variable queue_cv_;
  std::deque<std::vector<uint8_t>> frame_queue_;
  std::thread playback_thread_;
  std::atomic<bool> running_{false};

  rclcpp::Subscription<fv_audio::msg::AudioFrame>::SharedPtr audio_sub_;
};

}  // namespace fv_audio_output

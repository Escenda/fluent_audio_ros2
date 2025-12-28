#pragma once

#include <alsa/asoundlib.h>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "diagnostic_msgs/msg/diagnostic_status.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"
#include "std_msgs/msg/header.hpp"

#include "fv_audio/msg/audio_frame.hpp"
#include "fv_audio/srv/list_devices.hpp"
#include "fv_audio/srv/switch_device.hpp"
#include "fv_audio/srv/record.hpp"

namespace fv_audio
{

struct AudioConfig
{
  std::string device_mode{"auto"};
  std::string device_identifier{};
  int device_index{-1};
  uint32_t sample_rate{48000};
  uint32_t channels{1};
  uint32_t bit_depth{16};
  uint32_t chunk_ms{20};
  std::string encoding{"pcm16"};
  double gain_db{0.0};
  bool auto_attach{true};
  bool publish_waveform{false};
  uint32_t diag_period_ms{1000};
};

class FvAudioNode : public rclcpp::Node
{
public:
  FvAudioNode();
  ~FvAudioNode() override;

private:
  void loadParameters();
  void initializePortAudio();
  void shutdownPortAudio();
  bool configureDevice();
  bool reopenStream(const std::string &device_id);
  void startCaptureThread();
  void stopCaptureThread();
  void captureLoop();
  void publishFrame(const uint8_t *data, size_t data_size, double rms, double peak);
  void publishDiagnostics();
  double computeRms(const void *data, size_t samples, double &peak) const;
  bool determineDeviceFromConfig(std::string &device_id, std::string &device_name);
  std::vector<std::pair<std::string, std::string>> enumerateCaptureDevices() const;

  // Services
  void handleListDevices(
    const std::shared_ptr<fv_audio::srv::ListDevices::Request> request,
    std::shared_ptr<fv_audio::srv::ListDevices::Response> response);
  void handleSwitchDevice(
    const std::shared_ptr<fv_audio::srv::SwitchDevice::Request> request,
    std::shared_ptr<fv_audio::srv::SwitchDevice::Response> response);
  void handleRecord(
    const std::shared_ptr<fv_audio::srv::Record::Request> request,
    std::shared_ptr<fv_audio::srv::Record::Response> response);

  bool startRecordingToFile(const std::string &path);
  bool stopRecordingToFile();
  void appendRecordingData(const uint8_t *data, size_t size);
  void writeWavHeader(std::fstream &stream, uint32_t data_length);
  void finalizeWavHeader(std::fstream &stream, uint32_t data_length);

  AudioConfig config_;

  rclcpp::Publisher<fv_audio::msg::AudioFrame>::SharedPtr audio_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr levels_pub_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diag_pub_;

  rclcpp::Service<fv_audio::srv::ListDevices>::SharedPtr list_devices_srv_;
  rclcpp::Service<fv_audio::srv::SwitchDevice>::SharedPtr switch_device_srv_;
  rclcpp::Service<fv_audio::srv::Record>::SharedPtr record_srv_;

  rclcpp::TimerBase::SharedPtr diag_timer_;

  std::atomic<bool> capturing_{false};
  std::thread capture_thread_;
  snd_pcm_t *pcm_handle_{nullptr};
  std::string active_device_id_;
  std::string active_device_name_;
  size_t frames_per_buffer_{0};
  size_t bytes_per_frame_{0};
  double gain_linear_{1.0};

  std::atomic<uint64_t> xruns_{0};
  std::atomic<uint64_t> frames_published_{0};
  std::chrono::steady_clock::time_point last_frame_time_;

  std::mutex record_mutex_;
  std::fstream record_stream_;
  bool recording_{false};
  uint32_t recorded_bytes_{0};
  std::string record_path_;
};

}  // namespace fv_audio

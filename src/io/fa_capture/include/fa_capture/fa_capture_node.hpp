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

#include "fa_interfaces/msg/audio_frame.hpp"
#include "fa_interfaces/srv/list_devices.hpp"
#include "fa_interfaces/srv/switch_device.hpp"

namespace fa_capture
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
  bool publish_waveform{false};
  uint32_t diag_period_ms{1000};
};

class FaCaptureNode : public rclcpp::Node
{
public:
  FaCaptureNode();
  ~FaCaptureNode() override;

private:
  void loadParameters();
  void initializeAlsa();
  void shutdownAlsa();
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
    const std::shared_ptr<fa_interfaces::srv::ListDevices::Request> request,
    std::shared_ptr<fa_interfaces::srv::ListDevices::Response> response);
  void handleSwitchDevice(
    const std::shared_ptr<fa_interfaces::srv::SwitchDevice::Request> request,
    std::shared_ptr<fa_interfaces::srv::SwitchDevice::Response> response);

  AudioConfig config_;

  rclcpp::Publisher<fa_interfaces::msg::AudioFrame>::SharedPtr audio_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr levels_pub_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diag_pub_;

  rclcpp::Service<fa_interfaces::srv::ListDevices>::SharedPtr list_devices_srv_;
  rclcpp::Service<fa_interfaces::srv::SwitchDevice>::SharedPtr switch_device_srv_;

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
};

}  // namespace fa_capture

#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "diagnostic_msgs/msg/diagnostic_status.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/header.hpp"

#include "fa_interfaces/msg/audio_frame.hpp"
#include "fa_interfaces/srv/list_devices.hpp"
#include "fa_interfaces/srv/switch_device.hpp"
#include "fa_in/backends/factory.hpp"
#include "fa_in/backends/source_backend.hpp"

namespace fa_in
{

struct AudioConfig
{
  std::string backend_name{};
  std::string output_topic{};
  std::string device_mode{};
  std::string device_identifier{};
  int device_index{};
  std::string file_path{};
  std::string endpoint_uri{};
  std::string transport_identity{};
  std::string source_id{};
  bool playback_loop{false};
  uint32_t sample_rate{0};
  uint32_t channels{0};
  uint32_t bit_depth{0};
  uint32_t chunk_ms{0};
  std::string encoding{};
  std::string stream_id{};
  std::string layout{};
  uint32_t audio_qos_depth{0};
  bool audio_qos_reliable{false};
  uint32_t diagnostics_qos_depth{0};
  bool diagnostics_qos_reliable{false};
  uint32_t diag_period_ms{0};
  uint32_t network_max_packet_bytes{0};
  uint32_t polling_period_ms{0};
  uint32_t network_source_timeout_ms{0};
};

class FaInNode : public rclcpp::Node
{
public:
  using BackendFactory = fa_in::backends::SourceBackendFactory;

  explicit FaInNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  FaInNode(const rclcpp::NodeOptions & options, BackendFactory backend_factory);
  ~FaInNode() override;
  bool hasFatalError() const;

private:
  void loadParameters();
  void initializeBackend();
  void shutdownBackend();
  void failClosed(const std::string &reason);
  bool configureDevice();
  bool reopenStream(const std::string &device_id);
  void startCaptureThread();
  void stopCaptureThread();
  void captureLoop();
  void publishFrame(const uint8_t *data, size_t data_size);
  void publishDiagnostics();
  fa_in::backends::DeviceInfo determineDeviceFromConfig();
  std::vector<fa_in::backends::DeviceInfo> enumerateCaptureDevices() const;

  // Services
  void handleListDevices(
    const std::shared_ptr<fa_interfaces::srv::ListDevices::Request> request,
    std::shared_ptr<fa_interfaces::srv::ListDevices::Response> response);
  void handleSwitchDevice(
    const std::shared_ptr<fa_interfaces::srv::SwitchDevice::Request> request,
    std::shared_ptr<fa_interfaces::srv::SwitchDevice::Response> response);

  AudioConfig config_;
  BackendFactory backend_factory_;

  rclcpp::Publisher<fa_interfaces::msg::AudioFrame>::SharedPtr audio_pub_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diag_pub_;

  rclcpp::Service<fa_interfaces::srv::ListDevices>::SharedPtr list_devices_srv_;
  rclcpp::Service<fa_interfaces::srv::SwitchDevice>::SharedPtr switch_device_srv_;

  rclcpp::TimerBase::SharedPtr diag_timer_;

  std::unique_ptr<fa_in::backends::SourceBackend> source_backend_;
  std::atomic<bool> capturing_{false};
  std::atomic<bool> fatal_error_{false};
  std::thread capture_thread_;
  std::string active_device_id_;
  std::string active_device_name_;
  std::string active_source_id_;
  size_t frames_per_buffer_{0};
  size_t bytes_per_frame_{0};
  size_t bytes_per_buffer_{0};

  std::atomic<uint64_t> xruns_{0};
  std::atomic<uint64_t> frames_published_{0};
  std::chrono::steady_clock::time_point last_frame_time_;
};

}  // namespace fa_in

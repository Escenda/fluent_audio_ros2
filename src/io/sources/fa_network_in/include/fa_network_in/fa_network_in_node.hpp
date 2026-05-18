#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>

#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "fa_interfaces/msg/audio_frame.hpp"
#include "fa_network_in/backends/network_pcm_receiver_backend.hpp"

namespace fa_network_in
{

struct FaNetworkInConfig
{
  std::string backend_name{};
  std::string endpoint_uri{};
  std::string transport_identity{};
  std::string source_id{};
  std::string stream_id{};
  std::string output_topic{};
  int expected_sample_rate{-1};
  int expected_channels{-1};
  std::string expected_encoding{};
  int expected_bit_depth{-1};
  std::string expected_layout{};
  int max_packet_bytes{-1};
  int polling_period_ms{-1};
  int qos_depth{-1};
  bool qos_reliable{false};
  int diagnostics_publish_period_ms{-1};
  int diagnostics_qos_depth{-1};
  bool diagnostics_qos_reliable{false};
};

/**
 * @brief 明示 UDP endpoint で受け取った raw PCM packet を AudioFrame として publish する source adapter。
 */
class FaNetworkInNode : public rclcpp::Node
{
public:
  explicit FaNetworkInNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~FaNetworkInNode() override;

  bool hasFatalError() const;

private:
  void loadParameters();
  void validateConfig() const;
  void openEndpoint();
  void setupInterfaces();
  void pollEndpoint();
  void publishDiagnostics();
  void failClosed(const std::string & reason);

  size_t bytesPerFrame() const;
  fa_interfaces::msg::AudioFrame buildFrame(const uint8_t * data, size_t byte_count);

  FaNetworkInConfig config_{};
  std::unique_ptr<backends::NetworkPcmReceiverBackend> backend_{};

  rclcpp::Publisher<fa_interfaces::msg::AudioFrame>::SharedPtr audio_pub_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diag_pub_;
  rclcpp::TimerBase::SharedPtr poll_timer_;
  rclcpp::TimerBase::SharedPtr diag_timer_;

  std::atomic<bool> fatal_error_{false};
  std::atomic<uint64_t> packets_published_{0};
  std::atomic<uint64_t> bytes_published_{0};
  std::atomic<uint64_t> packets_rejected_{0};
};

}  // namespace fa_network_in

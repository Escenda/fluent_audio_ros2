#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>

#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "fa_interfaces/msg/audio_frame.hpp"
#include "fa_network_out/backends/network_pcm_sender_backend.hpp"

namespace fa_network_out
{

struct FaNetworkOutConfig
{
  std::string backend_name{};
  std::string endpoint_uri{};
  std::string transport_identity{};
  std::string input_topic{};
  int expected_sample_rate{-1};
  int expected_channels{-1};
  std::string expected_encoding{};
  int expected_bit_depth{-1};
  std::string expected_layout{};
  int qos_depth{-1};
  bool qos_reliable{false};
  int diagnostics_publish_period_ms{-1};
};

/**
 * @brief AudioFrame payload を明示 UDP endpoint へ raw PCM packet として送信する sink adapter。
 */
class FaNetworkOutNode : public rclcpp::Node
{
public:
  explicit FaNetworkOutNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~FaNetworkOutNode() override;

  bool hasFatalError() const;

private:
  void loadParameters();
  void validateConfig() const;
  void openEndpoint();
  void setupInterfaces();
  void handleFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg);
  void publishDiagnostics();
  void failClosed(const std::string & reason);

  bool validateFrame(const fa_interfaces::msg::AudioFrame & msg);
  size_t bytesPerFrame() const;

  FaNetworkOutConfig config_{};
  std::unique_ptr<backends::NetworkPcmSenderBackend> backend_{};

  rclcpp::Subscription<fa_interfaces::msg::AudioFrame>::SharedPtr audio_sub_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diag_pub_;
  rclcpp::TimerBase::SharedPtr diag_timer_;

  std::atomic<bool> fatal_error_{false};
  std::atomic<uint64_t> frames_sent_{0};
  std::atomic<uint64_t> frames_rejected_{0};
  std::atomic<uint64_t> bytes_sent_{0};
};

}  // namespace fa_network_out

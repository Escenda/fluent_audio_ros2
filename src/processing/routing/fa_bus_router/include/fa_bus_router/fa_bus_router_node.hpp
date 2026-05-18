#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "fa_interfaces/msg/audio_frame.hpp"

namespace fa_bus_router
{

struct BusRouterConfig
{
  std::string input_topic{};
  std::vector<std::string> output_topics{};
  std::string input_stream_id{};
  std::vector<std::string> output_stream_ids{};
  std::string resolved_input_topic{};
  std::vector<std::string> resolved_output_topics{};
  int expected_sample_rate{-1};
  int expected_channels{-1};
  std::string expected_encoding{};
  int expected_bit_depth{-1};
  std::string expected_layout{};
  int qos_depth{-1};
  bool qos_reliable{true};
  int diagnostics_publish_period_ms{-1};
  int diagnostics_qos_depth{-1};
  bool diagnostics_qos_reliable{false};
};

/**
 * @brief 1つの AudioFrame stream を明示された複数 topic へ byte-for-byte に複製する routing node。
 */
class FaBusRouterNode : public rclcpp::Node
{
public:
  FaBusRouterNode();
  ~FaBusRouterNode() override = default;

private:
  void loadParameters();
  void setupInterfaces();
  void handleFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg);
  void publishCopies(const fa_interfaces::msg::AudioFrame & msg);
  void publishDiagnostics();

  bool validateFrame(const fa_interfaces::msg::AudioFrame & msg);
  size_t bytesPerFrame() const;

  BusRouterConfig config_;

  rclcpp::Subscription<fa_interfaces::msg::AudioFrame>::SharedPtr audio_sub_;
  std::vector<rclcpp::Publisher<fa_interfaces::msg::AudioFrame>::SharedPtr> audio_pubs_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diag_pub_;
  rclcpp::TimerBase::SharedPtr diag_timer_;

  std::atomic<uint64_t> frames_in_{0};
  std::atomic<uint64_t> frames_dropped_{0};
  std::atomic<uint64_t> copies_out_{0};
};

}  // namespace fa_bus_router

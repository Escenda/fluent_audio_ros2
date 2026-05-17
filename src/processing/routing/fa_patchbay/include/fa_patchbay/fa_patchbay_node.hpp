#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "fa_interfaces/msg/audio_frame.hpp"

namespace fa_patchbay
{

struct PatchbayRoute
{
  std::string input_topic{};
  std::string output_topic{};
  rclcpp::Publisher<fa_interfaces::msg::AudioFrame>::SharedPtr publisher{};
};

struct PatchbayConfig
{
  std::vector<std::string> input_topics{};
  std::vector<std::string> output_topics{};
  int expected_sample_rate{-1};
  int expected_channels{-1};
  std::string expected_encoding{};
  int expected_bit_depth{-1};
  std::string expected_layout{};
  int qos_depth{-1};
  bool qos_reliable{true};
  int diagnostics_publish_period_ms{-1};
};

/**
 * @brief 明示された input/output topic 対を使って AudioFrame を静的に複製する patchbay node。
 *
 * @details DSP、format 変換、gain、mixing、device I/O は行わず、契約に合う frame だけを route する。
 */
class FaPatchbayNode : public rclcpp::Node
{
public:
  FaPatchbayNode();
  ~FaPatchbayNode() override = default;

private:
  void loadParameters();
  void setupInterfaces();
  void handleFrame(const std::string & input_topic, const fa_interfaces::msg::AudioFrame::SharedPtr msg);
  void publishCopies(const std::string & input_topic, const fa_interfaces::msg::AudioFrame & msg);
  void publishDiagnostics();

  bool validateFrame(const std::string & input_topic, const fa_interfaces::msg::AudioFrame & msg);
  size_t bytesPerFrame() const;

  PatchbayConfig config_;
  std::vector<PatchbayRoute> routes_;
  std::vector<std::string> unique_input_topics_;
  std::unordered_map<std::string, std::vector<size_t>> route_indices_by_input_;

  std::vector<rclcpp::Subscription<fa_interfaces::msg::AudioFrame>::SharedPtr> audio_subs_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diag_pub_;
  rclcpp::TimerBase::SharedPtr diag_timer_;

  std::atomic<uint64_t> frames_in_{0};
  std::atomic<uint64_t> frames_dropped_{0};
  std::atomic<uint64_t> copies_out_{0};
};

}  // namespace fa_patchbay

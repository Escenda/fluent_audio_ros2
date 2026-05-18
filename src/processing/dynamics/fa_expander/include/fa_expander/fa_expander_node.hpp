#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>

#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "fa_interfaces/msg/audio_frame.hpp"

namespace fa_expander
{

namespace backends
{
class InternalStaticExpanderBackend;
}  // namespace backends

struct ExpanderConfig
{
  std::string input_topic{};
  std::string output_topic{};
  double threshold_linear{-1.0};
  double ratio{-1.0};
  int expected_sample_rate{-1};
  int expected_channels{-1};
  std::string expected_encoding{};
  int expected_bit_depth{-1};
  std::string expected_layout{};
  int qos_depth{-1};
  bool qos_reliable{false};
  int diagnostics_publish_period_ms{-1};
};

class FaExpanderNode : public rclcpp::Node
{
public:
  explicit FaExpanderNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~FaExpanderNode() override;

private:
  void loadParameters();
  void setupInterfaces();
  void handleFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg);
  void publishDiagnostics();
  void configureBackend();

  bool validateFrame(const fa_interfaces::msg::AudioFrame & msg);
  bool applyExpansion(const fa_interfaces::msg::AudioFrame & in, fa_interfaces::msg::AudioFrame & out);

  ExpanderConfig config_;
  std::unique_ptr<backends::InternalStaticExpanderBackend> backend_{};

  rclcpp::Subscription<fa_interfaces::msg::AudioFrame>::SharedPtr audio_sub_;
  rclcpp::Publisher<fa_interfaces::msg::AudioFrame>::SharedPtr audio_pub_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diag_pub_;
  rclcpp::TimerBase::SharedPtr diag_timer_;

  std::atomic<uint64_t> frames_in_{0};
  std::atomic<uint64_t> frames_out_{0};
  std::atomic<uint64_t> frames_dropped_{0};
  std::atomic<uint64_t> samples_expanded_{0};
};

}  // namespace fa_expander

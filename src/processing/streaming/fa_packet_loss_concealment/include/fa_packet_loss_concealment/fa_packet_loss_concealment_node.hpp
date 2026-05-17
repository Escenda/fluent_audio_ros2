#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "fa_interfaces/msg/audio_frame.hpp"
#include "std_msgs/msg/header.hpp"

namespace fa_packet_loss_concealment
{

struct PacketLossConcealmentConfig
{
  std::string input_topic{};
  std::string output_topic{};
  int expected_sample_rate{-1};
  int expected_channels{-1};
  std::string expected_encoding{};
  int expected_bit_depth{-1};
  std::string expected_layout{};
  int max_gap_frames{-1};
  double attenuation_per_gap{-1.0};
  int qos_depth{-1};
  bool qos_reliable{false};
  int diagnostics_publish_period_ms{-1};
};

struct PreviousValidFrame
{
  std_msgs::msg::Header header{};
  std::string source_id{};
  std::string encoding{};
  uint32_t sample_rate{0};
  uint32_t channels{0};
  uint32_t bit_depth{0};
  std::string layout{};
  std::vector<uint8_t> data{};
  uint32_t epoch{0};
};

class FaPacketLossConcealmentNode : public rclcpp::Node
{
public:
  FaPacketLossConcealmentNode();
  ~FaPacketLossConcealmentNode() override = default;

private:
  void loadParameters();
  void setupInterfaces();
  void handleFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg);
  void publishDiagnostics();
  void publishCurrentFrame(const fa_interfaces::msg::AudioFrame & msg);
  void updatePreviousFrame(const fa_interfaces::msg::AudioFrame & msg);
  void resetPreviousFrame();

  bool validateFrame(const fa_interfaces::msg::AudioFrame & msg);
  bool sourceChanged(const fa_interfaces::msg::AudioFrame & msg) const;
  bool publishConcealedFrames(uint32_t missing_frame_count);
  bool buildConcealedFrame(uint32_t gap_index, fa_interfaces::msg::AudioFrame & out) const;
  bool timestampForGap(uint32_t gap_index, std_msgs::msg::Header & header) const;
  float readFloat32LeSample(const std::vector<uint8_t> & data, size_t byte_offset) const;
  void writeFloat32LeSample(std::vector<uint8_t> & data, size_t byte_offset, float sample) const;
  size_t bytesPerSampleFrame() const;

  PacketLossConcealmentConfig config_;
  PreviousValidFrame previous_frame_{};
  bool has_previous_frame_{false};

  rclcpp::Subscription<fa_interfaces::msg::AudioFrame>::SharedPtr audio_sub_;
  rclcpp::Publisher<fa_interfaces::msg::AudioFrame>::SharedPtr audio_pub_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diag_pub_;
  rclcpp::TimerBase::SharedPtr diag_timer_;

  std::atomic<uint64_t> frames_in_{0};
  std::atomic<uint64_t> frames_out_{0};
  std::atomic<uint64_t> frames_dropped_{0};
  std::atomic<uint64_t> concealed_frames_{0};
  std::atomic<uint64_t> duplicate_drops_{0};
  std::atomic<uint64_t> gap_resets_{0};
};

}  // namespace fa_packet_loss_concealment

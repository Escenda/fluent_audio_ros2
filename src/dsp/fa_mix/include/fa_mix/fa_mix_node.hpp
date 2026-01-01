#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "fa_interfaces/msg/audio_frame.hpp"

namespace fa_mix
{

struct MixConfig
{
  std::vector<std::string> input_topics;
  std::vector<double> input_gains_db;
  int master_index = -1;
  std::string output_topic;

  int expected_sample_rate = -1;
  int expected_channels = -1;
  int expected_bit_depth = -1;
  std::string expected_encoding;

  int max_frame_age_ms = -1;

  int qos_depth = -1;
  bool qos_reliable = true;

  int diagnostics_publish_period_ms = -1;
};

class FaMixNode : public rclcpp::Node
{
public:
  FaMixNode();
  ~FaMixNode() override = default;

private:
  void loadParameters();
  void setupInterfaces();
  void publishDiagnostics();

  void onInputFrame(size_t index, const fa_interfaces::msg::AudioFrame::SharedPtr msg);
  void mixAndPublish(const fa_interfaces::msg::AudioFrame & base);

  bool validateFrame(const fa_interfaces::msg::AudioFrame & msg) const;
  static bool decodePcm16ToFloat(const fa_interfaces::msg::AudioFrame & msg, std::vector<float> & out_samples);
  static void encodeFloatToPcm16(const std::vector<float> & samples, std::vector<uint8_t> & out_bytes);
  static void computeRmsPeak(const std::vector<float> & interleaved, float & out_rms, float & out_peak);

  MixConfig config_;

  std::vector<rclcpp::Subscription<fa_interfaces::msg::AudioFrame>::SharedPtr> subs_;
  rclcpp::Publisher<fa_interfaces::msg::AudioFrame>::SharedPtr pub_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diag_pub_;
  rclcpp::TimerBase::SharedPtr diag_timer_;

  std::mutex frames_mutex_;
  std::vector<fa_interfaces::msg::AudioFrame::SharedPtr> latest_frames_;
  std::vector<rclcpp::Time> latest_frames_time_;

  std::atomic<uint64_t> in_{0};
  std::atomic<uint64_t> out_{0};
  std::atomic<uint64_t> drop_{0};
};

}  // namespace fa_mix


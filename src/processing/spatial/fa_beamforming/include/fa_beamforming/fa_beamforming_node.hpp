#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "fa_interfaces/msg/audio_frame.hpp"

namespace fa_beamforming
{

struct BeamformingConfig
{
  std::string input_topic{};
  std::string output_topic{};
  std::string input_stream_id{};
  std::string output_stream_id{};
  std::vector<double> weights{};
  int output_channels{-1};
  int expected_sample_rate{-1};
  int expected_channels{-1};
  std::string expected_encoding{};
  int expected_bit_depth{-1};
  std::string expected_layout{};
  int qos_depth{-1};
  bool qos_reliable{false};
  int diagnostics_publish_period_ms{-1};
  int diagnostics_qos_depth{-1};
  bool diagnostics_qos_reliable{false};
};

/**
 * @brief FLOAT32LE interleaved multi-channel AudioFrame へ固定 weight beamforming を適用する node。
 *
 * @details weight は起動 config で明示された値だけを使う。weight 推定、equal weight fallback、
 * resampling、format conversion、gain、limiter、device I/O は扱わない。起動時 config が不完全な
 * 場合は fail closed し、runtime frame が契約と一致しない場合は publish せず drop する。
 */
class FaBeamformingNode : public rclcpp::Node
{
public:
  FaBeamformingNode();
  ~FaBeamformingNode() override = default;

private:
  void loadParameters();
  void declareRequiredParameter(const std::string & name);
  void setupInterfaces();
  void handleFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg);
  void publishDiagnostics();

  std::string requireStringParameter(const std::string & name);
  int requireIntegerParameter(const std::string & name);
  bool requireBoolParameter(const std::string & name);
  std::vector<double> requireDoubleArrayParameter(const std::string & name);

  bool validateFrame(const fa_interfaces::msg::AudioFrame & msg);
  bool beamformFrame(const fa_interfaces::msg::AudioFrame & in, fa_interfaces::msg::AudioFrame & out);

  static float readFloat32Le(const std::vector<uint8_t> & bytes, size_t sample_index);
  static void appendFloat32Le(float sample, std::vector<uint8_t> & out_bytes);
  static bool isNormalizedFinite(double sample);
  static std::string formatWeights(const std::vector<double> & weights);

  BeamformingConfig config_;
  double weights_sum_abs_{0.0};
  rclcpp::Subscription<fa_interfaces::msg::AudioFrame>::SharedPtr audio_sub_;
  rclcpp::Publisher<fa_interfaces::msg::AudioFrame>::SharedPtr audio_pub_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diag_pub_;
  rclcpp::TimerBase::SharedPtr diag_timer_;

  std::atomic<uint64_t> frames_in_{0};
  std::atomic<uint64_t> frames_out_{0};
  std::atomic<uint64_t> frames_dropped_{0};
};

}  // namespace fa_beamforming

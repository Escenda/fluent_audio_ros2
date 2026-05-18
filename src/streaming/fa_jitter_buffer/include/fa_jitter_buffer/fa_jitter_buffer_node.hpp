#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "fa_interfaces/msg/audio_frame.hpp"

namespace fa_jitter_buffer
{

struct JitterBufferConfig
{
  std::string input_topic{};
  std::string output_topic{};
  int expected_sample_rate{-1};
  int expected_channels{-1};
  std::string expected_encoding{};
  int expected_bit_depth{-1};
  std::string expected_layout{};
  int target_depth_frames{-1};
  int max_depth_frames{-1};
  bool reset_on_epoch_regression{false};
  int qos_depth{-1};
  bool qos_reliable{false};
  int diagnostics_publish_period_ms{-1};
  int diagnostics_qos_depth{-1};
  bool diagnostics_qos_reliable{false};
};

struct ActiveStreamIdentity
{
  std::string source_id{};
  std::string stream_id{};
  uint32_t sample_rate{0};
  uint32_t channels{0};
  std::string encoding{};
  uint32_t bit_depth{0};
  std::string layout{};
};

/**
 * @brief AudioFrame を epoch 順に保持し、target depth を超えた分だけ publish する node。
 */
class FaJitterBufferNode : public rclcpp::Node
{
public:
  explicit FaJitterBufferNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~FaJitterBufferNode() override = default;

private:
  void loadParameters();
  void setupInterfaces();
  void handleFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg);
  void publishDiagnostics();

  bool validateFrame(const fa_interfaces::msg::AudioFrame & msg);
  bool validateFloat32InterleavedSamples(const std::vector<uint8_t> & data);
  bool hasDifferentContract(const fa_interfaces::msg::AudioFrame & msg) const;
  bool isDuplicateEpoch(const fa_interfaces::msg::AudioFrame & msg) const;
  bool isLateEpoch(const fa_interfaces::msg::AudioFrame & msg) const;
  void activateStream(const fa_interfaces::msg::AudioFrame & msg);
  void resetBuffer();
  void insertFrame(const fa_interfaces::msg::AudioFrame & msg);
  void publishReadyFrames();
  float readFloat32LeSample(const std::vector<uint8_t> & data, size_t byte_offset) const;
  size_t bytesPerSampleFrame() const;

  JitterBufferConfig config_;
  std::map<uint32_t, fa_interfaces::msg::AudioFrame> buffered_frames_{};
  std::optional<ActiveStreamIdentity> active_stream_{};
  std::optional<uint32_t> last_published_epoch_{};

  rclcpp::Subscription<fa_interfaces::msg::AudioFrame>::SharedPtr audio_sub_;
  rclcpp::Publisher<fa_interfaces::msg::AudioFrame>::SharedPtr audio_pub_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diag_pub_;
  rclcpp::TimerBase::SharedPtr diag_timer_;

  std::atomic<uint64_t> frames_in_{0};
  std::atomic<uint64_t> frames_out_{0};
  std::atomic<uint64_t> frames_dropped_{0};
  std::atomic<uint64_t> duplicate_drops_{0};
  std::atomic<uint64_t> late_drops_{0};
  std::atomic<uint64_t> max_depth_resets_{0};
  std::atomic<uint64_t> resets_{0};
  std::atomic<uint64_t> buffered_frame_count_{0};
};

}  // namespace fa_jitter_buffer

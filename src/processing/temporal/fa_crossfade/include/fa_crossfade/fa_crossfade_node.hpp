#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

#include <rclcpp/rclcpp.hpp>

#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "fa_interfaces/msg/audio_frame.hpp"

namespace fa_crossfade
{

namespace backends
{
class InternalCrossfadeBackend;
}  // namespace backends

struct CrossfadeConfig
{
  std::string input_a_topic{};
  std::string input_b_topic{};
  std::string output_topic{};
  std::string input_a_stream_id{};
  std::string input_b_stream_id{};
  std::string output_stream_id{};
  int overlap_frames{-1};
  std::string curve{};
  int expected_sample_rate{-1};
  int expected_channels{-1};
  std::string expected_encoding{};
  int expected_bit_depth{-1};
  std::string expected_layout{};
  int qos_depth{-1};
  bool qos_reliable{false};
  int diagnostics_qos_depth{-1};
  bool diagnostics_qos_reliable{false};
  int diagnostics_publish_period_ms{-1};
};

/**
 * @brief 2つの完全な AudioFrame segment を明示 overlap で crossfade する temporal node。
 */
class FaCrossfadeNode : public rclcpp::Node
{
public:
  explicit FaCrossfadeNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~FaCrossfadeNode() override;

private:
  void loadParameters();
  void configureBackend();
  void setupInterfaces();
  void handleInputA(const fa_interfaces::msg::AudioFrame::SharedPtr msg);
  void handleInputB(const fa_interfaces::msg::AudioFrame::SharedPtr msg);
  void handleFrame(
    const fa_interfaces::msg::AudioFrame::SharedPtr msg,
    bool is_input_a);
  void tryPublishPairLocked();
  void clearPendingLocked();
  void publishDiagnostics();

  bool validateFrame(
    const fa_interfaces::msg::AudioFrame & msg,
    const std::string & expected_stream_id,
    const char * input_name);
  bool pairMetadataMatches(
    const fa_interfaces::msg::AudioFrame & segment_a,
    const fa_interfaces::msg::AudioFrame & segment_b) const;
  size_t bytesPerFrame() const;

  CrossfadeConfig config_;
  std::unique_ptr<backends::InternalCrossfadeBackend> backend_{};

  rclcpp::Subscription<fa_interfaces::msg::AudioFrame>::SharedPtr input_a_sub_;
  rclcpp::Subscription<fa_interfaces::msg::AudioFrame>::SharedPtr input_b_sub_;
  rclcpp::Publisher<fa_interfaces::msg::AudioFrame>::SharedPtr audio_pub_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diag_pub_;
  rclcpp::TimerBase::SharedPtr diag_timer_;

  mutable std::mutex pending_mutex_;
  bool has_pending_a_{false};
  bool has_pending_b_{false};
  fa_interfaces::msg::AudioFrame pending_a_{};
  fa_interfaces::msg::AudioFrame pending_b_{};

  std::atomic<uint64_t> input_a_frames_in_{0};
  std::atomic<uint64_t> input_b_frames_in_{0};
  std::atomic<uint64_t> pairs_out_{0};
  std::atomic<uint64_t> frames_dropped_{0};
  std::atomic<uint64_t> invalid_frames_dropped_{0};
  std::atomic<uint64_t> epoch_mismatches_{0};
  std::atomic<uint64_t> metadata_mismatches_{0};
  std::atomic<uint64_t> backend_rejections_{0};
};

}  // namespace fa_crossfade

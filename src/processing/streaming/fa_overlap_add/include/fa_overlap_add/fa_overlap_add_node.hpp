#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include "builtin_interfaces/msg/time.hpp"
#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "fa_interfaces/msg/audio_frame.hpp"

namespace fa_overlap_add
{

struct OverlapAddConfig
{
  std::string input_topic{};
  std::string output_topic{};
  int expected_sample_rate{-1};
  int expected_channels{-1};
  std::string expected_encoding{};
  int expected_bit_depth{-1};
  std::string expected_layout{};
  int frame_samples{-1};
  int hop_samples{-1};
  std::string window_type{};
  int max_buffered_chunks{-1};
  int qos_depth{-1};
  bool qos_reliable{false};
  int diagnostics_publish_period_ms{-1};
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
 * @brief Overlapped AudioFrame chunk を synthesis window で overlap-add 復元する node。
 */
class FaOverlapAddNode : public rclcpp::Node
{
public:
  FaOverlapAddNode();
  ~FaOverlapAddNode() override = default;

private:
  void loadParameters();
  void setupInterfaces();
  void handleFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg);
  void publishDiagnostics();

  bool validateFrame(const fa_interfaces::msg::AudioFrame & msg);
  bool hasInputEpochRegression(const fa_interfaces::msg::AudioFrame & msg) const;
  bool requiresStreamReset(const fa_interfaces::msg::AudioFrame & msg) const;
  bool hasFormatChange(const fa_interfaces::msg::AudioFrame & msg) const;
  void activateStream(const fa_interfaces::msg::AudioFrame & msg);
  bool canAccumulateChunk() const;
  void accumulateChunk(const fa_interfaces::msg::AudioFrame & msg);
  void publishAvailableFrames();
  bool buildOutputFrame(fa_interfaces::msg::AudioFrame & out) const;
  bool advanceNextOutputStamp();
  void consumePublishedHop();
  void resetOverlapState();
  void buildSynthesisWindow();

  float readFloat32LeSample(const std::vector<uint8_t> & data, size_t byte_offset) const;
  void writeFloat32LeSample(std::vector<uint8_t> & data, size_t byte_offset, float sample) const;
  size_t bytesPerSampleFrame() const;
  size_t chunkBytes() const;
  size_t maxBufferedSampleFrames() const;

  OverlapAddConfig config_;
  std::vector<double> synthesis_window_{};
  std::vector<double> sample_sums_{};
  std::vector<double> weight_sums_{};
  std::optional<ActiveStreamIdentity> active_stream_{};
  std::optional<uint32_t> next_expected_input_epoch_{};
  std::optional<builtin_interfaces::msg::Time> next_output_stamp_{};
  size_t next_chunk_start_sample_frames_{0};
  uint32_t next_output_epoch_{0};

  rclcpp::Subscription<fa_interfaces::msg::AudioFrame>::SharedPtr audio_sub_;
  rclcpp::Publisher<fa_interfaces::msg::AudioFrame>::SharedPtr audio_pub_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diag_pub_;
  rclcpp::TimerBase::SharedPtr diag_timer_;

  std::atomic<uint64_t> frames_in_{0};
  std::atomic<uint64_t> frames_out_{0};
  std::atomic<uint64_t> frames_dropped_{0};
  std::atomic<uint64_t> epoch_regression_drops_{0};
  std::atomic<uint64_t> chunks_accumulated_{0};
  std::atomic<uint64_t> resets_{0};
  std::atomic<uint64_t> buffered_sample_frames_{0};
};

}  // namespace fa_overlap_add

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "fa_interfaces/msg/audio_frame.hpp"
#include "std_msgs/msg/header.hpp"

namespace fa_chunk_overlap
{

struct ChunkOverlapConfig
{
  std::string input_topic{};
  std::string output_topic{};
  std::string input_stream_id{};
  std::string output_stream_id{};
  int expected_sample_rate{-1};
  int expected_channels{-1};
  std::string expected_encoding{};
  int expected_bit_depth{-1};
  std::string expected_layout{};
  int frame_samples{-1};
  int hop_samples{-1};
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

struct BufferedSegment
{
  std_msgs::msg::Header header{};
  size_t byte_count{0};
};

class FaChunkOverlapNode : public rclcpp::Node
{
public:
  explicit FaChunkOverlapNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~FaChunkOverlapNode() override = default;

private:
  void loadParameters();
  void setupInterfaces();
  void handleFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg);
  void appendFrame(const fa_interfaces::msg::AudioFrame & msg);
  void publishAvailableChunks();
  void publishDiagnostics();
  void resetActiveBuffer();
  void activateStream(const fa_interfaces::msg::AudioFrame & msg);
  bool consumeBufferedBytes(size_t byte_count);
  bool advanceSegmentHeader(BufferedSegment & segment, size_t consumed_bytes);

  bool validateFrame(const fa_interfaces::msg::AudioFrame & msg);
  bool hasDifferentSource(const fa_interfaces::msg::AudioFrame & msg) const;
  float readFloat32LeSample(const std::vector<uint8_t> & data, size_t byte_offset) const;
  size_t bytesPerSampleFrame() const;
  size_t windowBytes() const;
  size_t hopBytes() const;

  ChunkOverlapConfig config_;
  std::vector<uint8_t> buffer_{};
  std::deque<BufferedSegment> buffered_segments_{};
  std::optional<ActiveStreamIdentity> active_stream_{};
  uint32_t next_output_epoch_{0};

  rclcpp::Subscription<fa_interfaces::msg::AudioFrame>::SharedPtr audio_sub_;
  rclcpp::Publisher<fa_interfaces::msg::AudioFrame>::SharedPtr audio_pub_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diag_pub_;
  rclcpp::TimerBase::SharedPtr diag_timer_;

  std::atomic<uint64_t> frames_in_{0};
  std::atomic<uint64_t> input_frames_accepted_{0};
  std::atomic<uint64_t> input_frames_dropped_{0};
  std::atomic<uint64_t> chunks_out_{0};
  std::atomic<uint64_t> sample_frames_out_{0};
  std::atomic<uint64_t> source_resets_{0};
  std::atomic<uint64_t> buffered_sample_frames_{0};
};

}  // namespace fa_chunk_overlap

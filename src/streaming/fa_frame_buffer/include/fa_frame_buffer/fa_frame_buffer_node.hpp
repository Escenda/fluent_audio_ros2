#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "fa_interfaces/msg/audio_frame.hpp"
#include "std_msgs/msg/header.hpp"

namespace fa_frame_buffer
{

struct FrameBufferConfig
{
  std::string input_topic{};
  std::string output_topic{};
  int expected_sample_rate{-1};
  int expected_channels{-1};
  std::string expected_encoding{};
  int expected_bit_depth{-1};
  std::string expected_layout{};
  int frames_per_chunk{-1};
  int max_buffered_chunks{-1};
  int qos_depth{-1};
  bool qos_reliable{false};
  int diagnostics_publish_period_ms{-1};
};

struct BufferedFrameIdentity
{
  std_msgs::msg::Header header{};
  std::string source_id{};
  uint32_t sample_rate{0};
  uint32_t channels{0};
  std::string encoding{};
  uint32_t bit_depth{0};
  std::string layout{};
  uint32_t epoch{0};
};

struct BufferedSegment
{
  BufferedFrameIdentity identity{};
  size_t byte_count{0};
};

/**
 * @brief FLOAT32LE interleaved AudioFrame を固定フレーム数の chunk にまとめる streaming node。
 */
class FaFrameBufferNode : public rclcpp::Node
{
public:
  FaFrameBufferNode();
  ~FaFrameBufferNode() override = default;

private:
  void loadParameters();
  void setupInterfaces();
  void handleFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg);
  void appendFrame(const fa_interfaces::msg::AudioFrame & msg);
  void publishAvailableChunks();
  void publishDiagnostics();
  void clearBufferedStream();
  void dropOldestChunkForOverflow();
  void consumeBufferedBytes(size_t byte_count);

  bool validateFrame(const fa_interfaces::msg::AudioFrame & msg);
  bool isCompatibleWithBufferedStream(const fa_interfaces::msg::AudioFrame & msg) const;
  BufferedFrameIdentity identityFromFrame(const fa_interfaces::msg::AudioFrame & msg) const;
  size_t bytesPerFrame() const;
  size_t chunkBytes() const;
  size_t maxBufferedBytes() const;

  FrameBufferConfig config_;
  std::vector<uint8_t> buffer_{};
  std::deque<BufferedSegment> buffered_segments_{};

  rclcpp::Subscription<fa_interfaces::msg::AudioFrame>::SharedPtr audio_sub_;
  rclcpp::Publisher<fa_interfaces::msg::AudioFrame>::SharedPtr audio_pub_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diag_pub_;
  rclcpp::TimerBase::SharedPtr diag_timer_;

  std::atomic<uint64_t> frames_in_{0};
  std::atomic<uint64_t> frames_out_{0};
  std::atomic<uint64_t> frames_dropped_{0};
  std::atomic<uint64_t> chunks_out_{0};
  std::atomic<uint64_t> partial_frames_buffered_{0};
  std::atomic<uint64_t> buffer_resets_{0};
  std::atomic<uint64_t> overflow_count_{0};
};

}  // namespace fa_frame_buffer

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>

#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "fa_file_in/backends/pcm_file_reader_backend.hpp"
#include "fa_interfaces/msg/audio_frame.hpp"

namespace fa_file_in
{

struct FaFileInConfig
{
  std::string backend_name{};
  std::string file_path{};
  std::string output_topic{};
  std::string source_id{};
  std::string stream_id{};
  int expected_sample_rate{-1};
  int expected_channels{-1};
  std::string expected_encoding{};
  int expected_bit_depth{-1};
  std::string expected_layout{};
  int frames_per_chunk{-1};
  bool playback_loop{false};
  int playback_publish_period_ms{-1};
  int qos_depth{-1};
  bool qos_reliable{false};
  int diagnostics_publish_period_ms{-1};
};

/**
 * @brief 明示設定された raw PCM file を AudioFrame として publish する source adapter。
 */
class FaFileInNode : public rclcpp::Node
{
public:
  explicit FaFileInNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~FaFileInNode() override;

  bool hasFatalError() const;

private:
  void loadParameters();
  void validateConfig() const;
  void openFile();
  void setupInterfaces();
  void publishNextChunk();
  void publishDiagnostics();
  void failClosed(const std::string & reason);

  size_t bytesPerFrame() const;
  size_t bytesPerChunk() const;
  fa_interfaces::msg::AudioFrame buildFrame(const uint8_t * data, size_t byte_count);

  FaFileInConfig config_{};
  std::unique_ptr<backends::PcmFileReaderBackend> backend_{};

  rclcpp::Publisher<fa_interfaces::msg::AudioFrame>::SharedPtr audio_pub_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diag_pub_;
  rclcpp::TimerBase::SharedPtr publish_timer_;
  rclcpp::TimerBase::SharedPtr diag_timer_;

  std::atomic<bool> fatal_error_{false};
  std::atomic<bool> completed_{false};
  std::atomic<uint64_t> frames_published_{0};
  std::atomic<uint64_t> bytes_published_{0};
  std::atomic<uint64_t> loops_completed_{0};
  std::atomic<uint64_t> short_chunks_published_{0};
};

}  // namespace fa_file_in

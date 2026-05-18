#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>

#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "fa_file_out/backends/pcm_file_writer_backend.hpp"
#include "fa_interfaces/msg/audio_frame.hpp"

namespace fa_file_out
{

struct FaFileOutConfig
{
  std::string backend_name{};
  std::string file_path{};
  std::string input_topic{};
  int expected_sample_rate{-1};
  int expected_channels{-1};
  std::string expected_encoding{};
  int expected_bit_depth{-1};
  std::string expected_layout{};
  bool overwrite_enabled{false};
  int qos_depth{-1};
  bool qos_reliable{false};
  int diagnostics_publish_period_ms{-1};
  int diagnostics_qos_depth{-1};
  bool diagnostics_qos_reliable{false};
};

/**
 * @brief AudioFrame payload を明示設定された raw PCM file へ write する sink adapter。
 */
class FaFileOutNode : public rclcpp::Node
{
public:
  explicit FaFileOutNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~FaFileOutNode() override;

  bool hasFatalError() const;

private:
  void loadParameters();
  void validateConfig() const;
  void openFile();
  void setupInterfaces();
  void handleFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg);
  void publishDiagnostics();
  void failClosed(const std::string & reason);

  bool validateFrame(const fa_interfaces::msg::AudioFrame & msg);
  size_t bytesPerFrame() const;

  FaFileOutConfig config_{};
  std::unique_ptr<backends::PcmFileWriterBackend> backend_{};

  rclcpp::Subscription<fa_interfaces::msg::AudioFrame>::SharedPtr audio_sub_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diag_pub_;
  rclcpp::TimerBase::SharedPtr diag_timer_;

  std::atomic<bool> fatal_error_{false};
  std::atomic<uint64_t> frames_written_{0};
  std::atomic<uint64_t> frames_rejected_{0};
  std::atomic<uint64_t> bytes_written_{0};
};

}  // namespace fa_file_out

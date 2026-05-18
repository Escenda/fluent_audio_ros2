#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "fa_decode/backends/external_codec_decoder.hpp"
#include "fa_interfaces/msg/audio_frame.hpp"
#include "fa_interfaces/msg/encoded_audio_chunk.hpp"

namespace fa_decode
{

struct FaDecodeConfig
{
  std::string backend_name{};
  std::string command_executable{};
  std::vector<std::string> command_arguments{};
  int command_timeout_ms{-1};
  int command_max_output_bytes{-1};
  std::string input_topic{};
  std::string output_topic{};
  std::string input_stream_id{};
  std::string output_stream_id{};
  std::string input_codec{};
  std::string input_container{};
  std::string input_payload_format{};
  int input_sample_rate{-1};
  int input_channels{-1};
  int output_sample_rate{-1};
  int output_channels{-1};
  std::string output_encoding{};
  int output_bit_depth{-1};
  std::string output_layout{};
  int qos_depth{-1};
  bool qos_reliable{false};
  int diagnostics_publish_period_ms{-1};
  int diagnostics_qos_depth{-1};
  bool diagnostics_qos_reliable{false};
};

/**
 * @brief EncodedAudioChunk を明示 codec backend で PCM AudioFrame へ変換する node。
 *
 * @details device I/O、resample、bit-depth conversion、gain、VAD/ASR は扱わない。
 * backend は ROS2 message を知らず、node が EncodedAudioChunk と AudioFrame の契約を検証する。
 */
class FaDecodeNode : public rclcpp::Node
{
public:
  explicit FaDecodeNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~FaDecodeNode() override = default;

private:
  void loadParameters();
  void setupBackend();
  void setupInterfaces();
  void handleChunk(const fa_interfaces::msg::EncodedAudioChunk::SharedPtr msg);
  void publishDiagnostics();

  bool validateChunk(const fa_interfaces::msg::EncodedAudioChunk & msg);
  bool buildFrame(
    const fa_interfaces::msg::EncodedAudioChunk & in,
    const backends::DecodeResult & result,
    fa_interfaces::msg::AudioFrame & out);

  FaDecodeConfig config_;
  std::unique_ptr<backends::ExternalCodecDecoderBackend> backend_;
  rclcpp::Subscription<fa_interfaces::msg::EncodedAudioChunk>::SharedPtr encoded_sub_;
  rclcpp::Publisher<fa_interfaces::msg::AudioFrame>::SharedPtr audio_pub_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diag_pub_;
  rclcpp::TimerBase::SharedPtr diag_timer_;

  std::atomic<uint64_t> chunks_in_{0};
  std::atomic<uint64_t> frames_out_{0};
  std::atomic<uint64_t> chunks_dropped_{0};
  std::atomic<uint64_t> decoded_bytes_out_{0};
};

}  // namespace fa_decode

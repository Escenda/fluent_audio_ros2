#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <utility>

#include "fa_interfaces/msg/audio_frame.hpp"
#include "fa_interfaces/srv/record.hpp"
#include "rclcpp/rclcpp.hpp"

namespace fa_record
{

namespace
{
struct AudioFormat
{
  uint32_t sample_rate{48000};
  uint32_t channels{1};
  uint32_t bit_depth{16};
  std::string encoding{"pcm16"};
};

bool formatMatches(const AudioFormat &a, const AudioFormat &b)
{
  return a.sample_rate == b.sample_rate && a.channels == b.channels && a.bit_depth == b.bit_depth;
}

AudioFormat formatFromFrame(const fa_interfaces::msg::AudioFrame &msg)
{
  AudioFormat fmt;
  fmt.sample_rate = msg.sample_rate;
  fmt.channels = msg.channels;
  fmt.bit_depth = msg.bit_depth;
  fmt.encoding = msg.encoding;
  return fmt;
}

void writeWavHeader(std::fstream &stream, const AudioFormat &format, uint32_t data_length)
{
  struct WavHeader
  {
    char riff[4];
    uint32_t chunk_size;
    char wave[4];
    char fmt[4];
    uint32_t subchunk1_size;
    uint16_t audio_format;
    uint16_t num_channels;
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample;
    char data[4];
    uint32_t subchunk2_size;
  } header{};

  std::memcpy(header.riff, "RIFF", 4);
  std::memcpy(header.wave, "WAVE", 4);
  std::memcpy(header.fmt, "fmt ", 4);
  std::memcpy(header.data, "data", 4);

  header.subchunk1_size = 16;
  header.audio_format = (format.bit_depth == 16) ? 1 : 3;
  header.num_channels = static_cast<uint16_t>(format.channels);
  header.sample_rate = format.sample_rate;
  uint16_t bytes_per_sample = static_cast<uint16_t>(format.bit_depth / 8);
  header.byte_rate = format.sample_rate * format.channels * bytes_per_sample;
  header.block_align = format.channels * bytes_per_sample;
  header.bits_per_sample = static_cast<uint16_t>(format.bit_depth);
  header.subchunk2_size = data_length;
  header.chunk_size = 36 + data_length;

  stream.write(reinterpret_cast<const char *>(&header), sizeof(header));
}

void finalizeWavHeader(std::fstream &stream, uint32_t data_length)
{
  stream.seekp(4, std::ios::beg);
  uint32_t chunk_size = 36 + data_length;
  stream.write(reinterpret_cast<const char *>(&chunk_size), sizeof(chunk_size));

  stream.seekp(40, std::ios::beg);
  stream.write(reinterpret_cast<const char *>(&data_length), sizeof(data_length));
}

}  // namespace

class FaRecordNode : public rclcpp::Node
{
public:
  FaRecordNode()
  : rclcpp::Node("fa_record")
  {
    input_topic_ = this->declare_parameter<std::string>("input_topic", input_topic_);

    audio_sub_ = this->create_subscription<fa_interfaces::msg::AudioFrame>(
      input_topic_, rclcpp::SensorDataQoS(),
      std::bind(&FaRecordNode::handleFrame, this, std::placeholders::_1));

    record_srv_ = this->create_service<fa_interfaces::srv::Record>(
      "record",
      std::bind(&FaRecordNode::handleRecord, this, std::placeholders::_1, std::placeholders::_2));

    RCLCPP_INFO(this->get_logger(), "FA Record node: input_topic=%s", input_topic_.c_str());
  }

  ~FaRecordNode() override
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (recording_) {
      finalizeAndCloseLocked();
    }
  }

private:
  void handleFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg)
  {
    if (!msg) {
      return;
    }

    const AudioFormat fmt = formatFromFrame(*msg);

    std::lock_guard<std::mutex> lock(mutex_);
    last_seen_format_ = fmt;
    has_last_seen_format_ = true;

    if (!recording_ || !stream_.is_open()) {
      return;
    }

    if (!header_written_) {
      active_format_ = fmt;
      writeWavHeader(stream_, active_format_, 0);
      header_written_ = true;
      RCLCPP_INFO(this->get_logger(), "Recording started: %s", record_path_.c_str());
    } else if (!formatMatches(active_format_, fmt)) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 3000,
        "Frame format changed during recording; dropping frame (active=%uHz/%u/%u frame=%uHz/%u/%u)",
        active_format_.sample_rate, active_format_.channels, active_format_.bit_depth,
        fmt.sample_rate, fmt.channels, fmt.bit_depth);
      return;
    }

    stream_.write(reinterpret_cast<const char *>(msg->data.data()),
      static_cast<std::streamsize>(msg->data.size()));
    recorded_bytes_ += static_cast<uint32_t>(msg->data.size());
  }

  void handleRecord(
    const std::shared_ptr<fa_interfaces::srv::Record::Request> request,
    std::shared_ptr<fa_interfaces::srv::Record::Response> response)
  {
    if (!request || !response) {
      return;
    }

    if (request->command == "start") {
      response->success = startRecording(request->file_path);
      response->message = response->success ? "recording started" : "failed to start recording";
      return;
    }

    if (request->command == "stop") {
      response->success = stopRecording();
      response->message = response->success ? "recording stopped" : "recording was not active";
      return;
    }

    response->success = false;
    response->message = "unknown command";
  }

  bool startRecording(const std::string &path)
  {
    if (path.empty()) {
      RCLCPP_ERROR(this->get_logger(), "record path is empty");
      return false;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    if (recording_) {
      RCLCPP_WARN(this->get_logger(), "already recording");
      return false;
    }

    std::filesystem::path target_path(path);
    auto parent = target_path.parent_path();
    if (!parent.empty()) {
      std::error_code ec;
      std::filesystem::create_directories(parent, ec);
      if (ec) {
        RCLCPP_ERROR(this->get_logger(), "failed to create directory: %s", ec.message().c_str());
        return false;
      }
    }

    stream_.open(target_path, std::ios::binary | std::ios::out | std::ios::trunc);
    if (!stream_.is_open()) {
      RCLCPP_ERROR(this->get_logger(), "failed to open record file %s", path.c_str());
      return false;
    }

    recorded_bytes_ = 0;
    record_path_ = target_path.string();
    header_written_ = false;
    recording_ = true;

    // If we already saw frames, we can write the header immediately.
    if (has_last_seen_format_) {
      active_format_ = last_seen_format_;
      writeWavHeader(stream_, active_format_, 0);
      header_written_ = true;
      RCLCPP_INFO(this->get_logger(), "Recording started: %s", record_path_.c_str());
    }

    return true;
  }

  bool stopRecording()
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!recording_) {
      return false;
    }
    finalizeAndCloseLocked();
    return true;
  }

  void finalizeAndCloseLocked()
  {
    if (stream_.is_open() && header_written_) {
      finalizeWavHeader(stream_, recorded_bytes_);
    }
    if (stream_.is_open()) {
      stream_.close();
    }
    recording_ = false;
    header_written_ = false;
    recorded_bytes_ = 0;
    record_path_.clear();
  }

  std::string input_topic_{"audio/frame"};

  rclcpp::Subscription<fa_interfaces::msg::AudioFrame>::SharedPtr audio_sub_;
  rclcpp::Service<fa_interfaces::srv::Record>::SharedPtr record_srv_;

  std::mutex mutex_;
  bool recording_{false};
  bool header_written_{false};
  uint32_t recorded_bytes_{0};
  std::string record_path_;

  AudioFormat active_format_{};
  AudioFormat last_seen_format_{};
  bool has_last_seen_format_{false};
  std::fstream stream_;
};

}  // namespace fa_record

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<fa_record::FaRecordNode>());
  rclcpp::shutdown();
  return 0;
}


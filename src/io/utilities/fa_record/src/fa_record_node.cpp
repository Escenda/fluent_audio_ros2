#include <chrono>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

#include "fa_interfaces/msg/audio_frame.hpp"
#include "fa_interfaces/srv/record.hpp"
#include "fa_record/backends/file_writer_backend.hpp"
#include "rclcpp/rclcpp.hpp"

namespace fa_record
{

namespace
{
constexpr const char * kInterleavedLayout = "interleaved";

bool isRequiredParameterSet(const rclcpp::Parameter & parameter)
{
  return parameter.get_type() != rclcpp::ParameterType::PARAMETER_NOT_SET;
}

rclcpp::Parameter getRequiredParameter(const rclcpp::Node & node, const std::string & name)
{
  rclcpp::Parameter parameter;
  if (!node.get_parameter(name, parameter) || !isRequiredParameterSet(parameter)) {
    throw std::runtime_error(name + " is required");
  }
  return parameter;
}

std::string readRequiredString(const rclcpp::Node & node, const std::string & name)
{
  const rclcpp::Parameter parameter = getRequiredParameter(node, name);
  if (parameter.get_type() != rclcpp::ParameterType::PARAMETER_STRING) {
    throw std::runtime_error(name + " must be a string parameter");
  }
  return parameter.as_string();
}

int readRequiredInt(const rclcpp::Node & node, const std::string & name)
{
  const rclcpp::Parameter parameter = getRequiredParameter(node, name);
  if (parameter.get_type() != rclcpp::ParameterType::PARAMETER_INTEGER) {
    throw std::runtime_error(name + " must be an integer parameter");
  }
  const int64_t value = parameter.as_int();
  if (value < static_cast<int64_t>(std::numeric_limits<int>::min()) ||
      value > static_cast<int64_t>(std::numeric_limits<int>::max()))
  {
    throw std::runtime_error(name + " is outside supported integer range");
  }
  return static_cast<int>(value);
}

bool readRequiredBool(const rclcpp::Node & node, const std::string & name)
{
  const rclcpp::Parameter parameter = getRequiredParameter(node, name);
  if (parameter.get_type() != rclcpp::ParameterType::PARAMETER_BOOL) {
    throw std::runtime_error(name + " must be a bool parameter");
  }
  return parameter.as_bool();
}

rclcpp::QoS makeExplicitQos(int depth, bool reliable)
{
  if (depth <= 0) {
    throw std::runtime_error("input.qos.depth must be greater than zero");
  }
  rclcpp::QoS qos(rclcpp::KeepLast(static_cast<size_t>(depth)));
  if (reliable) {
    qos.reliable();
  } else {
    qos.best_effort();
  }
  return qos;
}

backends::AudioFormat formatFromFrame(const fa_interfaces::msg::AudioFrame &msg)
{
  backends::AudioFormat fmt;
  fmt.sample_rate = msg.sample_rate;
  fmt.channels = msg.channels;
  fmt.bit_depth = msg.bit_depth;
  fmt.encoding = msg.encoding;
  return fmt;
}

bool isSupportedFrame(const fa_interfaces::msg::AudioFrame &msg)
{
  if (msg.sample_rate == 0 || msg.channels == 0) {
    return false;
  }
  if (msg.bit_depth != 16 && msg.bit_depth != 32) {
    return false;
  }
  if (msg.encoding.empty() || msg.data.empty()) {
    return false;
  }
  if (!((msg.bit_depth == 16 && msg.encoding == "PCM16LE") ||
        (msg.bit_depth == 32 && msg.encoding == "FLOAT32LE"))) {
    return false;
  }
  if (msg.source_id.empty() || msg.stream_id.empty()) {
    return false;
  }
  if (msg.layout != kInterleavedLayout) {
    return false;
  }
  const size_t bytes_per_sample = static_cast<size_t>(msg.bit_depth / 8);
  const size_t bytes_per_frame = static_cast<size_t>(msg.channels) * bytes_per_sample;
  return bytes_per_frame > 0 && (msg.data.size() % bytes_per_frame) == 0;
}

}  // namespace

class FaRecordNode : public rclcpp::Node
{
public:
  FaRecordNode()
  : rclcpp::Node("fa_record"),
    writer_(std::make_unique<backends::WavFileWriterBackend>())
  {
    this->declare_parameter<std::string>("input_topic");
    this->declare_parameter<int>("input.qos.depth");
    this->declare_parameter<bool>("input.qos.reliable");
    input_topic_ = readRequiredString(*this, "input_topic");
    input_qos_depth_ = readRequiredInt(*this, "input.qos.depth");
    input_qos_reliable_ = readRequiredBool(*this, "input.qos.reliable");
    if (input_topic_.empty()) {
      throw std::runtime_error("input_topic is required");
    }

    audio_sub_ = this->create_subscription<fa_interfaces::msg::AudioFrame>(
      input_topic_, makeExplicitQos(input_qos_depth_, input_qos_reliable_),
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
    if (!isSupportedFrame(*msg)) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 3000,
        "Dropping invalid frame: sr=%u ch=%u bits=%u enc=%s bytes=%zu",
        msg->sample_rate, msg->channels, msg->bit_depth, msg->encoding.c_str(), msg->data.size());
      return;
    }

    const backends::AudioFormat fmt = formatFromFrame(*msg);

    std::lock_guard<std::mutex> lock(mutex_);
    last_seen_format_ = fmt;
    has_last_seen_format_ = true;

    if (!recording_ || !writer_ || !writer_->isOpen()) {
      return;
    }

    const bool starting_payload = !writer_->hasFormat();
    try {
      writer_->writeChunk(fmt, msg->data.data(), msg->data.size());
    } catch (const backends::FormatMismatchError &) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 3000,
        "Frame format changed during recording; dropping frame (active file=%s frame=%uHz/%u/%u)",
        writer_->path().c_str(), fmt.sample_rate, fmt.channels, fmt.bit_depth);
      return;
    } catch (const backends::FileWriterError &error) {
      RCLCPP_ERROR(this->get_logger(), "%s", error.what());
      finalizeAndCloseLocked();
      return;
    }

    if (starting_payload) {
      RCLCPP_INFO(this->get_logger(), "Recording started: %s", record_path_.c_str());
    }
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

    if (!writer_) {
      writer_ = std::make_unique<backends::WavFileWriterBackend>();
    }

    try {
      writer_->open(path);
    } catch (const backends::FileWriterError &error) {
      RCLCPP_ERROR(this->get_logger(), "%s", error.what());
      return false;
    }

    record_path_ = writer_->path();
    recording_ = true;

    // If we already saw frames, we can write the header immediately.
    if (has_last_seen_format_) {
      try {
        writer_->startFormat(last_seen_format_);
      } catch (const backends::FileWriterError &error) {
        RCLCPP_ERROR(this->get_logger(), "%s", error.what());
        finalizeAndCloseLocked();
        return false;
      }
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
    if (writer_) {
      try {
        writer_->close();
      } catch (const backends::FileWriterError &error) {
        RCLCPP_ERROR(this->get_logger(), "%s", error.what());
      }
    }
    recording_ = false;
    record_path_.clear();
  }

  std::string input_topic_{};
  int input_qos_depth_{};
  bool input_qos_reliable_{};

  rclcpp::Subscription<fa_interfaces::msg::AudioFrame>::SharedPtr audio_sub_;
  rclcpp::Service<fa_interfaces::srv::Record>::SharedPtr record_srv_;

  std::mutex mutex_;
  bool recording_{false};
  std::string record_path_;

  std::unique_ptr<backends::FileWriterBackend> writer_;
  backends::AudioFormat last_seen_format_{};
  bool has_last_seen_format_{false};
};

}  // namespace fa_record

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<fa_record::FaRecordNode>());
  rclcpp::shutdown();
  return 0;
}

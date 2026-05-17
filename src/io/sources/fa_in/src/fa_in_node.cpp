#include "fa_in/fa_in_node.hpp"

#include "fa_in/audio_config_validation.hpp"

#include <functional>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "fa_in/backends/alsa_capture_backend.hpp"

namespace fa_in
{

namespace
{
constexpr const char * kEncodingPcm16 = "PCM16LE";
constexpr const char * kEncodingPcm32 = "PCM32LE";
constexpr const char * kEncodingFloat32 = "FLOAT32LE";
constexpr const char * kInterleavedLayout = "interleaved";

bool isSupportedEncodingPair(const AudioConfig & config)
{
  return (config.encoding == kEncodingPcm16 && config.bit_depth == 16) ||
         (config.encoding == kEncodingPcm32 && config.bit_depth == 32) ||
         (config.encoding == kEncodingFloat32 && config.bit_depth == 32);
}

backends::AudioFormat backendFormatFromConfig(const AudioConfig & config)
{
  return backends::AudioFormat{
    config.sample_rate,
    config.channels,
    config.bit_depth,
    config.chunk_ms,
    config.encoding,
    config.layout};
}

backends::DeviceSelector backendSelectorFromConfig(const AudioConfig & config)
{
  return backends::DeviceSelector{
    config.device_mode,
    config.device_identifier,
    config.device_index};
}

std::string displayName(const backends::DeviceInfo & device)
{
  if (device.name.empty()) {
    return device.id;
  }
  return device.name;
}
}  // namespace

FaInNode::FaInNode()
: rclcpp::Node("fa_in_node")
{
  RCLCPP_INFO(this->get_logger(), "Initializing FA In node");
  loadParameters();

  audio_pub_ = this->create_publisher<fa_interfaces::msg::AudioFrame>("audio/frame", rclcpp::SensorDataQoS());
  diag_pub_ = this->create_publisher<diagnostic_msgs::msg::DiagnosticArray>("diagnostics", rclcpp::SystemDefaultsQoS());

  list_devices_srv_ = this->create_service<fa_interfaces::srv::ListDevices>(
    "list_devices",
    std::bind(&FaInNode::handleListDevices, this, std::placeholders::_1, std::placeholders::_2));

  switch_device_srv_ = this->create_service<fa_interfaces::srv::SwitchDevice>(
    "switch_device",
    std::bind(&FaInNode::handleSwitchDevice, this, std::placeholders::_1, std::placeholders::_2));

  diag_timer_ = this->create_wall_timer(
    std::chrono::milliseconds(config_.diag_period_ms),
    std::bind(&FaInNode::publishDiagnostics, this));

  initializeBackend();
  if (!configureDevice()) {
    throw std::runtime_error("Failed to configure audio input source");
  }
  startCaptureThread();
}

FaInNode::~FaInNode()
{
  stopCaptureThread();
  shutdownBackend();
}

bool FaInNode::hasFatalError() const
{
  return fatal_error_.load();
}

void FaInNode::loadParameters()
{
  this->declare_parameter<std::string>("backend.name");
  this->declare_parameter<std::string>("audio.device_selector.mode");
  this->declare_parameter("audio.device_selector.identifier", config_.device_identifier);
  this->declare_parameter<int>("audio.device_selector.index", config_.device_index);
  this->declare_parameter<int>("audio.sample_rate");
  this->declare_parameter<int>("audio.channels");
  this->declare_parameter<int>("audio.bit_depth");
  this->declare_parameter<int>("audio.chunk_ms");
  this->declare_parameter<std::string>("audio.encoding");
  this->declare_parameter<std::string>("audio.stream_id");
  this->declare_parameter<std::string>("audio.layout");
  this->declare_parameter<int>("diagnostics.publish_period_ms");

  config_.backend_name = this->get_parameter("backend.name").as_string();
  config_.device_mode = this->get_parameter("audio.device_selector.mode").as_string();
  config_.device_identifier = this->get_parameter("audio.device_selector.identifier").as_string();
  config_.device_index = this->get_parameter("audio.device_selector.index").as_int();
  config_.sample_rate = validation::requirePositiveUint32(
    "audio.sample_rate", this->get_parameter("audio.sample_rate").as_int());
  config_.channels = validation::requirePositiveUint32(
    "audio.channels", this->get_parameter("audio.channels").as_int());
  config_.bit_depth = validation::requirePositiveUint32(
    "audio.bit_depth", this->get_parameter("audio.bit_depth").as_int());
  config_.chunk_ms = validation::requirePositiveUint32(
    "audio.chunk_ms", this->get_parameter("audio.chunk_ms").as_int());
  config_.encoding = this->get_parameter("audio.encoding").as_string();
  config_.stream_id = this->get_parameter("audio.stream_id").as_string();
  config_.layout = this->get_parameter("audio.layout").as_string();
  config_.diag_period_ms = validation::requirePositiveUint32(
    "diagnostics.publish_period_ms",
    this->get_parameter("diagnostics.publish_period_ms").as_int());

  validation::requireDeviceSelector(
    config_.device_mode, config_.device_identifier, config_.device_index);

  if (config_.backend_name.empty()) {
    throw std::runtime_error("backend.name is required");
  }
  if (config_.backend_name != "alsa_capture") {
    throw std::runtime_error("unsupported fa_in backend.name: " + config_.backend_name);
  }
  if (config_.stream_id.empty()) {
    throw std::runtime_error("audio.stream_id is required");
  }
  if (config_.layout != kInterleavedLayout) {
    throw std::runtime_error("audio.layout must be interleaved for backend.name=alsa_capture");
  }
  if (!isSupportedEncodingPair(config_)) {
    throw std::runtime_error(
      "audio.encoding/audio.bit_depth must be one of PCM16LE/16, PCM32LE/32, FLOAT32LE/32");
  }

  frames_per_buffer_ = validation::captureFramesPerBuffer(config_.sample_rate, config_.chunk_ms);
  bytes_per_frame_ = validation::bytesPerFrame(config_.channels, config_.bit_depth);
  bytes_per_buffer_ = validation::bytesForFrames(
    "audio.chunk_ms", frames_per_buffer_, bytes_per_frame_);
  last_frame_time_ = std::chrono::steady_clock::now();

  RCLCPP_INFO(this->get_logger(), "Audio configuration: backend.name=%s mode=%s rate=%uHz channels=%u bits=%u chunk=%ums",
    config_.backend_name.c_str(), config_.device_mode.c_str(), config_.sample_rate, config_.channels,
    config_.bit_depth, config_.chunk_ms);
}

void FaInNode::initializeBackend()
{
  if (config_.backend_name == "alsa_capture") {
    source_backend_ = std::make_unique<backends::AlsaCaptureBackend>();
    return;
  }
  throw std::runtime_error("unsupported fa_in backend.name: " + config_.backend_name);
}

void FaInNode::shutdownBackend()
{
  if (source_backend_) {
    source_backend_->close();
  }
}

void FaInNode::failClosed(const std::string &reason)
{
  if (fatal_error_.exchange(true)) {
    return;
  }

  RCLCPP_FATAL(this->get_logger(), "Failing closed: %s", reason.c_str());
  capturing_.store(false);
  if (source_backend_) {
    source_backend_->close();
  }
  rclcpp::shutdown();
}

bool FaInNode::configureDevice()
{
  backends::DeviceInfo device;
  try {
    device = determineDeviceFromConfig();
  } catch (const backends::BackendError &e) {
    RCLCPP_ERROR(this->get_logger(), "%s", e.what());
    return false;
  }

  if (!reopenStream(device.id)) {
    RCLCPP_ERROR(this->get_logger(), "Failed to open audio input source %s", device.id.c_str());
    return false;
  }

  active_device_name_ = displayName(device);
  active_device_id_ = device.id;
  RCLCPP_INFO(
    this->get_logger(),
    "Using ALSA device: %s (%s)",
    active_device_id_.c_str(),
    active_device_name_.c_str());
  return true;
}

bool FaInNode::reopenStream(const std::string &device_id)
{
  stopCaptureThread();

  if (!source_backend_) {
    RCLCPP_ERROR(this->get_logger(), "source backend is not initialized");
    return false;
  }

  try {
    frames_per_buffer_ = source_backend_->open(device_id, backendFormatFromConfig(config_), frames_per_buffer_);
  } catch (const backends::BackendError &e) {
    RCLCPP_ERROR(this->get_logger(), "%s", e.what());
    return false;
  }

  return true;
}

void FaInNode::startCaptureThread()
{
  if (capturing_.load()) {
    return;
  }
  capturing_.store(true);
  capture_thread_ = std::thread(&FaInNode::captureLoop, this);
}

void FaInNode::stopCaptureThread()
{
  if (!capturing_.load()) {
    if (capture_thread_.joinable()) {
      capture_thread_.join();
    }
    return;
  }
  capturing_.store(false);
  if (capture_thread_.joinable()) {
    capture_thread_.join();
  }
  if (source_backend_) {
    source_backend_->drop();
  }
}

void FaInNode::captureLoop()
{
  if (!source_backend_) {
    failClosed("capture loop started without required source backend");
    return;
  }

  std::vector<uint8_t> buffer(bytes_per_buffer_);

  while (rclcpp::ok() && capturing_.load()) {
    const auto read_result = source_backend_->read(buffer.data(), frames_per_buffer_);
    if (read_result.status == backends::ReadStatus::kXrun) {
      xruns_.fetch_add(1);
      failClosed("ALSA capture XRUN on required input source " + active_device_id_);
      break;
    } else if (read_result.status == backends::ReadStatus::kError) {
      failClosed(
        "snd_pcm_readi failed on required input source " + active_device_id_ + ": " + read_result.message);
      break;
    } else if (read_result.status == backends::ReadStatus::kZeroFrames) {
      failClosed("snd_pcm_readi returned zero frames on required input source " + active_device_id_);
      break;
    }

    if (read_result.frames != frames_per_buffer_) {
      failClosed(
        "snd_pcm_readi returned " + std::to_string(read_result.frames) +
        " frames, expected configured capture chunk " + std::to_string(frames_per_buffer_) +
        " on required input source " + active_device_id_);
      break;
    }

    size_t byte_count = validation::bytesForFrames(
      "captured frame count", read_result.frames, bytes_per_frame_);
    publishFrame(buffer.data(), byte_count);
    frames_published_.fetch_add(1);
    last_frame_time_ = std::chrono::steady_clock::now();
  }
}

void FaInNode::publishFrame(const uint8_t *data, size_t data_size)
{
  fa_interfaces::msg::AudioFrame frame_msg;
  frame_msg.header.stamp = this->now();
  frame_msg.source_id = active_device_id_;
  frame_msg.stream_id = config_.stream_id;
  frame_msg.encoding = config_.encoding;
  frame_msg.sample_rate = config_.sample_rate;
  frame_msg.channels = config_.channels;
  frame_msg.bit_depth = config_.bit_depth;
  frame_msg.layout = config_.layout;
  frame_msg.data.assign(data, data + data_size);
  audio_pub_->publish(frame_msg);
}

void FaInNode::publishDiagnostics()
{
  diagnostic_msgs::msg::DiagnosticArray array_msg;
  array_msg.header.stamp = this->now();
  diagnostic_msgs::msg::DiagnosticStatus status;
  status.name = "fa_in_node";
  status.hardware_id = active_device_name_;
  status.level = capturing_.load() ? diagnostic_msgs::msg::DiagnosticStatus::OK
                                   : diagnostic_msgs::msg::DiagnosticStatus::WARN;
  status.message = capturing_.load() ? "running" : "stopped";

  auto now = std::chrono::steady_clock::now();
  double age_ms =
    std::chrono::duration_cast<std::chrono::milliseconds>(now - last_frame_time_).count();

  auto push_kv = [&status](const std::string &key, const std::string &value) {
    diagnostic_msgs::msg::KeyValue kv;
    kv.key = key;
    kv.value = value;
    status.values.push_back(kv);
  };

  status.values.reserve(6);
  push_kv("device_id", active_device_id_);
  push_kv("sample_rate", std::to_string(config_.sample_rate));
  push_kv("chunk_ms", std::to_string(config_.chunk_ms));
  push_kv("frames_published", std::to_string(frames_published_.load()));
  push_kv("xruns", std::to_string(xruns_.load()));
  push_kv("last_frame_age_ms", std::to_string(age_ms));

  array_msg.status.push_back(status);
  diag_pub_->publish(array_msg);
}

std::vector<backends::DeviceInfo> FaInNode::enumerateCaptureDevices() const
{
  if (!source_backend_) {
    throw backends::BackendError("source backend is not initialized");
  }
  return source_backend_->listDevices();
}

backends::DeviceInfo FaInNode::determineDeviceFromConfig()
{
  if (!source_backend_) {
    throw backends::BackendError("source backend is not initialized");
  }
  return source_backend_->selectDevice(backendSelectorFromConfig(config_));
}

void FaInNode::handleListDevices(
  const std::shared_ptr<fa_interfaces::srv::ListDevices::Request> /*request*/,
  std::shared_ptr<fa_interfaces::srv::ListDevices::Response> response)
{
  std::vector<backends::DeviceInfo> devices;
  try {
    devices = enumerateCaptureDevices();
  } catch (const backends::BackendError & e) {
    response->success = false;
    response->message = e.what();
    response->active_device_id = active_device_id_;
    response->active_device_name = active_device_name_;
    response->active_device_index = -1;
    return;
  }

  response->success = true;
  response->message = "ok";

  response->active_device_id = active_device_id_;
  response->active_device_name = active_device_name_;
  response->active_device_index = -1;

  int index = 0;
  for (const auto &dev : devices) {
    response->device_ids.push_back(dev.id);
    response->device_names.push_back(displayName(dev));
    response->host_api_names.push_back("ALSA");
    response->device_indices.push_back(index++);
    response->max_input_channels.push_back(config_.channels);
    response->default_sample_rates.push_back(config_.sample_rate);
  }

  if (!active_device_id_.empty()) {
    for (size_t i = 0; i < devices.size(); ++i) {
      if (devices[i].id == active_device_id_) {
        response->active_device_index = static_cast<int32_t>(i);
        if (response->active_device_name.empty()) {
          response->active_device_name = displayName(devices[i]);
        }
        break;
      }
    }
  }
}

void FaInNode::handleSwitchDevice(
  const std::shared_ptr<fa_interfaces::srv::SwitchDevice::Request> request,
  std::shared_ptr<fa_interfaces::srv::SwitchDevice::Response> response)
{
  try {
    validation::requireExactlyOneSwitchDeviceSelector(
      request->target_identifier,
      request->target_index);
  } catch (const std::runtime_error & e) {
    response->success = false;
    response->message = e.what();
    return;
  }

  std::vector<backends::DeviceInfo> devices;
  try {
    devices = enumerateCaptureDevices();
  } catch (const backends::BackendError & e) {
    response->success = false;
    response->message = e.what();
    return;
  }

  std::string device_id;
  std::string device_name;

  if (request->target_index >= 0) {
    if (static_cast<size_t>(request->target_index) >= devices.size()) {
      response->success = false;
      response->message = "device index not found";
      return;
    }
    device_id = devices[request->target_index].id;
    device_name = displayName(devices[request->target_index]);
  } else {
    for (const auto &dev : devices) {
      if (dev.id == request->target_identifier) {
        device_id = dev.id;
        device_name = displayName(dev);
        break;
      }
    }
    if (device_id.empty()) {
      std::vector<backends::DeviceInfo> display_name_matches;
      for (const auto &dev : devices) {
        if (displayName(dev) == request->target_identifier) {
          display_name_matches.push_back(dev);
        }
      }
      if (display_name_matches.size() == 1) {
        device_id = display_name_matches[0].id;
        device_name = displayName(display_name_matches[0]);
      } else if (display_name_matches.size() > 1) {
        response->success = false;
        response->message = "device name is ambiguous";
        return;
      }
    }
  }

  if (device_id.empty()) {
    response->success = false;
    response->message = "device not found";
    return;
  }

  if (!request->restart) {
    response->success = true;
    response->message = "device selected (restart=false)";
    return;
  }

  if (!reopenStream(device_id)) {
    response->success = false;
    response->message = "failed to open device; fa_in shutting down";
    failClosed("failed to reopen required audio input source " + device_id);
    return;
  }

  active_device_id_ = device_id;
  active_device_name_ = device_name.empty() ? device_id : device_name;

  startCaptureThread();
  response->success = true;
  response->message = "switched";
}
}  // namespace fa_in

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<fa_in::FaInNode>();
    rclcpp::spin(node);
    const bool fatal_error = node->hasFatalError();
    rclcpp::shutdown();
    return fatal_error ? EXIT_FAILURE : EXIT_SUCCESS;
  } catch (const std::exception &e) {
    RCLCPP_FATAL(rclcpp::get_logger("fa_in_node"), "Exception: %s", e.what());
    rclcpp::shutdown();
    return EXIT_FAILURE;
  }
}

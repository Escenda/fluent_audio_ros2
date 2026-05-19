#include "fa_in/fa_in_node.hpp"

#include "fa_in/audio_config_validation.hpp"
#include "fa_in/backends/factory.hpp"

#include <functional>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <utility>

#include "rclcpp/exceptions.hpp"

namespace fa_in
{

namespace
{
constexpr const char * kEncodingPcm16 = "PCM16LE";
constexpr const char * kEncodingPcm32 = "PCM32LE";
constexpr const char * kEncodingFloat32 = "FLOAT32LE";
constexpr const char * kInterleavedLayout = "interleaved";
constexpr const char * kBackendAlsaCapture = "alsa_capture";
constexpr const char * kBackendPcmFileReader = "pcm_file_reader";
constexpr const char * kBackendNetworkPcmReceiver = "network_pcm_receiver";

bool isSupportedEncodingPair(const AudioConfig & config)
{
  return (config.encoding == kEncodingPcm16 && config.bit_depth == 16) ||
         (config.encoding == kEncodingPcm32 && config.bit_depth == 32) ||
         (config.encoding == kEncodingFloat32 && config.bit_depth == 32);
}

bool isVariablePacketBackend(const std::string & backend_name)
{
  return backend_name == kBackendPcmFileReader || backend_name == kBackendNetworkPcmReceiver;
}

backends::AudioFormat backendFormatFromConfig(const AudioConfig & config)
{
  return backends::AudioFormat{
    config.sample_rate,
    config.channels,
    config.bit_depth,
    config.chunk_ms,
    config.encoding,
    config.layout,
    config.playback_loop};
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
  return device.name;
}

std::string removeLeadingSlashes(std::string value)
{
  while (!value.empty() && value.front() == '/') {
    value.erase(value.begin());
  }
  return value;
}

bool sameIdentityString(const std::string & left, const std::string & right)
{
  return left == right || removeLeadingSlashes(left) == removeLeadingSlashes(right);
}

bool isRequiredParameterSet(const rclcpp::Parameter & parameter)
{
  return parameter.get_type() != rclcpp::ParameterType::PARAMETER_NOT_SET;
}

rclcpp::QoS makeExplicitQos(uint32_t depth, bool reliable)
{
  if (depth == 0) {
    throw std::runtime_error("qos depth must be greater than zero");
  }
  rclcpp::QoS qos(static_cast<size_t>(depth));
  if (reliable) {
    qos.reliable();
  } else {
    qos.best_effort();
  }
  return qos;
}

rclcpp::Parameter getRequiredParameter(const rclcpp::Node & node, const std::string & name)
{
  rclcpp::Parameter parameter;
  bool has_parameter = false;
  try {
    has_parameter = node.get_parameter(name, parameter);
  } catch (const rclcpp::exceptions::ParameterUninitializedException &) {
    throw std::runtime_error(name + " is required");
  }
  if (!has_parameter || !isRequiredParameterSet(parameter)) {
    throw std::runtime_error(name + " is required");
  }
  return parameter;
}

std::string readRequiredString(const rclcpp::Node & node, const std::string & name)
{
  const rclcpp::Parameter parameter = getRequiredParameter(node, name);
  if (parameter.get_type() != rclcpp::ParameterType::PARAMETER_STRING) {
    throw std::runtime_error(name + " must be a string");
  }
  return parameter.as_string();
}

int readRequiredInt(const rclcpp::Node & node, const std::string & name)
{
  const rclcpp::Parameter parameter = getRequiredParameter(node, name);
  if (parameter.get_type() != rclcpp::ParameterType::PARAMETER_INTEGER) {
    throw std::runtime_error(name + " must be an integer");
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
    throw std::runtime_error(name + " must be a bool");
  }
  return parameter.as_bool();
}

rcl_interfaces::msg::ParameterDescriptor dynamicParameterDescriptor()
{
  rcl_interfaces::msg::ParameterDescriptor descriptor;
  descriptor.dynamic_typing = true;
  return descriptor;
}
}  // namespace

FaInNode::FaInNode(const rclcpp::NodeOptions & options)
: FaInNode(options, backends::defaultAlsaCaptureBackendFactory())
{
}

FaInNode::FaInNode(const rclcpp::NodeOptions & options, BackendFactory backend_factory)
: rclcpp::Node("fa_in", options), backend_factory_(std::move(backend_factory))
{
  RCLCPP_INFO(this->get_logger(), "Initializing FA In node");
  if (!backend_factory_) {
    throw std::invalid_argument("fa_in backend factory is required");
  }
  loadParameters();

  audio_pub_ = this->create_publisher<fa_interfaces::msg::AudioFrame>(
    config_.output_topic,
    makeExplicitQos(config_.audio_qos_depth, config_.audio_qos_reliable));

  const rclcpp::QoS diagnostics_qos =
    makeExplicitQos(config_.diagnostics_qos_depth, config_.diagnostics_qos_reliable);
  diag_pub_ = this->create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
    "diagnostics",
    diagnostics_qos);

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
  const auto dynamic_parameter = dynamicParameterDescriptor();
  this->declare_parameter("backend.name", rclcpp::ParameterValue{}, dynamic_parameter);
  this->declare_parameter<std::string>("output_topic");
  this->declare_parameter("audio.device_selector.mode", rclcpp::ParameterValue{}, dynamic_parameter);
  this->declare_parameter("audio.device_selector.identifier", rclcpp::ParameterValue{}, dynamic_parameter);
  this->declare_parameter("audio.device_selector.index", rclcpp::ParameterValue{}, dynamic_parameter);
  this->declare_parameter("file.path", rclcpp::ParameterValue{}, dynamic_parameter);
  this->declare_parameter("endpoint.uri", rclcpp::ParameterValue{}, dynamic_parameter);
  this->declare_parameter("transport.identity", rclcpp::ParameterValue{}, dynamic_parameter);
  this->declare_parameter("audio.source_id", rclcpp::ParameterValue{}, dynamic_parameter);
  this->declare_parameter("playback.loop", rclcpp::ParameterValue{}, dynamic_parameter);
  this->declare_parameter("network.max_packet_bytes", rclcpp::ParameterValue{}, dynamic_parameter);
  this->declare_parameter("polling.period_ms", rclcpp::ParameterValue{}, dynamic_parameter);
  this->declare_parameter<int>("audio.sample_rate");
  this->declare_parameter<int>("audio.channels");
  this->declare_parameter<int>("audio.bit_depth");
  this->declare_parameter<int>("audio.chunk_ms");
  this->declare_parameter<std::string>("audio.encoding");
  this->declare_parameter<std::string>("audio.stream_id");
  this->declare_parameter<std::string>("audio.layout");
  this->declare_parameter<int>("audio.qos.depth");
  this->declare_parameter<bool>("audio.qos.reliable");
  this->declare_parameter<int>("diagnostics.qos.depth");
  this->declare_parameter<bool>("diagnostics.qos.reliable");
  this->declare_parameter<int>("diagnostics.publish_period_ms");

  config_.backend_name = readRequiredString(*this, "backend.name");
  config_.output_topic = readRequiredString(*this, "output_topic");
  if (config_.backend_name == kBackendAlsaCapture) {
    config_.device_mode = readRequiredString(*this, "audio.device_selector.mode");
    config_.device_identifier = readRequiredString(*this, "audio.device_selector.identifier");
    config_.device_index = readRequiredInt(*this, "audio.device_selector.index");
  } else if (config_.backend_name == kBackendPcmFileReader) {
    config_.file_path = readRequiredString(*this, "file.path");
    config_.source_id = readRequiredString(*this, "audio.source_id");
    config_.playback_loop = readRequiredBool(*this, "playback.loop");
  } else if (config_.backend_name == kBackendNetworkPcmReceiver) {
    config_.endpoint_uri = readRequiredString(*this, "endpoint.uri");
    config_.transport_identity = readRequiredString(*this, "transport.identity");
    config_.source_id = readRequiredString(*this, "audio.source_id");
    config_.network_max_packet_bytes = validation::requirePositiveUint32(
      "network.max_packet_bytes",
      readRequiredInt(*this, "network.max_packet_bytes"));
    config_.polling_period_ms = validation::requirePositiveUint32(
      "polling.period_ms",
      readRequiredInt(*this, "polling.period_ms"));
  }
  config_.sample_rate = validation::requirePositiveUint32(
    "audio.sample_rate", readRequiredInt(*this, "audio.sample_rate"));
  config_.channels = validation::requirePositiveUint32(
    "audio.channels", readRequiredInt(*this, "audio.channels"));
  config_.bit_depth = validation::requirePositiveUint32(
    "audio.bit_depth", readRequiredInt(*this, "audio.bit_depth"));
  config_.chunk_ms = validation::requirePositiveUint32(
    "audio.chunk_ms", readRequiredInt(*this, "audio.chunk_ms"));
  config_.encoding = readRequiredString(*this, "audio.encoding");
  config_.stream_id = readRequiredString(*this, "audio.stream_id");
  config_.layout = readRequiredString(*this, "audio.layout");
  config_.audio_qos_depth = validation::requirePositiveUint32(
    "audio.qos.depth",
    readRequiredInt(*this, "audio.qos.depth"));
  config_.audio_qos_reliable = readRequiredBool(*this, "audio.qos.reliable");
  config_.diagnostics_qos_depth = validation::requirePositiveUint32(
    "diagnostics.qos.depth",
    readRequiredInt(*this, "diagnostics.qos.depth"));
  config_.diagnostics_qos_reliable = readRequiredBool(*this, "diagnostics.qos.reliable");
  config_.diag_period_ms = validation::requirePositiveUint32(
    "diagnostics.publish_period_ms",
    readRequiredInt(*this, "diagnostics.publish_period_ms"));

  if (config_.backend_name.empty()) {
    throw std::runtime_error("backend.name is required");
  }
  if (config_.backend_name != kBackendAlsaCapture && config_.backend_name != kBackendPcmFileReader &&
      config_.backend_name != kBackendNetworkPcmReceiver)
  {
    throw std::runtime_error("unsupported fa_in backend.name: " + config_.backend_name);
  }
  if (config_.backend_name == kBackendAlsaCapture) {
    validation::requireDeviceSelector(
      config_.device_mode, config_.device_identifier, config_.device_index);
  }
  if (config_.backend_name == kBackendPcmFileReader && config_.file_path.empty()) {
    throw std::runtime_error("file.path is required for backend.name=pcm_file_reader");
  }
  if (config_.backend_name == kBackendPcmFileReader && config_.source_id.empty()) {
    throw std::runtime_error("audio.source_id is required for backend.name=pcm_file_reader");
  }
  if (config_.backend_name == kBackendNetworkPcmReceiver && config_.endpoint_uri.empty()) {
    throw std::runtime_error("endpoint.uri is required for backend.name=network_pcm_receiver");
  }
  if (config_.backend_name == kBackendNetworkPcmReceiver && config_.transport_identity.empty()) {
    throw std::runtime_error("transport.identity is required for backend.name=network_pcm_receiver");
  }
  if (config_.backend_name == kBackendNetworkPcmReceiver && config_.source_id.empty()) {
    throw std::runtime_error("audio.source_id is required for backend.name=network_pcm_receiver");
  }
  if (config_.output_topic.empty()) {
    throw std::runtime_error("output_topic is required");
  }
  if (config_.stream_id.empty()) {
    throw std::runtime_error("audio.stream_id is required");
  }
  const std::string resolved_output_topic =
    this->get_node_topics_interface()->resolve_topic_name(config_.output_topic);
  if (sameIdentityString(config_.stream_id, config_.output_topic) ||
      sameIdentityString(config_.stream_id, resolved_output_topic))
  {
    throw std::runtime_error("audio.stream_id must be distinct from ROS output_topic");
  }
  if (config_.layout != kInterleavedLayout) {
    throw std::runtime_error("audio.layout must be interleaved");
  }
  if (!isSupportedEncodingPair(config_)) {
    throw std::runtime_error(
      "audio.encoding/audio.bit_depth must be one of PCM16LE/16, PCM32LE/32, FLOAT32LE/32");
  }

  bytes_per_frame_ = validation::bytesPerFrame(config_.channels, config_.bit_depth);
  if (config_.backend_name == kBackendNetworkPcmReceiver) {
    if ((config_.network_max_packet_bytes % bytes_per_frame_) != 0) {
      throw std::runtime_error("network.max_packet_bytes must be divisible by expected frame byte size");
    }
    frames_per_buffer_ = config_.network_max_packet_bytes / bytes_per_frame_;
    bytes_per_buffer_ = config_.network_max_packet_bytes;
  } else {
    frames_per_buffer_ = validation::captureFramesPerBuffer(config_.sample_rate, config_.chunk_ms);
    bytes_per_buffer_ = validation::bytesForFrames(
      "audio.chunk_ms", frames_per_buffer_, bytes_per_frame_);
  }
  last_frame_time_ = std::chrono::steady_clock::now();

  RCLCPP_INFO(this->get_logger(), "Audio configuration: backend.name=%s rate=%uHz channels=%u bits=%u chunk=%ums",
    config_.backend_name.c_str(), config_.sample_rate, config_.channels,
    config_.bit_depth, config_.chunk_ms);
}

void FaInNode::initializeBackend()
{
  source_backend_ = backends::buildSourceBackend(
    backends::SourceBackendSettings{config_.backend_name},
    backend_factory_);
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
  active_source_id_ = config_.backend_name == kBackendAlsaCapture ? active_device_id_ : config_.source_id;
  RCLCPP_INFO(
    this->get_logger(),
    "Using input source: %s (%s)",
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
    if (read_result.status == backends::ReadStatus::kNoData) {
      if (config_.backend_name != kBackendNetworkPcmReceiver) {
        failClosed("source backend returned no data on required input source " + active_device_id_);
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(config_.polling_period_ms));
      continue;
    } else if (read_result.status == backends::ReadStatus::kXrun) {
      xruns_.fetch_add(1);
      failClosed("ALSA capture XRUN on required input source " + active_device_id_);
      break;
    } else if (read_result.status == backends::ReadStatus::kError) {
      const std::string read_label =
        config_.backend_name == kBackendAlsaCapture ? "snd_pcm_readi failed" : "source backend read failed";
      failClosed(read_label + " on required input source " + active_device_id_ + ": " + read_result.message);
      break;
    } else if (read_result.status == backends::ReadStatus::kZeroFrames) {
      failClosed("snd_pcm_readi returned zero frames on required input source " + active_device_id_);
      break;
    } else if (read_result.status == backends::ReadStatus::kEndOfStream) {
      capturing_.store(false);
      break;
    }

    if (read_result.frames != frames_per_buffer_ && !isVariablePacketBackend(config_.backend_name)) {
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
  frame_msg.source_id = active_source_id_;
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
  status.name = "fa_in";
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

  status.values.reserve(14);
  push_kv("device_id", active_device_id_);
  push_kv("source_id", active_source_id_);
  push_kv("output_topic", config_.output_topic);
  push_kv("stream_id", config_.stream_id);
  push_kv("sample_rate", std::to_string(config_.sample_rate));
  push_kv("channels", std::to_string(config_.channels));
  push_kv("encoding", config_.encoding);
  push_kv("chunk_ms", std::to_string(config_.chunk_ms));
  push_kv("frames_published", std::to_string(frames_published_.load()));
  push_kv("xruns", std::to_string(xruns_.load()));
  push_kv("last_frame_age_ms", std::to_string(age_ms));
  push_kv("audio.qos.depth", std::to_string(config_.audio_qos_depth));
  push_kv("audio.qos.reliable", config_.audio_qos_reliable ? "true" : "false");
  push_kv("diagnostics.qos.depth", std::to_string(config_.diagnostics_qos_depth));
  push_kv("diagnostics.qos.reliable", config_.diagnostics_qos_reliable ? "true" : "false");

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
  if (config_.backend_name == kBackendPcmFileReader) {
    return backends::DeviceInfo{config_.file_path, config_.source_id, config_.channels, config_.sample_rate};
  }
  if (config_.backend_name == kBackendNetworkPcmReceiver) {
    return backends::DeviceInfo{
      config_.endpoint_uri,
      config_.transport_identity,
      config_.channels,
      config_.sample_rate};
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
    response->max_input_channels.push_back(dev.max_input_channels);
    response->default_sample_rates.push_back(dev.default_sample_rate);
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
    if (config_.backend_name != kBackendAlsaCapture) {
      response->success = false;
      response->message = "switch_device is only supported for backend.name=alsa_capture";
      return;
    }
    validation::requireSwitchDeviceSelector(
      request->target_selector_mode,
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

  if (request->target_selector_mode == "index") {
    if (static_cast<size_t>(request->target_index) >= devices.size()) {
      response->success = false;
      response->message = "device index not found";
      return;
    }
    device_id = devices[request->target_index].id;
    device_name = displayName(devices[request->target_index]);
  } else if (request->target_selector_mode == "id") {
    for (const auto &dev : devices) {
      if (dev.id == request->target_identifier) {
        device_id = dev.id;
        device_name = displayName(dev);
        break;
      }
    }
    if (device_id.empty()) {
      response->success = false;
      response->message = "device id not found";
      return;
    }
  } else if (request->target_selector_mode == "name") {
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

  if (device_id.empty()) {
    response->success = false;
    response->message = "device not found";
    return;
  }

  if (!reopenStream(device_id)) {
    response->success = false;
    response->message = "failed to open device; fa_in shutting down";
    failClosed("failed to reopen required audio input source " + device_id);
    return;
  }

  active_device_id_ = device_id;
  active_device_name_ = device_name;

  startCaptureThread();
  response->success = true;
  response->message = "switched";
}
}  // namespace fa_in

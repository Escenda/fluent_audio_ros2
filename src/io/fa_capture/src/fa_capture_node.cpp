#include "fa_capture/fa_capture_node.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace fa_capture
{

namespace
{
constexpr double kInt16Scale = 32768.0;

void silenceAlsaErrors(const char * /*file*/, int /*line*/, const char * /*function*/,
                       int /*err*/, const char * /*fmt*/, ...)
{
  // prevent ALSA from printing to stderr when devices are unplugged
}
}  // namespace

FaCaptureNode::FaCaptureNode()
: rclcpp::Node("fa_capture_node")
{
  RCLCPP_INFO(this->get_logger(), "Initializing FA Capture node");
  loadParameters();

  audio_pub_ = this->create_publisher<fa_interfaces::msg::AudioFrame>("audio/frame", rclcpp::SensorDataQoS());
  levels_pub_ = this->create_publisher<std_msgs::msg::Float32MultiArray>("audio/levels", rclcpp::SystemDefaultsQoS());
  diag_pub_ = this->create_publisher<diagnostic_msgs::msg::DiagnosticArray>("diagnostics", rclcpp::SystemDefaultsQoS());

  list_devices_srv_ = this->create_service<fa_interfaces::srv::ListDevices>(
    "list_devices",
    std::bind(&FaCaptureNode::handleListDevices, this, std::placeholders::_1, std::placeholders::_2));

  switch_device_srv_ = this->create_service<fa_interfaces::srv::SwitchDevice>(
    "switch_device",
    std::bind(&FaCaptureNode::handleSwitchDevice, this, std::placeholders::_1, std::placeholders::_2));

  diag_timer_ = this->create_wall_timer(
    std::chrono::milliseconds(config_.diag_period_ms),
    std::bind(&FaCaptureNode::publishDiagnostics, this));

  initializeAlsa();
  if (configureDevice()) {
    startCaptureThread();
  } else {
    RCLCPP_ERROR(this->get_logger(), "Failed to configure audio device");
  }
}

FaCaptureNode::~FaCaptureNode()
{
  stopCaptureThread();
  shutdownAlsa();
}

void FaCaptureNode::loadParameters()
{
  this->declare_parameter("audio.device_selector.mode", config_.device_mode);
  this->declare_parameter("audio.device_selector.identifier", config_.device_identifier);
  this->declare_parameter<int>("audio.device_selector.index", config_.device_index);
  this->declare_parameter<int>("audio.sample_rate", static_cast<int>(config_.sample_rate));
  this->declare_parameter<int>("audio.channels", static_cast<int>(config_.channels));
  this->declare_parameter<int>("audio.bit_depth", static_cast<int>(config_.bit_depth));
  this->declare_parameter<int>("audio.chunk_ms", static_cast<int>(config_.chunk_ms));
  this->declare_parameter("audio.encoding", config_.encoding);
  this->declare_parameter("pipeline.publish_waveform", config_.publish_waveform);
  this->declare_parameter("pipeline.gain_db", config_.gain_db);
  this->declare_parameter<int>("diagnostics.publish_period_ms", static_cast<int>(config_.diag_period_ms));

  config_.device_mode = this->get_parameter("audio.device_selector.mode").as_string();
  config_.device_identifier = this->get_parameter("audio.device_selector.identifier").as_string();
  config_.device_index = this->get_parameter("audio.device_selector.index").as_int();
  config_.sample_rate = this->get_parameter("audio.sample_rate").as_int();
  config_.channels = this->get_parameter("audio.channels").as_int();
  config_.bit_depth = this->get_parameter("audio.bit_depth").as_int();
  config_.chunk_ms = this->get_parameter("audio.chunk_ms").as_int();
  config_.encoding = this->get_parameter("audio.encoding").as_string();
  config_.publish_waveform = this->get_parameter("pipeline.publish_waveform").as_bool();
  config_.gain_db = this->get_parameter("pipeline.gain_db").as_double();
  config_.diag_period_ms = this->get_parameter("diagnostics.publish_period_ms").as_int();

  gain_linear_ = std::pow(10.0, config_.gain_db / 20.0);
  frames_per_buffer_ = std::max<uint32_t>(config_.sample_rate * config_.chunk_ms / 1000, 64u);
  bytes_per_frame_ = config_.channels * (config_.bit_depth / 8);
  last_frame_time_ = std::chrono::steady_clock::now();

  RCLCPP_INFO(this->get_logger(), "Audio configuration: mode=%s rate=%uHz channels=%u bits=%u chunk=%ums",
    config_.device_mode.c_str(), config_.sample_rate, config_.channels, config_.bit_depth, config_.chunk_ms);
}

void FaCaptureNode::initializeAlsa()
{
  snd_lib_error_set_handler(silenceAlsaErrors);
}

void FaCaptureNode::shutdownAlsa()
{
  if (pcm_handle_) {
    snd_pcm_drop(pcm_handle_);
    snd_pcm_close(pcm_handle_);
    pcm_handle_ = nullptr;
  }
}

bool FaCaptureNode::configureDevice()
{
  std::string device_id;
  std::string device_name;
  if (!determineDeviceFromConfig(device_id, device_name)) {
    RCLCPP_ERROR(this->get_logger(), "No suitable ALSA capture device found");
    return false;
  }

  if (!reopenStream(device_id)) {
    RCLCPP_ERROR(this->get_logger(), "Failed to open ALSA device %s", device_id.c_str());
    return false;
  }

  active_device_name_ = device_name.empty() ? device_id : device_name;
  active_device_id_ = device_id;
  RCLCPP_INFO(this->get_logger(), "Using ALSA device: %s (%s)", active_device_id_.c_str(), active_device_name_.c_str());
  return true;
}

bool FaCaptureNode::reopenStream(const std::string &device_id)
{
  stopCaptureThread();

  if (pcm_handle_) {
    snd_pcm_drop(pcm_handle_);
    snd_pcm_close(pcm_handle_);
    pcm_handle_ = nullptr;
  }

  std::string target = device_id.empty() ? "default" : device_id;
  int err = snd_pcm_open(&pcm_handle_, target.c_str(), SND_PCM_STREAM_CAPTURE, 0);
  if (err < 0) {
    RCLCPP_ERROR(this->get_logger(), "snd_pcm_open failed for %s: %s", target.c_str(), snd_strerror(err));
    pcm_handle_ = nullptr;
    return false;
  }

  snd_pcm_hw_params_t *params = nullptr;
  snd_pcm_hw_params_malloc(&params);
  snd_pcm_hw_params_any(pcm_handle_, params);
  snd_pcm_hw_params_set_access(pcm_handle_, params, SND_PCM_ACCESS_RW_INTERLEAVED);

  snd_pcm_format_t format = (config_.bit_depth == 16) ? SND_PCM_FORMAT_S16_LE : SND_PCM_FORMAT_FLOAT_LE;
  if ((err = snd_pcm_hw_params_set_format(pcm_handle_, params, format)) < 0) {
    RCLCPP_ERROR(this->get_logger(), "Failed to set format: %s", snd_strerror(err));
    snd_pcm_hw_params_free(params);
    snd_pcm_close(pcm_handle_);
    pcm_handle_ = nullptr;
    return false;
  }

  unsigned int channels = config_.channels;
  if ((err = snd_pcm_hw_params_set_channels_near(pcm_handle_, params, &channels)) < 0) {
    RCLCPP_ERROR(this->get_logger(), "Failed to set channels: %s", snd_strerror(err));
    snd_pcm_hw_params_free(params);
    snd_pcm_close(pcm_handle_);
    pcm_handle_ = nullptr;
    return false;
  }
  if (channels != config_.channels) {
    RCLCPP_WARN(this->get_logger(), "Requested channels %u, but device set %u", config_.channels, channels);
    config_.channels = channels;
    bytes_per_frame_ = config_.channels * (config_.bit_depth / 8);
  }

  unsigned int rate = config_.sample_rate;
  int dir = 0;
  if ((err = snd_pcm_hw_params_set_rate_near(pcm_handle_, params, &rate, &dir)) < 0) {
    RCLCPP_ERROR(this->get_logger(), "Failed to set rate: %s", snd_strerror(err));
    snd_pcm_hw_params_free(params);
    snd_pcm_close(pcm_handle_);
    pcm_handle_ = nullptr;
    return false;
  }
  if (rate != config_.sample_rate) {
    RCLCPP_WARN(this->get_logger(), "Requested sample rate %u, but device set %u", config_.sample_rate, rate);
    config_.sample_rate = rate;
    frames_per_buffer_ = std::max<uint32_t>(config_.sample_rate * config_.chunk_ms / 1000, 64u);
    bytes_per_frame_ = config_.channels * (config_.bit_depth / 8);
  }

  snd_pcm_uframes_t period_size = static_cast<snd_pcm_uframes_t>(frames_per_buffer_);
  if ((err = snd_pcm_hw_params_set_period_size_near(pcm_handle_, params, &period_size, &dir)) < 0) {
    RCLCPP_ERROR(this->get_logger(), "Failed to set period size: %s", snd_strerror(err));
    snd_pcm_hw_params_free(params);
    snd_pcm_close(pcm_handle_);
    pcm_handle_ = nullptr;
    return false;
  }

  if ((err = snd_pcm_hw_params(pcm_handle_, params)) < 0) {
    RCLCPP_ERROR(this->get_logger(), "Failed to apply hw params: %s", snd_strerror(err));
    snd_pcm_hw_params_free(params);
    snd_pcm_close(pcm_handle_);
    pcm_handle_ = nullptr;
    return false;
  }

  snd_pcm_hw_params_free(params);

  if ((err = snd_pcm_prepare(pcm_handle_)) < 0) {
    RCLCPP_ERROR(this->get_logger(), "snd_pcm_prepare failed: %s", snd_strerror(err));
    snd_pcm_close(pcm_handle_);
    pcm_handle_ = nullptr;
    return false;
  }

  frames_per_buffer_ = static_cast<size_t>(period_size);
  return true;
}

void FaCaptureNode::startCaptureThread()
{
  if (capturing_.load()) {
    return;
  }
  capturing_.store(true);
  capture_thread_ = std::thread(&FaCaptureNode::captureLoop, this);
}

void FaCaptureNode::stopCaptureThread()
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
  if (pcm_handle_) {
    snd_pcm_drop(pcm_handle_);
  }
}

void FaCaptureNode::captureLoop()
{
  if (!pcm_handle_) {
    RCLCPP_ERROR(this->get_logger(), "Capture loop started without ALSA handle");
    return;
  }

  std::vector<uint8_t> buffer(frames_per_buffer_ * bytes_per_frame_);

  while (rclcpp::ok() && capturing_.load()) {
    snd_pcm_sframes_t frames = snd_pcm_readi(pcm_handle_, buffer.data(), frames_per_buffer_);
    if (frames == -EPIPE) {
      snd_pcm_prepare(pcm_handle_);
      xruns_.fetch_add(1);
      continue;
    } else if (frames < 0) {
      RCLCPP_ERROR(this->get_logger(), "snd_pcm_readi failed: %s", snd_strerror(static_cast<int>(frames)));
      snd_pcm_prepare(pcm_handle_);
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
      continue;
    }

    size_t sample_count = static_cast<size_t>(frames) * config_.channels;
    size_t byte_count = static_cast<size_t>(frames) * bytes_per_frame_;
    double peak = 0.0;
    double rms = computeRms(buffer.data(), sample_count, peak);
    publishFrame(buffer.data(), byte_count, rms, peak);
    frames_published_.fetch_add(1);
    last_frame_time_ = std::chrono::steady_clock::now();
  }
}

void FaCaptureNode::publishFrame(const uint8_t *data, size_t data_size, double rms, double peak)
{
  fa_interfaces::msg::AudioFrame frame_msg;
  frame_msg.header.stamp = this->now();
  frame_msg.encoding = config_.encoding;
  frame_msg.sample_rate = config_.sample_rate;
  frame_msg.channels = config_.channels;
  frame_msg.bit_depth = config_.bit_depth;
  frame_msg.rms = static_cast<float>(rms);
  frame_msg.peak = static_cast<float>(peak);
  frame_msg.vad = false;
  frame_msg.data.assign(data, data + data_size);
  audio_pub_->publish(frame_msg);

  std_msgs::msg::Float32MultiArray levels_msg;
  levels_msg.data = {static_cast<float>(rms), static_cast<float>(peak)};
  levels_pub_->publish(levels_msg);

}

void FaCaptureNode::publishDiagnostics()
{
  diagnostic_msgs::msg::DiagnosticArray array_msg;
  array_msg.header.stamp = this->now();
  diagnostic_msgs::msg::DiagnosticStatus status;
  status.name = "fa_capture_node";
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

double FaCaptureNode::computeRms(const void *data, size_t samples, double &peak) const
{
  peak = 0.0;
  double accum = 0.0;
  if (config_.bit_depth == 16) {
    const int16_t *samples_ptr = static_cast<const int16_t *>(data);
    for (size_t i = 0; i < samples; ++i) {
      double normalized = samples_ptr[i] / kInt16Scale;
      peak = std::max(peak, std::abs(normalized));
      accum += normalized * normalized;
    }
  } else {
    const float *samples_ptr = static_cast<const float *>(data);
    for (size_t i = 0; i < samples; ++i) {
      double normalized = samples_ptr[i];
      peak = std::max(peak, std::abs(normalized));
      accum += normalized * normalized;
    }
  }

  if (samples == 0) {
    return 0.0;
  }
  return std::sqrt(accum / static_cast<double>(samples));
}

std::vector<std::pair<std::string, std::string>> FaCaptureNode::enumerateCaptureDevices() const
{
  std::vector<std::pair<std::string, std::string>> devices;
  void **hints = nullptr;
  if (snd_device_name_hint(-1, "pcm", &hints) != 0) {
    devices.emplace_back("default", "ALSA default");
    return devices;
  }

  for (void **hint = hints; *hint != nullptr; ++hint) {
    char *name = snd_device_name_get_hint(*hint, "NAME");
    char *desc = snd_device_name_get_hint(*hint, "DESC");
    char *io = snd_device_name_get_hint(*hint, "IOID");
    bool is_input = (io == nullptr) || (std::strcmp(io, "Input") == 0);
    if (name && is_input) {
      devices.emplace_back(std::string(name), desc ? std::string(desc) : "");
    }
    if (name) {
      free(name);
    }
    if (desc) {
      free(desc);
    }
    if (io) {
      free(io);
    }
  }
  snd_device_name_free_hint(hints);

  if (devices.empty()) {
    devices.emplace_back("default", "ALSA default");
  }

  return devices;
}

bool FaCaptureNode::determineDeviceFromConfig(std::string &device_id, std::string &device_name)
{
  auto devices = enumerateCaptureDevices();
  if (devices.empty()) {
    device_id = "default";
    device_name = "ALSA default";
    return true;
  }

  if (config_.device_mode == "index" &&
      config_.device_index >= 0 &&
      static_cast<size_t>(config_.device_index) < devices.size())
  {
    device_id = devices[config_.device_index].first;
    device_name = devices[config_.device_index].second.empty() ?
      devices[config_.device_index].first : devices[config_.device_index].second;
    return true;
  }

  if (config_.device_mode == "name" && !config_.device_identifier.empty()) {
    for (const auto &dev : devices) {
      std::string label = dev.second.empty() ? dev.first : dev.second;
      if (label.find(config_.device_identifier) != std::string::npos) {
        device_id = dev.first;
        device_name = label;
        return true;
      }
    }
  }

  for (const auto &dev : devices) {
    if (dev.first == "default") {
      device_id = dev.first;
      device_name = dev.second.empty() ? "default" : dev.second;
      return true;
    }
  }

  device_id = devices.front().first;
  device_name = devices.front().second.empty() ? devices.front().first : devices.front().second;
  return true;
}

void FaCaptureNode::handleListDevices(
  const std::shared_ptr<fa_interfaces::srv::ListDevices::Request> /*request*/,
  std::shared_ptr<fa_interfaces::srv::ListDevices::Response> response)
{
  auto devices = enumerateCaptureDevices();
  response->success = true;
  response->message = "ok";

  response->active_device_id = active_device_id_;
  response->active_device_name = active_device_name_;
  response->active_device_index = -1;

  int index = 0;
  for (const auto &dev : devices) {
    response->device_ids.push_back(dev.first);
    response->device_names.push_back(dev.second.empty() ? dev.first : dev.second);
    response->host_api_names.push_back("ALSA");
    response->device_indices.push_back(index++);
    response->max_input_channels.push_back(config_.channels);
    response->default_sample_rates.push_back(config_.sample_rate);
  }

  if (!active_device_id_.empty()) {
    for (size_t i = 0; i < devices.size(); ++i) {
      if (devices[i].first == active_device_id_) {
        response->active_device_index = static_cast<int32_t>(i);
        if (response->active_device_name.empty()) {
          response->active_device_name = devices[i].second.empty() ? devices[i].first : devices[i].second;
        }
        break;
      }
    }
  }
}

void FaCaptureNode::handleSwitchDevice(
  const std::shared_ptr<fa_interfaces::srv::SwitchDevice::Request> request,
  std::shared_ptr<fa_interfaces::srv::SwitchDevice::Response> response)
{
  auto devices = enumerateCaptureDevices();
  std::string device_id;
  std::string device_name;

  if (request->target_index >= 0 &&
      static_cast<size_t>(request->target_index) < devices.size()) {
    device_id = devices[request->target_index].first;
    device_name = devices[request->target_index].second.empty()
      ? device_id
      : devices[request->target_index].second;
  } else if (!request->target_identifier.empty()) {
    // Prefer matching device_id (ALSA identifier) directly.
    for (const auto &dev : devices) {
      if (dev.first == request->target_identifier) {
        device_id = dev.first;
        device_name = dev.second.empty() ? dev.first : dev.second;
        break;
      }
    }

    if (device_id.empty()) {
      for (const auto &dev : devices) {
        std::string label = dev.second.empty() ? dev.first : dev.second;
        if (dev.first.find(request->target_identifier) != std::string::npos ||
            label.find(request->target_identifier) != std::string::npos) {
          device_id = dev.first;
          device_name = label;
          break;
        }
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
    response->message = "failed to open device";
    return;
  }

  active_device_id_ = device_id;
  active_device_name_ = device_name.empty() ? device_id : device_name;

  startCaptureThread();
  response->success = true;
  response->message = "switched";
}
}  // namespace fa_capture

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<fa_capture::FaCaptureNode>();
    rclcpp::spin(node);
  } catch (const std::exception &e) {
    RCLCPP_FATAL(rclcpp::get_logger("fa_capture_node"), "Exception: %s", e.what());
    return EXIT_FAILURE;
  }
  rclcpp::shutdown();
  return EXIT_SUCCESS;
}

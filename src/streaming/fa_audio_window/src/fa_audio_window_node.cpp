#include "fa_audio_window/fa_audio_window_node.hpp"

#include <algorithm>
#include <builtin_interfaces/msg/time.hpp>
#include <chrono>
#include <cinttypes>
#include <cctype>
#include <cstddef>
#include <functional>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

#include "fa_audio_window/time_range_parser.hpp"
#include "fa_audio_window/wav_writer.hpp"

namespace fa_audio_window
{

namespace
{
constexpr const char * kEncodingPcm16Le = "PCM16LE";
constexpr const char * kInterleavedLayout = "interleaved";
constexpr uint64_t kDefaultRetentionSeconds = 1800u;
constexpr uint64_t kNsecPerSecond = 1000000000u;

rclcpp::Parameter getDeclaredParameter(const rclcpp::Node & node, const std::string & name)
{
  rclcpp::Parameter parameter;
  if (!node.get_parameter(name, parameter)) {
    throw std::runtime_error(name + " is required");
  }
  return parameter;
}

int readIntParameter(const rclcpp::Node & node, const std::string & name)
{
  const rclcpp::Parameter parameter = getDeclaredParameter(node, name);
  if (parameter.get_type() != rclcpp::ParameterType::PARAMETER_INTEGER) {
    throw std::runtime_error(name + " must be an integer parameter");
  }
  const int64_t value = parameter.as_int();
  if (value < static_cast<int64_t>(std::numeric_limits<int>::min()) ||
      value > static_cast<int64_t>(std::numeric_limits<int>::max()))
  {
    throw std::runtime_error(name + " must fit in a 32-bit signed integer");
  }
  return static_cast<int>(value);
}

uint32_t readPositiveUint32Parameter(const rclcpp::Node & node, const std::string & name)
{
  const int value = readIntParameter(node, name);
  if (value <= 0) {
    throw std::runtime_error(name + " must be > 0");
  }
  return static_cast<uint32_t>(value);
}

std::set<std::string> readStringSetParameter(const rclcpp::Node & node, const std::string & name)
{
  const rclcpp::Parameter parameter = getDeclaredParameter(node, name);
  if (parameter.get_type() != rclcpp::ParameterType::PARAMETER_STRING_ARRAY) {
    throw std::runtime_error(name + " must be a string array parameter");
  }
  const std::vector<std::string> values = parameter.as_string_array();
  return {values.begin(), values.end()};
}

int64_t stampToUnixNs(const builtin_interfaces::msg::Time & stamp)
{
  if (stamp.sec < 0) {
    throw std::runtime_error("AudioFrame header.stamp.sec must be >= 0");
  }
  if (stamp.nanosec >= kNsecPerSecond) {
    throw std::runtime_error("AudioFrame header.stamp.nanosec must be < 1e9");
  }
  return static_cast<int64_t>(stamp.sec) * static_cast<int64_t>(kNsecPerSecond) +
    static_cast<int64_t>(stamp.nanosec);
}

uint64_t checkedRetentionNs(const int retention_seconds)
{
  if (retention_seconds <= 0) {
    throw std::runtime_error("window.retention_seconds must be > 0");
  }
  return static_cast<uint64_t>(retention_seconds) * kNsecPerSecond;
}

std::string sanitizeForFile(const std::string & value)
{
  std::string result;
  result.reserve(value.size());
  for (const char c : value) {
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-') {
      result.push_back(c);
    } else {
      result.push_back('_');
    }
  }
  return result.empty() ? "audio" : result;
}

bool hasNonWhitespace(const std::string & value)
{
  return std::any_of(value.begin(), value.end(), [](const char c) {
    return std::isspace(static_cast<unsigned char>(c)) == 0;
  });
}
}  // namespace

FaAudioWindowNode::FaAudioWindowNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("fa_audio_window", options)
{
  RCLCPP_INFO(this->get_logger(), "Starting FA Audio Window node");
  loadParameters();
  buffer_ = std::make_unique<AudioWindowBuffer>(config_.retention_ns, config_.expected_format);
  setupInterfaces();
}

void FaAudioWindowNode::loadParameters()
{
  this->declare_parameter<std::string>("input_topic", "fa_audio_window/input");
  this->declare_parameter<std::string>("service_name", "export_audio_window");
  this->declare_parameter<std::string>("archive_service_name", "archive_audio_window");
  this->declare_parameter<std::string>("input.source_id", "mic");
  this->declare_parameter<std::string>("input.stream_id", "audio/mic");
  this->declare_parameter<std::string>("expected.encoding", kEncodingPcm16Le);
  this->declare_parameter<int>("expected.sample_rate", 16000);
  this->declare_parameter<int>("expected.channels", 1);
  this->declare_parameter<int>("expected.bit_depth", 16);
  this->declare_parameter<std::string>("expected.layout", kInterleavedLayout);
  this->declare_parameter<int>("window.retention_seconds", static_cast<int>(kDefaultRetentionSeconds));
  this->declare_parameter<std::string>("audio.default_scope", "mic");
  this->declare_parameter<std::vector<std::string>>(
    "audio.supported_scopes",
    std::vector<std::string>{"mic"});
  this->declare_parameter<std::string>("export.output_directory", "/tmp/fa_audio_window");
  this->declare_parameter<std::string>("export.codec", "pcm_s16le");
  this->declare_parameter<std::string>("export.container", "wav");
  this->declare_parameter<std::string>("export.payload_format", "audio/wav");
  this->declare_parameter<std::string>("window.id", "fa_audio_window");
  this->declare_parameter<int>("window.epoch", 1);
  this->declare_parameter<int>("qos.depth", 10);
  this->declare_parameter<bool>("qos.reliable", false);

  config_.input_topic = this->get_parameter("input_topic").as_string();
  config_.service_name = this->get_parameter("service_name").as_string();
  config_.archive_service_name = this->get_parameter("archive_service_name").as_string();
  config_.source_id = this->get_parameter("input.source_id").as_string();
  config_.stream_id = this->get_parameter("input.stream_id").as_string();
  config_.expected_format.encoding = this->get_parameter("expected.encoding").as_string();
  config_.expected_format.sample_rate = readPositiveUint32Parameter(*this, "expected.sample_rate");
  config_.expected_format.channels = readPositiveUint32Parameter(*this, "expected.channels");
  config_.expected_format.bit_depth = readPositiveUint32Parameter(*this, "expected.bit_depth");
  config_.expected_format.layout = this->get_parameter("expected.layout").as_string();
  config_.retention_ns = checkedRetentionNs(readIntParameter(*this, "window.retention_seconds"));
  config_.default_audio_scope = this->get_parameter("audio.default_scope").as_string();
  config_.supported_audio_scopes = readStringSetParameter(*this, "audio.supported_scopes");
  config_.output_directory = this->get_parameter("export.output_directory").as_string();
  config_.supported_codec = this->get_parameter("export.codec").as_string();
  config_.supported_container = this->get_parameter("export.container").as_string();
  config_.supported_payload_format = this->get_parameter("export.payload_format").as_string();
  config_.window_id = this->get_parameter("window.id").as_string();
  config_.window_epoch = readPositiveUint32Parameter(*this, "window.epoch");
  config_.qos_depth = readIntParameter(*this, "qos.depth");
  config_.qos_reliable = this->get_parameter("qos.reliable").as_bool();

  if (config_.input_topic.empty() || config_.service_name.empty() || config_.archive_service_name.empty()) {
    throw std::runtime_error("input_topic, service_name, and archive_service_name are required");
  }
  if (config_.service_name == config_.archive_service_name) {
    throw std::runtime_error("service_name and archive_service_name must be different service names");
  }
  if (config_.source_id.empty() || config_.stream_id.empty()) {
    throw std::runtime_error("input.source_id and input.stream_id are required");
  }
  if (config_.expected_format.encoding != kEncodingPcm16Le ||
      config_.expected_format.bit_depth != 16u ||
      config_.expected_format.layout != kInterleavedLayout)
  {
    throw std::runtime_error("fa_audio_window initial implementation requires PCM16LE/16-bit/interleaved input");
  }
  if (config_.expected_format.sample_rate == 0u || config_.expected_format.channels == 0u) {
    throw std::runtime_error("expected.sample_rate and expected.channels must be > 0");
  }
  if (config_.default_audio_scope.empty()) {
    throw std::runtime_error("audio.default_scope is required");
  }
  if (config_.supported_audio_scopes.count(config_.default_audio_scope) == 0u) {
    throw std::runtime_error("audio.default_scope must be included in audio.supported_scopes");
  }
  if (config_.supported_codec.empty() || config_.supported_container.empty() ||
      config_.supported_payload_format.empty())
  {
    throw std::runtime_error("export codec/container/payload_format are required");
  }
  if (config_.qos_depth <= 0) {
    throw std::runtime_error("qos.depth must be > 0");
  }

  RCLCPP_INFO(
    this->get_logger(),
    "Audio window config: topic=%s export_service=%s archive_service=%s source=%s stream=%s "
    "format=%s/%uHz/%uch/%ubit/%s "
    "retention_ns=%" PRIu64 " scope=%s output=%s export=%s/%s/%s",
    config_.input_topic.c_str(),
    config_.service_name.c_str(),
    config_.archive_service_name.c_str(),
    config_.source_id.c_str(),
    config_.stream_id.c_str(),
    config_.expected_format.encoding.c_str(),
    config_.expected_format.sample_rate,
    config_.expected_format.channels,
    config_.expected_format.bit_depth,
    config_.expected_format.layout.c_str(),
    config_.retention_ns,
    config_.default_audio_scope.c_str(),
    config_.output_directory.string().c_str(),
    config_.supported_codec.c_str(),
    config_.supported_container.c_str(),
    config_.supported_payload_format.c_str());
}

void FaAudioWindowNode::setupInterfaces()
{
  rclcpp::QoS qos(static_cast<size_t>(config_.qos_depth));
  if (config_.qos_reliable) {
    qos.reliable();
  } else {
    qos.best_effort();
  }

  audio_sub_ = this->create_subscription<fa_interfaces::msg::AudioFrame>(
    config_.input_topic,
    qos,
    std::bind(&FaAudioWindowNode::handleFrame, this, std::placeholders::_1));
  export_service_ = this->create_service<ExportAudioWindow>(
    config_.service_name,
    std::bind(
      &FaAudioWindowNode::handleExportRequest,
      this,
      std::placeholders::_1,
      std::placeholders::_2));
  archive_service_ = this->create_service<ArchiveAudioWindow>(
    config_.archive_service_name,
    std::bind(
      &FaAudioWindowNode::handleArchiveRequest,
      this,
      std::placeholders::_1,
      std::placeholders::_2));
}

void FaAudioWindowNode::handleFrame(const fa_interfaces::msg::AudioFrame::SharedPtr msg)
{
  frames_in_.fetch_add(1);
  if (!msg) {
    frames_dropped_.fetch_add(1);
    return;
  }
  if (!validateFrame(*msg)) {
    frames_dropped_.fetch_add(1);
    return;
  }

  try {
    const TimedAudioFrame timed_frame = toTimedAudioFrame(*msg);
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    buffer_->addFrame(timed_frame);
    frames_buffered_.fetch_add(1);
  } catch (const std::exception & e) {
    frames_dropped_.fetch_add(1);
    RCLCPP_WARN(this->get_logger(), "Dropping AudioFrame: %s", e.what());
  }
}

void FaAudioWindowNode::handleExportRequest(
  const std::shared_ptr<ExportAudioWindow::Request> request,
  std::shared_ptr<ExportAudioWindow::Response> response)
{
  response->success = false;
  response->error_code = ExportAudioWindow::Response::ERROR_NONE;

  const ClipOperationResult result = writeWindowClip(
    ClipOperationRequest{
      request->time_range_spec,
      request->audio_scope,
      request->codec,
      request->container,
      request->payload_format,
      "export",
    });
  response->time_range = result.time_range;
  response->audio_clip_ref = result.audio_clip_ref;
  if (!result.success) {
    setError(*response, exportErrorCode(result.error), result.message);
    return;
  }

  response->success = true;
  response->error_code = ExportAudioWindow::Response::ERROR_NONE;
  response->message.clear();
}

void FaAudioWindowNode::handleArchiveRequest(
  const std::shared_ptr<ArchiveAudioWindow::Request> request,
  std::shared_ptr<ArchiveAudioWindow::Response> response)
{
  response->success = false;
  response->error_code = ArchiveAudioWindow::Response::ERROR_NONE;

  if (!hasNonWhitespace(request->reason)) {
    setError(
      *response,
      ArchiveAudioWindow::Response::ERROR_INVALID_ARCHIVE_REQUEST,
      "archive reason is required");
    return;
  }

  const ClipOperationResult result = writeWindowClip(
    ClipOperationRequest{
      request->time_range_spec,
      request->audio_scope,
      request->codec,
      request->container,
      request->payload_format,
      "archive",
    });
  response->time_range = result.time_range;
  response->audio_clip_ref = result.audio_clip_ref;
  if (!result.success) {
    setError(*response, archiveErrorCode(result.error), result.message);
    return;
  }

  response->success = true;
  response->error_code = ArchiveAudioWindow::Response::ERROR_NONE;
  response->message.clear();
}

FaAudioWindowNode::ClipOperationResult FaAudioWindowNode::writeWindowClip(
  const ClipOperationRequest & request)
{
  ClipOperationResult result;

  const TimeRangeParseResult parsed = parseNumericUnixNsRange(request.time_range_spec);
  if (!parsed.success) {
    result.error = ClipOperationError::kTimeRangeUnresolved;
    result.message = parsed.message;
    return result;
  }
  fillResolvedRange(result.time_range, parsed.range);

  const std::string resolved_scope = resolveAudioScope(request.audio_scope);
  if (config_.supported_audio_scopes.count(resolved_scope) == 0u) {
    result.error = ClipOperationError::kUnsupportedAudioScope;
    result.message = "audio_scope is not in configured supported set";
    return result;
  }

  if (!isSupportedFormat(request.codec, request.container, request.payload_format)) {
    result.error = ClipOperationError::kUnsupportedFormat;
    result.message = "only configured PCM16LE WAV audio clip format is supported";
    return result;
  }

  WindowQueryResult query;
  {
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    query = buffer_->query(parsed.range);
  }

  if (query.status == WindowQueryStatus::kWindowNotFound) {
    result.error = ClipOperationError::kWindowNotFound;
    result.message = "audio window is empty";
    return result;
  }
  if (query.status == WindowQueryStatus::kRangeOutsideWindow) {
    result.error = ClipOperationError::kRangeOutsideWindow;
    result.message = "requested time range is outside retained audio window";
    return result;
  }
  if (query.status == WindowQueryStatus::kRangeNotContinuous) {
    result.error = ClipOperationError::kRangeNotContinuous;
    result.message = "requested time range is not continuously covered by retained audio";
    return result;
  }
  if (query.status == WindowQueryStatus::kNoSamplesSelected) {
    result.error = ClipOperationError::kNoSamplesSelected;
    result.message = "no PCM samples selected";
    return result;
  }

  const uint64_t sequence = clip_sequence_.fetch_add(1) + 1u;
  const std::filesystem::path output_path =
    clipPathFor(request.operation_name, query.exported_range, sequence);
  try {
    WavWriter::writePcm16Le(output_path, config_.expected_format, query.pcm_data);
  } catch (const std::exception & e) {
    result.error = ClipOperationError::kWriteFailed;
    result.message = e.what();
    return result;
  }

  result.success = true;
  result.error = ClipOperationError::kNone;
  result.message.clear();
  fillResolvedRange(result.time_range, query.exported_range);
  fillAudioClipRef(
    result.audio_clip_ref,
    output_path.stem().string(),
    output_path,
    query);
  return result;
}

bool FaAudioWindowNode::validateFrame(const fa_interfaces::msg::AudioFrame & msg)
{
  if (msg.source_id != config_.source_id || msg.stream_id != config_.stream_id) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(),
      *this->get_clock(),
      3000,
      "AudioFrame identity mismatch: source=%s stream=%s expected=%s/%s",
      msg.source_id.c_str(),
      msg.stream_id.c_str(),
      config_.source_id.c_str(),
      config_.stream_id.c_str());
    return false;
  }

  const AudioFormat frame_format{
    msg.encoding,
    msg.sample_rate,
    msg.channels,
    msg.bit_depth,
    msg.layout,
  };
  if (!sameAudioFormat(frame_format, config_.expected_format)) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(),
      *this->get_clock(),
      3000,
      "AudioFrame format mismatch: frame=%s/%uHz/%uch/%ubit/%s expected=%s/%uHz/%uch/%ubit/%s",
      msg.encoding.c_str(),
      msg.sample_rate,
      msg.channels,
      msg.bit_depth,
      msg.layout.c_str(),
      config_.expected_format.encoding.c_str(),
      config_.expected_format.sample_rate,
      config_.expected_format.channels,
      config_.expected_format.bit_depth,
      config_.expected_format.layout.c_str());
    return false;
  }

  const size_t sample_frame_bytes = bytesPerSampleFrame(config_.expected_format);
  if (msg.data.empty() || msg.data.size() % sample_frame_bytes != 0u) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(),
      *this->get_clock(),
      3000,
      "AudioFrame data must be non-empty and sample-frame aligned");
    return false;
  }
  return true;
}

TimedAudioFrame FaAudioWindowNode::toTimedAudioFrame(const fa_interfaces::msg::AudioFrame & msg) const
{
  const int64_t start_unix_ns = stampToUnixNs(msg.header.stamp);
  const size_t sample_count = msg.data.size() / bytesPerSampleFrame(config_.expected_format);
  const size_t whole_seconds = sample_count / config_.expected_format.sample_rate;
  const size_t remaining_samples = sample_count % config_.expected_format.sample_rate;
  if (whole_seconds > static_cast<size_t>(std::numeric_limits<int64_t>::max() / kNsecPerSecond)) {
    throw std::runtime_error("AudioFrame timestamp overflows int64 nanosecond range");
  }
  const int64_t duration_ns =
    static_cast<int64_t>(whole_seconds) * static_cast<int64_t>(kNsecPerSecond) +
    static_cast<int64_t>(
      remaining_samples * static_cast<size_t>(kNsecPerSecond) / config_.expected_format.sample_rate);
  if (duration_ns <= 0) {
    throw std::runtime_error("AudioFrame duration must be > 0");
  }
  if (start_unix_ns > std::numeric_limits<int64_t>::max() - duration_ns) {
    throw std::runtime_error("AudioFrame timestamp overflows int64 nanosecond range");
  }

  return TimedAudioFrame{
    {start_unix_ns, start_unix_ns + duration_ns},
    msg.epoch,
    msg.data,
  };
}

std::string FaAudioWindowNode::resolveAudioScope(const std::string & requested_scope) const
{
  if (requested_scope.empty()) {
    return config_.default_audio_scope;
  }
  return requested_scope;
}

bool FaAudioWindowNode::isSupportedFormat(
  const std::string & codec,
  const std::string & container,
  const std::string & payload_format) const
{
  return codec == config_.supported_codec &&
    container == config_.supported_container &&
    payload_format == config_.supported_payload_format;
}

void FaAudioWindowNode::setError(
  ExportAudioWindow::Response & response,
  const std::string & error_code,
  const std::string & message) const
{
  response.success = false;
  response.error_code = error_code;
  response.message = message;
}

void FaAudioWindowNode::setError(
  ArchiveAudioWindow::Response & response,
  const std::string & error_code,
  const std::string & message) const
{
  response.success = false;
  response.error_code = error_code;
  response.message = message;
}

std::string FaAudioWindowNode::exportErrorCode(const ClipOperationError error) const
{
  switch (error) {
    case ClipOperationError::kNone:
      return ExportAudioWindow::Response::ERROR_NONE;
    case ClipOperationError::kTimeRangeUnresolved:
      return ExportAudioWindow::Response::ERROR_TIME_RANGE_UNRESOLVED;
    case ClipOperationError::kWindowNotFound:
      return ExportAudioWindow::Response::ERROR_WINDOW_NOT_FOUND;
    case ClipOperationError::kRangeOutsideWindow:
    case ClipOperationError::kRangeNotContinuous:
      return ExportAudioWindow::Response::ERROR_RANGE_OUTSIDE_WINDOW;
    case ClipOperationError::kUnsupportedAudioScope:
      return ExportAudioWindow::Response::ERROR_UNSUPPORTED_AUDIO_SCOPE;
    case ClipOperationError::kUnsupportedFormat:
      return ExportAudioWindow::Response::ERROR_UNSUPPORTED_EXPORT_FORMAT;
    case ClipOperationError::kNoSamplesSelected:
    case ClipOperationError::kWriteFailed:
      return ExportAudioWindow::Response::ERROR_EXPORT_FAILED;
  }
  return ExportAudioWindow::Response::ERROR_EXPORT_FAILED;
}

std::string FaAudioWindowNode::archiveErrorCode(const ClipOperationError error) const
{
  switch (error) {
    case ClipOperationError::kNone:
      return ArchiveAudioWindow::Response::ERROR_NONE;
    case ClipOperationError::kTimeRangeUnresolved:
      return ArchiveAudioWindow::Response::ERROR_TIME_RANGE_UNRESOLVED;
    case ClipOperationError::kWindowNotFound:
      return ArchiveAudioWindow::Response::ERROR_WINDOW_NOT_FOUND;
    case ClipOperationError::kRangeOutsideWindow:
      return ArchiveAudioWindow::Response::ERROR_RANGE_OUTSIDE_WINDOW;
    case ClipOperationError::kRangeNotContinuous:
      return ArchiveAudioWindow::Response::ERROR_RANGE_NOT_CONTINUOUS;
    case ClipOperationError::kUnsupportedAudioScope:
      return ArchiveAudioWindow::Response::ERROR_UNSUPPORTED_AUDIO_SCOPE;
    case ClipOperationError::kUnsupportedFormat:
      return ArchiveAudioWindow::Response::ERROR_UNSUPPORTED_ARCHIVE_FORMAT;
    case ClipOperationError::kNoSamplesSelected:
    case ClipOperationError::kWriteFailed:
      return ArchiveAudioWindow::Response::ERROR_ARCHIVE_FAILED;
  }
  return ArchiveAudioWindow::Response::ERROR_ARCHIVE_FAILED;
}

void FaAudioWindowNode::fillResolvedRange(
  fa_interfaces::msg::ResolvedTimeRange & msg,
  const TimeRange & range) const
{
  msg.start_unix_ns = range.start_unix_ns;
  msg.end_unix_ns = range.end_unix_ns;
  msg.clock = fa_interfaces::msg::ResolvedTimeRange::CLOCK_MEDIA;
  msg.uncertainty_ns = 0u;
  msg.uncertainty_reason.clear();
}

void FaAudioWindowNode::fillAudioClipRef(
  fa_interfaces::msg::AudioClipRef & ref,
  const std::string & clip_id,
  const std::filesystem::path & path,
  const WindowQueryResult & query) const
{
  ref.clip_id = clip_id;
  ref.uri = "file://" + path.string();
  ref.codec = config_.supported_codec;
  ref.container = config_.supported_container;
  ref.payload_format = config_.supported_payload_format;
  ref.sample_rate = config_.expected_format.sample_rate;
  ref.channels = config_.expected_format.channels;
  ref.duration_ns =
    static_cast<uint64_t>(query.exported_range.end_unix_ns - query.exported_range.start_unix_ns);
  fillResolvedRange(ref.time_range, query.exported_range);
}

std::filesystem::path FaAudioWindowNode::clipPathFor(
  const std::string & operation_name,
  const TimeRange & range,
  const uint64_t sequence) const
{
  std::ostringstream filename;
  filename << sanitizeForFile(operation_name) << "_" << sanitizeForFile(config_.source_id) << "_"
           << sanitizeForFile(config_.stream_id) << "_"
           << range.start_unix_ns << "_" << range.end_unix_ns << "_" << sequence << ".wav";
  return config_.output_directory / filename.str();
}

}  // namespace fa_audio_window

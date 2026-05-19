#include "fa_audio_window/fa_audio_window_node.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <builtin_interfaces/msg/time.hpp>
#include <cerrno>
#include <chrono>
#include <cinttypes>
#include <cctype>
#include <cstddef>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <openssl/evp.h>
#include <unistd.h>

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

enum class PublishStatus
{
  kPublished,
  kExistingMatched,
};

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

std::string trimWhitespace(const std::string & value)
{
  auto first = value.begin();
  while (first != value.end() && std::isspace(static_cast<unsigned char>(*first)) != 0) {
    ++first;
  }
  auto last = value.end();
  while (last != first && std::isspace(static_cast<unsigned char>(*(last - 1))) != 0) {
    --last;
  }
  return {first, last};
}

std::string jsonEscape(const std::string & value)
{
  std::ostringstream escaped;
  for (const char c : value) {
    switch (c) {
      case '"':
        escaped << "\\\"";
        break;
      case '\\':
        escaped << "\\\\";
        break;
      case '\b':
        escaped << "\\b";
        break;
      case '\f':
        escaped << "\\f";
        break;
      case '\n':
        escaped << "\\n";
        break;
      case '\r':
        escaped << "\\r";
        break;
      case '\t':
        escaped << "\\t";
        break;
      default:
        if (static_cast<unsigned char>(c) < 0x20u) {
          escaped << "\\u";
          escaped << "00";
          const char * digits = "0123456789abcdef";
          escaped << digits[(static_cast<unsigned char>(c) >> 4u) & 0x0fu];
          escaped << digits[static_cast<unsigned char>(c) & 0x0fu];
        } else {
          escaped << c;
        }
        break;
    }
  }
  return escaped.str();
}

void writeJsonStringArray(std::ostream & out, const std::vector<std::string> & values)
{
  out << "[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i > 0u) {
      out << ",";
    }
    out << "\"" << jsonEscape(values[i]) << "\"";
  }
  out << "]";
}

void appendIdentityField(std::ostream & out, const std::string & name, const std::string & value)
{
  out << name << ":" << value.size() << ":" << value << "\n";
}

void appendIdentityField(std::ostream & out, const std::string & name, const uint64_t value)
{
  appendIdentityField(out, name, std::to_string(value));
}

void appendIdentityField(std::ostream & out, const std::string & name, const int64_t value)
{
  appendIdentityField(out, name, std::to_string(value));
}

uint64_t stableFnv1a64(const std::string & value)
{
  uint64_t hash = 14695981039346656037ull;
  for (const unsigned char c : value) {
    hash ^= static_cast<uint64_t>(c);
    hash *= 1099511628211ull;
  }
  return hash;
}

std::string hex64(const uint64_t value)
{
  std::ostringstream out;
  out << std::hex << std::nouppercase << std::setw(16) << std::setfill('0') << value;
  return out.str();
}

std::vector<uint8_t> readBinaryFile(const std::filesystem::path & path);

std::string hexBytes(const unsigned char * data, const unsigned int size)
{
  std::ostringstream out;
  out << std::hex << std::nouppercase << std::setfill('0');
  for (unsigned int i = 0u; i < size; ++i) {
    out << std::setw(2) << static_cast<unsigned int>(data[i]);
  }
  return out.str();
}

std::string sha256Hex(const std::vector<uint8_t> & bytes)
{
  EVP_MD_CTX * raw_context = EVP_MD_CTX_new();
  if (raw_context == nullptr) {
    throw std::runtime_error("failed to allocate SHA-256 context");
  }
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context(raw_context, EVP_MD_CTX_free);

  if (EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1) {
    throw std::runtime_error("failed to initialize SHA-256 digest");
  }
  if (!bytes.empty() && EVP_DigestUpdate(context.get(), bytes.data(), bytes.size()) != 1) {
    throw std::runtime_error("failed to update SHA-256 digest");
  }

  std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
  unsigned int digest_size = 0u;
  if (EVP_DigestFinal_ex(context.get(), digest.data(), &digest_size) != 1) {
    throw std::runtime_error("failed to finalize SHA-256 digest");
  }
  if (digest_size != 32u) {
    throw std::runtime_error("unexpected SHA-256 digest size");
  }
  return hexBytes(digest.data(), digest_size);
}

std::string sha256HexForFile(const std::filesystem::path & path)
{
  return sha256Hex(readBinaryFile(path));
}

bool startsWith(const std::string & value, const std::string & prefix)
{
  return value.rfind(prefix, 0u) == 0u;
}

ArchiveStoreBackend archiveStoreBackendFromName(const std::string & backend_name)
{
  if (backend_name == "local_file") {
    return ArchiveStoreBackend::kLocalFile;
  }
  if (backend_name == "filesystem") {
    return ArchiveStoreBackend::kFilesystem;
  }
  throw std::runtime_error("archive.store.backend is unsupported: " + backend_name);
}

std::string buildPrefixedUri(const std::string & prefix, const std::string & filename)
{
  return prefix + filename;
}

std::string operationIdentity(const std::string & operation_name)
{
  if (operation_name == "export") {
    return "export_audio_window";
  }
  if (operation_name == "archive") {
    return "archive_audio_window";
  }
  throw std::runtime_error("unsupported clip operation");
}

std::filesystem::path metadataPathFor(const std::filesystem::path & clip_path)
{
  return clip_path.string() + ".metadata.json";
}

std::string errnoMessage(const int error_number)
{
  return std::error_code(error_number, std::generic_category()).message();
}

std::filesystem::path reserveTemporaryPublishPath(const std::filesystem::path & target_path)
{
  const std::filesystem::path parent = target_path.parent_path();
  if (!parent.empty()) {
    std::filesystem::create_directories(parent);
  }

  static std::atomic<uint64_t> temp_sequence{0};
  for (uint32_t attempt = 0u; attempt < 64u; ++attempt) {
    const uint64_t sequence = temp_sequence.fetch_add(1u) + 1u;
    const std::filesystem::path temp_path =
      target_path.string() + ".tmp." + std::to_string(static_cast<long long>(::getpid())) + "." +
      std::to_string(sequence);
    const int fd = ::open(temp_path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0666);
    if (fd >= 0) {
      if (::close(fd) != 0) {
        const int close_errno = errno;
        std::error_code remove_error;
        std::filesystem::remove(temp_path, remove_error);
        throw std::runtime_error(
          "failed to close temporary publish file: " + errnoMessage(close_errno));
      }
      return temp_path;
    }

    const int open_errno = errno;
    if (open_errno == EEXIST) {
      continue;
    }
    throw std::runtime_error(
      "failed to reserve temporary publish file: " + errnoMessage(open_errno));
  }

  throw std::runtime_error("failed to reserve temporary publish file after repeated collisions");
}

std::vector<uint8_t> readBinaryFile(const std::filesystem::path & path)
{
  std::ifstream in(path, std::ios::binary);
  if (!in.is_open()) {
    throw std::runtime_error("failed to open file for comparison: " + path.string());
  }
  return {
    std::istreambuf_iterator<char>(in),
    std::istreambuf_iterator<char>(),
  };
}

bool fileBytesEqual(const std::filesystem::path & left, const std::filesystem::path & right)
{
  return readBinaryFile(left) == readBinaryFile(right);
}

PublishStatus publishTempFileNoClobber(
  const std::filesystem::path & temp_path,
  const std::filesystem::path & target_path,
  const std::string & inspect_error_message,
  const std::string & conflict_error_message,
  const std::string & publish_error_message)
{
  if (::link(temp_path.c_str(), target_path.c_str()) == 0) {
    std::error_code remove_error;
    std::filesystem::remove(temp_path, remove_error);
    return PublishStatus::kPublished;
  }

  const int link_errno = errno;
  std::error_code exists_error;
  const bool target_exists = std::filesystem::exists(target_path, exists_error);
  if (exists_error) {
    std::error_code remove_error;
    std::filesystem::remove(temp_path, remove_error);
    throw std::runtime_error(inspect_error_message + ": " + exists_error.message());
  }

  if (target_exists) {
    if (fileBytesEqual(target_path, temp_path)) {
      std::error_code remove_error;
      std::filesystem::remove(temp_path, remove_error);
      return PublishStatus::kExistingMatched;
    }
    std::error_code remove_error;
    std::filesystem::remove(temp_path, remove_error);
    throw std::runtime_error(conflict_error_message);
  }

  std::error_code remove_error;
  std::filesystem::remove(temp_path, remove_error);
  throw std::runtime_error(publish_error_message + ": " + errnoMessage(link_errno));
}

PublishStatus publishExistingFileNoClobber(
  const std::filesystem::path & source_path,
  const std::filesystem::path & target_path,
  const std::string & inspect_error_message,
  const std::string & conflict_error_message,
  const std::string & publish_error_message)
{
  std::filesystem::path temp_path;
  try {
    temp_path = reserveTemporaryPublishPath(target_path);
    std::filesystem::copy_file(
      source_path,
      temp_path,
      std::filesystem::copy_options::overwrite_existing);
    return publishTempFileNoClobber(
      temp_path,
      target_path,
      inspect_error_message,
      conflict_error_message,
      publish_error_message);
  } catch (...) {
    std::error_code remove_error;
    if (!temp_path.empty()) {
      std::filesystem::remove(temp_path, remove_error);
    }
    throw;
  }
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
  this->declare_parameter<std::string>("archive.store.backend", "local_file");
  this->declare_parameter<std::string>("archive.store.directory", "");
  this->declare_parameter<std::string>("archive.store.uri_prefix", "");
  this->declare_parameter<std::string>("archive.store.metadata_uri_prefix", "");
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
  config_.archive_store.backend_name = this->get_parameter("archive.store.backend").as_string();
  config_.archive_store.backend = archiveStoreBackendFromName(config_.archive_store.backend_name);
  config_.archive_store.directory = this->get_parameter("archive.store.directory").as_string();
  config_.archive_store.uri_prefix = this->get_parameter("archive.store.uri_prefix").as_string();
  config_.archive_store.metadata_uri_prefix =
    this->get_parameter("archive.store.metadata_uri_prefix").as_string();
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
  if (config_.archive_store.backend == ArchiveStoreBackend::kFilesystem) {
    if (config_.archive_store.directory.empty()) {
      throw std::runtime_error("archive.store.directory is required for filesystem archive store");
    }
    if (config_.archive_store.uri_prefix.empty() || config_.archive_store.metadata_uri_prefix.empty()) {
      throw std::runtime_error(
        "archive.store.uri_prefix and archive.store.metadata_uri_prefix are required for filesystem archive store");
    }
    if (startsWith(config_.archive_store.uri_prefix, "file://") ||
        startsWith(config_.archive_store.metadata_uri_prefix, "file://"))
    {
      throw std::runtime_error("filesystem archive store URI prefixes must not use file://");
    }
  }
  if (config_.qos_depth <= 0) {
    throw std::runtime_error("qos.depth must be > 0");
  }

  RCLCPP_INFO(
    this->get_logger(),
    "Audio window config: topic=%s export_service=%s archive_service=%s source=%s stream=%s "
    "format=%s/%uHz/%uch/%ubit/%s "
    "retention_ns=%" PRIu64 " scope=%s output=%s export=%s/%s/%s archive_store=%s",
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
    config_.supported_payload_format.c_str(),
    config_.archive_store.backend_name.c_str());
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
      "",
      {},
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
      trimWhitespace(request->reason),
      request->related_artifact_ids,
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

  const std::string clip_id = clipIdFor(request, resolved_scope, query.exported_range);
  const std::filesystem::path output_path = clipPathFor(clip_id);
  const bool is_archive = request.operation_name == "archive";
  const std::filesystem::path metadata_path = metadataPathFor(output_path);

  if (is_archive) {
    std::error_code clip_exists_error;
    std::error_code metadata_exists_error;
    const bool clip_exists = std::filesystem::exists(output_path, clip_exists_error);
    const bool metadata_exists = std::filesystem::exists(metadata_path, metadata_exists_error);
    if (clip_exists_error || metadata_exists_error) {
      result.error = ClipOperationError::kWriteFailed;
      result.message = "failed to inspect deterministic archive target";
      return result;
    }
    if (!clip_exists && metadata_exists) {
      result.error = ClipOperationError::kWriteFailed;
      result.message = "deterministic archive metadata exists without matching audio clip";
      return result;
    }
  }

  PublishStatus publish_status = PublishStatus::kPublished;
  std::filesystem::path temp_output_path;
  try {
    temp_output_path = reserveTemporaryPublishPath(output_path);
    WavWriter::writePcm16Le(temp_output_path, config_.expected_format, query.pcm_data);
    publish_status = publishTempFileNoClobber(
      temp_output_path,
      output_path,
      "failed to inspect deterministic audio clip path",
      "deterministic audio clip path exists with different bytes",
      "failed to publish audio clip file");
  } catch (const std::exception & e) {
    std::error_code remove_error;
    if (!temp_output_path.empty()) {
      std::filesystem::remove(temp_output_path, remove_error);
    }
    result.error = ClipOperationError::kWriteFailed;
    result.message = e.what();
    return result;
  }

  fillResolvedRange(result.time_range, query.exported_range);
  fillAudioClipRef(
    result.audio_clip_ref,
    clip_id,
    clipUriFor(clip_id, output_path, is_archive),
    is_archive ? metadataUriFor(clip_id, metadata_path) : std::string{},
    query);

  bool local_metadata_published = false;
  try {
    if (is_archive) {
      const std::string expected_metadata =
        archiveMetadataJson(request, result.audio_clip_ref, query.exported_range);
      local_metadata_published = writeTextAtomically(metadata_path, expected_metadata);

      if (config_.archive_store.backend == ArchiveStoreBackend::kFilesystem) {
        const std::filesystem::path store_content_path = archiveStoreContentPathFor(clip_id);
        const std::filesystem::path store_metadata_path = archiveStoreMetadataPathFor(clip_id);
        PublishStatus store_content_status = publishExistingFileNoClobber(
          output_path,
          store_content_path,
          "failed to inspect archive store audio clip path",
          "archive store audio clip path exists with different bytes",
          "failed to publish archive store audio clip file");
        try {
          (void)publishExistingFileNoClobber(
            metadata_path,
            store_metadata_path,
            "failed to inspect archive store metadata path",
            "archive store metadata path exists with different content",
            "failed to publish archive store metadata file");
        } catch (...) {
          if (store_content_status == PublishStatus::kPublished) {
            std::error_code remove_error;
            std::filesystem::remove(store_content_path, remove_error);
          }
          throw;
        }
        finalizeClipHashes(result.audio_clip_ref, store_content_path, store_metadata_path);
      } else {
        finalizeClipHashes(result.audio_clip_ref, output_path, metadata_path);
      }
    } else {
      finalizeClipHashes(result.audio_clip_ref, output_path, std::filesystem::path{});
    }
  } catch (const std::exception & e) {
      std::error_code remove_error;
      if (local_metadata_published) {
        std::filesystem::remove(metadata_path, remove_error);
      }
      if (publish_status == PublishStatus::kPublished) {
        std::filesystem::remove(output_path, remove_error);
      }
      result.success = false;
      result.error = ClipOperationError::kWriteFailed;
      result.message = e.what();
      result.audio_clip_ref = fa_interfaces::msg::AudioClipRef{};
      result.time_range = fa_interfaces::msg::ResolvedTimeRange{};
      return result;
  }

  result.success = true;
  result.error = ClipOperationError::kNone;
  result.message.clear();
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
  const std::string & uri,
  const std::string & metadata_uri,
  const WindowQueryResult & query) const
{
  ref.clip_id = clip_id;
  ref.uri = uri;
  ref.metadata_uri = metadata_uri;
  ref.content_sha256.clear();
  ref.metadata_sha256.clear();
  ref.codec = config_.supported_codec;
  ref.container = config_.supported_container;
  ref.payload_format = config_.supported_payload_format;
  ref.sample_rate = config_.expected_format.sample_rate;
  ref.channels = config_.expected_format.channels;
  ref.duration_ns =
    static_cast<uint64_t>(query.exported_range.end_unix_ns - query.exported_range.start_unix_ns);
  fillResolvedRange(ref.time_range, query.exported_range);
}

void FaAudioWindowNode::finalizeClipHashes(
  fa_interfaces::msg::AudioClipRef & ref,
  const std::filesystem::path & content_path,
  const std::filesystem::path & metadata_path) const
{
  ref.content_sha256 = sha256HexForFile(content_path);
  if (metadata_path.empty()) {
    ref.metadata_sha256.clear();
    return;
  }
  ref.metadata_sha256 = sha256HexForFile(metadata_path);
}

std::string FaAudioWindowNode::clipIdFor(
  const ClipOperationRequest & request,
  const std::string & resolved_scope,
  const TimeRange & exported_range) const
{
  std::ostringstream identity;
  appendIdentityField(identity, "operation", operationIdentity(request.operation_name));
  appendIdentityField(identity, "window_id", config_.window_id);
  appendIdentityField(identity, "window_epoch", config_.window_epoch);
  appendIdentityField(identity, "source_id", config_.source_id);
  appendIdentityField(identity, "stream_id", config_.stream_id);
  appendIdentityField(identity, "audio_scope", resolved_scope);
  appendIdentityField(identity, "start_unix_ns", exported_range.start_unix_ns);
  appendIdentityField(identity, "end_unix_ns", exported_range.end_unix_ns);
  appendIdentityField(identity, "codec", request.codec);
  appendIdentityField(identity, "container", request.container);
  appendIdentityField(identity, "payload_format", request.payload_format);
  if (request.operation_name == "archive") {
    appendIdentityField(identity, "reason", request.archive_reason);
    appendIdentityField(
      identity,
      "related_artifact_count",
      static_cast<uint64_t>(request.related_artifact_ids.size()));
    for (const std::string & artifact_id : request.related_artifact_ids) {
      appendIdentityField(identity, "related_artifact_id", artifact_id);
    }
  }

  const std::string hash = hex64(stableFnv1a64(identity.str()));
  std::ostringstream clip_id;
  clip_id << sanitizeForFile(operationIdentity(request.operation_name)) << "_"
          << sanitizeForFile(config_.window_id) << "_epoch" << config_.window_epoch << "_"
          << sanitizeForFile(config_.source_id) << "_" << sanitizeForFile(config_.stream_id) << "_"
          << exported_range.start_unix_ns << "_" << exported_range.end_unix_ns << "_"
          << sanitizeForFile(request.codec) << "_" << sanitizeForFile(request.container) << "_"
          << sanitizeForFile(request.payload_format) << "_" << hash;
  return clip_id.str();
}

std::filesystem::path FaAudioWindowNode::clipPathFor(const std::string & clip_id) const
{
  return config_.output_directory / (clip_id + ".wav");
}

std::filesystem::path FaAudioWindowNode::archiveStoreContentPathFor(const std::string & clip_id) const
{
  return config_.archive_store.directory / (clip_id + ".wav");
}

std::filesystem::path FaAudioWindowNode::archiveStoreMetadataPathFor(const std::string & clip_id) const
{
  return config_.archive_store.directory / (clip_id + ".metadata.json");
}

std::string FaAudioWindowNode::clipUriFor(
  const std::string & clip_id,
  const std::filesystem::path & local_clip_path,
  const bool is_archive) const
{
  if (is_archive && config_.archive_store.backend == ArchiveStoreBackend::kFilesystem) {
    return buildPrefixedUri(config_.archive_store.uri_prefix, clip_id + ".wav");
  }
  return "file://" + local_clip_path.string();
}

std::string FaAudioWindowNode::metadataUriFor(
  const std::string & clip_id,
  const std::filesystem::path & local_metadata_path) const
{
  if (config_.archive_store.backend == ArchiveStoreBackend::kFilesystem) {
    return buildPrefixedUri(config_.archive_store.metadata_uri_prefix, clip_id + ".metadata.json");
  }
  return "file://" + local_metadata_path.string();
}

std::string FaAudioWindowNode::archiveMetadataJson(
  const ClipOperationRequest & request,
  const fa_interfaces::msg::AudioClipRef & clip_ref,
  const TimeRange & exported_range) const
{
  std::ostringstream out;
  out << "{\n";
  out << "  \"schema\":\"fluent_audio.archive_metadata.v1\",\n";
  out << "  \"operation\":\"archive_audio_window\",\n";
  out << "  \"reason\":\"" << jsonEscape(request.archive_reason) << "\",\n";
  out << "  \"related_artifact_ids\":";
  writeJsonStringArray(out, request.related_artifact_ids);
  out << ",\n";
  out << "  \"source_id\":\"" << jsonEscape(config_.source_id) << "\",\n";
  out << "  \"stream_id\":\"" << jsonEscape(config_.stream_id) << "\",\n";
  out << "  \"window_id\":\"" << jsonEscape(config_.window_id) << "\",\n";
  out << "  \"window_epoch\":" << config_.window_epoch << ",\n";
  out << "  \"audio_scope\":\"" << jsonEscape(resolveAudioScope(request.audio_scope)) << "\",\n";
  out << "  \"time_range\":{";
  out << "\"start_unix_ns\":" << exported_range.start_unix_ns << ",";
  out << "\"end_unix_ns\":" << exported_range.end_unix_ns << ",";
  out << "\"clock\":\"" << jsonEscape(fa_interfaces::msg::ResolvedTimeRange::CLOCK_MEDIA) << "\",";
  out << "\"uncertainty_ns\":0,";
  out << "\"uncertainty_reason\":\"\"";
  out << "},\n";
  out << "  \"audio_clip_ref\":{";
  out << "\"clip_id\":\"" << jsonEscape(clip_ref.clip_id) << "\",";
  out << "\"uri\":\"" << jsonEscape(clip_ref.uri) << "\",";
  out << "\"codec\":\"" << jsonEscape(clip_ref.codec) << "\",";
  out << "\"container\":\"" << jsonEscape(clip_ref.container) << "\",";
  out << "\"payload_format\":\"" << jsonEscape(clip_ref.payload_format) << "\",";
  out << "\"sample_rate\":" << clip_ref.sample_rate << ",";
  out << "\"channels\":" << clip_ref.channels << ",";
  out << "\"duration_ns\":" << clip_ref.duration_ns;
  out << "}\n";
  out << "}\n";
  return out.str();
}

bool FaAudioWindowNode::writeTextAtomically(
  const std::filesystem::path & path,
  const std::string & content) const
{
  const std::filesystem::path temp_path = reserveTemporaryPublishPath(path);
  std::error_code remove_error;

  std::ofstream out(temp_path, std::ios::binary | std::ios::trunc);
  if (!out.is_open()) {
    std::filesystem::remove(temp_path, remove_error);
    throw std::runtime_error("failed to open archive metadata file");
  }
  out << content;
  out.close();
  if (!out.good()) {
    std::filesystem::remove(temp_path, remove_error);
    throw std::runtime_error("failed to write archive metadata file");
  }

  return publishTempFileNoClobber(
    temp_path,
    path,
    "failed to inspect deterministic metadata path",
    "deterministic archive metadata path exists with different content",
    "failed to publish archive metadata file") == PublishStatus::kPublished;
}

}  // namespace fa_audio_window

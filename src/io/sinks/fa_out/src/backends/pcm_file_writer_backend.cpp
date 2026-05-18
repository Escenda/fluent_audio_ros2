#include "fa_out/backends/pcm_file_writer_backend.hpp"

#include <filesystem>
#include <limits>
#include <utility>

namespace fa_out::backends
{

namespace
{
constexpr const char * kEncodingPcm16 = "PCM16LE";
constexpr const char * kEncodingPcm32 = "PCM32LE";
constexpr const char * kEncodingFloat32 = "FLOAT32LE";

bool isSupportedEncodingPair(const std::string & encoding, const uint32_t bit_depth)
{
  return (encoding == kEncodingPcm16 && bit_depth == 16u) ||
         (encoding == kEncodingPcm32 && bit_depth == 32u) ||
         (encoding == kEncodingFloat32 && bit_depth == 32u);
}
}  // namespace

PcmFileWriterBackend::PcmFileWriterBackend(PcmFileWriterConfig config)
: config_(std::move(config))
{
}

PcmFileWriterBackend::~PcmFileWriterBackend()
{
  close();
}

SinkOpenInfo PcmFileWriterBackend::open()
{
  validateConfig();
  close();

  const std::filesystem::path file_path(config_.file_path);
  const auto parent_path = file_path.parent_path();
  std::error_code error;
  if (!parent_path.empty() && !std::filesystem::is_directory(parent_path, error)) {
    throw SinkBackendError("file.path parent directory must exist: " + parent_path.string());
  }
  if (std::filesystem::is_directory(file_path, error)) {
    throw SinkBackendError("file.path must not be a directory: " + config_.file_path);
  }
  if (std::filesystem::exists(file_path, error) && !config_.overwrite_enabled) {
    throw SinkBackendError(
      "file.path already exists and overwrite.enabled=false: " + config_.file_path);
  }

  stream_.open(file_path, std::ios::binary | std::ios::trunc);
  if (!stream_.is_open()) {
    throw SinkBackendError("failed to open file.path for writing: " + config_.file_path);
  }

  bytes_per_frame_ =
    static_cast<size_t>(config_.channels) * static_cast<size_t>(config_.bit_depth / 8u);
  bytes_written_ = 0;

  SinkOpenInfo open_info;
  open_info.info_messages.push_back("PCM file writer target: " + config_.file_path);
  return open_info;
}

void PcmFileWriterBackend::close()
{
  if (stream_.is_open()) {
    stream_.close();
  }
}

bool PcmFileWriterBackend::isOpen() const
{
  return stream_.is_open();
}

bool PcmFileWriterBackend::isRunning() const
{
  return false;
}

size_t PcmFileWriterBackend::writeFrames(const uint8_t * data, const size_t frame_count)
{
  if (!stream_.is_open()) {
    throw SinkBackendError("pcm file writer is not open");
  }
  if (data == nullptr) {
    throw SinkBackendError("pcm file writer data is null");
  }
  if (frame_count == 0) {
    throw SinkBackendError("pcm file writer frame_count must be > 0");
  }
  if (bytes_per_frame_ == 0) {
    throw SinkBackendError("pcm file writer bytes_per_frame is not configured");
  }

  if (frame_count > (std::numeric_limits<size_t>::max() / bytes_per_frame_)) {
    throw SinkBackendError("pcm file writer byte count overflow");
  }
  const size_t byte_count = frame_count * bytes_per_frame_;
  if (byte_count > static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
    throw SinkBackendError("pcm file writer byte count exceeds streamsize range");
  }
  stream_.write(reinterpret_cast<const char *>(data), static_cast<std::streamsize>(byte_count));
  stream_.flush();
  if (!stream_) {
    throw SinkBackendError("failed while writing file.path: " + config_.file_path);
  }
  bytes_written_ += static_cast<uint64_t>(byte_count);
  return frame_count;
}

uint64_t PcmFileWriterBackend::bytesWritten() const
{
  return bytes_written_;
}

void PcmFileWriterBackend::validateConfig() const
{
  if (config_.file_path.empty()) {
    throw SinkBackendError("file.path is required for backend.name=pcm_file_writer");
  }
  if (config_.channels == 0) {
    throw SinkBackendError("audio.channels must be > 0");
  }
  if (config_.bit_depth == 0 || (config_.bit_depth % 8u) != 0u) {
    throw SinkBackendError("audio.bit_depth must be positive and byte-aligned");
  }
  if (!isSupportedEncodingPair(config_.encoding, config_.bit_depth)) {
    throw SinkBackendError(
      "audio.encoding/audio.bit_depth must be one of PCM16LE/16, PCM32LE/32, FLOAT32LE/32");
  }
}

}  // namespace fa_out::backends

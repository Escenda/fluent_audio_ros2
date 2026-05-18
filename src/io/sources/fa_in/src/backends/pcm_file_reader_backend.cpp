#include "fa_in/backends/pcm_file_reader_backend.hpp"

#include <chrono>
#include <filesystem>
#include <limits>
#include <thread>

namespace fa_in::backends
{

PcmFileReaderBackend::~PcmFileReaderBackend()
{
  close();
}

std::vector<DeviceInfo> PcmFileReaderBackend::listDevices() const
{
  if (path_.empty()) {
    return {};
  }
  return {DeviceInfo{path_, path_, 0, 0}};
}

DeviceInfo PcmFileReaderBackend::selectDevice(const DeviceSelector& /*selector*/) const
{
  throw BackendError("pcm_file_reader uses file.path and does not support device selection");
}

size_t PcmFileReaderBackend::open(
  const std::string& device_id,
  const AudioFormat& format,
  const size_t requested_frames)
{
  close();

  if (device_id.empty()) {
    throw BackendError("file.path is required for backend.name=pcm_file_reader");
  }
  if (requested_frames == 0) {
    throw BackendError("requested file source frames must be > 0");
  }
  if (format.channels == 0 || format.bit_depth == 0 || (format.bit_depth % 8u) != 0u) {
    throw BackendError("file source format must define positive byte-aligned channels and bit depth");
  }

  const std::filesystem::path file_path(device_id);
  std::error_code error;
  if (!std::filesystem::is_regular_file(file_path, error)) {
    throw BackendError("file.path must point to an existing regular file: " + device_id);
  }

  const auto size = std::filesystem::file_size(file_path, error);
  if (error) {
    throw BackendError("failed to read file size for " + device_id + ": " + error.message());
  }
  if (size == 0) {
    throw BackendError("file payload must not be empty: " + device_id);
  }
  if (size > std::numeric_limits<uint64_t>::max()) {
    throw BackendError("file payload is too large to address: " + device_id);
  }

  bytes_per_frame_ =
    static_cast<size_t>(format.channels) * static_cast<size_t>(format.bit_depth / 8u);
  if ((static_cast<uint64_t>(size) % static_cast<uint64_t>(bytes_per_frame_)) != 0u) {
    throw BackendError("file payload byte size must be divisible by expected frame byte size");
  }

  stream_.open(file_path, std::ios::binary);
  if (!stream_.is_open()) {
    throw BackendError("failed to open file.path: " + device_id);
  }

  path_ = device_id;
  file_size_bytes_ = static_cast<uint64_t>(size);
  loop_ = format.loop;
  chunk_ms_ = format.chunk_ms;
  return requested_frames;
}

void PcmFileReaderBackend::close()
{
  if (stream_.is_open()) {
    stream_.close();
  }
}

void PcmFileReaderBackend::drop()
{
}

ReadResult PcmFileReaderBackend::read(uint8_t* data, const size_t frames)
{
  if (data == nullptr) {
    return ReadResult{ReadStatus::kError, 0, "pcm file reader destination is null"};
  }
  if (frames == 0) {
    return ReadResult{ReadStatus::kError, 0, "pcm file reader requested frames must be > 0"};
  }
  if (bytes_per_frame_ == 0) {
    return ReadResult{ReadStatus::kError, 0, "pcm file reader bytes_per_frame is not configured"};
  }

  const size_t requested_bytes = frames * bytes_per_frame_;
  size_t read_count = 0;
  try {
    read_count = readBytes(data, requested_bytes);
    if (read_count == 0 && loop_) {
      reset();
      read_count = readBytes(data, requested_bytes);
    }
  } catch (const BackendError& e) {
    return ReadResult{ReadStatus::kError, 0, e.what()};
  }

  if (read_count == 0) {
    return ReadResult{ReadStatus::kEndOfStream, 0, ""};
  }
  if ((read_count % bytes_per_frame_) != 0) {
    return ReadResult{ReadStatus::kError, 0, "pcm file reader returned a partial frame"};
  }

  if (chunk_ms_ > 0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(chunk_ms_));
  }
  return ReadResult{ReadStatus::kOk, read_count / bytes_per_frame_, ""};
}

void PcmFileReaderBackend::reset()
{
  if (!stream_.is_open()) {
    throw BackendError("pcm file reader is not open");
  }
  stream_.clear();
  stream_.seekg(0, std::ios::beg);
  if (!stream_) {
    throw BackendError("failed to seek file.path to beginning: " + path_);
  }
}

size_t PcmFileReaderBackend::readBytes(uint8_t* destination, const size_t requested_bytes)
{
  if (!stream_.is_open()) {
    throw BackendError("pcm file reader is not open");
  }
  if (requested_bytes == 0) {
    throw BackendError("pcm file reader requested_bytes must be > 0");
  }

  stream_.read(reinterpret_cast<char*>(destination), static_cast<std::streamsize>(requested_bytes));
  const auto read_count = stream_.gcount();
  if (read_count < 0) {
    throw BackendError("pcm file reader returned a negative byte count");
  }
  if (stream_.bad()) {
    throw BackendError("failed while reading file.path: " + path_);
  }
  return static_cast<size_t>(read_count);
}

}  // namespace fa_in::backends

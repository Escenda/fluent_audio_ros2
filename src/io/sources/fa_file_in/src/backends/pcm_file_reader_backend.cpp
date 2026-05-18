#include "fa_file_in/backends/pcm_file_reader_backend.hpp"

#include <filesystem>
#include <limits>

namespace fa_file_in::backends
{

BackendError::BackendError(const std::string & message)
: std::runtime_error(message)
{
}

PcmFileReaderBackend::~PcmFileReaderBackend()
{
  close();
}

void PcmFileReaderBackend::open(const std::string & path)
{
  close();

  if (path.empty()) {
    throw BackendError("file.path is required");
  }

  const std::filesystem::path file_path(path);
  std::error_code error;
  if (!std::filesystem::is_regular_file(file_path, error)) {
    throw BackendError("file.path must point to an existing regular file: " + path);
  }

  const auto size = std::filesystem::file_size(file_path, error);
  if (error) {
    throw BackendError("failed to read file size for " + path + ": " + error.message());
  }
  if (size == 0) {
    throw BackendError("file payload must not be empty: " + path);
  }
  if (size > std::numeric_limits<uint64_t>::max()) {
    throw BackendError("file payload is too large to address: " + path);
  }

  stream_.open(file_path, std::ios::binary);
  if (!stream_.is_open()) {
    throw BackendError("failed to open file.path: " + path);
  }

  path_ = path;
  file_size_bytes_ = static_cast<uint64_t>(size);
}

void PcmFileReaderBackend::close()
{
  if (stream_.is_open()) {
    stream_.close();
  }
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

size_t PcmFileReaderBackend::read(uint8_t * destination, const size_t requested_bytes)
{
  if (!stream_.is_open()) {
    throw BackendError("pcm file reader is not open");
  }
  if (destination == nullptr) {
    throw BackendError("pcm file reader destination is null");
  }
  if (requested_bytes == 0) {
    throw BackendError("pcm file reader requested_bytes must be > 0");
  }

  stream_.read(reinterpret_cast<char *>(destination), static_cast<std::streamsize>(requested_bytes));
  const auto read_count = stream_.gcount();
  if (read_count < 0) {
    throw BackendError("pcm file reader returned a negative byte count");
  }
  if (stream_.bad()) {
    throw BackendError("failed while reading file.path: " + path_);
  }
  return static_cast<size_t>(read_count);
}

uint64_t PcmFileReaderBackend::fileSizeBytes() const
{
  return file_size_bytes_;
}

bool PcmFileReaderBackend::isOpen() const
{
  return stream_.is_open();
}

}  // namespace fa_file_in::backends

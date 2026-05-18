#include "fa_file_out/backends/pcm_file_writer_backend.hpp"

#include <filesystem>

namespace fa_file_out::backends
{

BackendError::BackendError(const std::string & message)
: std::runtime_error(message)
{
}

PcmFileWriterBackend::~PcmFileWriterBackend()
{
  close();
}

void PcmFileWriterBackend::open(const std::string & path, const bool overwrite_enabled)
{
  close();

  if (path.empty()) {
    throw BackendError("file.path is required");
  }

  const std::filesystem::path file_path(path);
  const auto parent_path = file_path.parent_path();
  std::error_code error;
  if (!parent_path.empty() && !std::filesystem::is_directory(parent_path, error)) {
    throw BackendError("file.path parent directory must exist: " + parent_path.string());
  }
  if (std::filesystem::is_directory(file_path, error)) {
    throw BackendError("file.path must not be a directory: " + path);
  }
  if (std::filesystem::exists(file_path, error) && !overwrite_enabled) {
    throw BackendError("file.path already exists and overwrite.enabled=false: " + path);
  }

  stream_.open(file_path, std::ios::binary | std::ios::trunc);
  if (!stream_.is_open()) {
    throw BackendError("failed to open file.path for writing: " + path);
  }

  path_ = path;
  bytes_written_ = 0;
}

void PcmFileWriterBackend::close()
{
  if (stream_.is_open()) {
    stream_.close();
  }
}

void PcmFileWriterBackend::write(const uint8_t * data, const size_t byte_count)
{
  if (!stream_.is_open()) {
    throw BackendError("pcm file writer is not open");
  }
  if (data == nullptr) {
    throw BackendError("pcm file writer data is null");
  }
  if (byte_count == 0) {
    throw BackendError("pcm file writer byte_count must be > 0");
  }

  stream_.write(reinterpret_cast<const char *>(data), static_cast<std::streamsize>(byte_count));
  stream_.flush();
  if (!stream_) {
    throw BackendError("failed while writing file.path: " + path_);
  }
  bytes_written_ += static_cast<uint64_t>(byte_count);
}

uint64_t PcmFileWriterBackend::bytesWritten() const
{
  return bytes_written_;
}

bool PcmFileWriterBackend::isOpen() const
{
  return stream_.is_open();
}

}  // namespace fa_file_out::backends

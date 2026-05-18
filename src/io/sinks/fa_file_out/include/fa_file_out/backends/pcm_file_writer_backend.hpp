#pragma once

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>

namespace fa_file_out::backends
{

class BackendError : public std::runtime_error
{
public:
  explicit BackendError(const std::string & message);
};

class PcmFileWriterBackend
{
public:
  PcmFileWriterBackend() = default;
  ~PcmFileWriterBackend();

  void open(const std::string & path, bool overwrite_enabled);
  void close();
  void write(const uint8_t * data, size_t byte_count);

  uint64_t bytesWritten() const;
  bool isOpen() const;

private:
  std::string path_{};
  std::ofstream stream_{};
  uint64_t bytes_written_{0};
};

}  // namespace fa_file_out::backends

#pragma once

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>

namespace fa_file_in::backends
{

class BackendError : public std::runtime_error
{
public:
  explicit BackendError(const std::string & message);
};

class PcmFileReaderBackend
{
public:
  PcmFileReaderBackend() = default;
  ~PcmFileReaderBackend();

  void open(const std::string & path);
  void close();
  void reset();
  size_t read(uint8_t * destination, size_t requested_bytes);

  uint64_t fileSizeBytes() const;
  bool isOpen() const;

private:
  std::string path_{};
  std::ifstream stream_{};
  uint64_t file_size_bytes_{0};
};

}  // namespace fa_file_in::backends

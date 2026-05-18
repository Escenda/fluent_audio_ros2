#pragma once

#include "fa_in/backends/source_backend.hpp"

#include <cstdint>
#include <fstream>
#include <string>

namespace fa_in::backends
{

class PcmFileReaderBackend final : public SourceBackend
{
public:
  PcmFileReaderBackend() = default;
  ~PcmFileReaderBackend() override;

  std::vector<DeviceInfo> listDevices() const override;
  DeviceInfo selectDevice(const DeviceSelector& selector) const override;
  size_t open(const std::string& device_id, const AudioFormat& format, size_t requested_frames) override;
  void close() override;
  void drop() override;
  ReadResult read(uint8_t* data, size_t frames) override;

private:
  void reset();
  size_t readBytes(uint8_t* destination, size_t requested_bytes);

  std::ifstream stream_;
  std::string path_;
  uint64_t file_size_bytes_{0};
  size_t bytes_per_frame_{0};
  bool loop_{false};
  uint32_t chunk_ms_{0};
};

}  // namespace fa_in::backends

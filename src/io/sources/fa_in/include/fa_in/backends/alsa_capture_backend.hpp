#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "fa_in/backends/source_backend.hpp"

namespace fa_in
{
namespace backends
{

class AlsaCaptureBackend final : public SourceBackend
{
public:
  AlsaCaptureBackend();
  ~AlsaCaptureBackend() override;

  AlsaCaptureBackend(const AlsaCaptureBackend&) = delete;
  AlsaCaptureBackend& operator=(const AlsaCaptureBackend&) = delete;
  AlsaCaptureBackend(AlsaCaptureBackend&&) = delete;
  AlsaCaptureBackend& operator=(AlsaCaptureBackend&&) = delete;

  std::vector<DeviceInfo> listDevices() const override;
  DeviceInfo selectDevice(const DeviceSelector& selector) const override;
  size_t open(const std::string& device_id, const AudioFormat& format, size_t requested_frames) override;
  void close() override;
  void drop() override;
  ReadResult read(uint8_t* data, size_t frames) override;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace backends
}  // namespace fa_in

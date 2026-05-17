#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace fa_in
{
namespace backends
{

struct AudioFormat
{
  uint32_t sample_rate{0};
  uint32_t channels{0};
  uint32_t bit_depth{0};
  uint32_t chunk_ms{0};
  std::string encoding{};
  std::string layout{};
};

struct DeviceSelector
{
  std::string mode{};
  std::string identifier{};
  int index{-1};
};

struct DeviceInfo
{
  std::string id{};
  std::string name{};
  uint32_t max_input_channels{0};
  uint32_t default_sample_rate{0};
};

enum class ReadStatus
{
  kOk,
  kXrun,
  kError,
  kZeroFrames,
};

struct ReadResult
{
  ReadStatus status{ReadStatus::kError};
  size_t frames{0};
  std::string message{};
};

class BackendError : public std::runtime_error
{
public:
  explicit BackendError(const std::string& message)
  : std::runtime_error(message)
  {
  }
};

class SourceBackend
{
public:
  virtual ~SourceBackend() = default;

  virtual std::vector<DeviceInfo> listDevices() const = 0;
  virtual DeviceInfo selectDevice(const DeviceSelector& selector) const = 0;
  virtual size_t open(const std::string& device_id, const AudioFormat& format, size_t requested_frames) = 0;
  virtual void close() = 0;
  virtual void drop() = 0;
  virtual ReadResult read(uint8_t* data, size_t frames) = 0;
};

}  // namespace backends
}  // namespace fa_in

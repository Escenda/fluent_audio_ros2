#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace fa_out::backends
{

struct AlsaPlaybackConfig
{
  std::string device_id{};
  std::string encoding{};
  uint32_t sample_rate{0};
  uint32_t channels{0};
  uint32_t bit_depth{0};
};

struct AlsaPlaybackOpenInfo
{
  unsigned long buffer_size_frames{0};
  unsigned long period_size_frames{0};
  unsigned long start_threshold_frames{0};
  bool software_params_applied{false};
  std::vector<std::string> warnings;
};

class AlsaPlaybackError : public std::runtime_error
{
public:
  explicit AlsaPlaybackError(const std::string & message);
};

class AlsaPlaybackBackend
{
public:
  explicit AlsaPlaybackBackend(AlsaPlaybackConfig config);
  ~AlsaPlaybackBackend();

  AlsaPlaybackBackend(const AlsaPlaybackBackend &) = delete;
  AlsaPlaybackBackend & operator=(const AlsaPlaybackBackend &) = delete;
  AlsaPlaybackBackend(AlsaPlaybackBackend &&) = delete;
  AlsaPlaybackBackend & operator=(AlsaPlaybackBackend &&) = delete;

  static bool isRawHardwareDevice(const std::string & device_id);

  AlsaPlaybackOpenInfo open();
  void close();
  bool isOpen() const;
  bool isRunning() const;
  void discardBuffer(const std::string & operation);
  size_t writeFrames(const uint8_t * data, size_t frame_count);

private:
  struct Impl;

  void validateConfig() const;

  AlsaPlaybackConfig config_;
  std::unique_ptr<Impl> impl_;
};

}  // namespace fa_out::backends

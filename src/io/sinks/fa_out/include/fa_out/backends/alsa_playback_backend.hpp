#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "fa_out/backends/sink_backend.hpp"

namespace fa_out::backends
{

struct AlsaPlaybackConfig
{
  std::string device_id{};
  std::string encoding{};
  uint32_t sample_rate{0};
  uint32_t channels{0};
  uint32_t bit_depth{0};
  size_t buffer_frames{0};
  size_t period_frames{0};
};

class AlsaPlaybackError : public SinkBackendError
{
public:
  explicit AlsaPlaybackError(const std::string & message);
};

class AlsaPlaybackBackend final : public SinkBackend
{
public:
  explicit AlsaPlaybackBackend(AlsaPlaybackConfig config);
  ~AlsaPlaybackBackend();

  AlsaPlaybackBackend(const AlsaPlaybackBackend &) = delete;
  AlsaPlaybackBackend & operator=(const AlsaPlaybackBackend &) = delete;
  AlsaPlaybackBackend(AlsaPlaybackBackend &&) = delete;
  AlsaPlaybackBackend & operator=(AlsaPlaybackBackend &&) = delete;

  SinkOpenInfo open() override;
  void close() override;
  bool isOpen() const override;
  bool isRunning() const override;
  void discardBuffer(const std::string & operation) override;
  size_t writeFrames(const uint8_t * data, size_t frame_count) override;

private:
  struct Impl;

  void validateConfig() const;

  AlsaPlaybackConfig config_;
  std::unique_ptr<Impl> impl_;
};

}  // namespace fa_out::backends

#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace fa_out::backends
{

struct SinkOpenInfo
{
  std::vector<std::string> info_messages;
};

class SinkBackendError : public std::runtime_error
{
public:
  explicit SinkBackendError(const std::string & message);
};

class SinkBackend
{
public:
  virtual ~SinkBackend() = default;

  virtual SinkOpenInfo open() = 0;
  virtual void close() = 0;
  virtual bool isOpen() const = 0;
  virtual bool isRunning() const = 0;
  virtual void discardBuffer(const std::string & operation) = 0;
  virtual size_t writeFrames(const uint8_t * data, size_t frame_count) = 0;
};

}  // namespace fa_out::backends

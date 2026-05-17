#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

namespace fa_record::backends
{

struct AudioFormat
{
  uint32_t sample_rate{0};
  uint32_t channels{0};
  uint32_t bit_depth{0};
  std::string encoding;
};

class FileWriterError : public std::runtime_error
{
public:
  explicit FileWriterError(const std::string & message);
};

class FormatMismatchError final : public FileWriterError
{
public:
  explicit FormatMismatchError(const std::string & message);
};

class FileWriterBackend
{
public:
  virtual ~FileWriterBackend() = default;

  virtual void open(const std::string & path) = 0;
  virtual void startFormat(const AudioFormat & format) = 0;
  virtual void writeChunk(const AudioFormat & format, const uint8_t * data, size_t data_size) = 0;
  virtual void close() = 0;
  virtual bool isOpen() const = 0;
  virtual bool hasFormat() const = 0;
  virtual const std::string & path() const = 0;
};

class WavFileWriterBackend final : public FileWriterBackend
{
public:
  WavFileWriterBackend();
  ~WavFileWriterBackend() override;

  void open(const std::string & path) override;
  void startFormat(const AudioFormat & format) override;
  void writeChunk(const AudioFormat & format, const uint8_t * data, size_t data_size) override;
  void close() override;
  bool isOpen() const override;
  bool hasFormat() const override;
  const std::string & path() const override;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace fa_record::backends

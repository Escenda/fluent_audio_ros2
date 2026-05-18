#pragma once

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <string>

#include "fa_out/backends/sink_backend.hpp"

namespace fa_out::backends
{

struct PcmFileWriterConfig
{
  std::string file_path{};
  std::string encoding{};
  uint32_t channels{0};
  uint32_t bit_depth{0};
  bool overwrite_enabled{false};
};

class PcmFileWriterBackend final : public SinkBackend
{
public:
  explicit PcmFileWriterBackend(PcmFileWriterConfig config);
  ~PcmFileWriterBackend();

  PcmFileWriterBackend(const PcmFileWriterBackend &) = delete;
  PcmFileWriterBackend & operator=(const PcmFileWriterBackend &) = delete;
  PcmFileWriterBackend(PcmFileWriterBackend &&) = delete;
  PcmFileWriterBackend & operator=(PcmFileWriterBackend &&) = delete;

  SinkOpenInfo open() override;
  void close() override;
  bool isOpen() const override;
  bool isRunning() const override;
  size_t writeFrames(const uint8_t * data, size_t frame_count) override;

  uint64_t bytesWritten() const;

private:
  void validateConfig() const;

  PcmFileWriterConfig config_;
  std::ofstream stream_{};
  size_t bytes_per_frame_{0};
  uint64_t bytes_written_{0};
};

}  // namespace fa_out::backends

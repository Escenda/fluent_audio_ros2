#include "fa_record/backends/file_writer_backend.hpp"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>

namespace fa_record::backends
{

namespace
{

bool formatMatches(const AudioFormat & a, const AudioFormat & b)
{
  return a.sample_rate == b.sample_rate &&
         a.channels == b.channels &&
         a.bit_depth == b.bit_depth &&
         a.encoding == b.encoding;
}

bool isSupportedEncoding(const AudioFormat & format)
{
  if (format.bit_depth == 16) {
    return format.encoding == "PCM16LE";
  }
  if (format.bit_depth == 32) {
    return format.encoding == "FLOAT32LE";
  }
  return false;
}

void validateFormat(const AudioFormat & format)
{
  if (format.sample_rate == 0 || format.channels == 0) {
    throw FileWriterError("recording AudioFormat requires positive sample_rate and channels");
  }
  if (format.bit_depth != 16 && format.bit_depth != 32) {
    throw FileWriterError("recording AudioFormat bit_depth must be 16 or 32");
  }
  if (format.encoding.empty()) {
    throw FileWriterError("recording AudioFormat encoding is required");
  }
  if (!isSupportedEncoding(format)) {
    throw FileWriterError("recording AudioFormat encoding must be PCM16LE/16-bit or FLOAT32LE/32-bit");
  }
}

void writeWavHeader(std::fstream & stream, const AudioFormat & format, uint32_t data_length)
{
  struct WavHeader
  {
    char riff[4];
    uint32_t chunk_size;
    char wave[4];
    char fmt[4];
    uint32_t subchunk1_size;
    uint16_t audio_format;
    uint16_t num_channels;
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample;
    char data[4];
    uint32_t subchunk2_size;
  } header{};

  std::memcpy(header.riff, "RIFF", 4);
  std::memcpy(header.wave, "WAVE", 4);
  std::memcpy(header.fmt, "fmt ", 4);
  std::memcpy(header.data, "data", 4);

  header.subchunk1_size = 16;
  header.audio_format = (format.bit_depth == 16) ? 1 : 3;
  header.num_channels = static_cast<uint16_t>(format.channels);
  header.sample_rate = format.sample_rate;
  const uint16_t bytes_per_sample = static_cast<uint16_t>(format.bit_depth / 8);
  header.byte_rate = format.sample_rate * format.channels * bytes_per_sample;
  header.block_align = format.channels * bytes_per_sample;
  header.bits_per_sample = static_cast<uint16_t>(format.bit_depth);
  header.subchunk2_size = data_length;
  header.chunk_size = 36 + data_length;

  stream.write(reinterpret_cast<const char *>(&header), sizeof(header));
  if (!stream) {
    throw FileWriterError("failed to write WAV header");
  }
}

void finalizeWavHeader(std::fstream & stream, uint32_t data_length)
{
  stream.seekp(4, std::ios::beg);
  const uint32_t chunk_size = 36 + data_length;
  stream.write(reinterpret_cast<const char *>(&chunk_size), sizeof(chunk_size));

  stream.seekp(40, std::ios::beg);
  stream.write(reinterpret_cast<const char *>(&data_length), sizeof(data_length));
  if (!stream) {
    throw FileWriterError("failed to finalize WAV header");
  }
}

}  // namespace

FileWriterError::FileWriterError(const std::string & message)
: std::runtime_error(message)
{
}

FormatMismatchError::FormatMismatchError(const std::string & message)
: FileWriterError(message)
{
}

struct WavFileWriterBackend::Impl
{
  void open(const std::string & path)
  {
    if (path.empty()) {
      throw FileWriterError("record path is empty");
    }

    std::filesystem::path target_path(path);
    const std::filesystem::path parent = target_path.parent_path();
    if (!parent.empty()) {
      std::error_code ec;
      if (!std::filesystem::exists(parent, ec) || ec) {
        throw FileWriterError("record parent directory does not exist: " + parent.string());
      }
      if (!std::filesystem::is_directory(parent, ec) || ec) {
        throw FileWriterError("record parent path is not a directory: " + parent.string());
      }
    }

    stream_.open(target_path, std::ios::binary | std::ios::out | std::ios::trunc);
    if (!stream_.is_open()) {
      throw FileWriterError("failed to open record file: " + path);
    }

    path_ = target_path.string();
    header_written_ = false;
    recorded_bytes_ = 0;
    active_format_ = AudioFormat{};
  }

  void startFormat(const AudioFormat & format)
  {
    ensureOpen();
    if (header_written_) {
      if (!formatMatches(active_format_, format)) {
        throw FormatMismatchError("recording format changed during active file");
      }
      return;
    }
    validateFormat(format);
    active_format_ = format;
    writeWavHeader(stream_, active_format_, 0);
    header_written_ = true;
  }

  void writeChunk(const AudioFormat & format, const uint8_t * data, size_t data_size)
  {
    ensureOpen();
    if (data == nullptr || data_size == 0) {
      throw FileWriterError("recording chunk data is required");
    }
    startFormat(format);
    if (data_size > (std::numeric_limits<uint32_t>::max() - recorded_bytes_)) {
      throw FileWriterError("recording data exceeds WAV uint32 data length");
    }
    stream_.write(reinterpret_cast<const char *>(data), static_cast<std::streamsize>(data_size));
    if (!stream_) {
      throw FileWriterError("write failed while recording: " + path_);
    }
    recorded_bytes_ += static_cast<uint32_t>(data_size);
  }

  void close()
  {
    if (stream_.is_open() && header_written_) {
      finalizeWavHeader(stream_, recorded_bytes_);
    }
    if (stream_.is_open()) {
      stream_.close();
    }
    header_written_ = false;
    recorded_bytes_ = 0;
    path_.clear();
    active_format_ = AudioFormat{};
  }

  bool isOpen() const
  {
    return stream_.is_open();
  }

  bool hasFormat() const
  {
    return header_written_;
  }

  const std::string & path() const
  {
    return path_;
  }

  void ensureOpen() const
  {
    if (!stream_.is_open()) {
      throw FileWriterError("record file is not open");
    }
  }

  std::fstream stream_;
  bool header_written_{false};
  uint32_t recorded_bytes_{0};
  std::string path_;
  AudioFormat active_format_{};
};

WavFileWriterBackend::WavFileWriterBackend()
: impl_(std::make_unique<Impl>())
{
}

WavFileWriterBackend::~WavFileWriterBackend() = default;

void WavFileWriterBackend::open(const std::string & path)
{
  impl_->open(path);
}

void WavFileWriterBackend::startFormat(const AudioFormat & format)
{
  impl_->startFormat(format);
}

void WavFileWriterBackend::writeChunk(const AudioFormat & format, const uint8_t * data, size_t data_size)
{
  impl_->writeChunk(format, data, data_size);
}

void WavFileWriterBackend::close()
{
  impl_->close();
}

bool WavFileWriterBackend::isOpen() const
{
  return impl_->isOpen();
}

bool WavFileWriterBackend::hasFormat() const
{
  return impl_->hasFormat();
}

const std::string & WavFileWriterBackend::path() const
{
  return impl_->path();
}

}  // namespace fa_record::backends

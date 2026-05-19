#include "fa_audio_window/wav_writer.hpp"

#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

namespace fa_audio_window
{

namespace
{
void writeAscii(std::ofstream & output, const char * value, const size_t size)
{
  output.write(value, static_cast<std::streamsize>(size));
}

void writeU16Le(std::ofstream & output, const uint16_t value)
{
  const char bytes[2] = {
    static_cast<char>(value & 0xffu),
    static_cast<char>((value >> 8u) & 0xffu),
  };
  output.write(bytes, 2);
}

void writeU32Le(std::ofstream & output, const uint32_t value)
{
  const char bytes[4] = {
    static_cast<char>(value & 0xffu),
    static_cast<char>((value >> 8u) & 0xffu),
    static_cast<char>((value >> 16u) & 0xffu),
    static_cast<char>((value >> 24u) & 0xffu),
  };
  output.write(bytes, 4);
}
}  // namespace

void WavWriter::writePcm16Le(
  const std::filesystem::path & path,
  const AudioFormat & format,
  const std::vector<uint8_t> & pcm_data)
{
  if (format.encoding != "PCM16LE" || format.bit_depth != 16u || format.layout != "interleaved") {
    throw std::runtime_error("WAV writer only supports PCM16LE/16-bit/interleaved");
  }
  if (format.sample_rate == 0u || format.channels == 0u) {
    throw std::runtime_error("WAV writer requires sample_rate and channels");
  }
  const size_t sample_frame_bytes = bytesPerSampleFrame(format);
  if (pcm_data.empty() || pcm_data.size() % sample_frame_bytes != 0u) {
    throw std::runtime_error("PCM data must be non-empty and sample-frame aligned");
  }
  if (pcm_data.size() > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
    throw std::runtime_error("PCM data is too large for RIFF WAV");
  }

  const uint32_t data_size = static_cast<uint32_t>(pcm_data.size());
  const uint32_t riff_size = 36u + data_size;
  const uint16_t channels = static_cast<uint16_t>(format.channels);
  const uint16_t bits_per_sample = static_cast<uint16_t>(format.bit_depth);
  const uint32_t byte_rate =
    format.sample_rate * static_cast<uint32_t>(format.channels) * static_cast<uint32_t>(bytesPerSample(format));
  const uint16_t block_align =
    static_cast<uint16_t>(static_cast<uint32_t>(format.channels) * static_cast<uint32_t>(bytesPerSample(format)));

  const std::filesystem::path parent = path.parent_path();
  if (!parent.empty()) {
    std::filesystem::create_directories(parent);
  }

  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) {
    throw std::runtime_error("failed to open WAV output: " + path.string());
  }

  writeAscii(output, "RIFF", 4);
  writeU32Le(output, riff_size);
  writeAscii(output, "WAVE", 4);
  writeAscii(output, "fmt ", 4);
  writeU32Le(output, 16u);
  writeU16Le(output, 1u);
  writeU16Le(output, channels);
  writeU32Le(output, format.sample_rate);
  writeU32Le(output, byte_rate);
  writeU16Le(output, block_align);
  writeU16Le(output, bits_per_sample);
  writeAscii(output, "data", 4);
  writeU32Le(output, data_size);
  output.write(
    reinterpret_cast<const char *>(pcm_data.data()),
    static_cast<std::streamsize>(pcm_data.size()));

  if (!output) {
    throw std::runtime_error("failed to write WAV output: " + path.string());
  }
}

}  // namespace fa_audio_window

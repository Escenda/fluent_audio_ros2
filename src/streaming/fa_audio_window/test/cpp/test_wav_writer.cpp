#include "fa_audio_window/wav_writer.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <vector>

#include <gtest/gtest.h>

namespace
{
uint32_t readU32Le(const std::vector<uint8_t> & bytes, const size_t offset)
{
  return static_cast<uint32_t>(bytes[offset]) |
    (static_cast<uint32_t>(bytes[offset + 1u]) << 8u) |
    (static_cast<uint32_t>(bytes[offset + 2u]) << 16u) |
    (static_cast<uint32_t>(bytes[offset + 3u]) << 24u);
}

uint16_t readU16Le(const std::vector<uint8_t> & bytes, const size_t offset)
{
  return static_cast<uint16_t>(
    static_cast<uint16_t>(bytes[offset]) |
    static_cast<uint16_t>(bytes[offset + 1u] << 8u));
}
}  // namespace

TEST(WavWriterTest, WritesPcm16LeHeaderAndData)
{
  const std::filesystem::path path =
    std::filesystem::temp_directory_path() / "fa_audio_window_wav_writer_test.wav";
  const fa_audio_window::AudioFormat format{"PCM16LE", 8000u, 1u, 16u, "interleaved"};
  const std::vector<uint8_t> pcm{0x01, 0x00, 0xff, 0x7f};

  fa_audio_window::WavWriter::writePcm16Le(path, format, pcm);

  std::ifstream input(path, std::ios::binary);
  ASSERT_TRUE(input.good());
  const std::vector<uint8_t> bytes(
    (std::istreambuf_iterator<char>(input)),
    std::istreambuf_iterator<char>());

  ASSERT_EQ(bytes.size(), 48u);
  EXPECT_EQ(bytes[0], 'R');
  EXPECT_EQ(bytes[1], 'I');
  EXPECT_EQ(bytes[2], 'F');
  EXPECT_EQ(bytes[3], 'F');
  EXPECT_EQ(readU32Le(bytes, 4u), 40u);
  EXPECT_EQ(bytes[8], 'W');
  EXPECT_EQ(bytes[9], 'A');
  EXPECT_EQ(bytes[10], 'V');
  EXPECT_EQ(bytes[11], 'E');
  EXPECT_EQ(readU16Le(bytes, 20u), 1u);
  EXPECT_EQ(readU16Le(bytes, 22u), 1u);
  EXPECT_EQ(readU32Le(bytes, 24u), 8000u);
  EXPECT_EQ(readU32Le(bytes, 28u), 16000u);
  EXPECT_EQ(readU16Le(bytes, 32u), 2u);
  EXPECT_EQ(readU16Le(bytes, 34u), 16u);
  EXPECT_EQ(readU32Le(bytes, 40u), 4u);

  const std::vector<uint8_t> data(bytes.begin() + 44, bytes.end());
  EXPECT_EQ(data, pcm);

  std::filesystem::remove(path);
}

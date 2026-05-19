#include "fa_audio_window/audio_window_buffer.hpp"

#include <cstdint>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

namespace
{
constexpr int64_t kSecond = 1000000000LL;
constexpr int64_t kMillisecond = 1000000LL;

fa_audio_window::AudioFormat pcm16Mono1Hz()
{
  return {"PCM16LE", 1u, 1u, 16u, "interleaved"};
}

fa_audio_window::AudioFormat pcm16Mono10Hz()
{
  return {"PCM16LE", 10u, 1u, 16u, "interleaved"};
}

fa_audio_window::TimedAudioFrame frameAt(const int64_t start_ns, const uint8_t low_byte)
{
  return {
    {start_ns, start_ns + kSecond},
    static_cast<uint32_t>(start_ns / kSecond),
    {low_byte, 0x00},
  };
}

fa_audio_window::TimedAudioFrame frameWithSamples(
  const int64_t start_ns,
  const std::vector<uint8_t> & low_bytes,
  const uint32_t sample_rate)
{
  std::vector<uint8_t> data;
  data.reserve(low_bytes.size() * 2u);
  for (const uint8_t low_byte : low_bytes) {
    data.push_back(low_byte);
    data.push_back(0x00);
  }
  const int64_t duration_ns =
    static_cast<int64_t>(low_bytes.size()) * kSecond / static_cast<int64_t>(sample_rate);
  return {
    {start_ns, start_ns + duration_ns},
    static_cast<uint32_t>(start_ns / kSecond),
    data,
  };
}
}  // namespace

TEST(AudioWindowBufferTest, DropsFramesOutsideRetention)
{
  fa_audio_window::AudioWindowBuffer buffer(2u * static_cast<uint64_t>(kSecond), pcm16Mono1Hz());
  buffer.addFrame(frameAt(0LL * kSecond, 0x10));
  buffer.addFrame(frameAt(1LL * kSecond, 0x20));
  buffer.addFrame(frameAt(2LL * kSecond, 0x30));
  buffer.addFrame(frameAt(3LL * kSecond, 0x40));

  const fa_audio_window::TimeRange retained = buffer.retainedRange();
  EXPECT_EQ(retained.start_unix_ns, 2LL * kSecond);
  EXPECT_EQ(retained.end_unix_ns, 4LL * kSecond);

  const fa_audio_window::WindowQueryResult old_range = buffer.query({1LL * kSecond, 3LL * kSecond});
  EXPECT_EQ(old_range.status, fa_audio_window::WindowQueryStatus::kRangeOutsideWindow);
}

TEST(AudioWindowBufferTest, QueryReturnsTimestampOrderAndRequestedRange)
{
  fa_audio_window::AudioWindowBuffer buffer(5u * static_cast<uint64_t>(kSecond), pcm16Mono1Hz());
  buffer.addFrame(frameAt(3LL * kSecond, 0x30));
  buffer.addFrame(frameAt(1LL * kSecond, 0x10));
  buffer.addFrame(frameAt(2LL * kSecond, 0x20));

  const fa_audio_window::WindowQueryResult query = buffer.query({1LL * kSecond, 4LL * kSecond});

  ASSERT_EQ(query.status, fa_audio_window::WindowQueryStatus::kOk);
  EXPECT_EQ(query.exported_range.start_unix_ns, 1LL * kSecond);
  EXPECT_EQ(query.exported_range.end_unix_ns, 4LL * kSecond);
  const std::vector<uint8_t> expected{0x10, 0x00, 0x20, 0x00, 0x30, 0x00};
  EXPECT_EQ(query.pcm_data, expected);
}

TEST(AudioWindowBufferTest, EmptyWindowIsExplicit)
{
  fa_audio_window::AudioWindowBuffer buffer(2u * static_cast<uint64_t>(kSecond), pcm16Mono1Hz());
  const fa_audio_window::WindowQueryResult query = buffer.query({0LL, kSecond});
  EXPECT_EQ(query.status, fa_audio_window::WindowQueryStatus::kWindowNotFound);
}

TEST(AudioWindowBufferTest, NonAlignedEndIncludesLastSampleBeforeEnd)
{
  fa_audio_window::AudioWindowBuffer buffer(2u * static_cast<uint64_t>(kSecond), pcm16Mono10Hz());
  buffer.addFrame(frameWithSamples(0LL, {0x10, 0x20, 0x30, 0x40}, 10u));

  const fa_audio_window::WindowQueryResult query = buffer.query({0LL, 350LL * kMillisecond});

  ASSERT_EQ(query.status, fa_audio_window::WindowQueryStatus::kOk);
  EXPECT_EQ(query.exported_range.start_unix_ns, 0LL);
  EXPECT_EQ(query.exported_range.end_unix_ns, 400LL * kMillisecond);
  const std::vector<uint8_t> expected{0x10, 0x00, 0x20, 0x00, 0x30, 0x00, 0x40, 0x00};
  EXPECT_EQ(query.pcm_data, expected);
}

TEST(AudioWindowBufferTest, NonAlignedStartAndEndSnapToContainedSamples)
{
  fa_audio_window::AudioWindowBuffer buffer(2u * static_cast<uint64_t>(kSecond), pcm16Mono10Hz());
  buffer.addFrame(frameWithSamples(0LL, {0x10, 0x20, 0x30, 0x40}, 10u));

  const fa_audio_window::WindowQueryResult query =
    buffer.query({50LL * kMillisecond, 350LL * kMillisecond});

  ASSERT_EQ(query.status, fa_audio_window::WindowQueryStatus::kOk);
  EXPECT_EQ(query.exported_range.start_unix_ns, 100LL * kMillisecond);
  EXPECT_EQ(query.exported_range.end_unix_ns, 400LL * kMillisecond);
  const std::vector<uint8_t> expected{0x20, 0x00, 0x30, 0x00, 0x40, 0x00};
  EXPECT_EQ(query.pcm_data, expected);
}

TEST(AudioWindowBufferTest, QueryFailsWhenFramesDoNotContinuouslyCoverRequestedRange)
{
  fa_audio_window::AudioWindowBuffer buffer(5u * static_cast<uint64_t>(kSecond), pcm16Mono1Hz());
  buffer.addFrame(frameAt(0LL, 0x10));
  buffer.addFrame(frameAt(2LL * kSecond, 0x20));

  const fa_audio_window::WindowQueryResult query = buffer.query({0LL, 3LL * kSecond});

  EXPECT_EQ(query.status, fa_audio_window::WindowQueryStatus::kRangeNotContinuous);
  EXPECT_TRUE(query.pcm_data.empty());
}

TEST(AudioWindowBufferTest, RejectsOverlappingFrames)
{
  fa_audio_window::AudioWindowBuffer buffer(5u * static_cast<uint64_t>(kSecond), pcm16Mono1Hz());
  buffer.addFrame(frameAt(0LL, 0x10));

  EXPECT_THROW(
    buffer.addFrame(frameWithSamples(500LL * kMillisecond, {0x20}, 1u)),
    std::runtime_error);
}

TEST(AudioWindowBufferTest, RejectedOverlapDoesNotAdvanceRetentionState)
{
  fa_audio_window::AudioWindowBuffer buffer(2u * static_cast<uint64_t>(kSecond), pcm16Mono1Hz());
  buffer.addFrame(frameAt(0LL, 0x10));

  const std::vector<uint8_t> rejected_samples{
    0x20, 0x30, 0x40, 0x50, 0x60,
    0x70, 0x80, 0x90, 0xa0, 0xb0,
  };
  EXPECT_THROW(
    buffer.addFrame(frameWithSamples(500LL * kMillisecond, rejected_samples, 1u)),
    std::runtime_error);

  buffer.addFrame(frameAt(1LL * kSecond, 0x20));

  const fa_audio_window::TimeRange retained = buffer.retainedRange();
  EXPECT_EQ(retained.start_unix_ns, 0LL);
  EXPECT_EQ(retained.end_unix_ns, 2LL * kSecond);

  const fa_audio_window::WindowQueryResult query = buffer.query({0LL, 2LL * kSecond});
  ASSERT_EQ(query.status, fa_audio_window::WindowQueryStatus::kOk);
  EXPECT_EQ(query.exported_range.start_unix_ns, 0LL);
  EXPECT_EQ(query.exported_range.end_unix_ns, 2LL * kSecond);
  const std::vector<uint8_t> expected{0x10, 0x00, 0x20, 0x00};
  EXPECT_EQ(query.pcm_data, expected);
}

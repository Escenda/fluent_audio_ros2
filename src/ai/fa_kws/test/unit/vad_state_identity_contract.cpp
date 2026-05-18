#include <gtest/gtest.h>

#include "fa_interfaces/msg/vad_state.hpp"
#include "fa_kws/vad_state_identity.hpp"

namespace
{

fa_interfaces::msg::VadState make_vad_state()
{
  fa_interfaces::msg::VadState msg;
  msg.source_id = "mic-a";
  msg.stream_id = "audio/raw/mic";
  msg.probability = 0.8F;
  msg.is_speech = true;
  return msg;
}

}  // namespace

TEST(VadStateIdentityContract, AcceptsMatchingAudioBinding)
{
  const auto msg = make_vad_state();

  EXPECT_TRUE(fa_kws::vadStateMatchesAudioBinding(msg, "mic-a", "audio/raw/mic"));
}

TEST(VadStateIdentityContract, RejectsMissingVadIdentity)
{
  auto missing_source = make_vad_state();
  missing_source.source_id.clear();

  auto missing_stream = make_vad_state();
  missing_stream.stream_id.clear();

  EXPECT_FALSE(fa_kws::vadStateMatchesAudioBinding(missing_source, "mic-a", "audio/raw/mic"));
  EXPECT_FALSE(fa_kws::vadStateMatchesAudioBinding(missing_stream, "mic-a", "audio/raw/mic"));
}

TEST(VadStateIdentityContract, RejectsMissingExpectedBinding)
{
  const auto msg = make_vad_state();

  EXPECT_FALSE(fa_kws::vadStateMatchesAudioBinding(msg, "", "audio/raw/mic"));
  EXPECT_FALSE(fa_kws::vadStateMatchesAudioBinding(msg, "mic-a", ""));
}

TEST(VadStateIdentityContract, RejectsUnexpectedSourceOrStream)
{
  const auto msg = make_vad_state();

  EXPECT_FALSE(fa_kws::vadStateMatchesAudioBinding(msg, "mic-b", "audio/raw/mic"));
  EXPECT_FALSE(fa_kws::vadStateMatchesAudioBinding(msg, "mic-a", "audio/other"));
}

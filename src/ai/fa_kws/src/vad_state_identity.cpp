#include "fa_kws/vad_state_identity.hpp"

namespace fa_kws
{

bool vadStateMatchesAudioBinding(
  const fa_interfaces::msg::VadState &msg,
  const std::string &expected_source_id,
  const std::string &expected_stream_id)
{
  if (msg.source_id.empty() || msg.stream_id.empty()) {
    return false;
  }
  if (expected_source_id.empty() || expected_stream_id.empty()) {
    return false;
  }
  return msg.source_id == expected_source_id && msg.stream_id == expected_stream_id;
}

}  // namespace fa_kws

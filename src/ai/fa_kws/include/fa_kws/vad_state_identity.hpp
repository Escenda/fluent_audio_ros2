#ifndef FA_KWS__VAD_STATE_IDENTITY_HPP_
#define FA_KWS__VAD_STATE_IDENTITY_HPP_

#include <string>

#include "fa_interfaces/msg/vad_state.hpp"

namespace fa_kws
{

bool vadStateMatchesAudioBinding(
  const fa_interfaces::msg::VadState &msg,
  const std::string &expected_source_id,
  const std::string &expected_stream_id);

}  // namespace fa_kws

#endif  // FA_KWS__VAD_STATE_IDENTITY_HPP_

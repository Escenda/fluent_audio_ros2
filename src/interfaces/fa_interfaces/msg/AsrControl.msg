# AsrControl.msg
# Dialogue turn controllerからASR nodeへのstream lifecycle制御。

uint8 ACTION_START=1
uint8 ACTION_STOP=2
uint8 ACTION_CANCEL=3

# 送信時刻
builtin_interfaces/Time timestamp

# ACTION_* のいずれか
uint8 action

# 対象session/turn
string session_id
uint32 user_turn_id

# 制御理由。例: wake, td_end, vad_fallback, cancel
string reason

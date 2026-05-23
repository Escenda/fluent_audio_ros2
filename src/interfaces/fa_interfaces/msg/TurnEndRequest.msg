# TurnEndRequest.msg
# dialogue から turn detector へ送る「終了確認をしてよい」要求。

builtin_interfaces/Time timestamp
string session_id
uint32 user_turn_id
uint32 request_id
uint32 quiet_ms

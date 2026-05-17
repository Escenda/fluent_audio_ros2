# TurnEnd.msg
# turn_detector（Smart Turn v3）からの「発話終了確率」通知（ID相関用）

# タイムスタンプ（送信時刻）
builtin_interfaces/Time timestamp

# セッションID
string session_id

# 入力ターンID
uint32 user_turn_id

# 発話終了確率（0.0-1.0）
float32 probability

# threshold 以上なら true
bool is_end

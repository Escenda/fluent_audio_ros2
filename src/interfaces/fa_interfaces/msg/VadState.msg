# VadState.msg
# VAD（Voice Activity Detection）の状態

std_msgs/Header header

# 発話確率 (0.0 - 1.0)
float32 probability

# 発話中フラグ
bool is_speech

# 発話開始フラグ
bool start

# 発話終了フラグ
bool end


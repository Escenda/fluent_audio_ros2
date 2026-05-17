# WakeWordResult.msg
# ウェイクワード検出結果

std_msgs/Header header

# 検出キーワード
string keyword

# スコア（0.0 - 1.0 を想定）
float32 score

# 検出フラグ
bool detected


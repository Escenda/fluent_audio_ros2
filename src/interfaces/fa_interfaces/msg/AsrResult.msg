# AsrResult.msg
# ASRの最終結果/タイムアウト/エラー通知（ID相関用）

uint8 STATUS_FINAL=1
uint8 STATUS_TIMEOUT=2
uint8 STATUS_ERROR=3
uint8 STATUS_PARTIAL=4

# タイムスタンプ（送信時刻）
builtin_interfaces/Time timestamp

# セッションID
string session_id

# 入力ターンID（session内で単調増加）
uint32 user_turn_id

# 結果種別
uint8 status

# status が STATUS_TIMEOUT / STATUS_ERROR の場合の理由
string reason

# 認識テキスト（STATUS_PARTIAL / STATUS_FINAL の場合に使用）
string text

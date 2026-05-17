# TurnContext.msg
# セッション/入力ターンの「現在値」を配布するためのコンテキストメッセージ。
#
# - 会話オーケストレータが session_id / user_turn_id を発番する
# - ASR / turn_detector はこのメッセージを購読して、結果にIDを付与する

# タイムスタンプ（送信時刻）
builtin_interfaces/Time timestamp

# セッションID（未開始の場合は空）
string session_id

# 入力ターンID（session内で単調増加）
uint32 user_turn_id

# 有効フラグ（true: このturn_idが有効、false: 無効）
bool active

# PlaybackDone.msg
# audio/output/frame の1フレーム再生完了通知。
#
# - header: 元の AudioFrame.header をそのまま転送
# - request_id: 相関用ID（通常は header.frame_id のコピー）
# - epoch: AudioFrame.epoch のコピー（stop世代）

std_msgs/Header header
string request_id
uint32 epoch

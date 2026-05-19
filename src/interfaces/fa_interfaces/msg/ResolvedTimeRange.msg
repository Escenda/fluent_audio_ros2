# Timeline tool boundaryで解決済みの時間範囲。
# timestampの不確実性は隠さず、同期不能時はuncertainty_reasonで明示する。

string CLOCK_AGENT=agent
string CLOCK_ROBOT=robot
string CLOCK_MEDIA=media

int64 start_unix_ns
int64 end_unix_ns
string clock
uint64 uncertainty_ns
string uncertainty_reason

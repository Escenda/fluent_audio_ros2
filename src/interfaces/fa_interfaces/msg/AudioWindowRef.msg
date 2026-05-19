# rolling audio window内の切り出し範囲を参照する。
# window_epochはrolling buffer再生成や世代差を識別するためのwindow側identity。

string window_id
uint64 window_epoch
string source_id
string stream_id
fa_interfaces/ResolvedTimeRange time_range

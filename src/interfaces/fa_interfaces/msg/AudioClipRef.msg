# archive/export済み音声clipへの参照。
# success=trueのservice responseではclip_idとuriを空にしない。

string clip_id
string uri
string metadata_uri
string content_sha256
string metadata_sha256
string codec
string container
string payload_format
uint32 sample_rate
uint32 channels
uint64 duration_ns
fa_interfaces/ResolvedTimeRange time_range

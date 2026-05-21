# Current fa_asr lifecycle state.

uint8 STATE_WAITING=1
uint8 STATE_COLLECTING=2
uint8 STATE_QUEUED=3
uint8 STATE_TRANSCRIBING=4
uint8 STATE_COMPLETED=5
uint8 STATE_TIMEOUT=6
uint8 STATE_FAILED=7

builtin_interfaces/Time timestamp
uint8 state
string reason
string session_id
uint32 user_turn_id
string control_id
string source_id
string stream_id
bool context_active
bool window_open
uint32 queued_jobs
uint32 sample_count
int64 start_unix_ns
int64 end_unix_ns
string error_code

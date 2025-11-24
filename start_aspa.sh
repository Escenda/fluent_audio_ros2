#!/bin/bash
set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="$(cd "${SELF_DIR}/../.." && pwd)"
STATE_DIR="${HOME}/.aspa_runtime"
PID_DIR="${STATE_DIR}/pids"
LOG_DIR="${STATE_DIR}/logs"
mkdir -p "${PID_DIR}" "${LOG_DIR}"

SETUP_FILE="${WS_DIR}/install/setup.bash"
if [ ! -f "${SETUP_FILE}" ]; then
  echo "[start_aspa] ${SETUP_FILE} が見つかりません。先に 'colcon build' を実行してください。" >&2
  exit 1
fi
# shellcheck disable=SC1090
set +u
source "${SETUP_FILE}"
set -u

if ! command -v ros2 >/dev/null 2>&1; then
  echo "[start_aspa] ros2 コマンドが見つかりません。ROS 2 環境を確認してください。" >&2
  exit 1
fi

python3 - <<'PY'
import sys
try:
    import pyopenjtalk  # noqa: F401
except Exception as exc:  # pragma: no cover
    print(f"[start_aspa] pyopenjtalk を import できません: {exc}", file=sys.stderr)
    sys.exit(1)
PY

fv_tts_params="${WS_DIR}/install/fv_tts/share/fv_tts/config/default.yaml"
if [ ! -f "${fv_tts_params}" ]; then
  fv_tts_params="${SELF_DIR}/src/audio/fv_tts/config/default.yaml"
fi

aspa_i2c_config="${WS_DIR}/install/aspa_i2c/share/aspa_i2c/config/hub_0.yaml"
if [ ! -f "${aspa_i2c_config}" ]; then
  aspa_i2c_config="${WS_DIR}/src/aspa_i2c/config/hub_0.yaml"
fi

audio_output_params="${WS_DIR}/install/fv_audio_output/share/fv_audio_output/config/default.yaml"
if [ ! -f "${audio_output_params}" ]; then
  audio_output_params="${SELF_DIR}/src/audio/fv_audio_output/config/default.yaml"
fi

start_unit() {
  local name="$1"
  shift
  local pid_file="${PID_DIR}/${name}.pid"
  local log_file="${LOG_DIR}/${name}.log"

  if [ -f "${pid_file}" ]; then
    local existing_pid
    existing_pid="$(cat "${pid_file}")"
    if kill -0 "${existing_pid}" >/dev/null 2>&1; then
      echo "[start_aspa] ${name} は既に起動しています (PID=${existing_pid})"
      return
    else
      rm -f "${pid_file}"
    fi
  fi

  echo "[start_aspa] ${name} を起動します..."
  ("$@" >"${log_file}" 2>&1 & echo $! >"${pid_file}")
  sleep 0.5
  local started_pid
  started_pid="$(cat "${pid_file}")"
  if kill -0 "${started_pid}" >/dev/null 2>&1; then
    echo "[start_aspa] ${name} (PID=${started_pid}) を起動しました。ログ: ${log_file}"
  else
    echo "[start_aspa] ${name} の起動に失敗しました。ログを確認してください: ${log_file}" >&2
  fi
}

start_unit "fv_tts" ros2 run fv_tts fv_tts_node --ros-args --params-file "${fv_tts_params}"
start_unit "fv_audio_output" ros2 run fv_audio_output fv_audio_output_node --ros-args --params-file "${audio_output_params}"
start_unit "aspa_i2c_hub_0" ros2 launch aspa_i2c aspa_i2c_hub_launch.py node_name:=aspa_i2c_hub_0 config_file:="${aspa_i2c_config}"

say_startup_message() {
  local attempt
  for attempt in {1..10}; do
    if ros2 service list | grep -q "/fv_tts/speak"; then
      echo "[start_aspa] TTSサービストリガを送ります..."
      ros2 service call /fv_tts/speak fv_tts/srv/Speak "{text: '起動しました', voice_id: '', play: true, volume_db: 0.0, cache_key: ''}" >/dev/null 2>&1 && return
    fi
    sleep 1
  done
  echo "[start_aspa] TTSサービスが起動せず、音声再生をスキップしました。" >&2
}

say_startup_message

echo "[start_aspa] すべてのノードを起動しました。停止するには ./stop_aspa.sh を実行してください。"

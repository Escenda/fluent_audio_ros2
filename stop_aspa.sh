#!/bin/bash
set -euo pipefail

STATE_DIR="${HOME}/.aspa_runtime"
PID_DIR="${STATE_DIR}/pids"

stop_unit() {
  local name="$1"
  local pid_file="${PID_DIR}/${name}.pid"

  if [ ! -f "${pid_file}" ]; then
    echo "[stop_aspa] ${name} は起動記録がありません。"
    return
  fi

  local pid
  pid="$(cat "${pid_file}")"
  if kill -0 "${pid}" >/dev/null 2>&1; then
    echo "[stop_aspa] ${name} (PID=${pid}) を停止します..."
    kill "${pid}" >/dev/null 2>&1 || true
    wait "${pid}" 2>/dev/null || true
    echo "[stop_aspa] ${name} を停止しました。"
  else
    echo "[stop_aspa] ${name} (PID=${pid}) は既に終了していました。"
  fi
  rm -f "${pid_file}"
}

if [ ! -d "${PID_DIR}" ]; then
  echo "[stop_aspa] 停止するノードはありません。"
  exit 0
fi

stop_unit "aspa_i2c_hub_0"
stop_unit "fv_tts"

# 念のための後片付け
rm -f "${PID_DIR}"/*.pid 2>/dev/null || true

# run tree 空なら親ディレクトリも掃除
rmdir "${PID_DIR}" 2>/dev/null || true
rmdir "${STATE_DIR}" 2>/dev/null || true

echo "[stop_aspa] すべてのノードを停止しました。"

#!/usr/bin/env bash
set -Eeuo pipefail

# Interface: OUT_DIR API_PID ENGINE_PID DURATION_SEC ROUND TAG
out_dir=${1:?OUT_DIR required}
api_pid=${2:?API_PID required}
engine_pid=${3:?ENGINE_PID required}
delay_sec=${4:?DURATION_SEC required}
round=${5:?ROUND required}
tag=${6:?TAG required}

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

mkdir -p "$out_dir"
exit_code_file="$out_dir/exit_code"
rc=0
trap 'rc=$?; printf "%s\n" "$rc" > "$exit_code_file"' EXIT

cat > "$out_dir/meta.env" <<EOF
COLLECTOR=snapshot_after_delay
OUT_DIR=$out_dir
API_PID=$api_pid
ENGINE_PID=$engine_pid
DELAY_SEC=$delay_sec
ROUND=$round
TAG=$tag
STARTED_AT=$(date -Iseconds)
EOF

sleep "$delay_sec"
"$script_dir/collect_snapshot.sh" "$tag" "$api_pid" "$engine_pid" "$out_dir" || rc=$?
exit "$rc"

#!/usr/bin/env bash
set -Eeuo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
run_script="$script_dir/run_experiment.sh"
affinity_script="$script_dir/affinity.sh"

fail() {
    printf 'FAIL: %s\n' "$*" >&2
    exit 1
}

[[ -x "$affinity_script" ]] || fail "affinity.sh must be an executable standalone binding script"

grep -q 'taskset -pc' "$affinity_script" \
    || fail "affinity.sh must own the concrete CPU binding operations"

if grep -q 'taskset -pc' "$run_script"; then
    fail "run_experiment.sh must not contain concrete CPU binding operations"
fi

serve_line=$(grep -n 'vllm serve' "$run_script" | tail -n 1 | cut -d: -f1)
first_affinity_call_line=$(grep -n '"\$script_dir/affinity.sh"' "$run_script" | head -n 1 | cut -d: -f1)

[[ -n "$serve_line" && -n "$first_affinity_call_line" ]] \
    || fail "could not find vllm serve launch or affinity.sh call"
(( serve_line < first_affinity_call_line )) \
    || fail "run_experiment.sh must call affinity.sh after launching vllm serve"

printf 'PASS: CPU binding is isolated in affinity.sh and called after vllm serve launch\n'

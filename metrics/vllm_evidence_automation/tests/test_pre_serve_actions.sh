#!/usr/bin/env bash
set -Eeuo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
run_script="$script_dir/run_experiment.sh"

fail() {
    printf 'FAIL: %s\n' "$*" >&2
    exit 1
}

grep -q '^run_pre_serve_actions() {' "$run_script" \
    || fail "run_pre_serve_actions function is missing"

grep -q 'sync || true' "$run_script" \
    || fail "pre-serve actions must sync before dropping caches"
grep -q 'echo 3 > /proc/sys/vm/drop_caches || true' "$run_script" \
    || fail "pre-serve actions must drop PageCache, dentries, and inodes"
grep -q 'echo 1 > /proc/sys/vm/compact_memory || true' "$run_script" \
    || fail "pre-serve actions must compact physical memory"

pre_serve_line=$(grep -n 'run_pre_serve_actions' "$run_script" | tail -n 1 | cut -d: -f1)
serve_line=$(grep -n 'vllm serve' "$run_script" | tail -n 1 | cut -d: -f1)

[[ -n "$pre_serve_line" && -n "$serve_line" ]] \
    || fail "could not find pre-serve action call or vllm serve launch"
(( pre_serve_line < serve_line )) \
    || fail "pre-serve actions must run before vllm serve is launched"

printf 'PASS: pre-serve actions are explicit and run before vllm serve\n'

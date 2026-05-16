#!/usr/bin/env bash
set -Eeuo pipefail

exp_root=${1:?EXP_ROOT required}

printf 'summary file:\n'
cat "$exp_root/summary.tsv"

printf '\nfirst-round TPOT by run:\n'
awk -F '\t' 'NR>1 {printf "%s\t%s\t%s\n", $1, $2, $3}' "$exp_root/summary.tsv"

printf '\nhot threads from T4 and T6 snapshots:\n'
while IFS=$'\t' read -r run_id run_class tpot api_pid engine_pid run_dir; do
    [[ "$run_id" == "run" ]] && continue
    printf '\n=== %s class=%s tpot=%s ===\n' "$run_id" "$run_class" "$tpot"
    for tag in T4_after_final_bind T6_after_round1 T8_after_round2; do
        file="$run_dir/snapshots/${tag}.ENGINE.threads.tsv"
        [[ -f "$file" ]] || continue
        printf -- '-- %s ENGINE top pcpu --\n' "$tag"
        awk -F '\t' 'NR>1 {print $5 "\t" $1 "\t" $2 "\tallowed=" $3 "\tpsr=" $4}' "$file" | sort -nr | head -n 20
    done
done < "$exp_root/summary.tsv"

printf '\nNUMA summaries for first good and bad runs when available:\n'
for cls in good bad; do
    link=$(find "$exp_root/by_class/$cls" -mindepth 1 -maxdepth 1 -type l 2>/dev/null | sort | head -n 1 || true)
    [[ -n "$link" ]] || continue
    run_dir=$(readlink -f "$link")
    file="$run_dir/snapshots/T6_after_round1.engine.numa_maps.txt"
    [[ -f "$file" ]] || continue
    printf '\n=== %s %s ===\n' "$cls" "$(basename "$run_dir")"
    "$(dirname "$0")/summarize_numa_maps.sh" "$file"
done


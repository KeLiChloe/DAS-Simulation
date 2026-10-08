#!/bin/bash
# Shared helper: snapshot the calling run script into SAVE_DIR for later inspection.
# Source from a discrete run_*.sh after SAVE_DIR is set:
#   source "$(dirname "$0")/_snapshot_run_setup.sh"
#   snapshot_run_setup "$SAVE_DIR"

snapshot_run_setup() {
    local save_dir="$1"
    if [[ -z "$save_dir" ]]; then
        echo "snapshot_run_setup: SAVE_DIR argument required" >&2
        return 1
    fi
    mkdir -p "$save_dir"

    # Caller is one frame up from this sourced helper.
    local caller_dir caller_base caller
    caller_dir="$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd)"
    caller_base="$(basename "${BASH_SOURCE[1]}")"
    caller="${caller_dir}/${caller_base}"

    cp "$caller" "${save_dir}/run_script.sh"

    cat > "${save_dir}/README_setup.txt" <<EOF
Experiment setup snapshot
=========================
source_script : ${caller}
copied_as     : ${save_dir}/run_script.sh
saved_utc     : $(date -u +%Y-%m-%dT%H:%M:%SZ)
host          : $(hostname)

How to re-read parameters
-------------------------
1. Open run_script.sh in this folder (exact bash used for the sweep).
2. Each *.pkl stores main.py "exp_params" (K/d/noise ranges, algorithms, cv_folds, ...).

Re-run tip: bump SAVE_DIR (e.g. set_3) before launching so you do not overwrite.
EOF

    echo "📋 Setup snapshot → ${save_dir}/run_script.sh (+ README_setup.txt)"
}

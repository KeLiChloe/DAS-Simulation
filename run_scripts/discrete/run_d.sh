#!/bin/bash
# Discrete: vary covariate dimension d.
# Noise: exactly one of --X_noise_std_scale | --target_mahalanobis_sep.
# With --partial_x 1, --disturb_covariate_noise is unused (disturb_d=0).

# Prevent numpy/BLAS thread-pool fork deadlocks on macOS.
# Must be set before Python starts (setting inside Python is too late).
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# ===== 参数配置 =====
SAVE_DIR="exp_sept_2026/discrete/varying_d/set_2_policy_tree_hybrid"
mkdir -p "$SAVE_DIR"
# shellcheck source=./_snapshot_run_setup.sh
source "$(cd "$(dirname "$0")" && pwd)/_snapshot_run_setup.sh"
snapshot_run_setup "$SAVE_DIR"

ALGOS=(
    policy_tree
    dast mst
    kmeans-standard
    gmm-standard
    clr-standard
    t_learner s_learner x_learner dr_learner causal_forest
)

# ===== 实验循环 =====
# varying d
for D in $(seq 16 1 20); do
    D_TAG=$(printf "%02d" "$D")

    echo "🚀 Running experiment with d=${D} ..."

    OUTFILE="${SAVE_DIR}/exp_d_${D_TAG}.pkl"
    python3 main.py \
        --target_p_range 0.01 0.15 \
        --winner_p_range 0.15 0.3 \
        --beta_range -0.1 0.1 \
        --delta_range -0.1 0.1 \
        --DR_generation_method lightgbm \
        --kmeans_coef 0.1 \
        --x_mean_range -50 50 \
        --N_segment_size 100 \
        --implementation_scale 10 \
        --X_noise_std_scale 0.2 \
        --K 5 \
        --d "$D" \
        --partial_x 1 \
        --action_num 3 \
        --N_sims 100 \
        --disallowed_ball_radius 0.2 \
        --cv_folds 5 \
        --action_method diff_in_means \
        --save_file "$OUTFILE" \
        --sequence_seed 999 \
        --algorithms "${ALGOS[@]}"

    echo "✅ Finished d=${D}. Saved to $OUTFILE"
    echo "----------------------------------------"
done

echo "🎉 All experiments completed! Results saved in $SAVE_DIR/"

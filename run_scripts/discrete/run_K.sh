#!/bin/bash
# Discrete: vary K (cluster count).
# Noise: exactly one of --X_noise_std_scale | --target_mahalanobis_sep.
# With --partial_x 1, --disturb_covariate_noise is unused (disturb_d=0).

# Prevent numpy/BLAS thread-pool fork deadlocks on macOS.
# Must be set before Python starts (setting inside Python is too late).
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# ===== 参数配置 =====
SAVE_DIR="exp_sept_2026/discrete/varying_K/set2_policy_tree_hybrid"
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
# varying K
for K in $(seq 2 1 10); do
    K_TAG=$(printf "%02d" "$K")

    echo "🚀 Running experiment with K=${K} ..."

    OUTFILE="${SAVE_DIR}/exp_K_${K_TAG}_oracle.pkl"

    # Discrete Bernoulli: P(Y=1)=sigmoid(alpha + beta@x + tau[D] + delta[D]@x)
    python3 main.py \
        --target_p_range 0.05 0.2 \
        --winner_p_range 0.2 0.5 \
        --beta_range -0.05 0.05 \
        --delta_range -0.05 0.05 \
        --DR_generation_method lightgbm \
        --kmeans_coef 0.1 \
        --x_mean_range -200 200 \
        --N_segment_size 100 \
        --implementation_scale 10 \
        --X_noise_std_scale 0.2 \
        --K "$K" \
        --d 7 \
        --partial_x 1 \
        --action_num 3 \
        --N_sims 50 \
        --disallowed_ball_radius 0.2 \
        --cv_folds 5 \
        --action_method diff_in_means \
        --save_file "$OUTFILE" \
        --sequence_seed 888 \
        --algorithms "${ALGOS[@]}"

    echo "✅ Finished K=${K}. Saved to $OUTFILE"
    echo "----------------------------------------"
done

echo "🎉 All experiments completed! Results saved in $SAVE_DIR/"

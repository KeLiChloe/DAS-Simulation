#!/bin/bash
# Discrete: vary cluster overlap via target_mahalanobis_sep
#   (do NOT also pass --X_noise_std_scale).
# Smaller sep → more overlap. With --partial_x 1, disturb noise is unused.

# Prevent numpy/BLAS thread-pool fork deadlocks on macOS.
# Must be set before Python starts (setting inside Python is too late).
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# ===== 参数配置 =====
SAVE_DIR="exp_april_2026/discrete/varying_overlap_mahal/set_2"
mkdir -p "$SAVE_DIR"
# shellcheck source=./_snapshot_run_setup.sh
source "$(cd "$(dirname "$0")" && pwd)/_snapshot_run_setup.sh"
snapshot_run_setup "$SAVE_DIR"

ALGOS=(
    policy_tree
    dast mst
    kmeans-standard kmeans-da
    gmm-standard gmm-da
    clr-standard clr-da
    t_learner s_learner x_learner dr_learner causal_forest
)

# ===== 实验循环 =====
# Use integer loop to avoid shell floating-point precision issues.
for SEP_INT in $(seq 50 50 800); do
    MAHAL_SEP=$(awk "BEGIN {printf \"%.3f\", $SEP_INT/100}")
    SEP_TAG=$(printf "%03d" "$SEP_INT")

    echo "🚀 Running experiment with target_mahalanobis_sep=${MAHAL_SEP} ..."

    OUTFILE="${SAVE_DIR}/exp_Msep_${SEP_TAG}_oracle.pkl"

    # Discrete Bernoulli: P(Y=1)=sigmoid(alpha + beta@x + tau[D] + delta[D]@x)
    python3 main.py \
        --target_p_range 0.05 0.2 \
        --winner_p_range 0.2 0.5 \
        --beta_range -0.2 0.2 \
        --delta_range -0.05 0.05 \
        --disturb_covariate_noise 3 \
        --DR_generation_method lightgbm \
        --kmeans_coef 0.15 \
        --x_mean_range -200 200 \
        --N_segment_size 100 \
        --implementation_scale 10 \
        --target_mahalanobis_sep "$MAHAL_SEP" \
        --K 5 \
        --d 6 \
        --partial_x 1 \
        --action_num 3 \
        --N_sims 100 \
        --disallowed_ball_radius 0.2 \
        --cv_folds 5 \
        --action_method diff_in_means \
        --save_file "$OUTFILE" \
        --sequence_seed 108 \
        --n_workers 8 \
        --algorithms "${ALGOS[@]}"

    echo "✅ Finished target_mahalanobis_sep=${MAHAL_SEP}. Saved to $OUTFILE"
    echo "----------------------------------------"
done

echo "🎉 All experiments completed! Results saved in $SAVE_DIR/"

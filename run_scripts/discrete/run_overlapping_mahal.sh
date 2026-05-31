#!/bin/bash

# Prevent numpy/BLAS thread-pool fork deadlocks on macOS.
# Must be set before Python starts (setting inside Python is too late).
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# ===== 参数配置 =====
SAVE_DIR="exp_april_2026/discrete/varying_overlap_mahal/set_2"
mkdir -p "$SAVE_DIR"

# ===== 实验循环 =====
# varying target_mahalanobis_sep: smaller values create more overlapping clusters
# Use integer loop to avoid shell floating-point precision issues.
for SEP_INT in $(seq 50 50 800); do
    MAHAL_SEP=$(awk "BEGIN {printf \"%.3f\", $SEP_INT/100}")
    SEP_TAG=$(printf "%03d" "$SEP_INT")

    echo "🚀 Running experiment with target_mahalanobis_sep=${MAHAL_SEP} ..."

    OUTFILE="${SAVE_DIR}/exp_Msep_${SEP_TAG}_oracle.pkl"

    # Discrete logistic model: p = sigmoid(alpha + beta@x + tau[D])
    python main.py \
        --outcome_type discrete \
        --target_p_range 0.05 0.2 \
        --winner_p_range 0.2 0.5\
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
        --save_file "$OUTFILE" \
        --sequence_seed 108 \
        --n_workers 8 \
        --algorithms policy_tree dast mst kmeans-standard gmm-standard clr-standard t_learner s_learner x_learner dr_learner causal_forest

    echo "✅ Finished target_mahalanobis_sep=${MAHAL_SEP}. Saved to $OUTFILE"
    echo "----------------------------------------"
done

echo "🎉 All experiments completed! Results saved in $SAVE_DIR/"

#!/bin/bash

# Prevent numpy/BLAS thread-pool fork deadlocks on macOS.
# Must be set before Python starts (setting inside Python is too late).
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# ===== 参数配置 =====
SAVE_DIR="exp_april_2026/discrete/varying_overlapping/set_4"
mkdir -p "$SAVE_DIR"

# ===== 实验循环 =====
# varying X_noise_std_scale: larger values create more overlapping clusters
# Use integer loop to avoid shell floating-point precision issues.
for X_INT in $(seq 50 25 500); do
    X_NOISE=$(awk "BEGIN {printf \"%.3f\", $X_INT/1000}")
    X_TAG=$(printf "%03d" "$X_INT")

    echo "🚀 Running experiment with X_noise_std_scale=${X_NOISE} ..."

    OUTFILE="${SAVE_DIR}/exp_Xnoise_${X_TAG}_oracle.pkl"

    # Discrete logistic model: p = sigmoid(alpha + beta@x + tau[D])
    python main.py \
        --outcome_type discrete \
        --target_p_range 0.01 0.10 \
        --winner_p_range 0.10 0.3 \
        --beta_range -0.3 0.3 \
        --delta_range -0.05 0.05 \
        --disturb_covariate_noise 3 \
        --DR_generation_method lightgbm \
        --kmeans_coef 0.15 \
        --x_mean_range -50 50 \
        --N_segment_size 20 \
        --implementation_scale 10 \
        --X_noise_std_scale "$X_NOISE" \
        --K 2 \
        --d 1 \
        --partial_x 1 \
        --action_num 3 \
        --N_sims 100 \
        --disallowed_ball_radius 0.2 \
        --save_file "$OUTFILE" \
        --sequence_seed 108 \
        --n_workers 6 \
        --algorithms dast mst kmeans-standard gmm-standard clr-standard t_learner s_learner x_learner dr_learner causal_forest

    echo "✅ Finished X_noise_std_scale=${X_NOISE}. Saved to $OUTFILE"
    echo "----------------------------------------"
done

echo "🎉 All experiments completed! Results saved in $SAVE_DIR/"

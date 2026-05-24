#!/bin/bash

# Prevent numpy/BLAS thread-pool fork deadlocks on macOS.
# Must be set before Python starts (setting inside Python is too late).
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# ===== 参数配置 =====
SAVE_DIR="exp_april_2026/discrete/varying_d/set_3"
mkdir -p "$SAVE_DIR"

# ===== 实验循环 =====
# varying d: same range as continuous counterpart
for D in $(seq 1 1 20); do
    D_TAG=$(printf "%02d" "$D")

    echo "🚀 Running experiment with d=${D} ..."

    OUTFILE="${SAVE_DIR}/exp_d_${D_TAG}_oracle.pkl"

    # Discrete logistic model: p = sigmoid(alpha + beta@x + tau[D])
    python main.py \
        --outcome_type discrete \
        --target_p_range 0.01 0.15 \
        --winner_p_range 0.15 0.3 \
        --beta_range -0.1 0.1 \
        --delta_range -0.1 0.1 \
        --disturb_covariate_noise 3 \
        --DR_generation_method lightgbm \
        --kmeans_coef 0.15 \
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
        --save_file "$OUTFILE" \
        --sequence_seed 999 \
        --n_workers 6 \
        --algorithms dast mst kmeans-standard gmm-standard clr-standard t_learner s_learner x_learner dr_learner causal_forest

    echo "✅ Finished d=${D}. Saved to $OUTFILE"
    echo "----------------------------------------"
done

echo "🎉 All experiments completed! Results saved in $SAVE_DIR/"

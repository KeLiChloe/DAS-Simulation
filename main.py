from ground_truth import PopulationSimulator
from plot import plot_ground_truth, plot_segmentation, plot_bernoulli_prob_histogram
from gmm import GMM_segment_and_estimate
from oracle import structure_oracle, policy_oracle, oracle_profit_on_customers
import pandas as pd
import numpy as np
from dast import DAST_segment_and_estimate
from mst import MST_segment_and_estimate
from kmeans import KMeans_segment_and_estimate
from clr import CLR_segment_and_estimate
from meta_learners import T_learner, S_learner, X_learner, DR_learner
from causal_forest import causal_forest_predict
from utils import assign_new_customers_to_segments, pick_M_for_algo, parse_args
from cv_utils import CV_M_ALGOS, make_m_selection_folds, attach_fold
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
from tqdm import tqdm
import random
import pickle
import time
import json
import os
import sys

# ── Constants ─────────────────────────────────────────────────────────────────
_HTE_METHODS    = frozenset(["t_learner", "x_learner", "dr_learner", "s_learner", "causal_forest", "policy_tree"])
_FULL_SPLIT_ALGOS = frozenset(["clr-standard", "kmeans-standard", "gmm-standard"])


# ── Helpers ───────────────────────────────────────────────────────────────────

def _restore_pop_split(pop, sp):
    """Re-attach a cached train/val split to *pop* without refitting gamma."""
    pop.train_customers = sp['train_customers']
    pop.val_customers   = sp['val_customers']
    pop.train_indices   = sp['train_indices']
    pop.val_indices     = sp['val_indices']
    pop.gamma_train     = sp['gamma_train']
    pop.gamma_val       = sp['gamma_val']


def _build_split(pop, frac, d):
    """Call split_pilot once and cache all derived arrays."""
    pop.split_pilot_customers_into_train_and_validate(train_frac=frac)
    return {
        'train_customers': list(pop.train_customers),
        'val_customers':   list(pop.val_customers),
        'train_indices':   pop.train_indices.copy(),
        'val_indices':     pop.val_indices.copy(),
        'gamma_train':     pop.gamma_train,
        'gamma_val':       pop.gamma_val,
        'x_mat_tr':  np.array([c.x   for c in pop.train_customers]),
        'D_vec_tr':  np.array([c.D_i for c in pop.train_customers]),
        'y_vec_tr':  np.array([c.y   for c in pop.train_customers]),
        'x_mat_val': np.array([c.x   for c in pop.val_customers])   if pop.val_customers else np.zeros((0, d)),
        'D_vec_val': np.array([c.D_i for c in pop.val_customers])   if pop.val_customers else np.zeros(0),
        'y_vec_val': np.array([c.y   for c in pop.val_customers])   if pop.val_customers else np.zeros(0),
    }


def _fit_val_score_for_algo(
    pop, algo, M, x_mat, D_vec, y_vec, *,
    include_interactions, seed, action_method, args,
):
    """Fit one candidate M on the current pop train/val split; return held-out DR (or BIC/sil)."""
    if algo == "gmm-standard":
        score, _ = GMM_segment_and_estimate(
            pop, M, x_mat, D_vec, y_vec, algo,
            include_interactions, random_state=seed, action_method=action_method)
        return score
    if algo == "gmm-da":
        score, _ = GMM_segment_and_estimate(
            pop, M, x_mat, D_vec, y_vec, algo,
            include_interactions, random_state=seed, action_method=action_method)
        return score
    if algo == "kmeans-standard":
        score, _ = KMeans_segment_and_estimate(
            pop, M, x_mat, D_vec, y_vec, algo,
            include_interactions, random_state=seed, action_method=action_method)
        return score
    if algo == "kmeans-da":
        score, _ = KMeans_segment_and_estimate(
            pop, M, x_mat, D_vec, y_vec, algo,
            include_interactions, random_state=seed, action_method=action_method)
        return score
    if algo == "clr-standard":
        score, _ = CLR_segment_and_estimate(
            pop, M, x_mat, D_vec, y_vec,
            kmeans_coef=args.kmeans_coef, num_tries=3,
            algo=algo, include_interactions=include_interactions,
            random_state=seed, action_method=action_method)
        return score
    if algo == "clr-da":
        score, _ = CLR_segment_and_estimate(
            pop, M, x_mat, D_vec, y_vec,
            kmeans_coef=args.kmeans_coef, num_tries=3,
            algo=algo, include_interactions=include_interactions,
            random_state=seed, action_method=action_method)
        return score
    if algo == "dast":
        _, score, _ = DAST_segment_and_estimate(
            pop, M, min_leaf_size=2, algo=algo,
            use_hybrid_method=args.use_hybrid_method,
            action_method=action_method,
            include_interactions=include_interactions)
        return score
    if algo == "mst":
        depth_mst = 1 if M <= 2 else (2 if M <= 4 else (3 if M <= 8 else 4))
        _, score, _ = MST_segment_and_estimate(
            pop, M, max_depth=depth_mst, min_leaf_size=2,
            epsilon=1e-2, algo=algo,
            include_interactions=include_interactions,
            action_method=action_method)
        return score
    raise ValueError(f"Unknown segmentation algorithm: {algo}")


# ── Per-simulation worker ─────────────────────────────────────────────────────

def _run_one_sim(packed_args):
    """
    Run one independent simulation.  Returns a result dict or None on failure.

    When quiet=True (parallel mode) all stdout/stderr from this worker process
    is suppressed so that concurrent workers do not interleave their output.
    A compact per-simulation summary is returned in the result dict and printed
    by the main process instead.

    Optimisations applied here:
      1. Full-pilot and (for mst) 80/20 splits are built once and re-attached.
      2. dast / *-da select M via K-fold CV (Gamma refit per fold on fold-train).
      3. CLR M-sweep uses num_tries=3; the final retrain uses num_tries=8.
    """
    args, param_range, seed, sim_idx, quiet = packed_args

    # Silence worker stdout/stderr so concurrent processes don't interleave output.
    # With spawn, workers have their own stdout; with fork they share the parent's.
    if quiet:
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    np.random.seed(seed)

    # Discrete-only codebase: estimation design matrix never includes x*D
    # (DGP may still use delta_range for treatment–covariate interactions).
    include_interactions = False
    action_method = args.action_method
    cv_folds = int(getattr(args, 'cv_folds', 5))
    N_pilot  = args.N_segment_size * args.K
    N_impl   = int(N_pilot * args.implementation_scale)
    M_range  = list(range(max(2, args.K - 3), args.K + 4))

    pop = PopulationSimulator(
        N_pilot, N_impl,
        args.d, args.K,
        args.disturb_covariate_noise,
        param_range,
        args.DR_generation_method,
        args.partial_x,
        action_num=getattr(args, 'action_num', 2),
        X_noise_std_scale=args.X_noise_std_scale,
        target_mahalanobis_sep=getattr(args, 'target_mahalanobis_sep', None),
        disallowed_ball_radius=getattr(args, 'disallowed_ball_radius', None),
    )

    if args.plot:
        plot_ground_truth(
            pop.to_dataframe(),
            run_idx=sim_idx,
        )
        plot_bernoulli_prob_histogram(
            pop.implement_customers,
            action_num=getattr(args, 'action_num'),
            run_idx=sim_idx,
        )

    # ── Pre-compute true segment ids for all pilot customers (never changes) ──
    all_true_seg_ids = np.array([c.true_segment.segment_id for c in pop.pilot_customers])

    # ── Cached splits: 80/20 for mst; 100% for standard + final retrain ───────
    split07 = _build_split(pop, 0.8, args.d)
    split10 = _build_split(pop, 1.0, args.d)
    cv_folds_idx = make_m_selection_folds(N_pilot, cv_folds, seed)

    # ── Oracle profit (algorithm-independent, computed once per sim) ──────────
    oracle_profit_impl = oracle_profit_on_customers(
        pop.implement_customers, signal_d=pop.signal_d)

    algo_result_dict = {}

    try:
        for algo in args.algorithms:
            is_meta = algo in _HTE_METHODS
            use_cv = (not is_meta) and (algo in CV_M_ALGOS)

            # Select cached split for non-CV paths ─────────────────────────────
            if is_meta or algo in _FULL_SPLIT_ALGOS:
                sp = split10
            else:
                sp = split07  # mst (and any leftover holdout-val algo)
            _restore_pop_split(pop, sp)

            x_mat_tr  = sp['x_mat_tr']
            D_vec_tr  = sp['D_vec_tr']
            y_vec_tr  = sp['y_vec_tr']

            true_seg_ids_tr = all_true_seg_ids[sp['train_indices']]

            # ── M sweep (segmentation methods only) ─────────────────────────────
            results_M = []

            if not is_meta:
                for M in M_range:
                    if use_cv:
                        fold_scores = []
                        for train_idx, val_idx in cv_folds_idx:
                            fold_sp = attach_fold(pop, train_idx, val_idx, args.d)
                            score = _fit_val_score_for_algo(
                                pop, algo, M,
                                fold_sp['x_mat_tr'], fold_sp['D_vec_tr'], fold_sp['y_vec_tr'],
                                include_interactions=include_interactions,
                                seed=seed, action_method=action_method, args=args,
                            )
                            if score is None:
                                continue
                            fold_scores.append(float(score))
                        if not fold_scores:
                            raise RuntimeError(
                                f"[sim {sim_idx}] {algo} M={M}: all CV folds failed to produce a score."
                            )
                        cv_score = float(np.mean(fold_scores))
                        results_M.append({
                            "M": M,
                            f"{algo}_val": cv_score,
                            "ARI": None,
                            "NMI": None,
                            "regret": None,
                            "mistreatment_rate": None,
                            "manager_profit": None,
                        })
                    else:
                        score = _fit_val_score_for_algo(
                            pop, algo, M, x_mat_tr, D_vec_tr, y_vec_tr,
                            include_interactions=include_interactions,
                            seed=seed, action_method=action_method, args=args,
                        )
                        est_seg_ids_tr = np.array(
                            [c.est_segment[algo].segment_id for c in pop.train_customers])
                        S = structure_oracle(true_seg_ids_tr, est_seg_ids_tr)
                        P = policy_oracle(pop.pilot_customers, algo=algo, signal_d=pop.signal_d)
                        results_M.append({
                            "M": M,
                            f"{algo}_val": score,
                            "ARI": S["ARI"],
                            "NMI": S["NMI"],
                            "regret": P["regret"],
                            "mistreatment_rate": P["mistreatment_rate"],
                            "manager_profit": P["manager_profit"],
                        })

            df_M = pd.DataFrame(results_M)

            # ── Pick optimal M ────────────────────────────────────────────────
            if is_meta:
                oracle_picked_M = {
                    'Oracle_ARI': 0, 'Oracle_NMI': 0,
                    'Oracle_Regret': 0, 'Oracle_Mistreat': 0,
                }
            else:
                if len(df_M) == 0:
                    print(f"[sim {sim_idx}] WARNING: No valid M results for {algo}, skipping.",
                          file=sys.__stderr__)
                    continue
                if use_cv:
                    # Structure/policy oracles filled after full-pilot retrain
                    oracle_picked_M = {
                        'Oracle_ARI': None, 'Oracle_NMI': None,
                        'Oracle_Regret': None, 'Oracle_Mistreat': None,
                    }
                else:
                    oracle_picked_M = {
                        'Oracle_ARI':      df_M.at[df_M['ARI'].idxmax(),              'M'],
                        'Oracle_NMI':      df_M.at[df_M['NMI'].idxmax(),              'M'],
                        'Oracle_Regret':   df_M.at[df_M['regret'].idxmin(),           'M'],
                        'Oracle_Mistreat': df_M.at[df_M['mistreatment_rate'].idxmin(),'M'],
                    }

            algo_picked_M = pick_M_for_algo(algo, df_M)
            picked_M      = {**oracle_picked_M, **algo_picked_M}

            if is_meta:
                row        = None
                retrain_M  = None
            else:
                retrain_M  = picked_M[f'{algo}_picked_M']
                row        = df_M.loc[df_M['M'] == retrain_M].iloc[0]

            algo_result_dict[algo] = {
                "picked_M":                   picked_M if not is_meta else "Not applicable",
                "profit_at_manager_picked_M": row['manager_profit']    if (row is not None and not use_cv) else None,
                "ARI":                        row['ARI']               if (row is not None and not use_cv) else None,
                "NMI":                        row['NMI']               if (row is not None and not use_cv) else None,
                "regret":                     row['regret']            if (row is not None and not use_cv) else None,
                "mistreatment_rate":          row['mistreatment_rate'] if (row is not None and not use_cv) else None,
            }

            # ── Final fit on full pilot data ──────────────────────────────────
            _restore_pop_split(pop, split10)
            x_mat_tr = split10['x_mat_tr']
            D_vec_tr = split10['D_vec_tr']
            y_vec_tr = split10['y_vec_tr']

            meta_labels = act_id = None
            plot_tree = None

            if is_meta:
                if algo == "t_learner":
                    meta_labels, act_id = T_learner(
                        pop.implement_customers, x_mat_tr, D_vec_tr, y_vec_tr)
                elif algo == "s_learner":
                    meta_labels, act_id = S_learner(
                        pop.implement_customers, x_mat_tr, D_vec_tr, y_vec_tr)
                elif algo == "x_learner":
                    meta_labels, act_id = X_learner(
                        pop.implement_customers, x_mat_tr, D_vec_tr, y_vec_tr)
                elif algo == "causal_forest":
                    meta_labels, act_id = causal_forest_predict(
                        pop.implement_customers, x_mat_tr, D_vec_tr, y_vec_tr)
                elif algo == "policy_tree":
                    from policy_tree import policy_tree_predict
                    meta_labels, act_id = policy_tree_predict(
                        pop.implement_customers, x_mat_tr, D_vec_tr, y_vec_tr)
                elif algo == "dr_learner":
                    meta_labels, act_id = DR_learner(
                        pop.implement_customers, x_mat_tr, D_vec_tr, y_vec_tr)
                else:
                    raise ValueError(f"Unknown HTE algorithm: {algo}")

            elif algo == "gmm-standard":
                _, gmm_model = GMM_segment_and_estimate(
                    pop, retrain_M, x_mat_tr, D_vec_tr, y_vec_tr, algo,
                    include_interactions, random_state=seed,
                    action_method=action_method)
                assign_new_customers_to_segments(pop, pop.implement_customers, gmm_model, algo)

            elif algo == "gmm-da":
                _, gmm_model = GMM_segment_and_estimate(
                    pop, retrain_M, x_mat_tr, D_vec_tr, y_vec_tr, algo,
                    include_interactions, random_state=seed,
                    action_method=action_method)
                assign_new_customers_to_segments(pop, pop.implement_customers, gmm_model, algo)

            elif algo == "dast":
                opt_tree, _, seg_dict = DAST_segment_and_estimate(
                    pop, retrain_M, min_leaf_size=2, algo=algo,
                    use_hybrid_method=args.use_hybrid_method,
                    action_method=action_method,
                    include_interactions=include_interactions)
                opt_tree.predict_segment(pop.implement_customers, seg_dict)
                plot_tree = opt_tree

            elif algo == "mst":
                d_mst = 1 if retrain_M <= 2 else (2 if retrain_M <= 4 else (3 if retrain_M <= 8 else 4))
                opt_tree, _, seg_dict = MST_segment_and_estimate(
                    pop, retrain_M, max_depth=d_mst, min_leaf_size=2,
                    epsilon=1e-2, algo=algo, include_interactions=include_interactions,
                    action_method=action_method)
                opt_tree.predict_segment(pop.implement_customers, seg_dict)
                plot_tree = opt_tree

            elif algo == "kmeans-standard":
                _, km_model = KMeans_segment_and_estimate(
                    pop, retrain_M, x_mat_tr, D_vec_tr, y_vec_tr, algo,
                    include_interactions, random_state=seed,
                    action_method=action_method)
                assign_new_customers_to_segments(pop, pop.implement_customers, km_model, algo)

            elif algo == "kmeans-da":
                _, km_model = KMeans_segment_and_estimate(
                    pop, retrain_M, x_mat_tr, D_vec_tr, y_vec_tr, algo,
                    include_interactions, random_state=seed,
                    action_method=action_method)
                assign_new_customers_to_segments(pop, pop.implement_customers, km_model, algo)

            elif algo == "clr-standard":
                _, CLR = CLR_segment_and_estimate(
                    pop, retrain_M, x_mat_tr, D_vec_tr, y_vec_tr,
                    args.kmeans_coef, num_tries=8, algo=algo,
                    include_interactions=include_interactions, random_state=seed,
                    action_method=action_method)
                assign_new_customers_to_segments(pop, pop.implement_customers, CLR, algo)

            elif algo == "clr-da":
                _, CLR = CLR_segment_and_estimate(
                    pop, retrain_M, x_mat_tr, D_vec_tr, y_vec_tr,
                    args.kmeans_coef, num_tries=8, algo=algo,
                    include_interactions=include_interactions, random_state=seed,
                    action_method=action_method)
                assign_new_customers_to_segments(pop, pop.implement_customers, CLR, algo)

            else:
                raise ValueError(f"Unknown algorithm: {algo}")

            # CV algos: fill structure/policy metrics from full-pilot retrain
            if use_cv:
                est_seg_ids_full = np.array(
                    [c.est_segment[algo].segment_id for c in pop.train_customers])
                S = structure_oracle(all_true_seg_ids[split10['train_indices']], est_seg_ids_full)
                P = policy_oracle(pop.pilot_customers, algo=algo, signal_d=pop.signal_d)
                algo_result_dict[algo]["ARI"] = S["ARI"]
                algo_result_dict[algo]["NMI"] = S["NMI"]
                algo_result_dict[algo]["regret"] = P["regret"]
                algo_result_dict[algo]["mistreatment_rate"] = P["mistreatment_rate"]
                algo_result_dict[algo]["profit_at_manager_picked_M"] = P["manager_profit"]

            if args.plot and not is_meta:
                labels_plot = np.array([c.est_segment[algo].segment_id for c in pop.train_customers])
                plot_segmentation(
                    labels_plot, x_mat_tr, y_vec_tr, D_vec_tr,
                    algo, M=retrain_M, tree=plot_tree, run_idx=sim_idx,
                )

            # ── Evaluate implementation outcome ───────────────────────────────
            impl_outcome = 0.0
            for i, cust in enumerate(pop.implement_customers):
                if is_meta:
                    impl_outcome += cust.evaluate_profits(algo, act_id[meta_labels[i]])
                else:
                    impl_outcome += cust.evaluate_profits(algo)

            algo_result_dict[algo]['implementation_profits'] = impl_outcome

    except Exception:
        import traceback
        # Always print errors to the real stderr so they are visible
        if quiet:
            sys.stderr = sys.__stderr__
        traceback.print_exc()
        return None

    # Build a compact per-sim summary string for the main process to print
    impl_lines = []
    for algo in args.algorithms:
        if algo in algo_result_dict:
            profit = algo_result_dict[algo].get('implementation_profits')
            m_val  = algo_result_dict[algo].get('picked_M', {})
            if isinstance(m_val, dict):
                m_val = m_val.get(f'{algo}_picked_M', 'N/A')
            impl_lines.append(f"  {algo}: profit={profit:.2f}, M={m_val}" if profit is not None else f"  {algo}: N/A")
    summary = (
        f"[sim {sim_idx}] seed={seed}  oracle={oracle_profit_impl:.2f}\n"
        + "\n".join(impl_lines)
    )

    covariate_overlap = {
        **getattr(pop, "covariate_distance_summary", {}),
        "signal_covariate_noise": getattr(pop, "signal_covariate_noise", None),
        "target_mahalanobis_sep": getattr(pop, "target_mahalanobis_sep", None),
        "realized_median_nn_mahalanobis_sep": getattr(pop, "realized_median_nn_mahalanobis_sep", None),
        "realized_min_nn_mahalanobis_sep": getattr(pop, "realized_min_nn_mahalanobis_sep", None),
        "realized_median_nn_bayes_error": getattr(pop, "realized_median_nn_bayes_error", None),
        "realized_min_nn_bayes_error": getattr(pop, "realized_min_nn_bayes_error", None),
    }

    return {
        'seed':                seed,
        'sim_idx':             sim_idx,
        'oracle_profits_impl': oracle_profit_impl,
        'covariate_overlap':   covariate_overlap,
        'algo_result_dict':    algo_result_dict,
        'summary':             summary,
    }


# ── Main orchestrator ─────────────────────────────────────────────────────────

def main(args, param_range):
    '''
    For each simulation: generate a random population, run all algorithms,
    pick M via validation / CV, assign implementation customers, evaluate profits.

    Simulations are run sequentially in one process (no multiprocessing).
    '''
    include_interactions = False
    cv_folds = int(getattr(args, 'cv_folds', 5))
    print(f"Outcome type: discrete (Bernoulli), Include interactions: {include_interactions}")
    print(f"CV folds for dast/*-da M selection: {cv_folds}"
          + (" (single 80/20 holdout)" if cv_folds == 1 else ""))

    if args.sequence_seed is not None:
        random.seed(args.sequence_seed)
        seq_seed = args.sequence_seed
    else:
        seq_seed = random.randint(0, 100000)
        random.seed(seq_seed)
    print(f"Using fixed sequence seed: {seq_seed}")

    seeds = [random.randint(0, 100000) for _ in range(args.N_sims)]

    exp_result_dict = {
        "exp_params": {
            "sequence_seed":           seq_seed,
            "action_num":              getattr(args, 'action_num', 2),
            "K":                       getattr(args, "K", None),
            "d":                       getattr(args, "d", None),
            "partial_x":               getattr(args, 'partial_x', None),
            "X_noise_std_scale":       getattr(args, 'X_noise_std_scale', None),
            "target_mahalanobis_sep":  getattr(args, 'target_mahalanobis_sep', None),
            "disturb_covariate_noise": getattr(args, 'disturb_covariate_noise', None),
            "disallowed_ball_radius":  getattr(args, 'disallowed_ball_radius', None),
            "param_range":             param_range,
            "N_segment_size":          getattr(args, 'N_segment_size', None),
            "DR_generation_method":    getattr(args, 'DR_generation_method', None),
            "kmeans_coef":             getattr(args, 'kmeans_coef', None),
            "N_total_pilot_customers": args.N_segment_size * args.K,
            "implementation_scale":    getattr(args, 'implementation_scale', None),
            "outcome_type":            "discrete",
            "cv_folds":                cv_folds,
            "action_method":           getattr(args, 'action_method', None),
            "use_hybrid_method":       bool(getattr(args, 'use_hybrid_method', True)),
            "N_sims":                  int(args.N_sims),
            "n_workers":               1,
            "algorithms":              list(args.algorithms),
            "save_file":               getattr(args, 'save_file', None),
        },
        "seed": [],
        "oracle_profits_impl": [],
        "covariate_overlap": [],
        **{algo: [] for algo in args.algorithms},
    }

    if getattr(args, 'n_workers', None) not in (None, 1):
        print(f"Note: --n_workers={args.n_workers} ignored; simulations always run sequentially.")
    print(f"Running {args.N_sims} simulations sequentially.")

    # quiet=False keeps per-sim logs visible (no worker processes to interleave).
    packed = [(args, param_range, seed, i, False) for i, seed in enumerate(seeds)]

    start_time = time.time()
    for res in tqdm(map(_run_one_sim, packed), total=args.N_sims, desc="Simulations"):
        if res is None:
            continue

        tqdm.write(res['summary'])

        exp_result_dict['seed'].append(res['seed'])
        exp_result_dict['oracle_profits_impl'].append(res['oracle_profits_impl'])
        exp_result_dict['covariate_overlap'].append(res['covariate_overlap'])
        for algo in args.algorithms:
            if algo in res['algo_result_dict']:
                exp_result_dict[algo].append(res['algo_result_dict'][algo])

        if args.save_file is not None:
            with open(args.save_file, "wb") as f:
                tqdm.write(f"Checkpoint saved → {args.save_file}")
                pickle.dump(exp_result_dict, f)

    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    args = parse_args()

    print("==== Final Experiment Configuration ====")
    print(json.dumps(vars(args), indent=4))

    def _is_set(name):
        return getattr(args, name, None) is not None

    for r in ['beta_range', 'x_mean_range', 'target_p_range']:
        if not _is_set(r):
            raise ValueError(f"'--{r}' is required.")

    if _is_set('target_mahalanobis_sep') == _is_set('X_noise_std_scale'):
        raise ValueError("Set exactly one of '--target_mahalanobis_sep' or legacy '--X_noise_std_scale'.")

    if getattr(args, 'cv_folds', 5) < 1:
        raise ValueError(f"'--cv_folds' must be >= 1, got {args.cv_folds}.")

    lo, hi = args.target_p_range
    if not (0 < lo < hi < 1):
        raise ValueError(f"--target_p_range must satisfy 0 < lo < hi < 1, got {lo} {hi}.")
    if _is_set('winner_p_range'):
        wlo, whi = args.winner_p_range
        if not (0 < wlo < whi < 1):
            raise ValueError(f"--winner_p_range must satisfy 0 < lo < hi < 1, got {wlo} {whi}.")
        if wlo < hi:
            print(f"Warning: winner_p_range [{wlo},{whi}] overlaps target_p_range [{lo},{hi}]. "
                  f"Consider setting winner_p_range > target_p_range for a clear gap.")
    param_range = {
        "alpha":    None,
        "beta":     tuple(args.beta_range),
        "tau":      None,
        "delta":    tuple(args.delta_range) if _is_set('delta_range') else None,
        "x_mean":   tuple(args.x_mean_range),
        "target_p": tuple(args.target_p_range),
        "winner_p": tuple(args.winner_p_range) if _is_set('winner_p_range') else None,
    }

    main(args, param_range)

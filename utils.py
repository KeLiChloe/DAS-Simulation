
import numpy as np
from action_estimation import (
    ACTION_METHODS,
    estimate_segment_parameters,
    recommend_segment_action,
)
from ground_truth import PopulationSimulator
import pandas as pd
import plotly.graph_objects as go
import argparse

def build_design_matrix(x_array, D_array, include_interactions, action_num=None):
    """
    Design matrix for CLR / MST Bernoulli fits.

    Treatment is one-hot with action 0 as reference (same as action_estimation
    logistic), so multi-arm D is not treated as a numeric scalar.

    Columns:
      [intercept, x, 1{D=1}, ..., 1{D=K-1}]
      + [x * 1{D=a} for a=1..K-1] if include_interactions
    """
    x_array = np.asarray(x_array, dtype=float)
    D = np.ravel(np.asarray(D_array)).astype(int)
    if x_array.ndim == 1:
        x_array = x_array.reshape(-1, 1)
    N, d = x_array.shape

    if action_num is None:
        K = int(max(D.max() + 1, 2)) if N > 0 else 2
    else:
        K = int(action_num)
        if K < 1:
            raise ValueError(f"action_num must be >= 1, got {K}")

    intercept = np.ones((N, 1))
    blocks = [intercept, x_array]

    dummies = []
    for a in range(1, K):
        dummies.append((D == a).astype(float).reshape(-1, 1))
    if dummies:
        D_oh = np.hstack(dummies)
        blocks.append(D_oh)
        if include_interactions:
            # x_j * 1{D=a} for each non-reference action
            blocks.append(x_array[:, :, None] * D_oh[:, None, :])
            # reshape last block: (N, d, K-1) -> (N, d*(K-1))
            blocks[-1] = blocks[-1].reshape(N, d * (K - 1))

    return np.hstack(blocks)


def assign_trained_customers_to_segments(pop: PopulationSimulator, segment_labels, algo):
    """
    Assign customers to estimated segments based on segment labels.
    
    Parameters:
        pop: PopulationSimulator object with all simulated data
        segment_labels: array of segment labels for each customer
        
    Returns:
        None, modifies pop.customers in-place
    """
    for i, cust in enumerate(pop.train_customers):
        m = segment_labels[i]
        cust.est_segment[algo] = pop.est_segments_list[algo][m]
        assert cust.est_segment[algo].segment_id == m, f"Segment ID mismatch for customer {cust.customer_id}: expected {m}, got {cust.est_segment[algo].segment_id}"
        



def plot_segment_sankey(original, pruned):
    df = pd.DataFrame({'orig': original, 'pruned': pruned})
    flow = df.groupby(['orig', 'pruned']).size().reset_index(name='count')

    # Create node labels
    orig_labels = sorted(df['orig'].unique())
    pruned_labels = sorted(df['pruned'].unique())
    all_labels = [f'Orig {i}' for i in orig_labels] + [f'Pruned {i}' for i in pruned_labels]

    label_to_index = {label: idx for idx, label in enumerate(all_labels)}

    # Build Sankey diagram
    source = [label_to_index[f'Orig {o}'] for o in flow['orig']]
    target = [label_to_index[f'Pruned {p}'] for p in flow['pruned']]
    value = flow['count'].tolist()

    fig = go.Figure(data=[go.Sankey(
        node=dict(label=all_labels),
        link=dict(source=source, target=target, value=value)
    )])
    fig.update_layout(title_text="Segment Merge Flow (Original → Pruned)", font_size=12)
    fig.show()
    


# estimated total profits of a segment after applying the learnt policy to the customers in that segment
def compute_node_DR_value(
    Y, D, gamma, indices, use_hybrid_method,
    *,
    X=None,
    action_method: str,
    action_num: int | None = None,
    include_interactions: bool = False,
):
    D_m = D[indices]
    Y_m = Y[indices]
    X_full = np.asarray(X) if X is not None else np.zeros((len(np.ravel(Y)), 1))

    a_i = recommend_segment_action(
        X_full,
        D,
        Y,
        method=action_method,
        gamma=gamma,
        indices=indices,
        action_num=action_num,
        include_interactions=include_interactions,
    )
    if a_i == 404:
        return 0.0

    if use_hybrid_method is True:
        # Method 1: Direct + Gamma
        # If D_m[i] = a_i, then we get profit Y_m[i]
        # If D_m[i] != a_i, then we get profit gamma_a_i_m[i]
        gamma_a_i_m = gamma[indices, a_i].reshape(-1, 1)
        value = np.sum(Y_m[D_m == a_i]) + np.sum(gamma_a_i_m[D_m != a_i])
    else:
        # Method 2: Gamma only
        # Sum of gamma values for the recommended action
        gamma_a_i_m = gamma[indices, a_i]
        value = np.sum(gamma_a_i_m)
    
    return value


def evaluate_on_validation(pop: PopulationSimulator, algo, Gamma_val, customers=None):
    """
    Evaluate estimated policy on a set of customers using doubly-robust value.
    
    Args:
        pop: PopulationSimulator object
        algo: Algorithm name
        Gamma_val: Gamma matrix for the customers
        customers: List of customers to evaluate (default: pop.val_customers)
    """
    # Use validation customers by default, or specified customers
    if customers is None:
        customers = pop.val_customers
    
    if len(customers) == 0:
        return None

    V = 0
    for i, cust in enumerate(customers):
        assigned_action = cust.est_segment[algo].est_action
        
        # BUG FIX: Only handle the case when action is undecided (404)
        if assigned_action == 404:
            if algo in ["mst", "gmm-standard", "gmm-da", "kmeans-standard", "kmeans-da"]:
                assigned_action = np.random.randint(0, pop.action_num)
                cust.est_segment[algo].est_action = assigned_action
            else:
                assigned_action = cust.true_segment.action
                cust.est_segment[algo].est_action = cust.true_segment.action
        
        if int(assigned_action) == int(cust.D_i):
            V += cust.y
        else:
            V += Gamma_val[i, assigned_action]

    return V 

def assign_new_customers_to_segments(pop: PopulationSimulator, customers, model, algo):
    """
    Assign new customers to segments based on the trained model.
    
    Parameters:
        pop: PopulationSimulator object with all simulated data
        algo: algorithm used for segmentation
    Returns:
        None, modifies pop.customers in-place
    """
    
    x_new = np.array([cust.x for cust in customers])
    if x_new.shape[0] > 0:
        labels = model.predict(x_new)
        for cust, seg_id in zip(customers, labels):
            segment = pop.est_segments_list[f"{algo}"][seg_id]
            cust.est_segment[f"{algo}"] = segment


def pick_M_for_algo(algo, df_results_M):

    val_col = f'{algo}_val'

    max_val_algos = ["gmm-da", "kmeans-da", "clr-da",
                     "dast", "mst", "kmeans-standard"]
    min_val_algos = ["gmm-standard", "clr-standard"]
    meta_learners = ["t_learner", "s_learner", "x_learner",
                     "dr_learner", "causal_forest", "policy_tree"]

    if algo not in max_val_algos and algo not in min_val_algos and algo not in meta_learners:
        raise ValueError(f"Unknown algorithm: {algo}")

    is_maximize = algo in max_val_algos
    is_minimize = algo in min_val_algos
    is_meta_learner = algo in meta_learners

    if is_maximize:
        best_score = df_results_M[val_col].max()
        candidates = df_results_M[df_results_M[val_col] == best_score].copy()
        
        picked_M = int(candidates["M"].min())
        
        # if algo == "dast":
        #     picked_M = int(candidates["M"].max())  
        # else:
        #     picked_M = int(candidates["M"].min())

    elif is_minimize:
        best_score = df_results_M[val_col].min()
        candidates = df_results_M[df_results_M[val_col] == best_score].copy()
        picked_M = int(candidates["M"].min())

    elif is_meta_learner:
        picked_M = "Not applicable"

    return {f'{algo}_picked_M': picked_M}




def parse_args():
    parser = argparse.ArgumentParser(description="Simulation configuration for experiment")

    parser.add_argument("--plot", action="store_true", help="Enable plotting")

    parser.add_argument("--beta_range",  type=float, nargs=2, help="Range for covariate effect beta")
    parser.add_argument("--target_p_range", type=float, nargs=2, required=True,
                        help="P(Y=1 | x=x_mean, D=a) range for NON-winner actions "
                             "(or all actions when --winner_p_range is not set). "
                             "Alpha and tau are back-computed to hit these targets exactly.")
    parser.add_argument("--winner_p_range", type=float, nargs=2, default=None,
                        help="[optional] P(Y=1 | x=x_mean, D=a*) range for the "
                             "WINNER action (randomly chosen per segment). When set, one action per "
                             "segment gets a probability drawn from this range while all other "
                             "actions use --target_p_range, ensuring a clear best action. "
                             "Example: --target_p_range 0.02 0.10 --winner_p_range 0.15 0.40")
    parser.add_argument("--delta_range", type=float, nargs=2,
                        help="Range for delta (treatment–covariate interaction) in the DGP")
    parser.add_argument("--x_mean_range", type=float, nargs=2, help="Range for x_mean parameter")

    
    parser.add_argument("--N_segment_size", type=int, help="Number of customers per segment")
    parser.add_argument("--d", type=int, help="Dimensionality of covariates")
    parser.add_argument("--partial_x", type=float, help="Fraction of x used in outcome generation")
    parser.add_argument("--K", type=int, help="Number of segments")
    parser.add_argument("--action_num", type=int, default=2, help="Number of actions (default: 2 for binary treatment)")
    parser.add_argument("--disallowed_ball_radius", type=float, help="Minimum distance between mean vectors as scale factor (e.g., 0.8 means min_dist = 0.8 * space_range/K^(1/d)). Default: 0.5")
    parser.add_argument("--X_noise_std_scale", type=float, default=None,
                        help="[legacy] Scale factor for within-cluster covariate noise as a multiple of average distance between mean vectors")
    parser.add_argument("--target_mahalanobis_sep", type=float, default=None,
                        help="Target nearest-neighbor Mahalanobis separation. "
                             "When set, within-cluster covariate noise is chosen so that "
                             "the median nearest-neighbor center distance is this many standard deviations.")
    parser.add_argument("--disturb_covariate_noise", type=float, help="Covariate noise across segments")
    
    parser.add_argument("--kmeans_coef", type=float, help="Coefficient for k-means weighting")

    parser.add_argument("--DR_generation_method", type=str,
                        choices=["reg", "mlp", "lightgbm", "random_forest", "xgboost"],
                        help="DR generation method. 'reg': logistic regression. 'mlp': neural network. "
                             "'lightgbm': LightGBM. 'random_forest': sklearn RandomForest. 'xgboost': XGBoost.")
    
    # default is True
    parser.add_argument("--use_hybrid_method", type=lambda x: str(x).lower() == 'true', 
                        default=True, 
                        help="Use hybrid method for tree splitting and evaluation (default: True). Pass True or False")

    parser.add_argument(
        "--action_method",
        type=str,
        choices=list(ACTION_METHODS),
        help="Node/segment action rule: diff_in_means, gamma (argmax mean DR score), "
             "or logistic (node-level logistic model).",
    )
    
    parser.add_argument("--implementation_scale", type=float, help="Scale of implementation population")

    parser.add_argument("--N_sims", type=int, help="Number of simulations to run")
    
    parser.add_argument("--n_workers", type=int, default=None,
                        help="Deprecated/ignored: simulations always run sequentially "
                             "(avoids macOS fork + LightGBM/OpenMP crashes).")

    parser.add_argument(
        "--cv_folds",
        type=int,
        default=5,
        help="Folds for M selection on dast / *-da (default: 5). "
             "cv_folds=1 uses a single 80/20 holdout instead of K-fold CV.",
    )

    parser.add_argument("--save_file", type=str, help="Path to save experiment results")
    
    parser.add_argument("--sequence_seed", type=int, help="Random seed for simulation sequence")

    # 算法列表
    parser.add_argument("--algorithms", type=str, nargs="+", default=[],
                        help="List of algorithms to run")

    args = parser.parse_args()
    return args


import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPRegressor, MLPClassifier
from econml.dr import DRLearner
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri, default_converter
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
from scipy.spatial.distance import pdist, squareform


ALGORITHMS = ["kmeans-standard", "kmeans-da", 
              "gmm-standard", "gmm-da", 
              "clr-standard", "clr-da",
              "policy_tree", 
              "mst", 
              "dast", "dast_old",
              "t_learner",
              "x_learner",
              "s_learner",
              "dr_learner",
              "causal_forest"]

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -6, 6)))


class SegmentTrue:
    """
    True segment parameters.

    Both outcome types share the same parameter structure:
        alpha  : scalar intercept
        beta   : (d,)            covariate main effects
        tau    : (action_num,)   treatment effects; tau[0] = 0 by convention
        delta  : (action_num, d) or None — treatment-covariate interactions (optional)

    Outcome generation:
        continuous : y = alpha + beta@x + tau[D] + (delta[D]@x if delta) + N(0, noise_std)
        discrete   : eta = alpha + beta@x + tau[D] + (delta[D]@x if delta)
                     p   = sigmoid(eta)
                     y   ~ Bernoulli(p)

    Optimal action (x-independent for both types because alpha and beta@x cancel
    across actions when comparing actions):
        action = argmax_a tau[a]
    """

    def __init__(self, segment_id=None, x_mean=None, outcome_type=None,
                 alpha=None, beta=None, tau=None, delta=None):
        if outcome_type is None:
            raise ValueError("outcome_type must be specified: 'continuous' or 'discrete'")
        if outcome_type not in ('continuous', 'discrete'):
            raise ValueError(f"Unknown outcome_type: '{outcome_type}'. Must be 'continuous' or 'discrete'.")

        assert alpha is not None, "alpha must be provided"
        assert beta  is not None, "beta must be provided"
        assert tau   is not None, "tau must be provided"

        self.outcome_type = outcome_type
        self.segment_id   = segment_id
        self.x_mean       = x_mean

        self.alpha  = alpha
        self.beta   = beta   # shape (d,)
        self.tau    = tau    # shape (action_num,); tau[0] = 0
        self.delta  = delta  # shape (action_num, d) or None
        self.action = self._best_action_at(x_mean)

    def _best_action_at(self, x):
        """Return argmax_a E[Y | x, D=a], accounting for delta interactions.

        Falls back to argmax(tau) when x or delta is not available.
        """
        if x is None or self.delta is None:
            return int(np.argmax(self.tau))
        scores = self.tau + self.delta @ np.asarray(x)
        return int(np.argmax(scores))

    def _linear_predictor(self, x, D_i, signal_d):
        """Shared linear predictor η = alpha + beta@x + tau[D] + (delta[D]@x)."""
        eta = self.alpha + self.beta[:signal_d] @ x[:signal_d] + self.tau[int(D_i)]
        if self.delta is not None:
            eta += self.delta[int(D_i), :signal_d] @ x[:signal_d]
        return eta

    def generate_outcome(self, x, D_i, noise_std, signal_d):
        """Stochastic outcome sample — use for data generation only."""
        eta = self._linear_predictor(x, D_i, signal_d)
        if self.outcome_type == 'discrete':
            # Y ~ Bernoulli(sigmoid(eta))
            p = _sigmoid(eta)
            return int(np.random.binomial(1, p))
        else:
            # Y = eta + N(0, noise_std)
            return eta + np.random.normal(0, noise_std)



class SegmentEstimate:
    def __init__(self, est_tau, est_action, segment_id=None):
        self.est_tau = est_tau
        self.est_action = est_action  # 0 or 1
        self.segment_id = segment_id


class Customer_pilot:
    def __init__(self, x, D_i, y, true_segment: SegmentTrue, signal_d, customer_id=None):
        self.x = x
        self.D_i = D_i
        self.true_segment = true_segment
        self.y = y
        self.signal_d = signal_d
        self.est_segment = {algo: None for algo in ALGORITHMS}
        self.customer_id = customer_id

    def expected_outcome(self, D_i):
        """E[Y | self.x, D=D_i] under true segment parameters."""
        seg = self.true_segment
        eta = seg._linear_predictor(self.x, D_i, self.signal_d)
        return float(_sigmoid(eta)) if seg.outcome_type == 'discrete' else float(eta)


class Customer_implement:
    def __init__(self, x, true_segment: SegmentTrue, noise_std, signal_d):
        self.x = x
        self.true_segment = true_segment
        self.noise_std = noise_std
        self.signal_d = signal_d
        self.est_segment = {algo: None for algo in ALGORITHMS}

    def expected_outcome(self, D_i):
        """E[Y | self.x, D=D_i] under true segment parameters."""
        seg = self.true_segment
        eta = seg._linear_predictor(self.x, D_i, self.signal_d)
        return float(_sigmoid(eta)) if seg.outcome_type == 'discrete' else float(eta)

    def evaluate_profits(self, algo, implement_action=None):
        """Evaluate deterministic expected profit under the algorithm's recommended action."""
        if implement_action is not None:
            self.implement_action = implement_action
        else:
            if self.est_segment[algo].est_action != 404:
                self.implement_action = self.est_segment[algo].est_action
            else:
                self.implement_action = 1  # 404 fallback

        self.y = self.expected_outcome(self.implement_action)
        return self.y
        
# ----------------------------------------
# Population Simulator
# ----------------------------------------

class PopulationSimulator:
    def __init__(self, N_total_pilot_customers, N_total_implement_customers, d, K, disturb_covariate_noise, param_range, DR_generation_method, partial_x, action_num, X_mean_vectors=None, X_noise_std_scale=None, Y_noise_std_scale=None, disallowed_ball_radius=None, outcome_type=None):
        self.N_total_pilot_customers = N_total_pilot_customers
        self.N_total_implement_customers = N_total_implement_customers
        self.d = d
        self.K = K
        self.action_num = action_num
        if outcome_type is None:
            raise ValueError("outcome_type is required. Please specify 'continuous' or 'discrete'.")
        self.outcome_type = outcome_type

        self.param_range = param_range
        self.disturb_covariate_noise = disturb_covariate_noise
        self.signal_d = d if partial_x == 1 else max(1, int(d * partial_x))
        self.disturb_d = d - self.signal_d
        self.disallowed_ball_radius = disallowed_ball_radius if disallowed_ball_radius is not None else 0

        self.true_segments = self._init_true_segments(X_mean_vectors)
        
        # Compute signal_covariate_noise based on X_noise_std_scale (required parameter)
        if X_noise_std_scale is None:
            raise ValueError("X_noise_std_scale is required. Please provide a scale factor for within-cluster covariate noise.")
        
        if self.K <= 1:
            raise ValueError(f"Cannot use X_noise_std_scale with K={self.K}. Need at least 2 clusters to compute average distance.")
        
        mean_vectors_signal = np.array([seg.x_mean[:self.signal_d] for seg in self.true_segments])
        pairwise_distances = pdist(mean_vectors_signal, metric='euclidean')
        
        if len(pairwise_distances) == 0:
            raise ValueError(f"No pairwise distances computed for K={self.K} clusters.")
        
        avg_distance = np.mean(pairwise_distances)
        self.signal_covariate_noise = X_noise_std_scale * avg_distance
        print(f"Computed X_covariate_noise: {self.signal_covariate_noise:.4f} (scale={X_noise_std_scale}, avg_distance={avg_distance:.4f})")
        
        
        
        # noise_std: only meaningful for continuous; discrete uses Bernoulli randomness
        if outcome_type == 'continuous':
            self._adjust_adjacent_cluster_tau()
            if Y_noise_std_scale is None:
                raise ValueError("Y_noise_std_scale is required for continuous outcome_type.")
            tau_values = np.array([seg.tau[1:] for seg in self.true_segments]).flatten()
            avg_tau_magnitude = np.mean(np.abs(tau_values)) if len(tau_values) > 0 else 1.0
            self.noise_std = Y_noise_std_scale * avg_tau_magnitude
            print(f"Computed Y_noise_std: {self.noise_std:.4f} (scale={Y_noise_std_scale}, avg_|tau|={avg_tau_magnitude:.4f})")
        else:
            # discrete: no Gaussian noise; noise_std is irrelevant but stored as 0
            self.noise_std = 0.0
            if Y_noise_std_scale is not None:
                print("Note: Y_noise_std_scale is ignored for discrete outcome_type.")

        self.pilot_customers = self._generate_pilot_customers()
        self.implement_customers = self._generate_implement_customers()

        if outcome_type == 'discrete':
            self._print_segment_sigmoids()
        
        self.est_segments_list = {algo: [] for algo in ALGORITHMS}
        
        # Store DR generation method for later use
        self.DR_generation_method = DR_generation_method
        self.gamma_train = None  # Will be computed after train/val split
        self.gamma_val = None    # Will be computed after train/val split
        
        self.train_customers, self.val_customers, self.train_indices, self.val_indices = None, None, None, None
        
            
    def _init_true_segments(self, X_mean_vectors):
        pr = self.param_range  # alias for convenience
        true_segments = []
        
        def _logit(p):
            return np.log(p / (1.0 - p))

        def _make_segment_params(k, x_mean=None):
            """Sample outcome parameters for one segment.

            Continuous : alpha ~ Uniform(alpha_range); beta, tau ~ Uniform(tau_range).

            Discrete (target_p mode):
              For EVERY action a (including a=0), we sample a target probability
                target_p_a ~ Uniform(target_p_range)
              and back-compute parameters so that
                P(Y=1 | D=a, x=x_mean) = target_p_a  for all a.

              Concretely:
                alpha  = logit(target_p_0) - beta@x_mean        [absorbs x_mean baseline]
                tau[a] = logit(target_p_a) - logit(target_p_0)
                         - delta[a]@x_mean                       [absorbs delta@x_mean offset]

              This ensures that delta@x only contributes *deviations* around x_mean,
              so the sigmoid is not saturated even when x_mean is large.
              tau_range is ignored in discrete mode (tau is derived, not sampled freely).
            """
            beta = np.random.uniform(*pr["beta"], size=self.d)

            # Generate delta FIRST — required before back-computing tau in discrete mode
            if pr.get("delta") is not None:
                delta_mat = np.zeros((self.action_num, self.d))
                for a in range(1, self.action_num):
                    delta_mat[a] = np.random.uniform(*pr["delta"], size=self.d)
            else:
                delta_mat = None

            sd = self.signal_d
            xm = x_mean[:sd] if x_mean is not None else None

            if pr.get("target_p") is not None:
                # --- Discrete mode: back-compute alpha and tau from target probabilities ---

                # If winner_p is set, one random action gets a high probability (winner);
                # all others get a low probability (loser). This guarantees a clear best action.
                winner_a = (np.random.randint(0, self.action_num)
                            if pr.get("winner_p") is not None else None)
                print(f"    Segment {k}: winner_a={winner_a} (if any)")

                def _sample_p(a):
                    if winner_a is not None and a == winner_a:
                        return np.clip(np.random.uniform(*pr["winner_p"]), 1e-6, 1 - 1e-6)
                    return np.clip(np.random.uniform(*pr["target_p"]), 1e-6, 1 - 1e-6)

                # Step 1: baseline (D=0)
                target_p_0 = _sample_p(0)
                logit_0    = _logit(target_p_0)
                beta_dot   = float(beta[:sd] @ xm) if xm is not None else 0.0
                alpha      = logit_0 - beta_dot          # sigmoid(alpha + beta@x_mean) = target_p_0

                # Step 2: non-baseline actions (a >= 1)
                tau_vec = np.zeros(self.action_num)      # tau[0] = 0 by convention
                for a in range(1, self.action_num):
                    target_p_a  = _sample_p(a)
                    logit_a     = _logit(target_p_a)
                    delta_dot   = float(delta_mat[a, :sd] @ xm) if (xm is not None and delta_mat is not None) else 0.0
                    # sigmoid(alpha + beta@x_mean + tau[a] + delta[a]@x_mean) = target_p_a
                    # => tau[a] = logit_a - logit_0 - delta[a]@x_mean
                    tau_vec[a]  = logit_a - logit_0 - delta_dot
            else:
                # --- Continuous mode: sample alpha and tau directly ---
                alpha   = np.random.uniform(*pr["alpha"])
                tau_vec = np.zeros(self.action_num)
                for a in range(1, self.action_num):
                    tau_vec[a] = np.random.uniform(*pr["tau"])

            return dict(alpha=alpha, beta=beta, tau=tau_vec, delta=delta_mat)

        # If generating mean vectors, use minimum distance constraint
        if X_mean_vectors is None:
            # Calculate a reasonable minimum distance based on space size
            space_range = pr["x_mean"][1] - pr["x_mean"][0]
            min_distance = (space_range / (self.K ** (1.0 / self.d))) * self.disallowed_ball_radius
            print(f"Generating mean vectors with minimum distance: {min_distance:.2f}")

            generated_means = []
            for k in range(self.K):
                # Sample x_mean FIRST so discrete can back-compute alpha from it
                max_attempts = 100
                x_mean = None
                for attempt in range(max_attempts):
                    x_mean_candidate = np.random.uniform(*pr["x_mean"], size=self.d)

                    if len(generated_means) == 0:
                        x_mean = x_mean_candidate
                        break

                    distances = [np.linalg.norm(x_mean_candidate - existing)
                                 for existing in generated_means]
                    min_dist_to_existing = min(distances)

                    if min_dist_to_existing >= min_distance:
                        x_mean = x_mean_candidate
                        break

                    if attempt == max_attempts - 1:
                        print(f"⚠️  Warning: Could not find mean vector for cluster {k} satisfying min distance.")
                        print(f"   Using best candidate with distance {min_dist_to_existing:.2f}")
                        x_mean = x_mean_candidate

                params = _make_segment_params(k, x_mean=x_mean)
                generated_means.append(x_mean)
                true_segments.append(SegmentTrue(
                    segment_id=k, x_mean=x_mean, outcome_type=self.outcome_type,
                    alpha=params["alpha"], beta=params["beta"],
                    tau=params["tau"],    delta=params["delta"]))
        else:
            # Use provided mean vectors
            for k in range(self.K):
                x_mean = X_mean_vectors[k]
                params = _make_segment_params(k, x_mean=x_mean)
                true_segments.append(SegmentTrue(
                    segment_id=k, x_mean=x_mean, outcome_type=self.outcome_type,
                    alpha=params["alpha"], beta=params["beta"],
                    tau=params["tau"],    delta=params["delta"]))
        
        return true_segments
    
    def _adjust_adjacent_cluster_tau(self):
        """
        Make sure "adjacent but not overlapping" clusters have opposite treatment effect signs.
        Adjacent means: distance ≈ 1-3 sigma (touching at boundaries, not fully overlapping).
        This creates a challenging scenario for algorithms.
        """
        if self.K < 2:
            return
        
        # Extract mean vectors (only signal dimensions matter for clustering)
        mean_vectors_signal = np.array([seg.x_mean[:self.signal_d] for seg in self.true_segments])
        
        # Compute pairwise distances in signal space
        dist_matrix = squareform(pdist(mean_vectors_signal, metric='euclidean'))
        
        # Set diagonal to infinity to exclude self-distances
        np.fill_diagonal(dist_matrix, np.inf)
        
        # Define "adjacent" as distance in range [lower_bound, upper_bound]
        # This means clusters touch at boundaries but don't heavily overlap
        sigma = self.signal_covariate_noise
        lower_bound = 1.0 * sigma  # Closer than this = too much overlap
        upper_bound = 4.0 * sigma  # Farther than this = well separated
        
        # Find pairs in the "adjacent" range
        adjacent_pairs = []
        for i in range(self.K):
            for j in range(i+1, self.K):
                dist = dist_matrix[i, j]
                if lower_bound <= dist <= upper_bound:
                    adjacent_pairs.append((i, j, dist))
        
        if adjacent_pairs:
            # Pick the pair with distance closest to 2*sigma (sweet spot for "touching")
            target_dist = 2.0 * sigma
            best_pair = min(adjacent_pairs, key=lambda x: abs(x[2] - target_dist))
            idx1, idx2, dist = best_pair
            
            action1_before = self.true_segments[idx1].action
            action2_before = self.true_segments[idx2].action
            
            # If same best action, fix seg2 to have a different best action
            if action1_before == action2_before:
                seg2 = self.true_segments[idx2]
                # Flip non-baseline tau signs so argmax changes
                # Works for both continuous and discrete (logistic) because
                # action = argmax(tau) in both cases.
                seg2.tau[1:] = -seg2.tau[1:]
                seg2.action = seg2._best_action_at(seg2.x_mean)

                action2_after = self.true_segments[idx2].action
                print(f"⚠️  Adjacent clusters (segments {idx1} and {idx2}) had same best action ({action1_before}).")
                print(f"   Distance: {dist:.2f}, sigma={sigma:.2f}")
                print(f"   Fixed seg {idx2}. New best action: {action2_after}")
            else:
                print(f"✓ Adjacent clusters (segments {idx1} and {idx2}) already have different best actions.")
                print(f"   Distance: {dist:.2f}, sigma={sigma:.2f}")
                print(f"   Seg{idx1}: action={action1_before}, Seg{idx2}: action={action2_before}")

    def _print_segment_sigmoids(self):
        """Print per-segment discrete outcome probabilities (discrete only).

        Two rows per segment:
          p(x_mean) : sigmoid(alpha + beta@x_mean + tau[D])  — the exact target_p guarantee point
          realized  : actual y=1 rate among pilot customers  — reflects the full x distribution
        """
        from collections import defaultdict

        # Collect pilot y=1 counts per (segment, action)
        seg_action_y = defaultdict(list)
        for c in self.pilot_customers:
            seg_action_y[(c.true_segment.segment_id, c.D_i)].append(c.y)

        action_labels = [f"D={a}" for a in range(self.action_num)]
        col_w = 10
        sep = "  "
        header_cols = sep.join(f"{lbl:>{col_w}}" for lbl in action_labels)
        header = f"{'Seg':>4}  {'best':>4}  {'':12}{header_cols}"
        divider = "-" * (4 + 2 + 4 + 2 + 12 + (col_w + 2) * self.action_num)

        print(f"\n{'─'*60}")
        print("  Discrete segment probabilities  P(Y=1 | D=a)")
        print(f"{'─'*60}")
        print(header)
        print(divider)

        for seg in self.true_segments:
            k = seg.segment_id
            # Row 1: p at x_mean (exact target_p guarantee)
            xmean_probs = [
                float(_sigmoid(seg._linear_predictor(seg.x_mean, a, self.signal_d)))
                if seg.outcome_type == 'discrete'
                else float(seg._linear_predictor(seg.x_mean, a, self.signal_d))
                for a in range(self.action_num)
            ]
            xmean_str = sep.join(f"{p:{col_w}.4f}" for p in xmean_probs)
            print(f"{k:>4}  {seg.action:>4}  {'p(x=x_mean)':12}{xmean_str}")

            # Row 2: realized y=1 rate among pilot customers
            realized = []
            for a in range(self.action_num):
                ys = seg_action_y.get((k, a), [])
                realized.append(np.mean(ys) if ys else float('nan'))
            real_str = sep.join(
                f"{p:{col_w}.4f}" if not np.isnan(p) else f"{'N/A':>{col_w}}"
                for p in realized
            )
            print(f"{'':>4}  {'':>4}  {'realized':12}{real_str}")

        print(f"{'─'*60}\n")

    def _generate_pilot_customers(self):
        pilot_customers = []

        cov_signal = np.eye(self.signal_d) * (self.signal_covariate_noise ** 2) 
        if self.disturb_d > 0:
            # === 1️⃣ Generate adversarial noise clusters ===
            cov_noise = np.eye(self.disturb_d) * (self.disturb_covariate_noise ** 2)
            N_noise_clusters = max(1, int(np.random.uniform(self.K - 5, self.K + 5)))
            noise_cluster_means = [
                np.random.uniform(*self.param_range["x_mean"], size=self.disturb_d)
                for _ in range(N_noise_clusters)
            ]
            noise_cluster_labels = np.random.randint(
                0, N_noise_clusters, size=self.N_total_pilot_customers
            )
        for i in range(self.N_total_pilot_customers):
            # 2️⃣ Choose true segment for signal part
            segment = np.random.choice(self.true_segments)

            # 3️⃣ Signal features: cluster-dependent
            # Only use first signal_d dimensions of x_mean
            x_signal = np.random.multivariate_normal(mean=segment.x_mean[:self.signal_d], cov=cov_signal)

            # 4️⃣ Disturbing features: come from random noise cluster
            if self.disturb_d > 0:
                w = noise_cluster_labels[i]
                x_noise = np.random.multivariate_normal(mean=noise_cluster_means[w], cov=cov_noise)
                x_full = np.concatenate([x_signal, x_noise])

            else:
                x_full = x_signal

            # 6️⃣ Outcome
            D_i = np.random.choice(self.action_num)  # Randomly assign action from 0 to action_num-1
            y = segment.generate_outcome(x_full, D_i, self.noise_std, self.signal_d)

            # 7️⃣ Save
            cust = Customer_pilot(x_full, D_i, y, segment, self.signal_d, customer_id=i)
            pilot_customers.append(cust)

        return pilot_customers
     
    def _generate_implement_customers(self):
        implement_customers = []

        # --- Signal and noise covariance ---
        cov_signal = np.eye(self.signal_d) * (self.signal_covariate_noise ** 2)
        

        # === 1️⃣  noise clusters ===
        if self.disturb_d > 0:
            cov_noise = np.eye(self.disturb_d) * (self.disturb_covariate_noise ** 2)
            N_noise_clusters = max(1, int(np.random.uniform(self.K - 5, self.K + 5)))

            # Randomly position these noise cluster means
            noise_cluster_means = [
                np.random.uniform(*self.param_range["x_mean"], size=self.disturb_d)
                for _ in range(N_noise_clusters)
            ]

            # Assign each implement customer to a random noise cluster
            noise_cluster_labels = np.random.randint(
                0, N_noise_clusters, size=self.N_total_implement_customers
            )

        for i in range(self.N_total_implement_customers):
            # 2️⃣ Choose true signal segment (same as before)
            segment = np.random.choice(self.true_segments)

            # 3️⃣ Signal part: segment-dependent
            # Only use first signal_d dimensions of x_mean
            x_signal = np.random.multivariate_normal(mean=segment.x_mean[:self.signal_d], cov=cov_signal)

            # 4️⃣ Noise part: from an unrelated latent noise cluster
            if self.disturb_d > 0:
                w = noise_cluster_labels[i]
                x_noise = np.random.multivariate_normal(mean=noise_cluster_means[w], cov=cov_noise)

                # 5️⃣ Combine signal + noise
                x_full = np.concatenate([x_signal, x_noise])
            else:
                x_full = x_signal

            # 6️⃣ Create Customer_implement object
            cust = Customer_implement(x_full, segment, self.noise_std, self.signal_d)
            implement_customers.append(cust)

        return implement_customers





    
    def compute_gamma_scores(self, method, train_customers, val_customers, n_crossfit_folds=5):
        """
        Compute doubly robust (DR) scores for both training and validation customers.

        Strategy:
        - Gamma_train : K-fold cross-fitting so every train sample gets an
                        out-of-fold mu prediction (avoids in-sample overfitting).
        - Gamma_val   : mu models fitted on ALL train data, predicted on val
                        (already out-of-sample, no cross-fitting needed).

        Parameters:
        method : str
            Outcome model type ('reg', 'mlp', 'lightgbm', 'random_forest', 'xgboost')
        train_customers : list
        val_customers : list  (can be empty)
        n_crossfit_folds : int
            Number of folds for cross-fitting Gamma_train (default 5)

        Returns:
        Gamma_train : np.ndarray, shape (N_train, n_actions)
        Gamma_val   : np.ndarray, shape (N_val, n_actions), or None
        """
        # Extract train data
        X_train = np.array([cust.x for cust in train_customers])
        D_train = np.array([cust.D_i for cust in train_customers])
        Y_train = np.array([cust.y  for cust in train_customers])

        if len(val_customers) > 0:
            X_val = np.array([cust.x  for cust in val_customers])
            D_val = np.array([cust.D_i for cust in val_customers])
            Y_val = np.array([cust.y  for cust in val_customers])
        else:
            X_val = D_val = Y_val = None

        n_actions  = self.action_num
        is_discrete = self.outcome_type == 'discrete'

        # Propensity scores estimated from full training data
        e = np.array([np.mean(D_train == a) for a in range(n_actions)])
        e = np.clip(e, 1e-6, 1.0)

        # ── helpers ───────────────────────────────────────────────────────────

        def _predict_mu(model, X):
            if hasattr(model, 'predict_proba'):
                return model.predict_proba(X)[:, 1]
            return model.predict(X)

        def _compute_gamma(X, D, Y, models):
            """Gamma[i,a] = mu_a(X_i) + (1/e[a]) * 1[D_i==a] * (Y_i - mu_a(X_i))"""
            N = X.shape[0]
            Gamma = np.zeros((N, n_actions))
            for a in range(n_actions):
                mu_a      = _predict_mu(models[a], X)
                indicator = (D == a).astype(float)
                Gamma[:, a] = mu_a + (indicator / e[a]) * (Y - mu_a)
            return Gamma

        def _make_model():
            """Return a fresh unfitted model for the current method."""
            if method == "reg":
                return LogisticRegression(max_iter=1000) if is_discrete else LinearRegression()
            elif method == "mlp":
                return (MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', max_iter=10000)
                        if is_discrete else
                        MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', max_iter=10000))
            elif method == "lightgbm":
                try:
                    from lightgbm import LGBMClassifier, LGBMRegressor
                except ImportError:
                    raise ImportError("lightgbm is not installed. Run: pip install lightgbm")
                return (LGBMClassifier(n_estimators=500, verbose=-1)
                        if is_discrete else LGBMRegressor(n_estimators=500, verbose=-1))
            elif method == "random_forest":
                from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
                return (RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42)
                        if is_discrete else
                        RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=42))
            elif method == "xgboost":
                try:
                    from xgboost import XGBClassifier, XGBRegressor
                except ImportError:
                    raise ImportError("xgboost is not installed. Run: pip install xgboost")
                return (XGBClassifier(n_estimators=500, verbosity=0, random_state=42)
                        if is_discrete else
                        XGBRegressor(n_estimators=500, verbosity=0, random_state=42))
            else:
                raise ValueError(f"Unknown DR generation method: '{method}'. "
                                 f"Choose from: reg, mlp, lightgbm, random_forest, xgboost")

        def _build_models(X, D, Y):
            """Fit one mu_a model per action on the provided subset."""
            models = {}
            for a in range(n_actions):
                mask = D == a
                X_a, Y_a = X[mask], Y[mask]
                if len(X_a) == 0:
                    raise ValueError(f"No training samples for action {a}. Cannot fit outcome model.")
                m = _make_model()
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*valid feature names.*",
                                            category=UserWarning)
                    m.fit(X_a, Y_a)
                models[a] = m
            return models

        def _crossfit_gamma_train():
            """K-fold cross-fitting: each train sample gets an out-of-fold prediction."""
            N = len(X_train)
            Gamma_cf   = np.zeros((N, n_actions))
            fold_idx   = np.array_split(np.arange(N), n_crossfit_folds)
            for k, val_idx in enumerate(fold_idx):
                tr_idx   = np.concatenate([fold_idx[j] for j in range(n_crossfit_folds) if j != k])
                models_k = _build_models(X_train[tr_idx], D_train[tr_idx], Y_train[tr_idx])
                Gamma_cf[val_idx] = _compute_gamma(
                    X_train[val_idx], D_train[val_idx], Y_train[val_idx], models_k)
            return Gamma_cf

        # ── compute ───────────────────────────────────────────────────────────

        # Full-train models are used only for Gamma_val (already out-of-sample)
        models_full = _build_models(X_train, D_train, Y_train)

        Gamma_train = _crossfit_gamma_train()
        Gamma_val   = (_compute_gamma(X_val, D_val, Y_val, models_full)
                       if X_val is not None and len(X_val) > 0 else None)

        return Gamma_train, Gamma_val


    def split_pilot_customers_into_train_and_validate(self, train_frac):
        indices = np.arange(self.N_total_pilot_customers)

        split = int(self.N_total_pilot_customers * train_frac)
        self.train_indices, self.val_indices = indices[:split], indices[split:]

        self.train_customers = [self.pilot_customers[i] for i in self.train_indices]
        self.val_customers = [self.pilot_customers[i] for i in self.val_indices]
        
        # Compute gamma scores
        if train_frac < 1.0:
            # Normal case: have both train and val
            self.gamma_train, self.gamma_val = self.compute_gamma_scores(
                self.DR_generation_method, self.train_customers, self.val_customers
            )
        else:
            # train_frac=1.0: all data is training, no validation
            # Still compute gamma_train (some algorithms need it), but gamma_val = None
            self.gamma_train, self.gamma_val = self.compute_gamma_scores(
                self.DR_generation_method, self.train_customers, []  # Empty val_customers
            )

    def to_dataframe(self):
        data = []
        for cust in self.pilot_customers:
            row = {
                "customer_id": cust.customer_id,
                "true_segment_id": cust.true_segment.segment_id,
                "D_i": cust.D_i,
                "outcome": cust.y,
            }

            # 动态添加每个算法的 est_segment_id
            for algo in ALGORITHMS:
                row[f"{algo}_est_segment_id"] = (
                    cust.est_segment[algo].segment_id if cust.est_segment[algo] is not None else None
                )
            # 添加特征向量 x
            for j in range(self.d):
                row[f"x_{j}"] = cust.x[j]

            data.append(row)

        return pd.DataFrame(data)










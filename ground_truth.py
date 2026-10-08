import warnings
import math
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from scipy.spatial.distance import pdist, squareform


ALGORITHMS = ["kmeans-standard", "kmeans-da", 
              "gmm-standard", "gmm-da", 
              "clr-standard", "clr-da",
              "policy_tree", 
              "mst", 
              "dast",
              "t_learner",
              "x_learner",
              "s_learner",
              "dr_learner",
              "causal_forest"]

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -6, 6)))


class SegmentTrue:
    """
    True segment parameters for discrete (Bernoulli) outcomes.

    Parameter structure:
        alpha  : scalar intercept
        beta   : (d,)            covariate main effects
        tau    : (action_num,)   treatment effects; tau[0] = 0 by convention
        delta  : (action_num, d) or None — treatment-covariate interactions (optional)

    Outcome generation:
        eta = alpha + beta@x + tau[D] + (delta[D]@x if delta)
        p   = sigmoid(eta)
        y   ~ Bernoulli(p)

    Optimal action at x:
        action = argmax_a E[Y | x, D=a]
    """

    def __init__(self, segment_id=None, x_mean=None,
                 alpha=None, beta=None, tau=None, delta=None):
        assert alpha is not None, "alpha must be provided"
        assert beta  is not None, "beta must be provided"
        assert tau   is not None, "tau must be provided"

        self.outcome_type = 'discrete'
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
        """Stochastic Bernoulli outcome sample — use for data generation only."""
        eta = self._linear_predictor(x, D_i, signal_d)
        p = _sigmoid(eta)
        return int(np.random.binomial(1, p))



class SegmentEstimate:
    def __init__(self, est_tau, est_action, segment_id=None):
        self.est_tau = est_tau
        self.est_action = est_action  # recommended action (int 0..K-1, or 404 = missing)
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
        """E[Y | self.x, D=D_i] = sigmoid(eta) under true segment parameters."""
        seg = self.true_segment
        eta = seg._linear_predictor(self.x, D_i, self.signal_d)
        return float(_sigmoid(eta))


class Customer_implement:
    def __init__(self, x, true_segment: SegmentTrue, noise_std, signal_d):
        self.x = x
        self.true_segment = true_segment
        self.noise_std = noise_std
        self.signal_d = signal_d
        self.action_num = len(true_segment.tau)
        self.est_segment = {algo: None for algo in ALGORITHMS}

    def expected_outcome(self, D_i):
        """E[Y | self.x, D=D_i] = sigmoid(eta) under true segment parameters."""
        seg = self.true_segment
        eta = seg._linear_predictor(self.x, D_i, self.signal_d)
        return float(_sigmoid(eta))

    def evaluate_profits(self, algo, implement_action=None):
        """Evaluate deterministic expected profit under the algorithm's recommended action."""
        if implement_action is not None:
            self.implement_action = implement_action
        else:
            if self.est_segment[algo].est_action != 404:
                self.implement_action = self.est_segment[algo].est_action
            else:
                self.implement_action = np.random.randint(0, self.action_num)  # 404 fallback

        self.y = self.expected_outcome(self.implement_action)
        return self.y
        
# ----------------------------------------
# Population Simulator
# ----------------------------------------

class PopulationSimulator:
    def __init__(self, N_total_pilot_customers, N_total_implement_customers, d, K, disturb_covariate_noise, param_range, DR_generation_method, partial_x, action_num, X_mean_vectors=None, X_noise_std_scale=None, target_mahalanobis_sep=None, disallowed_ball_radius=None):
        self.N_total_pilot_customers = N_total_pilot_customers
        self.N_total_implement_customers = N_total_implement_customers
        self.d = d
        self.K = K
        self.action_num = action_num
        self.outcome_type = 'discrete'

        self.param_range = param_range
        self.disturb_covariate_noise = disturb_covariate_noise
        self.signal_d = d if partial_x == 1 else max(1, int(d * partial_x))
        self.disturb_d = d - self.signal_d
        self.disallowed_ball_radius = disallowed_ball_radius if disallowed_ball_radius is not None else 0

        self.true_segments = self._init_true_segments(X_mean_vectors)
        
        if X_noise_std_scale is None and target_mahalanobis_sep is None:
            raise ValueError(
                "Provide either target_mahalanobis_sep or legacy X_noise_std_scale "
                "to set within-cluster covariate noise."
            )

        if X_noise_std_scale is not None and target_mahalanobis_sep is not None:
            raise ValueError("Use only one of target_mahalanobis_sep or X_noise_std_scale.")

        if target_mahalanobis_sep is not None and target_mahalanobis_sep <= 0:
            raise ValueError(f"target_mahalanobis_sep must be positive, got {target_mahalanobis_sep}.")
        
        if self.K <= 1:
            raise ValueError(f"Need at least 2 clusters to compute covariate noise, got K={self.K}.")
        
        mean_vectors_signal = np.array([seg.x_mean[:self.signal_d] for seg in self.true_segments])
        pairwise_distances = pdist(mean_vectors_signal, metric='euclidean')
        
        if len(pairwise_distances) == 0:
            raise ValueError(f"No pairwise distances computed for K={self.K} clusters.")
        
        avg_distance = float(np.mean(pairwise_distances))
        median_distance = float(np.median(pairwise_distances))
        dist_matrix = squareform(pairwise_distances)
        np.fill_diagonal(dist_matrix, np.inf)
        nearest_distances = dist_matrix.min(axis=1)
        median_nn_distance = float(np.median(nearest_distances))
        min_nn_distance = float(np.min(nearest_distances))

        self.covariate_distance_summary = {
            "pairwise_mean_distance": avg_distance,
            "pairwise_median_distance": median_distance,
            "nearest_neighbor_median_distance": median_nn_distance,
            "nearest_neighbor_min_distance": min_nn_distance,
        }

        if target_mahalanobis_sep is not None:
            self.signal_covariate_noise = median_nn_distance / target_mahalanobis_sep
            self.target_mahalanobis_sep = target_mahalanobis_sep
            self.realized_median_nn_mahalanobis_sep = median_nn_distance / self.signal_covariate_noise
            self.realized_min_nn_mahalanobis_sep = min_nn_distance / self.signal_covariate_noise
            self.realized_median_nn_bayes_error = 0.5 * math.erfc(
                self.realized_median_nn_mahalanobis_sep / (2.0 * np.sqrt(2.0))
            )
            self.realized_min_nn_bayes_error = 0.5 * math.erfc(
                self.realized_min_nn_mahalanobis_sep / (2.0 * np.sqrt(2.0))
            )
            print(
                "Computed X_covariate_noise: "
                f"{self.signal_covariate_noise:.4f} "
                f"(target_mahalanobis_sep={target_mahalanobis_sep}, "
                f"median_nn_distance={median_nn_distance:.4f}, "
                f"min_nn_distance={min_nn_distance:.4f})"
            )
        else:
            self.signal_covariate_noise = X_noise_std_scale * avg_distance
            self.target_mahalanobis_sep = None
            self.realized_median_nn_mahalanobis_sep = median_nn_distance / self.signal_covariate_noise
            self.realized_min_nn_mahalanobis_sep = min_nn_distance / self.signal_covariate_noise
            self.realized_median_nn_bayes_error = 0.5 * math.erfc(
                self.realized_median_nn_mahalanobis_sep / (2.0 * np.sqrt(2.0))
            )
            self.realized_min_nn_bayes_error = 0.5 * math.erfc(
                self.realized_min_nn_mahalanobis_sep / (2.0 * np.sqrt(2.0))
            )
            print(
                "Computed X_covariate_noise: "
                f"{self.signal_covariate_noise:.4f} "
                f"(legacy scale={X_noise_std_scale}, avg_distance={avg_distance:.4f}, "
                f"median_nn_mahalanobis={self.realized_median_nn_mahalanobis_sep:.4f})"
            )
        
        
        
        # Discrete Bernoulli: no Gaussian outcome noise
        self.noise_std = 0.0

        self.pilot_customers = self._generate_pilot_customers()
        self.implement_customers = self._generate_implement_customers()
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
            """Sample discrete outcome parameters for one segment.

            For EVERY action a (including a=0), sample a target probability
              target_p_a ~ Uniform(target_p_range)
            and back-compute parameters so that
              P(Y=1 | D=a, x=x_mean) = target_p_a  for all a.

            Concretely:
              alpha  = logit(target_p_0) - beta@x_mean
              tau[a] = logit(target_p_a) - logit(target_p_0) - delta[a]@x_mean

            If winner_p is set, one random action gets a high probability while
            others use target_p, guaranteeing a clear best action.
            """
            if pr.get("target_p") is None:
                raise ValueError(
                    "param_range['target_p'] is required for discrete DGP "
                    "(set via --target_p_range)."
                )

            beta = np.random.uniform(*pr["beta"], size=self.d)

            # Generate delta FIRST — required before back-computing tau
            if pr.get("delta") is not None:
                delta_mat = np.zeros((self.action_num, self.d))
                for a in range(1, self.action_num):
                    delta_mat[a] = np.random.uniform(*pr["delta"], size=self.d)
            else:
                delta_mat = None

            sd = self.signal_d
            xm = x_mean[:sd] if x_mean is not None else None

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
            alpha      = logit_0 - beta_dot

            # Step 2: non-baseline actions (a >= 1)
            tau_vec = np.zeros(self.action_num)
            for a in range(1, self.action_num):
                target_p_a  = _sample_p(a)
                logit_a     = _logit(target_p_a)
                delta_dot   = float(delta_mat[a, :sd] @ xm) if (xm is not None and delta_mat is not None) else 0.0
                tau_vec[a]  = logit_a - logit_0 - delta_dot

            return dict(alpha=alpha, beta=beta, tau=tau_vec, delta=delta_mat)

        # If generating mean vectors, use minimum distance constraint
        if X_mean_vectors is None:
            # Calculate a reasonable minimum distance based on space size
            space_range = pr["x_mean"][1] - pr["x_mean"][0]
            min_distance = (space_range / (self.K ** (1.0 / self.d))) * self.disallowed_ball_radius
            print(f"Generating mean vectors with minimum distance: {min_distance:.2f}")

            generated_means = []
            for k in range(self.K):
                # Sample x_mean FIRST so we can back-compute alpha from it
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
                    segment_id=k, x_mean=x_mean,
                    alpha=params["alpha"], beta=params["beta"],
                    tau=params["tau"],    delta=params["delta"]))
        else:
            # Use provided mean vectors
            for k in range(self.K):
                x_mean = X_mean_vectors[k]
                params = _make_segment_params(k, x_mean=x_mean)
                true_segments.append(SegmentTrue(
                    segment_id=k, x_mean=x_mean,
                    alpha=params["alpha"], beta=params["beta"],
                    tau=params["tau"],    delta=params["delta"]))
        
        return true_segments
    
    def _print_segment_sigmoids(self):
        """Print per-segment discrete outcome probabilities.

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





    
    def compute_gamma_scores(self, method, train_customers, val_customers):
        """
        Compute doubly robust (DR) scores for both training and validation customers.

        Strategy:
        - Fit mu_a once on the full train set, then compute Gamma_train in-sample.
        - Gamma_val uses the same train-fitted models (already out-of-sample on val).

        Parameters:
        method : str
            Outcome model type ('reg', 'mlp', 'lightgbm', 'random_forest', 'xgboost')
        train_customers : list
        val_customers : list  (can be empty)

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
            """Return a fresh unfitted Bernoulli outcome model."""
            if method == "reg":
                return LogisticRegression(max_iter=1000)
            elif method == "mlp":
                return MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', max_iter=10000)
            elif method == "lightgbm":
                try:
                    from lightgbm import LGBMClassifier
                except ImportError:
                    raise ImportError("lightgbm is not installed. Run: pip install lightgbm")
                return LGBMClassifier(n_estimators=500, n_jobs=1, verbose=-1)
            elif method == "random_forest":
                from sklearn.ensemble import RandomForestClassifier
                return RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42)
            elif method == "xgboost":
                try:
                    from xgboost import XGBClassifier
                except ImportError:
                    raise ImportError("xgboost is not installed. Run: pip install xgboost")
                return XGBClassifier(n_estimators=500, verbosity=0, random_state=42)
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

        # ── compute once on full train ────────────────────────────────────────
        models = _build_models(X_train, D_train, Y_train)
        Gamma_train = _compute_gamma(X_train, D_train, Y_train, models)
        Gamma_val   = (_compute_gamma(X_val, D_val, Y_val, models)
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








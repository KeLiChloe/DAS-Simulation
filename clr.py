from builtins import range
import numpy as np
from sklearn.base import BaseEstimator, clone
# from scipy.spatial.distance import cdist
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from sklearn.utils.validation import check_is_fitted
from ground_truth import PopulationSimulator, SegmentEstimate
from utils import assign_trained_customers_to_segments, estimate_segment_parameters, assign_new_customers_to_segments, evaluate_on_validation, build_design_matrix


# The expected shape of y is (N,) not (N, 1)!!!!!

class CLRpRegressor(BaseEstimator):
    def __init__(self, num_planes, kmeans_coef, clr_lr=None, max_iter=5, num_tries=8, clf=None,
                 include_interactions=False, n_features=None, action_num=None):
        self.num_planes = num_planes
        self.kmeans_coef = kmeans_coef
        self.num_tries = num_tries
        self.clr_lr = clr_lr
        self.max_iter = max_iter
        self.include_interactions = include_interactions
        # Used to strip treatment / interaction columns: X_D = [1, x(d), D dummies, ...]
        self.n_features = n_features
        self.action_num = action_num

        if clf is None:
            self.clf = RandomForestClassifier(n_estimators=10)
        else:
            self.clf = clf

    def fit(self, X_D, y, seed=None):
        if seed is not None:
            np.random.seed(seed)

        self.cluster_labels, self.models, _, _ = best_clr(
                    X_D, y, k=self.num_planes, kmeans_coef=self.kmeans_coef,
                    max_iter=self.max_iter, num_tries=self.num_tries,
                    lr=self.clr_lr,
        )
        
        if np.unique(self.cluster_labels).shape[0] == 1: # self.clf needs at least 2 classes
            self.cluster_labels[0] = 1 if self.cluster_labels[0] == 0 else 0
        
        # Assignment RF is fit on covariates only (no treatment / interactions).
        # X_D = [intercept, x(d), 1{D=1}..1{D=K-1}, (optional x*dummies)]
        if self.n_features is None:
            raise ValueError("CLRpRegressor.n_features must be set to strip treatment columns.")
        d = int(self.n_features)
        X_for_clf = X_D[:, 1 : 1 + d]
        self.clf.fit(X_for_clf, self.cluster_labels)

    def predict(self, X_only):
        check_is_fitted(self, ['cluster_labels', 'models'])
        test_labels = self.clf.predict(X_only)
        return test_labels

def best_clr(X_D, y, k, num_tries=5, **kwargs):
    clr_func = clr
    best_obj = np.inf
    for i in range(num_tries):
        cluster_labels, models, weights, obj = clr_func(X_D, y, k, **kwargs)
        if obj < best_obj:
            best_obj = obj
            best_cluster_labels = cluster_labels
            best_models = models
            best_weights = weights
    return best_cluster_labels, best_models, best_weights, best_obj

def clr(X_D, y, k, kmeans_coef, lr=None, max_iter=5, cluster_labels=None):

    if cluster_labels is None:
        cluster_labels = np.random.choice(k, size=X_D.shape[0])

    if lr is None:
        lr = LogisticRegression(fit_intercept=False, max_iter=2000, solver='lbfgs')

    models = [clone(lr) for _ in range(k)]
    # Init to inf: empty / unfitted clusters must never win argmin
    scores = np.full((X_D.shape[0], k), np.inf)
    y_float = y.astype(float)

    for _ in range(max_iter):

        # ── Fit + Score per cluster (combined to avoid stale-model issues) ──
        for cl_idx in range(k):
            mask = cluster_labels == cl_idx

            # Empty cluster: keep scores at inf, no fitting
            if np.sum(mask) == 0:
                scores[:, cl_idx] = np.inf
                continue

            y_cl = y[mask].astype(int)

            if len(np.unique(y_cl)) < 2:
                # Single-class cluster: use a near-constant probability.
                # This gives the lowest possible CE for same-class points,
                # making the cluster properly attractive for them.
                const_p = (1 - 1e-6) if y_cl[0] == 1 else 1e-6
                p_hat = np.full(X_D.shape[0], const_p)
            else:
                models[cl_idx] = clone(lr)
                models[cl_idx].fit(X_D[mask], y_cl)
                try:
                    p_hat = np.clip(
                        models[cl_idx].predict_proba(X_D)[:, 1], 1e-9, 1 - 1e-9
                    )
                except Exception:
                    p_hat = np.full(X_D.shape[0], 0.5)
            scores[:, cl_idx] = -(
                y_float * np.log(p_hat) + (1 - y_float) * np.log(1 - p_hat)
            )

            if kmeans_coef > 0:
                center = np.mean(X_D[mask], axis=0)
                scores[:, cl_idx] += kmeans_coef * np.sum(
                    np.square(X_D - center), axis=1
                )

        cluster_labels_prev = cluster_labels.copy()
        cluster_labels = np.argmin(scores, axis=1)

        if np.allclose(cluster_labels, cluster_labels_prev):
            break

    obj = np.mean(scores[np.arange(X_D.shape[0]), cluster_labels])

    weights = (cluster_labels == np.arange(k)[:, np.newaxis]).sum(axis=1).astype(float)
    weights /= np.sum(weights)
    return cluster_labels, models, weights, obj

def bic_score(X_D, y, cluster_labels, models):
    n, d = X_D.shape
    k = len(models)

    # Number of parameters: k regression models (d params each) + (k-1) mixture weights
    p = k * d + (k - 1)

    # Bernoulli log-likelihood using predicted probabilities (not hard labels)
    y_int = y.astype(int)
    log_lik = 0.0
    for cl_idx in range(k):
        mask = (cluster_labels == cl_idx)
        if np.sum(mask) == 0:
            continue
        try:
            p_hat = np.clip(models[cl_idx].predict_proba(X_D[mask])[:, 1], 1e-9, 1 - 1e-9)
        except Exception:
            p_hat = np.full(np.sum(mask), 0.5)
        log_lik += np.sum(y_int[mask] * np.log(p_hat) + (1 - y_int[mask]) * np.log(1 - p_hat))
    logL = log_lik

    BIC = -2 * logL + p * np.log(n)
    return BIC


def CLR_segment_and_estimate(pop: PopulationSimulator, n_segments: int, x_mat, D_vec, y_vec, kmeans_coef, num_tries, algo, include_interactions, action_method: str, random_state=None):
    
    
    # Important: y_vec needs to be shape (N,) not (N, 1).
    y_vec = y_vec.ravel()
    X_D = build_design_matrix(
        x_mat, D_vec, include_interactions, action_num=pop.action_num)
    CLR = CLRpRegressor(
        num_planes=n_segments,
        kmeans_coef=kmeans_coef,
        num_tries=num_tries,
        include_interactions=include_interactions,
        n_features=x_mat.shape[1],
        action_num=pop.action_num,
    )
    CLR.fit(X_D, y_vec, seed=random_state)
    clr_labels = CLR.cluster_labels
    bic = bic_score(X_D, y_vec, CLR.cluster_labels, CLR.models)
    
    # Assign each customer to estimated segment
    pop.est_segments_list[f"{algo}"] = []  # Reset!!!!

    # Estimate triplets via OLS per segment
    for m in range(n_segments):
        idx_m = np.where(clr_labels == m)[0]
        
        if len(idx_m) == 0:
            est_action = int(np.random.randint(0, pop.action_num))
            est_tau = 0.0
            est_seg = SegmentEstimate(est_tau, est_action, segment_id=m)
            pop.est_segments_list[f"{algo}"].append(est_seg)
            continue

        est_tau, est_action = estimate_segment_parameters(
            x_mat, D_vec, y_vec,
            method=action_method,
            gamma=pop.gamma_train,
            indices=idx_m,
            action_num=pop.action_num,
            include_interactions=include_interactions,
        )
        est_seg = SegmentEstimate(est_tau, est_action, segment_id=m)
        pop.est_segments_list[f"{algo}"].append(est_seg)

    # Link each customer to estimated segment
    assign_trained_customers_to_segments(pop, clr_labels, f"{algo}")
    
    model_selection = algo.split("-")[-1]
    if model_selection == "standard":
        return bic, CLR
    elif model_selection == "da":
        assign_new_customers_to_segments(pop, pop.val_customers, CLR, algo)
        Gamma_val = pop.gamma_val
        DA_score = evaluate_on_validation(pop, algo=algo, Gamma_val=Gamma_val)
        return  DA_score, CLR
    else:
        raise ValueError("model_selection must be either 'standard' or 'da'")

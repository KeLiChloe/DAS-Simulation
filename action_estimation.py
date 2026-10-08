"""
Node- / segment-level action recommendation for segmentation methods and DAST splits.

Methods
-------
diff_in_means : argmax_a mean(Y | D=a) within the node
                Binary: a = 1{mean(Y|D=1) >= mean(Y|D=0)}
gamma         : argmax_a mean(Gamma[:, a]) over customers in the node
                (Gamma = doubly-robust score matrix, shape (N, K))
logistic      : fit P(Y=1|x,D) = sigma(alpha + beta'x + theta_1*1{D=1} + ... + theta_{K-1}*1{D=K-1})
                recommend argmax_a theta_a  (theta_0 := 0, reference action 0)
                Binary: a = 1{theta_hat >= 0}  — identical to paper eq. (7)
"""

from __future__ import annotations

import warnings

import numpy as np

ACTION_METHODS = ("diff_in_means", "gamma", "logistic")
_MISSING_ACTION = 404


def normalize_action_method(method: str) -> str:
    key = str(method).strip().lower().replace("-", "_")
    aliases = {
        "diffs_in_means": "diff_in_means",
        "diff_in_mean": "diff_in_means",
        "means": "diff_in_means",
        "dr": "gamma",
        "dr_score": "gamma",
        "dr_scores": "gamma",
    }
    key = aliases.get(key, key)
    if key not in ACTION_METHODS:
        raise ValueError(
            f"Unknown action_method {method!r}. Choose from {ACTION_METHODS}."
        )
    return key


def _as_segment_arrays(
    X, D, Y, indices=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    Y = np.ravel(np.asarray(Y, dtype=float))
    D = np.ravel(np.asarray(D)).astype(int)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if indices is not None:
        idx = np.asarray(indices, dtype=int)
        X, D, Y = X[idx], D[idx], Y[idx]
    return X, D, Y


def _action_num_from_data(D: np.ndarray, action_num: int | None) -> int:
    if action_num is not None and action_num >= 1:
        return int(action_num)
    present = np.unique(D)
    if len(present) == 0:
        return 2
    return int(max(present) + 1)


def _diff_in_means_action(D: np.ndarray, Y: np.ndarray, K: int) -> int | None:
    action_means: dict[int, float] = {}
    for a in range(K):
        mask = D == a
        if np.any(mask):
            action_means[a] = float(np.mean(Y[mask]))
    if not action_means:
        return None
    return int(max(action_means, key=action_means.get))


def _gamma_action(
    gamma: np.ndarray, indices, K: int,
) -> int | None:
    G = np.asarray(gamma, dtype=float)
    idx = np.asarray(indices, dtype=int)
    if G.ndim != 2 or G.shape[1] < K:
        return None
    means = [float(np.mean(G[idx, a])) for a in range(K)]
    if all(np.isnan(m) for m in means):
        return None
    return int(np.argmax(means))


def _build_logistic_design(
    X: np.ndarray, D: np.ndarray, K: int,
) -> np.ndarray:
    """Columns: intercept, x, 1{D=1}, ..., 1{D=K-1} (no interaction terms)."""
    n, d = X.shape
    blocks = [np.ones((n, 1)), X]
    dummies = []
    for a in range(1, K):
        dummies.append((D == a).astype(float).reshape(-1, 1))
    if dummies:
        blocks.append(np.hstack(dummies))
    return np.hstack(blocks)


def _logistic_action(
    X: np.ndarray,
    D: np.ndarray,
    Y: np.ndarray,
    K: int,
) -> int | None:
    n = len(Y)
    if n < 2:
        return None
    if len(np.unique(Y)) < 2:
        return None

    Z = _build_logistic_design(X, D, K)
    d = X.shape[1]
    # Coefficient layout: intercept, x (d cols), theta_1..theta_{K-1}
    n_theta = max(K - 1, 0)

    try:
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(
            penalty=None,
            solver="lbfgs",
            max_iter=500,
            fit_intercept=False,
        )
        clf.fit(Z, Y.astype(int))
        coef = clf.coef_.ravel()
    except Exception:
        warnings.warn(
            "Logistic node action fit failed; falling back to diff_in_means.",
            stacklevel=3,
        )
        return _diff_in_means_action(D, Y, K)

    # theta_0 := 0 (reference action); theta_1..theta_{K-1} from coef
    off = 1 + d
    theta_rest = coef[off : off + n_theta] if n_theta else np.array([])
    full_theta = np.concatenate([[0.0], theta_rest])  # shape (K,)
    return int(np.argmax(full_theta))


def recommend_segment_action(
    X,
    D,
    Y,
    *,
    method: str,
    gamma: np.ndarray | None = None,
    indices=None,
    action_num: int | None = None,
    include_interactions: bool = False,   # kept for API compatibility; logistic always fits without interactions
) -> int:
    """
    Recommend a single action for a node / segment.

    Returns 404 if no valid recommendation can be formed.
    """
    method = normalize_action_method(method)
    X, D, Y = _as_segment_arrays(X, D, Y, indices)
    if len(Y) == 0:
        return _MISSING_ACTION

    K = _action_num_from_data(D, action_num)

    if method == "gamma":
        if gamma is None:
            raise ValueError("action_method='gamma' requires gamma scores.")
        idx = np.arange(len(Y)) if indices is None else np.asarray(indices, dtype=int)
        act = _gamma_action(gamma, idx, K)
    elif method == "logistic":
        act = _logistic_action(X, D, Y, K)
    else:
        act = _diff_in_means_action(D, Y, K)

    if act is None:
        return _MISSING_ACTION
    return int(act)


def estimate_segment_tau(
    D: np.ndarray,
    Y: np.ndarray,
    est_action: int,
    *,
    method: str,
    gamma: np.ndarray | None = None,
    indices=None,
    action_num: int | None = None,
) -> float:
    """Segment-level effect contrast for storage / debugging (relative to action 0)."""
    if est_action == _MISSING_ACTION:
        return 0.0

    _, D, Y = _as_segment_arrays(np.zeros((len(Y), 1)), D, Y, indices)
    K = _action_num_from_data(D, action_num)
    method = normalize_action_method(method)

    if method == "gamma":
        if gamma is None:
            return 0.0
        idx = np.arange(len(Y)) if indices is None else np.asarray(indices, dtype=int)
        G = np.asarray(gamma, dtype=float)[idx]
        if G.ndim == 2 and est_action < G.shape[1]:
            return float(np.mean(G[:, est_action]) - np.mean(G[:, 0]))
        return 0.0

    means = {}
    for a in range(K):
        mask = D == a
        if np.any(mask):
            means[a] = float(np.mean(Y[mask]))
    if 0 in means and est_action in means:
        return means[est_action] - means[0]
    baseline = 0 if 0 in means else min(means.keys(), default=0)
    if est_action in means and baseline in means:
        return means[est_action] - means[baseline]
    return 0.0


def estimate_segment_parameters(
    X,
    D,
    Y,
    *,
    method: str,
    gamma: np.ndarray | None = None,
    indices=None,
    action_num: int | None = None,
    include_interactions: bool = False,   # kept for API compatibility; logistic always fits without interactions
) -> tuple[float, int]:
    """
    Estimate (est_tau, est_action) for a segment / leaf.

    Backward-compatible positional call:
        estimate_segment_parameters(x_m, D_m, y_m)
    still works when passed as first three arguments only.
    """
    est_action = recommend_segment_action(
        X, D, Y,
        method=method,
        gamma=gamma,
        indices=indices,
        action_num=action_num,
        include_interactions=include_interactions,
    )
    est_tau = estimate_segment_tau(
        D, Y, est_action,
        method=method,
        gamma=gamma,
        indices=indices,
        action_num=action_num,
    )
    return est_tau, est_action

"""
Policy Tree individual-level policy recommendations via R's policytree package.

Workflow (aligned with causal_forest / meta-learners):
1. Fit a GRF forest on pilot data
2. Compute doubly-robust scores (Gamma)
3. Fit a shallow policy tree on (X, Gamma)
4. Predict recommended action per implementation customer (type="action.id")
"""

import numpy as np

_r_initialized = False
_ro = None
_grf = None
_policytree = None
_localconverter = None
_default_converter = None
_numpy2ri = None


def _init_r():
    global _r_initialized, _ro, _grf, _policytree
    global _localconverter, _default_converter, _numpy2ri
    if _r_initialized:
        return

    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri, default_converter
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects.packages import importr

    _ro = ro
    _localconverter = localconverter
    _default_converter = default_converter
    _numpy2ri = numpy2ri
    _grf = importr("grf")
    _policytree = importr("policytree")
    _r_initialized = True
    print("[policy_tree] R packages loaded (grf, policytree)", flush=True)


def _gamma_column_actions(gamma_r):
    """
    Action label for each column of Gamma, in R column order.

    policytree::predict(..., type='action.id') returns 1..ncol(Gamma), i.e. a
    1-based column index — NOT the raw treatment code from D_vec.
    """
    n_cols = int(np.asarray(_ro.r("ncol")(gamma_r)).item())
    if n_cols < 1:
        raise ValueError("Gamma matrix has no columns")

    colnames = _ro.r("colnames")(gamma_r)
    if _ro.r["is.null"](colnames)[0]:
        return np.arange(n_cols, dtype=int)

    labels = []
    for name in list(colnames):
        # GRF/policytree use "0","1","2" for numeric arms; parse to int.
        labels.append(int(float(str(name))))
    return np.asarray(labels, dtype=int)


def policy_tree_predict(implement_customers, x_mat, D_vec, y_vec, depth=2):
    """
    Fit a policy tree and recommend an action for each implementation customer.

    Parameters
    ----------
    implement_customers : list
        Implementation customers (only .x is used for prediction).
    x_mat, D_vec, y_vec : array-like
        Pilot/training data used to fit the forest and policy tree.
    depth : int
        Maximum tree depth passed to R's policy_tree().

    Returns
    -------
    recommended_idx : ndarray, shape (n_impl,)
        0-based column index into action_identity (same convention as
        np.argmax(..., axis=1) in meta-learners).
    action_identity : ndarray, shape (n_actions,)
        action_identity[j] is the treatment label for Gamma column j+1.
        main.py uses: act_id[recommended_idx[i]].
    """
    _init_r()

    x_mat = np.asarray(x_mat)
    D_vec = np.asarray(D_vec)
    y_vec = np.asarray(y_vec)
    unique_actions = np.unique(D_vec)
    if len(unique_actions) < 2:
        raise ValueError(
            f"policy_tree needs >= 2 actions in training data, got {unique_actions}"
        )

    X_impl = np.array([cust.x for cust in implement_customers])
    n_train, n_impl = x_mat.shape[0], X_impl.shape[0]
    print(
        f"[policy_tree] start: n_train={n_train}, n_impl={n_impl}, "
        f"d={x_mat.shape[1]}, depth={depth}, actions={unique_actions.tolist()}",
        flush=True,
    )

    with _localconverter(_default_converter + _numpy2ri.converter):
        X_r = _ro.conversion.py2rpy(x_mat)
        y_r = _ro.conversion.py2rpy(y_vec)
        D_r = _ro.conversion.py2rpy(D_vec)
        X_impl_r = _ro.conversion.py2rpy(X_impl)

        print("[policy_tree] fitting multi_arm_causal_forest ...", flush=True)
        forest = _grf.multi_arm_causal_forest(X_r, y_r, D_r)

        print("[policy_tree] computing double_robust_scores (Gamma) ...", flush=True)
        gamma_r = _policytree.double_robust_scores(forest)
        action_identity = _gamma_column_actions(gamma_r)
        n_gamma = int(np.asarray(_ro.r("nrow")(gamma_r)).item())
        print(
            f"[policy_tree] Gamma: {n_gamma} x {len(action_identity)}, "
            f"column actions (R order) = {action_identity.tolist()}",
            flush=True,
        )

        print(f"[policy_tree] fitting policy_tree(depth={depth}) ...", flush=True)
        tree = _policytree.policy_tree(X_r, gamma_r, depth=depth)

        print("[policy_tree] predicting actions on implementation set ...", flush=True)
        action_r = _policytree.predict_policy_tree(tree, X_impl_r, type="action.id")
        action_ids_raw = np.asarray(_ro.conversion.rpy2py(action_r), dtype=int)

    # R: 1..n_cols (column index)  →  Python: 0..n_cols-1 (for act_id[·])
    recommended_idx = action_ids_raw - 1
    recommended_actions = action_identity[recommended_idx]

    n_cols = len(action_identity)
    if np.any(recommended_idx < 0) or np.any(recommended_idx >= n_cols):
        raise ValueError(
            f"R action.id out of range [1, {n_cols}]; "
            f"got min={action_ids_raw.min()}, max={action_ids_raw.max()}"
        )

    unique_rec, counts = np.unique(recommended_actions, return_counts=True)
    dist = ", ".join(f"a={a}:{c}" for a, c in zip(unique_rec, counts))
    print(
        f"[policy_tree] done: R action.id in [{action_ids_raw.min()}, {action_ids_raw.max()}], "
        f"recommended actions {{{dist}}}",
        flush=True,
    )

    return recommended_idx.astype(int), action_identity

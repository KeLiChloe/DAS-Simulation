"""
Policy Tree individual-level policy recommendations via R's policytree package.

Workflow (aligned with causal_forest / meta-learners):
1. Fit a GRF forest on pilot data
2. Compute doubly-robust scores (Gamma)
3. Fit a shallow policy tree on (X, Gamma)
4. Predict recommended action per implementation customer (type="action.id")

NOTE: we do NOT use a numpy2ri localconverter block.  The numpy2ri converter
round-trips rpy2 R objects back through numpy, stripping class attributes and
turning R numeric vectors into R 'array' objects.  Instead every R object is
built explicitly from rpy2 primitive constructors so GRF sees the correct types:
  X → R matrix   (class: 'matrix' 'array')   — grf accepts this
  Y → R numeric  (class: 'numeric')
  W → R factor   (class: 'factor')
"""

import numpy as np

_r_initialized = False
_ro = None
_grf = None
_policytree = None


def _init_r():
    global _r_initialized, _ro, _grf, _policytree
    if _r_initialized:
        return

    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr

    _ro = ro
    _grf = importr("grf")
    _policytree = importr("policytree")
    _r_initialized = True
    print("[policy_tree] R packages loaded (grf, policytree)", flush=True)


def _r_class(obj):
    return list(_ro.r("class")(obj))


def _as_r_matrix(arr2d):
    """Convert a C-contiguous float64 (n, p) numpy array to an R matrix."""
    n, p = arr2d.shape
    flat = arr2d.reshape(-1).tolist()          # plain Python list → no numpy2ri
    rvec = _ro.FloatVector(flat)
    return _ro.r.matrix(rvec, nrow=n, ncol=p, byrow=True)


def _as_r_numeric(arr1d):
    """Convert a float64 1-D numpy array to an R numeric vector."""
    return _ro.FloatVector(arr1d.tolist())     # class 'numeric' guaranteed


def _as_r_factor(arr1d_int):
    """Convert an int 1-D numpy array to an R factor."""
    ivec = _ro.IntVector(arr1d_int.tolist())
    return _ro.r["as.factor"](ivec)            # class 'factor'


def _gamma_column_actions(gamma_r):
    """
    Return the action label for each column of Gamma, in R column order.
    policytree predict(type='action.id') returns 1-based column indices.
    """
    n_cols = int(list(_ro.r("ncol")(gamma_r))[0])
    if n_cols < 1:
        raise ValueError("Gamma matrix has no columns")

    colnames = _ro.r("colnames")(gamma_r)
    if list(_ro.r["is.null"](colnames))[0]:
        return np.arange(n_cols, dtype=int)

    return np.array([int(float(str(s))) for s in list(colnames)], dtype=int)


def policy_tree_predict(implement_customers, x_mat, D_vec, y_vec, depth=2):
    """
    Fit a policy tree and recommend an action for each implementation customer.

    Returns
    -------
    recommended_idx : ndarray, shape (n_impl,)
        0-based index into action_identity for each customer.
    action_identity : ndarray, shape (n_actions,)
        action_identity[j] = actual treatment label for Gamma column j+1.
        main.py: act_id[recommended_idx[i]]
    """
    _init_r()

    # ── prepare numpy arrays ──────────────────────────────────────────────────
    x_mat = np.asarray(x_mat, dtype=np.float64)
    if x_mat.ndim == 1:
        x_mat = x_mat.reshape(-1, 1)
    D_vec = np.asarray(D_vec).reshape(-1).astype(int)
    y_vec = np.asarray(y_vec, dtype=np.float64).reshape(-1)

    unique_actions = np.unique(D_vec)
    if len(unique_actions) < 2:
        raise ValueError(
            f"policy_tree needs >= 2 actions in training data, got {unique_actions}"
        )

    X_impl = np.asarray([cust.x for cust in implement_customers], dtype=np.float64)
    if X_impl.ndim == 1:
        X_impl = X_impl.reshape(-1, 1)

    n_train, n_impl = x_mat.shape[0], X_impl.shape[0]
    print(
        f"[policy_tree] start: n_train={n_train}, n_impl={n_impl}, "
        f"d={x_mat.shape[1]}, depth={depth}, actions={unique_actions.tolist()}",
        flush=True,
    )

    # ── build R objects explicitly (no localconverter) ────────────────────────
    X_r      = _as_r_matrix(x_mat)
    y_r      = _as_r_numeric(y_vec)
    D_r      = _as_r_factor(D_vec)
    X_impl_r = _as_r_matrix(X_impl)

    print(
        f"[policy_tree] R classes: X={_r_class(X_r)}, "
        f"Y={_r_class(y_r)}, W={_r_class(D_r)}",
        flush=True,
    )

    # ── GRF + policytree ─────────────────────────────────────────────────────
    print("[policy_tree] fitting multi_arm_causal_forest ...", flush=True)
    forest = _grf.multi_arm_causal_forest(X_r, y_r, D_r)

    print("[policy_tree] computing double_robust_scores (Gamma) ...", flush=True)
    gamma_r = _policytree.double_robust_scores(forest)
    action_identity = _gamma_column_actions(gamma_r)
    n_gamma = int(list(_ro.r("nrow")(gamma_r))[0])
    print(
        f"[policy_tree] Gamma: {n_gamma} x {len(action_identity)}, "
        f"column actions = {action_identity.tolist()}",
        flush=True,
    )

    print(f"[policy_tree] fitting policy_tree(depth={depth}) ...", flush=True)
    tree = _policytree.policy_tree(X_r, gamma_r, depth=depth)

    print("[policy_tree] predicting on implementation set ...", flush=True)
    action_r = _policytree.predict_policy_tree(tree, X_impl_r, type="action.id")
    action_ids_raw = np.array(list(action_r), dtype=int)   # 1-based column index

    # ── index mapping ─────────────────────────────────────────────────────────
    recommended_idx = action_ids_raw - 1   # → 0-based for act_id[·]

    n_cols = len(action_identity)
    if recommended_idx.min() < 0 or recommended_idx.max() >= n_cols:
        raise ValueError(
            f"R action.id out of range [1, {n_cols}]; "
            f"got [{action_ids_raw.min()}, {action_ids_raw.max()}]"
        )

    recommended_actions = action_identity[recommended_idx]
    unique_rec, counts = np.unique(recommended_actions, return_counts=True)
    dist = ", ".join(f"a={a}:{c}" for a, c in zip(unique_rec, counts))
    print(
        f"[policy_tree] done: recommended actions {{{dist}}}",
        flush=True,
    )

    return recommended_idx.astype(int), action_identity

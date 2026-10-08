"""M sweep / pick / retrain helpers shared by discrete demo scripts."""

from __future__ import annotations

import random
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from cv_utils import CV_FOLDS_DEFAULT, CV_M_ALGOS, attach_fold, make_m_selection_folds
from dast import DAST_segment_and_estimate
from gmm import GMM_segment_and_estimate
from kmeans import KMeans_segment_and_estimate
from mst import MST_segment_and_estimate
from utils import pick_M_for_algo

DEMO_SEG_ALGOS = (
    "dast",
    "mst",
    "kmeans-standard",
    "kmeans-da",
    "gmm-standard",
    "gmm-da",
)

DEMO_META_ALGOS = (
    "t_learner",
    "s_learner",
    "x_learner",
    "dr_learner",
    "causal_forest",
    "policy_tree",
)

DEMO_ALL_ALGOS = DEMO_SEG_ALGOS + DEMO_META_ALGOS


def resolve_seed(seed: Optional[int]) -> int:
    """Use explicit seed, or draw one at random (same range as main.py)."""
    if seed is None:
        seed = random.randint(0, 100_000)
        print(f"Using random seed: {seed}")
    return int(seed)


def m_range_for_k(K: int) -> List[int]:
    """Same M grid as main.py: range(max(2, K-3), K+4)."""
    return list(range(max(2, K - 3), K + 4))


def m_range_elbow_for_k(K: int) -> List[int]:
    """Elbow curve includes M=1 (needed so M=2 can be selected)."""
    return list(range(1, K + 4))


def pick_M_elbow_inertia(Ms: List[int], inertias: List[float]) -> int:
    """Pick M by max distance from (M, inertia) to the chord joining endpoints."""
    if len(Ms) != len(inertias):
        raise ValueError("Ms and inertias must have the same length.")
    if len(Ms) == 1:
        return int(Ms[0])
    xs = np.asarray(Ms, dtype=float)
    ys = np.asarray(inertias, dtype=float)
    p1 = np.array([xs[0], ys[0], 0.0])
    p2 = np.array([xs[-1], ys[-1], 0.0])
    line = p2 - p1
    line_norm = float(np.linalg.norm(line))
    if line_norm < 1e-12:
        return int(Ms[0])
    dists = [
        float(np.linalg.norm(np.cross(line, p1 - np.array([x, y, 0.0]))) / line_norm)
        for x, y in zip(xs, ys)
    ]
    return int(Ms[int(np.argmax(dists))])


def sweep_train_frac(algo: str, train_frac: float, train_frac_retrain: float) -> float:
    """Train fraction during M sweep (matches main.py split rules for non-CV algos)."""
    if algo in ("kmeans-standard", "gmm-standard", "clr-standard"):
        return train_frac_retrain
    return train_frac


def _mst_depth(M: int) -> int:
    if M <= 2:
        return 1
    if M <= 4:
        return 2
    if M <= 8:
        return 3
    return 4


def run_one_fit(
    pop,
    algo: str,
    M: int,
    x_mat: np.ndarray,
    D_vec: np.ndarray,
    y_vec: np.ndarray,
    *,
    include_interactions: bool,
    seed: int,
    min_leaf_size: int,
    use_hybrid_method: bool,
    action_method: str,
    kmeans_coef: float = 0.15,
) -> Dict[str, Any]:
    """Fit one candidate M; return score + labels on current train customers."""
    if algo in ("kmeans-standard", "kmeans-da"):
        score, model = KMeans_segment_and_estimate(
            pop,
            M,
            x_mat,
            D_vec,
            y_vec,
            algo,
            include_interactions=include_interactions,
            random_state=seed,
            action_method=action_method,
        )
        labels = np.array([c.est_segment[algo].segment_id for c in pop.train_customers])
        actions = np.array([c.est_segment[algo].est_action for c in pop.train_customers])
        return {
            "score": score,
            "model": model,
            "tree": None,
            "segment_dict": None,
            "labels": labels,
            "actions": actions,
        }

    if algo in ("gmm-standard", "gmm-da"):
        score, model = GMM_segment_and_estimate(
            pop,
            M,
            x_mat,
            D_vec,
            y_vec,
            algo,
            include_interactions=include_interactions,
            random_state=seed,
            action_method=action_method,
        )
        labels = np.array([c.est_segment[algo].segment_id for c in pop.train_customers])
        actions = np.array([c.est_segment[algo].est_action for c in pop.train_customers])
        return {
            "score": score,
            "model": model,
            "tree": None,
            "segment_dict": None,
            "labels": labels,
            "actions": actions,
        }

    if algo == "dast":
        tree, val_score, segment_dict = DAST_segment_and_estimate(
            pop,
            n_segments=M,
            min_leaf_size=min_leaf_size,
            algo=algo,
            use_hybrid_method=use_hybrid_method,
            debug=False,
            action_method=action_method,
            include_interactions=include_interactions,
        )
        labels = np.array([c.est_segment[algo].segment_id for c in pop.train_customers])
        actions = np.array([c.est_segment[algo].est_action for c in pop.train_customers])
        return {
            "score": val_score,
            "model": None,
            "tree": tree,
            "segment_dict": segment_dict,
            "labels": labels,
            "actions": actions,
        }

    if algo == "mst":
        tree, val_score, segment_dict = MST_segment_and_estimate(
            pop,
            n_segments=M,
            max_depth=_mst_depth(M),
            min_leaf_size=min_leaf_size,
            epsilon=1e-2,
            algo=algo,
            include_interactions=include_interactions,
            action_method=action_method,
        )
        labels = np.array([c.est_segment[algo].segment_id for c in pop.train_customers])
        actions = np.array([c.est_segment[algo].est_action for c in pop.train_customers])
        return {
            "score": val_score,
            "model": None,
            "tree": tree,
            "segment_dict": segment_dict,
            "labels": labels,
            "actions": actions,
        }

    raise ValueError(f"Unsupported demo algorithm '{algo}'.")


def sweep_pick_and_retrain(
    pop,
    algo: str,
    K: int,
    apply_split: Callable[[float], None],
    get_arrays: Callable[[], tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    include_interactions: bool,
    seed: int,
    min_leaf_size: int,
    use_hybrid_method: bool,
    train_frac: float,
    train_frac_retrain: float,
    action_method: str,
    cv_folds: int = CV_FOLDS_DEFAULT,
    kmeans_coef: float = 0.15,
) -> Dict[str, Any]:
    """
    M sweep → pick_M_for_algo → final retrain on train_frac_retrain pilot data.

    For dast / *-da: K-fold CV of held-out DR value (same as main.py).
    For mst: single holdout val DR (train_frac).
    For kmeans-standard: elbow on inertia with M starting at 1.
    For other *-standard: silhouette/BIC on full (retrain) pilot split.
    """
    M_range = m_range_for_k(K)
    use_cv = algo in CV_M_ALGOS
    n_pilot = len(pop.pilot_customers)
    d = int(np.asarray(pop.pilot_customers[0].x).shape[0])

    fit_kw = dict(
        include_interactions=include_interactions,
        seed=seed,
        min_leaf_size=min_leaf_size,
        use_hybrid_method=use_hybrid_method,
        action_method=action_method,
        kmeans_coef=kmeans_coef,
    )

    rows = []
    if algo == "kmeans-standard":
        # Elbow from M=1 (chord on inertia); also record silhouette for M>=2.
        sweep_frac = sweep_train_frac(algo, train_frac, train_frac_retrain)
        apply_split(sweep_frac)
        x_mat, _, _ = get_arrays()
        M_elbow = m_range_elbow_for_k(K)
        for M in M_elbow:
            km = KMeans(n_clusters=M, random_state=seed).fit(x_mat)
            row = {"M": M, f"{algo}_val": float(km.inertia_)}
            if M >= 2:
                row["silhouette"] = float(silhouette_score(x_mat, km.labels_))
            else:
                row["silhouette"] = np.nan
            rows.append(row)
        df_M = pd.DataFrame(rows)
        picked_M = pick_M_elbow_inertia(
            df_M["M"].astype(int).tolist(),
            df_M[f"{algo}_val"].astype(float).tolist(),
        )
        # Silhouette pick (max over M>=2) — for reporting / dual plot marker
        sil_pick = int(df_M.loc[df_M["M"] >= 2].sort_values("silhouette", ascending=False).iloc[0]["M"])
    elif use_cv:
        folds = make_m_selection_folds(n_pilot, cv_folds, seed)
        for M in M_range:
            fold_scores = []
            for train_idx, val_idx in folds:
                fold_sp = attach_fold(pop, train_idx, val_idx, d)
                fit = run_one_fit(
                    pop,
                    algo,
                    M,
                    fold_sp["x_mat_tr"],
                    fold_sp["D_vec_tr"],
                    fold_sp["y_vec_tr"],
                    **fit_kw,
                )
                if fit["score"] is not None:
                    fold_scores.append(float(fit["score"]))
            if not fold_scores:
                raise RuntimeError(f"{algo} M={M}: all CV folds failed to produce a score.")
            rows.append({"M": M, f"{algo}_val": float(np.mean(fold_scores))})
        sweep_frac = f"cv_folds={cv_folds}"
        df_M = pd.DataFrame(rows)
        val_col = f"{algo}_val"
        df_M = df_M.dropna(subset=[val_col]).reset_index(drop=True)
        if len(df_M) == 0:
            raise RuntimeError(f"{algo}: no valid M candidates after sweep.")
        picked_M = int(pick_M_for_algo(algo, df_M)[f"{algo}_picked_M"])
    else:
        sweep_frac = sweep_train_frac(algo, train_frac, train_frac_retrain)
        apply_split(sweep_frac)
        x_mat, D_vec, y_vec = get_arrays()
        for M in M_range:
            fit = run_one_fit(pop, algo, M, x_mat, D_vec, y_vec, **fit_kw)
            rows.append({"M": M, f"{algo}_val": fit["score"]})
        df_M = pd.DataFrame(rows)
        val_col = f"{algo}_val"
        df_M = df_M.dropna(subset=[val_col]).reset_index(drop=True)
        if len(df_M) == 0:
            raise RuntimeError(f"{algo}: no valid M candidates after sweep.")
        picked_M = int(pick_M_for_algo(algo, df_M)[f"{algo}_picked_M"])

    apply_split(train_frac_retrain)
    x_mat, D_vec, y_vec = get_arrays()
    final = run_one_fit(pop, algo, picked_M, x_mat, D_vec, y_vec, **fit_kw)
    final["picked_M"] = picked_M
    final["m_sweep"] = df_M
    final["train_frac_sweep"] = sweep_frac
    final["train_frac_retrain"] = train_frac_retrain
    final["cv_folds"] = cv_folds if use_cv else None
    if algo == "kmeans-standard":
        final["m_selection"] = "elbow_inertia_from_M1"
        final["silhouette_picked_M"] = sil_pick
    return final

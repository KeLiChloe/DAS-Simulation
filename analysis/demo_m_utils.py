"""M sweep / pick / retrain helpers shared by run_demo_{discrete,continuous}.py."""

from __future__ import annotations

import random
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from dast import DAST_segment_and_estimate
from kmeans import KMeans_segment_and_estimate
from utils import pick_M_for_algo


def resolve_seed(seed: Optional[int]) -> int:
    """Use explicit seed, or draw one at random (same range as main.py)."""
    if seed is None:
        seed = random.randint(0, 100_000)
        print(f"Using random seed: {seed}")
    return int(seed)


def m_range_for_k(K: int) -> List[int]:
    """Same M grid as main.py: range(max(2, K-3), K+4)."""
    return list(range(max(2, K - 3), K + 4))


def sweep_train_frac(algo: str, train_frac: float, train_frac_retrain: float) -> float:
    """Train fraction during M sweep (matches main.py split rules)."""
    if algo in ("kmeans-standard", "gmm-standard", "clr-standard"):
        return train_frac_retrain
    return train_frac


def run_one_fit(
    pop,
    algo: str,
    M: int,
    x_mat: np.ndarray,
    D_vec: np.ndarray,
    y_vec: np.ndarray,
    *,
    include_interactions: bool,
    is_discrete: bool,
    seed: int,
    min_leaf_size: int,
    use_hybrid_method: bool,
) -> Dict[str, Any]:
    if algo == "kmeans-standard":
        score, model = KMeans_segment_and_estimate(
            pop,
            M,
            x_mat,
            D_vec,
            y_vec,
            algo,
            include_interactions=include_interactions,
            random_state=seed,
            is_discrete=is_discrete,
        )
        labels = np.array([c.est_segment[algo].segment_id for c in pop.train_customers])
        return {
            "score": score,
            "model": model,
            "tree": None,
            "segment_dict": None,
            "labels": labels,
        }

    if algo == "dast":
        tree, val_score, segment_dict = DAST_segment_and_estimate(
            pop,
            n_segments=M,
            min_leaf_size=min_leaf_size,
            algo=algo,
            use_hybrid_method=use_hybrid_method,
            debug=False,
        )
        labels = np.array([c.est_segment[algo].segment_id for c in pop.train_customers])
        return {
            "score": val_score,
            "model": None,
            "tree": tree,
            "segment_dict": segment_dict,
            "labels": labels,
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
    is_discrete: bool,
    seed: int,
    min_leaf_size: int,
    use_hybrid_method: bool,
    train_frac: float,
    train_frac_retrain: float,
) -> Dict[str, Any]:
    """
    M sweep → pick_M_for_algo → final retrain on train_frac_retrain pilot data.

    Mirrors the segmentation branch of main._run_one_sim.
    """
    M_range = m_range_for_k(K)
    sweep_frac = sweep_train_frac(algo, train_frac, train_frac_retrain)
    apply_split(sweep_frac)
    x_mat, D_vec, y_vec = get_arrays()

    rows = []
    for M in M_range:
        fit = run_one_fit(
            pop,
            algo,
            M,
            x_mat,
            D_vec,
            y_vec,
            include_interactions=include_interactions,
            is_discrete=is_discrete,
            seed=seed,
            min_leaf_size=min_leaf_size,
            use_hybrid_method=use_hybrid_method,
        )
        rows.append({"M": M, f"{algo}_val": fit["score"]})

    df_M = pd.DataFrame(rows)
    picked_M = int(pick_M_for_algo(algo, df_M)[f"{algo}_picked_M"])

    apply_split(train_frac_retrain)
    x_mat, D_vec, y_vec = get_arrays()
    final = run_one_fit(
        pop,
        algo,
        picked_M,
        x_mat,
        D_vec,
        y_vec,
        include_interactions=include_interactions,
        is_discrete=is_discrete,
        seed=seed,
        min_leaf_size=min_leaf_size,
        use_hybrid_method=use_hybrid_method,
    )
    final["picked_M"] = picked_M
    final["m_sweep"] = df_M
    final["train_frac_sweep"] = sweep_frac
    final["train_frac_retrain"] = train_frac_retrain
    return final

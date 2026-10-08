"""Cross-validation helpers for selecting the number of segments M."""

from __future__ import annotations

import numpy as np

CV_FOLDS_DEFAULT = 5
CV_M_ALGOS = frozenset({"dast", "kmeans-da", "gmm-da", "clr-da"})


def make_m_selection_folds(n_pilot: int, cv_folds: int, seed: int):
    """
    Build (train_idx, val_idx) pairs for M selection.

    cv_folds == 1 : single 80/20 holdout (same convention as former split07).
    cv_folds >= 2 : seeded shuffle then K-fold partitions.
    """
    cv_folds = int(cv_folds)
    if cv_folds < 1:
        raise ValueError(f"cv_folds must be >= 1, got {cv_folds}")
    if n_pilot < 2:
        raise ValueError(f"Need at least 2 pilot customers for M selection, got {n_pilot}")

    if cv_folds == 1:
        split = int(n_pilot * 0.8)
        if split < 1 or split >= n_pilot:
            split = max(1, n_pilot - 1)
        train_idx = np.arange(split)
        val_idx = np.arange(split, n_pilot)
        return [(train_idx, val_idx)]

    if cv_folds > n_pilot:
        raise ValueError(
            f"cv_folds={cv_folds} exceeds n_pilot={n_pilot}; "
            f"use fewer folds or more pilot customers."
        )

    rng = np.random.RandomState(seed)
    order = rng.permutation(n_pilot)
    parts = np.array_split(order, cv_folds)
    folds = []
    for f in range(cv_folds):
        val_idx = np.sort(parts[f])
        train_idx = np.sort(np.concatenate([parts[j] for j in range(cv_folds) if j != f]))
        folds.append((train_idx, val_idx))
    return folds


def attach_fold(pop, train_idx, val_idx, d: int):
    """
    Attach a fold's train/val customers, fit Gamma on fold-train only,
    and return a split cache dict (same keys as main._build_split).
    """
    train_idx = np.asarray(train_idx, dtype=int)
    val_idx = np.asarray(val_idx, dtype=int)

    pop.train_indices = train_idx
    pop.val_indices = val_idx
    pop.train_customers = [pop.pilot_customers[i] for i in train_idx]
    pop.val_customers = [pop.pilot_customers[i] for i in val_idx]

    pop.gamma_train, pop.gamma_val = pop.compute_gamma_scores(
        pop.DR_generation_method, pop.train_customers, pop.val_customers
    )

    return {
        "train_customers": list(pop.train_customers),
        "val_customers": list(pop.val_customers),
        "train_indices": train_idx.copy(),
        "val_indices": val_idx.copy(),
        "gamma_train": pop.gamma_train,
        "gamma_val": pop.gamma_val,
        "x_mat_tr": np.array([c.x for c in pop.train_customers]),
        "D_vec_tr": np.array([c.D_i for c in pop.train_customers]),
        "y_vec_tr": np.array([c.y for c in pop.train_customers]),
        "x_mat_val": (
            np.array([c.x for c in pop.val_customers])
            if pop.val_customers
            else np.zeros((0, d))
        ),
        "D_vec_val": (
            np.array([c.D_i for c in pop.val_customers])
            if pop.val_customers
            else np.zeros(0)
        ),
        "y_vec_val": (
            np.array([c.y for c in pop.val_customers])
            if pop.val_customers
            else np.zeros(0)
        ),
    }

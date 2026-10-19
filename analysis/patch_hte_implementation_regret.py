#!/usr/bin/env python3
"""Add implementation_regret to HTE algo records in experiment pkls.

Regret definition (matches plot_trends_auto.compute_regret):
    (oracle_profits_impl - implementation_profits) / oracle_profits_impl

Only HTE algorithm dicts are updated; all other fields are left unchanged.
"""

from __future__ import annotations

import argparse
import glob
import pickle
import sys

HTE_ALGOS = frozenset(
    [
        "policy_tree",
        "t_learner",
        "s_learner",
        "x_learner",
        "dr_learner",
        "causal_forest",
    ]
)


def patch_pkl(path: str, eps: float = 1e-6) -> tuple[int, int]:
    with open(path, "rb") as f:
        data = pickle.load(f)

    if "oracle_profits_impl" not in data:
        print(f"  skip (no oracle_profits_impl): {path}", file=sys.stderr)
        return 0, 0

    oracle_list = data["oracle_profits_impl"]
    n_patched = 0
    n_skipped = 0

    for algo in HTE_ALGOS:
        runs = data.get(algo)
        if not isinstance(runs, list):
            continue
        for i, rec in enumerate(runs):
            if not isinstance(rec, dict):
                continue
            if i >= len(oracle_list):
                n_skipped += 1
                continue
            impl = rec.get("implementation_profits")
            oracle = oracle_list[i]
            if impl is None or oracle is None:
                n_skipped += 1
                continue
            impl_f = float(impl)
            oracle_f = float(oracle)
            if abs(oracle_f) < eps:
                n_skipped += 1
                continue
            rec["implementation_regret"] = (oracle_f - impl_f) / oracle_f
            n_patched += 1

    with open(path, "wb") as f:
        pickle.dump(data, f)

    return n_patched, n_skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dirs",
        nargs="*",
        default=[
            "exp_oct_2026/discrete/varying_d/set_1_policy_tree_depth_2",
            "exp_oct_2026/discrete/varying_K/set_1_policy_tree_depth_2",
        ],
        help="Directories containing exp_*.pkl files",
    )
    args = parser.parse_args()

    paths = []
    for d in args.dirs:
        paths.extend(sorted(glob.glob(f"{d}/exp_*.pkl")))

    if not paths:
        print("No pkls found.", file=sys.stderr)
        sys.exit(1)

    total_patched = 0
    total_skipped = 0
    for path in paths:
        n_patched, n_skipped = patch_pkl(path)
        total_patched += n_patched
        total_skipped += n_skipped
        print(f"{path}: patched {n_patched} HTE records, skipped {n_skipped}")

    print(f"Done. total patched={total_patched}, skipped={total_skipped}")


if __name__ == "__main__":
    main()

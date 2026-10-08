"""
Illustrative discrete-outcome demo (Bernoulli outcomes, k-means vs DAST).

Run from project root:
    python analysis/run_demo_discrete.py

Edit DEMO_CONFIG below — the only place to set parameters.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "analysis") not in sys.path:
    sys.path.insert(0, str(ROOT / "analysis"))

from demo_m_utils import m_range_for_k, resolve_seed, sweep_pick_and_retrain
from ground_truth import PopulationSimulator
from oracle import oracle_profit_on_customers
from plot_demo_discrete import (
    algo_panel_title,
    build_learned_segment_actions,
    plot_demo_combined,
    plot_demo_ground_truth,
    plot_demo_pilot,
    plot_demo_segmentation,
    plot_kmeans_silhouette_elbow,
)
from plot_style import set_demo_plot_style
from utils import assign_new_customers_to_segments

# =============================================================================
# Demo configuration — edit here only
# =============================================================================

DEMO_CONFIG: Dict[str, Any] = {
    "seed": 11298,  # None → random seed printed at runtime
    "K": 3,
    "d": 1,
    "action_num": 2,
    "n_per_segment": 200,
    "plot_n_per_segment": 50,  # subsample pilot for figures only (None = plot all)
    "n_implement": 6000,
    "target_mahalanobis_sep": 3.0,  # within-cluster covariate noise (same rule as main.py mahal runs)
    "min_leaf_size": 5,
    "train_frac": 0.8,          # pilot train share for M sweep (dast val score)
    "train_frac_retrain": 1.0,  # pilot train share for final fit after M is picked
    "use_hybrid_method": True,
    "action_method": "diff_in_means",  # diff_in_means | gamma | logistic
    "disturb_covariate_noise": 3.0,
    "partial_x": 1.0,
    "DR_generation_method": "lightgbm",
    "disallowed_ball_radius": 0.2,
    "algorithms": ("kmeans-standard", "dast"),
    "out_dir": "figures/demo",
    # Segment colors (demo only) — edit these two keys to change panel colors:
    #   segment_colors_ground_truth → ground-truth panel (b); None uses matplotlib "Set2"
    #   segment_colors_estimated    → K-Means / DAS panels (c, d)
    "segment_colors_ground_truth": [
        "#C77DFF",  # bright purple (💜-like) for True Seg 1
        "#FC8D62",  # Set2 coral (True Seg 2)
        "#5CB85C",  # green (True Seg 3)
    ],
    "segment_colors_estimated": [
        "#0D7377",  # teal
        "#E41A1C",  # red
        "#1A44C0",  # Klein blue, slightly brighter
        "#009E73",  # bluish green
        "#F0E442",  # yellow
        
    ],
    # fixed 1-D cluster centers for the demo plot (set None to randomize)
    "x_mean_vectors": np.array([[5.0], [25.0], [35.0]]),
    # DGP outcome parameters (discrete)
    "param_range": {
        "alpha": None,
        "beta": (-0.05, 0.05),
        "tau": None,
        "delta": (0.0, 0.0),
        "x_mean": (-100.0, 100.0),
        "target_p": (0.05, 0.12),
        "winner_p": (0.55, 0.80),
    },
}


class DiscreteDemo:
    """Build discrete DGP → M sweep / pick M → plot."""

    def __init__(self, config: Dict[str, Any]) -> None:
        c = config
        self.seed = resolve_seed(c["seed"])
        self.K = c["K"]
        self.d = c["d"]
        self.action_num = c["action_num"]
        self.n_per_segment = c["n_per_segment"]
        self.n_implement = c["n_implement"]
        self.target_mahalanobis_sep = c["target_mahalanobis_sep"]
        self.min_leaf_size = c["min_leaf_size"]
        self.train_frac = c["train_frac"]
        self.train_frac_retrain = c["train_frac_retrain"]
        self.use_hybrid_method = c["use_hybrid_method"]
        self.action_method = c["action_method"]
        self.disturb_covariate_noise = c["disturb_covariate_noise"]
        self.partial_x = c["partial_x"]
        self.DR_generation_method = c["DR_generation_method"]
        self.disallowed_ball_radius = c["disallowed_ball_radius"]
        self.algorithms = tuple(c["algorithms"])
        self.out_dir = Path(c["out_dir"])
        self.plot_n_per_segment = c.get("plot_n_per_segment")
        self.param_range = dict(c["param_range"])

        x_mean_vectors = c.get("x_mean_vectors")
        if x_mean_vectors is not None:
            x_mean_vectors = np.asarray(x_mean_vectors, dtype=float)
            if x_mean_vectors.shape != (self.K, self.d):
                raise ValueError(
                    f"x_mean_vectors must have shape ({self.K}, {self.d}), "
                    f"got {x_mean_vectors.shape}."
                )
        self.x_mean_vectors = x_mean_vectors
        self.segment_colors_ground_truth = c.get("segment_colors_ground_truth")
        self.segment_colors_estimated = c.get("segment_colors_estimated")

        self.pop: Optional[PopulationSimulator] = None
        self._arrays: Optional[Dict[str, np.ndarray]] = None
        self.results: Dict[str, Dict[str, Any]] = {}
        self.oracle_profits: Optional[float] = None

    def build(self) -> PopulationSimulator:
        np.random.seed(self.seed)
        n_pilot = self.n_per_segment * self.K
        self.pop = PopulationSimulator(
            n_pilot,
            self.n_implement,
            self.d,
            self.K,
            self.disturb_covariate_noise,
            self.param_range,
            self.DR_generation_method,
            self.partial_x,
            action_num=self.action_num,
            X_mean_vectors=self.x_mean_vectors,
            target_mahalanobis_sep=self.target_mahalanobis_sep,
            disallowed_ball_radius=self.disallowed_ball_radius,
        )
        self.pop.split_pilot_customers_into_train_and_validate(train_frac=self.train_frac_retrain)
        self._arrays = self._extract_train_arrays()
        return self.pop

    def _apply_split(self, train_frac: float) -> None:
        assert self.pop is not None
        self.pop.split_pilot_customers_into_train_and_validate(train_frac=train_frac)
        self._arrays = self._extract_train_arrays()

    def run_algorithms(self) -> Dict[str, Dict[str, Any]]:
        if self.pop is None or self._arrays is None:
            raise RuntimeError("Call build() before run_algorithms().")

        self.results = {}
        for algo in self.algorithms:
            self.results[algo] = sweep_pick_and_retrain(
                self.pop,
                algo,
                self.K,
                self._apply_split,
                self._get_train_arrays,
                include_interactions=False,
                seed=self.seed,
                min_leaf_size=self.min_leaf_size,
                use_hybrid_method=self.use_hybrid_method,
                train_frac=self.train_frac,
                train_frac_retrain=self.train_frac_retrain,
                action_method=self.action_method,
            )
        self.evaluate_implementation_profits()
        return self.results

    def evaluate_implementation_profits(self) -> float:
        if self.pop is None or not self.results:
            raise RuntimeError("Call run_algorithms() first.")

        pop = self.pop
        self.oracle_profits = oracle_profit_on_customers(
            pop.implement_customers, signal_d=pop.signal_d,
        )
        for algo, res in self.results.items():
            if algo == "kmeans-standard":
                assign_new_customers_to_segments(
                    pop, pop.implement_customers, res["model"], algo,
                )
            elif algo == "dast":
                res["tree"].predict_segment(pop.implement_customers, res["segment_dict"])
            else:
                raise ValueError(f"No implementation assignment for '{algo}'.")
            res["implementation_profits"] = sum(
                cust.evaluate_profits(algo) for cust in pop.implement_customers
            )
        return self.oracle_profits

    def plot(self) -> Dict[str, Path]:
        """Save PNG + PDF for each panel; returned paths are PDF (for LaTeX)."""
        if self.pop is None or self._arrays is None or not self.results:
            raise RuntimeError("Call run_algorithms() before plot().")

        self.out_dir.mkdir(parents=True, exist_ok=True)
        x_mat, D_vec, y_vec = self._get_train_arrays()
        plot_idx = self._plot_train_indices()
        df_plot = self._subsample_pilot_df(self.pop.to_dataframe())
        saved: Dict[str, Path] = {}
        seg_optimal_actions = {
            seg.segment_id: seg.action for seg in self.pop.true_segments
        }

        _, saved["pilot"] = plot_demo_pilot(
            df_plot,
            str(self.out_dir / "pilot_plot"),
        )

        _, saved["ground_truth"] = plot_demo_ground_truth(
            df_plot,
            str(self.out_dir / "ground_truth_plot"),
            segment_colors=self.segment_colors_ground_truth,
            seg_optimal_actions=seg_optimal_actions,
        )

        stack_panels = [
            {
                "kind": "pilot",
                "df": df_plot,
                "title": "Pilot Data",
            },
            {
                "kind": "ground_truth",
                "df": df_plot,
                "title": "Ground Truth Segmentation",
                "segment_colors": self.segment_colors_ground_truth,
                "segment_cmap_name": "Set2",
                "seg_optimal_actions": seg_optimal_actions,
            },
        ]
        for algo in self.algorithms:
            res = self.results[algo]
            picked_M = res["picked_M"]
            plot_labels = res["labels"][plot_idx]
            plot_D = D_vec[plot_idx]
            plot_y = y_vec[plot_idx]
            # Legend actions: recomputed on plotted points (same rule as kmeans/dast fit).
            seg_estimated_actions = build_learned_segment_actions(
                plot_labels,
                x_mat[plot_idx],
                plot_D,
                plot_y,
                method=self.action_method,
                gamma=self.pop.gamma_train[plot_idx] if self.pop.gamma_train is not None else None,
                action_num=self.action_num,
                include_interactions=False,
            )
            stem = self.out_dir / f"{algo}_segmentation_{picked_M}"
            if algo == "kmeans-standard" and res.get("m_selection") == "elbow_inertia_from_M1":
                sweep = res["m_sweep"]
                _, saved[algo] = plot_kmeans_silhouette_elbow(
                    sweep["M"].to_numpy(),
                    sweep["kmeans-standard_val"].to_numpy(),
                    sweep["silhouette"].to_numpy(),
                    elbow_M=int(picked_M),
                    silhouette_M=int(res.get("silhouette_picked_M", picked_M)),
                    out_path=str(stem),
                )
            else:
                _, saved[algo] = plot_demo_segmentation(
                    plot_labels,
                    x_mat[plot_idx],
                    plot_y,
                    plot_D,
                    algo=algo,
                    M=picked_M,
                    out_path=str(stem),
                    tree=res.get("tree"),
                    model=res.get("model"),
                    segment_colors=self.segment_colors_estimated,
                    seg_estimated_actions=seg_estimated_actions,
                )
            stack_panels.append({
                "kind": "segmentation",
                "labels": plot_labels,
                "X": x_mat[plot_idx],
                "y_vec": plot_y,
                "D_vec": plot_D,
                "algo": algo,
                "tree": res.get("tree"),
                "model": res.get("model"),
                "title": algo_panel_title(algo, picked_M),
                "segment_colors": self.segment_colors_estimated,
                "seg_estimated_actions": seg_estimated_actions,
            })

        _, saved["combined"] = plot_demo_combined(
            stack_panels, str(self.out_dir / "demo_combined"),
        )
        return saved

    def run(self) -> Tuple[PopulationSimulator, Dict[str, Dict[str, Any]], Dict[str, Path]]:
        self.build()
        self.run_algorithms()
        paths = self.plot()
        self._print_summary()
        return self.pop, self.results, paths

    def _extract_train_arrays(self) -> Dict[str, np.ndarray]:
        assert self.pop is not None
        customers = self.pop.train_customers
        return {
            "x_mat": np.array([c.x for c in customers]),
            "D_vec": np.array([c.D_i for c in customers]),
            "y_vec": np.array([c.y for c in customers]),
        }

    def _get_train_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert self._arrays is not None
        return self._arrays["x_mat"], self._arrays["D_vec"], self._arrays["y_vec"]

    def _true_segment_ids_train(self) -> np.ndarray:
        assert self.pop is not None
        return np.array([c.true_segment.segment_id for c in self.pop.train_customers])

    def _indices_per_true_segment(self, true_seg_ids: np.ndarray) -> np.ndarray:
        """Up to plot_n_per_segment rows per true segment (figures only)."""
        if self.plot_n_per_segment is None:
            return np.arange(len(true_seg_ids))
        rng = np.random.default_rng(self.seed)
        chosen: list[int] = []
        for seg in sorted(np.unique(true_seg_ids)):
            pool = np.where(true_seg_ids == seg)[0]
            k = min(self.plot_n_per_segment, len(pool))
            chosen.extend(rng.choice(pool, size=k, replace=False).tolist())
        return np.array(sorted(chosen))

    def _plot_train_indices(self) -> np.ndarray:
        return self._indices_per_true_segment(self._true_segment_ids_train())

    def _subsample_pilot_df(self, df):
        if self.plot_n_per_segment is None:
            return df
        idx = self._indices_per_true_segment(df["true_segment_id"].to_numpy())
        return df.iloc[idx].reset_index(drop=True)

    def _print_learned_segment_stats(self, algo: str, res: Dict[str, Any]) -> None:
        """Per-segment n_0, n_1, y_0, y_1 (means) on final train fit — for manual est_action checks."""
        assert self.pop is not None and self._arrays is not None
        labels = np.ravel(res["labels"]).astype(int)
        D_vec = self._arrays["D_vec"]
        y_vec = self._arrays["y_vec"]
        title = {
            "kmeans-standard": "k-means",
            "dast": "DAS",
        }.get(algo, algo)
        print(f"    {title} segment stats (final train fit; y_a = mean(Y|D=a)):")
        print(f"      {'seg':>3}  {'n_0':>4}  {'n_1':>4}  {'y_0':>8}  {'y_1':>8}  "
              f"{'sum_y0':>6}  {'sum_y1':>6}  {'est_tau':>8}  {'a_hat':>4}")
        for seg in self.pop.est_segments_list[algo]:
            m = int(seg.segment_id)
            idx = labels == m
            D_m = D_vec[idx]
            y_m = y_vec[idx]
            n0 = int(np.sum(D_m == 0))
            n1 = int(np.sum(D_m == 1))
            y0 = float(y_m[D_m == 0].mean()) if n0 else float("nan")
            y1 = float(y_m[D_m == 1].mean()) if n1 else float("nan")
            s0 = int(y_m[D_m == 0].sum()) if n0 else 0
            s1 = int(y_m[D_m == 1].sum()) if n1 else 0
            print(
                f"      {m:3d}  {n0:4d}  {n1:4d}  {y0:8.4f}  {y1:8.4f}  "
                f"{s0:6d}  {s1:6d}  {seg.est_tau:+8.4f}  {seg.est_action:4d}"
            )
        print("      (check: est_tau ≈ y_1 - y_0; a_hat = 1[y_1 - y_0 >= 0] when n_0,n_1 > 0)")

    def _print_summary(self) -> None:
        assert self.pop is not None
        print("\n=== Discrete Demo Summary ===")
        print(f"  seed={self.seed}, K={self.K}, M_range={m_range_for_k(self.K)}, d={self.d}")
        print(f"  train_frac={self.train_frac} (M sweep), train_frac_retrain={self.train_frac_retrain}")
        print(f"  pilot N={self.pop.N_total_pilot_customers}")
        print(f"  median NN Mahalanobis sep={self.pop.realized_median_nn_mahalanobis_sep:.3f}")
        for k, seg in enumerate(self.pop.true_segments):
            tau_str = ", ".join(f"{t:.3f}" for t in seg.tau)
            print(f"  true segment {k}: action={seg.action}, tau=[{tau_str}]")
        for algo, res in self.results.items():
            score = res["score"]
            score_str = f"{score:.4f}" if score is not None else "n/a"
            sel = res.get("m_selection")
            sel_str = f", m_selection={sel}" if sel else ""
            print(f"  {algo}: picked_M={res['picked_M']}, score={score_str}, "
                  f"sweep_frac={res['train_frac_sweep']}{sel_str}")
            print(f"    M sweep:\n{res['m_sweep'].to_string(index=False)}")
            if algo in ("kmeans-standard", "dast"):
                self._print_learned_segment_stats(algo, res)
        if self.oracle_profits is not None:
            print("\n=== Share of optimal policy ===")
            print(f"  optimal profit={self.oracle_profits:.4f}")
            for algo, res in self.results.items():
                p = res.get("implementation_profits")
                if p is not None:
                    rel = 100.0 * p / self.oracle_profits if self.oracle_profits > 0 else float("nan")
                    print(
                        f"  {algo}: profit={p:.4f} "
                        f"→ {rel:.1f}% of optimal policy"
                    )


if __name__ == "__main__":
    set_demo_plot_style()
    DiscreteDemo(DEMO_CONFIG).run()

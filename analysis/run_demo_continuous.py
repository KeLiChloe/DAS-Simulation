"""
Illustrative continuous-outcome demo (Gaussian noise, K-means vs DAST).

Run from project root:
    python analysis/run_demo_continuous.py

Edit DEMO_CONFIG below — the only place to set parameters.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "analysis") not in sys.path:
    sys.path.insert(0, str(ROOT / "analysis"))

from demo_m_utils import m_range_for_k, resolve_seed, sweep_pick_and_retrain
from ground_truth import PopulationSimulator
from oracle import oracle_profit_on_customers
from plot import plot_ground_truth, plot_segmentation
from plot_style import set_plot_style
from utils import assign_new_customers_to_segments

# =============================================================================
# Demo configuration — edit here only
# =============================================================================

DEMO_CONFIG: Dict[str, Any] = {
    "seed": None,  # None → random seed printed at runtime
    "K": 3,
    "d": 1,
    "action_num": 2,
    "n_per_segment": 100,
    "n_implement": 50,
    "target_mahalanobis_sep": 6.0,  # within-cluster covariate noise (same rule as main.py mahal runs)
    "Y_noise_std_scale": 0.15,
    "min_leaf_size": 2,
    "train_frac": 0.8,
    "train_frac_retrain": 1.0,
    "use_hybrid_method": True,
    "disturb_covariate_noise": 3.0,
    "partial_x": 1.0,
    "DR_generation_method": "mlp",
    "disallowed_ball_radius": 0.2,
    "algorithms": ("kmeans-standard", "dast"),
    "out_dir": "figures/demo_continuous",
    "x_mean_vectors": np.array([[5.0], [18.0], [26.0]]),
    "param_range": {
        "alpha": (-5.0, 5.0),
        "beta": (-0.5, 0.5),
        "tau": (-30.0, 30.0),
        "delta": None,          # e.g. (-0.8, 0.8) to enable x×D interactions
        "x_mean": (-25.0, 25.0),
        "target_p": None,
    },
}


class ContinuousDemo:
    """Build continuous DGP → M sweep / pick M → plot."""

    def __init__(self, config: Dict[str, Any]) -> None:
        c = config
        self.seed = resolve_seed(c["seed"])
        self.K = c["K"]
        self.d = c["d"]
        self.action_num = c["action_num"]
        self.n_per_segment = c["n_per_segment"]
        self.n_implement = c["n_implement"]
        self.target_mahalanobis_sep = c.get("target_mahalanobis_sep", 6.0)
        self.X_noise_std_scale = c.get("X_noise_std_scale")
        if self.target_mahalanobis_sep is not None and self.X_noise_std_scale is not None:
            raise ValueError("Set only one of target_mahalanobis_sep or X_noise_std_scale.")
        if self.target_mahalanobis_sep is None and self.X_noise_std_scale is None:
            self.target_mahalanobis_sep = 6.0
        self.Y_noise_std_scale = c["Y_noise_std_scale"]
        self.min_leaf_size = c["min_leaf_size"]
        self.train_frac = c["train_frac"]
        self.train_frac_retrain = c["train_frac_retrain"]
        self.use_hybrid_method = c["use_hybrid_method"]
        self.disturb_covariate_noise = c["disturb_covariate_noise"]
        self.partial_x = c["partial_x"]
        self.DR_generation_method = c["DR_generation_method"]
        self.disallowed_ball_radius = c["disallowed_ball_radius"]
        self.algorithms = tuple(c["algorithms"])
        self.out_dir = Path(c["out_dir"])
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

        self.pop: Optional[PopulationSimulator] = None
        self._arrays: Optional[Dict[str, np.ndarray]] = None
        self.results: Dict[str, Dict[str, Any]] = {}
        self.oracle_profits: Optional[float] = None

    @property
    def include_interactions(self) -> bool:
        return self.param_range.get("delta") is not None

    def build(self) -> PopulationSimulator:
        np.random.seed(self.seed)
        n_pilot = self.n_per_segment * self.K
        pop_kwargs: Dict[str, Any] = dict(
            X_mean_vectors=self.x_mean_vectors,
            Y_noise_std_scale=self.Y_noise_std_scale,
            disallowed_ball_radius=self.disallowed_ball_radius,
        )
        if self.target_mahalanobis_sep is not None:
            pop_kwargs["target_mahalanobis_sep"] = self.target_mahalanobis_sep
        else:
            pop_kwargs["X_noise_std_scale"] = self.X_noise_std_scale

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
            outcome_type="continuous",
            **pop_kwargs,
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
                include_interactions=self.include_interactions,
                is_discrete=False,
                seed=self.seed,
                min_leaf_size=self.min_leaf_size,
                use_hybrid_method=self.use_hybrid_method,
                train_frac=self.train_frac,
                train_frac_retrain=self.train_frac_retrain,
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
        if self.pop is None or self._arrays is None or not self.results:
            raise RuntimeError("Call run_algorithms() before plot().")

        self.out_dir.mkdir(parents=True, exist_ok=True)
        saved: Dict[str, Path] = {}

        plot_ground_truth(
            self.pop.to_dataframe(),
            title="Ground Truth (Continuous Outcomes)",
            x_col="x_0",
            x2_col="x_1" if self.d >= 2 else None,
            y_col="outcome",
            D_col="D_i",
            run_idx=None,
            out_dir=str(self.out_dir),
        )
        saved["ground_truth"] = self.out_dir / "ground_truth_plot.png"

        x_mat, D_vec, y_vec = self._get_train_arrays()
        for algo, res in self.results.items():
            picked_M = res["picked_M"]
            plot_segmentation(
                res["labels"], x_mat, y_vec, D_vec,
                algo=algo, M=picked_M, tree=res["tree"], run_idx=None,
            )
            seg_name = f"{algo}_segmentation_{picked_M}.png"
            src = ROOT / "figures" / seg_name
            dst = self.out_dir / seg_name
            if src.exists() and src != dst:
                dst.write_bytes(src.read_bytes())
                src.unlink()
            saved[algo] = dst
        return saved

    def run(self) -> Tuple[PopulationSimulator, Dict[str, Dict[str, Any]], Dict[str, Path]]:
        self.build()
        self.run_algorithms()
        paths = self.plot()
        self._print_summary(paths)
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

    def _print_summary(self, paths: Dict[str, Path]) -> None:
        assert self.pop is not None
        print("\n=== Continuous Demo Summary ===")
        print(f"  seed={self.seed}, K={self.K}, M_range={m_range_for_k(self.K)}, d={self.d}")
        print(f"  train_frac={self.train_frac} (M sweep), train_frac_retrain={self.train_frac_retrain}")
        print(f"  pilot N={self.pop.N_total_pilot_customers}, Y_noise_std={self.pop.noise_std:.4f}")
        if self.target_mahalanobis_sep is not None:
            print(f"  median NN Mahalanobis sep={self.pop.realized_median_nn_mahalanobis_sep:.3f}")
        else:
            print(f"  X_noise_std_scale={self.X_noise_std_scale}")
        for k, seg in enumerate(self.pop.true_segments):
            tau_str = ", ".join(f"{t:.3f}" for t in seg.tau)
            print(f"  true segment {k}: action={seg.action}, tau=[{tau_str}]")
        for algo, res in self.results.items():
            score = res["score"]
            score_str = f"{score:.4f}" if score is not None else "n/a"
            print(f"  {algo}: picked_M={res['picked_M']}, score={score_str}, "
                  f"sweep_frac={res['train_frac_sweep']}")
            print(f"    M sweep:\n{res['m_sweep'].to_string(index=False)}")
        if self.oracle_profits is not None:
            print(f"\n  oracle profit={self.oracle_profits:.4f}")
            for algo, res in self.results.items():
                p = res.get("implementation_profits")
                if p is not None:
                    rel = 100.0 * p / self.oracle_profits if self.oracle_profits > 0 else float("nan")
                    print(f"  {algo}: implementation_profit={p:.4f} ({rel:.1f}% of oracle)")
        print("\nFigures saved:")
        for name, path in paths.items():
            print(f"  - {name}: {path}")


if __name__ == "__main__":
    set_plot_style()
    ContinuousDemo(DEMO_CONFIG).run()

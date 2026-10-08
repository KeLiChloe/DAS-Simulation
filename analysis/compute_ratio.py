import argparse
import pickle
import numpy as np
from scipy import stats


def print_params(data):
    dast_parameters = {
        'K': data.get('exp_params').get('K'),
        'd': data.get('exp_params').get('d'),
        'X_noise_std_scale': data.get('exp_params').get('X_noise_std_scale'),
        'target_mahalanobis_sep': data.get('exp_params').get('target_mahalanobis_sep'),
        'Y_noise_std_scale': data.get('exp_params').get('Y_noise_std_scale'),
        'alpha_param_range': data.get('exp_params').get('param_range').get('alpha'),
        'beta_param_range': data.get('exp_params').get('param_range').get('beta'),
        'tau_param_range': data.get('exp_params').get('param_range').get('tau'),
        'delta_param_range': data.get('exp_params').get('param_range').get('delta'),
        'x_param_range': data.get('exp_params').get('param_range').get('x_mean'),
        'partial_x': data.get('exp_params').get('partial_x'),
        'N_segment_size': data.get('exp_params').get('N_segment_size'),
        'implementation_scale': data.get('exp_params').get('implementation_scale', 1),
        'disallowed_ball_radius': data.get('exp_params').get('disallowed_ball_radius'),
        'outcome_type': data.get('exp_params').get('outcome_type'),
        'alpha_range': data.get('exp_params').get('param_range', {}).get('alpha'),
        'tau_range': data.get('exp_params').get('param_range', {}).get('tau'),
        'sequence_seed': data.get('exp_params').get('sequence_seed'),
    }

    print("=== Experiment Parameters ===")
    for key, value in dast_parameters.items():
        print(f"{key:>20}: {value}")


def compute_improvement_ratio(data, comparators, relative=False, eps=1e-6):
    """
    relative=False  ->  (dast - comp) / |comp|
                        standard improvement ratio
    relative=True   ->  (dast - comp) / (oracle - comp)
                        relative improvement ratio: fraction of oracle gap captured by DAST
                        requires data['oracle_profits_impl']
    """
    if relative and 'oracle_profits_impl' not in data:
        raise ValueError(
            "--relative requested but 'oracle_profits_impl' not found in pkl. "
            "Re-run the experiment with the current main.py which computes oracle profits."
        )

    improvement_ratios = {comp: [] for comp in comparators}
    n = len(data['dast'])

    for i in range(n):
        dast_profit = data['dast'][i]['implementation_profits']

        for comp in comparators:
            comp_profit = data[comp][i]['implementation_profits']

            if relative:
                oracle_profit = data['oracle_profits_impl'][i]
                denom = oracle_profit - comp_profit
                if abs(denom) < eps:
                    continue
            else:
                denom = abs(comp_profit)
                if denom < eps:
                    continue

            improvement_ratios[comp].append((dast_profit - comp_profit) / denom)

    return improvement_ratios


def compute_oracle_gap_ratio(data, algos, eps=1e-6):
    """
    Oracle Gap Ratio:  (oracle - algo) / oracle   for each algo.

    - Value in [0, 1]: 0 = algo achieves oracle, 1 = algo gets nothing.
    - Lower is better.
    - Stable denominator: oracle >= 0 always.
    - Requires data['oracle_profits_impl'].
    """
    if 'oracle_profits_impl' not in data:
        raise ValueError(
            "--regret requested but 'oracle_profits_impl' not found in pkl. "
            "Re-run the experiment with the current main.py which computes oracle profits."
        )

    result = {algo: [] for algo in algos}
    n = len(data['oracle_profits_impl'])

    for i in range(n):
        oracle = data['oracle_profits_impl'][i]
        if abs(oracle) < eps:
            continue
        for algo in algos:
            if algo == 'dast':
                profit = data['dast'][i]['implementation_profits']
            else:
                profit = data[algo][i]['implementation_profits']
            result[algo].append((oracle - profit) / abs(oracle))

    return result


def print_summary(filtered_ratios, title="Summary Statistics"):
    preferred_order = [
        "dast",
        "kmeans-standard", "gmm-standard", "clr-standard", "mst",
        "t_learner", "s_learner", "x_learner", "dr_learner",
        "causal_forest", "policy_tree",
    ]

    print(f"\n{title}:")
    keys = sorted(
        filtered_ratios.keys(),
        key=lambda k: preferred_order.index(k) if k in preferred_order else 999,
    )
    for key in keys:
        ratios = np.asarray(filtered_ratios[key])
        if len(ratios) == 0:
            continue
        ratios_pct = ratios * 100
        n = len(ratios_pct)
        mean = float(np.mean(ratios_pct))
        se = stats.sem(ratios_pct)
        ci = se * stats.t.ppf(0.975, n - 1) if n > 1 else 0.0
        print(
            f"{key:>20}: {n:3d} runs | Avg: {mean:7.2f}% | "
            f"95% CI: [{mean - ci:7.2f}%, {mean + ci:7.2f}%]"
        )


def filter_ratios(improvement_ratios, apply_remove_extreme, apply_sigma_clip):
    filtered_ratios = {}

    for comp, ratios in improvement_ratios.items():
        ratios_np = np.array(ratios)

        if apply_remove_extreme.get(comp, False):
            min_indices = np.argsort(ratios_np)[:3]
            mask = np.ones_like(ratios_np, dtype=bool)
            mask[min_indices] = False
            ratios_np = ratios_np[mask]

            max_indices = np.argsort(ratios_np)[-3:]
            mask = np.ones_like(ratios_np, dtype=bool)
            mask[max_indices] = False
            ratios_np = ratios_np[mask]

        if apply_sigma_clip:
            mean = np.mean(ratios_np)
            std = np.std(ratios_np)
            ratios_np = ratios_np[np.abs(ratios_np - mean) <= 1.5 * std]

        filtered_ratios[comp] = ratios_np
    return filtered_ratios


_IGNORE = {'dast', 'exp_params', 'seed', 'oracle_profits_impl', 'covariate_overlap'}

apply_remove_extreme = {
    "gmm-standard": True,
    "kmeans-standard": True,
    "mst": True,
    "clr-standard": True,
    "t_learner": True,
    "x_learner": True,
    "dr_learner": True,
    "s_learner": True,
    "policy_tree": True,
    "causal_forest": True,
}

_METRICS = ("relative_comp", "relative_oracle", "regret")


def main():
    parser = argparse.ArgumentParser(
        description="Compute DAST improvement ratios from a pkl experiment file."
    )
    parser.add_argument("file", help="Path to the .pkl experiment file")
    parser.add_argument(
        "--metric",
        choices=_METRICS,
        default="relative_comp",
        help=(
            "relative_comp:   (dast - comp) / |comp|  [default]\n"
            "relative_oracle: (dast - comp) / (oracle - comp)  — requires oracle_profits_impl\n"
            "regret:          (oracle - algo) / oracle  — requires oracle_profits_impl"
        ),
    )
    parser.add_argument(
        "--sigma_clip", action="store_true",
        help="Apply 3-sigma clipping after extreme-value trimming.",
    )
    args = parser.parse_args()

    with open(args.file, "rb") as f:
        data = pickle.load(f)

    print_params(data)

    if args.metric == "regret":
        algos = ["dast"] + [
            k for k in data.keys()
            if k not in _IGNORE
            and isinstance(data[k], list)
            and len(data[k]) > 0
            and isinstance(data[k][0], dict)
        ]
        print(f"Algorithms found in data: {algos}")
        remove_extreme_map = {**apply_remove_extreme, "dast": True}
        ratios = compute_oracle_gap_ratio(data, algos)
        filtered_ratios = filter_ratios(ratios, remove_extreme_map, args.sigma_clip)
        print_summary(filtered_ratios, title="Oracle Gap Ratio  (lower = closer to oracle)")
    else:
        comparators = [
            k for k in data.keys()
            if k not in _IGNORE
            and isinstance(data[k], list)
            and len(data[k]) > 0
            and isinstance(data[k][0], dict)
        ]
        print(f"Comparators found in data: {comparators}")
        relative = (args.metric == "relative_oracle")
        ratios = compute_improvement_ratio(data, comparators, relative=relative)
        filtered_ratios = filter_ratios(ratios, apply_remove_extreme, args.sigma_clip)
        print_summary(filtered_ratios)


if __name__ == "__main__":
    main()

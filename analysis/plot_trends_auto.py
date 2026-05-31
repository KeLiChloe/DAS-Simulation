#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified trend plot script.

Scans experiment pkls, builds one CSV with both metrics, and plots two figures:

  relative_comp   (dast - comp) / |comp|
  relative_oracle (dast - comp) / (oracle - comp)  [requires oracle_profits_impl in pkl]

Usage (build + plot in one step):
  python analysis/plot_trends_auto.py build-csv \\
      --dir exp_feb_2026/discrete/varying_d_set8
  # -> figures/curves.csv
  # -> figures/curves_relative_comp.pdf/.png
  # -> figures/curves_relative_oracle.pdf/.png

Re-plot from an existing CSV only:
  python analysis/plot_trends_auto.py plot-csv --csv figures/curves.csv
"""

import os
import re
import glob
import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats
from plot_style import (
    set_plot_style,
    IGNORE_KEYS_DEFAULT, DEFAULT_REMOVE_EXTREME,
    DEFAULT_COLORS, LABEL_MAP, X_LABEL_MAP, COMPARATOR_ORDER,
)

BUILD_METRICS = ("relative_comp", "relative_oracle")

TITLE_MAP = {
    ("relative_comp",   "d"):     "DAS Improvement Ratio (Relative to Comparator) vs. Dimension ",
    ("relative_comp",   "K"):     "DAS Improvement Ratio (Relative to Comparator) vs. Num. of Clusters",
    ("relative_comp",   "delta"):  "DAS Improvement Ratio (Relative to Comparator) vs. Interaction Strength",
    ("relative_comp",   "Xnoise"): "DAS Improvement Ratio (Relative to Comparator) vs. $X$ Noise Scale",
    ("relative_comp",   "mahalanobis_sep"): "DAS Improvement Ratio (Relative to Comparator) vs. Mahalanobis Separation",
    ("relative_oracle", "d"):      "DAS Improvement Ratio (Relative to Oracle) vs. Dimension",
    ("relative_oracle", "K"):      "DAS Improvement Ratio (Relative to Oracle) vs. Num. of Clusters",
    ("relative_oracle", "delta"):  "DAS Improvement Ratio (Relative to Oracle) vs. Interaction Strength",
    ("relative_oracle", "Xnoise"): "DAS Improvement Ratio (Relative to Oracle) vs. $X$ Noise Scale",
    ("relative_oracle", "mahalanobis_sep"): "DAS Improvement Ratio (Relative to Oracle) vs. Mahalanobis Separation",
    ("regret",          "d"):     "Regret Ratio vs. Dimension",
    ("regret",          "K"):     "Regret Ratio vs. Num. of Clusters",
    ("regret",          "delta"): "Regret Ratio vs. Interaction Strength",
}

YLABEL_MAP = {
    "relative_comp":   "(dast - comp) / |comp| (%)",
    "relative_oracle": "(dast - comp) / (oracle - comp) (%)",
    "regret":          "(oracle - algo) / oracle (%)",
}

# dast / dast_old get distinct colors; both highlighted in regret mode
DAST_COLOR     = "#E41A1CAF"
DAST_LABEL     = "DAST"


# ---------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------
def extract_experiment_param(filepath):
    """Parse exp_{param}_{tag}.pkl filenames (tag encoding only; see resolve_experiment_param)."""
    base = os.path.basename(filepath)
    base_stripped = re.sub(r"(_with_oracle|_oracle)$", "", os.path.splitext(base)[0])
    m = re.match(r"exp_([A-Za-z]+)_(.+)$", base_stripped)
    if not m:
        raise ValueError(f"Filename not recognised: {base}")
    param_name = m.group(1)
    raw_val = m.group(2).replace("p", ".") if param_name == "delta" else m.group(2)
    try:
        param_value = float(raw_val)
    except ValueError as e:
        raise ValueError(f"Cannot parse param_value from filename: {base}") from e
    if param_name.lower() == "xnoise":
        param_name = "Xnoise"
    elif param_name.lower() in {"msep", "mahalanobissep"}:
        param_name = "mahalanobis_sep"
    return param_name, param_value


def _decode_xnoise_filename_tag(tag_value):
    """
    Fallback when pkl has no exp_params.

    Continuous run scripts encode X_noise_std_scale as hundredths in the tag
    (exp_Xnoise_100 -> 1.00, exp_Xnoise_125 -> 1.25). Discrete scripts encode
    millis (050 -> 0.05); those pkls should supply exp_params instead.
    """
    v = float(tag_value)
    if v < 10:
        return v
    return v / 100.0


def _format_numeric_tick(value, precision=3):
    """Format ticks without hiding meaningful decimal places."""
    text = f"{value:.{precision}f}".rstrip("0").rstrip(".")
    return text if text else "0"


def resolve_experiment_param(filepath, data):
    """
    Return (param_name, param_value) for plotting.

    For X-noise sweeps, use exp_params['X_noise_std_scale'] (the actual scale passed
    to main.py), not the integer tag in the filename (100 -> 1.00, not 100).
    """
    param_name, param_value = extract_experiment_param(filepath)
    exp_params = data.get("exp_params") or {}

    if param_name == "mahalanobis_sep":
        sep = exp_params.get("target_mahalanobis_sep")
        if sep is not None:
            return "mahalanobis_sep", float(sep)
        return "mahalanobis_sep", _decode_xnoise_filename_tag(param_value)

    if param_name != "Xnoise":
        return param_name, param_value

    scale = exp_params.get("X_noise_std_scale")
    if scale is not None:
        return "Xnoise", float(scale)
    return "Xnoise", _decode_xnoise_filename_tag(param_value)


def extract_algo_keys(data):
    """Return all list-valued keys that contain per-sim dicts."""
    return [
        k for k in data
        if k not in IGNORE_KEYS_DEFAULT
        and isinstance(data[k], list)
        and len(data[k]) > 0
        and isinstance(data[k][0], dict)
    ]


# ---------------------------------------------------------------
# Compute functions
# ---------------------------------------------------------------
def _require_oracle(data):
    if "oracle_profits_impl" not in data:
        raise ValueError(
            "'oracle_profits_impl' not found in pkl. "
            "Re-run the experiment with the current main.py."
        )


def compute_relative_comp(data, comparators, eps=1e-12):
    """(dast - comp) / |comp|"""
    dast_runs = data["dast"]
    n = len(dast_runs)
    result = {}
    for comp in comparators:
        comp_runs = data.get(comp)
        if not isinstance(comp_runs, list) or len(comp_runs) != n:
            continue
        ratios = []
        for i in range(n):
            d = float(dast_runs[i]["implementation_profits"])
            c = float(comp_runs[i]["implementation_profits"])
            if abs(c) < eps:
                continue
            ratios.append((d - c) / abs(c))
        result[comp] = np.asarray(ratios, dtype=float)
    return result


def compute_relative_oracle(data, comparators, eps=1e-3):
    """(dast - comp) / (oracle - comp)"""
    _require_oracle(data)
    dast_runs   = data["dast"]
    oracle_list = data["oracle_profits_impl"]
    n = len(dast_runs)
    result = {}
    for comp in comparators:
        comp_runs = data.get(comp)
        if not isinstance(comp_runs, list) or len(comp_runs) != n:
            continue
        ratios = []
        for i in range(n):
            d = float(dast_runs[i]["implementation_profits"])
            c = float(comp_runs[i]["implementation_profits"])
            o = float(oracle_list[i])
            denom = o - c
            if abs(denom) < eps:
                continue
            ratios.append((d - c) / denom)
        result[comp] = np.asarray(ratios, dtype=float)
    return result


def compute_regret(data, algos, eps=1e-6):
    """(oracle - algo) / oracle  for each algo (including dast)"""
    _require_oracle(data)
    oracle_list = data["oracle_profits_impl"]
    n = len(oracle_list)
    result = {}
    for algo in algos:
        runs = data.get(algo)
        if not isinstance(runs, list) or len(runs) != n:
            continue
        ratios = []
        for i in range(n):
            o = float(oracle_list[i])
            p = float(runs[i]["implementation_profits"])
            if abs(o) < eps:
                continue
            ratios.append((o - p) / abs(o))
        result[algo] = np.asarray(ratios, dtype=float)
    return result


def compute_ratios(data, metric):
    """Dispatch to the right compute function. Returns dict entity -> np.array."""
    algo_keys = extract_algo_keys(data)
    comparators = [k for k in algo_keys if k != "dast"]

    if metric == "relative_comp":
        return compute_relative_comp(data, comparators)
    elif metric == "relative_oracle":
        return compute_relative_oracle(data, comparators)
    elif metric == "regret":
        algos = ["dast"] + comparators
        return compute_regret(data, algos)
    else:
        raise ValueError(f"Unknown metric: {metric!r}. Choose from {BUILD_METRICS + ('regret',)}.")


# ---------------------------------------------------------------
# Filtering + summarisation
# ---------------------------------------------------------------
def filter_ratios(ratios, remove_extreme_map, method="trim", trim_k=3,
                  sigma_clip=False, sigma=3.0):
    filtered = {}
    for key, arr in ratios.items():
        x = np.asarray(arr, dtype=float)
        x = x[np.isfinite(x)]
        if x.size == 0:
            filtered[key] = x
            continue
        if remove_extreme_map.get(key, False) and x.size > 2 * trim_k + 2:
            xs = np.sort(x)
            if method == "trim":
                xs = xs[trim_k: -trim_k]
            elif method == "winsorize":
                lo, hi = xs[trim_k], xs[-trim_k - 1]
                xs = np.clip(xs, lo, hi)
            x = xs
        if sigma_clip and x.size > 2:
            mu, sd = float(np.mean(x)), float(np.std(x, ddof=1))
            if sd > 0:
                x = x[np.abs(x - mu) <= sigma * sd]
        filtered[key] = x
    return filtered


def summarize_ratios(filtered_ratios):
    out = {}
    for key, arr in filtered_ratios.items():
        if arr is None or len(arr) == 0:
            out[key] = (np.nan, np.nan, 0)
            continue
        pct   = np.asarray(arr, dtype=float) * 100.0
        n     = int(pct.size)
        mu    = float(np.mean(pct))
        if n < 2:
            out[key] = (mu, np.nan, n)
        else:
            se    = float(stats.sem(pct, nan_policy="omit"))
            tcrit = float(stats.t.ppf(0.95, n - 1))
            out[key] = (mu, se * tcrit, n)
    return out


# ---------------------------------------------------------------
# STEP 1: Build CSV
# ---------------------------------------------------------------
def build_csv_from_dir(in_dir, out_csv,
                       file_pattern="exp_*.pkl",
                       sigma_clip=False, sigma=3.0,
                       extreme_method="trim", trim_k=3):
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    files = sorted(glob.glob(os.path.join(in_dir, file_pattern)))
    if not files:
        raise FileNotFoundError(
            f"No files matched '{os.path.join(in_dir, file_pattern)}'."
        )

    remove_extreme_map = {**DEFAULT_REMOVE_EXTREME, "dast": True}
    method  = None if extreme_method == "none" else extreme_method
    records = []
    all_keys, param_names = set(), set()

    for fp in files:
        with open(fp, "rb") as f:
            data = pickle.load(f)

        exp_params = data.get("exp_params", {})
        print(f"\n=== {os.path.basename(fp)} ===")
        print(exp_params)

        param_name, param_value = resolve_experiment_param(fp, data)
        param_names.add(param_name)

        for metric in BUILD_METRICS:
            try:
                ratios = compute_ratios(data, metric)
            except ValueError as e:
                print(f"  [SKIP {metric}] {e}")
                continue

            all_keys.update(ratios.keys())

            filtered = filter_ratios(ratios, remove_extreme_map=remove_extreme_map,
                                     method=method, trim_k=trim_k,
                                     sigma_clip=sigma_clip, sigma=sigma)
            summary = summarize_ratios(filtered)

            for entity, (mean_pct, err_pct, n) in summary.items():
                records.append({
                    "metric":      metric,
                    "param_name":  param_name,
                    "param_value": param_value,
                    "algo":        entity,
                    "mean_pct":    mean_pct,
                    "err_pct":     err_pct,
                    "n":           n,
                    "file":        os.path.basename(fp),
                })

    df = pd.DataFrame(records).sort_values(["param_name", "algo", "param_value"])
    df.to_csv(out_csv, index=False)
    return df, sorted(all_keys), sorted(param_names), len(files)


# ---------------------------------------------------------------
# STEP 2: Plot from CSV
# ---------------------------------------------------------------
def _plot_one_metric(df, metric, param_name, out_base, show_band=False, band_alpha=0.12):
    """Render and save one figure for a single (metric, param_name) slice."""
    xlabel = X_LABEL_MAP.get(param_name, param_name)
    title  = TITLE_MAP.get((metric, param_name),
                           f"{metric} vs. {param_name}")
    ylabel = YLABEL_MAP.get(metric, "ratio (%)")

    os.makedirs(os.path.dirname(out_base) or ".", exist_ok=True)
    set_plot_style()

    AX_TOP   = 0.76
    LEGEND_Y = AX_TOP + 0.15
    TITLE_Y  = AX_TOP + 0.2

    LEFT, RIGHT = 0.14, 0.98
    AX_CENTER_X = (LEFT + RIGHT) / 2

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    fig.subplots_adjust(left=LEFT, right=RIGHT, bottom=0.14, top=AX_TOP)
    fig.suptitle(title, x=AX_CENTER_X, y=TITLE_Y,
                 fontsize=15, fontweight="bold", fontfamily="serif",
                 ha="center")

    algos_present = list(df["algo"].unique())

    if metric == "regret":
        # All algos as lines; dast / dast_old highlighted
        order = ["dast"] + [a for a in COMPARATOR_ORDER if a in algos_present]
        algos = order + [a for a in algos_present if a not in order]
    else:
        # Only comparators (no dast line); dast_old treated as comparator
        order = [a for a in COMPARATOR_ORDER if a in algos_present]
        algos = order + [a for a in algos_present if a not in order and a != "dast"]

    for algo in algos:
        sub  = df[df["algo"] == algo].sort_values("param_value")
        x    = sub["param_value"].to_numpy(dtype=float)
        y    = sub["mean_pct"].to_numpy(dtype=float)

        if algo == "dast":
            ls = "--" if metric == "regret" else "-"
            color, label, lw, zo = DAST_COLOR, DAST_LABEL, 2.5, 5
        # elif algo == "dast_old":
        #     ls = "--" if metric == "regret" else "-"
        #     color, label, lw, zo = DAST_OLD_COLOR, DAST_OLD_LABEL, 2.3, 4
        else:
            color = DEFAULT_COLORS.get(algo, None)
            label = LABEL_MAP.get(algo, algo)
            lw, ls, zo = 2.0, "-", 3

        ax.plot(x, y, marker="o", label=label, color=color,
                linewidth=lw, linestyle=ls,
                markeredgewidth=0.8, zorder=zo)

        if show_band:
            e  = sub["err_pct"].to_numpy(dtype=float)
            ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(e)
            if np.any(ok):
                ax.fill_between(x[ok], (y - e)[ok], (y + e)[ok],
                                alpha=band_alpha, color=color, linewidth=0)

    # Reference lines
    ax.axhline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.75)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")

    unique_x = sorted(df["param_value"].dropna().unique())
    MAX_TICKS = 15
    if unique_x and np.all(np.isfinite(unique_x)):
        ticks = (unique_x if len(unique_x) <= MAX_TICKS
                 else [unique_x[i] for i in
                       np.linspace(0, len(unique_x) - 1, MAX_TICKS, dtype=int)])
        ax.set_xticks(ticks)
        if param_name in {"Xnoise", "mahalanobis_sep"}:
            ax.xaxis.set_major_formatter(
                mpl.ticker.FuncFormatter(lambda x, pos: _format_numeric_tick(x, 3))
            )
        else:
            ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.2g"))

    ymin, ymax = ax.get_ylim()
    if np.isfinite(ymin) and np.isfinite(ymax) and ymax > ymin:
        pad = 0.06 * (ymax - ymin)
        ax.set_ylim(ymin - pad, ymax + pad)

    handles, labels_leg = ax.get_legend_handles_labels()
    ncol = min(max(len(labels_leg), 1), 3)
    fig.legend(handles, labels_leg, ncol=ncol, frameon=False,
               loc="upper center", bbox_to_anchor=(AX_CENTER_X, LEGEND_Y),
               columnspacing=1.2, handlelength=2.2, handletextpad=0.6)

    fig.savefig(out_base + ".pdf")
    fig.savefig(out_base + ".png")
    plt.close(fig)

    return out_base + ".pdf", out_base + ".png"


def plot_from_csv(csv_path, out_fig, show_band=False, band_alpha=0.12):
    df = pd.read_csv(csv_path)

    for col in ("param_value", "mean_pct", "err_pct"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "err_pct" not in df.columns:
        df["err_pct"] = np.nan

    df = df.dropna(subset=["param_name", "param_value", "algo", "mean_pct"]).copy()
    if "metric" not in df.columns:
        df["metric"] = "relative_comp"

    out_dir = os.path.dirname(out_fig) or "."
    out_stem = os.path.splitext(os.path.basename(out_fig))[0]
    metrics = sorted(df["metric"].unique())
    param_names = sorted(df["param_name"].unique())

    saved = []
    for metric in metrics:
        for param_name in param_names:
            sub = df[(df["metric"] == metric) & (df["param_name"] == param_name)]
            if sub.empty:
                continue
            # Always one file per metric (relative_comp + relative_oracle).
            if len(param_names) == 1:
                out_base = os.path.join(out_dir, f"{out_stem}_{metric}")
            else:
                out_base = os.path.join(out_dir, f"{out_stem}_{param_name}_{metric}")
            saved.extend(_plot_one_metric(
                sub, str(metric), str(param_name), out_base,
                show_band=show_band, band_alpha=band_alpha,
            ))
    return saved


def default_out_fig(out_csv):
    """figures/curves.csv -> figures/curves.pdf (per-metric suffixes added in plot_from_csv)."""
    return os.path.splitext(out_csv)[0] + ".pdf"


# ---------------------------------------------------------------
# CLI
# ---------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Unified trend plot for DAST experiment results."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("build-csv",
                        help="Scan pkls, write CSV, and plot both metrics (default).")
    p1.add_argument("--dir",          required=True)
    p1.add_argument("--out_csv",      default="figures/curves.csv")
    p1.add_argument("--out_fig",      default=None,
                    help="Figure stem (default: same as --out_csv with .pdf)")
    p1.add_argument("--file_pattern", default="exp_*.pkl")
    p1.add_argument("--sigma_clip",   action="store_true")
    p1.add_argument("--sigma",        type=float, default=3.0)
    p1.add_argument("--extreme_method", choices=["trim", "winsorize", "none"],
                    default="trim")
    p1.add_argument("--trim_k",       type=int, default=3)
    p1.add_argument("--show_band",    action="store_true")
    p1.add_argument("--no_plot",      action="store_true",
                    help="Only write CSV; skip plotting.")

    p2 = sub.add_parser("plot-csv", help="Re-plot from an existing CSV.")
    p2.add_argument("--csv",       required=True)
    p2.add_argument("--out_fig",   default=None,
                    help="Figure stem (default: same as --csv with .pdf)")
    p2.add_argument("--show_band", action="store_true")

    args = parser.parse_args()

    if args.cmd == "build-csv":
        df, keys, params, nfiles = build_csv_from_dir(
            in_dir=args.dir,
            out_csv=args.out_csv,
            file_pattern=args.file_pattern,
            sigma_clip=args.sigma_clip,
            sigma=args.sigma,
            extreme_method=args.extreme_method,
            trim_k=args.trim_k,
        )
        print(f"\n[OK] Processed {nfiles} files")
        print(f"[OK] Metrics:     {', '.join(BUILD_METRICS)}")
        print(f"[OK] Param names: {params}")
        print(f"[OK] Entities:    {keys}")
        print(f"[OK] Saved CSV -> {args.out_csv}")

        if not args.no_plot:
            out_fig = args.out_fig or default_out_fig(args.out_csv)
            saved = plot_from_csv(
                csv_path=args.out_csv,
                out_fig=out_fig,
                show_band=args.show_band,
            )
            for path in saved:
                print(f"[OK] Saved plot -> {path}")

    elif args.cmd == "plot-csv":
        out_fig = args.out_fig or default_out_fig(args.csv)
        saved = plot_from_csv(
            csv_path=args.csv,
            out_fig=out_fig,
            show_band=args.show_band,
        )
        for path in saved:
            print(f"[OK] Saved plot -> {path}")


if __name__ == "__main__":
    main()

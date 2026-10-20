"""
Build / plot ARI–NMI segmentation recovery from experiment pkls.

Preferred (always refresh CSV from pkls, then plot):
  python3 analysis/plot_segmentation_metrics.py build-csv --dir <exp_dir>

plot-csv is for rare styling-only replot and requires --allow-stale-csv.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "analysis") not in sys.path:
    sys.path.insert(0, str(ROOT / "analysis"))

from plot_style import (  # noqa: E402
    DEFAULT_COLORS,
    LABEL_MAP,
    SAVEFIG_KW,
    X_LABEL_MAP,
    set_plot_style,
)
from plot_trends_auto import resolve_experiment_param  # noqa: E402

DAST_COLOR = "#E41A1CAF"

# Internal algo keys that report ARI/NMI (segmentation methods)
_SEG_KEYS = [
    "dast",
    "kmeans-standard",
    "mst",
    "gmm-standard",
    "clr-standard",
]

_KEY_TO_DISPLAY = {
    "dast": "DAS",
    "kmeans-standard": "K-Means",
    "mst": "MST",
    "gmm-standard": "GMM",
    "clr-standard": "CLR",
}

_DISPLAY_ORDER = ["DAS", "K-Means", "MST", "GMM", "CLR"]

_ALGO_TO_KEY = {v: k for k, v in _KEY_TO_DISPLAY.items()}
_ALGO_TO_KEY["DAST"] = "dast"

_PLOT_ORDER = ["GMM", "K-Means", "MST", "CLR", "DAS"]


def _style_for(algo: str) -> dict:
    key = _ALGO_TO_KEY.get(algo, algo)
    if key == "dast" or algo in ("DAS", "DAST"):
        return dict(
            color=DAST_COLOR,
            label=LABEL_MAP.get("dast", "DAS"),
            lw=2.6,
            zo=6,
        )
    return dict(
        color=DEFAULT_COLORS.get(key, "#555555AF"),
        label=LABEL_MAP.get(key, algo),
        lw=2.0,
        zo=3,
    )


def _sem(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n <= 1:
        return 0.0
    return float(np.std(x, ddof=1) / np.sqrt(n))


def build_long_csv_from_dir(
    in_dir: str | Path,
    *,
    file_pattern: str = "exp_*.pkl",
) -> pd.DataFrame:
    in_dir = Path(in_dir)
    paths = sorted(in_dir.glob(file_pattern))
    if not paths:
        raise FileNotFoundError(f"No files matching {file_pattern} in {in_dir}")

    rows = []
    for path in paths:
        with open(path, "rb") as f:
            data = pickle.load(f)
        param_name, param_value = resolve_experiment_param(str(path), data)
        for key in _SEG_KEYS:
            runs = data.get(key)
            if not isinstance(runs, list) or not runs:
                continue
            if not isinstance(runs[0], dict):
                continue
            ari = np.array([r.get("ARI") for r in runs], dtype=float)
            nmi = np.array([r.get("NMI") for r in runs], dtype=float)
            # Drop runs with missing metrics
            ok = np.isfinite(ari) & np.isfinite(nmi)
            ari, nmi = ari[ok], nmi[ok]
            if len(ari) == 0:
                continue
            rows.append({
                "param_name": param_name,
                "param_value": float(param_value),
                "algo": _KEY_TO_DISPLAY[key],
                "n_sims": int(len(ari)),
                "ARI_mean": float(np.mean(ari)),
                "ARI_se": _sem(ari),
                "NMI_mean": float(np.mean(nmi)),
                "NMI_se": _sem(nmi),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No ARI/NMI found in pkls under {in_dir}")
    df = df.sort_values(["param_value", "algo"]).reset_index(drop=True)
    return df


def _write_wide_metric(df: pd.DataFrame, metric: str, out_path: Path) -> None:
    """metric in {'ARI','NMI'} -> Method,param1,param2,..."""
    col = f"{metric}_mean"
    piv = (
        df.pivot(index="algo", columns="param_value", values=col)
        .reindex([a for a in _DISPLAY_ORDER if a in set(df["algo"])])
    )
    piv.columns = [str(int(c)) if float(c).is_integer() else str(c) for c in piv.columns]
    piv = piv.reset_index().rename(columns={"algo": "Method"})
    # Round like existing tables
    for c in piv.columns:
        if c != "Method":
            piv[c] = piv[c].map(lambda v: f"{v:.3f}" if pd.notna(v) else "")
    piv.to_csv(out_path, index=False)


def _write_wide_both(df: pd.DataFrame, out_path: Path) -> None:
    params = sorted(df["param_value"].unique())
    algos = [a for a in _DISPLAY_ORDER if a in set(df["algo"])]
    rows = []
    for algo in algos:
        for metric, col in (("ARI", "ARI_mean"), ("NMI", "NMI_mean")):
            sub = df[df["algo"] == algo].set_index("param_value")
            row = {"Method": algo, "Metric": metric}
            for p in params:
                key = str(int(p)) if float(p).is_integer() else str(p)
                val = sub.loc[p, col] if p in sub.index else np.nan
                row[key] = f"{val:.3f}" if pd.notna(val) else ""
            rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False)


def _write_sweep_mean(df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    """Average ARI/NMI means across the param sweep (equal weight per param)."""
    rows = []
    for algo in [a for a in _DISPLAY_ORDER if a in set(df["algo"])]:
        sub = df[df["algo"] == algo]
        rows.append({
            "Method": algo,
            "ARI": float(sub["ARI_mean"].mean()),
            "NMI": float(sub["NMI_mean"].mean()),
        })
    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False, float_format="%.6f")
    return out


def _tex_escape_param(param_name: str, value: float) -> str:
    if param_name == "d":
        return f"$d={int(value)}$"
    if param_name == "K":
        return f"$K={int(value)}$"
    return f"${param_name}={value:g}$"


def _write_tex_tables(
    df: pd.DataFrame,
    out_path: Path,
    *,
    chunk_size: int = 8,
    fixed_K: int | None = None,
) -> None:
    param_name = str(df["param_name"].iloc[0])
    params = sorted(df["param_value"].unique())
    algos = [a for a in _DISPLAY_ORDER if a in set(df["algo"])]
    K_str = str(fixed_K) if fixed_K is not None else "5"

    if param_name == "d":
        caption_base = (
            "Segmentation recovery on the pilot sample under varying covariate "
            f"dimension ($K={K_str}$). Entries are Monte Carlo means "
            "of ARI and NMI. Higher values indicate closer agreement with the "
            "ground-truth latent segments."
        )
        label = "tab:app-ari-nmi-varying-d"
    elif param_name == "K":
        caption_base = (
            "Segmentation recovery on the pilot sample under varying number of "
            "latent segments. Entries are Monte Carlo means of ARI and NMI. "
            "Higher values indicate closer agreement with the ground-truth "
            "latent segments."
        )
        label = "tab:app-ari-nmi-varying-K"
    else:
        caption_base = (
            f"Segmentation recovery vs. {param_name}. "
            "Entries are Monte Carlo means of ARI and NMI."
        )
        label = f"tab:app-ari-nmi-varying-{param_name}"

    chunks = [params[i:i + chunk_size] for i in range(0, len(params), chunk_size)]
    # If last chunk is tiny and we already have chunks, keep it (matches prior d=17,18 table)
    blocks = []
    for ci, chunk in enumerate(chunks):
        n = len(chunk)
        col_spec = "ll" + "c" * n
        header = "Method & Metric & " + " & ".join(
            _tex_escape_param(param_name, p) for p in chunk
        ) + r" \\"
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
        ]
        if ci == 0:
            lines.append(rf"\caption{{{caption_base}}}")
            lines.append(rf"\label{{{label}}}")
        else:
            lines.append(
                rf"\caption{{{caption_base} Continued.}}"
            )
            lines.append(rf"\label{{{label}-cont}}")
        lines += [
            rf"\begin{{tabular}}{{{col_spec}}}",
            r"\toprule",
            header,
            r"\midrule",
        ]
        for ai, algo in enumerate(algos):
            sub = df[df["algo"] == algo].set_index("param_value")
            for mi, (metric, col) in enumerate((("ARI", "ARI_mean"), ("NMI", "NMI_mean"))):
                cells = []
                for p in chunk:
                    val = sub.loc[p, col] if p in sub.index else np.nan
                    cells.append(f"{val:.3f}" if pd.notna(val) else "---")
                name = algo if mi == 0 else ""
                met = metric
                lines.append(
                    f"{name} & {met} & " + " & ".join(cells) + r" \\"
                )
            if ai < len(algos) - 1:
                lines.append(r"\addlinespace")
        lines += [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
        blocks.append("\n".join(lines))

    out_path.write_text("\n".join(blocks), encoding="utf-8")


def _write_sweep_mean_tex(
    mean_df: pd.DataFrame,
    out_path: Path,
    param_name: str,
    *,
    fixed_K: int | None = None,
) -> None:
    K_str = str(fixed_K) if fixed_K is not None else "5"
    if param_name == "d":
        caption = (
            "How well each method recovers the true customer groups on the pilot "
            f"data when we vary the covariate dimension $d$ (with $K={K_str}$). Numbers "
            "are ARI and NMI, averaged over $d$ in the sweep and over simulation "
            "replications. Higher is better."
        )
        label = "tab:app-ari-nmi-varying-d-avg"
    elif param_name == "K":
        caption = (
            "How well each method recovers the true customer groups on the pilot "
            "data when we vary the number of latent segments $K$. Numbers are ARI "
            "and NMI, averaged over $K$ in the sweep and over simulation "
            "replications. Higher is better."
        )
        label = "tab:app-ari-nmi-varying-K-avg"
    else:
        caption = (
            f"ARI and NMI averaged over the {param_name} sweep. Higher is better."
        )
        label = f"tab:app-ari-nmi-varying-{param_name}-avg"

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Method & ARI & NMI \\",
        r"\midrule",
    ]
    for _, row in mean_df.iterrows():
        lines.append(
            f"{row['Method']} & {row['ARI']:.3f} & {row['NMI']:.3f} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def export_segmentation_artifacts(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    fixed_K: int | None = None,
) -> dict:
    """Write long/wide/mean CSV + tex next to experiment outputs."""
    out_dir = Path(out_dir)
    param_name = str(df["param_name"].iloc[0])
    stem = f"segmentation_{param_name}_ari_nmi"

    long_path = out_dir / f"{stem}.csv"
    df.to_csv(long_path, index=False)

    ari_wide = out_dir / f"segmentation_{param_name}_ari_wide.csv"
    nmi_wide = out_dir / f"segmentation_{param_name}_nmi_wide.csv"
    both_wide = out_dir / f"{stem}_wide.csv"
    _write_wide_metric(df, "ARI", ari_wide)
    _write_wide_metric(df, "NMI", nmi_wide)
    _write_wide_both(df, both_wide)

    mean_path = out_dir / f"{stem}_sweep_mean.csv"
    mean_df = _write_sweep_mean(df, mean_path)

    tex_path = out_dir / f"{stem}.tex"
    mean_tex = out_dir / f"{stem}_sweep_mean.tex"
    _write_tex_tables(df, tex_path, fixed_K=fixed_K)
    _write_sweep_mean_tex(mean_df, mean_tex, param_name, fixed_K=fixed_K)

    return {
        "long": str(long_path),
        "ari_wide": str(ari_wide),
        "nmi_wide": str(nmi_wide),
        "both_wide": str(both_wide),
        "sweep_mean": str(mean_path),
        "tex": str(tex_path),
        "sweep_mean_tex": str(mean_tex),
    }


def plot_ari_nmi(
    csv_path: str | Path,
    out_base: str | Path | None = None,
    *,
    show_band: bool = True,
    band_alpha: float = 0.12,
) -> tuple[str, str]:
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    required = {"param_name", "param_value", "algo", "ARI_mean", "NMI_mean"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")

    param_name = str(df["param_name"].iloc[0])
    xlabel = X_LABEL_MAP.get(param_name, param_name)

    if out_base is None:
        out_base = csv_path.with_name("curves_ari_nmi")
    else:
        out_base = Path(out_base)

    set_plot_style()

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), sharex=True)
    LEFT, RIGHT, BOTTOM, AX_TOP = 0.07, 0.99, 0.14, 0.72
    fig.subplots_adjust(left=LEFT, right=RIGHT, bottom=BOTTOM, top=AX_TOP, wspace=0.22)
    ax_center = (LEFT + RIGHT) / 2
    legend_y = AX_TOP + 0.14
    title_y = AX_TOP + 0.20

    fig.suptitle(
        f"Segmentation recovery (ARI / NMI) vs. {xlabel}",
        x=ax_center,
        y=title_y,
        fontsize=14,
        fontweight="bold",
        fontfamily="serif",
        ha="center",
    )

    panels = [
        (axes[0], "ARI_mean", "ARI_se", "Adjusted Rand Index (ARI)"),
        (axes[1], "NMI_mean", "NMI_se", "Normalized Mutual Information (NMI)"),
    ]

    algos_present = list(df["algo"].unique())
    order = [a for a in _PLOT_ORDER if a in algos_present]
    order += [a for a in algos_present if a not in order]
    if "DAS" in order:
        order = [a for a in order if a != "DAS"] + ["DAS"]

    style_by_algo = {}
    for ax, mean_col, se_col, ylabel in panels:
        for algo in order:
            sub = df[df["algo"] == algo].sort_values("param_value")
            x = sub["param_value"].to_numpy(dtype=float)
            y = sub[mean_col].to_numpy(dtype=float)
            st = _style_for(algo)
            style_by_algo[algo] = st

            ax.plot(
                x, y,
                marker="o", markersize=5.5,
                color=st["color"], label=st["label"],
                linewidth=st["lw"], linestyle="-",
                markeredgewidth=0.8, zorder=st["zo"],
            )
            if show_band and se_col in sub.columns:
                se = sub[se_col].to_numpy(dtype=float)
                # Two-sided 95% CI half-width: SE * t_{0.975, n-1}
                if "n_sims" in sub.columns:
                    n = sub["n_sims"].to_numpy(dtype=float)
                    tcrit = np.array([
                        float(stats.t.ppf(0.975, max(ni - 1, 1)))
                        if np.isfinite(ni) and ni >= 2 else np.nan
                        for ni in n
                    ])
                    e = se * tcrit
                else:
                    e = se * 1.96
                ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(e)
                if np.any(ok):
                    ax.fill_between(
                        x[ok], (y - e)[ok], (y + e)[ok],
                        alpha=band_alpha, color=st["color"],
                        linewidth=0, zorder=st["zo"] - 1,
                    )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, axis="y", linestyle="-", linewidth=0.5, alpha=0.28)
        ax.grid(False, axis="x")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        unique_x = sorted(df["param_value"].dropna().unique())
        if unique_x and np.all(np.isfinite(unique_x)):
            if param_name in {"d", "K"}:
                ticks = [int(v) for v in unique_x]
                ax.set_xticks(ticks)
                ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%d"))
                if len(ticks) >= 16:
                    ax.tick_params(axis="x", labelsize=8.5)

    proxy_handles, labels_leg = [], []
    legend_order = [a for a in _PLOT_ORDER if a in style_by_algo]
    legend_order += [a for a in style_by_algo if a not in legend_order]
    for algo in legend_order:
        st = style_by_algo[algo]
        proxy_handles.append(
            mpl.lines.Line2D(
                [0], [0],
                color=st["color"], linestyle="-", linewidth=st["lw"],
                marker="o", markersize=5, markeredgewidth=0.8,
            )
        )
        labels_leg.append(st["label"])

    fig.legend(
        proxy_handles, labels_leg,
        ncol=min(len(labels_leg), 5),
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(ax_center, legend_y),
        columnspacing=1.0,
        handlelength=3.2,
        handletextpad=0.45,
        fontsize=9.5,
    )

    pdf = str(out_base) + ".pdf"
    png = str(out_base) + ".png"
    fig.savefig(pdf, **SAVEFIG_KW)
    fig.savefig(png, **SAVEFIG_KW)
    plt.close(fig)
    return pdf, png


def main():
    parser = argparse.ArgumentParser(
        description="Build/plot ARI–NMI segmentation recovery curves."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("build-csv", help="Scan pkls → CSV/tex (+ optional plot).")
    p1.add_argument("--dir", required=True)
    p1.add_argument("--file_pattern", default="exp_*.pkl")
    p1.add_argument("--show_band", action="store_true")
    p1.add_argument("--no_plot", action="store_true")

    p2 = sub.add_parser(
        "plot-csv",
        help="Re-plot from an existing CSV (stale; requires --allow-stale-csv).",
    )
    p2.add_argument("--csv", required=True)
    p2.add_argument("--out_fig", default=None)
    p2.add_argument("--show_band", action="store_true")
    p2.add_argument("--no_band", action="store_true")
    p2.add_argument(
        "--allow-stale-csv",
        action="store_true",
        help="Confirm you intentionally skip refreshing from pkls.",
    )

    args = parser.parse_args()

    if args.cmd == "build-csv":
        in_dir = Path(args.dir)
        df = build_long_csv_from_dir(in_dir, file_pattern=args.file_pattern)
        fixed_K = None
        sample = next(iter(sorted(in_dir.glob(args.file_pattern))), None)
        if sample is not None:
            with open(sample, "rb") as f:
                ep = (pickle.load(f).get("exp_params") or {})
            if ep.get("K") is not None:
                fixed_K = int(ep["K"])
        paths = export_segmentation_artifacts(df, in_dir, fixed_K=fixed_K)
        for k, v in paths.items():
            print(f"[OK] Saved {k} -> {v}")
        if not args.no_plot:
            pdf, png = plot_ari_nmi(
                paths["long"],
                in_dir / "curves_ari_nmi",
                show_band=True,
            )
            print(f"[OK] Saved plot -> {pdf}")
            print(f"[OK] Saved plot -> {png}")

    elif args.cmd == "plot-csv":
        if not args.allow_stale_csv:
            raise SystemExit(
                "plot-csv would reuse a possibly stale CSV.\n"
                "Prefer:  python3 analysis/plot_segmentation_metrics.py build-csv --dir <exp_dir>\n"
                "Override only if intentional: add --allow-stale-csv"
            )
        show_band = not args.no_band
        pdf, png = plot_ari_nmi(args.csv, args.out_fig, show_band=show_band)
        print(f"[OK] Saved plot -> {pdf}")
        print(f"[OK] Saved plot -> {png}")


if __name__ == "__main__":
    main()

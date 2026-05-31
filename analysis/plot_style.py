"""
Shared plot style, color palette, and label maps for all analysis scripts.
Import this module instead of duplicating constants across scripts.
"""

import matplotlib as mpl
import numpy as np


# ------------------------------------------------------------------
# Keys to skip when extracting comparator algorithms from pkl files
# ------------------------------------------------------------------
IGNORE_KEYS_DEFAULT = {
    "dast",
    "exp_params",
    "seed",
    "oracle_profits_impl",
    "covariate_overlap",
}

# ------------------------------------------------------------------
# Outlier handling: which comparators get trimmed/winsorized
# ------------------------------------------------------------------
DEFAULT_REMOVE_EXTREME = {
    "gmm-standard":    True,
    "kmeans-standard": True,
    "mst":             True,
    "clr-standard":    True,
    "dast_old":        True,
    "t_learner":       True,
    "x_learner":       True,
    "dr_learner":      True,
    "s_learner":       True,
    "policy_tree":     True,
}

# ------------------------------------------------------------------
# Color palette
# ------------------------------------------------------------------
DEFAULT_COLORS = {
    "kmeans-standard": "#FFD22FAF",
    "gmm-standard":    "#006135AF",
    "clr-standard":    "#F59134AF",
    "mst":             "#937860AF",
    "dast_old":        "#C2C2C2AE",
    "t_learner":       "#6BC735AF",
    "s_learner":       "#1F5BFFAF",
    "x_learner":       "#FF5832AD",
    "dr_learner":      "#7A5CFFAF",
    "policy_tree":     "#333333AF",
}

# ------------------------------------------------------------------
# Display labels
# ------------------------------------------------------------------
LABEL_MAP = {
    "kmeans-standard": "K-Means",
    "gmm-standard":    "GMM",
    "clr-standard":    "CLR",
    "mst":             "MST",
    "dast_old":        "DAST (old)",
    "t_learner":       "T-Learner",
    "s_learner":       "S-Learner",
    "x_learner":       "X-Learner",
    "dr_learner":      "DR-Learner",
    "policy_tree":     "Policy Tree",
    "causal_forest":   "Causal Forest",
}

# ------------------------------------------------------------------
# Axis / title labels indexed by experiment parameter name
# ------------------------------------------------------------------
X_LABEL_MAP = {
    "d":      "Dimension $d$",
    "K":      "Ground-truth number of clusters ($K$)",
    "delta":  r"Magnitude of interaction effects ($\delta$)",
    "Xnoise": r"Within-cluster covariate noise scale ($X_{\mathrm{noise}}$ std. scale)",
    "mahalanobis_sep": r"Target nearest-neighbor Mahalanobis separation",
}

# ------------------------------------------------------------------
# Default plot comparator order
# ------------------------------------------------------------------
COMPARATOR_ORDER = [
    "gmm-standard",
    "kmeans-standard",
    "mst",
    "clr-standard",
    "dast_old",
    # "t_learner",
    # "s_learner",
    # "x_learner",
    # "dr_learner",
    # "causal_forest",
]

# ------------------------------------------------------------------
# Demo / segmentation figure layout (single-column panel)
# ------------------------------------------------------------------
DEMO_FIGSIZE_2D = (5.2, 3.5)
DEMO_FIGSIZE_2D_DISCRETE = (6.2, 3.8)  # ylim (y_min-1, y_max+1); legend above y=1 band
DEMO_FIGSIZE_STACK = (11.0, 6.2)     # 2×2 Bernoulli demo composite
DEMO_FIGSIZE_3D = (6.5, 5.0)

# Discrete Bernoulli demo axis labels (combined + standalone figures)
DEMO_DISCRETE_XLABEL = "Monthly watch hours ($x$)"
DEMO_DISCRETE_YLABEL = "Subscribe ($y$)"
DEMO_SCATTER_SIZE = 32
DEMO_SCATTER_ALPHA = 0.82
BINARY_Y_PAD = 0.04  # room for markers at y in {0, 1} without clipping
def discrete_y_display_limits(y_min: float = 0.0, y_max: float = 1.0) -> tuple[float, float]:
    """Bernoulli panels: show (y_min - 1, y_max + 1) on the axis; ticks stay at 0 and 1."""
    return y_min - 1.0, y_max + 1.0


def discrete_legend_axes_y_floor(ax, *, pad: float = 0.05) -> float:
    """Axes-fraction lower bound for legend, just above the y=1 outcome band."""
    ymin, ymax = ax.get_ylim()
    span = ymax - ymin
    if span <= 0:
        return 0.70
    return (1.0 - ymin) / span + pad

SAVEFIG_KW = dict(
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.08,
    facecolor="white",
    edgecolor="none",
)


def demo_figsize_2d(*, discrete_outcome: bool = False) -> tuple:
    return DEMO_FIGSIZE_2D_DISCRETE if discrete_outcome else DEMO_FIGSIZE_2D


def demo_legend_style(*, ncol: int = 1) -> dict:
    """Legend styling; ncol=1 gives one line per segment entry."""
    return dict(
        ncol=ncol,
        fontsize=8.0,
        frameon=True,
        fancybox=False,
        framealpha=0.98,
        edgecolor="0.72",
        facecolor="white",
        handletextpad=0.4,
        labelspacing=0.35,
        borderpad=0.35,
    )


def layout_demo_2d(fig, *, discrete_outcome: bool) -> None:
    if discrete_outcome:
        fig.subplots_adjust(top=0.84, left=0.10, right=0.97, bottom=0.22)
    else:
        fig.subplots_adjust(top=0.88, left=0.12, right=0.97, bottom=0.14)


def layout_demo_stack(fig) -> None:
    fig.subplots_adjust(hspace=0.48, wspace=0.22,
                        top=0.94, left=0.08, right=0.97, bottom=0.20)


def style_binary_y_axis(ax, *, z_axis: bool = False, y_vals=None) -> None:
    """Bernoulli: ylim (y_min - 1, y_max + 1); tick labels only at 0 and 1."""
    if y_vals is not None:
        y = np.asarray(y_vals, dtype=float).ravel()
        y_min, y_max = float(y.min()), float(y.max())
    else:
        y_min, y_max = 0.0, 1.0
    lo, hi = discrete_y_display_limits(y_min, y_max)
    ticks = [0, 1]
    if z_axis:
        ax.set_zlim(lo, hi)
        ax.set_zticks(ticks)
    else:
        ax.set_ylim(lo, hi)
        ax.set_yticks(ticks)


def style_demo_2d_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", direction="out", length=3.5, width=0.8, pad=3)
    ax.set_axisbelow(True)
    ax.grid(True, which="major", linestyle="-", linewidth=0.5, alpha=0.22)


# ------------------------------------------------------------------
# Publication-ready rcParams
# ------------------------------------------------------------------
def set_plot_style():
    mpl.rcParams.update({
        "font.family":        "serif",
        "font.serif":         ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset":   "stix",
        "pdf.fonttype":       42,
        "ps.fonttype":        42,
        "axes.unicode_minus": False,

        "axes.labelsize":     11,
        "axes.titlesize":     11,
        "axes.titleweight":   "normal",
        "axes.labelweight":   "normal",
        "xtick.labelsize":    10,
        "ytick.labelsize":    10,
        "legend.fontsize":    9,

        "lines.linewidth":    1.6,
        "lines.markersize":   5.0,

        "axes.spines.top":    False,
        "axes.spines.right":  False,

        "axes.grid":          True,
        "grid.alpha":         0.22,
        "grid.linestyle":     "-",
        "grid.linewidth":     0.5,

        "figure.dpi":         150,
        "savefig.dpi":        300,
    })

"""
Shared plot style, color palette, and label maps for all analysis scripts.
Import this module instead of duplicating constants across scripts.
"""

import matplotlib as mpl


# ------------------------------------------------------------------
# Keys to skip when extracting comparator algorithms from pkl files
# ------------------------------------------------------------------
IGNORE_KEYS_DEFAULT = {
    "dast",
    "exp_params",
    "seed",
    "oracle_profits_impl",
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
# Publication-ready rcParams
# ------------------------------------------------------------------
def set_plot_style():
    mpl.rcParams.update({
        "font.family":       "serif",
        "font.serif":        ["Times New Roman", "Times", "DejaVu Serif"],
        "pdf.fonttype":      42,
        "ps.fonttype":       42,

        "axes.labelsize":    12,
        "axes.titlesize":    13,
        "xtick.labelsize":   11,
        "ytick.labelsize":   11,
        "legend.fontsize":   10,

        "lines.linewidth":   2.0,
        "lines.markersize":  5.5,

        "axes.spines.top":   False,
        "axes.spines.right": False,

        "axes.grid":         True,
        "grid.alpha":        0.25,
        "grid.linestyle":    "-",
        "grid.linewidth":    0.8,

        "figure.dpi":        120,
        "savefig.dpi":       600,
    })

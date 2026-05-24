import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
import os

# Markers for up to 8 actions
_ACTION_MARKERS = ['o', 'x', '^', 's', 'D', 'v', 'P', '*']


def plot_segmentation(labels, X, y_vec, D_vec, algo, M=None, tree=None, run_idx=None):
    """
    Visualize segmentation with:
      - Color per segment
      - Marker per action (supports multi-action)
      - 2D  when X has 1 feature  : x_0  vs outcome
      - 3D  when X has ≥2 features : x_0 vs x_1 vs outcome
      - Decision boundaries for tree-based methods (DAST, MST)
    """
    os.makedirs("figures", exist_ok=True)

    labels  = np.ravel(labels).astype(int)
    y_vec   = np.ravel(y_vec)
    D_vec   = np.ravel(D_vec).astype(int)   # ensure 1-D regardless of (N,1) input

    unique_labels  = sorted(np.unique(labels))
    unique_actions = sorted(np.unique(D_vec))

    cmap = plt.colormaps.get_cmap("Set1").resampled(max(len(unique_labels), 2))
    label_to_color  = {lab: cmap(i) for i, lab in enumerate(unique_labels)}
    action_to_marker = {a: _ACTION_MARKERS[i % len(_ACTION_MARKERS)]
                        for i, a in enumerate(unique_actions)}

    use_3d = X.ndim == 2 and X.shape[1] >= 2

    # ── 3-D plot ──────────────────────────────────────────────────────────────
    if use_3d:
        x0, x1 = X[:, 0], X[:, 1]

        fig = plt.figure(figsize=(10, 8))
        ax  = fig.add_subplot(111, projection='3d')

        for lab in unique_labels:
            for act in unique_actions:
                idx = (labels == lab) & (D_vec == act)
                if not np.any(idx):
                    continue
                ax.scatter(x0[idx], x1[idx], y_vec[idx],
                           color=label_to_color[lab],
                           marker=action_to_marker[act],
                           alpha=0.6, s=18,
                           label=f"Seg {lab}, D={act}")

        # Decision planes for tree splits – clipped to each node's bounding box
        if tree is not None and algo in ["dast", "mst"]:
            splits = _extract_splits_all_features(tree)
            x0_dmin, x0_dmax = x0.min(), x0.max()
            x1_dmin, x1_dmax = x1.min(), x1.max()
            y_dmin,  y_dmax  = y_vec.min(), y_vec.max()
            plane_colors = {0: 'red', 1: 'steelblue'}

            for feat, thresh, bbox in splits:
                # Clip bbox to actual data range
                x0_lo = max(bbox.get(0, (-np.inf, np.inf))[0], x0_dmin)
                x0_hi = min(bbox.get(0, (-np.inf, np.inf))[1], x0_dmax)
                x1_lo = max(bbox.get(1, (-np.inf, np.inf))[0], x1_dmin)
                x1_hi = min(bbox.get(1, (-np.inf, np.inf))[1], x1_dmax)

                if feat == 0:
                    # Plane x0 = thresh, spanning clipped x1 range and full y
                    g1, gy = np.meshgrid(np.linspace(x1_lo, x1_hi, 3),
                                         np.linspace(y_dmin, y_dmax, 3))
                    g0 = np.full_like(g1, thresh)
                    ax.plot_surface(g0, g1, gy, alpha=0.15,
                                    color=plane_colors.get(feat, 'gray'))
                elif feat == 1:
                    # Plane x1 = thresh, spanning clipped x0 range and full y
                    g0, gy = np.meshgrid(np.linspace(x0_lo, x0_hi, 3),
                                         np.linspace(y_dmin, y_dmax, 3))
                    g1_surf = np.full_like(g0, thresh)
                    ax.plot_surface(g0, g1_surf, gy, alpha=0.15,
                                    color=plane_colors.get(feat, 'gray'))

        ax.set_xlabel("x_0"); ax.set_ylabel("x_1"); ax.set_zlabel("outcome")
        ax.set_title(f"{algo.upper()}-Based Segmentation (3D), M={M}")

    # ── 2-D plot ──────────────────────────────────────────────────────────────
    else:
        x0 = X[:, 0] if X.ndim == 2 else np.ravel(X)

        fig, ax = plt.subplots(figsize=(8, 6))

        for lab in unique_labels:
            for act in unique_actions:
                idx = (labels == lab) & (D_vec == act)
                if not np.any(idx):
                    continue
                ax.scatter(x0[idx], y_vec[idx],
                           color=label_to_color[lab],
                           marker=action_to_marker[act],
                           alpha=0.7, s=20,
                           label=f"Seg {lab}, D={act}")

        if tree is not None and algo in ["dast", "mst"]:
            x0_dmin, x0_dmax = x0.min(), x0.max()
            y_dmin,  y_dmax  = y_vec.min(), y_vec.max()
            for feat, thresh, bbox in _extract_splits_all_features(tree):
                if feat != 0:
                    continue
                # Clip the line's y-extent to customers in this node's x0 region
                x0_lo = max(bbox.get(0, (-np.inf, np.inf))[0], x0_dmin)
                x0_hi = min(bbox.get(0, (-np.inf, np.inf))[1], x0_dmax)
                mask  = (x0 >= x0_lo) & (x0 <= x0_hi)
                y_lo  = y_vec[mask].min() if mask.any() else y_dmin
                y_hi  = y_vec[mask].max() if mask.any() else y_dmax
                ax.plot([thresh, thresh], [y_lo, y_hi],
                        color='red', linestyle='--', linewidth=1.8,
                        alpha=0.8, label='split')

        ax.set_xlabel("x_0");  ax.set_ylabel("outcome")
        ax.set_title(f"{algo.upper()}-Based Segmentation, M={M}")
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
        ax.grid(True, alpha=0.4)

    # Deduplicated legend
    handles, leg_labels = ax.get_legend_handles_labels()
    by_label = dict(zip(leg_labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=7, ncol=2, loc='best')

    plt.tight_layout()
    if run_idx is None:
        out_name = f"{algo}_segmentation_{M}.png"
    else:
        out_name = f"{algo}_segmentation_{M}_run{run_idx}.png"
    plt.savefig(os.path.join("figures", out_name), dpi=300)
    plt.close()


def _extract_splits_all_features(tree):
    """Return list of (feature_idx, threshold, bbox) for every internal split.

    bbox is a dict {feature_idx: (lo, hi)} representing the axis-aligned region
    that the node is responsible for.  Infinite bounds mean "no constraint yet".
    The split plane / line should only be drawn within this region.
    """
    splits = []

    def traverse(node, bbox):
        if node is None or node.is_leaf:
            return
        feat   = node.split_feature
        thresh = node.split_threshold
        splits.append((feat, thresh, dict(bbox)))

        lo, hi = bbox.get(feat, (-np.inf, np.inf))

        left_bbox = dict(bbox)
        left_bbox[feat] = (lo, thresh)
        traverse(node.left, left_bbox)

        right_bbox = dict(bbox)
        right_bbox[feat] = (thresh, hi)
        traverse(node.right, right_bbox)

    traverse(tree.root, {})
    return splits


def extract_tree_splits(tree):
    """Return sorted split thresholds on feature 0 only (legacy 1-D helper)."""
    return sorted(t for f, t, _ in _extract_splits_all_features(tree) if f == 0)



def plot_ground_truth(df, title="Ground-Truth Segmentation",
                      segment_col='true_segment_id',
                      x_col='x_0', x2_col='x_1',
                      y_col='outcome', D_col='D_i',
                      run_idx=None, out_dir="figures"):
    """
    Plot ground-truth segmentation.
      - Color per true segment
      - Marker per action (multi-action supported)
      - 2D when x2_col is absent: x_0 vs outcome
      - 3D when x2_col exists  : x_0 vs x_1 vs outcome

    Parameters
    ----------
    df         : DataFrame from pop.to_dataframe()
    x2_col     : second covariate column for 3D; set to None to force 2D
    """
    os.makedirs(out_dir, exist_ok=True)

    segments       = sorted(df[segment_col].unique())
    unique_actions = sorted(df[D_col].unique())

    cmap = plt.colormaps.get_cmap("Set1").resampled(max(len(segments), 2))
    seg_to_color    = {k: cmap(i) for i, k in enumerate(segments)}
    action_to_marker = {a: _ACTION_MARKERS[i % len(_ACTION_MARKERS)]
                        for i, a in enumerate(unique_actions)}

    use_3d = (x2_col is not None) and (x2_col in df.columns)

    if use_3d:
        fig = plt.figure(figsize=(10, 8))
        ax  = fig.add_subplot(111, projection='3d')

        for k in segments:
            for a in unique_actions:
                subset = df[(df[segment_col] == k) & (df[D_col] == a)]
                if subset.empty:
                    continue
                ax.scatter(subset[x_col], subset[x2_col], subset[y_col],
                           color=seg_to_color[k],
                           marker=action_to_marker[a],
                           alpha=0.6, s=18,
                           label=f"Seg {k}, D={a}")

        ax.set_xlabel(f"${x_col}$",  fontsize=12)
        ax.set_ylabel(f"${x2_col}$", fontsize=12)
        ax.set_zlabel(f"${y_col}$",  fontsize=12)

    else:
        fig, ax = plt.subplots(figsize=(8, 6))

        for k in segments:
            for a in unique_actions:
                subset = df[(df[segment_col] == k) & (df[D_col] == a)]
                if subset.empty:
                    continue
                ax.scatter(subset[x_col], subset[y_col],
                           color=seg_to_color[k],
                           marker=action_to_marker[a],
                           alpha=0.7, s=20,
                           label=f"Seg {k}, D={a}")

        ax.set_xlabel(f"${x_col}$", fontsize=16)
        ax.set_ylabel(f"${y_col}$", fontsize=16)
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
        ax.grid(True, alpha=0.4)

    ax.set_title(title, fontsize=16)

    handles, leg_labels = ax.get_legend_handles_labels()
    by_label = dict(zip(leg_labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=7, ncol=2, loc='best')

    plt.tight_layout()
    if run_idx is None:
        fname = os.path.join(out_dir, "ground_truth_plot.png")
    else:
        fname = os.path.join(out_dir, f"ground_truth_plot_run{run_idx}.png")
    plt.savefig(fname, dpi=300)
    plt.close()


def plot_implementation_clustering(implement_customers, algo, title=None):
    """
    Plot implementation customers with their clustering results and assigned treatment.
    
    Left plot: True segments with potential outcomes under BOTH treatments (counterfactual)
    Right plot: Estimated segments with algorithm-assigned treatment (factual)
    
    Parameters:
    - implement_customers: list of Customer_implement objects
    - algo: algorithm name (e.g., 'dast', 'kmeans-da', etc.)
    - title: optional custom title
    """
    # Extract data from customers
    X = np.array([cust.x[0] if len(cust.x) > 0 else 0 for cust in implement_customers])
    true_segment_ids = np.array([cust.true_segment.segment_id for cust in implement_customers])
    est_segment_ids = np.array([cust.est_segment[algo].segment_id if cust.est_segment[algo] is not None else -1 for cust in implement_customers])
    assigned_treatments = np.array([cust.implement_action for cust in implement_customers])  # Treatment assigned by algo
    
    # Generate ground truth potential outcomes for BOTH treatments
    Y_0 = np.array([cust.expected_outcome(0) for cust in implement_customers])
    Y_1 = np.array([cust.expected_outcome(1) for cust in implement_customers])
    
    # Actual observed outcome (based on assigned treatment)
    Y_observed = np.array([cust.y for cust in implement_customers])
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # Define colors and markers
    colors = ['red', 'green', 'blue', 'purple', 'orange', 'brown']
    markers = {0: 'o', 1: 'x'}
    marker_sizes = {0: 60, 1: 80}  # Make 'x' larger for better visibility
    
    # Left subplot: True segments with BOTH potential outcomes (counterfactual)
    ax1 = axes[0]
    unique_true_segments = sorted(np.unique(true_segment_ids))
    for seg_id in unique_true_segments:
        for treatment in [0, 1]:
            mask = (true_segment_ids == seg_id)
            if np.any(mask):
                Y_treatment = Y_0[mask] if treatment == 0 else Y_1[mask]
                # For 'o' marker (filled), use edgecolors; for 'x' marker (unfilled), don't use edgecolors
                if treatment == 0:
                    ax1.scatter(
                        X[mask],
                        Y_treatment,
                        color=colors[seg_id % len(colors)],
                        marker=markers[treatment],
                        alpha=0.6,
                        s=marker_sizes[treatment],
                        linewidths=1,
                        label=f"True Seg {seg_id}, T={treatment} (potential)"
                    )
                else:  # treatment == 1, 'x' marker
                    ax1.scatter(
                        X[mask],
                        Y_treatment,
                        color=colors[seg_id % len(colors)],
                        marker=markers[treatment],
                        alpha=0.6,
                        s=marker_sizes[treatment],
                        linewidths=2,  # Make 'x' marker thicker
                        label=f"True Seg {seg_id}, T={treatment} (potential)"
                    )
    
    ax1.set_xlabel("Covariate $x_0$", fontsize=14)
    ax1.set_ylabel("Potential Outcome $y$", fontsize=14)
    ax1.set_title(f"True Segments + Potential Outcomes Under Both Treatments\n(Ground Truth Counterfactuals)", fontsize=15, fontweight='bold')
    ax1.axhline(0, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    ax1.legend(loc='best', fontsize=9, framealpha=0.9, ncol=2)
    ax1.grid(True, alpha=0.3)
    
    # Right subplot: Estimated segments with algo-assigned treatment (factual)
    ax2 = axes[1]
    unique_est_segments = sorted(np.unique(est_segment_ids[est_segment_ids >= 0]))
    for seg_id in unique_est_segments:
        for treatment in [0, 1]:
            mask = (est_segment_ids == seg_id) & (assigned_treatments == treatment)
            if np.any(mask):
                # For 'o' marker (filled), use edgecolors; for 'x' marker (unfilled), don't use edgecolors
                if treatment == 0:
                    ax2.scatter(
                        X[mask],
                        Y_observed[mask],
                        color=colors[seg_id % len(colors)],
                        marker=markers[treatment],
                        alpha=0.7,
                        s=marker_sizes[treatment],
                        linewidths=1,
                        label=f"Est Seg {seg_id}, Assigned T={treatment}"
                    )
                else:  # treatment == 1, 'x' marker
                    ax2.scatter(
                        X[mask],
                        Y_observed[mask],
                        color=colors[seg_id % len(colors)],
                        marker=markers[treatment],
                        alpha=0.7,
                        s=marker_sizes[treatment],
                        linewidths=2,  # Make 'x' marker thicker
                        label=f"Est Seg {seg_id}, Assigned T={treatment}"
                    )
    
    ax2.set_xlabel("Covariate $x_0$", fontsize=14)
    ax2.set_ylabel("Observed Outcome $y$", fontsize=14)
    ax2.set_title(f"Estimated Segments + Algorithm-Assigned Treatment\n({algo.upper()} Policy)", fontsize=15, fontweight='bold')
    ax2.axhline(0, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    ax2.legend(loc='best', fontsize=9, framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"figures/implementation_{algo}_clustering.png", dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory


def plot_bernoulli_prob_histogram(customers, action_num, run_idx=0, out_dir="figures"):
    """
    Plot P(Y=1 | x_i, D=a) distributions broken down by segment × action.

    Layout : rows = segments (K),  cols = actions (A)
    Each cell shows:
      - histogram + KDE of P(Y=1|x_i, D=a) for customers in that segment
      - red dashed vertical line at the anchor probability p*_{k,a}  (evaluated at x_mean)
      - green border / star in title when this action is the segment's optimal action

    Parameters
    ----------
    customers  : list of Customer_pilot or Customer_implement
    action_num : number of actions
    run_idx    : simulation index (used in filename)
    out_dir    : output directory
    """
    from scipy.stats import gaussian_kde
    from ground_truth import _sigmoid

    os.makedirs(out_dir, exist_ok=True)

    action_colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e',
                     '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

    # Group customers by segment
    seg_map = {}
    for c in customers:
        sid = c.true_segment.segment_id
        seg_map.setdefault(sid, []).append(c)
    seg_ids = sorted(seg_map.keys())
    K = len(seg_ids)
    A = action_num

    fig, axes = plt.subplots(K, A,
                             figsize=(4.0 * A, 3.2 * K),
                             squeeze=False)

    for row, sid in enumerate(seg_ids):
        seg_custs = seg_map[sid]
        seg       = seg_custs[0].true_segment
        best_a    = seg.action          # optimal action at x_mean

        for col in range(A):
            ax    = axes[row, col]
            color = action_colors[col % len(action_colors)]

            probs = np.array([c.expected_outcome(col) for c in seg_custs])

            # Histogram
            ax.hist(probs, bins=30, alpha=0.45, color=color,
                    edgecolor='none', density=True)

            # KDE overlay
            if probs.std() > 1e-6:
                kde = gaussian_kde(probs, bw_method='scott')
                xs  = np.linspace(0, 1, 300)
                ax.plot(xs, kde(xs), color=color, linewidth=1.8)

            # Mean of distribution
            ax.axvline(probs.mean(), color='red', linestyle='--',
                       linewidth=1.4, alpha=0.9,
                       label=f"mean={probs.mean():.3f}")

            # Column header: "D = a" only on the top row
            if row == 0:
                ax.set_title(f"D = {col}", fontsize=11, fontweight='bold', pad=6)

            # Row label: "Seg k  (★ D=best_a)" only on the leftmost column
            if col == 0:
                ax.set_ylabel(
                    f"Seg {sid} \n\nDensity",
                    fontsize=9, labelpad=6
                )
            else:
                ax.set_ylabel("")

            # x-axis label only on the bottom row
            if row == K - 1:
                ax.set_xlabel("P(Y=1 | x, D=a)", fontsize=9)
            else:
                ax.set_xlabel("")

            # Green border for best action column within this row
            for spine in ax.spines.values():
                spine.set_linewidth(2.0 if col == best_a else 0.6)
                spine.set_edgecolor('#2ca02c' if col == best_a else '#cccccc')

            ax.set_xlim(0, 1)
            ax.legend(frameon=False, fontsize=8)
            ax.grid(True, axis='y', alpha=0.25)

    fig.suptitle(f"Bernoulli Probability by Segment × Action",
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()

    fname = os.path.join(out_dir, f"bernoulli_prob_hist_run{run_idx}.png")
    plt.savefig(fname, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"[plot] Bernoulli prob histogram saved -> {fname}")

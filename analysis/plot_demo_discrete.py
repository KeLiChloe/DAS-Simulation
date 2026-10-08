"""
Publication-quality figures for the discrete Bernoulli demo (analysis/run_demo_discrete.py).

Self-contained: does not import from plot.py.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle

from plot_style import (
    DEMO_DISCRETE_XLABEL,
    DEMO_DISCRETE_YLABEL,
    DEMO_FIGSIZE_STACK,
    DEMO_FIGSIZE_STACK_2x3,
    DEMO_SCATTER_ALPHA,
    DEMO_SCATTER_SIZE,
    LABEL_MAP,
    SAVEFIG_KW,
    demo_figsize_2d,
    demo_legend_style,
    discrete_y_display_limits,
    discrete_legend_axes_y_floor,
    layout_demo_2d,
    layout_demo_stack,
    layout_demo_stack_2x3,
    style_binary_y_axis,
    style_demo_2d_axes,
)

_ACTION_MARKERS = ["o", "+", "^", "s", "D", "v", "P", "*"]

_SPLIT_LINE_COLOR = "black"
_SPLIT_LINE_ALPHA = 0.95
_SPLIT_LINE_LW = 1.2

# Manual legend geometry (axes fraction, tuned for 2×2 demo panels)
_LEGEND_PAD = 0.014
_LEGEND_ROW_H_MIN = 0.052
_LEGEND_ROW_H_MAX = 0.080
_LEGEND_SWATCH_W = 0.024
_LEGEND_SWATCH_H = 0.026
_LEGEND_SWATCH_GAP = 0.010
_LEGEND_TEXT_GAP = 0.020
_LEGEND_LINE_W = 0.048
_LEGEND_BOX_RIGHT = 0.008
_LEGEND_WIDTH_BUFFER = 0.020
_LEGEND_MARKER_W = 0.018
_LEGEND_MARKER_MS = 5.0


def _axes_text_width(ax, text: str, fontsize: float) -> float:
    """Measure rendered text width in axes-fraction coordinates."""
    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    if renderer is None:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
    tmp = ax.text(
        0.0, 0.0, text, fontsize=fontsize, transform=ax.transAxes,
        alpha=0.0, clip_on=False,
    )
    try:
        bbox = tmp.get_window_extent(renderer=renderer).transformed(ax.transAxes.inverted())
        return float(bbox.width)
    finally:
        tmp.remove()


def _seg_display_num(seg_id) -> int:
    """1-based segment index for figure labels (data IDs stay 0-based)."""
    return int(seg_id) + 1


def _true_seg_label(seg_id) -> str:
    return f"True Seg {_seg_display_num(seg_id)}"


def _learned_seg_label(seg_id) -> str:
    return f"Learned Seg {_seg_display_num(seg_id)}"


def _est_action_math(display_k: int, est_act) -> str:
    """Mathtext for segment-wise estimated action, e.g. $\\widehat{a}_1=1$."""
    return rf"$\widehat{{a}}_{{{int(display_k)}}}={int(est_act)}$"


def _kmeans_seg_ids_by_centroid_x(model, feature_idx: int = 0) -> list[int]:
    """Cluster ids ordered by increasing centroid coordinate (left to right on x-axis)."""
    centers = np.asarray(model.cluster_centers_)
    if centers.ndim == 1:
        centers = centers.reshape(-1, 1)
    order = np.argsort(centers[:, feature_idx])
    return [int(i) for i in order]


def _kmeans_legend_seg_order(model, labels) -> list[int]:
    """Left-to-right cluster ids, keeping only labels present in the plotted sample."""
    present = set(np.ravel(labels).astype(int))
    return [s for s in _kmeans_seg_ids_by_centroid_x(model) if s in present]


def _row_width_swatch_label(ax, label: str, fs: float) -> float:
    return (
        _LEGEND_PAD + _LEGEND_SWATCH_W + _LEGEND_SWATCH_GAP
        + _axes_text_width(ax, label, fs) + _LEGEND_PAD + _LEGEND_WIDTH_BUFFER
    )


def _row_width_gt(ax, seg_id, opt_act, fs: float) -> float:
    fs_opt = fs - 0.35
    seg_text = _true_seg_label(seg_id)
    opt_text = f"optimal action: {opt_act}"
    return (
        _LEGEND_PAD + _LEGEND_SWATCH_W + _LEGEND_SWATCH_GAP
        + _axes_text_width(ax, seg_text, fs) + _LEGEND_TEXT_GAP
        + _axes_text_width(ax, opt_text, fs_opt) + _LEGEND_TEXT_GAP
        + _LEGEND_MARKER_W + _LEGEND_PAD + _LEGEND_WIDTH_BUFFER
    )


def _row_width_learned_seg(ax, seg_text: str, display_k: int, est_act, fs: float) -> float:
    fs_est = fs - 0.35
    est_text = _est_action_math(display_k, est_act)
    return (
        _LEGEND_PAD + _LEGEND_SWATCH_W + _LEGEND_SWATCH_GAP
        + _axes_text_width(ax, seg_text, fs) + _LEGEND_TEXT_GAP
        + _axes_text_width(ax, est_text, fs_est) + _LEGEND_TEXT_GAP
        + _LEGEND_MARKER_W + _LEGEND_PAD + _LEGEND_WIDTH_BUFFER
    )


def _row_width_split(ax, label: str, fs: float) -> float:
    return (
        _LEGEND_PAD + _LEGEND_LINE_W + _LEGEND_TEXT_GAP
        + _axes_text_width(ax, label, fs) + _LEGEND_PAD + _LEGEND_WIDTH_BUFFER
    )


def _draw_legend_rect(ax, x: float, y: float, color) -> None:
    ax.add_patch(Rectangle(
        (x, y - _LEGEND_SWATCH_H / 2), _LEGEND_SWATCH_W, _LEGEND_SWATCH_H,
        transform=ax.transAxes,
        facecolor=color, edgecolor="0.40", linewidth=0.55,
        clip_on=False, zorder=11,
    ))


def _draw_legend_box(ax, x_left: float, y_bot: float, box_w: float, y_top: float) -> None:
    ax.add_patch(FancyBboxPatch(
        (x_left, y_bot), box_w, y_top - y_bot,
        transform=ax.transAxes, boxstyle="square,pad=0",
        facecolor="white", edgecolor="black", linewidth=0.75,
        zorder=10, clip_on=False,
    ))


def save_demo_figure(fig, out_path: str | Path) -> tuple[Path, Path]:
    """Save the same figure as PNG (preview) and PDF (vector, for LaTeX)."""
    stem = Path(out_path).with_suffix("")
    png_path = stem.with_suffix(".png")
    pdf_path = stem.with_suffix(".pdf")
    os.makedirs(stem.parent or ".", exist_ok=True)
    fig.savefig(png_path, **SAVEFIG_KW)
    fig.savefig(pdf_path, **SAVEFIG_KW)
    return png_path, pdf_path


def plot_kmeans_silhouette_elbow(
    Ms,
    inertias,
    silhouettes,
    *,
    elbow_M: int,
    silhouette_M: int,
    out_path: str,
) -> tuple[Path, Path]:
    """Side-by-side $k$-means model selection: silhouette (left) and elbow (right)."""
    Ms = np.asarray(Ms, dtype=int)
    inertias = np.asarray(inertias, dtype=float)
    silhouettes = np.asarray(silhouettes, dtype=float)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(9.2, 3.6))

    # Left: silhouette (M >= 2)
    sil_ok = np.isfinite(silhouettes)
    Ms_s = Ms[sil_ok]
    sil_s = silhouettes[sil_ok]
    ax_l.plot(Ms_s, sil_s, color="0.25", linewidth=1.8, marker="o", markersize=6, zorder=3)
    if len(Ms_s):
        pick_l = Ms_s == int(silhouette_M)
        if np.any(pick_l):
            ax_l.scatter(
                Ms_s[pick_l], sil_s[pick_l],
                s=90, color="#E41A1C", zorder=4,
                label=rf"silhouette $M={int(silhouette_M)}$",
            )
    ax_l.set_xlabel(r"Number of clusters ($M$)")
    ax_l.set_ylabel("Silhouette score")
    ax_l.set_xticks(list(Ms_s) if len(Ms_s) else list(Ms))
    ax_l.legend(frameon=True, fancybox=False, framealpha=0.98, edgecolor="0.72")
    style_demo_2d_axes(ax_l)
    ax_l.set_title("(a) Silhouette", fontsize=plt.rcParams.get("axes.titlesize", 11))

    # Right: elbow / inertia (M from 1)
    ax_r.plot(Ms, inertias, color="0.25", linewidth=1.8, marker="o", markersize=6, zorder=3)
    ax_r.plot(
        [Ms[0], Ms[-1]], [inertias[0], inertias[-1]],
        color="0.65", linestyle="--", linewidth=1.1, zorder=2, label="chord",
    )
    pick_r = Ms == int(elbow_M)
    if np.any(pick_r):
        ax_r.scatter(
            Ms[pick_r], inertias[pick_r],
            s=90, color="#E41A1C", zorder=4,
            label=rf"elbow $M={int(elbow_M)}$",
        )
    ax_r.set_xlabel(r"Number of clusters ($M$)")
    ax_r.set_ylabel(r"Inertia (WCSS)")
    ax_r.set_xticks(list(Ms))
    ax_r.legend(frameon=True, fancybox=False, framealpha=0.98, edgecolor="0.72")
    style_demo_2d_axes(ax_r)
    ax_r.set_title("(b) Elbow", fontsize=plt.rcParams.get("axes.titlesize", 11))

    fig.tight_layout()
    paths = save_demo_figure(fig, out_path)
    plt.close(fig)
    return paths


def _extract_splits_all_features(tree):
    """Return list of (feature_idx, threshold, bbox) for every internal split."""
    splits = []

    def traverse(node, bbox):
        if node is None or node.is_leaf:
            return
        feat = node.split_feature
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


def algo_panel_title(algo: str, M: int) -> str:
    if algo == "dast":
        name = "DAS"
    elif algo in ("kmeans-standard", "kmeans"):
        name = r"$k$-means"
    else:
        name = LABEL_MAP.get(algo, algo.replace("-", " ").title())
    return f"{name}, $M={M}$"


def _label_color_map(seg_ids, segment_colors=None, segment_cmap_name: str = "Set2"):
    seg_ids = list(seg_ids)
    if segment_colors is not None:
        colors = list(segment_colors)
        return {s: colors[i % len(colors)] for i, s in enumerate(seg_ids)}
    cmap = plt.colormaps.get_cmap(segment_cmap_name).resampled(max(len(seg_ids), 2))
    return {s: cmap(i) for i, s in enumerate(seg_ids)}


def _action_marker_map(unique_actions):
    acts = sorted(unique_actions)
    return {a: _ACTION_MARKERS[i % len(_ACTION_MARKERS)] for i, a in enumerate(acts)}


def _draw_discrete_scatter(ax, x0, y_vec, labels, D_vec, seg_to_color, action_to_marker):
    labels = np.ravel(labels).astype(int)
    D_vec = np.ravel(D_vec).astype(int)
    y_vec = np.ravel(y_vec)
    x0 = np.ravel(x0)
    seg_ids = sorted(np.unique(labels))
    acts = sorted(np.unique(D_vec))
    for seg in seg_ids:
        color = seg_to_color[seg]
        for act in acts:
            idx = (labels == seg) & (D_vec == act)
            if not np.any(idx):
                continue
            mk = action_to_marker[act]
            ax.scatter(
                x0[idx], y_vec[idx],
                color=color, marker=mk, s=DEMO_SCATTER_SIZE, alpha=DEMO_SCATTER_ALPHA,
                linewidths=0.8 if mk in ("x", "+") else 0.0,
                edgecolors=color if mk in ("x", "+") else "none",
                zorder=3,
            )


def _draw_discrete_splits(ax, tree, x0, y_vec):
    if tree is None:
        return None
    x0_dmin, x0_dmax = float(x0.min()), float(x0.max())
    # Target range: (y_min - 0.5) to 1.5  — matches style_binary_y_axis limits
    y_lo_draw, _ = discrete_y_display_limits(float(y_vec.min()), float(y_vec.max()))
    y_hi_draw = 1.5
    split_handle = None
    for feat, thresh, bbox in _extract_splits_all_features(tree):
        if feat != 0:
            continue
        x0_lo = max(bbox.get(0, (-np.inf, np.inf))[0], x0_dmin)
        x0_hi = min(bbox.get(0, (-np.inf, np.inf))[1], x0_dmax)
        (line,) = ax.plot(
            [thresh, thresh], [y_lo_draw, y_hi_draw],
            color=_SPLIT_LINE_COLOR, linestyle="--", linewidth=_SPLIT_LINE_LW,
            alpha=_SPLIT_LINE_ALPHA, zorder=2,
        )
        split_handle = line
    if split_handle is not None:
        split_handle.set_label("split")
    return split_handle


def _legend_marker_axes(ax, x_ax, y_ax, act, color, action_to_marker, *, ms=None):
    """Treatment marker at axes-fraction (x_ax, y_ax)."""
    if ms is None:
        ms = _LEGEND_MARKER_MS
    mk = action_to_marker[act]
    ax.plot(
        [x_ax], [y_ax], linestyle="None", marker=mk, color=color,
        markerfacecolor="none" if mk in ("x", "+") else color,
        markeredgecolor=color, markeredgewidth=0.75, markersize=ms,
        transform=ax.transAxes, clip_on=False, zorder=11,
    )


def _legend_layout(ax, n_rows: int):
    fs = demo_legend_style(ncol=1)["fontsize"]
    y_top = 1.02
    y_floor = discrete_legend_axes_y_floor(ax)
    row_h = min(
        _LEGEND_ROW_H_MAX,
        max(_LEGEND_ROW_H_MIN, (y_top - y_floor - 2 * _LEGEND_PAD) / max(n_rows, 1)),
    )
    box_h = n_rows * row_h + 2 * _LEGEND_PAD
    y_bot = max(y_top - box_h, y_floor)
    return fs, y_top, y_bot, row_h


def _draw_learned_seg_legend_row(
    ax, x_left: float, y: float, seg_id, color, est_act, action_to_marker, fs: float,
    *, seg_text: str | None = None, display_k: int | None = None,
) -> None:
    if seg_text is None:
        seg_text = _learned_seg_label(seg_id)
    k = _seg_display_num(seg_id) if display_k is None else int(display_k)
    est_text = _est_action_math(k, est_act)
    fs_est = fs - 0.35
    x = x_left + _LEGEND_PAD
    _draw_legend_rect(ax, x, y, color)
    x_seg = x + _LEGEND_SWATCH_W + _LEGEND_SWATCH_GAP
    ax.text(
        x_seg, y, seg_text, transform=ax.transAxes, fontsize=fs, color="black",
        va="center", ha="left", zorder=11, clip_on=False,
    )
    x_est = x_seg + _axes_text_width(ax, seg_text, fs) + _LEGEND_TEXT_GAP
    ax.text(
        x_est, y, est_text, transform=ax.transAxes,
        fontsize=fs_est, color="black", va="center", ha="left", zorder=11, clip_on=False,
    )
    x_mk = x_est + _axes_text_width(ax, est_text, fs_est) + _LEGEND_TEXT_GAP
    _legend_marker_axes(ax, x_mk, y, est_act, "black", action_to_marker)


def _draw_gt_legend_row(
    ax, x_left: float, y: float, seg_id, color, opt_act, action_to_marker, fs: float,
) -> None:
    seg_text = _true_seg_label(seg_id)
    opt_text = f"optimal action: {opt_act}"
    fs_opt = fs - 0.35
    x = x_left + _LEGEND_PAD
    _draw_legend_rect(ax, x, y, color)
    x_seg = x + _LEGEND_SWATCH_W + _LEGEND_SWATCH_GAP
    ax.text(
        x_seg, y, seg_text, transform=ax.transAxes, fontsize=fs, color="black",
        va="center", ha="left", zorder=11, clip_on=False,
    )
    x_opt = x_seg + _axes_text_width(ax, seg_text, fs) + _LEGEND_TEXT_GAP
    ax.text(
        x_opt, y, opt_text, transform=ax.transAxes,
        fontsize=fs_opt, color="black", va="center", ha="left", zorder=11, clip_on=False,
    )
    x_mk = x_opt + _axes_text_width(ax, opt_text, fs_opt) + _LEGEND_TEXT_GAP
    _legend_marker_axes(ax, x_mk, y, opt_act, "black", action_to_marker)


def _draw_split_legend_row(ax, x_left: float, y: float, handle, label: str, fs: float) -> None:
    x0 = x_left + _LEGEND_PAD
    x1 = x0 + _LEGEND_LINE_W
    ax.plot(
        [x0, x1], [y, y], transform=ax.transAxes,
        color=handle.get_color(), linestyle=handle.get_linestyle(),
        linewidth=1.2, clip_on=False, zorder=11,
    )
    ax.text(
        x1 + _LEGEND_TEXT_GAP, y, label, transform=ax.transAxes,
        fontsize=fs - 0.35, color="black", va="center", ha="left", zorder=11, clip_on=False,
    )


def build_learned_segment_actions(
    labels,
    X,
    D_vec,
    y_vec,
    *,
    method: str = "diff_in_means",
    gamma=None,
    action_num: int | None = None,
    include_interactions: bool = False,
) -> dict[int, int]:
    """Per learned segment: same action rule as fitted segmentation (see action_estimation.py)."""
    from utils import estimate_segment_parameters

    labels = np.ravel(labels).astype(int)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    D_vec = np.ravel(D_vec).astype(int)
    y_vec = np.ravel(y_vec)
    out: dict[int, int] = {}
    for seg in sorted(np.unique(labels)):
        idx = np.where(labels == seg)[0]
        if len(idx) == 0:
            out[int(seg)] = 0
            continue
        _, act = estimate_segment_parameters(
            X, D_vec, y_vec,
            method=method,
            gamma=gamma,
            indices=idx,
            action_num=action_num,
            include_interactions=include_interactions,
        )
        out[int(seg)] = 0 if act == 404 else int(act)
    return out


def _build_segment_legend(
    ax, seg_ids, seg_to_color, _unique_actions, action_to_marker,
    extra_handles=None, *, show_treatments=True,
    seg_estimated_actions: dict | None = None,
    seg_legend_labels: dict[int, str] | None = None,
    legend_x_right: float | None = None,
    legend_y_top: float | None = None,
    width_trim: float = 0.0,
    y_floor_pad: float | None = None,
    row_h_min: float | None = None,
    row_h_max: float | None = None,
):
    """Learned-segment legend: swatch + Learned Seg k + estimated action + marker; optional split."""
    if show_treatments:
        raise NotImplementedError("Treatment rows are only shown in the pilot panel.")
    n_segs = len(seg_ids)
    n_rows = n_segs + (1 if extra_handles else 0)
    fs = demo_legend_style(ncol=1)["fontsize"]
    y_top = 1.02 if legend_y_top is None else legend_y_top
    floor_kw = {} if y_floor_pad is None else {"pad": y_floor_pad}
    y_floor = discrete_legend_axes_y_floor(ax, **floor_kw)
    rmin = _LEGEND_ROW_H_MIN if row_h_min is None else row_h_min
    rmax = _LEGEND_ROW_H_MAX if row_h_max is None else row_h_max
    row_h = min(
        rmax,
        max(rmin, (y_top - y_floor - 2 * _LEGEND_PAD) / max(n_rows, 1)),
    )
    box_h = n_rows * row_h + 2 * _LEGEND_PAD
    y_bot = max(y_top - box_h, y_floor)

    def _seg_label(seg) -> str:
        if seg_legend_labels is not None and seg in seg_legend_labels:
            return seg_legend_labels[seg]
        return _learned_seg_label(seg)

    if seg_estimated_actions is None:
        row_widths = [_row_width_swatch_label(ax, _seg_label(seg), fs) for seg in seg_ids]
    else:
        row_widths = [
            _row_width_learned_seg(
                ax, _seg_label(seg), i + 1, seg_estimated_actions.get(seg, 0), fs,
            )
            for i, seg in enumerate(seg_ids)
        ]
    if extra_handles:
        split_label = extra_handles[0].get_label()
        row_widths.append(_row_width_split(ax, split_label, fs - 0.35))
    box_w = max(0.08, max(row_widths) - width_trim)
    x_right = _LEGEND_BOX_RIGHT if legend_x_right is None else legend_x_right
    x_left = 1.0 - box_w - x_right
    _draw_legend_box(ax, x_left, y_bot, box_w, y_top)

    for i, seg in enumerate(seg_ids):
        y = y_top - _LEGEND_PAD - (i + 0.5) * row_h
        label = _seg_label(seg)
        if seg_estimated_actions is None:
            x = x_left + _LEGEND_PAD
            _draw_legend_rect(ax, x, y, seg_to_color[seg])
            ax.text(
                x + _LEGEND_SWATCH_W + _LEGEND_SWATCH_GAP, y, label,
                transform=ax.transAxes, fontsize=fs, color="black",
                va="center", ha="left", zorder=11, clip_on=False,
            )
        else:
            _draw_learned_seg_legend_row(
                ax, x_left, y, seg, seg_to_color[seg],
                seg_estimated_actions.get(seg, 0), action_to_marker, fs,
                seg_text=label, display_k=i + 1,
            )

    if extra_handles:
        h = extra_handles[0]
        y = y_top - _LEGEND_PAD - (n_segs + 0.5) * row_h
        _draw_split_legend_row(ax, x_left, y, h, h.get_label(), fs)


def _build_gt_legend(ax, seg_ids, seg_to_color, seg_optimal_actions, action_to_marker):
    """Ground-truth legend: swatch + True Seg k + optimal action + marker."""
    n_rows = len(seg_ids)
    fs, y_top, y_bot, row_h = _legend_layout(ax, n_rows)
    row_widths = [
        _row_width_gt(ax, seg, seg_optimal_actions.get(seg, 0), fs) for seg in seg_ids
    ]
    box_w = max(row_widths)
    x_left = 1.0 - box_w - _LEGEND_BOX_RIGHT
    _draw_legend_box(ax, x_left, y_bot, box_w, y_top)

    for i, seg in enumerate(seg_ids):
        y = y_top - _LEGEND_PAD - (i + 0.5) * row_h
        _draw_gt_legend_row(
            ax, x_left, y, seg, seg_to_color[seg],
            seg_optimal_actions.get(seg, 0), action_to_marker, fs,
        )


def _style_discrete_2d_panel(ax, y_vec, *, show_xlabel: bool = True):
    style_binary_y_axis(ax, y_vals=y_vec)
    style_demo_2d_axes(ax)
    if show_xlabel:
        ax.set_xlabel(DEMO_DISCRETE_XLABEL)
    ax.set_ylabel(DEMO_DISCRETE_YLABEL)
    ax.margins(x=0.02)


def _panel_caption(ax, text: str, *, is_bottom_row: bool = False) -> None:
    """Place x-axis label and panel caption below the plot area."""
    ax.set_xlabel(DEMO_DISCRETE_XLABEL, labelpad=4)
    caption_y = -0.36 if is_bottom_row else -0.24
    ax.text(
        0.5, caption_y, text,
        transform=ax.transAxes, ha="center", va="top",
        fontsize=plt.rcParams.get("axes.titlesize", 11),
        clip_on=False,
    )


def _draw_discrete_pilot_panel(
    ax, df, *, title="Pilot Data",
    x_col="x_0", y_col="outcome", D_col="D_i",
):
    """All points black; legend shows only treatment markers (D=0, D=1, …)."""
    x0 = df[x_col].to_numpy()
    y_vec = df[y_col].to_numpy()
    D_vec = df[D_col].to_numpy().astype(int)
    unique_actions = sorted(np.unique(D_vec))
    action_to_marker = _action_marker_map(unique_actions)

    for act in unique_actions:
        idx = D_vec == act
        mk = action_to_marker[act]
        ax.scatter(
            x0[idx], y_vec[idx],
            color="black", marker=mk,
            s=DEMO_SCATTER_SIZE, alpha=DEMO_SCATTER_ALPHA,
            linewidths=0.8 if mk in ("x", "+") else 0.0,
            edgecolors="black",
            zorder=3,
        )

    _style_discrete_2d_panel(ax, y_vec, show_xlabel=False)

    # Treatment-only legend (simple, tight, top-right)
    fs = demo_legend_style()["fontsize"]
    fs_leg = fs - 0.35
    y_floor = discrete_legend_axes_y_floor(ax)
    pad = _LEGEND_PAD
    row_h = min(
        _LEGEND_ROW_H_MAX,
        max(_LEGEND_ROW_H_MIN, (0.97 - y_floor - 2 * pad) / max(len(unique_actions), 1)),
    )
    label_for = lambda a: f"treatment $D = {a}$"
    dtext_w = max(_axes_text_width(ax, label_for(a), fs_leg) for a in unique_actions)
    gap_txt_mk = 0.012
    marker_w = _LEGEND_MARKER_W
    pad_sides = pad
    box_w = pad_sides + dtext_w + gap_txt_mk + marker_w + pad_sides
    box_w = min(box_w, 0.72)
    n_rows = len(unique_actions)
    box_h = n_rows * row_h + 2 * pad
    y_top = 1.02
    y_bot = max(y_top - box_h, y_floor)
    x_left = 1.0 - box_w - 0.01
    # Markers sit optically low vs text; nudge up for baseline alignment.
    marker_y_nudge = 0.008

    ax.add_patch(FancyBboxPatch(
        (x_left, y_bot), box_w, y_top - y_bot,
        transform=ax.transAxes, boxstyle="square,pad=0",
        facecolor="white", edgecolor="black", linewidth=0.75,
        zorder=10, clip_on=False,
    ))
    x_cur = x_left + pad_sides
    for i, act in enumerate(unique_actions):
        y = y_top - pad - (i + 0.5) * row_h
        x_txt = x_cur
        x_mk  = x_txt + dtext_w + gap_txt_mk
        ax.text(
            x_txt, y, label_for(act),
            transform=ax.transAxes, fontsize=fs_leg, color="0.20",
            va="center", ha="left", zorder=11, clip_on=False,
        )
        _legend_marker_axes(
            ax, x_mk, y + marker_y_nudge, act, "black", action_to_marker,
        )


def _draw_discrete_ground_truth_panel(
    ax, df, *, title="", segment_colors=None, segment_cmap_name: str = "Set2",
    segment_col="true_segment_id", x_col="x_0", y_col="outcome", D_col="D_i",
    seg_optimal_actions=None,
):
    seg_ids = sorted(df[segment_col].unique())
    unique_actions = sorted(df[D_col].unique())
    seg_to_color = _label_color_map(seg_ids, segment_colors, segment_cmap_name)
    action_to_marker = _action_marker_map(unique_actions)

    labels = df[segment_col].to_numpy()
    D_vec = df[D_col].to_numpy()
    y_vec = df[y_col].to_numpy()
    x0 = df[x_col].to_numpy()

    _draw_discrete_scatter(ax, x0, y_vec, labels, D_vec, seg_to_color, action_to_marker)
    _style_discrete_2d_panel(ax, y_vec, show_xlabel=False)
    if seg_optimal_actions is not None:
        _build_gt_legend(ax, seg_ids, seg_to_color, seg_optimal_actions, action_to_marker)
    else:
        _build_segment_legend(ax, seg_ids, seg_to_color, unique_actions, action_to_marker)


def _draw_kmeans_splits(ax, x0, y_vec, model):
    """Draw vertical lines at midpoints between sorted K-means centroids."""
    try:
        centers = np.sort(model.cluster_centers_[:, 0])
    except (AttributeError, IndexError):
        return None
    if len(centers) < 2:
        return None
    y_lo, _ = discrete_y_display_limits(float(y_vec.min()), float(y_vec.max()))
    y_hi = 1.5
    midpoints = (centers[:-1] + centers[1:]) / 2
    split_handle = None
    for mx in midpoints:
        (line,) = ax.plot(
            [mx, mx], [y_lo, y_hi], color=_SPLIT_LINE_COLOR, linewidth=_SPLIT_LINE_LW,
            linestyle="--", alpha=_SPLIT_LINE_ALPHA, zorder=2,
        )
        split_handle = line
    if split_handle is not None:
        split_handle.set_label("midpoint of centroids")
    return split_handle


def _draw_discrete_segmentation_panel(
    ax, labels, X, y_vec, D_vec, *, title="", tree=None, algo=None, model=None,
    segment_colors=None, segment_cmap_name: str = "tab10",
    seg_estimated_actions=None,
):
    labels = np.ravel(labels).astype(int)
    D_vec = np.ravel(D_vec).astype(int)
    y_vec = np.ravel(y_vec)
    x0 = X[:, 0] if X.ndim == 2 else np.ravel(X)

    unique_actions = sorted(np.unique(D_vec))
    seg_legend_labels = None
    if algo in ("kmeans-standard", "kmeans") and model is not None:
        seg_ids = _kmeans_legend_seg_order(model, labels)
        seg_legend_labels = {
            cid: f"Learned Seg {rank}"
            for rank, cid in enumerate(seg_ids, start=1)
        }
    else:
        seg_ids = sorted(np.unique(labels))
    seg_to_color = _label_color_map(seg_ids, segment_colors, segment_cmap_name)
    action_to_marker = _action_marker_map(unique_actions)

    _draw_discrete_scatter(ax, x0, y_vec, labels, D_vec, seg_to_color, action_to_marker)
    extra = []
    if algo == "dast" and tree is not None:
        split_h = _draw_discrete_splits(ax, tree, x0, y_vec)
        if split_h is not None:
            extra = [split_h]
    elif algo in ("kmeans-standard", "kmeans") and model is not None:
        split_h = _draw_kmeans_splits(ax, x0, y_vec, model)
        if split_h is not None:
            extra = [split_h]
    _style_discrete_2d_panel(ax, y_vec, show_xlabel=False)
    legend_kw: dict = {}
    if algo == "dast":
        # Panel (d): multiple segs + split → more vertical room / row height
        legend_kw = dict(
            legend_x_right=0.003,
            legend_y_top=1.04,
            width_trim=0.016,
            y_floor_pad=0.015,
            row_h_min=0.068,
            row_h_max=0.090,
        )
    _build_segment_legend(
        ax, seg_ids, seg_to_color, unique_actions, action_to_marker,
        extra_handles=extra or None, show_treatments=False,
        seg_estimated_actions=seg_estimated_actions,
        seg_legend_labels=seg_legend_labels,
        **legend_kw,
    )


def plot_demo_combined(panels, out_path: str) -> tuple[Path, Path]:
    """Save a combined figure.

    Expected panel order:
      [0] pilot, [1] ground_truth, then one segmentation panel per algorithm
      (e.g. kmeans, mst, dast).

    Layout:
      4 panels → 2×2
      5 panels → 2×3 (last cell blank)
    """
    n = len(panels)
    if n == 4:
        nrows, ncols = 2, 2
        figsize = DEMO_FIGSIZE_STACK
        layout_fn = layout_demo_stack
    elif n == 5:
        nrows, ncols = 2, 3
        figsize = DEMO_FIGSIZE_STACK_2x3
        layout_fn = layout_demo_stack_2x3
    else:
        raise AssertionError(f"Expected 4 or 5 panels, got {n}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True, sharey=False)
    layout_fn(fig)
    fig.canvas.draw()

    labels = [f"({chr(ord('a') + i)})" for i in range(nrows * ncols)]

    def _draw(ax, panel, label_prefix: str, is_bottom_row: bool):
        kind = panel["kind"]
        title_text = panel.get("title", "")
        caption = f"{label_prefix} {title_text}".strip()
        if kind == "pilot":
            _draw_discrete_pilot_panel(ax, panel["df"])
        elif kind == "ground_truth":
            _draw_discrete_ground_truth_panel(
                ax, panel["df"],
                segment_colors=panel.get("segment_colors"),
                segment_cmap_name=panel.get("segment_cmap_name", "Set2"),
                seg_optimal_actions=panel.get("seg_optimal_actions"),
            )
        elif kind == "segmentation":
            _draw_discrete_segmentation_panel(
                ax,
                panel["labels"], panel["X"], panel["y_vec"], panel["D_vec"],
                tree=panel.get("tree"),
                algo=panel.get("algo"),
                model=panel.get("model"),
                segment_colors=panel.get("segment_colors"),
                segment_cmap_name=panel.get("segment_cmap_name", "tab10"),
                seg_estimated_actions=panel.get("seg_estimated_actions"),
            )
        else:
            raise ValueError(f"Unknown panel kind: {kind!r}")
        _panel_caption(ax, caption, is_bottom_row=is_bottom_row)

    flat_axes = np.asarray(axes).ravel()
    for i, panel in enumerate(panels):
        row, col = divmod(i, ncols)
        _draw(flat_axes[i], panel, labels[i], is_bottom_row=(row == nrows - 1))
    for j in range(n, nrows * ncols):
        flat_axes[j].set_visible(False)

    layout_fn(fig)
    paths = save_demo_figure(fig, out_path)
    plt.close(fig)
    return paths


def plot_demo_pilot(
    df,
    out_path: str,
    *,
    x_col: str = "x_0",
    y_col: str = "outcome",
    D_col: str = "D_i",
) -> tuple[Path, Path]:
    """Save a single-panel pilot-data figure for the discrete demo."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=demo_figsize_2d(discrete_outcome=True))
    layout_demo_2d(fig, discrete_outcome=True)
    fig.canvas.draw()
    _draw_discrete_pilot_panel(ax, df, x_col=x_col, y_col=y_col, D_col=D_col)
    ax.set_xlabel(DEMO_DISCRETE_XLABEL)
    paths = save_demo_figure(fig, out_path)
    plt.close(fig)
    return paths


def plot_demo_ground_truth(
    df,
    out_path: str,
    *,
    segment_colors=None,
    segment_cmap_name: str = "Set2",
    seg_optimal_actions=None,
    segment_col: str = "true_segment_id",
    x_col: str = "x_0",
    y_col: str = "outcome",
    D_col: str = "D_i",
) -> tuple[Path, Path]:
    """Save a single-panel ground-truth figure for the discrete demo."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=demo_figsize_2d(discrete_outcome=True))
    layout_demo_2d(fig, discrete_outcome=True)
    fig.canvas.draw()
    _draw_discrete_ground_truth_panel(
        ax, df,
        segment_colors=segment_colors,
        segment_cmap_name=segment_cmap_name,
        seg_optimal_actions=seg_optimal_actions,
        segment_col=segment_col, x_col=x_col, y_col=y_col, D_col=D_col,
    )
    ax.set_xlabel(DEMO_DISCRETE_XLABEL)
    paths = save_demo_figure(fig, out_path)
    plt.close(fig)
    return paths


def plot_demo_segmentation(
    labels,
    X,
    y_vec,
    D_vec,
    algo: str,
    M: int,
    out_path: str,
    *,
    tree=None,
    model=None,
    segment_colors=None,
    segment_cmap_name: str = "tab10",
    seg_estimated_actions=None,
) -> tuple[Path, Path]:
    """Save a single-panel algorithm segmentation figure for the discrete demo."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=demo_figsize_2d(discrete_outcome=True))
    layout_demo_2d(fig, discrete_outcome=True)
    fig.canvas.draw()
    _draw_discrete_segmentation_panel(
        ax, labels, X, y_vec, D_vec,
        title=algo_panel_title(algo, M),
        tree=tree, algo=algo, model=model,
        segment_colors=segment_colors, segment_cmap_name=segment_cmap_name,
        seg_estimated_actions=seg_estimated_actions,
    )
    ax.set_xlabel(DEMO_DISCRETE_XLABEL)
    paths = save_demo_figure(fig, out_path)
    plt.close(fig)
    return paths

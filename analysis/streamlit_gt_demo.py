"""2D GT + algorithm compare (Streamlit). Run: streamlit run analysis/streamlit_gt_demo.py"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "analysis") not in sys.path:
    sys.path.insert(0, str(ROOT / "analysis"))

from demo_m_utils import (  # noqa: E402
    DEMO_META_ALGOS,
    DEMO_SEG_ALGOS,
    sweep_pick_and_retrain,
)
from ground_truth import PopulationSimulator, _sigmoid  # noqa: E402
from meta_learners import DR_learner, S_learner, T_learner, X_learner  # noqa: E402
from oracle import oracle_profit_on_customers, policy_oracle, structure_oracle  # noqa: E402
from utils import assign_new_customers_to_segments  # noqa: E402

SEGMENT_COLORS = [
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#CC79A7",
    "#D55E00",
    "#56B4E9",
    "#F0E442",
    "#999999",
]

ACTION_MARKERS = ["circle", "square", "diamond", "cross", "x", "triangle-up"]

DEFAULT_CHECKED = ("dast", "kmeans-standard", "t_learner")


def _meta_fn(algo: str):
    if algo == "t_learner":
        return T_learner
    if algo == "s_learner":
        return S_learner
    if algo == "x_learner":
        return X_learner
    if algo == "dr_learner":
        return DR_learner
    if algo == "causal_forest":
        from causal_forest import causal_forest_predict
        return causal_forest_predict
    if algo == "policy_tree":
        import importlib
        import policy_tree as _policy_tree_mod

        importlib.reload(_policy_tree_mod)
        return _policy_tree_mod.policy_tree_predict
    raise ValueError(f"Unknown meta algorithm: {algo}")


def _oracle_action_at(seg, x: np.ndarray, signal_d: int, action_num: int) -> int:
    scores = [
        _sigmoid(seg._linear_predictor(x, a, signal_d)) for a in range(action_num)
    ]
    return int(np.argmax(scores))


def _p_at_xmean(seg, a: int, signal_d: int) -> float:
    return float(_sigmoid(seg._linear_predictor(seg.x_mean, a, signal_d)))


@contextlib.contextmanager
def _silence_stdout():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield


def dgp_fingerprint(params: dict[str, Any]) -> tuple:
    keys = (
        "seed",
        "K",
        "action_num",
        "N_segment_size",
        "x_space",
        "disallowed_ball_radius",
        "noise_mode",
        "X_noise_std_scale",
        "target_mahalanobis_sep",
        "beta_scale",
        "enable_delta",
        "delta_scale",
        "target_p_lo",
        "target_p_hi",
        "use_winner",
        "winner_p_lo",
        "winner_p_hi",
        "cv_folds",
        "action_method",
        "implementation_scale",
        "DR_generation_method",
    )
    return tuple(params[k] for k in keys)


def _param_range_from(params: dict[str, Any]) -> dict:
    return {
        "alpha": None,
        "beta": (
            (-params["beta_scale"], params["beta_scale"])
            if params["beta_scale"] > 0
            else (0.0, 0.0)
        ),
        "tau": None,
        "delta": (
            (-params["delta_scale"], params["delta_scale"])
            if params["enable_delta"] and params["delta_scale"] > 0
            else None
        ),
        "x_mean": (-params["x_space"], params["x_space"]),
        "target_p": (params["target_p_lo"], params["target_p_hi"]),
        "winner_p": (
            (params["winner_p_lo"], params["winner_p_hi"])
            if params["use_winner"]
            else None
        ),
    }


def build_population(params: dict[str, Any]) -> PopulationSimulator:
    np.random.seed(params["seed"])
    N_pilot = params["N_segment_size"] * params["K"]
    scale = float(params.get("implementation_scale", 2.0))
    N_impl = max(10, int(N_pilot * scale))
    dr_method = str(params.get("DR_generation_method", "reg"))
    noise_mode = str(params.get("noise_mode", "mahalanobis"))
    kw: dict[str, Any] = dict(
        N_total_pilot_customers=N_pilot,
        N_total_implement_customers=N_impl,
        d=2,
        K=params["K"],
        disturb_covariate_noise=1.0,  # unused when partial_x=1 (disturb_d=0)
        param_range=_param_range_from(params),
        DR_generation_method=dr_method,
        partial_x=1.0,
        action_num=params["action_num"],
        disallowed_ball_radius=params["disallowed_ball_radius"],
    )
    if noise_mode == "X_noise_std_scale":
        kw["X_noise_std_scale"] = float(params.get("X_noise_std_scale", 0.2))
    else:
        kw["target_mahalanobis_sep"] = float(params.get("target_mahalanobis_sep", 2.5))
    with _silence_stdout():
        return PopulationSimulator(**kw)


@st.cache_data(show_spinner="Generating ground-truth population…")
def generate_gt_bundle(_cache_ver: int = 2, **params) -> dict[str, Any]:
    """Serializable GT arrays for plotting (cached)."""
    pop = build_population(params)
    action_num = params["action_num"]
    X = np.asarray([c.x for c in pop.pilot_customers], dtype=float)
    seg_ids = np.asarray(
        [c.true_segment.segment_id for c in pop.pilot_customers], dtype=int
    )
    oracle_a = np.asarray(
        [
            _oracle_action_at(c.true_segment, c.x, pop.signal_d, action_num)
            for c in pop.pilot_customers
        ],
        dtype=int,
    )
    centers = np.asarray([seg.x_mean[:2] for seg in pop.true_segments], dtype=float)

    rows = []
    for seg in pop.true_segments:
        row = {"segment": int(seg.segment_id), "best@x_mean": int(seg.action)}
        for a in range(action_num):
            row[f"p(D={a})"] = round(_p_at_xmean(seg, a, pop.signal_d), 4)
        rows.append(row)

    seg_packs = []
    for seg in pop.true_segments:
        seg_packs.append(
            {
                "segment_id": int(seg.segment_id),
                "x_mean": np.asarray(seg.x_mean, dtype=float),
                "alpha": float(seg.alpha),
                "beta": np.asarray(seg.beta, dtype=float),
                "tau": np.asarray(seg.tau, dtype=float),
                "delta": None if seg.delta is None else np.asarray(seg.delta, dtype=float),
            }
        )

    return {
        "X": X,
        "seg_ids": seg_ids,
        "oracle_a": oracle_a,
        "centers": centers,
        "seg_packs": seg_packs,
        "prob_rows": rows,
        "metrics": {
            "signal_covariate_noise": float(pop.signal_covariate_noise),
            "noise_mode": str(params.get("noise_mode", "mahalanobis")),
            "X_noise_std_scale": (
                None
                if params.get("noise_mode", "mahalanobis") != "X_noise_std_scale"
                else float(params.get("X_noise_std_scale", 0.2))
            ),
            "target_mahalanobis_sep": (
                None
                if params.get("noise_mode", "mahalanobis") == "X_noise_std_scale"
                else float(params.get("target_mahalanobis_sep", 2.5))
            ),
            "median_nn_mahalanobis": float(pop.realized_median_nn_mahalanobis_sep),
            "min_nn_mahalanobis": float(pop.realized_min_nn_mahalanobis_sep),
            "median_nn_bayes_error": float(pop.realized_median_nn_bayes_error),
            "N_pilot": int(len(pop.pilot_customers)),
        },
        "action_num": int(action_num),
        "signal_d": int(pop.signal_d),
    }


def _linear_predictor_from_pack(pack: dict, x: np.ndarray, a: int, signal_d: int) -> float:
    eta = (
        pack["alpha"]
        + float(pack["beta"][:signal_d] @ x[:signal_d])
        + float(pack["tau"][a])
    )
    if pack["delta"] is not None:
        eta += float(pack["delta"][a, :signal_d] @ x[:signal_d])
    return eta


def oracle_action_grid(
    centers: np.ndarray,
    seg_packs: list[dict],
    action_num: int,
    signal_d: int,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    resolution: int = 160,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.linspace(xlim[0], xlim[1], resolution)
    ys = np.linspace(ylim[0], ylim[1], resolution)
    xx, yy = np.meshgrid(xs, ys)
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    d2 = ((pts[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    nearest = np.argmin(d2, axis=1)
    actions = np.empty(len(pts), dtype=int)
    for i, (pt, sid) in enumerate(zip(pts, nearest)):
        pack = seg_packs[int(sid)]
        scores = [
            _sigmoid(_linear_predictor_from_pack(pack, pt, a, signal_d))
            for a in range(action_num)
        ]
        actions[i] = int(np.argmax(scores))
    return xs, ys, actions.reshape(resolution, resolution)


def _axis_limits(X: np.ndarray, pad: float = 0.08) -> tuple[tuple[float, float], tuple[float, float]]:
    x_min, x_max = float(X[:, 0].min()), float(X[:, 0].max())
    y_min, y_max = float(X[:, 1].min()), float(X[:, 1].max())
    dx = max(x_max - x_min, 1e-6)
    dy = max(y_max - y_min, 1e-6)
    return (
        (x_min - pad * dx, x_max + pad * dx),
        (y_min - pad * dy, y_max + pad * dy),
    )


def _add_action_background(
    fig: go.Figure,
    bundle: dict[str, Any],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    xs, ys, grid = oracle_action_grid(
        bundle["centers"],
        bundle["seg_packs"],
        bundle["action_num"],
        bundle["signal_d"],
        xlim,
        ylim,
    )
    action_num = bundle["action_num"]
    colorscale = []
    rgbs = [
        (100, 149, 237),
        (240, 128, 128),
        (144, 238, 144),
        (221, 160, 221),
    ]
    for a in range(action_num):
        t0 = a / action_num
        t1 = (a + 1) / action_num
        rgb = rgbs[a % 4]
        hexcol = f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"
        colorscale.append([t0, hexcol])
        colorscale.append([t1 - 1e-9, hexcol])
    fig.add_trace(
        go.Heatmap(
            x=xs,
            y=ys,
            z=grid,
            colorscale=colorscale,
            zmin=-0.5,
            zmax=action_num - 0.5,
            opacity=0.35,
            showscale=False,
            hoverinfo="skip",
            name="oracle action bg",
        )
    )


def _scatter_by_segment_action(
    fig: go.Figure,
    X: np.ndarray,
    seg_ids: np.ndarray,
    actions: np.ndarray,
    action_num: int,
    *,
    action_label: str,
) -> None:
    if len(seg_ids) == 0:
        return
    n_seg = int(seg_ids.max()) + 1
    for sid in range(n_seg):
        mask_s = seg_ids == sid
        color = SEGMENT_COLORS[sid % len(SEGMENT_COLORS)]
        for a in range(action_num):
            mask = mask_s & (actions == a)
            if not np.any(mask):
                continue
            fig.add_trace(
                go.Scatter(
                    x=X[mask, 0],
                    y=X[mask, 1],
                    mode="markers",
                    name=f"Seg {sid} · a={a}",
                    legendgroup=f"seg{sid}",
                    marker=dict(
                        size=9,
                        color=color,
                        symbol=ACTION_MARKERS[a % len(ACTION_MARKERS)],
                        line=dict(width=0.6, color="rgba(30,30,30,0.55)"),
                        opacity=0.88,
                    ),
                    hovertemplate=(
                        f"segment={sid}<br>{action_label}={a}<br>"
                        "x=%{x:.2f}, y=%{y:.2f}<extra></extra>"
                    ),
                )
            )


def _layout_figure(
    fig: go.Figure,
    *,
    title: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    height: int = 520,
) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0.02, xanchor="left", font=dict(size=15)),
        xaxis_title="x₀",
        yaxis_title="x₁",
        xaxis=dict(range=list(xlim), zeroline=False, showgrid=True, gridcolor="#eee"),
        yaxis=dict(
            range=list(ylim),
            zeroline=False,
            showgrid=True,
            gridcolor="#eee",
            scaleanchor="x",
            scaleratio=1,
        ),
        plot_bgcolor="#fafafa",
        paper_bgcolor="white",
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#ddd",
            borderwidth=1,
            font=dict(size=11),
        ),
        margin=dict(l=50, r=160, t=55, b=45),
        height=height,
        hovermode="closest",
    )
    return fig


def build_gt_figure(bundle: dict[str, Any], *, show_background: bool) -> go.Figure:
    X = bundle["X"]
    xlim, ylim = _axis_limits(X)
    fig = go.Figure()
    if show_background:
        _add_action_background(fig, bundle, xlim, ylim)
    _scatter_by_segment_action(
        fig,
        X,
        bundle["seg_ids"],
        bundle["oracle_a"],
        bundle["action_num"],
        action_label="oracle",
    )
    return _layout_figure(fig, title="Ground truth", xlim=xlim, ylim=ylim)


def build_algo_figure(
    bundle: dict[str, Any],
    algo_result: dict[str, Any],
    *,
    algo: str,
    height: int = 380,
) -> go.Figure:
    X = np.asarray(algo_result.get("X", bundle["X"]), dtype=float)
    xlim, ylim = _axis_limits(X)
    # Align GT axes when plotting pilot
    if X.shape == bundle["X"].shape:
        xlim, ylim = _axis_limits(bundle["X"])
    fig = go.Figure()
    labels = np.asarray(algo_result["labels"], dtype=int)
    actions = np.asarray(algo_result["actions"], dtype=int)
    action_num = bundle["action_num"]
    actions_plot = actions.copy()
    actions_plot[actions_plot == 404] = action_num
    plot_A = action_num + (1 if np.any(actions == 404) else 0)
    _scatter_by_segment_action(
        fig,
        X,
        labels,
        np.clip(actions_plot, 0, plot_A - 1),
        plot_A,
        action_label="est_action",
    )
    impl = float(algo_result["implementation_profits"])
    if algo_result.get("is_meta"):
        title = f"{algo} (impl={impl:.1f})"
    else:
        picked = algo_result.get("picked_M", "?")
        title = f"{algo} M={picked} (impl={impl:.1f})"
    return _layout_figure(fig, title=title, xlim=xlim, ylim=ylim, height=height)


def fit_seg_algorithm(params: dict[str, Any], algo: str) -> dict[str, Any]:
    """Segmentation method: M-selection + retrain + impl profits."""
    pop = build_population(params)

    def apply_split(frac: float) -> None:
        if isinstance(frac, str):
            return
        with _silence_stdout():
            pop.split_pilot_customers_into_train_and_validate(train_frac=float(frac))

    def get_arrays():
        x = np.array([c.x for c in pop.train_customers])
        D = np.array([c.D_i for c in pop.train_customers])
        y = np.array([c.y for c in pop.train_customers])
        return x, D, y

    with _silence_stdout():
        result = sweep_pick_and_retrain(
            pop,
            algo,
            params["K"],
            apply_split,
            get_arrays,
            include_interactions=False,
            seed=params["seed"],
            min_leaf_size=2,
            use_hybrid_method=True,
            train_frac=0.8,
            train_frac_retrain=1.0,
            action_method=params["action_method"],
            cv_folds=params["cv_folds"],
            kmeans_coef=0.15,
        )

    labels = np.asarray(result["labels"], dtype=int)
    actions = np.asarray(result["actions"], dtype=int)
    true_ids = np.array(
        [c.true_segment.segment_id for c in pop.train_customers], dtype=int
    )
    S = structure_oracle(true_ids, labels)
    # Match main.py post-retrain: policy on all pilot (train_frac=1 ⇒ train==pilot)
    P = policy_oracle(pop.pilot_customers, algo=algo, signal_d=pop.signal_d)

    if result.get("tree") is not None and result.get("segment_dict") is not None:
        result["tree"].predict_segment(pop.implement_customers, result["segment_dict"])
    elif result.get("model") is not None:
        assign_new_customers_to_segments(
            pop, pop.implement_customers, result["model"], algo
        )
    else:
        raise RuntimeError(f"{algo}: no model/tree to assign implement customers.")

    impl_profit = float(sum(c.evaluate_profits(algo) for c in pop.implement_customers))
    oracle_impl = float(
        oracle_profit_on_customers(pop.implement_customers, signal_d=pop.signal_d)
    )

    return {
        "algo": algo,
        "is_meta": False,
        "labels": labels,
        "actions": actions,
        "picked_M": int(result["picked_M"]),
        "ARI": float(S["ARI"]),
        "regret": float(P["regret"]),
        "implementation_profits": impl_profit,
        "oracle_profits_impl": oracle_impl,
        "N_impl": int(len(pop.implement_customers)),
    }


def fit_meta_algorithm(params: dict[str, Any], algo: str) -> dict[str, Any]:
    """Meta-learner: one fit on full pilot (same as main.py), predict pilot+implement."""
    fn = _meta_fn(algo)
    pop = build_population(params)
    with _silence_stdout():
        pop.split_pilot_customers_into_train_and_validate(train_frac=1.0)

    x_mat = np.array([c.x for c in pop.train_customers])
    D_vec = np.array([c.D_i for c in pop.train_customers])
    y_vec = np.array([c.y for c in pop.train_customers])

    # One fit only (main.py). Pass pilot+implement so we can viz pilot without a
    # second fit that would consume RNG and change implement profits.
    n_pilot = len(pop.pilot_customers)
    customers_all = list(pop.pilot_customers) + list(pop.implement_customers)
    with _silence_stdout():
        labels_all, act_id = fn(customers_all, x_mat, D_vec, y_vec)

    act_id = np.asarray(act_id, dtype=int)
    labels_all = np.asarray(labels_all, dtype=int)
    labels_pilot = labels_all[:n_pilot]
    labels_impl = labels_all[n_pilot:]
    actions_pilot = act_id[labels_pilot]

    impl_profit = 0.0
    for i, cust in enumerate(pop.implement_customers):
        impl_profit += float(
            cust.evaluate_profits(algo, int(act_id[labels_impl[i]]))
        )
    oracle_impl = float(
        oracle_profit_on_customers(pop.implement_customers, signal_d=pop.signal_d)
    )

    return {
        "algo": algo,
        "is_meta": True,
        "labels": labels_pilot,
        "actions": actions_pilot,
        "picked_M": None,
        "ARI": None,
        "regret": None,
        "implementation_profits": float(impl_profit),
        "oracle_profits_impl": oracle_impl,
        "N_impl": int(len(pop.implement_customers)),
    }


def fit_one_algorithm(params: dict[str, Any], algo: str) -> dict[str, Any]:
    if algo in DEMO_META_ALGOS:
        return fit_meta_algorithm(params, algo)
    if algo in DEMO_SEG_ALGOS:
        return fit_seg_algorithm(params, algo)
    raise ValueError(f"Unsupported algorithm: {algo}")


def render_dgp_sidebar() -> dict[str, Any]:
    """Left sidebar: data-generating process only."""
    st.sidebar.title("DGP")

    st.sidebar.subheader("Reproducibility")
    seed = st.sidebar.number_input(
        "seed", min_value=0, max_value=10**9, value=42, step=1,
        key="seed", bind="query-params",
    )

    st.sidebar.subheader("Structure")
    K = st.sidebar.slider("K", 2, 6, 3, key="K", bind="query-params")
    action_num = st.sidebar.slider(
        "action_num", 2, 4, 3, key="action_num", bind="query-params",
    )
    N_segment_size = st.sidebar.slider(
        "N_per_segment",
        min_value=20,
        max_value=500,
        value=80,
        step=10,
        key="N_per_segment",
        bind="query-params",
    )
    implementation_scale = st.sidebar.slider(
        "implementation_scale", 0.5, 10.0, 2.0, step=0.5,
        key="implementation_scale", bind="query-params",
    )

    st.sidebar.subheader("Geometry")
    x_space = st.sidebar.slider(
        "x_space", 5.0, 200.0, 10.0, step=1.0,
        key="x_space", bind="query-params",
    )
    disallowed_ball_radius = st.sidebar.slider(
        "disallowed_ball_radius", 0.05, 0.9, 0.3, step=0.05,
        key="disallowed_ball_radius", bind="query-params",
    )

    st.sidebar.subheader("Overlap")
    noise_mode = st.sidebar.radio(
        "within-cluster noise",
        options=["mahalanobis", "X_noise_std_scale"],
        index=0,
        key="noise_mode",
        bind="query-params",
        help=(
            "mahalanobis: set target nearest-neighbor separation in σ units "
            "(σ = median_nn_distance / target_mahalanobis_sep). "
            "X_noise_std_scale: σ = scale × mean pairwise center distance."
        ),
    )
    target_mahalanobis_sep = 2.5
    X_noise_std_scale = 0.2
    if noise_mode == "mahalanobis":
        target_mahalanobis_sep = st.sidebar.slider(
            "target_mahalanobis_sep", 0.8, 6.0, 2.5, step=0.1,
            key="target_mahalanobis_sep", bind="query-params",
            help="Larger → tighter clusters / less overlap.",
        )
    else:
        X_noise_std_scale = st.sidebar.number_input(
            "X_noise_std_scale", min_value=0.01, value=0.2, step=0.05,
            key="X_noise_std_scale", bind="query-params",
        )

    st.sidebar.subheader("Outcome")
    beta_scale = st.sidebar.slider(
        "beta_scale", 0.0, 0.3, 0.05, step=0.01,
        key="beta_scale", bind="query-params",
    )
    enable_delta = st.sidebar.checkbox(
        "enable_delta", value=False, key="enable_delta", bind="query-params",
    )
    delta_scale = 0.0
    if enable_delta:
        delta_scale = st.sidebar.slider(
            "delta_scale", 0.0, 0.2, 0.05, step=0.01,
            key="delta_scale", bind="query-params",
        )

    t_lo, t_hi = st.sidebar.slider(
        "target_p", 0.02, 0.45, (0.10, 0.25), step=0.01,
        key="target_p", bind="query-params",
    )
    use_winner = st.sidebar.checkbox(
        "use_winner_p", value=True, key="use_winner_p", bind="query-params",
    )
    w_lo, w_hi = 0.40, 0.60
    if use_winner:
        w_lo, w_hi = st.sidebar.slider(
            "winner_p", 0.15, 0.85, (0.40, 0.60), step=0.01,
            key="winner_p", bind="query-params",
        )

    st.sidebar.subheader("Display")
    show_background = st.sidebar.checkbox(
        "voronoi_bg", value=True, key="voronoi_bg", bind="query-params",
    )

    return dict(
        seed=int(seed),
        K=int(K),
        action_num=int(action_num),
        N_segment_size=int(N_segment_size),
        implementation_scale=float(implementation_scale),
        x_space=float(x_space),
        disallowed_ball_radius=float(disallowed_ball_radius),
        noise_mode=str(noise_mode),
        X_noise_std_scale=float(X_noise_std_scale),
        target_mahalanobis_sep=float(target_mahalanobis_sep),
        beta_scale=float(beta_scale),
        enable_delta=bool(enable_delta),
        delta_scale=float(delta_scale),
        target_p_lo=float(t_lo),
        target_p_hi=float(t_hi),
        use_winner=bool(use_winner),
        winner_p_lo=float(w_lo),
        winner_p_hi=float(w_hi),
        show_background=bool(show_background),
    )


def render_algo_panel() -> dict[str, Any]:
    """Right-column controls: algorithms + run settings (vertical)."""
    st.markdown("### Algorithms")
    run_clicked = st.button("Run", type="primary", width="stretch")

    # Seed defaults once when URL has no prior selection.
    if "ms_seg" not in st.session_state and "ms_seg" not in st.query_params:
        st.session_state.ms_seg = [a for a in DEFAULT_CHECKED if a in DEMO_SEG_ALGOS]
    if "ms_meta" not in st.session_state and "ms_meta" not in st.query_params:
        st.session_state.ms_meta = [a for a in DEFAULT_CHECKED if a in DEMO_META_ALGOS]

    seg_checked = st.multiselect(
        "Segmentation",
        options=list(DEMO_SEG_ALGOS),
        key="ms_seg",
        bind="query-params",
    )
    meta_checked = st.multiselect(
        "Meta / individual",
        options=list(DEMO_META_ALGOS),
        key="ms_meta",
        bind="query-params",
    )
    checked = list(seg_checked) + list(meta_checked)

    st.markdown("**Fit options**")
    cv_folds = st.slider(
        "cv_folds", 1, 5, 5, key="cv_folds", bind="query-params",
    )
    action_method = st.selectbox(
        "action_method",
        options=["diff_in_means", "gamma", "logistic"],
        key="action_method",
        bind="query-params",
    )
    DR_generation_method = st.selectbox(
        "DR_generation_method",
        options=["reg", "mlp", "lightgbm", "random_forest", "xgboost"],
        key="DR_generation_method",
        bind="query-params",
    )

    return dict(
        checked_algos=checked,
        run_clicked=bool(run_clicked),
        cv_folds=int(cv_folds),
        action_method=str(action_method),
        DR_generation_method=str(DR_generation_method),
    )


def main() -> None:
    st.set_page_config(page_title="GT + Algo", layout="wide", initial_sidebar_state="expanded")
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.0rem; padding-bottom: 2rem; }
        div[data-testid="stMetricValue"] { font-size: 1.1rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if "algo_results" not in st.session_state:
        st.session_state.algo_results = {}
    if "algo_fp" not in st.session_state:
        st.session_state.algo_fp = None

    dgp = render_dgp_sidebar()

    # Left: GT + results · Right: algo controls
    left, right = st.columns([3.4, 1.15], gap="large")

    with right:
        algo_ui = render_algo_panel()

    params = {**dgp, **algo_ui}

    if params["target_p_lo"] >= params["target_p_hi"]:
        st.error("target_p range needs lo < hi.")
        st.stop()
    if params["use_winner"] and params["winner_p_lo"] >= params["winner_p_hi"]:
        st.error("winner_p range needs lo < hi.")
        st.stop()

    fp = dgp_fingerprint(params)
    gt_keys = {
        k: params[k]
        for k in (
            "seed",
            "K",
            "action_num",
            "N_segment_size",
            "x_space",
            "disallowed_ball_radius",
            "noise_mode",
            "X_noise_std_scale",
            "target_mahalanobis_sep",
            "beta_scale",
            "enable_delta",
            "delta_scale",
            "target_p_lo",
            "target_p_hi",
            "use_winner",
            "winner_p_lo",
            "winner_p_hi",
        )
    }

    with left:
        try:
            bundle = generate_gt_bundle(_cache_ver=2, **gt_keys)
        except Exception as exc:
            st.error(f"`{exc}`")
            st.stop()

        st.subheader("Ground truth")
        m = bundle["metrics"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Pilot N", f"{m['N_pilot']}")
        if m.get("noise_mode") == "X_noise_std_scale":
            c2.metric("X_noise_std_scale", f"{m['X_noise_std_scale']:.3f}")
        else:
            c2.metric("target_mahal", f"{m['target_mahalanobis_sep']:.2f}")
        c3.metric("σ", f"{m['signal_covariate_noise']:.3f}")
        c4.metric("NN Mahal", f"{m['median_nn_mahalanobis']:.2f}")

        st.plotly_chart(
            build_gt_figure(bundle, show_background=params["show_background"]),
            use_container_width=True,
        )

        st.dataframe(
            pd.DataFrame(bundle["prob_rows"]).set_index("segment"),
            use_container_width=True,
            hide_index=False,
        )

        if params["run_clicked"]:
            if not params["checked_algos"]:
                st.warning("No algorithm selected.")
            else:
                st.session_state.algo_results = {}
                st.session_state.algo_fp = fp
                progress = st.progress(0.0)
                n = len(params["checked_algos"])
                errors = []
                for i, algo in enumerate(params["checked_algos"]):
                    progress.progress((i + 1) / n)
                    try:
                        st.session_state.algo_results[algo] = fit_one_algorithm(params, algo)
                    except Exception as exc:
                        errors.append(f"{algo}: {exc}")
                progress.empty()
                if errors:
                    st.error("\n".join(errors))

        stale = (
            st.session_state.algo_fp is not None
            and st.session_state.algo_fp != fp
            and bool(st.session_state.algo_results)
        )
        if stale:
            st.warning("stale — re-Run")

        results = st.session_state.algo_results
        ordered = [a for a in params["checked_algos"] if a in results]
        if ordered:
            any_res = results[ordered[0]]
            st.subheader("Results")
            st.caption(
                f"N_impl={any_res['N_impl']}  ·  oracle_impl={any_res['oracle_profits_impl']:.2f}"
            )
            for i in range(0, len(ordered), 2):
                cols = st.columns(2)
                for j, col in enumerate(cols):
                    idx = i + j
                    if idx >= len(ordered):
                        break
                    algo = ordered[idx]
                    res = results[algo]
                    with col:
                        st.plotly_chart(
                            build_algo_figure(bundle, res, algo=algo),
                            use_container_width=True,
                        )


if __name__ == "__main__":
    main()

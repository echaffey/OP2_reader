# op2_native/plots.py
"""
Quick-look Plotly visualisation helpers for OP2 result DataFrames.

All functions return a ``plotly.graph_objects.Figure`` that can be shown
with ``fig.show()`` or embedded in a notebook with ``fig``.

Functions
---------
plot_vm_stress(df, subcase=1, fiber="max", title=None)
    Bar/scatter chart of Von Mises stress per element.

plot_displacement_magnitude(df, subcase=1, component="mag", title=None)
    Bar/scatter chart of displacement magnitude (or a single DOF) per grid.

plot_element_forces(df, component="NX", title=None)
    Bar chart of a chosen element force component per element.

plot_stress_histogram(df, column="VON_MISES1", bins=40, title=None)
    Histogram of a stress/strain column across all elements.
"""
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _require_plotly():
    try:
        import plotly.graph_objects as go  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "plotly is required for plotting. Install with: pip install plotly"
        ) from e


def _go():
    import plotly.graph_objects as go

    return go


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def plot_vm_stress(
    df: pd.DataFrame,
    fiber: str = "max",
    title: Optional[str] = None,
    color_scale: str = "Plasma",
    height: int = 500,
):
    """
    Plot Von Mises stress per element as a scatter/bar chart sorted by EID.

    Parameters
    ----------
    df : DataFrame
        Output of ``op2.stresses()[subcase]``.
        Must contain ``EID`` and at least one of ``VON_MISES1``, ``VON_MISES2``.
    fiber : {"max", "1", "2"}
        Which fiber layer to plot:
        * ``"max"`` — element-wise maximum of VON_MISES1 and VON_MISES2
        * ``"1"``   — bottom fiber (VON_MISES1)
        * ``"2"``   — top fiber (VON_MISES2)
    title : str, optional
    color_scale : str
        Plotly color scale name (default ``"Plasma"``).
    height : int
        Figure height in pixels.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    _require_plotly()
    go = _go()

    if "VON_MISES1" not in df.columns and "VON_MISES2" not in df.columns:
        raise ValueError(
            "DataFrame has no VON_MISES1/VON_MISES2 columns — is this a stress table?"
        )

    df = df.copy()
    if fiber == "max":
        cols = [c for c in ("VON_MISES1", "VON_MISES2") if c in df.columns]
        df["_vm"] = df[cols].max(axis=1)
        y_label = "VM stress (max fiber)"
    elif fiber == "1":
        df["_vm"] = df["VON_MISES1"]
        y_label = "VM stress (fiber 1, bottom)"
    else:
        df["_vm"] = df["VON_MISES2"]
        y_label = "VM stress (fiber 2, top)"

    df = df.sort_values("EID").reset_index(drop=True)
    vm_max = df["_vm"].max()

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df["EID"],
            y=df["_vm"],
            marker=dict(
                color=df["_vm"],
                colorscale=color_scale,
                colorbar=dict(title=y_label),
                cmin=0,
                cmax=vm_max,
            ),
            hovertemplate="EID=%{x}<br>VM=%{y:.4g}<extra></extra>",
            name=y_label,
        )
    )
    fig.update_layout(
        title=title or f"Von Mises Stress ({fiber} fiber) — {len(df)} elements",
        xaxis_title="Element ID",
        yaxis_title=y_label,
        height=height,
        template="plotly_white",
    )
    return fig


def plot_displacement_magnitude(
    df: pd.DataFrame,
    component: str = "mag",
    title: Optional[str] = None,
    color_scale: str = "Viridis",
    height: int = 500,
    max_points: int = 2000,
):
    """
    Plot displacement magnitude (or a single DOF) per grid node.

    Parameters
    ----------
    df : DataFrame
        Output of ``op2.displacements()[subcase]``.
        Must contain ``GRID`` and ``TX, TY, TZ``.
    component : {"mag", "TX", "TY", "TZ", "RX", "RY", "RZ"}
        ``"mag"`` computes sqrt(TX²+TY²+TZ²); any other value selects that
        column directly.
    title : str, optional
    color_scale : str
    height : int
    max_points : int
        Downsample to this many points if the DataFrame is larger (keeps the
        figure responsive).

    Returns
    -------
    plotly.graph_objects.Figure
    """
    _require_plotly()
    go = _go()

    required = {"GRID", "TX", "TY", "TZ"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing columns: {missing}")

    df = df.copy().sort_values("GRID").reset_index(drop=True)

    if component == "mag":
        df["_val"] = np.sqrt(df["TX"] ** 2 + df["TY"] ** 2 + df["TZ"] ** 2)
        y_label = "|U| displacement magnitude"
    else:
        if component not in df.columns:
            raise ValueError(f"Column {component!r} not found in DataFrame")
        df["_val"] = df[component]
        y_label = f"{component} displacement"

    # Downsample for large models
    if len(df) > max_points:
        df = (
            df.sample(max_points, random_state=0)
            .sort_values("GRID")
            .reset_index(drop=True)
        )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["GRID"],
            y=df["_val"],
            mode="markers",
            marker=dict(
                color=df["_val"],
                colorscale=color_scale,
                colorbar=dict(title=y_label),
                size=4,
            ),
            hovertemplate="GRID=%{x}<br>%{y:.4g}<extra></extra>",
            name=y_label,
        )
    )
    fig.update_layout(
        title=title or f"Displacement {component} — {len(df)} nodes",
        xaxis_title="Grid ID",
        yaxis_title=y_label,
        height=height,
        template="plotly_white",
    )
    return fig


def plot_element_forces(
    df: pd.DataFrame,
    component: str = "NX",
    title: Optional[str] = None,
    color_scale: str = "RdBu",
    height: int = 500,
):
    """
    Plot a chosen element force component per element.

    Parameters
    ----------
    df : DataFrame
        Output of ``op2.element_forces()[subcase]``.
    component : str
        Column name to plot. For shell elements: ``NX, NY, NXY, MX, MY, MXY, QX, QY``.
    title : str, optional
    color_scale : str
    height : int

    Returns
    -------
    plotly.graph_objects.Figure
    """
    _require_plotly()
    go = _go()

    if component not in df.columns:
        available = [c for c in df.columns if c != "EID"]
        raise ValueError(f"Column {component!r} not found. Available: {available}")

    df = df.sort_values("EID").reset_index(drop=True)
    vals = df[component]
    abs_max = vals.abs().max()

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df["EID"],
            y=vals,
            marker=dict(
                color=vals,
                colorscale=color_scale,
                colorbar=dict(title=component),
                cmin=-abs_max,
                cmax=abs_max,
            ),
            hovertemplate="EID=%{x}<br>" + component + "=%{y:.4g}<extra></extra>",
            name=component,
        )
    )
    fig.update_layout(
        title=title or f"Element Force: {component} — {len(df)} elements",
        xaxis_title="Element ID",
        yaxis_title=component,
        height=height,
        template="plotly_white",
    )
    return fig


def plot_stress_histogram(
    df: pd.DataFrame,
    column: str = "VON_MISES1",
    bins: int = 40,
    title: Optional[str] = None,
    height: int = 450,
):
    """
    Histogram of a stress (or strain) column across all elements.

    Parameters
    ----------
    df : DataFrame
        Output of ``op2.stresses()[subcase]`` or similar.
    column : str
        Column to histogram. Defaults to ``"VON_MISES1"``.
    bins : int
        Number of histogram bins.
    title : str, optional
    height : int

    Returns
    -------
    plotly.graph_objects.Figure
    """
    _require_plotly()
    go = _go()

    if column not in df.columns:
        raise ValueError(
            f"Column {column!r} not in DataFrame. Available: {list(df.columns)}"
        )

    vals = df[column].dropna()

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=vals,
            nbinsx=bins,
            marker_color="steelblue",
            opacity=0.85,
            hovertemplate="range=%{x}<br>count=%{y}<extra></extra>",
            name=column,
        )
    )
    # Overlay a vertical line at mean
    mean_val = float(vals.mean())
    fig.add_vline(
        x=mean_val,
        line_dash="dash",
        line_color="firebrick",
        annotation_text=f"mean={mean_val:.4g}",
        annotation_position="top right",
    )
    fig.update_layout(
        title=title or f"Histogram of {column} — {len(vals)} elements",
        xaxis_title=column,
        yaxis_title="Count",
        height=height,
        template="plotly_white",
        bargap=0.05,
    )
    return fig


def plot_top_n_stress(
    df: pd.DataFrame,
    n: int = 20,
    fiber: str = "max",
    title: Optional[str] = None,
    color_scale: str = "Plasma",
    height: int = 500,
):
    """
    Bar chart of the *n* most-stressed elements, annotated with their EIDs.

    Useful for quickly identifying the governing elements in a model without
    scrolling through hundreds of bars.

    Parameters
    ----------
    df : DataFrame
        Output of ``op2.stresses()[subcase]``.  Must contain ``EID`` and at
        least one of ``VON_MISES1``, ``VON_MISES2``.
    n : int
        Number of top elements to display.  Default 20.
    fiber : {"max", "1", "2"}
        Which fiber layer to use for ranking.
    title : str, optional
    color_scale : str
    height : int

    Returns
    -------
    plotly.graph_objects.Figure
    """
    _require_plotly()
    go = _go()

    if "VON_MISES1" not in df.columns and "VON_MISES2" not in df.columns:
        raise ValueError(
            "DataFrame has no VON_MISES1/VON_MISES2 columns — is this a stress table?"
        )

    df = df.copy()
    if fiber == "max":
        cols = [c for c in ("VON_MISES1", "VON_MISES2") if c in df.columns]
        df["_vm"] = df[cols].max(axis=1)
        y_label = "VM stress (max fiber)"
    elif fiber == "1":
        df["_vm"] = df["VON_MISES1"]
        y_label = "VM stress (fiber 1)"
    else:
        df["_vm"] = df["VON_MISES2"]
        y_label = "VM stress (fiber 2)"

    top = (
        df.nlargest(n, "_vm").sort_values("_vm", ascending=True).reset_index(drop=True)
    )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=top["_vm"],
            y=top["EID"].astype(str),
            orientation="h",
            marker=dict(
                color=top["_vm"],
                colorscale=color_scale,
                colorbar=dict(title=y_label),
                cmin=0,
                cmax=float(top["_vm"].max()),
            ),
            hovertemplate="EID=%{y}<br>VM=%{x:.4g}<extra></extra>",
            text=top["_vm"].map(lambda v: f"{v:.4g}"),
            textposition="outside",
            name=y_label,
        )
    )
    fig.update_layout(
        title=title or f"Top {n} Stressed Elements ({fiber} fiber)",
        xaxis_title=y_label,
        yaxis_title="Element ID",
        height=max(height, 30 * n + 100),
        template="plotly_white",
        yaxis=dict(tickmode="linear"),
    )
    return fig


def plot_principal_stress(
    df: pd.DataFrame,
    fiber: str = "1",
    title: Optional[str] = None,
    height: int = 550,
    max_elements: int = 300,
):
    """
    Grouped bar chart showing MAJOR, MINOR, and VM stress side-by-side for
    each element, making it easy to see the spread between principal values.

    Parameters
    ----------
    df : DataFrame
        Output of ``op2.stresses()[subcase]``.  Must contain
        ``EID, MAX_PRIN1, MIN_PRIN1, VON_MISES1`` (or ``…2`` for fiber 2).
    fiber : {"1", "2"}
        Which fiber to plot.
    title : str, optional
    height : int
    max_elements : int
        Downsample to this many elements (by VM rank) if the model is large,
        to keep the chart readable.  Default 300.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    _require_plotly()
    go = _go()

    suf = fiber  # "1" or "2"
    needed = {"EID", f"MAX_PRIN{suf}", f"MIN_PRIN{suf}", f"VON_MISES{suf}"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing columns: {missing}")

    df = df.copy().sort_values("EID").reset_index(drop=True)

    # Downsample: keep the highest-VM elements if there are too many
    vm_col = f"VON_MISES{suf}"
    if len(df) > max_elements:
        df = df.nlargest(max_elements, vm_col).sort_values("EID").reset_index(drop=True)

    eids = df["EID"].astype(str)
    major = df[f"MAX_PRIN{suf}"]
    minor = df[f"MIN_PRIN{suf}"]
    vm = df[vm_col]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=eids,
            y=major,
            name=f"σ_major (fiber {suf})",
            marker_color="royalblue",
            opacity=0.85,
            hovertemplate="EID=%{x}<br>MAJOR=%{y:.4g}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=eids,
            y=minor,
            name=f"σ_minor (fiber {suf})",
            marker_color="tomato",
            opacity=0.85,
            hovertemplate="EID=%{x}<br>MINOR=%{y:.4g}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=eids,
            y=vm,
            name=f"VM (fiber {suf})",
            marker_color="mediumseagreen",
            opacity=0.85,
            hovertemplate="EID=%{x}<br>VM=%{y:.4g}<extra></extra>",
        )
    )
    fig.update_layout(
        barmode="group",
        title=title or f"Principal Stress — fiber {suf} ({len(df)} elements)",
        xaxis_title="Element ID",
        yaxis_title="Stress",
        height=height,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_forces_polar(
    df: pd.DataFrame,
    title: Optional[str] = None,
    height: int = 500,
):
    """
    Polar scatter of in-plane element forces (NX, NY, NXY) plotted as
    magnitudes at orientations 0°, 90°, and 45° respectively.

    Each element is represented by three points in polar coordinates —
    a quick visual of the force resultant field distribution.

    Parameters
    ----------
    df : DataFrame
        Output of ``op2.element_forces()[subcase]``.
        Must contain ``EID, NX, NY, NXY``.
    title : str, optional
    height : int

    Returns
    -------
    plotly.graph_objects.Figure
    """
    _require_plotly()
    go = _go()

    needed = {"EID", "NX", "NY", "NXY"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing columns: {missing}")

    # Build long-form data: (theta_deg, |force|, label) per element per component
    import numpy as _np

    records = []
    for _, row in df.iterrows():
        eid = int(row["EID"])
        for comp, theta in [("NX", 0.0), ("NY", 90.0), ("NXY", 45.0)]:
            records.append(
                {
                    "EID": eid,
                    "component": comp,
                    "theta": theta,
                    "r": abs(float(row[comp])),
                }
            )

    long_df = pd.DataFrame(records)

    colors = {"NX": "royalblue", "NY": "tomato", "NXY": "goldenrod"}

    fig = go.Figure()
    for comp in ("NX", "NY", "NXY"):
        sub = long_df[long_df["component"] == comp]
        fig.add_trace(
            go.Scatterpolar(
                r=sub["r"],
                theta=sub["theta"],
                mode="markers",
                marker=dict(color=colors[comp], size=5, opacity=0.65),
                name=comp,
                hovertemplate=f"EID=%{{text}}<br>{comp}=%{{r:.4g}}<extra></extra>",
                text=sub["EID"].astype(str),
            )
        )
    fig.update_layout(
        title=title or f"In-plane Forces (polar) — {len(df)} elements",
        polar=dict(
            radialaxis=dict(visible=True, title="Force magnitude"),
            angularaxis=dict(
                tickmode="array",
                tickvals=[0, 45, 90, 135, 180, 225, 270, 315],
            ),
        ),
        height=height,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="right", x=1),
    )
    return fig


def plot_stress_summary(
    df: pd.DataFrame,
    fiber: str = "1",
    bins: int = 30,
    title: Optional[str] = None,
    height: int = 700,
):
    """
    4×2 subplot grid showing histograms of all eight stress components for
    one fiber layer: SX, SY, TXY, ANG, MAJOR, MINOR, VM, and FD.

    Gives an at-a-glance overview of the full stress state distribution
    across all elements — useful for spotting skewed distributions, outliers,
    or near-zero components.

    Parameters
    ----------
    df : DataFrame
        Output of ``op2.stresses()[subcase]``.
    fiber : {"1", "2"}
        Which fiber layer to plot.  Default ``"1"`` (bottom).
    bins : int
        Number of histogram bins per subplot.  Default 30.
    title : str, optional
    height : int
        Total figure height in pixels.  Default 700.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    _require_plotly()
    # Use make_subplots — part of plotly.subplots
    try:
        from plotly.subplots import make_subplots
    except ImportError:
        _require_plotly()  # raises ImportError with instructions

    go = _go()

    suf = fiber
    component_map = [
        (f"SX{suf}", f"σx (fiber {suf})"),
        (f"SY{suf}", f"σy (fiber {suf})"),
        (f"TXY{suf}", f"τxy (fiber {suf})"),
        (f"ANG{suf}", f"Angle (fiber {suf})"),
        (f"MAJOR{suf}", f"σ_major (fiber {suf})"),
        (f"MINOR{suf}", f"σ_minor (fiber {suf})"),
        (f"VM{suf}", f"VM (fiber {suf})"),
        (f"FD{suf}", f"Fiber dist {suf}"),
    ]

    # Only keep pairs where the column actually exists
    available = [(col, label) for col, label in component_map if col in df.columns]
    if not available:
        raise ValueError(
            f"No stress columns for fiber {fiber!r} found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )

    n = len(available)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    subplot_titles = [label for _, label in available]
    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.08,
        vertical_spacing=0.14,
    )

    _COLORS = [
        "royalblue",
        "tomato",
        "goldenrod",
        "mediumseagreen",
        "mediumpurple",
        "darkorange",
        "steelblue",
        "firebrick",
    ]

    for idx, (col, label) in enumerate(available):
        row = idx // ncols + 1
        col_idx = idx % ncols + 1
        vals = df[col].dropna()
        mean_val = float(vals.mean())

        fig.add_trace(
            go.Histogram(
                x=vals,
                nbinsx=bins,
                marker_color=_COLORS[idx % len(_COLORS)],
                opacity=0.80,
                showlegend=False,
                hovertemplate=f"{col}=%{{x:.4g}}<br>count=%{{y}}<extra></extra>",
                name=col,
            ),
            row=row,
            col=col_idx,
        )
        # Mean line via shape — use add_vline only available at fig level with row/col
        fig.add_vline(
            x=mean_val,
            line_dash="dash",
            line_color="black",
            line_width=1,
            row=row,
            col=col_idx,
        )

    fig.update_layout(
        title_text=title
        or f"Stress Component Summary — fiber {fiber}, {len(df)} elements",
        height=height,
        template="plotly_white",
        bargap=0.04,
    )
    return fig

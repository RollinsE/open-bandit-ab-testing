# -*- coding: utf-8 -*-
"""Plotly figures for the A/B test and the off-policy evaluation.

How the figures are built:

1. Plotly only. Every chart is a Plotly figure, so the saved HTML is
   interactive: hover any point for its exact values, click a legend entry to
   hide a series, drag to zoom.
2. File first. Figures are saved by ``src.reporting.save_plotly_bundle`` as
   HTML (always) and PNG (when Kaleido is installed). Nothing is shown inline
   unless ``--show-plots`` is set, so command-line and Colab subprocess runs
   never stall on a renderer.
3. One look. All figures share ``_base_layout``: the same font, surface,
   recessive grid and two series colors (blue for control, orange for
   treatment), so the report reads as one set.
"""

import logging
from contextlib import contextmanager

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.constants import ESTIMATOR_LABELS, LOGGER_NAME, POLICY_LABELS
from src.model_evaluation import SAME_CTR_BASELINE

LOG = logging.getLogger(LOGGER_NAME)

# Categorical series colors and neutral inks, for a light and a dark background.
# The light theme is the default; the dark one matches GitHub's dark mode, so
# README images can be shown in whichever theme the reader uses.
THEMES = {
    "light": dict(COLOR_CONTROL="#2a78d6", COLOR_TREATMENT="#eb6834", COLOR_THIRD="#1baf7a",
                  COLOR_FOURTH="#eda100", INK="#0b0b0b", INK_MUTED="#52514e", GRID="#e4e3df",
                  SURFACE="#fcfcfb", HOVER_BG="#ffffff"),
    "dark": dict(COLOR_CONTROL="#4a9ff5", COLOR_TREATMENT="#f27a4a", COLOR_THIRD="#2fc48d",
                 COLOR_FOURTH="#f2b53a", INK="#e6edf3", INK_MUTED="#9198a1", GRID="#30363d",
                 SURFACE="#0d1117", HOVER_BG="#161b22"),
}
COLOR_CONTROL = COLOR_TREATMENT = COLOR_THIRD = COLOR_FOURTH = ""
INK = INK_MUTED = GRID = SURFACE = HOVER_BG = ""
FONT = "Inter, system-ui, -apple-system, Segoe UI, Roboto, sans-serif"


def set_theme(name: str = "light") -> None:
    """Switch every figure built after this call to the ``light`` or ``dark`` palette."""
    if name not in THEMES:
        raise ValueError(f"Unknown theme {name!r}; choose from {sorted(THEMES)}")
    globals().update(THEMES[name])


@contextmanager
def theme(name: str):
    """Build figures in ``name`` theme inside a ``with`` block, then restore the light theme."""
    set_theme(name)
    try:
        yield
    finally:
        set_theme("light")


set_theme("light")



# =============================================================================
# COLAB DISPLAY
# =============================================================================

def _show_fig(fig, show_plot: bool) -> None:
    """Show a figure inline only when asked; CLI and subprocess runs stay file-first."""
    if not show_plot:
        return
    try:
        fig.show()
    except Exception as exc:
        LOG.debug(f"Inline Plotly display not available: {exc}")


def configure_colab_plotly() -> None:
    """Use Colab's own renderer when running inside a Colab notebook kernel."""
    import plotly.io as pio
    try:
        import sys
        if "google.colab" in sys.modules:
            pio.renderers.default = "colab"
            LOG.info("Plotly configured for Colab rendering")
    except Exception as exc:
        LOG.warning(f"Could not configure Plotly renderer: {exc}")


# =============================================================================
# SHARED STYLE
# =============================================================================

def _base_layout(fig: go.Figure, title: str, subtitle: str = "", height: int = 420) -> go.Figure:
    """Apply the shared font, surface, grid and title to a figure."""
    title_text = f"{title}<br><span style='font-size:12px;color:{INK_MUTED}'>{subtitle}</span>" if subtitle else title
    # The legend sits in its own row below the x-axis title, so it never collides
    # with the title, subtitle or panel titles. Its position is set in pixels.
    has_legend = fig.layout.showlegend is not False and any(
        t.name and t.showlegend is not False for t in fig.data)
    top, legend_gap = (90 if subtitle else 70), 62
    bottom = 95 if has_legend else 55
    height += bottom - 55  # keep the plot area the same size whether or not there is a legend
    plot_px = max(height - top - bottom, 50)
    fig.update_layout(
        title=dict(text=title_text, x=0.01, xanchor="left", font=dict(size=16, color=INK)),
        font=dict(family=FONT, size=12, color=INK),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        height=height, margin=dict(l=10, r=20, t=top, b=bottom),
        legend=dict(orientation="h", yanchor="top", y=-legend_gap / plot_px, xanchor="left", x=0, title_text=""),
        hoverlabel=dict(bgcolor=HOVER_BG, font=dict(family=FONT, size=12, color=INK), bordercolor=GRID),
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=False, linecolor=GRID,
                     tickfont=dict(color=INK_MUTED), title_font=dict(color=INK_MUTED, size=12))
    fig.update_yaxes(showgrid=False, linecolor=GRID, tickfont=dict(color=INK), automargin=True)
    return fig


def _compact(value: float) -> str:
    """1000 -> '1k', 2000000 -> '2M'."""
    for size, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if value >= size:
            return f"{value / size:g}{suffix}"
    return f"{value:g}"


def _pct(x: float, digits: int = 2, signed: bool = False) -> str:
    if pd.isna(x):
        return "n/a"
    return f"{100 * x:+.{digits}f}%" if signed else f"{100 * x:.{digits}f}%"


# =============================================================================
# A/B TEST FIGURES
# =============================================================================

def create_ab_ctr_plot(ab: pd.DataFrame, show_plot: bool = False) -> go.Figure:
    """CTR of each arm per campaign, with 95% Wilson intervals.

    Args:
        ab: Output of ``execute_ab_test``.
        show_plot: Show inline as well as returning the figure.
    """
    campaigns = [c.capitalize() for c in ab["campaign"]]
    fig = go.Figure()
    for arm, policy, color in (("control", "random", COLOR_CONTROL), ("treatment", "bts", COLOR_TREATMENT)):
        ctr = 100 * ab[f"{arm}_ctr"]
        fig.add_trace(go.Scatter(
            x=ctr, y=campaigns, mode="markers", orientation="h", name=POLICY_LABELS[policy],
            marker=dict(size=12, color=color, line=dict(color=SURFACE, width=2)),
            error_x=dict(type="data", symmetric=False, thickness=2, width=0, color=color,
                         array=100 * ab[f"{arm}_ctr_upper"] - ctr,
                         arrayminus=ctr - 100 * ab[f"{arm}_ctr_lower"]),
            customdata=np.stack([100 * ab[f"{arm}_ctr_lower"], 100 * ab[f"{arm}_ctr_upper"],
                                 ab[f"{arm}_clicks"], ab[f"{arm}_impressions"]], axis=-1),
            hovertemplate=("<b>%{y}</b> · " + POLICY_LABELS[policy] + "<br>CTR %{x:.3f}%"
                           "<br>95% CI %{customdata[0]:.3f}% to %{customdata[1]:.3f}%"
                           "<br>%{customdata[2]:,} clicks / %{customdata[3]:,} impressions<extra></extra>"),
        ))
    fig.update_layout(scattermode="group", scattergap=0.6)
    fig.update_xaxes(title_text="Click-through rate (%)", ticksuffix="%")
    fig.update_yaxes(autorange="reversed")
    _base_layout(fig, "A/B test: click-through rate by arm", "Dots are observed CTR; bars are 95% Wilson intervals",
                 height=160 + 110 * len(ab))
    _show_fig(fig, show_plot)
    return fig


def create_ab_lift_plot(ab: pd.DataFrame, show_plot: bool = False) -> go.Figure:
    """Relative lift of treatment over control per campaign, with its 95% interval and the zero line."""
    campaigns = [c.capitalize() for c in ab["campaign"]]
    lift = 100 * ab["rel_lift"]
    colors = [COLOR_TREATMENT if s else INK_MUTED for s in ab["significant"]]
    fig = go.Figure(go.Scatter(
        x=lift, y=campaigns, mode="markers", name="Lift",
        marker=dict(size=13, color=colors, line=dict(color=SURFACE, width=2)),
        error_x=dict(type="data", symmetric=False, thickness=2, width=0, color=INK_MUTED,
                     array=100 * ab["rel_lift_upper"] - lift, arrayminus=lift - 100 * ab["rel_lift_lower"]),
        # object dtype keeps the numbers numeric next to the decision text, so hover formats apply
        customdata=pd.DataFrame({"lo": 100 * ab["rel_lift_lower"], "hi": 100 * ab["rel_lift_upper"],
                                 "p": ab["p_value"], "power": 100 * ab["achieved_power"],
                                 "decision": ab["decision"]}).to_numpy(dtype=object),
        hovertemplate=("<b>%{y}</b><br>Lift %{x:+.1f}%<br>95% CI %{customdata[0]:+.1f}% to %{customdata[1]:+.1f}%"
                       "<br>p = %{customdata[2]:.3f} · power %{customdata[3]:.0f}%"
                       "<br><b>%{customdata[4]}</b><extra></extra>"),
        showlegend=False,
    ))
    fig.add_vline(x=0, line=dict(color=INK_MUTED, width=1, dash="dash"))
    fig.update_xaxes(title_text="Relative lift of Thompson Sampling over random (%)", ticksuffix="%")
    fig.update_yaxes(autorange="reversed")
    _base_layout(fig, "A/B test: lift with 95% interval",
                 "Orange = significant at the chosen alpha; grey = not significant. Right of zero favours Thompson Sampling.",
                 height=160 + 90 * len(ab))
    _show_fig(fig, show_plot)
    return fig


def create_power_curve_plot(power_table: pd.DataFrame, show_plot: bool = False) -> go.Figure:
    """Impressions per arm needed to detect each target lift, one line per campaign (log scale)."""
    fig = go.Figure()
    palette = [COLOR_CONTROL, COLOR_TREATMENT, COLOR_THIRD, COLOR_FOURTH]
    for i, (campaign, grp) in enumerate(power_table.groupby("campaign", sort=False)):
        fig.add_trace(go.Scatter(
            x=100 * grp["target_lift"], y=grp["n_per_arm"], mode="lines+markers", name=campaign.capitalize(),
            line=dict(color=palette[i % len(palette)], width=2), marker=dict(size=8),
            customdata=np.stack([100 * grp["baseline_ctr"], 100 * grp["target_ctr"]], axis=-1),
            hovertemplate=(f"<b>{campaign.capitalize()}</b><br>Lift %{{x:.0f}}%"
                           "<br>%{customdata[0]:.3f}% → %{customdata[1]:.3f}% CTR"
                           "<br><b>%{y:,.0f}</b> impressions per arm<extra></extra>"),
        ))
    fig.update_xaxes(title_text="Lift to detect (%)", ticksuffix="%")
    ticks = [m * 10 ** e for e in range(2, 9) for m in (1, 2, 5)]
    fig.update_yaxes(title_text="Impressions per arm", type="log", showgrid=True, gridcolor=GRID,
                     tickvals=ticks, ticktext=[_compact(v) for v in ticks],
                     title_font=dict(color=INK_MUTED, size=12))
    _base_layout(fig, "Planning the next test: sample size needed",
                 "80% power, two-sided, at each campaign's observed control CTR")
    _show_fig(fig, show_plot)
    return fig


# =============================================================================
# MODEL MARATHON FIGURE
# =============================================================================

def create_model_leaderboard_plot(leaderboard: pd.DataFrame, show_plot: bool = False) -> go.Figure:
    """Each click model's skill over the same-CTR baseline, one panel per campaign; the winner is highlighted."""
    campaigns = list(dict.fromkeys(leaderboard["campaign"]))
    # The same-CTR baseline is the zero line itself, so it is not drawn as a bar.
    data = leaderboard[leaderboard["Model"] != SAME_CTR_BASELINE]
    fig = make_subplots(rows=1, cols=len(campaigns), shared_xaxes=True, horizontal_spacing=0.18,
                        subplot_titles=[c.capitalize() for c in campaigns])
    for col, campaign in enumerate(campaigns, start=1):
        grp = data[data["campaign"] == campaign].sort_values("Log loss")
        colors = [COLOR_TREATMENT if w else (GRID if role == "baseline" else COLOR_CONTROL)
                  for w, role in zip(grp["winner"], grp["Role"])]
        skill = 100 * grp["Skill vs baseline"]
        fig.add_trace(go.Bar(
            x=skill, y=grp["Model"], orientation="h", marker=dict(color=colors, cornerradius=4),
            showlegend=False, name=campaign,
            customdata=np.stack([grp["Log loss"], grp["ROC AUC"], grp["Brier"], grp["Type"]], axis=-1).astype(object),
            hovertemplate=("<b>%{y}</b> (%{customdata[3]})<br>Skill vs baseline %{x:+.2f}%"
                           "<br>Log loss %{customdata[0]:.5f} · AUC %{customdata[1]:.3f}"
                           "<br>Brier %{customdata[2]:.6f}<extra></extra>"),
        ), row=1, col=col)
        fig.update_yaxes(autorange="reversed", row=1, col=col, tickfont=dict(size=11))
        fig.update_xaxes(title_text="Skill vs baseline (%)", ticksuffix="%", zeroline=True,
                         zerolinecolor=INK_MUTED, zerolinewidth=1, row=1, col=col)
    n_models = data.groupby("campaign").size().max()
    _base_layout(fig, "Reward model marathon: click model leaderboard",
                 "Out-of-fold log loss improvement over predicting the same CTR in every slot. "
                 "Orange = winner (simplest model within the margin of the best); grey = baseline.",
                 height=140 + 26 * int(n_models))
    _left_align_subplot_titles(fig, len(campaigns))
    _show_fig(fig, show_plot)
    return fig


# =============================================================================
# OFF-POLICY EVALUATION FIGURES
# =============================================================================

def create_ope_error_plot(summary: pd.DataFrame, show_plot: bool = False) -> go.Figure:
    """Mean relative error of each estimator against the true Thompson Sampling CTR, one panel per campaign.

    Each campaign uses its own marathon-winning click model, named in the panel title.
    """
    campaigns = list(dict.fromkeys(summary["campaign"]))
    titles = [f"{c.capitalize()} · click model: {summary.loc[summary['campaign'] == c, 'base_model'].iloc[0]}"
              for c in campaigns]
    fig = make_subplots(rows=1, cols=len(campaigns), shared_xaxes=True, horizontal_spacing=0.09,
                        subplot_titles=titles)
    for col, campaign in enumerate(campaigns, start=1):
        grp = summary[summary["campaign"] == campaign].sort_values("relative_error_mean")
        err = 100 * grp["relative_error_mean"]
        sd = 100 * grp["relative_error_std"].fillna(0)
        fig.add_trace(go.Scatter(
            x=err, y=[ESTIMATOR_LABELS.get(e, e) for e in grp["estimator"]], mode="markers",
            name="Relative error", showlegend=False,
            marker=dict(size=11, color=COLOR_CONTROL, line=dict(color=SURFACE, width=1.5)),
            error_x=dict(type="data", symmetric=False, array=sd, arrayminus=np.minimum(sd, err),
                         thickness=2, width=0, color=COLOR_CONTROL),
            customdata=np.stack([100 * grp["mean_estimate"], 100 * grp["ground_truth"], sd, grp["n_runs"]], axis=-1),
            hovertemplate=("<b>%{y}</b><br>Relative error %{x:.1f}% ± %{customdata[2]:.1f}%"
                           "<br>Mean estimate %{customdata[0]:.3f}% vs true %{customdata[1]:.3f}%"
                           "<br>%{customdata[3]} bootstrap runs<extra></extra>"),
        ), row=1, col=col)
        fig.update_yaxes(autorange="reversed", row=1, col=col)
        fig.update_xaxes(title_text="Relative error (%)", ticksuffix="%", rangemode="tozero", row=1, col=col)
    _base_layout(fig, "Off-policy estimates vs the true Thompson Sampling CTR",
                 "Lower is better. Bars show ±1 std over bootstrap runs.")
    _left_align_subplot_titles(fig, len(campaigns))
    _show_fig(fig, show_plot)
    return fig


def create_ab_vs_ope_plot(comparison: pd.DataFrame, show_plot: bool = False) -> go.Figure:
    """Each estimator's estimated lift over the A/B test's lift and 95% interval, one panel per campaign."""
    data = comparison
    campaigns = list(dict.fromkeys(data["campaign"]))
    fig = make_subplots(rows=1, cols=len(campaigns), horizontal_spacing=0.09,
                        subplot_titles=[c.capitalize() for c in campaigns])
    for col, campaign in enumerate(campaigns, start=1):
        grp = data[data["campaign"] == campaign].reset_index(drop=True)
        ab = grp.iloc[0]
        labels = [ESTIMATOR_LABELS.get(e, e) for e in grp["estimator"]]
        # Shapes do not appear in the legend or widen the axis, so the A/B lift gets a legend-only
        # entry and the axis range is set to cover zero, the A/B interval and every estimate.
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines", line=dict(color=COLOR_TREATMENT, width=2),
            name="A/B lift (shaded: 95% CI)", legendgroup="ab", showlegend=(col == 1), hoverinfo="skip",
        ), row=1, col=col)
        values = [0.0, 100 * ab["ab_lift_lower"], 100 * ab["ab_lift_upper"], *(100 * grp["ope_lift"])]
        low, high = min(values), max(values)
        pad = 0.06 * (high - low or 1.0)
        fig.update_xaxes(range=[low - pad, high + pad], row=1, col=col)
        fig.add_trace(go.Scatter(
            x=100 * grp["ope_lift"], y=labels, mode="markers", name="Off-policy estimate",
            legendgroup="ope", showlegend=(col == 1),
            marker=dict(size=10, color=COLOR_CONTROL, line=dict(color=SURFACE, width=1.5)),
            customdata=np.stack([grp["within_ab_ci"].map({True: "inside", False: "outside"})], axis=-1),
            hovertemplate="<b>%{y}</b><br>Estimated lift %{x:+.1f}%<br>%{customdata[0]} the A/B interval<extra></extra>",
        ), row=1, col=col)
        if "selected_by_campaign_split" in grp.columns:
            pick = grp[grp["selected_by_campaign_split"].astype(bool)]
            if not pick.empty:
                fig.add_trace(go.Scatter(
                    x=100 * pick["ope_lift"], y=[ESTIMATOR_LABELS.get(e, e) for e in pick["estimator"]],
                    mode="markers", name="Chosen without looking at this campaign",
                    legendgroup="pick", showlegend=(col == 1),
                    marker=dict(size=17, color="rgba(0,0,0,0)", symbol="circle",
                                line=dict(color=INK, width=2)),
                    hovertemplate="<b>%{y}</b> (fair pick)<br>Estimated lift %{x:+.1f}%<extra></extra>",
                ), row=1, col=col)
        # Shapes go in after the traces: plotly skips shapes added to a subplot that has no data yet.
        fig.add_vrect(x0=100 * ab["ab_lift_lower"], x1=100 * ab["ab_lift_upper"], fillcolor=COLOR_TREATMENT,
                      opacity=0.15, line_width=0, layer="below", row=1, col=col)
        fig.add_vline(x=0, line=dict(color=INK_MUTED, width=1, dash="dash"), row=1, col=col)
        fig.add_vline(x=100 * ab["ab_lift"], line=dict(color=COLOR_TREATMENT, width=2), row=1, col=col)
        fig.update_yaxes(type="category", categoryorder="array", categoryarray=labels, autorange="reversed",
                         row=1, col=col)
        fig.update_xaxes(title_text="Lift over random (%)", ticksuffix="%", row=1, col=col)
    _base_layout(fig, "A/B lift vs off-policy estimate",
                 "Estimates use only the random arm's logs and each campaign's winning click model. "
                 "Ringed: the estimator chosen on the other campaigns.")
    _left_align_subplot_titles(fig, len(campaigns))
    _show_fig(fig, show_plot)
    return fig


def create_estimator_selection_plot(selection: pd.DataFrame, show_plot: bool = False) -> go.Figure:
    """Held-out error of the fairly chosen estimator, next to the best in hindsight and the median."""
    data = selection.copy()
    split_names = {"campaign": "Campaign split", "time": "Time split"}
    data["label"] = [f"{split_names.get(sp, sp)} · {h}" + (" (2nd half)" if sp == "time" else "")
                     for sp, h in zip(data["split"], data["held_out"])]
    fig = go.Figure()
    for column, name, color, size in (
        ("median_error", "Median estimator", GRID, 10),
        ("oracle_error", "Best in hindsight (optimistic)", COLOR_CONTROL, 10),
        ("test_error", "Fairly chosen estimator", COLOR_TREATMENT, 13),
    ):
        custom = data[["selected_estimator", "oracle_estimator", "selected_on", "selected_rank", "n_estimators"]]
        fig.add_trace(go.Scatter(
            x=100 * data[column], y=data["label"], mode="markers", name=name,
            marker=dict(size=size, color=color, line=dict(color=INK_MUTED if color == GRID else SURFACE, width=1.5)),
            customdata=custom.to_numpy(dtype=object),
            hovertemplate=("<b>%{y}</b><br>" + name + ": %{x:.1f}% error"
                           "<br>Chosen: %{customdata[0]} (on %{customdata[2]}), ranked %{customdata[3]} of %{customdata[4]}"
                           "<br>Best in hindsight: %{customdata[1]}<extra></extra>"),
        ))
    fig.update_xaxes(title_text="Relative error on held-out data (%)", ticksuffix="%", rangemode="tozero")
    fig.update_yaxes(autorange="reversed")
    _base_layout(fig, "Fair estimator selection: how wrong is the estimator you would have picked?",
                 "Each estimator is chosen on one slice of data and scored on another it never saw.",
                 height=180 + 48 * len(data))
    _show_fig(fig, show_plot)
    return fig


def _left_align_subplot_titles(fig: go.Figure, n_cols: int) -> None:
    """Move each panel title to its panel's left edge, matching the figure title."""
    for i, annotation in enumerate(fig.layout.annotations[:n_cols]):
        axis = fig.layout["xaxis" if i == 0 else f"xaxis{i + 1}"]
        annotation.update(x=axis.domain[0], xanchor="left", font=dict(size=13, color=INK))

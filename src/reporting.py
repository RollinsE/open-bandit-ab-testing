# -*- coding: utf-8 -*-
"""Stage 6: interactive Plotly figures and the written report.

What gets written to ``builds/<run_id>/reports/``:

1. ``figures/<name>.html`` for every figure: interactive Plotly, opens in any
   browser. ``figures/<name>.png`` as well when Kaleido is installed; a missing
   PNG is a logged warning, never a failed run.
2. ``report.html``: the whole report in one page, with every figure embedded
   as an interactive chart.
3. ``report.md``: the same text for GitHub, linking each figure (and showing
   its PNG where one was written).

Figures:

* ``ab_ctr_by_campaign``: each arm's CTR with its 95% interval
* ``ab_lift``: the lift with its 95% interval and significance
* ``ab_sample_size``: impressions per arm needed for a range of lifts
* ``model_leaderboard``: the reward model marathon, winner highlighted
* ``ope_relative_error``: each estimator's error against the true CTR
* ``estimator_selection``: held-out error of the fairly chosen estimator
* ``ab_vs_ope_lift``: the A/B lift interval with each estimator's lift on top
"""

import html
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from src.constants import ESTIMATOR_LABELS, PROJECT_NAME
from src.utils import get_logger
from src.visualization import (
    configure_colab_plotly,
    create_ab_ctr_plot,
    create_ab_lift_plot,
    create_ab_vs_ope_plot,
    create_estimator_selection_plot,
    create_model_leaderboard_plot,
    create_ope_error_plot,
    create_power_curve_plot,
)

FIGURE_MARKER = "@@figure:{}@@"
PNG_WIDTH = 1000  # pixels; the figures are laid out for this width (Kaleido's default is 700)


# =============================================================================
# SAVING FIGURES
# =============================================================================

def save_plotly_bundle(fig, base_name: str, config: dict) -> dict:
    """Save one figure as interactive HTML and, when Kaleido is available, PNG.

    Args:
        fig: Plotly figure.
        base_name: File name without extension.
        config: Run configuration with ``figure_dir``.

    Returns:
        ``{"html": path, "png": path or None, "figure": fig}``.
    """
    LOG = get_logger()
    figure_dir = Path(config["figure_dir"])
    html_path = figure_dir / f"{base_name}.html"
    png_path = figure_dir / f"{base_name}.png"
    fig.write_html(str(html_path), include_plotlyjs="cdn", full_html=True)
    try:
        fig.write_image(str(png_path), width=PNG_WIDTH, height=fig.layout.height, scale=2)
        png_written = str(png_path)
    except Exception as exc:
        reason = next((line.strip() for line in str(exc).splitlines() if line.strip()), type(exc).__name__)
        LOG.warning(f"Could not write PNG for {base_name}: {reason}")
        png_written = None
    LOG.info(f"Saved figure bundle: {html_path}{' / ' + png_written if png_written else ''}")
    return {"html": str(html_path), "png": png_written, "figure": fig}


# =============================================================================
# WRITTEN REPORT
# =============================================================================

def _pct(x: float, digits: int = 1, signed: bool = False) -> str:
    if pd.isna(x):
        return "n/a"
    return f"{100 * x:+.{digits}f}%" if signed else f"{100 * x:.{digits}f}%"


def build_report_markdown(summary: pd.DataFrame, ab: Optional[pd.DataFrame], leaderboard: Optional[pd.DataFrame],
                          ope: Optional[pd.DataFrame], selection: Optional[pd.DataFrame],
                          comparison: Optional[pd.DataFrame], config: dict, figures: Dict[str, dict]) -> str:
    """Assemble the report text from the stage outputs.

    Figures appear as ``@@figure:<name>@@`` marker lines, which
    ``render_markdown`` and ``markdown_to_html`` turn into an image link or an
    embedded interactive chart.
    """
    def fig_line(name: str) -> list:
        return [FIGURE_MARKER.format(name), ""] if name in figures else []

    alpha = config.get("alpha", 0.05)
    lines = [f"# {PROJECT_NAME}: run {config.get('run_id', '')}", ""]
    if config.get("using_bundled_sample"):
        lines += ["> **Sample data.** This run used the 10,000-row sample per arm and campaign. "
                  "The figures show the method working; they are too small for real conclusions.", ""]

    lines += ["## Data", "",
              "| Arm | Campaign | Impressions | Clicks | CTR |", "|---|---|---:|---:|---:|"]
    for _, r in summary.iterrows():
        lines.append(f"| {r['arm']} | {r['campaign']} | {r['impressions']:,} | {r['clicks']:,} | {_pct(r['ctr'], 3)} |")

    if ab is not None:
        lines += ["", "## A/B test: Thompson Sampling vs random", ""]
        lines += fig_line("ab_ctr_by_campaign") + fig_line("ab_lift")
        lines += ["| Campaign | Control share | Control CTR | Treatment CTR | Lift (95% CI) | p-value | Power | Impressions/arm needed | Decision |",
                  "|---|---:|---:|---:|---|---:|---:|---:|---|"]
        for _, r in ab.iterrows():
            needed = "n/a" if r["abs_diff"] == 0 else f"{r['n_per_arm_needed']:,.0f}"
            share = _pct(r["control_share"], 1) if "control_share" in r else "n/a"
            lines.append(
                f"| {r['campaign']} | {share} | {_pct(r['control_ctr'], 3)} | {_pct(r['treatment_ctr'], 3)} | "
                f"{_pct(r['rel_lift'], 1, True)} ({_pct(r['rel_lift_lower'], 0, True)} to {_pct(r['rel_lift_upper'], 0, True)}) | "
                f"{r['p_value']:.3f} | {_pct(r['achieved_power'], 0)} | {needed} | {r['decision']} |")
        checked = "expected_control_share" in ab and ab["expected_control_share"].notna().any()
        lines += ["", ("Control share is the random arm's share of impressions; the sample ratio check compared it "
                       f"with an intended {_pct(ab['expected_control_share'].iloc[0], 1)}." if checked else
                       "Control share is the random arm's share of impressions. The intended split is not "
                       "documented, so no sample ratio check was run (`--expected-control-share` turns it on).")]
        lines += ["", f"Lift is treatment CTR / control CTR - 1. Power is for the observed lift at alpha {alpha}. "
                  "Impressions are the unit of analysis; one page view shows three items, so intervals are somewhat too narrow.", ""]
        lines += ["### Planning the next test", ""] + fig_line("ab_sample_size")

    if leaderboard is not None:
        rule = config.get("winner_rule", "simplest")
        if rule == "simplest":
            tolerance = float(config.get("winner_tolerance", 0.0) or 0.0)
            rule_text = ("The winner is the simplest model whose log loss is within a margin of the lowest: the wider "
                         f"of {100 * tolerance:.2f}% of the best log loss and one standard error across the folds. "
                         "The click model is retrained in every bootstrap run, so a slightly better but much "
                         "slower model would make the off-policy stage take hours longer for a negligible gain.")
        else:
            rule_text = "The model with the lowest log loss wins."
        lines += ["## Reward model marathon", "",
                  "The model-based estimators need a click model. Every candidate is scored on out-of-fold log loss "
                  f"of the random arm's logged clicks. {rule_text} "
                  "Thompson Sampling's true CTR is never used to choose.", ""]
        lines += fig_line("model_leaderboard")
        lines += ["| Campaign | Winner | Log loss | Lowest log loss (model) | ROC AUC | Skill vs baseline | Models entered |",
                  "|---|---|---:|---|---:|---:|---:|"]
        for campaign, grp in leaderboard.groupby("campaign", sort=False):
            w = grp[grp["winner"].astype(bool)].iloc[0]
            candidates = grp[grp["Role"] != "baseline"]
            best = candidates.loc[candidates["Log loss"].idxmin()]
            entered = len(candidates)
            lines.append(f"| {campaign} | **{w['Model']}** | {w['Log loss']:.5f} | "
                         f"{best['Log loss']:.5f} ({best['Model']}) | {w['ROC AUC']:.3f} | "
                         f"{_pct(w['Skill vs baseline'], 2, True)} | {entered} |")
        lines += ["", "The full leaderboard, with every model's log loss, Brier score and AUC, is in "
                  "`artifacts/model_performance.csv`.", ""]

    if ope is not None:
        lines += ["## Off-policy evaluation", "",
                  "Estimating Thompson Sampling's CTR from the random arm's logs only, scored against its true CTR.", ""]
        lines += fig_line("ope_relative_error")
        lines += ["| Campaign | Click model | Estimator | Mean estimate | True CTR | Relative error |",
                  "|---|---|---|---:|---:|---:|"]
        for _, r in ope.iterrows():
            lines.append(f"| {r['campaign']} | {r['base_model']} | "
                         f"{ESTIMATOR_LABELS.get(r['estimator'], r['estimator'])} | {_pct(r['mean_estimate'], 3)} | "
                         f"{_pct(r['ground_truth'], 3)} | {_pct(r['relative_error_mean'])} ± {_pct(r['relative_error_std'])} |")
        lines.append("")

    if selection is not None and not selection.empty:
        lines += ["## Fair estimator selection", "",
                  "Picking the estimator closest to the truth and reporting that same error flatters it. "
                  "Here each estimator is chosen on one slice of data and scored on another it never saw: "
                  "leave one campaign out, and first half of the logging period then second half. "
                  "The best estimator in hindsight is shown for context only.", ""]
        lines += fig_line("estimator_selection")
        lines += ["| Split | Held out | Chosen on | Chosen estimator | Held-out error | Rank | Best in hindsight | Median error |",
                  "|---|---|---|---|---:|---:|---|---:|"]
        for _, r in selection.iterrows():
            lines.append(
                f"| {r['split']} | {r['held_out']}{' (2nd half)' if r['split'] == 'time' else ''} | {r['selected_on']} | "
                f"{ESTIMATOR_LABELS.get(r['selected_estimator'], r['selected_estimator'])} | {_pct(r['test_error'])} | "
                f"{int(r['selected_rank'])} of {int(r['n_estimators'])} | "
                f"{ESTIMATOR_LABELS.get(r['oracle_estimator'], r['oracle_estimator'])} ({_pct(r['oracle_error'])}) | "
                f"{_pct(r['median_error'])} |")
        if (selection["split"] == "campaign").any():
            lines += ["", "The \"all\" campaign shares many items with \"men\" and \"women\", so the campaign split "
                      "is not a fully independent test; the time split is the cleaner check."]
        lines.append("")

    if comparison is not None:
        lines += ["## A/B test vs off-policy estimate", ""] + fig_line("ab_vs_ope_lift")
        if "selected_by_campaign_split" in comparison.columns:
            for _, r in comparison[comparison["selected_by_campaign_split"].astype(bool)].iterrows():
                lines.append(
                    f"- **{r['campaign']}**, using the estimator chosen on the other campaigns "
                    f"({ESTIMATOR_LABELS.get(r['estimator'], r['estimator'])}): estimated lift "
                    f"{_pct(r['ope_lift'], 1, True)}; the A/B test measured {_pct(r['ab_lift'], 1, True)} "
                    f"(95% CI {_pct(r['ab_lift_lower'], 0, True)} to {_pct(r['ab_lift_upper'], 0, True)}).")
            lines.append("")
        for campaign, grp in comparison.groupby("campaign"):
            inside = int(grp["within_ab_ci"].astype(bool).sum())
            direction = ("the A/B test saw no difference, so direction does not apply"
                         if grp["same_direction"].isna().all()
                         else f"{int(grp['same_direction'].sum())} point the same way as the A/B result")
            lines.append(f"- **{campaign}**: {inside} of {len(grp)} estimates fall inside the A/B lift interval; "
                         f"{direction}.")

    lines += ["", "## Files", "",
              "Every table above is in `artifacts/` as CSV. Each figure is in `reports/figures/` as interactive HTML. "
              "The full log is in `logs/`."]
    return "\n".join(lines) + "\n"


def render_markdown(markdown: str, figures: Dict[str, dict]) -> str:
    """Replace figure markers with a PNG image (when written) and a link to the interactive HTML."""
    out = []
    for line in markdown.splitlines():
        name = _marker_name(line)
        if name is None:
            out.append(line)
            continue
        bundle = figures[name]
        if bundle.get("png"):
            out.append(f"![{name}](figures/{Path(bundle['png']).name})")
        out.append(f"[Open interactive figure: {name}](figures/{Path(bundle['html']).name})")
    return "\n".join(out) + "\n"


def markdown_to_html(markdown: str, title: str, figures: Dict[str, dict]) -> str:
    """Convert the report's own markdown to one HTML page with embedded interactive figures."""
    out, in_table, first_figure = [], False, True
    for line in markdown.splitlines():
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if set("".join(cells)) <= set("-:"):
                continue
            tag = "th" if not in_table else "td"
            if not in_table:
                out.append("<table>")
                in_table = True
            out.append("<tr>" + "".join(f"<{tag}>{_inline(c)}</{tag}>" for c in cells) + "</tr>")
            continue
        if in_table:
            out.append("</table>")
            in_table = False
        name = _marker_name(line)
        if name is not None:
            # The first figure loads plotly.js from the CDN at the version Plotly wrote it for.
            out.append('<div class="figure">' + figures[name]["figure"].to_html(
                full_html=False, include_plotlyjs="cdn" if first_figure else False,
                config={"displaylogo": False, "responsive": True}) + "</div>")
            first_figure = False
        elif line.startswith("# "):
            out.append(f"<h1>{_inline(line[2:])}</h1>")
        elif line.startswith("### "):
            out.append(f"<h3>{_inline(line[4:])}</h3>")
        elif line.startswith("## "):
            out.append(f"<h2>{_inline(line[3:])}</h2>")
        elif line.startswith("> "):
            out.append(f"<blockquote>{_inline(line[2:])}</blockquote>")
        elif line.startswith("- "):
            out.append(f"<p>• {_inline(line[2:])}</p>")
        elif line.strip():
            out.append(f"<p>{_inline(line)}</p>")
    if in_table:
        out.append("</table>")
    style = ("body{font-family:Inter,system-ui,sans-serif;max-width:64rem;margin:2rem auto;padding:0 1rem;"
             "color:#0b0b0b;background:#fcfcfb}"
             "table{border-collapse:collapse;font-size:.85rem;margin:1rem 0}"
             "td,th{border-bottom:1px solid #e4e3df;padding:.3rem .6rem;text-align:left}"
             ".figure{margin:1rem 0}"
             "blockquote{border-left:3px solid #eda100;margin:1rem 0;padding:.2rem 1rem;background:#fff8e6}")
    return (f"<!doctype html><html><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width, initial-scale=1'>"
            f"<title>{html.escape(title)}</title><style>{style}</style></head><body>{''.join(out)}</body></html>")


def _marker_name(line: str) -> Optional[str]:
    """Return the figure name if ``line`` is a figure marker, else ``None``."""
    prefix, suffix = FIGURE_MARKER.split("{}")
    if line.startswith(prefix) and line.endswith(suffix):
        return line[len(prefix):-len(suffix)]
    return None


def _inline(text: str) -> str:
    """Escape text and render **bold** and `code`."""
    text = html.escape(text)
    parts = text.split("**")
    text = "".join(f"<b>{p}</b>" if i % 2 else p for i, p in enumerate(parts))
    parts = text.split("`")
    return "".join(f"<code>{p}</code>" if i % 2 else p for i, p in enumerate(parts))


# =============================================================================
# STAGE ENTRY POINT
# =============================================================================

def build_reports(summary: pd.DataFrame, ab: Optional[pd.DataFrame], power_table: Optional[pd.DataFrame],
                  leaderboard: Optional[pd.DataFrame], ope: Optional[pd.DataFrame],
                  selection: Optional[pd.DataFrame], comparison: Optional[pd.DataFrame], config: dict) -> dict:
    """Run stage 6: save every figure and the report for the stages that ran.

    Returns:
        Mapping of output name to path for the files written.
    """
    LOG = get_logger()
    configure_colab_plotly()
    show = bool(config.get("enable_inline_display"))
    figures: Dict[str, dict] = {}
    if ab is not None:
        figures["ab_ctr_by_campaign"] = save_plotly_bundle(create_ab_ctr_plot(ab, show), "ab_ctr_by_campaign", config)
        figures["ab_lift"] = save_plotly_bundle(create_ab_lift_plot(ab, show), "ab_lift", config)
    if power_table is not None:
        figures["ab_sample_size"] = save_plotly_bundle(create_power_curve_plot(power_table, show),
                                                       "ab_sample_size", config)
    if leaderboard is not None:
        figures["model_leaderboard"] = save_plotly_bundle(create_model_leaderboard_plot(leaderboard, show),
                                                          "model_leaderboard", config)
    if ope is not None:
        figures["ope_relative_error"] = save_plotly_bundle(create_ope_error_plot(ope, show),
                                                           "ope_relative_error", config)
    if selection is not None and not selection.empty:
        figures["estimator_selection"] = save_plotly_bundle(create_estimator_selection_plot(selection, show),
                                                            "estimator_selection", config)
    if comparison is not None:
        figures["ab_vs_ope_lift"] = save_plotly_bundle(create_ab_vs_ope_plot(comparison, show),
                                                       "ab_vs_ope_lift", config)

    markdown = build_report_markdown(summary, ab, leaderboard, ope, selection, comparison, config, figures)
    files = config["output_files"]
    Path(files["report_md"]).write_text(render_markdown(markdown, figures), encoding="utf-8")
    Path(files["report_html"]).write_text(markdown_to_html(markdown, PROJECT_NAME, figures), encoding="utf-8")
    LOG.info("Wrote report: %s", files["report_html"])
    LOG.info("Wrote report: %s", files["report_md"])

    written = {f"figure_{name}": bundle["html"] for name, bundle in figures.items()}
    written.update(report_md=files["report_md"], report_html=files["report_html"])
    return written

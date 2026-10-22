"""Export a build's figures as static PNG images, for example for the README.

GitHub cannot display interactive Plotly charts, so the README uses PNG copies.
The pipeline writes PNGs itself when Kaleido is installed; this script rebuilds
the figures from a finished build's artifacts, so images can be made later
without re-running anything.

Each figure is written twice, for light and dark backgrounds (``<name>.png`` and
``<name>-dark.png``), so a README can show the one matching the reader's theme.

Usage:
    python scripts/export_figures.py outputs/releases/latest docs/images
    python scripts/export_figures.py outputs/releases/latest docs/images --only ab_vs_ope_lift estimator_selection

Requires Kaleido (``pip install kaleido==0.2.1``, included in requirements/colab.txt).
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import visualization as V  # noqa: E402

FIGURES = {
    "ab_ctr_by_campaign": (V.create_ab_ctr_plot, "ab_test_results.csv"),
    "ab_lift": (V.create_ab_lift_plot, "ab_test_results.csv"),
    "ab_sample_size": (V.create_power_curve_plot, "ab_test_power.csv"),
    "model_leaderboard": (V.create_model_leaderboard_plot, "model_performance.csv"),
    "ope_relative_error": (V.create_ope_error_plot, "ope_summary.csv"),
    "ab_vs_ope_lift": (V.create_ab_vs_ope_plot, "ab_vs_ope_comparison.csv"),
    "estimator_selection": (V.create_estimator_selection_plot, "estimator_selection.csv"),
}


def build_figures(build_dir: Path, only: list[str] | None = None, theme: str = "light") -> dict:
    """Rebuild the named figures (default: all) from ``build_dir/artifacts`` in ``theme``."""
    artifacts = Path(build_dir) / "artifacts"
    figures = {}
    with V.theme(theme):
        for name, (make, table) in FIGURES.items():
            if only and name not in only:
                continue
            path = artifacts / table
            if path.exists():
                figures[name] = make(pd.read_csv(path))
    return figures


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("build_dir", type=Path, help="A build or release folder")
    parser.add_argument("out_dir", type=Path, help="Folder for the PNG files")
    parser.add_argument("--only", nargs="+", choices=sorted(FIGURES), help="Export only these figures")
    parser.add_argument("--width", type=int, default=1000, help="Image width in pixels (default 1000)")
    parser.add_argument("--scale", type=float, default=2, help="Resolution multiplier (default 2)")
    parser.add_argument("--themes", nargs="+", choices=sorted(V.THEMES), default=["light", "dark"],
                        help="Themes to export (default: both)")
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for theme in args.themes:
        suffix = "" if theme == "light" else f"-{theme}"
        for name, fig in build_figures(args.build_dir, args.only, theme).items():
            path = args.out_dir / f"{name}{suffix}.png"
            fig.write_image(path, width=args.width, height=fig.layout.height, scale=args.scale)
            print(f"Wrote {path}")


if __name__ == "__main__":
    main()

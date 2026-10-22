# -*- coding: utf-8 -*-
"""Runtime detection, data paths and the per-build output layout.

How a run is configured:

1. Runtime. ``detect_runtime`` returns "colab", "production" or "local". It can
   be forced with ``--mode`` or the ``OBD_ENV`` environment variable.
2. Data. The bundled 10,000-row sample in ``data/obd`` is used unless
   ``--data-dir`` or ``OBD_DATA_DIR`` points at the full Open Bandit Dataset.
3. Outputs. Each run writes to ``<output_root>/builds/<run_id>/`` with
   ``artifacts/``, ``reports/``, ``logs/`` and ``checkpoints/`` inside, so runs
   never overwrite each other. A successful build can be promoted to ``releases/<name>``.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from src.constants import CAMPAIGNS, ENV_DATA_DIR, ENV_RUNTIME

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BUNDLED_DATA_DIR = PROJECT_ROOT / "data" / "obd"


# =============================================================================
# RUNTIME
# =============================================================================

def detect_runtime() -> str:
    """Return "colab", "production" or "local" for the current environment."""
    try:
        if "google.colab" in sys.modules:
            return "colab"
    except Exception:
        pass

    try:
        from IPython import get_ipython

        if "google.colab" in str(get_ipython()):
            return "colab"
    except Exception:
        pass

    if os.environ.get(ENV_RUNTIME) == "production":
        return "production"
    return "local"


# =============================================================================
# DATA
# =============================================================================

def find_data_directory(data_dir: Optional[str] = None) -> Path:
    """Resolve the Open Bandit Dataset folder.

    Args:
        data_dir: Explicit folder from ``--data-dir``. Takes priority.

    Returns:
        A folder containing ``<policy>/<campaign>/<campaign>.csv``. Falls back
        to the bundled sample when nothing else is given.
    """
    candidate = data_dir or os.environ.get(ENV_DATA_DIR)
    return Path(candidate).expanduser().resolve() if candidate else BUNDLED_DATA_DIR


def is_bundled_sample(data_dir: Path) -> bool:
    """True when the run uses the small sample shipped with the repository."""
    return Path(data_dir).resolve() == BUNDLED_DATA_DIR.resolve()


# =============================================================================
# OUTPUT LAYOUT
# =============================================================================

def apply_output_directory(config: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
    """Create the build folders and record every output path in ``config``.

    Args:
        config: Configuration dictionary to update in place.
        output_dir: The build folder, normally ``<output_root>/builds/<run_id>``.

    Returns:
        The same dictionary with ``*_dir`` keys and an ``output_files`` map.
    """
    build_dir = Path(output_dir)
    artifact_dir = build_dir / "artifacts"
    report_dir = build_dir / "reports"
    figure_dir = report_dir / "figures"
    log_dir = build_dir / "logs"
    checkpoint_dir = build_dir / "checkpoints"

    for path in (build_dir, artifact_dir, report_dir, figure_dir, log_dir, checkpoint_dir):
        path.mkdir(parents=True, exist_ok=True)

    config["output_dir"] = str(build_dir)
    config["artifact_dir"] = str(artifact_dir)
    config["report_dir"] = str(report_dir)
    config["figure_dir"] = str(figure_dir)
    config["log_dir"] = str(log_dir)
    config["checkpoint_dir"] = str(checkpoint_dir)

    config["output_files"] = {
        "dataset_summary": str(artifact_dir / "dataset_summary.csv"),
        "ab_test_results": str(artifact_dir / "ab_test_results.csv"),
        "ab_test_power": str(artifact_dir / "ab_test_power.csv"),
        "model_performance": str(artifact_dir / "model_performance.csv"),
        "reward_model_winners": str(artifact_dir / "reward_model_winners.json"),
        "model_diagnostics": str(artifact_dir / "model_diagnostics.json"),
        "ope_estimates_per_run": str(artifact_dir / "ope_estimates_per_run.csv"),
        "ope_summary": str(artifact_dir / "ope_summary.csv"),
        "ope_time_split": str(artifact_dir / "ope_time_split.csv"),
        "estimator_selection": str(artifact_dir / "estimator_selection.csv"),
        "ab_vs_ope": str(artifact_dir / "ab_vs_ope_comparison.csv"),
        "report_md": str(report_dir / "report.md"),
        "report_html": str(report_dir / "report.html"),
        "run_manifest": str(build_dir / "run_manifest.json"),
    }
    return config


def get_config(runtime: Optional[str] = None, data_dir: Optional[str] = None) -> Dict[str, Any]:
    """Build the base configuration for a run.

    Args:
        runtime: "colab", "production" or "local". Detected when ``None``.
        data_dir: Optional override for the dataset folder.

    Returns:
        Configuration dictionary. Output paths are added by
        ``apply_output_directory``.
    """
    runtime = runtime or detect_runtime()
    data_directory = find_data_directory(data_dir)
    return {
        "runtime": runtime,
        "project_root": str(PROJECT_ROOT),
        "data_directory": str(data_directory),
        "using_bundled_sample": is_bundled_sample(data_directory),
        "campaigns": list(CAMPAIGNS),
    }


def validate_runtime_configuration(config: Dict[str, Any]) -> None:
    """Fail early when the dataset folder is missing any campaign file.

    Raises:
        FileNotFoundError: Listing every missing file, so all can be fixed at once.
    """
    data_dir = Path(config["data_directory"])
    missing = []
    for policy in ("random", "bts"):
        for campaign in config["campaigns"]:
            for name in (f"{campaign}.csv", "item_context.csv"):
                path = data_dir / policy / campaign / name
                if not path.exists():
                    missing.append(str(path))
    if missing:
        raise FileNotFoundError(
            "Open Bandit Dataset files not found:\n  " + "\n  ".join(missing)
        )

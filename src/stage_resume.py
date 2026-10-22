# -*- coding: utf-8 -*-
"""Stage selection and resuming from an earlier build.

Stage 1 is cheap and always runs, because the other stages need the loaded data.
Stage 4 (off-policy evaluation) needs the reward-model winners from stage 3,
stage 5 needs the outputs of stages 2 and 4, and stage 6 reports whatever has run.
When a stage is skipped, its outputs are read from ``--resume-build`` instead, so
for example ``--only-stage 6`` rebuilds the report from an old run without
re-running the model marathon or the off-policy benchmark.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pandas as pd

from src.constants import STAGES


class ResumeArtifactError(RuntimeError):
    """Raised when a resume build is missing or lacks a needed file."""


@dataclass
class StagePlan:
    """Which stages run in this build."""

    from_stage: int = 1
    to_stage: int = 6
    stages: List[int] = field(default_factory=list)

    def includes(self, stage: int) -> bool:
        return stage in self.stages


def resolve_stage_plan(only_stage: Optional[int] = None, from_stage: int = 1, to_stage: int = 6) -> StagePlan:
    """Turn the stage arguments into a ``StagePlan``.

    Raises:
        ValueError: If the range is empty or names an unknown stage.
    """
    if only_stage is not None:
        from_stage = to_stage = only_stage
    if from_stage not in STAGES or to_stage not in STAGES:
        raise ValueError(f"Stages must be between {min(STAGES)} and {max(STAGES)}")
    if from_stage > to_stage:
        raise ValueError(f"--from-stage {from_stage} is after --to-stage {to_stage}")
    return StagePlan(from_stage, to_stage, list(range(from_stage, to_stage + 1)))


def validate_resume_build(resume_build: Optional[str]) -> Optional[Path]:
    """Check a resume build folder exists and has an ``artifacts`` folder."""
    if not resume_build:
        return None
    path = Path(resume_build)
    if not (path / "artifacts").is_dir():
        raise ResumeArtifactError(f"Resume build not found or has no artifacts/: {path}")
    return path


def require_resume_for_skipped_dependencies(plan: StagePlan, resume_build: Optional[str]) -> None:
    """Stages 4 to 6 need earlier outputs; insist on a resume build when they were skipped."""
    needs_earlier = (plan.includes(4) and not plan.includes(3)) or \
                    (plan.includes(5) and not (plan.includes(2) and plan.includes(4))) or \
                    (plan.includes(6) and not any(plan.includes(s) for s in (2, 3, 4)))
    if needs_earlier and not resume_build:
        raise ResumeArtifactError(
            "This stage plan skips stages whose outputs are needed. Pass --resume-build <builds/run_id>."
        )


def load_json_artifact(resume_build: Optional[Path], filename: str) -> Optional[dict]:
    """Read one JSON file from a resume build's ``artifacts`` folder, or ``None`` if absent."""
    if resume_build is None:
        return None
    path = resume_build / "artifacts" / filename
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def load_artifact(resume_build: Optional[Path], filename: str) -> Optional[pd.DataFrame]:
    """Read one CSV from a resume build's ``artifacts`` folder, or ``None`` if absent."""
    if resume_build is None:
        return None
    path = resume_build / "artifacts" / filename
    return pd.read_csv(path) if path.exists() else None

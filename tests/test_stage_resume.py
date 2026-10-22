import pytest

from src.stage_resume import ResumeArtifactError, require_resume_for_skipped_dependencies, resolve_stage_plan


def test_only_stage_sets_both_ends():
    plan = resolve_stage_plan(only_stage=3)
    assert plan.stages == [3]


def test_reversed_range_is_rejected():
    with pytest.raises(ValueError):
        resolve_stage_plan(from_stage=4, to_stage=2)


def test_skipped_dependencies_need_resume_build():
    for stage in (4, 5, 6):
        with pytest.raises(ResumeArtifactError):
            require_resume_for_skipped_dependencies(resolve_stage_plan(only_stage=stage), None)
    require_resume_for_skipped_dependencies(resolve_stage_plan(from_stage=1, to_stage=6), None)
    require_resume_for_skipped_dependencies(resolve_stage_plan(from_stage=1, to_stage=3), None)

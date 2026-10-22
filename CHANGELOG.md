# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-10-22

### Added

- Six-stage pipeline: data loading and validation, A/B test, reward model marathon, off-policy evaluation with fair estimator selection, A/B vs off-policy comparison, and reporting.
- A/B analysis with Wilson, delta-method and bootstrap intervals, z-test and Fisher's exact test, power analysis and an optional sample ratio check.
- Reward model marathon with nested tuning of the top families, an ensemble, and a simplest-within-margin winner rule (`--winner-rule`, `--winner-tolerance`).
- Seven off-policy estimators in `obp_lite`, verified against the original Open Bandit Pipeline.
- Fair estimator selection by leave-one-campaign-out and time split.
- Interactive Plotly figures and HTML and Markdown reports.
- Checkpointing with `--continue-build auto|latest|<build>` and a build lock, for interrupted runs.
- Partial runs with `--from-stage`, `--to-stage`, `--only-stage` and `--resume-build`, which reuse only upstream results and refuse data that differs from the earlier build.
- Full-scale support: light loading, per-context prediction, marathon sampling, a click-model training cap in the benchmark (`--ope-fit-max-rows`) and memory-aware parallelism.
- Colab notebooks, Makefile, pytest suite and GitHub Actions workflow.
- Light and dark figure themes, and `scripts/export_figures.py` to export a build's figures as PNG images in both.

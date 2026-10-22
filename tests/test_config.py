from pathlib import Path

from src.config import BUNDLED_DATA_DIR, apply_output_directory, detect_runtime, find_data_directory, get_config


def test_apply_output_directory_builds_expected_layout(tmp_path):
    config = apply_output_directory({}, str(tmp_path / "outputs" / "builds" / "run1"))
    for key in ("artifact_dir", "report_dir", "figure_dir", "log_dir"):
        assert Path(config[key]).exists()
    assert config["output_files"]["ab_test_results"].endswith("artifacts/ab_test_results.csv")
    assert config["output_files"]["report_html"].endswith("reports/report.html")


def test_detect_runtime_is_local_outside_colab(monkeypatch):
    monkeypatch.delenv("OBD_ENV", raising=False)
    assert detect_runtime() == "local"


def test_data_directory_defaults_to_bundled_sample(monkeypatch):
    monkeypatch.delenv("OBD_DATA_DIR", raising=False)
    assert find_data_directory() == BUNDLED_DATA_DIR
    assert get_config("local")["using_bundled_sample"] is True


def test_data_directory_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("OBD_DATA_DIR", str(tmp_path))
    assert find_data_directory() == tmp_path.resolve()

"""Tests for path resolution (terravision.config)."""

from pathlib import Path

import pytest

from terravision.config import ENV_PREFIX, find_repo_root, get_paths

DIR_VARS = ["ROOT", "DATA_DIR", "RUNS_DIR", "MODELS_DIR", "RESULTS_DIR", "OUTPUTS_DIR"]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Start every test with no TERRAVISION_* overrides."""
    for name in DIR_VARS:
        monkeypatch.delenv(ENV_PREFIX + name, raising=False)


def test_repo_root_found_from_installed_package():
    root = find_repo_root()
    assert (root / "pyproject.toml").is_file()
    assert (root / "src" / "terravision").is_dir()


def test_defaults_live_inside_repo():
    paths = get_paths()
    assert paths.data == paths.root / "data"
    assert paths.runs == paths.root / "runs"
    assert paths.models == paths.root / "models"
    assert paths.results == paths.root / "results"
    assert paths.figures == paths.root / "outputs" / "figures"
    assert paths.predictions == paths.root / "outputs" / "predictions"


def test_absolute_override(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_PREFIX + "RUNS_DIR", str(tmp_path / "drive_runs"))
    assert get_paths().runs == (tmp_path / "drive_runs").resolve()


def test_relative_override_resolves_against_root(monkeypatch):
    monkeypatch.setenv(ENV_PREFIX + "DATA_DIR", "data_sample")
    paths = get_paths()
    assert paths.data == paths.root / "data_sample"


def test_root_override(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_PREFIX + "ROOT", str(tmp_path))
    paths = get_paths()
    assert paths.root == tmp_path.resolve()
    assert paths.models == tmp_path.resolve() / "models"


def test_root_not_found_raises(tmp_path):
    with pytest.raises(RuntimeError):
        find_repo_root(start=Path(tmp_path))


def test_ensure_creates_folders(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_PREFIX + "ROOT", str(tmp_path))
    paths = get_paths().ensure()
    for folder in (paths.data, paths.runs, paths.models, paths.results,
                   paths.figures, paths.predictions):
        assert folder.is_dir()

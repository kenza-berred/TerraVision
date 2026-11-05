"""Project paths, resolved the same way locally and on Colab.

Every folder has a default inside the repository and can be overridden with an
environment variable. Locally the defaults are used as-is; on Colab the setup
cell points the heavy folders to fast local disk or to Google Drive:

    TERRAVISION_ROOT          repository root (normally auto-detected)
    TERRAVISION_DATA_DIR      extracted datasets         default: <root>/data
    TERRAVISION_RUNS_DIR      training runs (checkpoints, histories)
                                                         default: <root>/runs
    TERRAVISION_MODELS_DIR    checkpoints the backend serves
                                                         default: <root>/models
    TERRAVISION_RESULTS_DIR   metric / history JSONs     default: <root>/results
    TERRAVISION_OUTPUTS_DIR   figures and predictions    default: <root>/outputs

Relative override values are resolved against the repository root, so
``TERRAVISION_DATA_DIR=data_sample`` means ``<root>/data_sample``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ENV_PREFIX = "TERRAVISION_"

# Folder name -> default location relative to the repository root.
_DEFAULT_DIRS = {
    "data": "data",
    "runs": "runs",
    "models": "models",
    "results": "results",
    "outputs": "outputs",
}


def find_repo_root(start: Path | None = None) -> Path:
    """Return the repository root.

    Uses ``TERRAVISION_ROOT`` if set, otherwise walks up from ``start``
    (default: this file) to the first folder containing ``pyproject.toml``.
    This works for an editable install (``pip install -e .``), which is how
    the package is installed both locally and on Colab.
    """
    override = os.environ.get(f"{ENV_PREFIX}ROOT")
    if override:
        return Path(override).expanduser().resolve()

    here = (start or Path(__file__)).resolve()
    for folder in [here, *here.parents]:
        if (folder / "pyproject.toml").is_file():
            return folder
    raise RuntimeError(
        "Could not find the TerraVision repository root (no pyproject.toml "
        f"above {here}). Set {ENV_PREFIX}ROOT or install with `pip install -e .`."
    )


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    data: Path
    runs: Path
    models: Path
    results: Path
    outputs: Path

    @property
    def figures(self) -> Path:
        return self.outputs / "figures"

    @property
    def predictions(self) -> Path:
        return self.outputs / "predictions"

    def ensure(self) -> "ProjectPaths":
        """Create every folder that doesn't exist yet. Returns self."""
        for path in (self.data, self.runs, self.models, self.results,
                     self.figures, self.predictions):
            path.mkdir(parents=True, exist_ok=True)
        return self


def _resolve(root: Path, name: str, default: str) -> Path:
    value = os.environ.get(f"{ENV_PREFIX}{name.upper()}_DIR")
    path = Path(value).expanduser() if value else Path(default)
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def get_paths() -> ProjectPaths:
    """Read the environment and return the current project paths.

    Not cached on purpose: a notebook can change an environment variable and
    call this again. It is cheap (no filesystem writes).
    """
    root = find_repo_root()
    dirs = {name: _resolve(root, name, default) for name, default in _DEFAULT_DIRS.items()}
    return ProjectPaths(root=root, **dirs)

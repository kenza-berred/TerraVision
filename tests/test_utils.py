"""Tests for reproducibility helpers (terravision.utils)."""

import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from terravision.utils import environment_info, get_device, set_seed  # noqa: E402


def _draw():
    return random.random(), np.random.rand(), torch.rand(1).item()


def test_set_seed_makes_draws_repeatable():
    set_seed(42)
    first = _draw()
    set_seed(42)
    assert _draw() == first


def test_cpu_can_be_forced():
    assert get_device(prefer="cpu").type == "cpu"


def test_environment_info_has_core_keys():
    info = environment_info()
    assert {"python", "torch", "cuda_available"} <= info.keys()

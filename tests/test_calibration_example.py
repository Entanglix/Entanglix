"""The calibration example's datasets regenerate, and its spec fits training data only."""

import hashlib
import importlib.util
from pathlib import Path

import numpy as np

from entanglix.corrector.calibration import holdout_form
from entanglix.measurements import load_dataset
from entanglix.protocols.calibrate import load_calibration

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/transmon-calibration"
spec = importlib.util.spec_from_file_location("generate", EXAMPLE / "generate.py")
generate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generate)


def test_the_generator_reproduces_the_tracked_datasets(tmp_path):
    generate.generate(tmp_path)
    for name in ("training", "holdout"):
        tracked = load_dataset(EXAMPLE / f"{name}.toml")
        regenerated = load_dataset(tmp_path / f"{name}.toml")
        assert (tmp_path / f"{name}.toml").read_bytes() == (EXAMPLE / f"{name}.toml").read_bytes()
        assert tracked.arrays.keys() == regenerated.arrays.keys()
        for role, values in tracked.arrays.items():
            np.testing.assert_allclose(regenerated.arrays[role], values, rtol=0, atol=1e-9)


def test_the_spec_fits_training_data_only_and_the_holdout_is_checkable():
    loaded = load_calibration(EXAMPLE / "calibrate.toml")
    assert loaded.spec.datasets == ["training.toml"]
    withheld = {
        hashlib.sha256((EXAMPLE / name).read_bytes()).hexdigest()
        for name in ("holdout.toml", "holdout.npz")
    }
    assert not withheld & set(loaded.input_digests.values())
    assert not any("holdout" in name for name in loaded.input_digests)
    holdout = load_dataset(EXAMPLE / "holdout.toml")
    training = load_dataset(EXAMPLE / "training.toml")
    assert holdout.arrays["map.bias"].min() > training.arrays["map.bias"].max()
    assert holdout_form(loaded, holdout) == ("map", None)

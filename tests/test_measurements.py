import hashlib
import io
import json
import tomllib
import zipfile
from datetime import UTC, datetime

import numpy as np
import pytest
import tomli_w

from entanglix.cli import main
from entanglix.measurements import DatasetError, dataset_report, load_dataset, split_dataset

BIAS = np.linspace(-0.5, 0.5, 11)
FREQUENCY = np.linspace(4.0, 6.0, 21)


def _document(**levels):
    return {
        "format_version": 1,
        "id": "two-tone",
        "kind": "spectroscopy",
        "array_file": "data.npz",
        "conditions": {
            "drive_power": {"value": -30, "unit": "dBm"},
            "averages": 1000,
            "readout": "dispersive, 7.1 GHz",
        },
        "provenance": {
            "device": "chip-A",
            "measured_at": datetime(2026, 10, 1, 12, tzinfo=UTC),
            "converter": {"name": "synthetic", "version": "1"},
        },
        **levels,
    }


MAP = {
    "signal": {"key": "signal", "unit": "V"},
    "frequency": {"key": "frequency", "unit": "GHz"},
    "bias": {"key": "bias", "unit": "mA"},
}
POINTS = {
    "centre": {"key": "centre", "unit": "GHz"},
    "centre_uncertainty": {"key": "centre_sigma", "unit": "GHz"},
    "bias": {"key": "point_bias", "unit": "mA"},
    "linewidth": {"key": "hwhm", "unit": "GHz"},
    "amplitude": {"key": "amplitude", "unit": "V"},
    "assignment": {"key": "label"},
}


def _arrays():
    rng = np.random.default_rng(1)
    signal = rng.normal(size=(BIAS.size, FREQUENCY.size)) + 1j * rng.normal(
        size=(BIAS.size, FREQUENCY.size)
    )
    return {
        "signal": signal,
        "frequency": FREQUENCY,
        "bias": BIAS,
        "centre": np.linspace(4.5, 5.5, BIAS.size),
        "centre_sigma": np.full(BIAS.size, 1e-3),
        "point_bias": BIAS[::-1].copy(),
        "hwhm": np.full(BIAS.size, 2e-3),
        "amplitude": np.ones(BIAS.size),
        "label": np.array(["0-1"] * 5 + [""] * 6),
    }


def _write(directory, document, arrays, *, name="dataset.toml"):
    directory.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    (directory / document["array_file"]).write_bytes(buffer.getvalue())
    path = directory / name
    path.write_bytes(tomli_w.dumps(document).encode())
    return path


def _full(tmp_path):
    return _write(tmp_path / "full", _document(map=MAP, points=POINTS), _arrays())


def test_map_points_and_both_validate(tmp_path):
    arrays = _arrays()
    points = {key: arrays[key] for key in ("centre", "centre_sigma", "point_bias", "hwhm")}
    points["amplitude"], points["label"] = arrays["amplitude"], arrays["label"]
    grid = {key: arrays[key] for key in ("signal", "frequency", "bias")}
    for name, levels, values in [
        ("map", {"map": MAP}, grid),
        ("points", {"points": POINTS}, points),
        ("both", {"map": MAP, "points": POINTS}, arrays),
    ]:
        dataset = load_dataset(_write(tmp_path / name, _document(**levels), values))
        report = dataset_report(dataset)
        assert report["bias_unit"] == "mA"
        assert ("map" in report) == ("map" in levels)
        assert ("points" in report) == ("points" in levels)
    assert report["map"]["shape"] == [BIAS.size, FREQUENCY.size]
    assert report["map"]["signal"] == "complex"
    assert report["points"]["labelled"] == 5


def test_minimal_points_and_fixed_bias_map(tmp_path):
    points = {
        "centre": {"key": "centre", "unit": "GHz"},
        "centre_uncertainty": {"key": "centre_sigma", "unit": "GHz"},
    }
    arrays = {"centre": np.array([5.0, 5.2]), "centre_sigma": np.array([1e-3, 2e-3])}
    report = dataset_report(
        load_dataset(_write(tmp_path / "points", _document(points=points), arrays))
    )
    assert report["bias_unit"] is None
    assert report["points"] == {
        "count": 2,
        "labelled": 0,
        "fields": ["centre", "centre_uncertainty"],
    }
    fixed = {key: value for key, value in MAP.items() if key != "bias"}
    arrays = {"signal": np.abs(_arrays()["signal"][0]), "frequency": FREQUENCY}
    report = dataset_report(load_dataset(_write(tmp_path / "fixed", _document(map=fixed), arrays)))
    assert report["map"] == {
        "shape": [FREQUENCY.size],
        "signal": "real",
        "frequency_range_ghz": [4.0, 6.0],
    }


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda d, a: a.update(signal=np.array([object()] * 3)), "Object arrays"),
        (lambda d, a: a.update(signal=a["signal"][:, :-1]), "does not match its axes"),
        (lambda d, a: a.update(frequency=FREQUENCY[::-1].copy()), "strictly increasing"),
        (lambda d, a: a.update(hwhm=np.zeros(BIAS.size)), "points.linewidth: values must be"),
        (lambda d, a: a.update(centre_sigma=np.ones(3)), "matching points.centre"),
        (lambda d, a: a.update(label=np.arange(BIAS.size)), "text labels"),
        (lambda d, a: a.update(extra=np.ones(2)), "unreferenced NPZ arrays ['extra']"),
        (lambda d, a: a.pop("hwhm"), "missing NPZ keys ['hwhm']"),
        (lambda d, a: d["map"]["bias"].pop("unit"), "map.bias.unit: Field required"),
        (lambda d, a: d["map"]["frequency"].update(unit="MHz"), "map.frequency.unit"),
        (lambda d, a: d["points"]["bias"].update(unit="V"), "share one bias axis and unit"),
        (lambda d, a: d["points"].pop("bias"), "share one bias axis and unit"),
        (lambda d, a: d["points"]["linewidth"].update(key="centre"), "its own NPZ key"),
        (lambda d, a: d["provenance"].update(measured_at=datetime(2026, 10, 1)), "UTC offset"),
        (lambda d, a: d["conditions"].update(power=-30.0), "conditions.power"),
        (lambda d, a: d.pop("map") and d.pop("points"), "a map, points, or both"),
        (lambda d, a: d.update(array_file="data.bin"), "beside the dataset file"),
    ],
)
def test_refusals_carry_diagnostics(tmp_path, change, message):
    document = _document(
        map={key: dict(value) for key, value in MAP.items()},
        points={key: dict(value) for key, value in POINTS.items()},
    )
    arrays = _arrays()
    change(document, arrays)
    path = _write(tmp_path, document, arrays)
    with pytest.raises(DatasetError) as raised:
        load_dataset(path)
    assert message in str(raised.value)


def test_digest_is_of_validated_bytes_and_survives_freezing(tmp_path):
    dataset = load_dataset(_full(tmp_path))
    sidecar, archive = dataset.files["dataset.toml"], dataset.files["data.npz"]
    expected = hashlib.sha256(
        str(len(sidecar)).encode() + b"\0" + sidecar + str(len(archive)).encode() + b"\0" + archive
    ).hexdigest()
    assert dataset.sha256 == expected
    frozen = tmp_path / "run" / "spec"
    frozen.mkdir(parents=True)
    for name, content in dataset.files.items():
        (frozen / name).write_bytes(content)
    assert load_dataset(frozen / "dataset.toml").sha256 == dataset.sha256
    (frozen / "data.npz").write_bytes(archive + b"\0")
    assert load_dataset(frozen / "dataset.toml").sha256 != dataset.sha256


def test_raw_file_digest_is_checked_only_when_supplied(tmp_path):
    raw = tmp_path / "raw.hdf5"
    raw.write_bytes(b"laboratory bytes")
    document = _document(map=MAP, points=POINTS)
    document["provenance"]["raw_file_sha256"] = hashlib.sha256(b"laboratory bytes").hexdigest()
    path = _write(tmp_path / "declared", document, _arrays())
    assert dataset_report(load_dataset(path))["raw_file"] == "not supplied"
    report = dataset_report(load_dataset(path, raw_file=raw))
    assert report["raw_file"] == "digest matched"
    assert report["conversion"] == "recorded by its converter; not verified by Entanglix"
    raw.write_bytes(b"other bytes")
    with pytest.raises(DatasetError, match="does not match provenance.raw_file_sha256"):
        load_dataset(path, raw_file=raw)
    with pytest.raises(DatasetError, match="declares no raw_file_sha256"):
        load_dataset(_full(tmp_path), raw_file=raw)


def _parts(result):
    return result["holdout"]["split"], result["remainder"]["split"]


def test_end_range_split_is_disjoint_and_records_its_parent(tmp_path):
    source = _full(tmp_path)
    parent = load_dataset(source)
    result = split_dataset(
        source, 0.15, 0.5, holdout=tmp_path / "h.toml", remainder=tmp_path / "r.toml"
    )
    holdout, remainder = _parts(result)
    assert holdout["parent_sha256"] == remainder["parent_sha256"] == parent.sha256
    assert holdout["columns"] == [[7, 11]] and remainder["columns"] == [[0, 7]]
    # Point biases run in reverse, so the points inside the range come first.
    assert holdout["points"] == [[0, 4]] and remainder["points"] == [[4, 11]]
    held, rest = load_dataset(tmp_path / "h.toml"), load_dataset(tmp_path / "r.toml")
    assert held.spec.id == "two-tone-holdout" and rest.spec.id == "two-tone-remainder"
    np.testing.assert_array_equal(held.arrays["map.signal"], parent.arrays["map.signal"][7:])
    np.testing.assert_array_equal(held.arrays["map.frequency"], FREQUENCY)
    np.testing.assert_array_equal(
        rest.arrays["points.assignment"], parent.arrays["points.assignment"][4:]
    )
    assert held.spec.provenance == parent.spec.provenance


def test_interior_split_leaves_a_two_interval_remainder(tmp_path):
    source = _full(tmp_path)
    result = split_dataset(
        source, -0.15, 0.15, holdout=tmp_path / "h.toml", remainder=tmp_path / "r.toml"
    )
    holdout, remainder = _parts(result)
    assert holdout["columns"] == [[4, 7]]
    assert remainder["columns"] == [[0, 4], [7, 11]]
    assert holdout["points"] == [[4, 7]] and remainder["points"] == [[0, 4], [7, 11]]
    parent = load_dataset(source).arrays["map.bias"]
    rest = load_dataset(tmp_path / "r.toml").arrays["map.bias"]
    np.testing.assert_array_equal(rest, np.concatenate([parent[:4], parent[7:]]))
    columns = [
        index
        for start, stop in holdout["columns"] + remainder["columns"]
        for index in range(start, stop)
    ]
    assert sorted(columns) == list(range(BIAS.size))


def test_split_refusals(tmp_path):
    source = _full(tmp_path)
    outputs = {"holdout": tmp_path / "h.toml", "remainder": tmp_path / "r.toml"}
    with pytest.raises(DatasetError, match="remainder would be empty"):
        split_dataset(source, -1, 1, **outputs)
    with pytest.raises(DatasetError, match="finite lower < upper"):
        split_dataset(source, 0.2, 0.1, **outputs)
    split_dataset(source, 0.15, 0.5, **outputs)
    with pytest.raises(DatasetError, match="already exists"):
        split_dataset(source, 0.15, 0.5, **outputs)
    with pytest.raises(DatasetError, match="already a split part"):
        split_dataset(
            outputs["remainder"],
            -0.5,
            -0.2,
            holdout=tmp_path / "a.toml",
            remainder=tmp_path / "b.toml",
        )
    fixed = {key: value for key, value in MAP.items() if key != "bias"}
    arrays = {"signal": _arrays()["signal"][0], "frequency": FREQUENCY}
    unbiased = _write(tmp_path / "fixed", _document(map=fixed), arrays)
    with pytest.raises(DatasetError, match="needs a bias axis"):
        split_dataset(unbiased, 0, 1, holdout=tmp_path / "c.toml", remainder=tmp_path / "d.toml")


def test_split_record_must_match_the_kept_entries(tmp_path):
    source = _full(tmp_path)
    split_dataset(source, 0.15, 0.5, holdout=tmp_path / "h.toml", remainder=tmp_path / "r.toml")
    document = tomllib.loads((tmp_path / "h.toml").read_text())
    document["split"]["columns"] = [[6, 11]]
    (tmp_path / "h.toml").write_text(tomli_w.dumps(document))
    with pytest.raises(DatasetError, match="split.columns: records 5 indices for 4 kept"):
        load_dataset(tmp_path / "h.toml")


def test_cli_validates_and_splits(tmp_path, capsys):
    source = _full(tmp_path)
    assert main(["validate-dataset", str(source)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["conversion"] == "recorded by its converter; not verified by Entanglix"
    arguments = ["split-dataset", str(source), "--lower", "-0.15", "--upper", "0.15"]
    outputs = ["--holdout", str(tmp_path / "h.toml"), "--remainder", str(tmp_path / "r.toml")]
    assert main([*arguments, *outputs]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["remainder"]["split"]["columns"] == [[0, 4], [7, 11]]
    assert main([*arguments, *outputs]) == 64
    assert "already exists" in capsys.readouterr().err
    (tmp_path / "full" / "data.npz").write_bytes(b"not an archive")
    assert main(["validate-dataset", str(source)]) == 64
    assert "entanglix: invalid dataset: array_file" in capsys.readouterr().err


def test_sparse_points_leave_a_map_only_part(tmp_path):
    arrays = _arrays()
    for key in ("centre", "centre_sigma", "point_bias", "hwhm", "amplitude", "label"):
        arrays[key] = arrays[key][:3]
    arrays["point_bias"] = np.array([-0.5, -0.4, -0.3])
    source = _write(tmp_path / "sparse", _document(map=MAP, points=POINTS), arrays)
    result = split_dataset(
        source, 0.15, 0.5, holdout=tmp_path / "h.toml", remainder=tmp_path / "r.toml"
    )
    assert "points" not in result["holdout"] and result["holdout"]["split"]["points"] == []
    assert result["holdout"]["split"]["columns"] == [[7, 11]]
    assert result["remainder"]["points"]["count"] == 3
    held = load_dataset(tmp_path / "h.toml")
    assert held.spec.points is None and set(held.arrays) == {
        "map.signal",
        "map.frequency",
        "map.bias",
    }


def test_failed_split_leaves_no_outputs(tmp_path):
    source = _full(tmp_path)
    holdout = tmp_path / "h.toml"
    with pytest.raises(DatasetError, match="does not exist"):
        split_dataset(source, 0.15, 0.5, holdout=holdout, remainder=tmp_path / "missing/r.toml")
    assert not holdout.exists() and not holdout.with_suffix(".npz").exists()
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    try:
        with pytest.raises(PermissionError):
            split_dataset(source, 0.15, 0.5, holdout=holdout, remainder=locked / "r.toml")
    finally:
        locked.chmod(0o700)
    assert not holdout.exists() and not holdout.with_suffix(".npz").exists()
    split_dataset(source, 0.15, 0.5, holdout=holdout, remainder=tmp_path / "r.toml")


def test_cli_refuses_a_malformed_archive_member(tmp_path, capsys):
    source = _full(tmp_path)
    archive = tmp_path / "full" / "data.npz"
    with zipfile.ZipFile(archive) as original:
        members = {name: original.read(name) for name in original.namelist()}
    members["signal.npy"] = b"not an array"
    with zipfile.ZipFile(archive, "w") as rewritten:
        for name, content in members.items():
            rewritten.writestr(name, content)
    assert main(["validate-dataset", str(source)]) == 64
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "invalid dataset: array_file: 'signal' is not a valid NPY array" in captured.err

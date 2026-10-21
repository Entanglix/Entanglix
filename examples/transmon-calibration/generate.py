"""Synthetic two-tone spectroscopy for the transmon calibration example, in the uniform format.

Example code, not a harness feature. The known parameters below generate f01, f12 and the
two-photon f02/2 at each coil bias through the example's own system file, with declared
amplitudes, linewidth, I/Q background and noise from a fixed seed. Each map is written as an
Entanglix dataset, with complex signal, frequency in GHz and coil bias in mA, and the generator
recorded as its converter. Synthetic data has no laboratory raw file, so no raw-file digest is
recorded. The training map and the separate holdout measurement cover different bias ranges.

    uv run --locked --exact python examples/transmon-calibration/generate.py --output DIR
"""

import argparse
import io
import tomllib
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import tomli_w

from entanglix.models import load_system

EXAMPLE = Path(__file__).resolve().parent
TRUTH = {"EJ1": 12.0, "EJ2": 8.0, "Csh": 70.0}
OFFSET, PERIOD = 0.1, 1.0  # mA: phi = (bias - OFFSET) / PERIOD
# Visible lines as (initial, final, photons) with their amplitudes in V; f01 is the strongest.
VISIBLE = {(0, 1, 1): 1.0, (1, 2, 1): 0.3, (0, 2, 2): 0.2}
HWHM = 0.002  # GHz, Lorentzian
NOISE = 0.02  # V per quadrature
BACKGROUND = (0.5 + 0.3j, 0.02)  # V, and V per GHz
PHASE = 0.7  # rad
SEED = 20261021
# Coil biases in mA, the frequency window in GHz at 1 MHz steps, and the measurement time.
MEASUREMENTS = {
    "training": (np.linspace(-0.25, 0.38, 21), (3.5, 6.5), datetime(2026, 10, 21, 9, tzinfo=UTC)),
    "holdout": (np.linspace(0.40, 0.65, 8), (2.0, 5.0), datetime(2026, 10, 21, 11, tzinfo=UTC)),
}


def lines(bias):
    """Visible line frequencies at each bias, from the system file at the known parameters."""
    path = EXAMPLE / "system.toml"
    document = tomllib.loads(path.read_text())
    for name, value in TRUTH.items():
        document["parameters"][name]["value"] = value
    rows = []
    for value in bias:
        document["parameters"]["phi"]["value"] = float((value - OFFSET) / PERIOD)
        system = load_system(path, content=tomli_w.dumps(document).encode())
        energies = np.linalg.eigvalsh(system.hamiltonian_ghz)
        rows.append([(energies[f] - energies[i]) / k for i, f, k in VISIBLE])
    return np.array(rows)


def measure(rng, bias, window):
    frequency = np.round(np.arange(window[0], window[1], 0.001), 6)
    clean = np.zeros((bias.size, frequency.size))
    for column, centres in enumerate(lines(bias)):
        for centre, amplitude in zip(centres, VISIBLE.values(), strict=True):
            clean[column] += amplitude * HWHM**2 / ((frequency - centre) ** 2 + HWHM**2)
    background = BACKGROUND[0] + BACKGROUND[1] * (frequency - frequency[0])
    noise = NOISE * (rng.standard_normal(clean.shape) + 1j * rng.standard_normal(clean.shape))
    return frequency, np.exp(1j * PHASE) * clean + background + noise


def write(directory, name, bias, frequency, signal, measured_at):
    buffer = io.BytesIO()
    np.savez(buffer, signal=signal, frequency=frequency, bias=bias)
    (directory / f"{name}.npz").write_bytes(buffer.getvalue())
    document = {
        "format_version": 1,
        "id": f"transmon-calibration-{name}",
        "kind": "spectroscopy",
        "array_file": f"{name}.npz",
        "conditions": {
            "measurement": "synthetic two-tone",
            "noise_per_quadrature": {"value": NOISE, "unit": "V"},
        },
        "provenance": {
            "device": "synthetic squid_transmon",
            "measured_at": measured_at,
            "converter": {"name": "examples/transmon-calibration/generate.py", "version": "1"},
        },
        "map": {
            "signal": {"key": "signal", "unit": "V"},
            "frequency": {"key": "frequency", "unit": "GHz"},
            "bias": {"key": "bias", "unit": "mA"},
        },
    }
    (directory / f"{name}.toml").write_text(tomli_w.dumps(document))


def generate(directory):
    rng = np.random.default_rng(SEED)
    for name, (bias, window, measured_at) in MEASUREMENTS.items():
        write(directory, name, bias, *measure(rng, bias, window), measured_at)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="existing directory")
    generate(parser.parse_args().output)


if __name__ == "__main__":
    main()

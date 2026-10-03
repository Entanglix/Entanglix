"""Trusted retained-basis observables and endpoint metric formulas."""

from typing import Annotated, Literal

import numpy as np
from pydantic import Field

from entanglix.models import StrictModel


class Population(StrictModel):
    kind: Literal["population"]
    subsystem: str
    level: int = Field(strict=True, ge=0)


class Coherence(StrictModel):
    kind: Literal["coherence"]
    subsystem: str


class EndpointMetric(StrictModel):
    kind: Literal["endpoint_real", "endpoint_coherence"]
    observable: str
    segment_tolerance: float | None = Field(default=None, gt=0, allow_inf_nan=False)


class DispersiveShift(StrictModel):
    kind: Literal["dispersive_shift"]
    qubit: str
    resonator: str
    segment_tolerance: float | None = Field(default=None, gt=0, allow_inf_nan=False)


Metric = Annotated[EndpointMetric | DispersiveShift, Field(discriminator="kind")]
Observable = Annotated[Population | Coherence, Field(discriminator="kind")]


def spectral_arrays(system, levels, definitions):
    """Lowest energies plus four uniquely assigned dressed energies for each shift."""
    if not definitions:
        return {"energies_ghz": system.qutip_hamiltonian().eigenenergies()[:levels]}
    energies, vectors = np.linalg.eigh(system.hamiltonian_ghz)
    arrays = {"energies_ghz": energies[:levels]}
    for name, definition in definitions.items():
        if len(system.dimensions) != 2 or {definition.qubit, definition.resonator} != set(
            system.subsystem_ids
        ):
            raise ValueError("dispersive_shift requires the two named qubit/resonator subsystems")
        if min(system.dimensions) < 2:
            raise ValueError("dispersive_shift requires two levels in each subsystem")
        labels = [(0, 0), (0, 1), (1, 0), (1, 1)]
        coordinates = [
            label if system.subsystem_ids[0] == definition.qubit else label[::-1]
            for label in labels
        ]
        bare = [np.ravel_multi_index(label, system.dimensions) for label in coordinates]
        weights = abs(vectors[bare, :levels]) ** 2
        indices = weights.argmax(axis=1)
        overlaps = weights[np.arange(4), indices]
        if len(set(indices)) != 4 or overlaps.min() < 0.9:
            raise ValueError(
                f"dispersive_shift {name}: ambiguous dressed labels within {levels} levels; indices={indices.tolist()}, overlaps={overlaps.tolist()} (minimum 0.9)"
            )
        arrays[f"dressed.{name}"] = energies[indices]
        arrays[f"dressed_indices.{name}"] = indices
        arrays[f"dressed_overlaps.{name}"] = overlaps
    return arrays


def observable_operator(system, definition):
    if isinstance(definition, str):
        if definition not in system.operators:
            raise ValueError(f"unknown observable operator: {definition}")
        return system.operators[definition]
    if definition.subsystem not in system.subsystem_ids:
        raise ValueError(f"unknown population subsystem: {definition.subsystem}")
    axis = system.subsystem_ids.index(definition.subsystem)
    level = definition.level if definition.kind == "population" else 1
    if level >= system.dimensions[axis]:
        raise ValueError("population level exceeds the subsystem's retained basis")
    result = np.ones((1, 1))
    for index, size in enumerate(system.dimensions):
        local = np.eye(size)
        if index == axis:
            local = np.zeros((size, size))
            if definition.kind == "population":
                local[level, level] = 1
            else:
                local[1, 0] = 1
        result = np.kron(result, local)
    return result


def metric_values(definitions, arrays):
    values = {}
    for name, definition in definitions.items():
        if definition.kind == "dispersive_shift":
            e00, e01, e10, e11 = arrays[f"dressed.{name}"]
            value = (e11 - e10 - e01 + e00) / 2
        else:
            endpoint = arrays[f"observable.{definition.observable}"][-1]
            value = 2 * abs(endpoint) if definition.kind == "endpoint_coherence" else endpoint.real
        if not np.isfinite(value):
            raise ValueError(f"metric {name}: nonfinite endpoint")
        values[name] = float(value)
    return values


def metric_tolerance(definition, policy):
    """Absolute local endpoint tolerance in the metric's units."""
    return definition.segment_tolerance or policy.thresholds["refinement"]["observable_atol"]

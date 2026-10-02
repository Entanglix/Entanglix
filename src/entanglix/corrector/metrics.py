"""Trusted retained-basis observables and endpoint metric formulas."""

from typing import Literal

import numpy as np
from pydantic import Field

from entanglix.models import StrictModel


class Population(StrictModel):
    kind: Literal["population"]
    subsystem: str
    level: int = Field(strict=True, ge=0)


class EndpointMetric(StrictModel):
    kind: Literal["endpoint_real"]
    observable: str
    rwa_bound: float = Field(default=0, ge=0, allow_inf_nan=False)


def observable_operator(system, definition):
    if isinstance(definition, str):
        if definition not in system.operators:
            raise ValueError(f"unknown observable operator: {definition}")
        return system.operators[definition]
    if definition.subsystem not in system.subsystem_ids:
        raise ValueError(f"unknown population subsystem: {definition.subsystem}")
    axis = system.subsystem_ids.index(definition.subsystem)
    if definition.level >= system.dimensions[axis]:
        raise ValueError("population level exceeds the subsystem's retained basis")
    result = np.ones((1, 1))
    for index, size in enumerate(system.dimensions):
        local = np.eye(size)
        if index == axis:
            local = np.zeros((size, size))
            local[definition.level, definition.level] = 1
        result = np.kron(result, local)
    return result


def metric_values(definitions, arrays):
    values = {}
    for name, definition in definitions.items():
        endpoint = arrays[f"observable.{definition.observable}"][-1]
        value = endpoint.real
        if not np.isfinite(value):
            raise ValueError(f"metric {name}: nonfinite endpoint")
        values[name] = float(value)
    return values

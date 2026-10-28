"""Shared engine interface and QuTiP CPU entry points; GPU adapters stay separate."""

from entanglix.engines.qutip_cpu import QuTiPIntegrator, evolve_cpu
from entanglix.evolution import (
    Carrier,
    EngineCapabilities,
    EvolutionInput,
    EvolutionInputError,
    EvolutionResult,
    Integrator,
    RotatingFrame,
)

__all__ = [
    "Carrier",
    "EngineCapabilities",
    "EvolutionInput",
    "EvolutionInputError",
    "EvolutionResult",
    "Integrator",
    "QuTiPIntegrator",
    "RotatingFrame",
    "evolve_cpu",
]

"""Shared engine interface and QuTiP CPU entry points; GPU adapters stay separate."""

from entanglix.engines.qutip_cpu import QuTiPIntegrator, evolve_cpu
from entanglix.evolution import (
    EngineCapabilities,
    EvolutionInput,
    EvolutionInputError,
    EvolutionResult,
    Integrator,
)

__all__ = [
    "EngineCapabilities",
    "EvolutionInput",
    "EvolutionInputError",
    "EvolutionResult",
    "Integrator",
    "QuTiPIntegrator",
    "evolve_cpu",
]

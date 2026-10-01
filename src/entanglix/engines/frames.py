"""Numeric rotating-wave projection, retaining local tensor factors for all engines."""

import math
from dataclasses import dataclass
from itertools import product

import numpy as np

from entanglix.evolution import EvolutionInputError, NormalizedInput
from entanglix.models import ConstructedSystem, TensorTerm

FREQUENCY_ATOL_GHZ = 1e-10


def matrix(terms):
    return sum(term.coefficient_ghz * tensor(term.factors) for term in terms)


def tensor(factors):
    value = np.array([[1.0]], dtype=complex)
    for factor in factors:
        value = np.kron(value, factor)
    return value


@dataclass(frozen=True)
class PreparedDynamics:
    hamiltonian: tuple[TensorTerm, ...]
    drives: dict[str, tuple[TensorTerm, ...]]
    collapse: tuple[tuple[TensorTerm, ...], ...]
    frame_levels_ghz: np.ndarray | None
    diagnostics: dict

    def initial_density(self, request):
        if self.frame_levels_ghz is None:
            return request.initial_density
        phase = np.exp(2j * np.pi * request.times_ns[0] * self.frame_levels_ghz)
        return phase[:, None] * request.initial_density * phase.conj()[None, :]

    def lab_states(self, states, times, *, xp=np):
        if self.frame_levels_ghz is None:
            return states
        phase = xp.exp(-2j * xp.pi * xp.asarray(times)[:, None] * xp.asarray(self.frame_levels_ghz))
        return phase[:, :, None] * xp.asarray(states) * phase.conj()[:, None, :]


def _sectors(factor, levels):
    differences = levels[:, None] - levels[None, :]
    remaining = np.abs(factor) > 0
    parts = []
    while np.any(remaining):
        row, column = np.argwhere(remaining)[0]
        shift = differences[row, column]
        mask = np.abs(differences - shift) <= FREQUENCY_ATOL_GHZ
        parts.append((float(shift), np.where(mask, factor, 0)))
        remaining &= ~mask
    return parts


def _split(term, levels):
    for parts in product(*(_sectors(factor, local) for factor, local in zip(term.factors, levels))):
        yield (
            sum(part[0] for part in parts),
            TensorTerm(term.coefficient_ghz, tuple(part[1] for part in parts)),
        )


def prepare_dynamics(system: ConstructedSystem, request: NormalizedInput) -> PreparedDynamics:
    hamiltonian = system.hamiltonian_terms or (TensorTerm(1, (system.hamiltonian_ghz,)),)
    drives = {
        name: (TensorTerm(1, system.drive_factors.get(name, (operator,))),)
        for name, operator in system.drives.items()
    }
    collapse = tuple(
        (TensorTerm(math.sqrt(rate), system.dissipator_factors.get(name, (operator,))),)
        for name, (operator, rate) in system.dissipators.items()
        if rate > 0
    )
    carriers = {
        name: {"frequency_ghz": value.frequency_ghz, "phase_rad": value.phase_rad}
        for name, value in request.carriers.items()
    }
    if request.frame is None:
        return PreparedDynamics(
            hamiltonian, drives, collapse, None, {"frame": {"kind": "lab"}, "carriers": carriers}
        )

    frame = request.frame
    if frame.rwa is not True:
        raise EvolutionInputError("rotating frames require an explicit rwa=True")
    ids = system.subsystem_ids or (system.id,)
    if not frame.frequencies_ghz or set(frame.frequencies_ghz) - set(ids):
        raise EvolutionInputError(f"frame frequencies must name subsystems from {list(ids)}")
    if set(frame.excitation_numbers) - set(ids):
        raise EvolutionInputError("frame excitation numbers name an unknown subsystem")
    bases = system.provenance.get("subsystem_bases", {ids[0]: system.basis})
    levels, numbers = [], {}
    for name, dimension in zip(ids, system.dimensions):
        frequency = frame.frequencies_ghz.get(name, 0.0)
        if not math.isfinite(frequency) or frequency < 0:
            raise EvolutionInputError("frame frequencies must be finite and nonnegative")
        values = frame.excitation_numbers.get(name)
        if values is None:
            if frequency and bases.get(name) != "isolated_eigenstates":
                raise EvolutionInputError(
                    f"explicit-operator subsystem {name!r} requires excitation_numbers"
                )
            values = list(range(dimension))
        array = np.asarray(values)
        if array.shape != (dimension,) or array.dtype.kind not in "iu" or np.any(array < 0):
            raise EvolutionInputError(
                f"{name}: excitation_numbers must be {dimension} nonnegative integers"
            )
        numbers[name] = array.tolist()
        levels.append(frequency * array)
    total_levels = np.zeros(system.dimensions)
    for axis, local in enumerate(levels):
        shape = [1] * len(levels)
        shape[axis] = len(local)
        total_levels += local.reshape(shape)
    if not np.all(np.isfinite(total_levels)):
        raise EvolutionInputError("total frame frequencies must be finite")
    if any(len(term.factors) != len(levels) for term in hamiltonian):
        raise EvolutionInputError("rotating frames require local Hamiltonian factors")
    identities = tuple(np.eye(d) for d in system.dimensions)
    static, removed_shifts = [], set()
    for term in hamiltonian:
        for shift, projected in _split(term, levels):
            if abs(shift) <= FREQUENCY_ATOL_GHZ:
                static.append(projected)
            else:
                removed_shifts.add(shift)
    static_matrix = matrix(static) if static else np.zeros_like(system.hamiltonian_ghz)
    removed_hamiltonian = float(np.linalg.norm(system.hamiltonian_ghz - static_matrix))
    for axis, local in enumerate(levels):
        factors = list(identities)
        factors[axis] = np.diag(local)
        static.append(TensorTerm(-1, tuple(factors)))

    projected_drives, removed_drives, removed_drive_shifts = {}, {}, {}
    for name, terms in drives.items():
        carrier = request.carriers.get(name)
        retained, dropped_terms = [], []
        dropped = set()
        for term in terms:
            for shift, projected in _split(term, levels):
                selected = False
                if carrier is None:
                    if abs(shift) <= FREQUENCY_ATOL_GHZ:
                        retained.append(projected)
                        selected = True
                else:
                    for sign in (-1, 1):
                        if abs(shift - sign * carrier.frequency_ghz) <= FREQUENCY_ATOL_GHZ:
                            retained.append(
                                TensorTerm(
                                    projected.coefficient_ghz
                                    * np.exp(-sign * 1j * carrier.phase_rad)
                                    / 2,
                                    projected.factors,
                                )
                            )
                            selected = True
                if not selected:
                    dropped.add(shift)
                    dropped_terms.append(projected)
        # A zero drive remains a valid control, including a differentiated override.
        if not retained:
            retained = [TensorTerm(0, identities)]
        projected_drives[name] = tuple(retained)
        removed_drives[name] = (
            float(np.linalg.norm(matrix(dropped_terms))) if dropped_terms else 0.0
        )
        removed_drive_shifts[name] = sorted(dropped)

    for terms in collapse:
        shifts = [shift for term in terms for shift, _ in _split(term, levels)]
        if shifts and max(shifts) - min(shifts) > FREQUENCY_ATOL_GHZ:
            raise EvolutionInputError(
                "rotating-frame dissipators must have one excitation-frequency shift"
            )
    diagnostics = {
        "kind": "rotating",
        "rwa": True,
        "frequencies_ghz": dict(frame.frequencies_ghz),
        "excitation_numbers": numbers,
        "frequency_atol_ghz": FREQUENCY_ATOL_GHZ,
        "removed_hamiltonian_frobenius_ghz": removed_hamiltonian,
        "removed_hamiltonian_shifts_ghz": sorted(removed_shifts),
        "removed_drive_frobenius": removed_drives,
        "removed_drive_shifts_ghz": removed_drive_shifts,
        "returned_state_basis": "lab",
    }
    return PreparedDynamics(
        tuple(static),
        projected_drives,
        collapse,
        total_levels.ravel(),
        {"frame": diagnostics, "carriers": carriers},
    )

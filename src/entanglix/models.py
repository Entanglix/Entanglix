"""Authored system specifications and the first CPU model construction path."""

from __future__ import annotations

import hashlib
import io
import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal
from zipfile import BadZipFile

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from scipy.constants import e, h


class SpecError(ValueError):
    """A system cannot be validated or constructed as authored."""


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Units(StrictModel):
    energy: Literal["GHz"]
    time: Literal["ns"]


class Parameter(StrictModel):
    value: float
    unit: Literal["GHz", "ns", "1/ns", "fF", "nH", "Phi0", "dimensionless"]
    source: str | None = None

    @model_validator(mode="after")
    def check_value(self) -> Parameter:
        if not math.isfinite(self.value):
            raise ValueError("parameter value must be finite")
        return self


class Quantity(StrictModel):
    value: float
    unit: Literal["GHz", "ns", "1/ns", "fF", "nH", "Phi0", "dimensionless"]

    @model_validator(mode="after")
    def check_value(self) -> Quantity:
        if not math.isfinite(self.value):
            raise ValueError("quantity value must be finite")
        return self


class ParameterRef(StrictModel):
    parameter: str


Value = ParameterRef | Quantity


class Branch(StrictModel):
    id: str
    kind: Literal["JJ", "C"]
    nodes: list[int] = Field(min_length=2, max_length=2)
    josephson_energy: Value | None = None
    charging_energy: Value | None = None
    capacitance: Value | None = None

    @model_validator(mode="after")
    def check_fields(self) -> Branch:
        if self.nodes[0] == self.nodes[1]:
            raise ValueError("branch endpoints must differ")
        if (self.charging_energy is None) == (self.capacitance is None):
            raise ValueError("branch requires exactly one of charging_energy or capacitance")
        if self.kind == "JJ" and self.josephson_energy is None:
            raise ValueError("JJ branch requires josephson_energy")
        if self.kind == "C" and self.josephson_energy is not None:
            raise ValueError("C branch cannot have josephson_energy")
        return self


class CircuitSpec(StrictModel):
    kind: Literal["circuit"]
    nodes: list[int]
    ground: int
    offset_charges: dict[str, float]
    branches: list[Branch]


class OperatorsSpec(StrictModel):
    kind: Literal["operators"]
    dimensions: list[int]
    basis: str
    array_file: str
    hamiltonian: str
    operators: dict[str, str]


ModelSpec = Annotated[CircuitSpec | OperatorsSpec, Field(discriminator="kind")]


class Truncation(StrictModel):
    cutoffs: dict[str, int]
    retained_levels: int = Field(gt=0)


class Dissipator(StrictModel):
    id: str
    factors: list[str]
    rate: Value


class SystemSpec(StrictModel):
    format_version: Literal[1]
    id: str = Field(min_length=1)
    units: Units
    parameters: dict[str, Parameter] = Field(default_factory=dict)
    model: ModelSpec
    truncation: Truncation | None = None
    dissipators: list[Dissipator] = Field(default_factory=list)


@dataclass(frozen=True)
class ConstructedSystem:
    id: str
    hamiltonian_ghz: np.ndarray
    operators: dict[str, np.ndarray]
    dissipators: dict[str, tuple[np.ndarray, float]]
    dimensions: tuple[int, ...]
    basis: str
    provenance: dict[str, object]

    def qutip_hamiltonian(self):
        import qutip as qt

        return qt.Qobj(self.hamiltonian_ghz, dims=[list(self.dimensions), list(self.dimensions)])


def _resolved(value: Value, parameters: dict[str, Parameter], unit: str, label: str) -> float:
    if isinstance(value, ParameterRef):
        if value.parameter not in parameters:
            raise SpecError(f"{label}: unknown parameter {value.parameter!r}")
        quantity = parameters[value.parameter]
    else:
        quantity = value
    if quantity.unit != unit:
        raise SpecError(f"{label}: expected {unit}, received {quantity.unit}")
    return quantity.value


def _positive(value: float, label: str) -> float:
    if not math.isfinite(value) or value <= 0:
        raise SpecError(f"{label}: expected a positive finite value")
    return value


def _charging_energy(branch: Branch, parameters: dict[str, Parameter]) -> float:
    if branch.charging_energy is not None:
        return _positive(_resolved(branch.charging_energy, parameters, "GHz", branch.id), branch.id)
    capacitance_ff = _positive(
        _resolved(branch.capacitance, parameters, "fF", branch.id), branch.id
    )
    return e**2 / (2 * h * capacitance_ff * 1e-15 * 1e9)


def _array(value: np.ndarray, dimension: int, label: str) -> np.ndarray:
    if value.shape != (dimension, dimension):
        raise SpecError(f"{label}: expected matrix shape ({dimension}, {dimension})")
    if value.dtype.kind not in "iufc" or not np.all(np.isfinite(value)):
        raise SpecError(f"{label}: expected a finite numeric matrix")
    return np.asarray(value, dtype=np.complex128)


def _construct_operators(spec: SystemSpec, source: Path):
    model = spec.model
    if spec.truncation is not None:
        raise SpecError("operators model has a declared basis; omit truncation")
    if len(model.dimensions) != 1 or model.dimensions[0] < 2:
        raise SpecError("operators model requires one dimension of at least 2")
    if not model.basis.strip():
        raise SpecError("model.basis must name the declared array basis")
    asset_name = Path(model.array_file)
    if asset_name.name != model.array_file or asset_name.suffix != ".npz":
        raise SpecError("model.array_file must name an NPZ file beside the system spec")
    asset = (source.parent / asset_name).resolve()
    if asset.parent != source.parent:
        raise SpecError(
            f"model.array_file: {model.array_file!r} resolves outside the spec's directory"
        )
    if not asset.is_file():
        raise SpecError(f"model.array_file: missing NPZ file beside the spec: {model.array_file!r}")
    content = asset.read_bytes()
    dimension = model.dimensions[0]
    try:
        with np.load(io.BytesIO(content), allow_pickle=False) as archive:
            if model.hamiltonian not in archive:
                raise SpecError(f"model.hamiltonian: missing NPZ key {model.hamiltonian!r}")
            hamiltonian = _array(archive[model.hamiltonian], dimension, "model.hamiltonian")
            operators = {}
            for name, key in model.operators.items():
                if key not in archive:
                    raise SpecError(f"model.operators.{name}: missing NPZ key {key!r}")
                operators[name] = _array(archive[key], dimension, f"model.operators.{name}")
    except (OSError, ValueError, TypeError, BadZipFile) as exc:
        raise SpecError(f"model.array_file: cannot read numeric NPZ: {exc}") from exc
    return (
        hamiltonian,
        operators,
        (dimension,),
        model.basis,
        {"asset_sha256": hashlib.sha256(content).hexdigest()},
    )


def _construct_circuit(spec: SystemSpec):
    import scqubits as scq

    model = spec.model
    if model.nodes != [0, 1] or model.ground != 0:
        raise SpecError("the current circuit path supports nodes [0, 1] with ground 0")
    if len(model.branches) != 2 or sorted(branch.kind for branch in model.branches) != ["C", "JJ"]:
        raise SpecError("the current circuit path requires one JJ and one shunt C branch")
    if len({branch.id for branch in model.branches}) != len(model.branches):
        raise SpecError("model.branches: branch IDs must be unique")
    if any(set(branch.nodes) != {0, 1} for branch in model.branches):
        raise SpecError("circuit branches must connect ground 0 and node 1")
    if set(model.offset_charges) != {"1"} or not math.isfinite(model.offset_charges["1"]):
        raise SpecError("model.offset_charges: node 1 requires a finite explicit value")
    truncation = spec.truncation
    if truncation is None or set(truncation.cutoffs) != {"theta1"}:
        raise SpecError("the current circuit path requires truncation.cutoffs.theta1")
    cutoff = truncation.cutoffs["theta1"]
    if cutoff < 1 or truncation.retained_levels > 2 * cutoff + 1:
        raise SpecError("truncation: cutoff must be positive and retainable levels <= 2N+1")
    lines = ["branches:"]
    converted = {}
    for branch in model.branches:
        charging_energy = _charging_energy(branch, spec.parameters)
        converted[branch.id] = {"charging_energy_ghz": charging_energy}
        if branch.kind == "JJ":
            ej = _positive(
                _resolved(branch.josephson_energy, spec.parameters, "GHz", branch.id), branch.id
            )
            converted[branch.id]["josephson_energy_ghz"] = ej
            lines.append(f"- [JJ, 0, 1, {ej:.17g}, {charging_energy:.17g}]")
        else:
            lines.append(f"- [C, 0, 1, {charging_energy:.17g}]")
    try:
        circuit = scq.Circuit(
            "\n".join(lines) + "\n", from_file=False, truncated_dim=truncation.retained_levels
        )
        circuit.cutoff_n_1 = cutoff
        offset = model.offset_charges["1"]
        circuit.ng1 = -offset  # scqubits Circuit uses n + ng.
        eigenvalues, eigenvectors = circuit.eigensys(evals_count=truncation.retained_levels)
        projection = eigenvectors.conj().T
        operators = {
            "charge": projection @ circuit.n1_operator() @ eigenvectors
            - offset * np.eye(truncation.retained_levels),
            "cos_phase": projection @ circuit.cosθ1_operator() @ eigenvectors,
            "sin_phase": projection @ circuit.sinθ1_operator() @ eigenvectors,
        }
    except (ValueError, RuntimeError, KeyError, AttributeError, np.linalg.LinAlgError) as exc:
        raise SpecError(f"circuit construction failed: {exc}") from exc
    provenance = {
        "scqubits_version": scq.__version__,
        "converted_branches": converted,
        "offset_node_1": offset,
        "charge_cutoff": cutoff,
    }
    return (
        np.diag(eigenvalues),
        operators,
        (truncation.retained_levels,),
        "isolated_eigenstates",
        provenance,
    )


def load_system(path: str | Path) -> ConstructedSystem:
    """Validate a v1 system file and construct the supported CPU model."""
    source = Path(path).resolve()
    try:
        content = source.read_bytes()
        spec = SystemSpec.model_validate(tomllib.loads(content.decode("utf-8")))
    except OSError as exc:
        raise SpecError(f"cannot read system file {source}: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise SpecError(f"invalid TOML: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise SpecError(f"system file must be UTF-8: {exc}") from exc
    except ValidationError as exc:
        issues = "; ".join(
            f"{'.'.join(map(str, error['loc']))}: {error['msg']}" for error in exc.errors()
        )
        raise SpecError(issues) from exc
    if isinstance(spec.model, OperatorsSpec):
        hamiltonian, operators, dimensions, basis, provenance = _construct_operators(spec, source)
    else:
        hamiltonian, operators, dimensions, basis, provenance = _construct_circuit(spec)
    if not np.allclose(hamiltonian, hamiltonian.conj().T, rtol=0, atol=1e-10):
        raise SpecError("model.hamiltonian: matrix must be Hermitian")
    dissipators = {}
    for dissipator in spec.dissipators:
        if dissipator.id in dissipators:
            raise SpecError(f"duplicate dissipator ID {dissipator.id!r}")
        if len(dissipator.factors) != 1 or dissipator.factors[0] not in operators:
            raise SpecError(f"dissipators.{dissipator.id}: expected one of {sorted(operators)}")
        rate = _resolved(dissipator.rate, spec.parameters, "1/ns", dissipator.id)
        if rate < 0:
            raise SpecError(f"dissipators.{dissipator.id}: rate must be nonnegative")
        dissipators[dissipator.id] = (operators[dissipator.factors[0]], rate)
    provenance["parameters"] = {
        name: parameter.model_dump(exclude_none=True) for name, parameter in spec.parameters.items()
    }
    provenance["system_sha256"] = hashlib.sha256(content).hexdigest()
    return ConstructedSystem(
        spec.id,
        np.asarray(hamiltonian, dtype=np.complex128),
        operators,
        dissipators,
        dimensions,
        basis,
        provenance,
    )

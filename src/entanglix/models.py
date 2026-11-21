"""Authored system specifications and the first CPU model construction path."""

from __future__ import annotations

import functools
import hashlib
import io
import itertools
import math
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal
from zipfile import BadZipFile

import numpy as np
import scipy.sparse
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from scipy.constants import e, h


class SpecError(ValueError):
    """A system cannot be validated or constructed as authored."""


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Units(StrictModel):
    energy: Literal["GHz"]
    time: Literal["ns"]


class UniformPrior(StrictModel):
    kind: Literal["uniform"]
    lower: float
    upper: float

    @model_validator(mode="after")
    def check_bounds(self) -> UniformPrior:
        if not (
            math.isfinite(self.lower) and math.isfinite(self.upper) and self.lower < self.upper
        ):
            raise ValueError("uniform prior requires finite lower < upper")
        return self


class Parameter(StrictModel):
    value: float
    unit: Literal["GHz", "ns", "1/ns", "fF", "nH", "Phi0", "dimensionless"]
    source: str | None = None
    prior: UniformPrior | None = None

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
    id: str = Field(min_length=1)
    kind: Literal["JJ", "C", "L"]
    nodes: list[int] = Field(min_length=2, max_length=2)
    josephson_energy: Value | None = None
    charging_energy: Value | None = None
    capacitance: Value | None = None
    inductive_energy: Value | None = None
    inductance: Value | None = None

    @model_validator(mode="after")
    def check_fields(self) -> Branch:
        if self.nodes[0] == self.nodes[1]:
            raise ValueError("branch endpoints must differ")
        if self.kind in ("JJ", "C"):
            if (self.charging_energy is None) == (self.capacitance is None):
                raise ValueError(
                    "JJ/C branch requires exactly one of charging_energy or capacitance"
                )
            if self.kind == "JJ" and self.josephson_energy is None:
                raise ValueError("JJ branch requires josephson_energy")
            if self.kind == "C" and self.josephson_energy is not None:
                raise ValueError("C branch cannot have josephson_energy")
            if self.inductive_energy is not None or self.inductance is not None:
                raise ValueError("JJ/C branch cannot have inductive quantities")
        else:
            if (self.inductive_energy is None) == (self.inductance is None):
                raise ValueError("L branch requires exactly one of inductive_energy or inductance")
            if any(
                x is not None
                for x in (self.charging_energy, self.capacitance, self.josephson_energy)
            ):
                raise ValueError("L branch cannot have charging or Josephson quantities")
        return self


class FluxBias(StrictModel):
    branch: str
    value: float | ParameterRef
    unit: Literal["Phi0"] | None = None

    @model_validator(mode="after")
    def check_value(self) -> FluxBias:
        if isinstance(self.value, ParameterRef):
            if self.unit is not None:
                raise ValueError("a flux parameter reference takes the parameter's unit; omit unit")
        elif self.unit is None:
            raise ValueError('a literal external flux requires unit = "Phi0"')
        elif not math.isfinite(self.value):
            raise ValueError("external flux must be finite")
        return self


class CircuitSpec(StrictModel):
    kind: Literal["circuit"]
    nodes: list[int]
    ground: int
    offset_charges: dict[str, float]
    branches: list[Branch]
    transformation: list[list[float]] | None = None
    external_fluxes: list[FluxBias] = Field(default_factory=list)


class OperatorsSpec(StrictModel):
    kind: Literal["operators"]
    dimensions: list[int]
    basis: str
    array_file: str
    hamiltonian: str
    operators: dict[str, str]


class SubsystemSpec(StrictModel):
    id: str = Field(min_length=1)
    source: str | None = None
    kind: Literal["oscillator", "transmon", "fluxonium"] | None = None
    frequency: Value | None = None
    josephson_energy: Value | None = None
    charging_energy: Value | None = None
    inductive_energy: Value | None = None
    ng: float | None = None
    flux: Value | None = None
    cutoff: int | None = None
    retained_levels: int | None = None

    @model_validator(mode="after")
    def check_kind(self) -> SubsystemSpec:
        fields = {
            "source": {"source"},
            "oscillator": {"frequency", "retained_levels"},
            "transmon": {"josephson_energy", "charging_energy", "ng", "cutoff", "retained_levels"},
            "fluxonium": {
                "josephson_energy",
                "charging_energy",
                "inductive_energy",
                "flux",
                "cutoff",
                "retained_levels",
            },
        }
        choice = "source" if self.source is not None else self.kind
        if choice not in fields or (self.source is not None and self.kind is not None):
            raise ValueError("subsystem requires a source or one constructor kind")
        supplied = {
            key
            for key in fields["fluxonium"] | fields["transmon"] | fields["oscillator"] | {"source"}
            if getattr(self, key) is not None
        }
        if supplied != fields[choice]:
            raise ValueError(f"{choice} subsystem requires exactly {sorted(fields[choice])}")
        if self.retained_levels is not None and self.retained_levels < 2:
            raise ValueError("subsystem retained_levels must be at least 2")
        if self.cutoff is not None and self.cutoff < 1:
            raise ValueError("subsystem cutoff must be positive")
        if self.ng is not None and not math.isfinite(self.ng):
            raise ValueError("ng must be finite")
        return self


class Interaction(StrictModel):
    coefficient: Value
    factors: list[str] = Field(min_length=1)
    add_hermitian_conjugate: bool = False


class CompositeSpec(StrictModel):
    kind: Literal["composite"]
    subsystems: list[SubsystemSpec] = Field(min_length=2)
    interactions: list[Interaction] = Field(default_factory=list)


ModelSpec = Annotated[CircuitSpec | OperatorsSpec | CompositeSpec, Field(discriminator="kind")]


class Truncation(StrictModel):
    cutoffs: dict[str, int]
    retained_levels: int = Field(gt=0)
    hierarchy: list[list[int]] | None = None
    hierarchy_levels: list[int] | None = None

    @model_validator(mode="after")
    def check_hierarchy(self) -> Truncation:
        if (self.hierarchy is None) != (self.hierarchy_levels is None):
            raise ValueError("hierarchy and hierarchy_levels are declared together")
        return self


class Dissipator(StrictModel):
    id: str
    factors: list[str]
    rate: Value


class Drive(StrictModel):
    id: str = Field(min_length=1)
    factors: list[str] = Field(min_length=1)


class SystemSpec(StrictModel):
    format_version: Literal[1, 2]
    id: str = Field(min_length=1)
    units: Units
    parameters: dict[str, Parameter] = Field(default_factory=dict)
    model: ModelSpec
    truncation: Truncation | None = None
    drives: list[Drive] = Field(default_factory=list)
    dissipators: list[Dissipator] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_version(self) -> SystemSpec:
        if self.format_version == 1 and self.truncation and self.truncation.hierarchy is not None:
            raise ValueError("truncation.hierarchy requires format_version = 2")
        return self


@dataclass(frozen=True)
class TensorTerm:
    coefficient_ghz: complex
    factors: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class ConstructedSystem:
    id: str
    hamiltonian_ghz: np.ndarray
    operators: dict[str, np.ndarray]
    dissipators: dict[str, tuple[np.ndarray, float]]
    dimensions: tuple[int, ...]
    basis: str
    provenance: dict[str, object]
    drives: dict[str, np.ndarray] = field(default_factory=dict)
    subsystem_ids: tuple[str, ...] = ()
    hamiltonian_terms: tuple[TensorTerm, ...] = ()
    operator_factors: dict[str, tuple[np.ndarray, ...]] = field(default_factory=dict)
    drive_factors: dict[str, tuple[np.ndarray, ...]] = field(default_factory=dict)
    dissipator_factors: dict[str, tuple[np.ndarray, ...]] = field(default_factory=dict)
    # Retained eigenstates in the charge basis of a single periodic circuit coordinate.
    charge_basis_vectors: np.ndarray | None = None
    # The scqubits HilbertSpace built at load; a model derived with another Hamiltonian has none.
    scqubits_space: object | None = field(default=None, repr=False, compare=False)

    def qutip_hamiltonian(self):
        import qutip as qt

        return qt.Qobj(self.hamiltonian_ghz, dims=[list(self.dimensions), list(self.dimensions)])

    def hilbert_space(self):
        """The scqubits `HilbertSpace` of a composite or a hierarchical circuit, for labels."""
        if self.scqubits_space is None:
            raise ValueError(
                f"{self.id} provides no HilbertSpace: only a loaded composite or hierarchical "
                "circuit does, not a single-coordinate circuit, an explicit model, or a model "
                "derived with another Hamiltonian"
            )
        return self.scqubits_space


@dataclass(frozen=True)
class _Leaf:
    """A subsystem as a composite's `HilbertSpace` takes it."""

    subsystem: object
    energies: np.ndarray
    vectors: np.ndarray
    # Operators in the subsystem's own basis, offset-relative; the export projects them on vectors.
    native: dict[str, object]
    # An explicit subsystem's off-diagonal Hamiltonian, a term on that subsystem alone.
    local: np.ndarray | None = None


@functools.cache
def _declared_class():
    from scqubits.core.qubit_base import QuantumSystem

    class Declared(QuantumSystem):
        """An explicit-operator subsystem whose bare states are its declared basis states."""

        def __init__(self, energies: np.ndarray):
            # First, as in scqubits' classes: subsystems compare by attributes, the ID leading.
            super().__init__(id_str=None)
            self.energies = energies
            self.truncated_dim = len(energies)

        @staticmethod
        def default_params():
            return {}

        def hilbertdim(self):
            return self.truncated_dim

        def eigenvals(self, evals_count=None):
            return self.energies[:evals_count]

        def eigensys(self, evals_count=None):
            return self.energies[:evals_count], np.eye(self.truncated_dim)[:, :evals_count]

    return Declared


def _declared_leaf(system: ConstructedSystem) -> _Leaf:
    """Declared states in declared order, the diagonal as bare energies."""
    hamiltonian = system.hamiltonian_ghz
    energies = np.real(np.diag(hamiltonian))
    local = hamiltonian - np.diag(energies)
    return _Leaf(
        _declared_class()(energies),
        energies,
        np.eye(len(energies)),
        dict(system.operators),
        local if np.any(local) else None,
    )


def _shifted(operator, offset: float):
    """The operator minus offset times the identity, kept sparse when it is sparse."""
    if offset == 0:
        return operator
    size = operator.shape[0]
    identity = (
        scipy.sparse.identity(size, format="csc")
        if scipy.sparse.issparse(operator)
        else np.eye(size)
    )
    return operator - offset * identity


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


def _inductive_energy(branch: Branch, parameters: dict[str, Parameter]) -> float:
    if branch.inductive_energy is not None:
        return _positive(
            _resolved(branch.inductive_energy, parameters, "GHz", branch.id), branch.id
        )
    inductance_nh = _positive(_resolved(branch.inductance, parameters, "nH", branch.id), branch.id)
    return (h / (2 * e * 2 * math.pi)) ** 2 / (inductance_nh * 1e-9 * h * 1e9)


def hierarchy_problem(
    coordinates: dict[str, str],
    cutoffs: dict[str, int],
    groups: list[list[int]],
    levels: list[int],
    retained_levels: int,
) -> str:
    """Why these hierarchy levels cannot be kept, or an empty string."""
    # scqubits' truncated harmonic construction adds a spurious state at index N/2, which lies
    # above at least N/2 states of any group holding that coordinate.
    for group, count in zip(groups, levels, strict=True):
        for index in group:
            cutoff = cutoffs[f"theta{index}"]
            if coordinates[f"theta{index}"] == "extended" and count > cutoff // 2:
                return (
                    f"truncation: harmonic cutoff theta{index} = {cutoff} gives at most "
                    f"{cutoff // 2} usable levels to group {group}; set theta{index} to at "
                    f"least {2 * count}"
                )
    # scqubits' sparse eigensolver returns at most two fewer states than its basis holds.
    sizes = [
        math.prod(
            2 * cutoffs[f"theta{index}"] + 1
            if coordinates[f"theta{index}"] == "periodic"
            else cutoffs[f"theta{index}"]
            for index in group
        )
        for group in groups
    ]
    for index, (group, size, count) in enumerate(zip(groups, sizes, levels, strict=True)):
        if count > size - 2:
            return (
                f"truncation.hierarchy_levels[{index}] = {count} exceeds {size - 2}, the most "
                f"kept from group {group}'s {size} construction states"
            )
    product = math.prod(levels)
    if retained_levels > product - 2:
        return (
            f"truncation.retained_levels = {retained_levels} exceeds {product - 2}, the most "
            f"kept from the groups' {product} product states"
        )
    return ""


def _tensor(factors: tuple[np.ndarray, ...]) -> np.ndarray:
    result = np.array([[1]], dtype=np.complex128)
    for factor in factors:
        result = np.kron(result, factor)
    return result


def _scqubits():
    import scqubits as scq

    if scq.get_units() != "GHz":
        raise SpecError(f"scqubits units must be GHz, currently {scq.get_units()}")
    return scq


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
    try:
        content = asset.read_bytes()
    except OSError as exc:
        raise SpecError(f"model.array_file: cannot read {model.array_file!r}: {exc}") from exc
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


def _determinant(rows: list[list[int]]) -> int:
    """Exact determinant of a square integer matrix, by fraction-free (Bareiss) elimination."""
    rows = [list(row) for row in rows]
    sign, previous = 1, 1
    for k in range(len(rows) - 1):
        if rows[k][k] == 0:
            swap = next((i for i in range(k + 1, len(rows)) if rows[i][k]), None)
            if swap is None:
                return 0
            rows[k], rows[swap], sign = rows[swap], rows[k], -sign
        for i in range(k + 1, len(rows)):
            for j in range(k + 1, len(rows)):
                rows[i][j] = (rows[i][j] * rows[k][k] - rows[i][k] * rows[k][j]) // previous
        previous = rows[k][k]
    return sign * rows[-1][-1]


def _primitive(columns: list[list[int]]) -> bool:
    """Whether integer columns extend to a unimodular basis: their maximal minors have gcd 1."""
    width = len(columns[0])
    minors = (
        _determinant([columns[row] for row in chosen])
        for chosen in itertools.combinations(range(len(columns)), width)
    )
    return math.gcd(*minors) == 1


def _construct_circuit(spec: SystemSpec):
    scq = _scqubits()

    model = spec.model
    if model.ground != 0 or model.nodes != list(range(len(model.nodes))) or len(model.nodes) < 2:
        raise SpecError("circuit nodes must be [0, ..., N] with ground 0")
    if len({branch.id for branch in model.branches}) != len(model.branches):
        raise SpecError("model.branches: branch IDs must be unique")
    if any(not set(branch.nodes) <= set(model.nodes) for branch in model.branches):
        raise SpecError("circuit branch endpoints must be declared nodes")
    if not model.branches or not any(branch.kind in ("JJ", "L") for branch in model.branches):
        raise SpecError("circuit requires at least one JJ or L branch")
    truncation = spec.truncation
    if truncation is None:
        raise SpecError("circuit requires truncation.cutoffs and retained_levels")
    lines = ["branches:"]
    converted = {}
    for branch in model.branches:
        first, second = branch.nodes
        if branch.kind == "L":
            el = _inductive_energy(branch, spec.parameters)
            converted[branch.id] = {
                "supplied": branch.model_dump(exclude_none=True),
                "inductive_energy_ghz": el,
            }
            lines.append(f"- [L, {first}, {second}, {el:.17g}]")
            continue
        ec = _charging_energy(branch, spec.parameters)
        converted[branch.id] = {
            "supplied": branch.model_dump(exclude_none=True),
            "charging_energy_ghz": ec,
        }
        if branch.kind == "JJ":
            ej = _positive(
                _resolved(branch.josephson_energy, spec.parameters, "GHz", branch.id), branch.id
            )
            converted[branch.id]["josephson_energy_ghz"] = ej
            lines.append(f"- [JJ, {first}, {second}, {ej:.17g}, {ec:.17g}]")
        else:
            lines.append(f"- [C, {first}, {second}, {ec:.17g}]")
    try:
        circuit = scq.Circuit.from_yaml_string(
            "\n".join(lines) + "\n",
            truncated_dim=truncation.retained_levels,
            ext_basis="harmonic",
        )
        count = len(model.nodes) - 1
        if count == 1 and model.transformation is not None:
            raise SpecError("single-coordinate circuit does not need a transformation matrix")
        if count > 1 and model.transformation is None:
            raise SpecError(
                "multi-node circuit requires an explicit transformation matrix; "
                f"scqubits default to adopt: {np.asarray(circuit.transformation_matrix).tolist()}"
            )
        categories = circuit.var_categories
        active = set(categories["periodic"] + categories["extended"])
        periodic_count = len(categories["periodic"])
        if active != set(range(1, count + 1)):
            raise SpecError("circuit has unsupported frozen or free coordinates")
        matrix = np.asarray(
            model.transformation
            if model.transformation is not None
            else circuit.transformation_matrix,
            dtype=float,
        )
        if matrix.shape != (count, count) or not np.all(np.isfinite(matrix)):
            raise SpecError(f"transformation must be a finite {count}x{count} matrix")
        if count == 1 and not np.isclose(abs(matrix[0, 0]), 1, atol=1e-12):
            raise SpecError("single-node transformation must preserve node orientation")
        if count > 1 and np.linalg.matrix_rank(matrix) < count:
            raise SpecError("transformation must be invertible")
        closure_ids = [bias.branch for bias in model.external_fluxes]
        if len(set(closure_ids)) != len(closure_ids):
            raise SpecError("external_fluxes: closure branches must be unique")
        branch_ids = {branch.id: index for index, branch in enumerate(model.branches)}
        if any(branch_id not in branch_ids for branch_id in closure_ids):
            raise SpecError("external_fluxes: unknown closure branch")
        if any(model.branches[branch_ids[branch_id]].kind == "C" for branch_id in closure_ids):
            raise SpecError("external_fluxes: a capacitor cannot close a flux loop")
        if model.transformation is not None or closure_ids:
            circuit.configure(
                transformation_matrix=matrix if model.transformation is not None else None,
                closure_branches=[circuit.branches[branch_ids[key]] for key in closure_ids]
                if closure_ids
                else None,
                ext_basis="harmonic",
            )
        if len(circuit.external_fluxes) != len(closure_ids):
            raise SpecError(
                "external_fluxes: give one closure branch and value per independent loop"
            )
        fluxes = [
            _resolved(bias.value, spec.parameters, "Phi0", f"external_fluxes.{bias.branch}")
            if isinstance(bias.value, ParameterRef)
            else bias.value
            for bias in model.external_fluxes
        ]
        for index, flux in enumerate(fluxes, start=1):
            setattr(circuit, f"Φ{index}", flux)
        categories = circuit.var_categories
        # scqubits counts a column outside the periodic subspace as extended, with a warning only.
        if (
            set(categories["periodic"] + categories["extended"]) != active
            or len(categories["periodic"]) != periodic_count
        ):
            raise SpecError("transformation changes the quantized coordinate categories")
        # The periodic columns must be a primitive basis of the periodic identification lattice.
        # They are checked exactly: construction uses these entries, so a tiny fraction counts.
        periodic = matrix[:, [index - 1 for index in categories["periodic"]]]
        if (
            count > 1
            and periodic.size
            and (
                not np.array_equal(periodic, np.rint(periodic))
                or not _primitive(np.rint(periodic).astype(int).tolist())
            )
        ):
            raise SpecError(
                "transformation: periodic columns must be integer and primitive "
                "(integer and unimodular when every coordinate is periodic)"
            )
        if set(truncation.cutoffs) != {f"theta{index}" for index in active}:
            raise SpecError(f"truncation.cutoffs must name {sorted(active)} as thetaK")
        groups, levels = truncation.hierarchy, truncation.hierarchy_levels
        singletons = [[index] for index in sorted(active)]
        if count > 1 and groups is None:
            needs = "need format_version = 2 with" if spec.format_version == 1 else "require"
            raise SpecError(
                f"multi-node circuits {needs} truncation.hierarchy and truncation.hierarchy_levels; "
                f"partition to adopt: hierarchy = {singletons}, with hierarchy_levels giving "
                "the eigenstates each group keeps"
            )
        if count == 1 and groups is not None:
            raise SpecError("a single-coordinate circuit takes no truncation.hierarchy")
        if groups is not None:
            members = sorted(index for group in groups for index in group)
            if len(groups) < 2 or not all(groups) or members != sorted(active):
                raise SpecError(
                    f"truncation.hierarchy must split coordinates {sorted(active)} into at least "
                    f"two nonempty groups, each coordinate in one; to adopt: {singletons}"
                )
            if len(levels) != len(groups) or any(level < 1 for level in levels):
                raise SpecError("truncation.hierarchy_levels needs one positive count per group")
        coordinates = {
            f"theta{index}": "periodic" if index in categories["periodic"] else "extended"
            for index in sorted(active)
        }
        for index in active:
            cutoff = truncation.cutoffs[f"theta{index}"]
            if cutoff < 1 or (index in categories["extended"] and cutoff < 2):
                raise SpecError("truncation cutoffs must be positive (extended: at least 2)")
            if index in categories["periodic"]:
                setattr(circuit, f"cutoff_n_{index}", cutoff)
            else:
                setattr(circuit, f"cutoff_ext_{index}", cutoff)
        # scqubits' truncated harmonic construction adds a spurious state at index N/2; under a
        # hierarchy the groups keep it out instead.
        for index in categories["extended"] if groups is None else ():
            cutoff = truncation.cutoffs[f"theta{index}"]
            if truncation.retained_levels > cutoff // 2:
                raise SpecError(
                    f"truncation: harmonic cutoff theta{index} = {cutoff} gives at most "
                    f"{cutoff // 2} usable levels; set theta{index} to at least "
                    f"{2 * truncation.retained_levels}"
                )
        if groups is not None:
            problem = hierarchy_problem(
                coordinates, truncation.cutoffs, groups, levels, truncation.retained_levels
            )
            if problem:
                raise SpecError(problem)
        else:
            basis_size = math.prod(
                2 * truncation.cutoffs[f"theta{index}"] + 1
                if index in categories["periodic"]
                else truncation.cutoffs[f"theta{index}"]
                for index in sorted(active)
            )
            if truncation.retained_levels > basis_size:
                raise SpecError("truncation: insufficient cutoff for retainable levels")
        node_keys = {str(index) for index in range(1, count + 1)}
        if not set(model.offset_charges) <= node_keys or any(
            not math.isfinite(value) for value in model.offset_charges.values()
        ):
            raise SpecError("offset_charges must use finite values on declared non-ground nodes")
        # A node reaches a periodic coordinate through a nonzero entry in that coordinate's column.
        required = {
            str(node)
            for node in range(1, count + 1)
            for index in categories["periodic"]
            if matrix[node - 1, index - 1] != 0
        }
        if missing := sorted(required - set(model.offset_charges)):
            raise SpecError(
                "offset_charges requires an explicit value for every node reaching a periodic "
                f"coordinate; missing: {missing}"
            )
        node_offsets = np.array(
            [model.offset_charges.get(str(index), 0.0) for index in range(1, count + 1)]
        )
        coordinate_offsets = matrix.T @ node_offsets
        dropped_offsets = {
            f"theta{index}": float(coordinate_offsets[index - 1])
            for index in categories["extended"]
            if coordinate_offsets[index - 1] != 0
        }
        for index in categories["extended"]:
            coordinate_offsets[index - 1] = 0
        for index in categories["periodic"]:
            setattr(circuit, f"ng{index}", -coordinate_offsets[index - 1])
        if groups is not None:
            # scqubits sizes each group from the cutoffs set before the hierarchy is configured;
            # fluxes and offsets carry over.
            circuit.configure(
                transformation_matrix=matrix,
                closure_branches=[circuit.branches[branch_ids[key]] for key in closure_ids]
                if closure_ids
                else None,
                system_hierarchy=groups,
                subsystem_trunc_dims=levels,
                ext_basis="harmonic",
            )
        eigenvalues, eigenvectors = circuit.eigensys(evals_count=truncation.retained_levels)
        charge_vectors = (
            np.asarray(eigenvectors) if count == 1 and categories["periodic"] == [1] else None
        )
        unavailable = {}
        # Operators are built in the circuit's own basis, then projected on its eigenvectors.
        coordinate_charges = {
            index: _shifted(getattr(circuit, f"n{index}_operator")(), coordinate_offsets[index - 1])
            if index in categories["periodic"]
            else getattr(circuit, f"Q{index}_operator")()
            for index in active
        }
        if count == 1:
            direction = float(matrix[0, 0])
            native = {"charge": coordinate_charges[1] / direction}
            if 1 not in categories["periodic"]:
                native["phase"] = direction * circuit.θ1_operator()
            if 1 in categories["periodic"] or any(branch.kind == "JJ" for branch in model.branches):
                native["cos_phase"] = circuit.cosθ1_operator()
                native["sin_phase"] = direction * circuit.sinθ1_operator()
        else:
            inverse_transpose = np.linalg.inv(matrix.T)
            native = {
                f"node{node}.charge": sum(
                    inverse_transpose[node - 1, index - 1] * coordinate_charges[index]
                    for index in sorted(active)
                )
                for node in range(1, count + 1)
            }
            # φ_K = Σⱼ M_Kj θⱼ is single-valued only when every weighted coordinate is extended.
            for node in range(1, count + 1):
                weights = {
                    index: matrix[node - 1, index - 1]
                    for index in sorted(active)
                    if matrix[node - 1, index - 1] != 0
                }
                blocking = [f"theta{index}" for index in weights if index in categories["periodic"]]
                if blocking:
                    unavailable[f"node{node}.phase"] = (
                        f"node {node}'s phase depends on periodic coordinate {', '.join(blocking)}"
                    )
                else:
                    native[f"node{node}.phase"] = sum(
                        weight * getattr(circuit, f"θ{index}_operator")()
                        for index, weight in weights.items()
                    )
        projection = eigenvectors.conj().T
        operators = {
            name: np.asarray(projection @ operator @ eigenvectors)
            for name, operator in native.items()
        }
    except SpecError:
        raise
    except (ValueError, RuntimeError, KeyError, AttributeError, np.linalg.LinAlgError) as exc:
        raise SpecError(f"circuit construction failed: {exc}") from exc
    provenance = {
        "scqubits_version": scq.__version__,
        "converted_branches": converted,
        "node_offsets": dict(model.offset_charges),
        "coordinate_offsets": coordinate_offsets.tolist(),
        "dropped_offsets": dropped_offsets,
        "transformation": matrix.tolist(),
        "transformation_source": "authored"
        if model.transformation is not None
        else "scqubits default",
        "cutoffs": dict(truncation.cutoffs),
        "coordinates": coordinates,
        **({"hierarchy": groups, "hierarchy_levels": levels} if groups is not None else {}),
        **({"unavailable_operators": unavailable} if unavailable else {}),
        "oriented_loops": [
            {
                "branch": bias.branch,
                "nodes": list(model.branches[branch_ids[bias.branch]].nodes),
                "value": flux,
                "unit": "Phi0",
                **(
                    {"parameter": bias.value.parameter}
                    if isinstance(bias.value, ParameterRef)
                    else {}
                ),
            }
            for bias, flux in zip(model.external_fluxes, fluxes, strict=True)
        ],
    }
    return (
        np.diag(eigenvalues),
        operators,
        (truncation.retained_levels,),
        "isolated_eigenstates",
        provenance,
        charge_vectors,
        _Leaf(circuit, eigenvalues, eigenvectors, native),
    )


def _construct_named_subsystem(
    item: SubsystemSpec, parameters: dict[str, Parameter]
) -> tuple[ConstructedSystem, _Leaf]:
    scq = _scqubits()

    count = item.retained_levels
    try:
        if item.kind == "oscillator":
            frequency = _positive(_resolved(item.frequency, parameters, "GHz", item.id), item.id)
            source = scq.Oscillator(E_osc=frequency, truncated_dim=count)
            lower = np.asarray(source.annihilation_operator(), dtype=np.complex128)
            native = {
                "annihilation": lower,
                "number": lower.conj().T @ lower,
                "position": lower + lower.conj().T,
            }
        else:
            ej = _positive(_resolved(item.josephson_energy, parameters, "GHz", item.id), item.id)
            ec = _positive(_resolved(item.charging_energy, parameters, "GHz", item.id), item.id)
            cutoff = item.cutoff
            if item.kind == "transmon":
                if count > 2 * cutoff + 1:
                    raise SpecError(f"{item.id}: retained_levels exceed charge cutoff basis")
                source = scq.Transmon(EJ=ej, EC=ec, ng=item.ng, ncut=cutoff, truncated_dim=count)
            else:
                el = _positive(
                    _resolved(item.inductive_energy, parameters, "GHz", item.id), item.id
                )
                flux = _resolved(item.flux, parameters, "Phi0", item.id)
                if count > cutoff:
                    raise SpecError(f"{item.id}: retained_levels exceed harmonic cutoff")
                source = scq.Fluxonium(
                    EJ=ej, EC=ec, EL=el, flux=flux, cutoff=cutoff, truncated_dim=count
                )
            native = {
                "charge": _shifted(source.n_operator(), item.ng if item.kind == "transmon" else 0),
                "cos_phase": source.cos_phi_operator(),
                "sin_phase": source.sin_phi_operator(),
            }
            if item.kind == "fluxonium":
                native["phase"] = source.phi_operator()
        energies, eigenvectors = source.eigensys(evals_count=count)
        projection = eigenvectors.conj().T
        operators = {
            name: projection @ operator @ eigenvectors for name, operator in native.items()
        }
    except SpecError:
        raise
    except (ValueError, RuntimeError, AttributeError, np.linalg.LinAlgError) as exc:
        raise SpecError(f"{item.id}: constructor failed: {exc}") from exc
    hamiltonian = np.diag(energies).astype(np.complex128)
    system = ConstructedSystem(
        item.id,
        hamiltonian,
        operators,
        {},
        (count,),
        "isolated_eigenstates",
        {"constructor": item.kind, "scqubits_version": scq.__version__},
        subsystem_ids=(item.id,),
        hamiltonian_terms=(TensorTerm(1, (hamiltonian,)),),
        operator_factors={name: (operator,) for name, operator in operators.items()},
    )
    return system, _Leaf(source, energies, eigenvectors, native)


def unavailable_operators(provenance: dict) -> dict[str, str]:
    """Names a circuit cannot export, with the reason, as a spec would reference them."""
    reasons = dict(provenance.get("unavailable_operators", {}))
    for subsystem, child in provenance.get("subsystems", {}).items():
        for name, reason in child.get("unavailable_operators", {}).items():
            reasons[f"{subsystem}.{name}"] = reason
    return reasons


def _composite_factors(
    names: list[str],
    operators: dict[str, tuple[np.ndarray, ...]],
    dimensions: tuple[int, ...],
    label: str,
    unavailable: dict[str, str],
) -> tuple[np.ndarray, ...]:
    factors = [np.eye(dimension, dtype=np.complex128) for dimension in dimensions]
    for name in names:
        if name in unavailable:
            raise SpecError(f"{label}: operator {name!r} is unavailable: {unavailable[name]}")
        if name not in operators:
            raise SpecError(f"{label}: unknown operator {name!r}; available: {sorted(operators)}")
        for index, factor in enumerate(operators[name]):
            factors[index] = factors[index] @ factor
    return tuple(factors)


def _construct_composite(spec: SystemSpec, source: Path):
    model = spec.model
    if spec.truncation is not None:
        raise SpecError("composite uses subsystem truncations; omit top-level truncation")
    ids = [item.id for item in model.subsystems]
    if len(set(ids)) != len(ids) or any("." in item_id for item_id in ids):
        raise SpecError("subsystem IDs must be unique and contain no dot")
    leaves = []
    for item in model.subsystems:
        if item.source is None:
            leaves.append(_construct_named_subsystem(item, spec.parameters))
            continue
        path = Path(item.source)
        if path.name != item.source or path.suffix != ".toml":
            raise SpecError(f"{item.id}: source must name a TOML file beside the composite")
        leaf_path = (source.parent / path).resolve()
        if leaf_path.parent != source.parent:
            raise SpecError(f"{item.id}: source resolves outside the composite directory")
        leaf, bare = _load_system(leaf_path, allow_composite=False)
        if leaf.drives or leaf.dissipators:
            raise SpecError(f"{item.id}: declare leaf drives and dissipators on the composite")
        if len(leaf.dimensions) != 1:
            raise SpecError(f"{item.id}: source must be a single subsystem")
        leaves.append((leaf, bare or _declared_leaf(leaf)))
    dimensions = tuple(leaf.dimensions[0] for leaf, _ in leaves)
    identities = tuple(np.eye(dimension, dtype=np.complex128) for dimension in dimensions)
    # The leaves give GHz matrices, so scqubits' unit setting does not apply here.
    from scqubits import HilbertSpace

    space = HilbertSpace([bare.subsystem for _, bare in leaves])
    terms = []
    operators = {}
    operator_factors = {}
    native = {}
    for index, (item, (leaf, bare)) in enumerate(zip(model.subsystems, leaves)):
        local = list(identities)
        local[index] = leaf.hamiltonian_ghz
        terms.append(TensorTerm(1, tuple(local)))
        if bare.local is not None:
            space.add_interaction(g=1, op1=(bare.local, bare.subsystem), check_validity=False)
        for name, operator in leaf.operators.items():
            factors = list(identities)
            factors[index] = operator
            key = f"{item.id}.{name}"
            operator_factors[key] = tuple(factors)
            operators[key] = _tensor(tuple(factors))
            native[key] = (bare.native[name], bare.subsystem)
    provenance = {
        "subsystems": {name: leaf.provenance for name, (leaf, _) in zip(ids, leaves)},
        "subsystem_bases": {name: leaf.basis for name, (leaf, _) in zip(ids, leaves)},
    }
    unavailable = unavailable_operators(provenance)
    for index, interaction in enumerate(model.interactions):
        label = f"model.interactions.{index}"
        coefficient = _resolved(interaction.coefficient, spec.parameters, "GHz", label)
        factors = _composite_factors(
            interaction.factors, operator_factors, dimensions, label, unavailable
        )
        terms.append(TensorTerm(coefficient, factors))
        if interaction.add_hermitian_conjugate:
            terms.append(TensorTerm(coefficient, tuple(factor.conj().T for factor in factors)))
        space.add_interaction(
            g=coefficient,
            **{f"op{place}": native[name] for place, name in enumerate(interaction.factors, 1)},
            add_hc=interaction.add_hermitian_conjugate,
            check_validity=False,
        )
    # scqubits projects each native operator on the eigenvectors the export was projected on.
    bare_esys = {index: (bare.energies, bare.vectors) for index, (_, bare) in enumerate(leaves)}
    hamiltonian = space.hamiltonian(bare_esys=bare_esys).full()
    return (
        hamiltonian,
        operators,
        dimensions,
        "product",
        provenance,
        tuple(terms),
        operator_factors,
        tuple(ids),
        space,
    )


def _load_system(
    path: str | Path, *, allow_composite: bool, content: bytes | None = None
) -> tuple[ConstructedSystem, _Leaf | None]:
    """Validate a system file and construct the CPU model, with a circuit's leaf for composites."""
    source = Path(path).resolve()
    try:
        content = source.read_bytes() if content is None else content
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
    charge_vectors = None
    leaf = None
    space = None
    if isinstance(spec.model, CompositeSpec):
        if not allow_composite:
            raise SpecError("referenced subsystem cannot itself be a composite")
        (
            hamiltonian,
            operators,
            dimensions,
            basis,
            provenance,
            terms,
            operator_factors,
            subsystem_ids,
            space,
        ) = _construct_composite(spec, source)
    elif isinstance(spec.model, OperatorsSpec):
        hamiltonian, operators, dimensions, basis, provenance = _construct_operators(spec, source)
        terms = (TensorTerm(1, (hamiltonian,)),)
        operator_factors = {name: (operator,) for name, operator in operators.items()}
        subsystem_ids = (spec.id,)
    else:
        hamiltonian, operators, dimensions, basis, provenance, charge_vectors, leaf = (
            _construct_circuit(spec)
        )
        if spec.truncation.hierarchy is not None:
            space = leaf.subsystem.hilbert_space
        terms = (TensorTerm(1, (hamiltonian,)),)
        operator_factors = {name: (operator,) for name, operator in operators.items()}
        subsystem_ids = (spec.id,)
    if not np.allclose(hamiltonian, hamiltonian.conj().T, rtol=0, atol=1e-10):
        raise SpecError("model.hamiltonian: matrix must be Hermitian")
    unavailable = unavailable_operators(provenance)
    drives = {}
    dissipators = {}
    drive_factors = {}
    dissipator_factors = {}
    for drive in spec.drives:
        if drive.id in drives or drive.id in dissipators:
            raise SpecError(f"duplicate drive or dissipator ID {drive.id!r}")
        factors = _composite_factors(
            drive.factors, operator_factors, dimensions, f"drives.{drive.id}", unavailable
        )
        operator = _tensor(factors)
        if not np.allclose(operator, operator.conj().T, rtol=0, atol=1e-10):
            raise SpecError(f"drives.{drive.id}: operator must be Hermitian")
        drives[drive.id] = operator
        drive_factors[drive.id] = factors
    for dissipator in spec.dissipators:
        if dissipator.id in dissipators or dissipator.id in drives:
            raise SpecError(f"duplicate drive or dissipator ID {dissipator.id!r}")
        factors = _composite_factors(
            dissipator.factors,
            operator_factors,
            dimensions,
            f"dissipators.{dissipator.id}",
            unavailable,
        )
        rate = _resolved(dissipator.rate, spec.parameters, "1/ns", dissipator.id)
        if rate < 0:
            raise SpecError(f"dissipators.{dissipator.id}: rate must be nonnegative")
        dissipators[dissipator.id] = (_tensor(factors), rate)
        dissipator_factors[dissipator.id] = factors
    provenance["parameters"] = {
        name: parameter.model_dump(exclude_none=True) for name, parameter in spec.parameters.items()
    }
    provenance["system_sha256"] = hashlib.sha256(content).hexdigest()
    system = ConstructedSystem(
        spec.id,
        np.asarray(hamiltonian, dtype=np.complex128),
        operators,
        dissipators,
        dimensions,
        basis,
        provenance,
        drives,
        subsystem_ids,
        terms,
        operator_factors,
        drive_factors,
        dissipator_factors,
        charge_vectors,
        space,
    )
    return system, leaf


def load_system(path: str | Path, *, content: bytes | None = None) -> ConstructedSystem:
    """Load a system file; ``content`` replaces its bytes, resolving references beside it."""
    return _load_system(path, allow_composite=True, content=content)[0]

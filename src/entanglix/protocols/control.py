"""Validated closed-system control models and shared ProblemSpec export."""

import hashlib
import io
import math
import re
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
import tomli_w
from pydantic import Field, model_validator

from entanglix.engines.frames import matrix, prepare_dynamics
from entanglix.evolution import Carrier, EvolutionInput, RotatingFrame, normalize_input
from entanglix.models import ConstructedSystem, StrictModel, load_system
from entanglix.protocols import (
    SystemSource,
    base_files,
    override_system,
    read_inputs,
    with_parameters,
    write_frozen,
)
from entanglix.protocols.plans import FrameSpec, Refinement
from entanglix.solvers import Pulse as NeutralPulse

AMICODE_REVISION = "cb717f57cb63fa292917188c8ca32f50ddcca539"
STARTING_PULSE = "starting-pulse.npz"
STEP_RELATIVE_HALF_WIDTH = 1e-7
# Derived values agree with declared ones, and across channels, within this tolerance.
RESOLVED_RELATIVE_TOLERANCE = 1e-9
Finite = Annotated[float, Field(allow_inf_nan=False)]
Nonnegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Level = Annotated[int, Field(strict=True, ge=0)]
EVALUATION_TARGET = re.compile(r"^(parameters|model\.offset_charges)\.[^.]+$")


class ControlFrame(FrameSpec):
    frequencies_ghz: dict[str, float] = Field(default_factory=dict)
    transitions: dict[str, Annotated[list[Level], Field(min_length=2, max_length=2)]] = Field(
        default_factory=dict
    )

    @model_validator(mode="after")
    def check_transitions(self):
        if not self.frequencies_ghz and not self.transitions:
            raise ValueError("frame requires frequencies_ghz or transitions")
        if any(lower == upper for lower, upper in self.transitions.values()):
            raise ValueError("frame.transitions require two distinct levels")
        return self


class Channel(StrictModel):
    id: str = Field(min_length=1)
    drive: str = Field(min_length=1)
    frequency_ghz: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    follows: str | None = Field(default=None, min_length=1)
    phase_rad: Finite = 0.0
    amplitude_bound_ghz: Nonnegative | None = None


class RabiBounds(StrictModel):
    subsystem: str = Field(min_length=1)
    amplitude_ghz: Nonnegative | None = None
    du_ghz_per_ns: Nonnegative | None = None
    ddu_ghz_per_ns2: Nonnegative | None = None


class Goal(StrictModel):
    kind: Literal["unitary"]
    gate: Literal["I", "X", "Y", "Z", "H"] | None = None
    matrix: list[list[Finite | list[Finite]]] | None = None
    subspace: list[list[Annotated[int, Field(strict=True, gt=0)]]] = Field(
        min_length=1, max_length=1
    )

    @model_validator(mode="after")
    def check_goal(self):
        if (self.gate is None) == (self.matrix is None):
            raise ValueError("goal requires exactly one of gate or matrix")
        indices = self.subspace[0]
        if not indices or len(set(indices)) != len(indices):
            raise ValueError("goal.subspace requires distinct one-based flattened basis indices")
        operator = self.operator()
        if operator.shape != (len(indices), len(indices)) or not np.allclose(
            operator.conj().T @ operator, np.eye(len(indices)), rtol=0, atol=1e-10
        ):
            raise ValueError("goal must be a unitary of the declared subspace dimension")
        return self

    def operator(self):
        if self.matrix is not None:
            try:
                if any(
                    isinstance(entry, list) and len(entry) != 2
                    for row in self.matrix
                    for entry in row
                ):
                    raise ValueError("complex entries require two components")
                return np.asarray(
                    [
                        [
                            complex(*entry) if isinstance(entry, list) else complex(entry)
                            for entry in row
                        ]
                        for row in self.matrix
                    ],
                    dtype=complex,
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "goal.matrix entries must be real numbers or [real, imaginary]"
                ) from exc
        return np.asarray(
            {
                "I": [[1, 0], [0, 1]],
                "X": [[0, 1], [1, 0]],
                "Y": [[0, -1j], [1j, 0]],
                "Z": [[1, 0], [0, -1]],
                "H": np.asarray([[1, 1], [1, -1]]) / np.sqrt(2),
            }[self.gate],
            dtype=complex,
        )


class Pulse(StrictModel):
    kind: Literal["zero_order"]
    T: float = Field(gt=0, allow_inf_nan=False)
    init: Literal["default", "random", "zeros"] | None = None
    # Seeds stay within the 64-bit integers of TOML records and the Julia adapter.
    seed: int | None = Field(default=None, strict=True, ge=0, lt=2**63)


class StartingPulse(StrictModel):
    """A neutral pulse naming its channels: a promoted entry's, a control run's, or a file."""

    entry: str | None = Field(default=None, min_length=1)
    run: str | None = Field(default=None, min_length=1)
    file: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def one_source(self):
        if sum(value is not None for value in (self.entry, self.run, self.file)) != 1:
            raise ValueError("starting_pulse names exactly one of entry, run or file")
        return self


class Problem(StrictModel):
    template: Literal["SmoothPulseProblem"]
    N: int = Field(strict=True, ge=2)
    Q: Nonnegative | None = None
    R: Nonnegative | None = None
    R_u: Nonnegative | None = None
    R_du: Nonnegative | None = None
    R_ddu: Nonnegative | None = None
    du_bound: Nonnegative | None = None
    ddu_bound: Nonnegative | None = None
    final_fidelity: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    free_dt: Literal[False] = False
    free_phase: Literal[False] = False


class Solver(StrictModel):
    backend: str = Field(min_length=1)
    fallback: str | None = Field(default=None, min_length=1)
    options: dict[str, Any] = Field(default_factory=dict)


class Evaluation(StrictModel):
    """Samples for the requested uncertainty diagnostic; not part of the ProblemSpec."""

    quantities: dict[str, Literal["drifting", "calibrated"]] = Field(min_length=1)
    samples: list[dict[str, Finite]] = Field(min_length=1)
    refinement: dict[str, Refinement] = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_samples(self):
        unknown = sorted(name for name in self.quantities if not EVALUATION_TARGET.fullmatch(name))
        if unknown:
            raise ValueError(
                f"evaluation quantities must be parameters.<name> or "
                f"model.offset_charges.<node>: {unknown}"
            )
        for index, sample in enumerate(self.samples):
            if not sample or sample.keys() - self.quantities.keys():
                raise ValueError(f"evaluation.samples[{index}] must set declared quantities")
        if len({tuple(sorted(sample.items())) for sample in self.samples}) != len(self.samples):
            raise ValueError("evaluation samples must be unique")
        return self


class ControlSpec(SystemSource):
    format_version: Literal[1]
    parameters: dict[str, Finite] = Field(default_factory=dict)
    frame: ControlFrame
    channels: list[Channel] = Field(min_length=1)
    rabi_bounds: RabiBounds | None = None
    evaluation: Evaluation | None = None
    goal: Goal
    pulse: Pulse
    starting_pulse: StartingPulse | None = None
    problem: Problem
    solver: Solver
    timeout_seconds: float | None = Field(default=None, gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def check_channels(self):
        names = [channel.id for channel in self.channels]
        if len(set(names)) != len(names):
            raise ValueError("channel IDs must be unique")
        if self.starting_pulse is not None and (
            self.pulse.init is not None or self.pulse.seed is not None
        ):
            raise ValueError("a starting pulse replaces pulse.init and pulse.seed")
        return self


def _encoded(operator):
    return np.stack([operator.real, operator.imag], axis=-1).tolist()


@dataclass(frozen=True)
class LoadedControl:
    definition: ControlSpec
    system: ConstructedSystem
    problem_spec: dict
    problem_bytes: bytes
    digest: str
    inputs: dict[str, Path]
    input_digests: dict[str, str]
    listing: dict
    catalog: dict


def _run_pulse(name: str, runs_root: Path) -> tuple[Path, list[str]]:
    """A completed control run's neutral pulse, with the channel IDs its frozen spec declares."""
    from entanglix.catalog import RUN_ID
    from entanglix.runner import _digest, _load_record, _read_attempt, _script_result

    path = Path(name)
    root = runs_root.resolve()
    run = root / path
    if (
        path.is_absolute()
        or len(path.parts) != 2
        or ".." in path.parts
        or not RUN_ID.fullmatch(path.name)
        or not run.resolve().is_relative_to(root)
        or not (run / "run.toml").is_file()
    ):
        raise ValueError(
            f"starting_pulse.run must name <system_id>/<run_id> in the runs root: {name}"
        )
    record = _read_attempt(run)
    if record["manifest"]["protocol"] != "control" or record["status"] != "completed":
        raise ValueError(f"starting_pulse.run must name a completed control run: {name}")
    spec = run / "spec/control.toml"
    if _digest(spec.read_bytes()) != record["manifest"]["spec_sha256"]["control.toml"]:
        raise ValueError(f"source input digest mismatch: {name}/spec/control.toml")
    result = _script_result(run)
    channels = [channel["id"] for channel in _load_record(spec)["channels"]]
    return (run / result["artifacts"]["pulse"]["path"]).resolve(), channels


def _starting_source(start: StartingPulse, protocol: Path, runs_root: Path, replay_source):
    """Locate a supplied start; a replay uses its frozen copy and keeps the recorded source."""
    from entanglix.catalog import pulse_source
    from entanglix.runner import _load_record

    lineage = {"source_entry_ids": [], "parent_run_ids": []}
    channels = None
    if replay_source is not None:
        path = replay_source / "spec" / STARTING_PULSE
        source = _load_record(replay_source / "run.toml")["launch"]["starting_pulse"]["source"]
    elif start.entry is not None:
        path = pulse_source(start.entry, runs_root)
        source = {"entry": start.entry}
        lineage["source_entry_ids"].append(start.entry)
    elif start.run is not None:
        path, channels = _run_pulse(start.run, runs_root)
        source = {"run": start.run}
        lineage["parent_run_ids"].append(Path(start.run).name)
    else:
        path = protocol.parent / start.file
        if Path(start.file).name != start.file or path.resolve().parent != protocol.parent:
            raise ValueError("starting_pulse.file must name a file beside the control spec")
        source = {"file": start.file}
    return {"path": path, "channels": channels, "source": source}, lineage


def channel_pulse(arrays: dict, channels: list[str]) -> NeutralPulse:
    """A neutral pulse's rows in the problem's channel order, matched by ID."""
    names = arrays["channels"].tolist()
    missing = [name for name in channels if name not in names]
    extra = [name for name in names if name not in channels]
    if missing or extra:
        raise ValueError(
            f"starting pulse channels differ from the problem's: missing {missing}, extra {extra}"
        )
    rows = [names.index(name) for name in channels]
    return NeutralPulse(
        np.asarray(arrays["times_ns"], dtype=float),
        np.asarray(arrays["amplitudes_ghz"], dtype=float)[rows],
    )


def _match_start(
    loaded: LoadedControl, start: dict, content: bytes
) -> tuple[NeutralPulse, bytes, dict]:
    """Refuse a start the constraints check would fail; it is never clipped or resampled."""
    from entanglix.catalog import _neutral_pulse
    from entanglix.corrector import load_policy
    from entanglix.corrector.control import constraints

    arrays = _neutral_pulse(io.BytesIO(content), start["channels"])
    pulse = channel_pulse(arrays, [channel.id for channel in loaded.definition.channels])
    failed = [
        item.get("diagnostic")
        or f"{item['name']} exceeds its limit by {item['residual']:.3g} "
        f"(tolerance {item['tolerance']:.3g})"
        for item in constraints(loaded, pulse, load_policy())["checks"]
        if item["outcome"] == "fail"
    ]
    if failed:
        raise ValueError(f"starting pulse does not fit the problem: {'; '.join(failed)}")
    if start["channels"] is not None:
        # A control run's pulse gains the channel IDs of its frozen spec.
        buffer = io.BytesIO()
        np.savez(buffer, **arrays)
        content = buffer.getvalue()
    digest = hashlib.sha256(content).hexdigest()
    source = start["source"]
    if "file" in source:
        source = {"file": source["file"], "sha256": digest}
    return pulse, content, {"sha256": digest, "source": source}


def _vary(document: dict, values: dict) -> None:
    """Apply investigation values to a parsed control input; its authored bytes stay unchanged."""
    from entanglix.protocols.sweep import _set_target

    for target, value in values.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"varied value must be finite: {target}")
        if target == "pulse.seed":
            if not isinstance(document.get("pulse"), dict) or "seed" not in document["pulse"]:
                raise ValueError("varied target must name a declared value: pulse.seed")
            if value < 0 or value != int(value):
                raise ValueError("pulse.seed must vary over non-negative integers")
            document["pulse"]["seed"] = int(value)
        elif target.startswith("parameters."):
            _set_target(document, target, value)
        else:
            raise ValueError(f"control varies only parameters.<name> and pulse.seed: {target}")


def _agreed(declared, derived, label):
    """Keep a declared value that agrees with the derived one; frozen inputs carry both."""
    if declared is None:
        return derived
    if not math.isclose(declared, derived, rel_tol=RESOLVED_RELATIVE_TOLERANCE, abs_tol=0):
        raise ValueError(f"{label}: declared {declared!r} disagrees with derived {derived!r}")
    return declared


def _resolve(spec: ControlSpec, system: ConstructedSystem) -> tuple[ControlSpec, dict]:
    """Resolve named transitions, following carriers and Rabi-rate bounds on the model."""
    ids = system.subsystem_ids or (system.id,)
    bases = system.provenance.get("subsystem_bases", {ids[0]: system.basis})
    data = spec.model_dump()
    frequencies = data["frame"]["frequencies_ghz"]
    resolved = {}
    for name, (lower, upper) in spec.frame.transitions.items():
        if name not in ids:
            raise ValueError(
                f"frame.transitions: unknown subsystem {name!r}; available: {list(ids)}"
            )
        if bases.get(name) != "isolated_eigenstates":
            raise ValueError(f"frame.transitions.{name}: named levels need isolated eigenstates")
        index = ids.index(name)
        # Local Hamiltonians precede interactions, in subsystem order.
        energies = np.diag(system.hamiltonian_terms[index].factors[index]).real
        if max(lower, upper) >= len(energies):
            raise ValueError(f"frame.transitions.{name}: {len(energies)} levels are retained")
        frequency = float(energies[upper] - energies[lower])
        if frequency <= 0:
            raise ValueError(f"frame.transitions.{name}: the first level must lie lower")
        frequencies[name] = _agreed(frequencies.get(name), frequency, f"frame.{name}")
        resolved.setdefault("transition_frequencies_ghz", {})[name] = frequency
    for channel in data["channels"]:
        label = f"channel {channel['id']!r}"
        if channel["follows"] is not None:
            if channel["follows"] not in frequencies:
                raise ValueError(
                    f"{label}: follows {channel['follows']!r}, which has no frame frequency"
                )
            # Followers use the frame's own number, so the RWA projection keeps them.
            _agreed(channel["frequency_ghz"], frequencies[channel["follows"]], label)
            channel["frequency_ghz"] = frequencies[channel["follows"]]
        if channel["frequency_ghz"] is None:
            raise ValueError(f"{label}: declare frequency_ghz or follows")
    rabi = spec.rabi_bounds
    if rabi is not None:
        if rabi.subsystem not in spec.frame.transitions:
            raise ValueError("rabi_bounds.subsystem must name a frame transition")
        lower, upper = spec.frame.transitions[rabi.subsystem]
        index = ids.index(rabi.subsystem)
        elements = {}
        for channel in spec.channels:
            factors = system.drive_factors.get(channel.drive, (system.drives[channel.drive],))
            if any(
                not np.array_equal(factor, np.eye(len(factor)))
                for position, factor in enumerate(factors)
                if position != index
            ):
                raise ValueError(
                    f"channel {channel.id!r}: a Rabi-rate bound needs drive {channel.drive!r} "
                    f"to act on {rabi.subsystem!r} alone"
                )
            # The cosine-carrier convention: Rabi rate = amplitude x |<lower|drive|upper>|.
            local = factors[index]
            element = float(abs(local[lower, upper]))
            if element <= 1e-12 * np.abs(local).max():
                raise ValueError(
                    f"channel {channel.id!r}: drive {channel.drive!r} has a zero "
                    f"{lower}-{upper} matrix element"
                )
            elements[channel.id] = element
        resolved["rabi_matrix_elements"] = elements
        if rabi.amplitude_ghz is not None:
            for channel in data["channels"]:
                channel["amplitude_bound_ghz"] = _agreed(
                    channel["amplitude_bound_ghz"],
                    rabi.amplitude_ghz / elements[channel["id"]],
                    f"channel {channel['id']!r} amplitude_bound_ghz",
                )
        for field, rate in (("du_bound", rabi.du_ghz_per_ns), ("ddu_bound", rabi.ddu_ghz_per_ns2)):
            if rate is None:
                continue
            bounds = {name: rate / element for name, element in elements.items()}
            if max(bounds.values()) > min(bounds.values()) * (1 + RESOLVED_RELATIVE_TOLERANCE):
                raise ValueError(
                    f"rabi_bounds: derived {field} differs across channels "
                    f"({', '.join(f'{name}={value!r}' for name, value in bounds.items())}); "
                    f"declare an explicit problem.{field}"
                )
            data["problem"][field] = _agreed(
                data["problem"][field], min(bounds.values()), f"problem.{field}"
            )
    for channel in data["channels"]:
        if channel["amplitude_bound_ghz"] is None:
            raise ValueError(
                f"channel {channel['id']!r}: declare amplitude_bound_ghz or rabi_bounds.amplitude_ghz"
            )
    if spec.frame.transitions or rabi is not None:
        resolved["frame_frequencies_ghz"] = frequencies
        resolved["carrier_frequencies_ghz"] = {
            c["id"]: c["frequency_ghz"] for c in data["channels"]
        }
        resolved["amplitude_bounds_ghz"] = {
            c["id"]: c["amplitude_bound_ghz"] for c in data["channels"]
        }
        resolved.update(
            {
                field: data["problem"][field]
                for field in ("du_bound", "ddu_bound")
                if data["problem"][field] is not None
            }
        )
    return ControlSpec.model_validate(data), resolved


def load_control(
    protocol: Path,
    *,
    values: dict[str, float] | None = None,
    runs_root=Path("research_runs"),
    replay_source=None,
) -> LoadedControl:
    """Validate a control input and build the matrices, without selecting a solver."""
    from entanglix.runner import _system_inputs

    protocol = protocol.resolve()
    content = protocol.read_bytes()
    document = tomllib.loads(content.decode("utf-8"))
    _vary(document, values or {})
    spec = ControlSpec.model_validate(document)
    source, catalog = spec.resolve_system(protocol, runs_root, replay_source)
    authored = source.read_bytes()
    original = load_system(source, content=with_parameters(authored, spec.parameters))
    if spec.evaluation is not None:
        # Only the names are checked here; the corrector builds each sampled model.
        override_system(authored, dict.fromkeys(spec.evaluation.quantities, 0.0))
    paths, hashes = _system_inputs(
        source, original, reserved={"control.toml", "problem.toml", STARTING_PULSE}
    )
    # Base digests and freezing checks use the authored bytes, before any override.
    hashes["system.toml"] = hashlib.sha256(authored).hexdigest()
    if protocol in {path.resolve() for path in paths.values()}:
        raise ValueError("control spec must be distinct from its system inputs")
    paths["control.toml"] = protocol
    hashes["control.toml"] = hashlib.sha256(content).hexdigest()
    if max(spec.goal.subspace[0]) > math.prod(original.dimensions):
        raise ValueError("goal.subspace exceeds the system dimension")
    for channel in spec.channels:
        if channel.drive not in original.drives:
            raise ValueError(
                f"channel {channel.id!r}: unknown drive {channel.drive!r}; "
                f"available: {sorted(original.drives)}"
            )
    spec, resolved = _resolve(spec, original)
    closed = replace(
        original,
        drives={channel.id: original.drives[channel.drive] for channel in spec.channels},
        drive_factors={
            channel.id: original.drive_factors[channel.drive]
            for channel in spec.channels
            if channel.drive in original.drive_factors
        },
        dissipators={},
        dissipator_factors={},
    )
    initial = np.zeros(math.prod(closed.dimensions), dtype=complex)
    initial[0] = 1
    request = EvolutionInput(
        initial,
        np.array([0.0, spec.pulse.T]),
        carriers={
            channel.id: Carrier(channel.frequency_ghz, channel.phase_rad)
            for channel in spec.channels
        },
        frame=RotatingFrame(**spec.frame.model_dump(exclude={"transitions"})),
    )
    prepared = prepare_dynamics(closed, normalize_input(closed, request))
    drift = matrix(prepared.hamiltonian)
    energy_origin_ghz = float(drift[0, 0].real)
    exported_drift = drift - energy_origin_ghz * np.eye(len(drift))
    drives = {name: matrix(terms) for name, terms in prepared.drives.items()}
    for name, operator in drives.items():
        if not np.any(operator):
            raise ValueError(
                f"channel {name!r}: exported drive is zero under the declared frame; "
                "check its carrier frequency and source operator"
            )
    if spec.rabi_bounds is not None:
        subsystem = spec.rabi_bounds.subsystem
        lower, upper = spec.frame.transitions[subsystem]
        ids = original.subsystem_ids or (original.id,)
        states = []
        for level in (lower, upper):
            state = [0] * len(original.dimensions)
            state[ids.index(subsystem)] = level
            states.append(np.ravel_multi_index(state, original.dimensions))
        first, second = states
        for name, element in resolved["rabi_matrix_elements"].items():
            # A transition kept by the RWA carries half its element under the cosine convention.
            retained = max(abs(drives[name][first, second]), abs(drives[name][second, first]))
            if not math.isclose(retained, element / 2, rel_tol=RESOLVED_RELATIVE_TOLERANCE):
                raise ValueError(
                    f"channel {name!r}: the exported drive does not retain the {lower}-{upper} "
                    f"transition of {subsystem!r}; check its carrier against the frame "
                    "and excitation numbers"
                )
    control_model = replace(
        closed,
        hamiltonian_ghz=drift,
        hamiltonian_terms=prepared.hamiltonian,
        drives=drives,
        drive_factors={},
        provenance={**closed.provenance, "control": prepared.diagnostics},
    )
    goal = spec.goal.model_dump(exclude_none=True)
    if spec.goal.matrix is not None:
        goal["matrix"] = _encoded(spec.goal.operator())
    problem = spec.problem.model_dump(exclude_none=True)
    step = spec.pulse.T / (spec.problem.N - 1)
    problem["free_dt"] = [
        step * (1 - STEP_RELATIVE_HALF_WIDTH),
        step * (1 + STEP_RELATIVE_HALF_WIDTH),
    ]
    problem["options"] = {"timesteps_all_equal": True}
    generated = {
        "schema_version": 1,
        "kind": "control",
        "system": {
            "kind": "raw",
            "params": {"drive_bounds": [channel.amplitude_bound_ghz for channel in spec.channels]},
            "H_drift": _encoded(2 * np.pi * exported_drift),
            "H_drives": [_encoded(2 * np.pi * drives[channel.id]) for channel in spec.channels],
        },
        "goal": goal,
        "pulse": spec.pulse.model_dump(exclude_none=True),
        "problem": problem,
    }
    data = tomli_w.dumps(generated).encode("utf-8")
    digest = hashlib.sha256(data).hexdigest()
    listing = {
        "protocol": "control",
        "system_id": original.id,
        "dimensions": list(original.dimensions),
        "channel_order": [channel.id for channel in spec.channels],
        "omitted_dissipators": sorted(original.dissipators),
        "frame": prepared.diagnostics["frame"],
        "energy_origin_ghz": energy_origin_ghz,
        "solver": spec.solver.model_dump(exclude_none=True),
        "duration_ns": spec.pulse.T,
        "interval_count": spec.problem.N - 1,
        "problem_sha256": digest,
        "amicode_revision": AMICODE_REVISION,
        **({"resolved": resolved} if resolved else {}),
    }
    return LoadedControl(
        spec, control_model, generated, data, digest, paths, hashes, listing, catalog
    )


def compile_control(
    protocol: Path,
    directory: Path,
    *,
    repository: Path | None = None,
    extras=(),
    retain_base=False,
    values=None,
    runs_root=Path("research_runs"),
    replay_source=None,
) -> dict:
    """Prepare immutable input bytes for the runner; this creates no attempt."""
    loaded = load_control(protocol, values=values, runs_root=runs_root, replay_source=replay_source)
    catalog = loaded.catalog
    start = {}
    if loaded.definition.starting_pulse is not None:
        # Only a launch reads the start; checks and their sampled models never do.
        located, lineage = _starting_source(
            loaded.definition.starting_pulse, protocol.resolve(), runs_root, replay_source
        )
        initial, start_bytes, start_record = _match_start(
            loaded, located, located["path"].read_bytes()
        )
        catalog = {name: [*ids, *lineage[name]] for name, ids in catalog.items()}
        start = {"initial_pulse": initial}
    selection = None
    runtime = None
    if repository is not None:
        from entanglix.solvers import select_solver

        backend, selection = select_solver(
            loaded.definition.solver.backend, loaded.definition.solver.fallback
        )
        capabilities = backend.capabilities()
        if start:
            # Support is read from the solver that will run, after any fallback.
            if capabilities.get("initial_pulse") is not True:
                raise ValueError(
                    f"solver {selection['actual']!r} does not support a starting pulse"
                )
            selection["initial_pulse"] = True
        if preflight := getattr(backend, "validate", None):
            runtime = preflight(loaded.problem_spec, loaded.definition.solver.options, **start)
        required = {"amplitude"}
        required.update(
            name
            for name, field in (("du", "du_bound"), ("ddu", "ddu_bound"))
            if field in loaded.problem_spec["problem"]
        )
        if "unitary" not in capabilities["problem_kinds"] or not required <= set(
            capabilities["constraints"]
        ):
            raise ValueError(
                f"solver {selection['actual']!r} does not support the declared CPU control problem and constraints"
            )
    frozen = read_inputs(loaded.inputs, loaded.input_digests, "control")
    base_inputs = loaded.input_digests if retain_base else {}
    if start and loaded.definition.starting_pulse.file is not None:
        # An authored file is a base input under its fixed name; entries and runs are not.
        frozen[STARTING_PULSE] = start_bytes
        if retain_base:
            base_inputs = {**base_inputs, STARTING_PULSE: start_record["sha256"]}
    base = base_files(frozen, loaded.definition.system, "control") if retain_base else {}
    definition = loaded.definition.frozen_document()
    if start:
        # The start is frozen beside the unchanged ProblemSpec, so reruns need no catalog.
        definition["starting_pulse"] = {"file": STARTING_PULSE}
        frozen[STARTING_PULSE] = start_bytes
    frozen["control.toml"] = tomli_w.dumps(definition).encode("utf-8")
    frozen["system.toml"] = with_parameters(frozen["system.toml"], loaded.definition.parameters)
    frozen["problem.toml"] = loaded.problem_bytes
    frozen.update(base)
    write_frozen(directory, frozen)
    result = {
        **catalog,
        "system_id": loaded.system.id,
        "inputs": {name: str(directory / name) for name in frozen},
        "spec_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in frozen.items()},
        "problem_sha256": loaded.digest,
        "preview": loaded.listing,
        "base_inputs": base_inputs,
        "scope_values": [values or {}],
        **({"starting_pulse": start_record} if start else {}),
    }
    if repository is not None:
        from entanglix.corrector.control import control_listing
        from entanglix.runner import capture_environment

        result.update(
            engine=selection["actual"],
            device="cpu",
            needs={"gradients": True, "device": "cpu"},
            environment=capture_environment(
                repository / "uv.lock", extras=tuple(extras), solver_runtime=runtime
            ),
            timeout_seconds=loaded.definition.timeout_seconds,
            kind="control",
            selection=selection,
            protocol_sha256=loaded.input_digests["control.toml"],
        )
        result["preview"]["solver_selection"] = selection
        result["preview"].update(control_listing(loaded))
    return result


def control(
    protocol: Path,
    *,
    runs_root: Path = Path("research_runs"),
    repository: Path | None = None,
    parent_run_ids=(),
    dry_run=False,
    investigation_id: str | None = None,
    base: str | None = None,
    values: dict[str, float] | None = None,
):
    """Solve a closed-system control problem as one owned CPU job."""
    from entanglix.protocols.simulate import _launch_protocol

    return _launch_protocol(
        protocol,
        operation="control",
        runs_root=runs_root,
        repository=repository,
        parent_run_ids=parent_run_ids,
        dry_run=dry_run,
        investigation_id=investigation_id,
        base=base,
        values=values,
    )

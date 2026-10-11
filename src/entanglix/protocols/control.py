"""Validated closed-system control models and shared ProblemSpec export."""

import hashlib
import math
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
from entanglix.protocols.plans import FrameSpec

AMICODE_REVISION = "cb717f57cb63fa292917188c8ca32f50ddcca539"
STEP_RELATIVE_HALF_WIDTH = 1e-7
Finite = Annotated[float, Field(allow_inf_nan=False)]
Nonnegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class Channel(StrictModel):
    id: str = Field(min_length=1)
    drive: str = Field(min_length=1)
    frequency_ghz: float = Field(gt=0, allow_inf_nan=False)
    phase_rad: Finite = 0.0
    amplitude_bound_ghz: Nonnegative


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
    seed: int | None = Field(default=None, strict=True, ge=0)


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


class ControlSpec(StrictModel):
    format_version: Literal[1]
    system: str
    frame: FrameSpec
    channels: list[Channel] = Field(min_length=1)
    goal: Goal
    pulse: Pulse
    problem: Problem
    solver: Solver
    timeout_seconds: float | None = Field(default=None, gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def check_channels(self):
        names = [channel.id for channel in self.channels]
        if len(set(names)) != len(names):
            raise ValueError("channel IDs must be unique")
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


def load_control(protocol: Path) -> LoadedControl:
    """Validate a control input and build the matrices, without selecting a solver."""
    from entanglix.runner import _system_inputs

    protocol = protocol.resolve()
    content = protocol.read_bytes()
    spec = ControlSpec.model_validate(tomllib.loads(content.decode("utf-8")))
    source = protocol.parent / spec.system
    if Path(spec.system).name != spec.system or source.resolve().parent != protocol.parent:
        raise ValueError("control.system must name a file beside the control spec")
    original = load_system(source)
    paths, hashes = _system_inputs(source, original, reserved={"control.toml", "problem.toml"})
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
        frame=RotatingFrame(**spec.frame.model_dump()),
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
    }
    return LoadedControl(spec, control_model, generated, data, digest, paths, hashes, listing)


def compile_control(
    protocol: Path,
    directory: Path,
    *,
    repository: Path | None = None,
    extras=(),
    retain_base=False,
) -> dict:
    """Prepare immutable input bytes for the runner; this creates no attempt."""
    loaded = load_control(protocol)
    selection = None
    runtime = None
    if repository is not None:
        from entanglix.solvers import select_solver

        backend, selection = select_solver(
            loaded.definition.solver.backend, loaded.definition.solver.fallback
        )
        if preflight := getattr(backend, "validate", None):
            runtime = preflight(loaded.problem_spec, loaded.definition.solver.options)
        required = {"amplitude"}
        required.update(
            name
            for name, field in (("du", "du_bound"), ("ddu", "ddu_bound"))
            if field in loaded.problem_spec["problem"]
        )
        capabilities = backend.capabilities()
        if "unitary" not in capabilities["problem_kinds"] or not required <= set(
            capabilities["constraints"]
        ):
            raise ValueError(
                f"solver {selection['actual']!r} does not support the declared CPU control problem and constraints"
            )
    frozen = {}
    for name, source in loaded.inputs.items():
        content = source.read_bytes()
        if hashlib.sha256(content).hexdigest() != loaded.input_digests[name]:
            raise ValueError(f"control input changed during compilation: {name}")
        frozen[name] = content
    base_inputs = loaded.input_digests if retain_base else {}
    base_files = {}
    if retain_base:
        # Admission and reruns use the authored bytes, before canonicalization.
        base_files = {f"base/{name}": content for name, content in frozen.items()}
        if loaded.definition.system != "system.toml":
            base_files[f"base/{loaded.definition.system}"] = frozen["system.toml"]
    # Use the runner's canonical system filename in the frozen control input.
    definition = loaded.definition.model_dump(exclude_none=True)
    definition["system"] = "system.toml"
    frozen["control.toml"] = tomli_w.dumps(definition).encode("utf-8")
    frozen["problem.toml"] = loaded.problem_bytes
    frozen.update(base_files)
    for name, content in frozen.items():
        (directory / name).parent.mkdir(parents=True, exist_ok=True)
        with (directory / name).open("xb") as target:
            target.write(content)
        (directory / name).chmod(0o444)
    result = {
        "system_id": loaded.system.id,
        "inputs": {name: str(directory / name) for name in frozen},
        "spec_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in frozen.items()},
        "problem_sha256": loaded.digest,
        "preview": loaded.listing,
        "base_inputs": base_inputs,
        "scope_values": [{}],
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
    )


def rerun_control(run: Path, *, repository: Path | None = None, runs_root: Path | None = None):
    """Re-solve retained control inputs as a new linked attempt under current terms."""
    from entanglix.protocols.simulate import _rerun_protocol

    return _rerun_protocol(run, operation="control", repository=repository, runs_root=runs_root)

"""Construct one refined CPU plan, keeping tensor order and eigenstate gauge."""

import shutil
import tomllib
from pathlib import Path

import numpy as np
import tomli_w

from entanglix.models import load_system
from entanglix.protocols.plans import load_plan


def _phases(original, refined, count):
    # Align independently diagonalized retained eigenstates using local operators.
    # A spanning forest allows symmetries that split the operator algebra into sectors.
    edges = []
    for old, new in zip(original, refined):
        for first in range(count):
            for second in range(first + 1, count):
                weight = min(abs(old[first, second]), abs(new[first, second]))
                if weight > 1e-10:
                    ratio = old[first, second] / new[first, second]
                    edges.append((-weight, first, second, ratio / abs(ratio)))
    phases = np.ones(count, dtype=complex)
    edges.sort(key=lambda edge: edge[:3])
    known = set()
    for root in range(count):
        if root in known:
            continue
        known.add(root)
        while True:
            edge = next(
                (edge for edge in edges if (edge[1] in known) != (edge[2] in known)),
                None,
            )
            if edge is None:
                break
            _, first, second, ratio = edge
            if first in known:
                phases[second] = phases[first] * ratio
                known.add(second)
            else:
                phases[first] = phases[second] / ratio
                known.add(first)
    return phases


def embed_state(system, refined, state):
    phases = np.array([1], dtype=complex)
    for axis, count in enumerate(system.dimensions):
        names = sorted(system.operators)
        if len(system.dimensions) == 1:
            original = [system.operators[name] for name in names]
            target = [refined.operators[name] for name in names]
        else:
            names = [name for name in names if name.startswith(system.subsystem_ids[axis] + ".")]
            original = [system.operator_factors[name][axis] for name in names]
            target = [refined.operator_factors[name][axis] for name in names]
        phases = np.kron(phases, _phases(original, target, count))
    indices = np.ravel_multi_index(
        np.unravel_index(np.arange(len(phases)), system.dimensions), refined.dimensions
    )
    state = np.asarray(state)
    if state.ndim == 1:
        result = np.zeros(np.prod(refined.dimensions), dtype=complex)
        result[indices] = state * phases
    else:
        result = np.zeros((np.prod(refined.dimensions),) * 2, dtype=complex)
        result[np.ix_(indices, indices)] = phases[:, None] * state * phases.conj()[None, :]
    return result


def embedding_matrix(system, refined):
    """Map the ordered source basis into the refined basis with aligned phases."""
    return np.column_stack(
        [embed_state(system, refined, state) for state in np.eye(np.prod(system.dimensions))]
    )


def prepare_refinement(loaded, directory: Path, policy):
    """Write a combined refinement; returned plan is validated before any solve."""
    for name, path in loaded.inputs.items():
        shutil.copyfile(path, directory / name)
    document = tomllib.loads((directory / "system.toml").read_text())
    model = document["model"]

    def refine(leaf, step):
        target = leaf.get("truncation", leaf["model"])
        target["retained_levels"] += step.retained_levels_step
        for name, increment in step.cutoff_steps.items():
            if name == "cutoff":
                target["cutoff"] += increment
            else:
                target["cutoffs"][name] += increment

    if model["kind"] == "composite":
        for index, item in enumerate(model["subsystems"]):
            step = loaded.definition.refinement.get(item["id"])
            if step is None:
                continue
            if "source" in item:
                leaf = tomllib.loads((directory / item["source"]).read_text())
                refine(leaf, step)
                # Two subsystems may share a file but request different refinements.
                name = f"refined-subsystem-{index}.toml"
                while (directory / name).exists():
                    name = "refined-" + name
                (directory / name).write_text(tomli_w.dumps(leaf))
                item["source"] = name
            else:
                refine({"model": item}, step)
    elif loaded.definition.refinement:
        refine(document, loaded.definition.refinement[document["id"]])
    (directory / "system.toml").write_text(tomli_w.dumps(document))
    refined = load_system(directory / "system.toml")
    plan = loaded.definition.model_dump(exclude_none=True)
    if loaded.request is not None:
        plan["engine"] = plan["engine"] if plan["device"] == "cpu" else "qutip-cpu"
        plan["device"] = "cpu"
        evolution = plan["evolution"]
        factor = policy.thresholds["refinement"]["tolerance_factor"]
        evolution["rtol"] = min(evolution["rtol"] * factor, policy.thresholds["refinement"]["rtol"])
        evolution["atol"] = min(evolution["atol"] * factor, policy.thresholds["refinement"]["atol"])
        # Retain the numerical initial state in the same ordered low-level subspace.
        name = "reference-state.npz"
        while (directory / name).exists():
            name = "refined-" + name
        np.savez(
            directory / name,
            initial_state=embed_state(loaded.system, refined, loaded.request.initial_state)
            if loaded.definition.refinement
            else loaded.request.initial_state,
        )
        evolution["array_file"] = name
        evolution["initial_state"] = "initial_state"
    (directory / "plan.toml").write_text(tomli_w.dumps(plan))
    return load_plan(directory / "system.toml", directory / "plan.toml")

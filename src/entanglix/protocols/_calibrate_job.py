"""The calibration fit, using the common launch gate and inherited start barrier."""

import os
from pathlib import Path

from entanglix.calibration import UNEXPLAINED, UNRESOLVED, calibrate_problem
from entanglix.protocols.calibrate import ENGINE, NUISANCE, load_calibration, prepare, tolerances
from entanglix.runner import _digest, _load_record, publish_toml


def execute(spec: Path):
    run = Path(os.environ["ENTANGLIX_RUN_DIR"])
    launch = _load_record(run / "run.toml")["launch"]
    if _digest(spec.read_bytes()) != launch["calibration_sha256"]:
        raise ValueError("frozen calibration spec digest changed")
    loaded = load_calibration(spec)
    problem, record = prepare(loaded)
    starts = [[start[name] for name in loaded.names] for start in loaded.spec.starts]
    found = calibrate_problem(problem, starts)
    labels = [member.label for member in loaded.members]
    names = {UNEXPLAINED: "unexplained", UNRESOLVED: "unresolved"}
    peaks = problem.peaks
    assignment = {
        "columns": record["columns"],
        "peaks": [
            {
                "column": int(peaks.column[i]),
                "centre_ghz": float(peaks.centre[i]),
                "uncertainty_ghz": float(peaks.uncertainty[i]),
                **({"linewidth_ghz": float(w)} if (w := peaks.linewidth[i]) == w else {}),
                **({"amplitude": float(a)} if (a := peaks.amplitude[i]) == a else {}),
                **({"label": labels[peaks.line[i]]} if loaded.labelled else {}),
            }
            for i in range(peaks.centre.size)
        ],
        "extraction_failures": record["failures"],
        "solutions": [
            {
                "assignment": [names.get(j, labels[j]) for j in solution.pop("assignment")],
                "missing": solution.pop("missing"),
            }
            for solution in found["solutions"]
        ],
    }
    output = run / "artifacts/assignment.toml"
    publish_toml(output, assignment)
    axis = loaded.spec.axis_map
    publish_toml(
        run / "result.toml",
        {
            "kind": "calibrate",
            "engine": ENGINE,
            "device": "cpu",
            "calibration_sha256": launch["calibration_sha256"],
            "outcome": found["outcome"],
            # Every run result has a metrics table; a calibration's statistics are per solution.
            "metrics": {},
            "procedure": loaded.listing["procedure"],
            "parameters": {
                "fitted": list(loaded.spec.parameters),
                **({"nuisance": list(NUISANCE), "flux": axis.flux} if axis else {}),
            },
            "tolerances": tolerances(loaded.spec),
            "datasets": record["datasets"],
            "starts": found["starts"],
            "solutions": found["solutions"],
            "artifacts": {
                "assignment": {
                    "path": str(output.relative_to(run)),
                    "sha256": _digest(output.read_bytes()),
                }
            },
        },
    )


if __name__ == "__main__":
    from entanglix.protocols._simulate_job import main

    raise SystemExit(main("calibrate"))

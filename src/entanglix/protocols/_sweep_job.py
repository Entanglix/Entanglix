"""Serial sweep gate and worker behind the runner's inherited start barrier."""

import os
from pathlib import Path

import numpy as np

from entanglix.corrector.metrics import metric_values
from entanglix.protocols._plan_job import compute
from entanglix.protocols.sweep import load_sweep
from entanglix.runner import _digest, _load_record, publish_toml


def draw_metric(axes, grid_axes, values, name):
    grid_axes = [axis for axis in grid_axes if len(axis.values) > 1]
    values = values.squeeze()
    if len(grid_axes) == 1:
        axes.plot(grid_axes[0].values, values, "o-")
        axes.set(xlabel=grid_axes[0].id, ylabel=name)
    elif len(grid_axes) == 2:
        image = axes.pcolormesh(
            grid_axes[0].values, grid_axes[1].values, values.T, shading="nearest"
        )
        axes.set(xlabel=grid_axes[0].id, ylabel=grid_axes[1].id)
        axes.figure.colorbar(image, ax=axes, label=name)
    else:
        axes.plot(values.ravel(), "o-")
        axes.set(xlabel="Point index (last axis varies fastest)", ylabel=name)


def execute(path):
    run = Path(os.environ["ENTANGLIX_RUN_DIR"])
    loaded = load_sweep(path.parent)
    if loaded.digest != _load_record(run / "run.toml")["launch"]["plan_sha256"]:
        raise ValueError("frozen sweep digest changed")
    artifacts, rows = {}, []
    for index, point in enumerate(loaded.points):
        arrays, details = compute(point)
        target = run / "artifacts/points" / f"{index:06d}"
        target.mkdir(parents=True)
        temporary = target / "output.tmp.npz"
        np.savez(temporary, **arrays)
        output = target / "output.npz"
        temporary.replace(output)
        metrics = metric_values(point.definition.metrics, arrays)
        record = {
            "index": index,
            "coordinate": loaded.manifest["points"][index]["coordinate"],
            "plan_sha256": point.digest,
            "metrics": metrics,
            "dimensions": list(point.system.dimensions),
            "device": point.definition.device,
            "output_sha256": _digest(output.read_bytes()),
            **details,
        }
        publish_toml(target / "result.toml", record)
        for name in ("output.npz", "result.toml"):
            file = target / name
            artifacts[f"point_{index}_{name.split('.')[0]}"] = {
                "path": str(file.relative_to(run)),
                "sha256": _digest(file.read_bytes()),
            }
        rows.append(metrics)
        print(f"point {index + 1}/{len(loaded.points)} completed: {metrics}", flush=True)
    grids = {
        name: np.array([row[name] for row in rows]).reshape(loaded.manifest["shape"])
        for name in rows[0]
    }
    output = run / "artifacts/output.npz"
    np.savez(output, **grids)
    artifacts["output"] = {
        "path": str(output.relative_to(run)),
        "sha256": _digest(output.read_bytes()),
    }
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for plot_index, (name, values) in enumerate(grids.items()):
        figure, axes = plt.subplots()
        draw_metric(axes, loaded.definition.axes, values, name)
        figure.tight_layout()
        plot = run / "artifacts" / f"metric-{plot_index}.png"
        figure.savefig(plot)
        plt.close(figure)
        artifacts[f"plot_{name}"] = {
            "path": str(plot.relative_to(run)),
            "sha256": _digest(plot.read_bytes()),
        }
    publish_toml(
        run / "result.toml",
        {
            "kind": loaded.manifest["kind"],
            "engine": loaded.points[0].definition.engine,
            "device": loaded.points[0].definition.device,
            "dimensions": list(loaded.system.dimensions),
            "plan_sha256": loaded.digest,
            "shape": loaded.manifest["shape"],
            "metrics": {name: values.tolist() for name, values in grids.items()},
            "summary": {
                name: {
                    "minimum": float(values.min()),
                    "maximum": float(values.max()),
                    "minimum_point": int(values.argmin()),
                    "maximum_point": int(values.argmax()),
                }
                for name, values in grids.items()
            },
            "artifacts": artifacts,
        },
        replace=True,
    )


if __name__ == "__main__":
    from entanglix.protocols._simulate_job import main

    raise SystemExit(main("sweep"))

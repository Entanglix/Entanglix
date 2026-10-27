"""Shared source declaration, system overrides and input freezing for built-in protocols."""

import hashlib
import math
import tomllib
from pathlib import Path

import tomli_w
from pydantic import Field, model_validator

from entanglix.models import StrictModel


class SystemSource(StrictModel):
    system: str | None = Field(default=None, min_length=1)
    system_entry: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def one_system_source(self):
        if (self.system is None) == (self.system_entry is None):
            raise ValueError("declare exactly one of system or system_entry")
        return self

    def resolve_system(self, protocol: Path, runs_root: Path, replay_source: Path | None = None):
        from entanglix.catalog import device_source
        from entanglix.runner import _load_record

        if self.system_entry is not None and replay_source is None:
            return device_source(self.system_entry, runs_root)
        name = "system.toml" if self.system_entry is not None else self.system
        source = protocol.parent / name
        if Path(name).name != name or source.resolve().parent != protocol.parent:
            raise ValueError("system must name a file beside the protocol spec")
        inherited = (
            _load_record(replay_source / "run.toml").get("source_entry_ids", [])
            if replay_source
            else []
        )
        return source, {"source_entry_ids": inherited, "parent_run_ids": []}

    def frozen_document(self) -> dict:
        """The spec as frozen: the runner's canonical system filename and no catalog entry."""
        document = self.model_dump(exclude_none=True)
        document.pop("system_entry", None)
        document["system"] = "system.toml"
        return document


def override_system(content: bytes, changes: dict[str, float]) -> bytes:
    """Set declared parameters and circuit offset charges, named by evaluation target."""
    if not changes:
        return content
    document = tomllib.loads(content.decode("utf-8"))
    for target, value in changes.items():
        if target.startswith("parameters."):
            name = target.removeprefix("parameters.")
            if name not in document.get("parameters", {}) or not math.isfinite(value):
                raise ValueError(
                    f"parameter override must name a finite declared parameter: {name}"
                )
            document["parameters"][name]["value"] = value
        else:
            node = target.removeprefix("model.offset_charges.")
            charges = document["model"].get("offset_charges", {})
            if node not in charges or not math.isfinite(value):
                raise ValueError(
                    f"offset charge override must name a finite declared circuit node: {node}"
                )
            charges[node] = value
    return tomli_w.dumps(document).encode("utf-8")


def with_parameters(content: bytes, parameters: dict[str, float]) -> bytes:
    """Override declared system parameters; the result is the frozen system."""
    return override_system(
        content, {f"parameters.{name}": value for name, value in parameters.items()}
    )


def read_inputs(inputs: dict[str, Path], digests: dict[str, str], kind: str) -> dict[str, bytes]:
    """Read each loaded input once, refusing one that changed since it was loaded."""
    frozen = {}
    for name, source in inputs.items():
        data = source.read_bytes()
        if hashlib.sha256(data).hexdigest() != digests[name]:
            raise ValueError(f"{kind} input changed during compilation: {name}")
        frozen[name] = data
    return frozen


def base_files(authored: dict[str, bytes], system: str | None, kind: str) -> dict[str, bytes]:
    """Authored inputs under base/, with the system file also under its authored name."""
    # Admission and reruns use the authored bytes, before canonicalization.
    files = {f"base/{name}": data for name, data in authored.items()}
    if system is not None and system != "system.toml":
        alias = f"base/{system}"
        if alias in files:
            raise ValueError(
                f"the system file {system!r} would replace the {kind} input of that name "
                "in the investigation base; rename the system file"
            )
        files[alias] = authored["system.toml"]
    return files


def write_frozen(directory: Path, files: dict[str, bytes]) -> None:
    """Write each frozen input once, read-only."""
    for name, data in files.items():
        (directory / name).parent.mkdir(parents=True, exist_ok=True)
        with (directory / name).open("xb") as target:
            target.write(data)
        (directory / name).chmod(0o444)

"""Shared source declaration for built-in protocols."""

from pathlib import Path

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

"""Scripted CPU threshold characterization; draft first, then wait for user start.

Completed work is recovered from investigation-linked run records. The workspace
holds the five base inputs and a locator, not a second progress ledger.
"""

import argparse
import copy
import hashlib
import json
import shutil
import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import tomli_w

from entanglix.corrector import policy_digest
from entanglix.corrector.reference import embed_state
from entanglix.corrector.verification import check_run
from entanglix.investigations import (
    create_investigation,
    read_investigation,
    record_finding,
    stop_investigation,
)
from entanglix.models import load_system
from entanglix.protocols.simulate import rerun_simulation, simulate
from entanglix.protocols.sweep import rerun_sweep, sweep
from entanglix.runner import read_run

EXAMPLE = Path(__file__).resolve().parent
TARGET = 0.01
WIDTH = 0.000125
TIGHT_WIDTH = 0.000015625
FREQUENCIES = {12.0: 7.0017536727417085, 13.5: 7.002206406997239, 15.0: 7.002760675881172}
AMPLITUDE = "evolution.drive_amplitudes_ghz.readout"
CARRIER = "evolution.carriers.readout.frequency_ghz"
FRAMES = [f"evolution.frame.frequencies_ghz.{name}" for name in ("q", "r")]
LIMITS = [
    "One crossing assumes decreasing coherence; only sampled amplitudes are checked.",
    "Bracket width describes the RWA crossing; lab confirmation is separate.",
    "Coarse/refined differences are convergence evidence, not an error bound.",
    "No check bounds whole-window or accumulated RWA error.",
]


def draft(workspace, *, runs_root, case, checkpoint, job_count=None):
    """Prepare fixed bases and propose terms. This never creates a start decision."""
    workspace = workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=False)
    model = tomllib.loads((EXAMPLE / "system.toml").read_text())
    protocol = tomllib.loads((EXAMPLE / "readout.toml").read_text())
    protocol["needs"]["device"] = "cpu"
    protocol["timeout_seconds"] = 180
    bases = []
    for name, ej in [
        ("rwa", None),
        ("lab", None),
        *[(f"refined-{ej:g}".replace(".", "_"), ej) for ej in FREQUENCIES],
    ]:
        directory = workspace / name
        directory.mkdir()
        document, spec = copy.deepcopy(model), copy.deepcopy(protocol)
        shutil.copyfile(EXAMPLE / "readout-state.npz", directory / "readout-state.npz")
        if name != "rwa":
            spec["evolution"].pop("frame")
        if ej is not None:
            document["parameters"]["EJ"]["value"] = ej
            (directory / "system.toml").write_text(tomli_w.dumps(document))
            source = load_system(directory / "system.toml")
            document["model"]["subsystems"][0].update(retained_levels=6, cutoff=32)
            document["model"]["subsystems"][1]["retained_levels"] = 8
            (directory / "system.toml").write_text(tomli_w.dumps(document))
            with np.load(EXAMPLE / "readout-state.npz", allow_pickle=False) as archive:
                state = embed_state(source, load_system(directory / "system.toml"), archive["psi"])
            np.savez(directory / "readout-state.npz", psi=state)
            spec["parameters"]["EJ"] = ej
            spec["evolution"]["carriers"]["readout"]["frequency_ghz"] = FREQUENCIES[ej]
            spec["evolution"].update(rtol=1e-11, atol=1e-13)
        (directory / "system.toml").write_text(tomli_w.dumps(document))
        (directory / "simulate.toml").write_text(tomli_w.dumps(spec))
        variations = [{"target": AMPLITUDE, "range": [0.0, 0.002]}]
        if ej is None:
            variations.append(
                {
                    "target": "parameters.EJ",
                    "values": list(FREQUENCIES),
                    "linked_values": {
                        target: list(FREQUENCIES.values())
                        for target in [CARRIER, *(FRAMES if name == "rwa" else [])]
                    },
                }
            )
        bases.append(
            {
                "name": name,
                "inputs": {
                    path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in directory.iterdir()
                },
                "variations": variations,
            }
        )
    seed = tomllib.loads((EXAMPLE / "dephasing.toml").read_text())
    seed["base"] = "simulate.toml"
    (workspace / "rwa/sweep.toml").write_text(tomli_w.dumps(seed))
    terms = {
        "format_version": 1,
        "question": (
            "Which amplitude adds endpoint-coherence loss 0.01 at 100 ns for each EJ? "
            f"Establish RWA brackets of width {WIDTH:g} GHz with lab-frame confirmation."
        ),
        "assumptions": [*LIMITS, f"Case: {case}; normal results precede tighter requests."],
        "scope": {"operations": ["simulate", "sweep", "check"], "bases": bases},
        "evidence_requirements": {
            "requirements": [
                "Seed integrity, diagnostics and sampled decreasing expectation.",
                "Final RWA ends: integrity, segments, diagnostics; refined lab runs: integrity.",
                "Lab confirmation at zero and both ends; target margins exceed observed refinement changes.",
                "No crossing: propose wider scope; failed confirmation: retain best bracket and hand off.",
                *(
                    [
                        f"After establishing every normal-width result, request RWA brackets of width {TIGHT_WIDTH:g} GHz "
                        "in ascending EJ order; at the first failed lab confirmation, open a precision review "
                        "and leave later tighter cases unattempted."
                    ]
                    if case == "early-stop"
                    else []
                ),
            ],
            "checks": {
                **{f"refined-{ej:g}".replace(".", "_"): ["integrity"] for ej in FREQUENCIES},
            },
        },
        "checkpoint": {"finished_attempts": checkpoint},
        "review_criteria": ["Failed lab confirmation opens precision review; no lab-frame search."],
        "stop_criteria": ["Declared evidence complete, or first lab confirmation failure."],
        "limits": {} if job_count is None else {"job_count": job_count},
    }
    record = create_investigation(terms, runs_root=runs_root)
    locator = {"investigation": record["path"], "case": case}
    (workspace / "client.json").write_text(json.dumps(locator, indent=2) + "\n")
    return record


class Client:
    def __init__(self, workspace, *, repository=None):
        self.workspace = workspace.resolve()
        self.repository = repository
        locator = json.loads((self.workspace / "client.json").read_text())
        self.path = Path(locator["investigation"])
        self.case = locator["case"]
        self.runs_root = self.path.parent.parent
        self.samples, self.records = {}, {}
        self.failed = {}
        self.seed_run = None
        for manifest in sorted(self.runs_root.glob("*/*/run.toml")):
            data = tomllib.loads(manifest.read_text())
            if data.get("investigation_id") != self.path.name:
                continue
            record = read_run(manifest.parent)
            if record["status"] in {"active", "unknown"} or (
                record["status"] == "interrupted" and "reconciliation" not in record
            ):
                raise RuntimeError(
                    f"Settle outstanding attempt before continuing: {manifest.parent}"
                )
            self.remember(manifest.parent, record)

    def remember(self, path, record=None):
        record = record or read_run(path)
        self.records[path.name] = (path, record)
        manifest = record["manifest"]
        if manifest["protocol"] not in {"simulate", "sweep"}:
            return
        key = (
            manifest["investigation_base"],
            tuple(sorted(manifest["launch"].get("varied_values", {}).items())),
        )
        if record["status"] != "completed":
            self.failed[key] = path
            return
        metrics = record["result"]["metrics"]["coherence"]
        if manifest["protocol"] == "sweep":
            self.seed_run = path
            for row, ej in enumerate(FREQUENCIES):
                for column, amplitude in enumerate((0.0, 0.001, 0.002)):
                    self.samples[ej, amplitude, "rwa"] = {
                        "amplitude_ghz": amplitude,
                        "coherence": metrics[row][column],
                        "source_run_id": path.name,
                        "point_index": row * 3 + column,
                    }
        else:
            spec = tomllib.loads((path / "spec/simulate.toml").read_text())
            ej = spec["parameters"]["EJ"]
            amplitude = spec["evolution"]["drive_amplitudes_ghz"]["readout"]
            profile = manifest["investigation_base"]
            profile = "lab_refined" if profile.startswith("refined-") else profile
            self.samples[ej, amplitude, profile] = {
                "amplitude_ghz": amplitude,
                "coherence": metrics,
                "source_run_id": path.name,
            }

    def finding(
        self, learned, *, kind="assessment", decision="continue", unresolved=LIMITS, review=None
    ):
        text = json.dumps(learned, sort_keys=True)
        if kind == "interpretation":
            record = read_investigation(self.path)
            for event in reversed(record["events"]):
                if (
                    event["kind"] == "finding"
                    and event["revision"] == record["active_revision"]
                    and event["finding"]["kind"] == kind
                    and event["finding"]["learned"] == text
                ):
                    return event
        return record_finding(
            self.path,
            {
                "kind": kind,
                "runs": list(self.records),
                "learned": text,
                "unresolved": list(unresolved),
                "decision": decision,
                "rationale": (
                    "Lab confirmation does not straddle the target beyond the observed refinement change. "
                    "Further RWA bisection cannot support this claim; unused allowance does not change the method limit."
                    if review
                    else "The reported lab margins support the RWA bracket under the stated convergence limitations."
                    if kind in {"report", "interpretation"}
                    else "Continue the declared midpoint search and pending lab confirmations using the retained source values. "
                    "Local checks do not establish whole-window or RWA accuracy."
                ),
                "next_experiment": "Lab-frame search needs a user decision."
                if review
                else "Continue within the declared range and evidence plan."
                if decision == "continue"
                else "Ask for a wider range if crossings are needed at EJ="
                + ", ".join(f"{item['ej_ghz']:g}" for item in learned["no_crossing"])
                + " GHz."
                if learned.get("no_crossing")
                else "No further experiment is required by this policy.",
                **({"review_name": review} if review else {}),
            },
        )

    def before_job(self):
        record = read_investigation(self.path)
        if record["state"] != "active" or record["open_reviews"]:
            raise RuntimeError(
                f"User decision needed: state={record['state']}, reviews={record['open_reviews']}"
            )
        if record["checkpoint"]["due"]:
            self.finding(
                {
                    "samples": [
                        {"ej_ghz": ej, "profile": profile, **sample}
                        for (ej, _, profile), sample in self.samples.items()
                    ]
                }
            )

    def completed(self, job):
        path = Path(job["path"])
        self.remember(path)
        if job["exit_code"]:
            raise RuntimeError(f"Job did not complete: {path}; exit={job['exit_code']}")
        return path

    def checked(self, source, names):
        history = read_run(source).get("verification", [])
        for item in history:
            if (
                item["status"] == "completed"
                and item.get("source_matches")
                and item["policy_version"] == policy_digest()
                and [entry["name"] for entry in item["checks"]] == list(names)
            ):
                report = item["checks"]
                break
        else:
            self.before_job()
            path = self.completed(check_run(source, names, repository=self.repository))
            report = read_run(path)["result"]["report"]["checks"]
        if any(entry.get("outcome") in {"fail", "inapplicable"} for entry in report):
            raise RuntimeError(f"Requested evidence did not pass: {source}; {report}")

    def seed(self):
        if self.seed_run is None:
            self.before_job()
            previous = self.failed.get(("rwa", ()))
            job = (
                rerun_sweep(previous, repository=self.repository)
                if previous
                else sweep(
                    self.workspace / "rwa/sweep.toml",
                    runs_root=self.runs_root,
                    repository=self.repository,
                    investigation_id=self.path.name,
                    base="rwa",
                )
            )
            self.completed(job)
        self.checked(self.seed_run, ("integrity", "diagnostics", "declared_expectation"))

    def solve(self, ej, amplitude, profile, parent):
        if (ej, amplitude, profile) not in self.samples:
            base = f"refined-{ej:g}".replace(".", "_") if profile == "lab_refined" else profile
            values = {AMPLITUDE: amplitude}
            if profile != "lab_refined":
                values.update({"parameters.EJ": ej, CARRIER: FREQUENCIES[ej]})
                if profile == "rwa":
                    values.update(dict.fromkeys(FRAMES, FREQUENCIES[ej]))
            self.before_job()
            previous = self.failed.get((base, tuple(sorted(values.items()))))
            job = (
                rerun_simulation(previous, repository=self.repository)
                if previous
                else simulate(
                    self.workspace / base / "simulate.toml",
                    runs_root=self.runs_root,
                    repository=self.repository,
                    investigation_id=self.path.name,
                    base=base,
                    values=values,
                    parent_run_ids=(parent,),
                )
            )
            self.completed(job)
        sample = self.samples[ej, amplitude, profile]
        if profile == "lab_refined":
            self.checked(self.records[sample["source_run_id"]][0], ("integrity",))
        return sample

    def bisect(self, ej, bracket, width):
        baseline = self.samples[ej, 0.0, "rwa"]["coherence"]
        lower, upper = bracket
        while upper["amplitude_ghz"] - lower["amplitude_ghz"] > width * (1 + 1e-12):
            midpoint = (lower["amplitude_ghz"] + upper["amplitude_ghz"]) / 2
            sample = self.solve(ej, midpoint, "rwa", upper["source_run_id"])
            if baseline - sample["coherence"] < TARGET:
                lower = sample
            else:
                upper = sample
        return lower, upper

    def compare(self, ej, bracket, stage):
        for sample in bracket:
            self.checked(
                self.records[sample["source_run_id"]][0], ("integrity", "segments", "diagnostics")
            )
        rows = []
        for amplitude in (0.0, *(sample["amplitude_ghz"] for sample in bracket)):
            rwa = self.samples[ej, amplitude, "rwa"]
            lab = self.solve(ej, amplitude, "lab", rwa["source_run_id"])
            refined = self.solve(ej, amplitude, "lab_refined", lab["source_run_id"])
            rows.append({"rwa": rwa, "lab": lab, "lab_refined": refined})
        endpoints = []
        for row in rows[1:]:
            losses = {
                name: rows[0][name]["coherence"] - sample["coherence"]
                for name, sample in row.items()
            }
            endpoints.append(
                {
                    "amplitude_ghz": row["rwa"]["amplitude_ghz"],
                    "coherences": {name: sample["coherence"] for name, sample in row.items()},
                    "source_runs": {name: sample["source_run_id"] for name, sample in row.items()},
                    "additional_loss": losses,
                    "observed_lab_refinement_change": abs(losses["lab_refined"] - losses["lab"]),
                }
            )
        margins = [row["additional_loss"]["lab_refined"] - TARGET for row in endpoints]
        changes = [row["observed_lab_refinement_change"] for row in endpoints]
        return {
            "ej_ghz": ej,
            "stage": stage,
            "width_ghz": bracket[1]["amplitude_ghz"] - bracket[0]["amplitude_ghz"],
            "baselines": rows[0],
            "endpoints": endpoints,
            "target_margins": margins,
            "confirmed": margins[0] < -changes[0] and margins[1] > changes[1],
        }

    def run(self):
        record = read_investigation(self.path)
        reports = [
            event
            for event in record["events"]
            if event["kind"] == "finding" and event["finding"]["kind"] == "report"
        ]
        if reports:
            result = json.loads(reports[-1]["finding"]["learned"])
            if record["state"] == "active":
                stop_investigation(
                    self.path, result["outcome"], note="Retained report closes the client policy."
                )
            return result
        self.before_job()
        self.seed()
        result = {
            "case": self.case,
            "brackets": [],
            "no_crossing": [],
            "unattempted": [],
            "limits": LIMITS,
        }
        brackets = {}
        for ej in FREQUENCIES:
            baseline = self.samples[ej, 0.0, "rwa"]["coherence"]
            samples = [self.samples[ej, a, "rwa"] for a in (0.0, 0.001, 0.002)]
            interval = next(
                (
                    (a, b)
                    for a, b in zip(samples, samples[1:])
                    if baseline - a["coherence"] < TARGET <= baseline - b["coherence"]
                ),
                None,
            )
            if interval is None:
                result["no_crossing"].append(
                    {
                        "ej_ghz": ej,
                        "samples": samples,
                        "additional_losses": [baseline - sample["coherence"] for sample in samples],
                        "proposal": "Ask for wider scope; do not extend the range.",
                    }
                )
                continue
            brackets[ej] = self.bisect(ej, interval, WIDTH)
            item = self.compare(ej, brackets[ej], "normal")
            result["brackets"].append(item)
            if item["confirmed"]:
                self.finding(item, kind="interpretation")
            if not item["confirmed"]:
                result["unattempted"] = [
                    {"ej_ghz": later, "stage": "normal"} for later in FREQUENCIES if later > ej
                ]
                break
        if all(item["confirmed"] for item in result["brackets"]) and self.case == "early-stop":
            for index, ej in enumerate(brackets):
                tighter = self.bisect(ej, brackets[ej], TIGHT_WIDTH)
                item = self.compare(ej, tighter, "tighter")
                result["brackets"].append(item)
                if not item["confirmed"]:
                    result["unattempted"] = [
                        {"ej_ghz": later, "stage": "tighter"}
                        for later in list(brackets)[index + 1 :]
                    ]
                    break
        handoff = any(not item["confirmed"] for item in result["brackets"])
        record = read_investigation(self.path)
        result.update(
            investigation_id=self.path.name,
            usage=record["usage"],
            elapsed_seconds=(
                datetime.now(UTC)
                - next(
                    event["recorded_at"]
                    for event in record["events"]
                    if event["kind"] == "decision" and event["action"] == "start"
                )
            ).total_seconds(),
            user_decisions_after_start=sum(
                event["kind"] == "decision" and event["action"] != "start"
                for event in record["events"]
            ),
            outcome="review_handoff" if handoff else "evidence_completion",
        )
        self.finding(
            result,
            kind="report",
            decision="review" if handoff else "stop",
            review="precision" if handoff else None,
            unresolved=[
                *LIMITS,
                *[
                    f"EJ={item['ej_ghz']:g} crossing needs wider scope."
                    for item in result["no_crossing"]
                ],
                *(
                    ["Tighter lab-frame claim unresolved; propose lab-frame search."]
                    if handoff
                    else []
                ),
            ],
        )
        stop_investigation(
            self.path,
            result["outcome"],
            note="Scripted policy completed; see the source-linked report.",
        )
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser(
        "draft", help="prepare bases and propose a revision; never start it"
    )
    prepare.add_argument("workspace", type=Path)
    prepare.add_argument("--runs-root", type=Path, required=True)
    prepare.add_argument("--case", choices=("normal", "early-stop"), required=True)
    prepare.add_argument("--checkpoint", type=int, required=True)
    prepare.add_argument("--job-count", type=int)
    execute = commands.add_parser("run", help="continue a user-started investigation")
    execute.add_argument("workspace", type=Path)
    args = parser.parse_args()
    if args.command == "draft":
        result = draft(
            args.workspace,
            runs_root=args.runs_root,
            case=args.case,
            checkpoint=args.checkpoint,
            job_count=args.job_count,
        )
    else:
        client = Client(args.workspace)
        try:
            result = client.run()
        except Exception:
            record = read_investigation(client.path)
            if record["state"] == "active" and not record["open_reviews"]:
                raise
            print(
                f"User decision needed: state={record['state']}, reviews={record['open_reviews']}; "
                "continue with the same run command after the operator decision.",
                file=sys.stderr,
            )
            raise SystemExit(1) from None
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()

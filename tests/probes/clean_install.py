"""Clean installs on the workstation, isolated from developer paths and variables.

Each world gets a fresh HOME and a PATH of linked system tools only, so the developer's uv,
Julia, environments and caches cannot stand in for missing pieces. The installer clones a
local fixture repository built from this worktree; that override is the only added variable.
Run: python tests/probes/clean_install.py --work DIR --output FILE [--worlds cpu,gpu].
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import tomllib
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
TOOLS = (
    "bash sh env git curl tar gzip sha256sum uname mkdir rm mv cp grep sed sort tail head "
    "dirname basename cat ls readlink sleep id flock"
).split()
CLIENT = """
import asyncio, json, os, sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    command, arguments = sys.argv[1], json.loads(sys.argv[2])
    params = StdioServerParameters(command=command, args=arguments, env=dict(os.environ))
    async with stdio_client(params) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()
            def text(result):
                assert not result.is_error, result.content[0].text
                return json.loads(result.content[0].text)
            out = {"capabilities": text(await session.call_tool("list_capabilities", {}))}
            if len(sys.argv) > 3:
                launch = text(await session.call_tool("simulate", {"protocol": sys.argv[3]}))
                out["run"] = text(await session.call_tool(
                    "read_run", {"run": launch["path"], "wait_seconds": 600}))
            print(json.dumps(out))

asyncio.run(main())
"""
SLOW_SCRIPT = """
import os, time
from pathlib import Path
time.sleep(float(os.environ.get("SLOW_SECONDS", "30")))
Path(os.environ["ENTANGLIX_RUN_DIR"], "result.toml").write_text(
    'engine = "authored-script"\\n[metrics]\\nslept = 1\\n'
)
"""
# The plugin reference's minimal project, with an adapter around an open engine.
PLUGIN_PROJECT = """[project]
name = "probe-engine"
version = "0.1.0"
dependencies = ["numpy"]

[project.entry-points."entanglix.integrators"]
probe-engine = "probe_engine:Engine"

[build-system]
requires = ["uv_build>=0.12,<0.13"]
build-backend = "uv_build"
"""
PLUGIN_ENGINE = """
from dataclasses import replace

from entanglix.engines import EngineCapabilities, QuTiPIntegrator


class Engine:
    def capabilities(self):
        return EngineCapabilities("probe-engine", "cpu", True, False)

    def evolve(self, system, request):
        return replace(QuTiPIntegrator().evolve(system, request), engine="probe-engine")
"""
HOSTILE = {
    "UV_NO_SYNC": "1",
    "UV_FROZEN": "1",
    "UV_PROJECT_ENVIRONMENT": "/nonexistent/env",
    "UV_CACHE_DIR": "/nonexistent/cache",
    "UV_PYTHON": "/nonexistent/python",
    "VIRTUAL_ENV": "/nonexistent/venv",
    "PYTHONPATH": "/nonexistent/lib",
    "JULIA_DEPOT_PATH": "/nonexistent/depot",
}

results = {"steps": []}


def step(name, check, **details):
    """Record one observation; a failed check stops the world."""
    results["steps"].append({"step": name, "passed": bool(check), **details})
    print(f"{'PASS' if check else 'FAIL'} {name}", flush=True)
    if not check:
        raise SystemExit(f"failed: {name}: {json.dumps(details, default=str)[:4000]}")


def run(command, env, cwd, timeout=3600, check=False):
    started = time.monotonic()
    done = subprocess.run(
        command, env=env, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False
    )
    done.seconds = round(time.monotonic() - started, 1)
    if check and done.returncode:
        raise SystemExit(f"{command} failed ({done.returncode}):\n{done.stdout}\n{done.stderr}")
    return done


def fixture_repository(work: Path, versions: list[str]) -> Path:
    """A local repository holding this worktree's files, one commit and tag per version."""
    repository = work / "repository"
    names = subprocess.run(
        [
            "git",
            "-C",
            str(REPOSITORY),
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        capture_output=True,
        check=True,
    ).stdout.split(b"\0")
    for raw in filter(None, names):
        name = os.fsdecode(raw)
        source = REPOSITORY / name
        if source.is_file():
            target = repository / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target, follow_symlinks=False)
    git = ["git", "-C", str(repository), "-c", "user.name=Probe", "-c", "user.email=probe@invalid"]
    subprocess.run([*git, "init", "-q", "-b", "fixture"], check=True)
    current = tomllib.loads((repository / "pyproject.toml").read_text())["project"]["version"]
    for version in versions:
        for name, pattern, replacement in (
            ("pyproject.toml", rf'^version = "{current}"$', f'version = "{version}"'),
            ("install.sh", rf"^  version={current}$", f"  version={version}"),
            (
                "uv.lock",
                rf'(\nname = "entanglix"\n)version = "{current}"',
                rf'\1version = "{version}"',
            ),
        ):
            path = repository / name
            text, count = re.subn(pattern, replacement, path.read_text(), count=1, flags=re.M)
            assert count == 1 or version == current, (name, version)
            path.write_text(text)
        current = version
        subprocess.run([*git, "add", "-A"], check=True)
        subprocess.run([*git, "commit", "-qm", f"fixture {version}", "--allow-empty"], check=True)
        subprocess.run([*git, "tag", f"v{version}"], check=True)
    return repository


def world(work: Path, name: str, gpu: bool, repository: Path):
    home = work / name / "home"
    tools = work / name / "tools"
    home.mkdir(parents=True)
    tools.mkdir()
    for tool in TOOLS + (["nvidia-smi"] if gpu else []):
        found = shutil.which(tool, path="/usr/bin:/bin")
        assert found, tool
        (tools / tool).symlink_to(found)
    env = {
        "HOME": str(home),
        "PATH": str(tools),
        "ENTANGLIX_INSTALL_REPOSITORY": f"file://{repository}",
    }
    return home, env


def install(repository, version, home, env):
    script = home.parent / f"install-{version}.sh"
    script.write_text(
        subprocess.run(
            ["git", "-C", str(repository), "show", f"v{version}:install.sh"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )
    return run(["bash", str(script)], env | {"ENTANGLIX_INSTALL_REF": f"v{version}"}, home)


def entanglix(home, env, cwd, *arguments, timeout=3600):
    return run([str(home / ".local/bin/entanglix"), *arguments], env, cwd, timeout=timeout)


def job(home, env, cwd, *arguments):
    done = entanglix(home, env, cwd, *arguments)
    try:
        result = json.loads(done.stdout)
    except json.JSONDecodeError:
        result = {"stdout": done.stdout[-2000:], "stderr": done.stderr[-4000:]}
    return done, result


def manifest(path):
    return tomllib.loads((Path(path) / "run.toml").read_text())


def inputs(project: Path):
    """Tiny specs from the shipped examples."""
    readout = project / "inputs/readout"
    control = project / "inputs/control"
    shutil.copytree(REPOSITORY / "examples/readout", readout)
    shutil.copytree(REPOSITORY / "examples/transmon-control", control)
    base = (readout / "purcell.toml").read_text()
    short = re.sub(
        r"times_ns = \{[^}]*\}", "times_ns = { start = 0.0, stop = 10.0, count = 11 }", base
    )
    (readout / "cpu.toml").write_text(short)
    (readout / "cuda.toml").write_text(short.replace('device = "cpu"', 'device = "cuda"'))
    (readout / "cuda-gradients.toml").write_text(
        short.replace('device = "cpu"', 'device = "cuda"\ngradients = true')
    )
    for name in ("control", "piccolo"):
        text = (control / f"{name}.toml").read_text()
        (control / f"{name}-short.toml").write_text(re.sub(r"max_iter = \d+", "max_iter = 3", text))
    (project / "inputs/slow.py").write_text(SLOW_SCRIPT)
    return readout, control


def completed(name, done, result, project, revision):
    record = manifest(result["path"]) if "path" in result else {}
    step(
        name,
        done.returncode == 0
        and result.get("status") == "completed"
        and record.get("code", {}).get("dirty") is False
        and record.get("code", {}).get("revision") == revision,
        seconds=done.seconds,
        engine=record.get("engine"),
        env_python=record.get("environment", {}).get("python"),
        stderr=done.stderr[-1500:] if done.returncode else "",
        result={key: result.get(key) for key in ("status", "exit_code", "stdout", "stderr")},
    )
    return record


def tree_state(root: Path):
    """Environment identity that a resynchronization would change."""
    state = {}
    for name in (".venv", ".venv-runner"):
        site = next((root / name / "lib").glob("python*/site-packages"))
        state[name] = sorted(
            (entry.name, entry.stat().st_mtime_ns) for entry in site.iterdir() if entry.is_dir()
        )
    return state


def cpu_world(work, repository, revisions):
    home, env = world(work, "cpu", gpu=False, repository=repository)
    data = home / ".local/share/entanglix"
    done = install(repository, "0.1.0", home, env)
    step(
        "base install, nvidia-smi hidden",
        done.returncode == 0,
        seconds=done.seconds,
        output=done.stdout[-3000:],
        stderr=done.stderr[-3000:],
    )
    step(
        "installer reports no driver and the Piccolo command",
        "No NVIDIA driver found" in done.stdout and "components add piccolo" in done.stdout,
        output=done.stdout[-1500:],
    )
    first = data / "0.1.0"
    status = json.loads(entanglix(home, env, home, "components").stdout)
    step("components: base only", status["components"] == ["base"], status=status)
    refused = entanglix(home, env, home, "components", "add", "gpu")
    step(
        "components add gpu refused without a driver",
        refused.returncode == 64 and "needs an NVIDIA driver" in refused.stderr,
        stderr=refused.stderr,
    )

    projects = {}
    for runtime in ("claude", "codex", "other"):
        project = home / f"project-{runtime}"
        setup = entanglix(home, env, home, "setup-project", str(project), "--runtime", runtime)
        step(
            f"setup-project {runtime}",
            setup.returncode == 0,
            output=setup.stdout,
            stderr=setup.stderr,
        )
        projects[runtime] = project
    claude = projects["claude"]
    readout, control = inputs(claude)
    entry = json.loads((claude / ".mcp.json").read_text())["mcpServers"]["entanglix"]
    codex = tomllib.loads((projects["codex"] / ".codex/config.toml").read_text())
    codex_entry = codex["mcp_servers"]["entanglix"]
    python = str(first / ".venv/bin/python")
    for label, server in (("claude", entry), ("codex", codex_entry)):
        listed = run(
            [python, "-c", CLIENT, server["command"], json.dumps(server["args"])],
            env,
            projects[label],
            timeout=300,
        )
        capabilities = json.loads(listed.stdout)["capabilities"] if listed.returncode == 0 else {}
        step(
            f"{label} server starts and lists capabilities",
            capabilities.get("installation", {}).get("version") == "0.1.0",
            stderr=listed.stderr[-2000:],
            solvers=capabilities.get("solvers"),
        )

    done, result = job(home, env, claude, "simulate", str(readout / "cpu.toml"))
    completed(
        "CPU simulation through the installed command", done, result, claude, revisions["0.1.0"]
    )
    done, result = job(home, env, claude, "control", str(control / "control-short.toml"))
    completed("GRAPE control solve", done, result, claude, revisions["0.1.0"])
    through = run(
        [
            python,
            "-c",
            CLIENT,
            entry["command"],
            json.dumps(entry["args"]),
            str(readout / "cpu.toml"),
        ],
        env,
        claude,
        timeout=900,
    )
    served = json.loads(through.stdout)["run"] if through.returncode == 0 else {}
    step(
        "CPU simulation through the project's MCP server (scripted client)",
        served.get("status") == "completed",
        stderr=through.stderr[-2000:],
    )
    listed = entanglix(home, env, claude, "list-runs")
    found = json.loads(listed.stdout) if listed.returncode == 0 else []
    step(
        "the CLI's default runs root finds the server's run",
        served.get("manifest", {}).get("run_id") in {entry.get("run_id") for entry in found}
        and Path(served.get("path", "/")).is_relative_to(claude / "research_runs"),
        run=served.get("path"),
        stderr=listed.stderr,
    )

    before = tree_state(first)
    again = install(repository, "0.1.0", home, env)
    step(
        "same installer run again leaves both environments untouched",
        again.returncode == 0
        and tree_state(first) == before
        and json.loads(entanglix(home, env, home, "components").stdout)["components"] == ["base"],
        seconds=again.seconds,
        output=again.stdout[-1000:],
    )

    hostile = env | HOSTILE
    # A project of the user's may pin another Python; jobs keep the installation's.
    (claude / ".python-version").write_text("3.12\n")
    done, result = job(home, hostile, claude, "simulate", str(readout / "cpu.toml"))
    record = completed(
        "job with hostile uv, Python and Julia variables", done, result, claude, revisions["0.1.0"]
    )
    # A used /nonexistent cache or environment would have failed the job.
    step(
        "hostile variables redirected nothing",
        record["environment"]["python"] == (REPOSITORY / ".python-version").read_text().strip()
        and record["environment"]["extras"] == ["differentiable"],
        python=record["environment"]["python"],
        extras=record["environment"]["extras"],
    )

    slow = subprocess.Popen(
        [
            str(home / ".local/bin/entanglix"),
            "run-script",
            str(claude / "inputs/slow.py"),
            "--system",
            str(readout / "system.toml"),
        ],
        env=env | {"SLOW_SECONDS": "40"},
        cwd=claude,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    time.sleep(15)
    during = entanglix(home, env, home, "components", "add", "piccolo")
    step(
        "components add refused while a launch holds the environment",
        during.returncode == 64 and "a launch or job is using" in during.stderr,
        stderr=during.stderr,
    )
    out, err = slow.communicate(timeout=600)
    step("that launch completes", slow.returncode == 0, stderr=err[-1500:])

    server = subprocess.Popen(
        [entry["command"], *entry["args"]],
        env=env,
        cwd=claude,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(5)
    removal = entanglix(home, env, home, "remove", "0.1.0")
    step(
        "remove refused while an idle server uses the installation",
        removal.returncode == 64 and "in use" in removal.stderr,
        stderr=removal.stderr,
    )
    added = entanglix(home, env, home, "components", "add", "piccolo", timeout=5400)
    step(
        "components add piccolo succeeds beside the idle server",
        added.returncode == 0,
        seconds=added.seconds,
        stderr=added.stderr[-3000:],
        output=added.stdout[-1500:],
    )
    server.stdin.close()
    server.wait(timeout=60)
    done, result = job(home, env, claude, "control", str(control / "piccolo-short.toml"))
    record = completed("minimal Piccolo solve", done, result, claude, revisions["0.1.0"])
    runtime = record["environment"]["solver_runtime"]
    step(
        "Piccolo used the installation's own Julia",
        runtime["julia_executable"].startswith(str(first / ".installation/julia-1.13.0")),
        runtime=runtime,
    )

    # A second version beside the first, sharing Entanglix's uv cache.
    pinned = home / "project-pinned"
    entanglix(home, env, home, "setup-project", str(pinned), "--runtime", "claude")
    inputs(pinned)
    terms = home / "terms.toml"
    digest = hashlib.sha256((readout / "system.toml").read_bytes()).hexdigest()
    terms.write_text(
        'format_version = 1\nquestion = "Probe operator commands."\nassumptions = []\n'
        'review_criteria = []\nstop_criteria = []\n[scope]\noperations = ["simulate"]\n'
        f'[[scope.bases]]\nname = "probe"\ninputs = {{ "system.toml" = "{digest}" }}\n'
        '[evidence_requirements]\nrequirements = ["The project runs its own version."]\n'
        "[checkpoint]\nfinished_attempts = 1\n[limits]\n"
    )
    newer = install(repository, "0.1.1", home, env)
    step(
        "second version installs beside the first",
        newer.returncode == 0,
        seconds=newer.seconds,
        stderr=newer.stderr[-2000:],
    )
    # du counts a hardlinked file once, under the first argument that holds it.
    sizes = run(
        ["du", "-sk", str(data / "cache"), str(first), str(data / "0.1.1")],
        {"PATH": "/usr/bin:/bin"},
        home,
    ).stdout
    results["disk"] = {"du_kib_cache_then_first_then_second": sizes.splitlines()}
    old_version = entanglix(home, env, claude, "--version").stdout.strip()
    created = entanglix(home, env, claude, "create-investigation", str(terms))
    investigation = json.loads(created.stdout)["path"] if created.returncode == 0 else ""
    decided = entanglix(
        home, env, claude, "investigation-decision", investigation, "start", "--revision", "1"
    )
    done, result = job(home, env, claude, "simulate", str(readout / "cpu.toml"))
    completed(
        "older project still runs jobs on its version", done, result, claude, revisions["0.1.0"]
    )
    step(
        "older project's operator commands run its version",
        old_version == "entanglix 0.1.0" and decided.returncode == 0,
        version=old_version,
        stderr=decided.stderr + created.stderr,
    )
    refreshed = entanglix(home, env, claude, "setup-project", ".", "--refresh")
    moved = entanglix(home, env, claude, "--version").stdout.strip()
    step(
        "--refresh from inside the project moves it to the newer version",
        refreshed.returncode == 0 and moved == "entanglix 0.1.1",
        version=moved,
        stderr=refreshed.stderr,
    )

    slow = subprocess.Popen(
        [
            str(home / ".local/bin/entanglix"),
            "run-script",
            str(pinned / "inputs/slow.py"),
            "--system",
            str(pinned / "inputs/readout/system.toml"),
        ],
        env=env | {"SLOW_SECONDS": "30"},
        cwd=pinned,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    time.sleep(15)
    during = entanglix(home, env, home, "remove", "0.1.0")
    step(
        "remove refused during one of its jobs",
        during.returncode == 64 and "in use" in during.stderr,
        stderr=during.stderr,
    )
    slow.communicate(timeout=600)
    pinned_entry = json.loads((pinned / ".mcp.json").read_text())["mcpServers"]["entanglix"]
    server = subprocess.Popen(
        [pinned_entry["command"], *pinned_entry["args"]],
        env=env,
        cwd=pinned,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(5)
    idle = entanglix(home, env, home, "remove", "0.1.0")
    step("remove refused while its server is idle", idle.returncode == 64, stderr=idle.stderr)
    server.stdin.close()
    server.wait(timeout=60)
    removed = entanglix(home, env, home, "remove", "0.1.0")
    step(
        "remove succeeds once nothing uses the version",
        removed.returncode == 0 and not first.exists(),
        output=removed.stdout,
    )
    stale = entanglix(home, env, pinned, "list-runs")
    step(
        "a project on the removed version is refused with the refresh command",
        stale.returncode == 64 and "setup-project" in stale.stderr and "--refresh" in stale.stderr,
        stderr=stale.stderr,
    )
    recovered = entanglix(home, env, pinned, "setup-project", ".", "--refresh")
    step(
        "that refresh, run from inside the project, succeeds",
        recovered.returncode == 0,
        stderr=recovered.stderr,
    )
    last = entanglix(home, env, home, "remove", "0.1.1")
    step(
        "removing the last version deletes the cache and the launcher; only the data lock stays",
        last.returncode == 0
        and sorted(path.name for path in data.iterdir()) == [".lock"]
        and not (home / ".local/bin/entanglix").exists()
        and not (home / ".cache/uv").exists(),
        output=last.stdout,
        stderr=last.stderr,
    )


def gpu_world(work, repository, revisions):
    home, env = world(work, "gpu", gpu=True, repository=repository)
    done = install(repository, "0.1.0", home, env)
    step(
        "base install with a driver",
        done.returncode == 0 and "components add gpu" in done.stdout,
        seconds=done.seconds,
        output=done.stdout[-1500:],
        stderr=done.stderr[-2000:],
    )
    plugin = home / "probe-engine"
    (plugin / "src/probe_engine").mkdir(parents=True)
    (plugin / "pyproject.toml").write_text(PLUGIN_PROJECT)
    (plugin / "src/probe_engine/__init__.py").write_text(PLUGIN_ENGINE)
    installed = entanglix(home, env, home, "plugins", "add", str(plugin))
    step(
        "plugins add installs a plugin project with the installation's uv",
        installed.returncode == 0
        and json.loads(installed.stdout).get("engines") == ["probe-engine"],
        output=installed.stdout,
        stderr=installed.stderr[-2000:],
    )
    added = entanglix(home, env, home, "components", "add", "gpu", timeout=5400)
    report = json.loads(added.stdout) if added.returncode == 0 else {}
    step(
        "components add gpu, with the GPU environment check",
        report.get("status") == "added"
        and all(item["status"] == "passed" for item in report.get("gpu_check", []))
        and report.get("gpu_check"),
        seconds=added.seconds,
        report=report,
        stderr=added.stderr[-3000:],
    )
    project = home / "project"
    setup = entanglix(home, env, home, "setup-project", str(project), "--runtime", "claude")
    step(
        "setup-project picks the single GPU",
        setup.returncode == 0 and "device GPU-" in setup.stdout,
        output=setup.stdout,
    )
    readout, _ = inputs(project)
    listed = entanglix(home, env, home, "plugins")
    done, result = job(
        home,
        env | {"ENTANGLIX_ENGINES": "cpu=probe-engine"},
        project,
        "simulate",
        str(readout / "cpu.toml"),
    )
    record = completed(
        "the plugin survives the GPU component and runs a job",
        done,
        result,
        project,
        revisions["0.1.0"],
    )
    step(
        "the plugin's engine ran, recorded with its distribution",
        "probe-engine" in json.loads(listed.stdout or "{}")
        and record.get("engine") == "probe-engine"
        and "probe-engine" in record.get("environment", {}).get("distributions", {}),
        plugins=listed.stdout,
        engine=record.get("engine"),
    )
    for label, spec in (("cuDensityMat", "cuda.toml"), ("dynamiqs on CUDA", "cuda-gradients.toml")):
        done, result = job(home, env, project, "simulate", str(readout / spec))
        completed(f"CUDA job on {label}", done, result, project, revisions["0.1.0"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worlds", default="cpu,gpu")
    args = parser.parse_args()
    args.work.mkdir(parents=True)
    repository = fixture_repository(args.work, ["0.1.0", "0.1.1"])
    revisions = {
        version: subprocess.run(
            ["git", "-C", str(repository), "rev-parse", f"v{version}^{{commit}}"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        for version in ("0.1.0", "0.1.1")
    }
    results["fixture"] = {
        "source": "worktree files (tracked and untracked, not ignored)",
        "revisions": revisions,
    }
    started = time.monotonic()
    try:
        for name in args.worlds.split(","):
            {"cpu": cpu_world, "gpu": gpu_world}[name](args.work, repository, revisions)
    finally:
        results["wall_seconds"] = round(time.monotonic() - started)
        results["limit"] = "no clean-OS test: this workstation has no Docker or Podman"
        args.output.write_text(json.dumps(results, indent=2, default=str) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

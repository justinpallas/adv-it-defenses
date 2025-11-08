"""Command line interface for advdef."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import click

from advdef.config import ExperimentConfig
from advdef.core.context import RunContext
from advdef.core.pipeline import Pipeline
from advdef.utils import capture_environment, load_yaml_file, utc_timestamp


def _slugify(text: str) -> str:
    sanitized = "".join(char if char.isalnum() or char in ("-", "_") else "-" for char in text.strip().lower())
    sanitized = "-".join(filter(None, sanitized.split("-")))
    return sanitized or "experiment"


def _load_experiment(config_path: Path) -> ExperimentConfig:
    raw_config = load_yaml_file(config_path)
    return ExperimentConfig.model_validate(raw_config)


def _config_digest(experiment: ExperimentConfig) -> str:
    payload = experiment.model_dump(mode="json")
    serialized = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _directory_has_contents(path: Path) -> bool:
    return path.exists() and any(path.iterdir())


def _apply_overrides(config: dict, overrides: Dict[str, object]) -> dict:
    updated = copy.deepcopy(config)
    for key, value in overrides.items():
        parts = key.split(".")
        cursor = updated
        for idx, part in enumerate(parts):
            is_last = idx == len(parts) - 1
            if part.isdigit():
                index = int(part)
                if not isinstance(cursor, list):
                    raise TypeError(f"Cannot index into non-list at {'.'.join(parts[:idx])}")
                while len(cursor) <= index:
                    cursor.append({})
                if is_last:
                    cursor[index] = value
                else:
                    if not isinstance(cursor[index], (dict, list)):
                        cursor[index] = {}
                    cursor = cursor[index]
            else:
                if is_last:
                    if isinstance(cursor, list):
                        raise TypeError(f"Attempting to assign key '{part}' within a list at {'.'.join(parts[:idx])}")
                    cursor[part] = value
                else:
                    if isinstance(cursor, list):
                        raise TypeError(f"Attempting to traverse key '{part}' within a list at {'.'.join(parts[:idx])}")
                    cursor = cursor.setdefault(part, {})
    return updated


def _execute_experiment(
    config_path: Path,
    experiment: ExperimentConfig,
    run_name: str | None = None,
    context_options: Dict[str, object] | None = None,
    log_level: str = "warning",
    resume: bool = False,
) -> Tuple[str, dict]:
    timestamp = utc_timestamp()
    run_id = run_name or f"{timestamp}_{_slugify(experiment.name)}"
    run_dir = experiment.run_directory(run_id)

    if resume and run_name is None:
        raise click.ClickException("--resume requires --run-name.")
    if resume and not run_dir.exists():
        raise click.ClickException(f"Cannot resume run '{run_id}' because {run_dir} does not exist.")
    if not resume and _directory_has_contents(run_dir):
        raise click.ClickException(
            f"Run directory {run_dir} already exists. Use --run-name for a new run ID or --resume to continue it."
        )

    logging.basicConfig(level=getattr(logging, log_level.upper(), logging.WARNING))

    context = RunContext(
        experiment=experiment,
        run_id=run_id,
        run_dir=run_dir,
        timestamp=timestamp,
        work_dir=Path.cwd(),
        options=context_options or {},
        resume=resume,
    )

    stored_timestamp = context.state.get("timestamp")
    if stored_timestamp:
        context.timestamp = stored_timestamp
    else:
        context.state["timestamp"] = context.timestamp

    digest = _config_digest(experiment)
    existing_digest = context.state.get("config_digest")
    if existing_digest and existing_digest != digest:
        raise click.ClickException(
            "Experiment configuration has changed since this run started. "
            "Start a new run or delete the existing directory."
        )
    context.state.setdefault("config_digest", digest)
    context.state.setdefault("steps", {})
    context.state["run_id"] = run_id

    if resume and context.state.get("status") == "completed" and context.metrics_path.exists():
        with context.metrics_path.open("r") as handle:
            metrics = json.load(handle)
        click.echo(f"Run '{run_id}' already completed. Returning stored metrics.")
        return run_id, metrics

    context.state["status"] = "running"
    context.save_state()
    context.save_config()

    click.echo(f"Running experiment '{experiment.name}' (run id: {run_id})")
    pipeline = Pipeline(context)
    metrics = pipeline.run()

    metadata = {
        "run_id": run_id,
        "timestamp": timestamp,
        "config_path": str(config_path),
        "environment": capture_environment(),
        "metrics": metrics,
        "variants": pipeline.variant_records,
        "resumed": bool(resume),
    }
    context.save_metadata(metadata)
    context.state["status"] = "completed"
    context.save_state()

    return run_id, metrics


@click.group()
def app() -> None:
    """Adversarial defense experimentation toolkit."""


@app.command()
@click.argument("config", type=click.Path(path_type=Path, exists=True))
@click.option("--run-name", type=str, default=None, help="Optional explicit run identifier.")
@click.option(
    "--imagenet-root",
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True, writable=True, resolve_path=True),
    default=None,
    help="Override base directory for ImageNet data (val set + devkit)."
)
@click.option(
    "--log-level",
    type=click.Choice(["warning", "info", "debug"], case_sensitive=False),
    default="warning",
    help="Set logging verbosity for the run (warning/info/debug).",
)
@click.option(
    "--resume/--no-resume",
    default=False,
    help="Resume an unfinished run (requires --run-name).",
)
def run(config: Path, run_name: str | None, imagenet_root: Path | None, log_level: str, resume: bool) -> None:
    """Run a single experiment using the provided YAML configuration file."""
    if resume and not run_name:
        raise click.ClickException("--resume requires --run-name.")
    experiment = _load_experiment(config)
    run_id, metrics = _execute_experiment(
        config,
        experiment,
        run_name=run_name,
        context_options={"imagenet_root": imagenet_root} if imagenet_root else None,
        log_level=log_level,
        resume=resume,
    )
    click.echo(f"Experiment finished. Run id: {run_id}")
    click.echo(json.dumps(metrics, indent=2))


@app.command()
@click.argument("queue", type=click.Path(path_type=Path, exists=True))
@click.option(
    "--imagenet-root",
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True, writable=True, resolve_path=True),
    default=None,
    help="Override base directory for ImageNet data (val set + devkit).",
)
@click.option(
    "--log-level",
    type=click.Choice(["warning", "info", "debug"], case_sensitive=False),
    default="warning",
    help="Set logging verbosity for the run (warning/info/debug).",
)
@click.option(
    "--resume/--no-resume",
    default=False,
    help="Resume jobs that define run_name entries rather than starting over.",
)
def queue(queue: Path, imagenet_root: Path | None, log_level: str, resume: bool) -> None:
    """Execute multiple experiments described in a queue YAML file."""
    queue_data = load_yaml_file(queue)
    jobs = queue_data.get("jobs", [])
    if not jobs:
        click.echo("Queue is empty; nothing to do.")
        return

    results: List[Tuple[str, str]] = []
    for idx, job in enumerate(jobs, start=1):
        config_path = job.get("path")
        if not config_path:
            raise ValueError(f"Queue job {idx} is missing the 'path' entry.")
        config_file = (queue.parent / config_path).resolve()
        raw_config = load_yaml_file(config_file)
        overrides = job.get("overrides", {})
        if overrides:
            raw_config = _apply_overrides(raw_config, overrides)
        experiment = ExperimentConfig.model_validate(raw_config)
        run_name = job.get("run_name")
        click.echo(f"[{idx}/{len(jobs)}] {experiment.name}")
        job_resume = resume and run_name is not None
        if resume and run_name is None:
            click.echo(
                f"[warn] Job {idx} has no run_name; starting a new run instead of resuming."
            )
        run_id, _ = _execute_experiment(
            config_file,
            experiment,
            run_name=run_name,
            context_options={"imagenet_root": imagenet_root} if imagenet_root else None,
            log_level=log_level,
            resume=job_resume,
        )
        results.append((experiment.name, run_id))

    click.echo("Queue complete. Runs executed:")
    for name, run_id in results:
        click.echo(f" - {name}: {run_id}")


@app.command()
def version() -> None:
    """Print advdef version."""
    from advdef import __version__

    click.echo(__version__)


@app.group()
def setup() -> None:
    """Environment helpers for optional components."""


@setup.command("r-smoe")
@click.option(
    "--root",
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True, resolve_path=True),
    default=Path("external/r-smoe"),
    help="Path to the R-SMOE submodule root.",
)
def setup_rsmoe(root: Path) -> None:
    """Install R-SMOE runtime prerequisites (extensions + Python deps)."""
    click.echo(f"[setup] Using R-SMOE root: {root}")
    if not root.exists():
        raise click.ClickException(
            f"R-SMOE directory not found at {root}. Ensure the submodule is present."
        )

    # Ensure submodule contents are downloaded (no-op if already done)
    click.echo("[setup] Updating submodules (if necessary)...")
    subprocess.run(
        ["git", "submodule", "update", "--init", "--recursive", str(root)],
        check=True,
    )

    ext_dir = root / "submodules-2d-smoe"
    simple_knn = ext_dir / "simple-knn"
    diff_gauss = ext_dir / "diff-gaussian-rasterization"
    for path in (simple_knn, diff_gauss):
        if not path.exists():
            raise click.ClickException(f"Expected extension directory missing: {path}")

    def pip_install(args: List[str]) -> None:
        cmd = [sys.executable, "-m", "pip", "install"] + args
        click.echo(f"[setup] Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    pip_install(["-e", str(simple_knn)])
    pip_install(["-e", str(diff_gauss)])
    pip_install(["scikit-image", "imageio"])

    click.echo("[setup] R-SMOE dependencies installed successfully.")


if __name__ == "__main__":
    app(prog_name="advdef")

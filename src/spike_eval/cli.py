"""spike-eval CLI — one invocation = one idea = one run dir (design §2).

Wires the deterministic pipeline to interactive gates. The agentic executors are not
wired in scope A: `run` reaches gate 1 (idea-spec) and reports that the extractor is a
scope-B stub, unless an idea_spec.yaml is supplied for `resume`.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import click

from spike_eval.pipeline import Executors, resume_pipeline, run_pipeline
from spike_eval.planstage import format_scale
from spike_eval.report import render_reports


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _default_runs_dir() -> Path:
    return Path.cwd() / "runs"


def _approve_spec(spec) -> bool:
    click.echo(f"\n[gate 1] idea-spec extracted: {spec.idea_name}")
    click.echo(f"  claim: {spec.claim.statement}")
    click.echo(f"  extension point: {spec.extension_point.file} "
               f"({spec.extension_point.kind})")
    return click.confirm("Approve this idea-spec?", default=False)


def _approve_plan(plan) -> bool:
    click.echo("\n[gate 2] scale surface:")
    click.echo(format_scale(plan))
    return click.confirm("Approve running this scale?", default=False)


@click.group()
def cli() -> None:
    """SpikeEval — validate a new-algorithm idea against a repo's own baseline, fast."""


def _executors_for(family: str | None) -> Executors:
    """Select the repo-family executors. None -> bare Executors (agentic stages are
    scope-B stubs)."""
    if family in (None, "", "none"):
        return Executors()
    if family == "gptq-opt":
        from spike_eval.executors.gptq_opt import make_executors
        return make_executors()
    raise click.BadParameter(f"unknown --family: {family}")


@cli.command()
@click.argument("repo")
@click.argument("idea")
@click.option("--runs-dir", type=click.Path(), default=None,
              help="base dir for run directories (default: ./runs)")
@click.option("--family", default=None,
              help="repo-family executors to wire (e.g. gptq-opt)")
@click.option("--spec", "spec_path", type=click.Path(exists=True), default=None,
              help="hand-authored idea_spec.yaml to seed (skips headless extraction)")
def run(repo: str, idea: str, runs_dir: str | None, family: str | None,
        spec_path: str | None) -> None:
    """Validate IDEA on REPO. REPO: path|github url. IDEA: text|file|arxiv id."""
    base = Path(runs_dir) if runs_dir else _default_runs_dir()
    base.mkdir(parents=True, exist_ok=True)
    res = run_pipeline(repo, idea, base, _timestamp(),
                       ex=_executors_for(family), approve_spec=_approve_spec,
                       approve_plan=_approve_plan,
                       spec_path=Path(spec_path) if spec_path else None)
    _report_result(res)


@cli.command()
@click.argument("run_dir", type=click.Path(exists=True))
def resume(run_dir: str) -> None:
    """Resume a run from the plan stage using its idea_spec.yaml."""
    res = resume_pipeline(Path(run_dir), ex=Executors(), approve_plan=_approve_plan)
    _report_result(res)


@cli.command()
@click.argument("run_dir", type=click.Path(exists=True))
def report(run_dir: str) -> None:
    """Re-render the report from persisted spec + grade."""
    from spike_eval.rundir import RunDir
    rd = RunDir.open(Path(run_dir))
    spec, grade = rd.read_spec(), rd.read_grade()
    if spec is None or grade is None:
        click.echo("run dir has no spec/grade yet", err=True)
        sys.exit(1)
    analysis, en, zh = render_reports(spec, grade)
    (rd.root / "analysis.md").write_text(analysis)
    (rd.root / "README.md").write_text(en)
    (rd.root / "README_zh.md").write_text(zh)
    click.echo(f"re-rendered reports in {rd.root}")


def _report_result(res) -> None:
    if res.aborted_at:
        click.echo(f"\naborted at: {res.aborted_at}")
        click.echo(f"run dir: {res.root}")
        return
    g = res.grade
    click.echo(f"\nverdict: {g.verdict} — {g.reason}" if g else "\n(no grade)")
    for note in res.notes:
        click.echo(f"note: {note}")
    click.echo(f"run dir: {res.root}")


if __name__ == "__main__":
    cli()

"""Deterministic orchestration of the SpikeEval stages (design §2).

Two user gates (idea-spec, plan) + one internal correctness hard-gate. All side-effecting
stages are injected as callables so the whole pipeline runs offline with fakes (mirrors
paper_reprise.pipeline).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from spike_eval import ingest as ingest_mod
from spike_eval.ablate import run_ablation
from spike_eval.baseline import infra_ok, run_baseline
from spike_eval.correctness import gate_passed, run_correctness
from spike_eval.grade import grade_idea
from spike_eval.ideaspec import extract_spec
from spike_eval.implement import run_implement
from spike_eval.models import Grade, IdeaSpec, Measurement
from spike_eval.planstage import build_plan
from spike_eval.report import render_reports
from spike_eval.rundir import RunDir
from spike_eval.validate import run_ladder


@dataclass
class PipelineResult:
    root: Path
    aborted_at: Optional[str] = None
    grade: Optional[Grade] = None
    notes: list = field(default_factory=list)


@dataclass
class Executors:
    """The injected side-effecting callables. Any left None raises NotImplementedError
    if its stage is reached (scope B fills the real ones)."""
    fetch: Callable = ingest_mod.default_fetch
    fetch_arxiv: Optional[Callable] = None
    spec_extractor: Optional[Callable] = None
    baseline: Optional[Callable] = None
    implementer: Optional[Callable] = None
    check_runner: Optional[Callable] = None
    tier: Optional[Callable] = None
    ablation: Optional[Callable] = None


def _finish_with_grade(rd: RunDir, spec: IdeaSpec, grade: Grade) -> PipelineResult:
    rd.write_grade(grade)
    analysis, en, zh = render_reports(spec, grade)
    (rd.root / "analysis.md").write_text(analysis)
    (rd.root / "README.md").write_text(en)
    (rd.root / "README_zh.md").write_text(zh)
    return PipelineResult(root=rd.root, aborted_at=None, grade=grade)


def run_pipeline(
    repo_arg: str,
    idea_arg: str,
    base_dir: Path,
    timestamp: str,
    *,
    ex: Executors,
    approve_spec: Callable[[IdeaSpec], bool],
    approve_plan: Callable,
) -> PipelineResult:
    # --- ingest ---
    info = ingest_mod.classify(repo_arg, idea_arg)
    idea_name = f"idea-{timestamp}"  # refined once the spec is known (dir already stamped)
    rd = RunDir.create(base_dir, idea_name=idea_name, repo_name=Path(repo_arg).name or "repo",
                       timestamp=timestamp)
    ex.fetch(info, rd.repo_dir)
    rd.write_idea(ingest_mod.read_idea_text(info, ex.fetch_arxiv))

    # --- ideaspec + gate 1 ---
    spec = extract_spec(rd, ex.spec_extractor)
    if spec is None:
        return PipelineResult(root=rd.root, aborted_at="ideaspec")
    if not approve_spec(spec):
        return PipelineResult(root=rd.root, aborted_at="spec-approval")
    rd.write_spec(spec)
    rd.write_public_spec(spec)

    return _finish_pipeline(rd, spec, ex=ex, approve_plan=approve_plan)


def _finish_pipeline(rd: RunDir, spec: IdeaSpec, *, ex: Executors,
                     approve_plan: Callable) -> PipelineResult:
    # --- plan + gate 2 (surface scale) ---
    plan = build_plan(spec)
    (rd.root / "plan.json").write_text(plan.model_dump_json(indent=2))
    if plan.needs_user_decision and not approve_plan(plan):
        return PipelineResult(root=rd.root, aborted_at="plan")

    # --- baseline (first) + infra sanity ---
    first_model = spec.ladder[0].model if spec.ladder else spec.baseline.method
    base_probe: Measurement = run_baseline(rd, spec.baseline, spec.claim.protocol,
                                           first_model, ex.baseline)
    sane = infra_ok(base_probe, spec.baseline)

    # --- implement ---
    impl = run_implement(rd, spec, ex.implementer)
    if not impl.ok:
        return PipelineResult(root=rd.root, aborted_at="implement")

    # --- correctness HARD GATE (before any eval) ---
    checks = run_correctness(rd, spec, ex.check_runner)
    correctness_ok = gate_passed(checks)
    if not correctness_ok:
        grade = grade_idea(spec.claim, base_probe, None, correctness_ok=False,
                           infra_ok=sane)
        return _finish_with_grade(rd, spec, grade)

    # --- validate: cost ladder ---
    # Reaching here means gate 2 was cleared: either no expensive tier exists, or the
    # user approved the surfaced scale above. So expensive tiers may now run.
    ladder = run_ladder(rd, spec, ex.tier, allow_expensive=True)

    deciding = ladder.outcomes[-1] if ladder.outcomes else None
    base_m = deciding.baseline if deciding else base_probe
    idea_m = deciding.idea if deciding else None

    # --- ablate (attribution at the deciding tier) ---
    deciding_tier_obj = None
    if deciding is not None:
        deciding_tier_obj = next((t for t in spec.ladder if t.name == deciding.tier), None)
    run_ablation(rd, deciding_tier_obj, spec.ablations, ex.ablation)

    # --- grade (pure code) ---
    grade = grade_idea(spec.claim, base_m, idea_m, correctness_ok=correctness_ok,
                       infra_ok=sane,
                       deciding_tier=ladder.deciding_tier)
    return _finish_with_grade(rd, spec, grade)


def resume_pipeline(run_dir: Path, *, ex: Executors, approve_plan: Callable
                    ) -> PipelineResult:
    """Continue from the plan stage using the idea_spec.yaml already on disk (reviewing/
    editing it and resuming IS the gate-1 approval)."""
    rd = RunDir.open(Path(run_dir))
    spec = rd.read_spec()
    if spec is None:
        return PipelineResult(root=rd.root, aborted_at="no-spec")
    return _finish_pipeline(rd, spec, ex=ex, approve_plan=approve_plan)

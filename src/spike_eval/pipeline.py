"""Deterministic orchestration of the SpikeEval stages (design §2).

Two user gates (idea-spec, plan) + one internal correctness hard-gate. All side-effecting
stages are injected as callables so the whole pipeline runs offline with fakes (mirrors
paper_reprise.pipeline).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
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


def _blocked(spec: IdeaSpec, reason: str) -> Grade:
    """A BLOCKED verdict for a structural precondition (e.g. no ladder), independent of
    correctness / infra checks."""
    return Grade(claim_id=spec.claim.id, verdict="BLOCKED",
                 metric=spec.claim.protocol.metric,
                 lower_is_better=spec.claim.protocol.lower_is_better, reason=reason)


def _finish_with_grade(rd: RunDir, spec: IdeaSpec, grade: Grade) -> PipelineResult:
    rd.write_grade(grade)
    analysis_en, analysis_zh, en, zh = render_reports(spec, grade)
    (rd.root / "analysis_en.md").write_text(analysis_en)
    (rd.root / "analysis_zh.md").write_text(analysis_zh)
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
    spec_path: Optional[Path] = None,
) -> PipelineResult:
    # --- ingest ---
    info = ingest_mod.classify(repo_arg, idea_arg)
    # Slug the run dir from the idea itself (text head / file stem / arxiv id), known at
    # ingest — the spec's refined idea_name is not available until after gate 1.
    rd = RunDir.create(base_dir, idea_name=ingest_mod.idea_slug_source(info),
                       repo_name=Path(repo_arg).name or "repo", timestamp=timestamp)
    ex.fetch(info, rd.repo_dir)
    (rd.root / "ingest.json").write_text(json.dumps(asdict(info), indent=2))
    rd.write_idea(ingest_mod.read_idea_text(info, ex.fetch_arxiv))
    # A hand-authored spec (demo / no headless extractor) is seeded here so `extract_spec`
    # loads it and gate 1 still applies.
    if spec_path is not None:
        (rd.root / "idea_spec.yaml").write_text(Path(spec_path).read_text())

    # --- ideaspec + gate 1 ---
    spec = extract_spec(rd, ex.spec_extractor)
    if spec is None:
        return PipelineResult(root=rd.root, aborted_at="ideaspec")
    if not approve_spec(spec):
        return PipelineResult(root=rd.root, aborted_at="spec-approval")
    rd.write_spec(spec)

    return _finish_pipeline(rd, spec, ex=ex, approve_plan=approve_plan)


def _finish_pipeline(rd: RunDir, spec: IdeaSpec, *, ex: Executors,
                     approve_plan: Callable) -> PipelineResult:
    # --- plan (record scale; gate 2 is surfaced lazily inside validate) ---
    plan = build_plan(spec)
    (rd.root / "plan.json").write_text(plan.model_dump_json(indent=2))

    # A spec with no ladder has nothing to validate -> BLOCKED, not a silent pass.
    if not spec.ladder:
        return _finish_with_grade(rd, spec, _blocked(
            spec, "no ladder tiers defined — nothing to validate"))

    # --- baseline (first): infra-sanity probe on the cheapest tier's model.
    # NB: this is the eval-infra probe only; the ladder re-measures baseline per tier for
    # the fair per-tier comparison (design §5). ---
    probe_model = spec.ladder[0].model
    base_probe: Measurement = run_baseline(rd, spec.baseline, spec.claim.protocol,
                                           probe_model, ex.baseline)
    sane = infra_ok(base_probe, spec.baseline)

    # --- implement ---
    impl = run_implement(rd, spec, ex.implementer)
    if not impl.ok:
        return PipelineResult(root=rd.root, aborted_at="implement")

    # --- correctness HARD GATE (before any eval); zero checks does NOT clear it ---
    checks = run_correctness(rd, spec, ex.check_runner)
    correctness_ok = gate_passed(checks)
    if not correctness_ok:
        grade = grade_idea(spec.claim, base_probe, None, correctness_ok=False,
                           infra_ok=sane)
        return _finish_with_grade(rd, spec, grade)

    # --- validate: cost ladder; cheap tiers run, the first expensive tier is surfaced ---
    ladder = run_ladder(rd, spec, ex.tier,
                        approve_expensive=lambda tier: approve_plan(plan))

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
                       infra_ok=sane, deciding_tier=ladder.deciding_tier)
    notes = ([f"skipped expensive tiers (not approved): {ladder.skipped_expensive}"]
             if ladder.skipped_expensive else [])
    res = _finish_with_grade(rd, spec, grade)
    res.notes = notes
    return res


def resume_pipeline(run_dir: Path, *, ex: Executors, approve_plan: Callable
                    ) -> PipelineResult:
    """Continue from the plan stage using the idea_spec.yaml already on disk (reviewing/
    editing it and resuming IS the gate-1 approval)."""
    rd = RunDir.open(Path(run_dir))
    spec = rd.read_spec()
    if spec is None:
        return PipelineResult(root=rd.root, aborted_at="no-spec")
    return _finish_pipeline(rd, spec, ex=ex, approve_plan=approve_plan)

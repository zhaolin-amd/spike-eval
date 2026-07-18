"""plan — feasibility + cost-ladder plan + SURFACE SCALE (design §2.1, §6, gate 2).

Pure code: turns an IdeaSpec's ladder into a human-surfaceable scale summary and flags
whether user approval is required before spending. Encodes the "surface scale before
large runs" and "no default --yes" principles.
"""
from __future__ import annotations

from pydantic import BaseModel

from spike_eval.models import IdeaSpec, LadderTier


class TierScale(BaseModel):
    name: str
    model: str
    metric: str
    budget_gpu_hours: float
    est_cost_usd: float
    is_expensive: bool


class PlanReport(BaseModel):
    idea_name: str
    tiers: list[TierScale]
    total_gpu_hours: float
    total_cost_usd: float
    biggest_model: str
    # True when any tier is expensive -> gate 2 must be shown before validate runs.
    needs_user_decision: bool
    notes: list[str] = []


def _tier_scale(t: LadderTier) -> TierScale:
    return TierScale(
        name=t.name, model=t.model, metric=t.protocol.metric,
        budget_gpu_hours=t.budget_gpu_hours, est_cost_usd=t.est_cost_usd,
        is_expensive=t.is_expensive,
    )


def build_plan(spec: IdeaSpec) -> PlanReport:
    """Summarize the ladder's scale and decide whether gate 2 is required."""
    tiers = [_tier_scale(t) for t in spec.ladder]
    total_h = sum(t.budget_gpu_hours for t in tiers)
    total_c = sum(t.est_cost_usd for t in tiers)
    biggest = max((t for t in spec.ladder), key=lambda t: t.budget_gpu_hours,
                  default=None)
    notes: list[str] = []
    if not tiers:
        notes.append("no ladder tiers defined — validation would be a no-op")
    return PlanReport(
        idea_name=spec.idea_name,
        tiers=tiers,
        total_gpu_hours=round(total_h, 3),
        total_cost_usd=round(total_c, 3),
        biggest_model=biggest.model if biggest else "",
        needs_user_decision=any(t.is_expensive for t in tiers),
        notes=notes,
    )


def format_scale(plan: PlanReport) -> str:
    """One-screen scale surface for the CLI gate (design §6). Never hides a big model."""
    lines = [f"Idea: {plan.idea_name}",
             f"Ladder: {len(plan.tiers)} tier(s) | biggest model: {plan.biggest_model or 'n/a'}",
             f"Total est: {plan.total_gpu_hours} GPU-h, ${plan.total_cost_usd}"]
    for t in plan.tiers:
        flag = "  <-- EXPENSIVE, needs approval" if t.is_expensive else ""
        lines.append(f"  - {t.name:9s} {t.model:16s} {t.metric:11s} "
                     f"{t.budget_gpu_hours}h ${t.est_cost_usd}{flag}")
    for n in plan.notes:
        lines.append(f"note: {n}")
    return "\n".join(lines)

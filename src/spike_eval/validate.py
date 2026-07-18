"""validate — the cost ladder (design §6).

Real control flow: walk tiers cheapest-first, measure baseline + idea at each, stop as
soon as a tier fails the pass margin (fail fast). Expensive tiers are only entered when
`allow_expensive` is set (the pipeline sets it from gate 2). The per-tier measurement is
an injected callable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from spike_eval.models import Claim, IdeaSpec, LadderTier, Measurement
from spike_eval.rundir import RunDir

# Executor contract: measure one variant ("baseline" | "idea") at one tier, returning a
# Measurement. Injected so the ladder is offline-testable.
TierExecutor = Callable[[RunDir, LadderTier, str], Measurement]


@dataclass
class TierOutcome:
    tier: str
    baseline: Optional[Measurement] = None
    idea: Optional[Measurement] = None
    advanced: bool = False           # cleared the pass margin -> go to next tier
    reason: str = ""


@dataclass
class LadderResult:
    outcomes: list[TierOutcome] = field(default_factory=list)
    deciding_tier: Optional[str] = None
    stopped_early: bool = False
    skipped_expensive: list[str] = field(default_factory=list)


def _margin(tier: LadderTier, claim: Claim) -> float:
    return tier.pass_margin if tier.pass_margin is not None else claim.min_delta


def _cleared(baseline: Measurement, idea: Measurement, lower_is_better: bool,
             margin: float) -> bool:
    if not (baseline.ok and idea.ok) or baseline.value is None or idea.value is None:
        return False
    delta = (baseline.value - idea.value) if lower_is_better else (idea.value - baseline.value)
    return delta >= margin


def run_ladder(rd: RunDir, spec: IdeaSpec, executor: Optional[TierExecutor],
               *, allow_expensive: bool = False) -> LadderResult:
    """Walk the ladder, fail fast, respect the expensive-tier gate."""
    if executor is None and spec.ladder:
        raise NotImplementedError("tier executor not wired (scope B)")
    res = LadderResult()
    lib = spec.claim.protocol.lower_is_better
    for tier in spec.ladder:
        if tier.is_expensive and not allow_expensive:
            res.skipped_expensive.append(tier.name)
            res.stopped_early = True
            break
        base_m = executor(rd, tier, "baseline")
        idea_m = executor(rd, tier, "idea")
        cleared = _cleared(base_m, idea_m, lib, _margin(tier, spec.claim))
        out = TierOutcome(tier=tier.name, baseline=base_m, idea=idea_m,
                          advanced=cleared,
                          reason="cleared margin" if cleared else "below margin")
        res.outcomes.append(out)
        res.deciding_tier = tier.name
        if not cleared:
            res.stopped_early = True
            break
    return res

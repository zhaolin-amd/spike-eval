"""grade — pure-code go/no-go judge, isolated from execution (design §5).

No LLM. A claim is WIN only if correctness passed AND infra is sane AND the idea beats
the measured baseline by >= min_delta in the claim's direction. Mirrors paper-reprise's
process-faithful + value-in-tolerance rule, adapted to "beats the baseline".
"""
from __future__ import annotations

from typing import Optional

from spike_eval.models import Claim, Grade, Measurement, TierName


def grade_idea(
    claim: Claim,
    baseline: Optional[Measurement],
    idea: Optional[Measurement],
    *,
    correctness_ok: bool,
    infra_ok: bool,
    deciding_tier: Optional[TierName] = None,
) -> Grade:
    lib = claim.protocol.lower_is_better
    b = baseline.value if baseline else None
    i = idea.value if idea else None

    def mk(verdict: str, reason: str, delta: Optional[float] = None) -> Grade:
        return Grade(
            claim_id=claim.id, verdict=verdict, metric=claim.protocol.metric,
            lower_is_better=lib, baseline_value=b, idea_value=i, delta=delta,
            correctness_ok=correctness_ok, infra_ok=infra_ok,
            deciding_tier=deciding_tier, reason=reason,
        )

    # --- BLOCKED: can't trust a comparison ---
    if not correctness_ok:
        return mk("BLOCKED", "correctness gate failed — idea not equivalent/valid")
    if not infra_ok:
        return mk("BLOCKED", "eval-infra sanity failed — baseline out of band; "
                             "suspect the protocol, not the idea")
    if b is None or i is None:
        return mk("BLOCKED", "missing measurement at the deciding tier")

    # signed improvement, positive = better
    delta = (b - i) if lib else (i - b)

    if delta >= claim.min_delta:
        return mk("WIN", f"beats baseline by {delta:.6g} >= min_delta {claim.min_delta:g}",
                  delta)
    if delta < -claim.tolerance:
        return mk("LOSE", f"regresses by {-delta:.6g} beyond tolerance {claim.tolerance:g}",
                  delta)
    return mk("NEUTRAL", f"delta {delta:.6g} within noise band "
                         f"[-{claim.tolerance:g}, {claim.min_delta:g})", delta)

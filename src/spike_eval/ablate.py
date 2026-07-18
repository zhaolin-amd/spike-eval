"""ablate — attribution: baseline / +idea / per-subcomponent (design §2.1).

One variant per declared ablation flag, measured at the deciding tier, so a moved number
can be traced to a specific sub-component rather than "changed five things at once".
Measurement is an injected callable; scope A ships the interface.
"""
from __future__ import annotations

from typing import Callable, Optional

from spike_eval.models import Ablation, LadderTier, Measurement
from spike_eval.rundir import RunDir

# Executor contract: measure one ablation variant at a tier, return a Measurement.
AblationExecutor = Callable[[RunDir, LadderTier, Ablation], Measurement]


def run_ablation(rd: RunDir, tier: Optional[LadderTier], ablations: list[Ablation],
                 executor: Optional[AblationExecutor]) -> list[Measurement]:
    if not ablations or tier is None:
        return []
    if executor is None:
        raise NotImplementedError("ablation executor not wired (scope B)")
    return [executor(rd, tier, a) for a in ablations]

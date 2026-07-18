"""baseline — run the repo's ORIGINAL algorithm first (design §1.2, baseline-first).

Its measured number is BOTH the grade reference AND the eval-infra sanity probe: a
baseline outside its sane band means suspect the eval protocol, not the idea.
The GPU execution is an injected callable (scope B wires the real one).
"""
from __future__ import annotations

from typing import Callable, Optional

from spike_eval.models import Baseline, EvalProtocol, Measurement
from spike_eval.rundir import RunDir

# Executor contract: measure the baseline method under a protocol at a given tier model,
# returning a Measurement. Injected for offline tests.
BaselineExecutor = Callable[[RunDir, Baseline, EvalProtocol, str], Measurement]


def infra_ok(m: Measurement, baseline: Baseline) -> bool:
    """The eval-infra sanity check (design §7). True when the measured baseline is inside
    its declared sane band; True (unchecked) when no band was declared; False when the
    number is missing or out of band."""
    if not m.ok or m.value is None:
        return False
    if baseline.sane_low is not None and m.value < baseline.sane_low:
        return False
    if baseline.sane_high is not None and m.value > baseline.sane_high:
        return False
    return True


def run_baseline(rd: RunDir, baseline: Baseline, protocol: EvalProtocol, model: str,
                 executor: Optional[BaselineExecutor]) -> Measurement:
    if executor is None:
        raise NotImplementedError("baseline executor not wired (scope B)")
    return executor(rd, baseline, protocol, model)

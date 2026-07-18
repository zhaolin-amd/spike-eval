"""correctness — HARD GATE before any eval (design §1.2, §6 L0).

Runs the spec's correctness checks (unit / closed-form / degenerate-equivalence). The
verdict aggregation is pure code; running each check is an injected callable. Green
tests+lint never substitute for this gate (self-review-is-a-hard-gate principle).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from spike_eval.models import CorrectnessCheck, IdeaSpec
from spike_eval.rundir import RunDir


@dataclass
class CheckResult:
    kind: str
    passed: bool
    detail: str = ""


# Executor contract: run one correctness check in the run dir, return its result.
CheckRunner = Callable[[RunDir, CorrectnessCheck], CheckResult]


def gate_passed(results: list[CheckResult]) -> bool:
    """The hard gate: ALL declared checks must pass, and there must be at least one
    (an idea with no correctness check does not clear the gate)."""
    return bool(results) and all(r.passed for r in results)


def run_correctness(rd: RunDir, spec: IdeaSpec, runner: Optional[CheckRunner]
                    ) -> list[CheckResult]:
    if not spec.correctness:
        return []
    if runner is None:
        raise NotImplementedError("correctness runner not wired (scope B)")
    return [runner(rd, c) for c in spec.correctness]

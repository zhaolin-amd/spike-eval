"""Drive the GPTQ bias-correction demo end-to-end (scope B first example).

Auto-approves both gates: the scale (one L2_tiny tier, opt-125m, ~0.05 GPU-h, no expensive
tier) was surfaced and approved out of band. Run with an interpreter that can import
spike_eval, and point SPIKE_EVAL_PYTHON at one that can run the GPTQ repo (torch), e.g.:

    SPIKE_EVAL_PYTHON=/path/to/torch/python \
        python scripts/run_gptq_demo.py
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from spike_eval.executors.gptq_opt import GITHUB_URL, make_executors
from spike_eval.pipeline import run_pipeline

REPO_ROOT = Path(__file__).resolve().parent.parent
SPEC = REPO_ROOT / "examples" / "gptq-bias-correction" / "idea_spec.yaml"
IDEA = ("GPTQ bias-correction: after quantizing each OPT block, fold alpha * mean output "
        "error into each Linear's bias; alpha=0 == vanilla GPTQ.")


def main() -> None:
    base = REPO_ROOT / "runs"
    base.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    res = run_pipeline(
        GITHUB_URL, IDEA, base, ts,
        ex=make_executors(),
        approve_spec=lambda spec: True,
        approve_plan=lambda plan: True,
        spec_path=SPEC,
    )
    print("\n===== SpikeEval GPTQ demo =====")
    if res.aborted_at:
        print(f"aborted at: {res.aborted_at}")
    elif res.grade:
        g = res.grade
        print(f"verdict : {g.verdict}")
        print(f"reason  : {g.reason}")
        print(f"baseline: {g.baseline_value}")
        print(f"idea    : {g.idea_value}")
        print(f"delta   : {g.delta}")
        print(f"gates   : correctness={g.correctness_ok} infra={g.infra_ok}")
    for n in res.notes:
        print(f"note: {n}")
    print(f"run dir : {res.root}")


if __name__ == "__main__":
    main()

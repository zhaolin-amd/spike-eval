"""Offline end-to-end pipeline tests: every side-effecting stage is a fake, so the whole
deterministic flow + two gates + correctness hard-gate run without network or GPU."""
from spike_eval.correctness import CheckResult
from spike_eval.implement import ImplementResult
from spike_eval.models import (
    Claim, EvalProtocol, IdeaSpec, LadderTier, Measurement,
)
from spike_eval.pipeline import Executors, run_pipeline


def _proto():
    return EvalProtocol(command="e", metric="ppl", lower_is_better=True)


def _spec(with_checks=True, tiers=None):
    if tiers is None:
        tiers = [LadderTier(name="L1_proxy", model="random", protocol=_proto())]
    return IdeaSpec(
        idea_name="GPTQ+comp",
        target_repo="/tmp/gptq",
        extension_point={"file": "gptq.py", "symbol": "GPTQ", "kind": "subclass"},
        baseline={"method": "GPTQ", "command": "run", "sane_low": 5.0, "sane_high": 20.0},
        claim=Claim(id="c1", statement="beats GPTQ ppl by >=0.1", protocol=_proto(),
                    min_delta=0.1, tolerance=0.02),
        correctness=([{"kind": "degenerate_equivalence",
                       "degenerate_params": {"alpha": 0}}] if with_checks else []),
        ladder=tiers,
    )


def _executors(spec, *, baseline_val, idea_val, checks_pass=True):
    def fetch(info, repo_dir):
        (repo_dir / "gptq.py").write_text("# fake repo\n")

    def extractor(rd):
        return spec

    def baseline_exec(rd, base, proto, model):
        return Measurement(tier="L1_proxy", variant="baseline", metric="ppl",
                           value=baseline_val)

    def implementer(rd, sp):
        return ImplementResult(ok=True, patch_path="impl/idea.patch")

    def check_runner(rd, check):
        return CheckResult(kind=check.kind, passed=checks_pass)

    def tier_exec(rd, tier, variant):
        v = baseline_val if variant == "baseline" else idea_val
        return Measurement(tier=tier.name, variant=variant, metric="ppl", value=v)

    return Executors(fetch=fetch, spec_extractor=extractor, baseline=baseline_exec,
                     implementer=implementer, check_runner=check_runner, tier=tier_exec)


def test_pipeline_win(tmp_path):
    spec = _spec()
    ex = _executors(spec, baseline_val=10.0, idea_val=9.8)
    res = run_pipeline("/tmp/gptq", "add a compensation term", tmp_path, "20260718-000000",
                       ex=ex, approve_spec=lambda s: True, approve_plan=lambda p: True)
    assert res.aborted_at is None
    assert res.grade.verdict == "WIN"
    assert (res.root / "README.md").exists()
    assert (res.root / "README_zh.md").exists()
    assert (res.root / "analysis.md").exists()
    assert (res.root / "idea_spec.yaml").exists()


def test_pipeline_correctness_hard_gate_blocks(tmp_path):
    spec = _spec()
    ex = _executors(spec, baseline_val=10.0, idea_val=9.0, checks_pass=False)
    res = run_pipeline("/tmp/gptq", "idea", tmp_path, "20260718-000001",
                       ex=ex, approve_spec=lambda s: True, approve_plan=lambda p: True)
    assert res.grade.verdict == "BLOCKED"
    assert "correctness" in res.grade.reason


def test_pipeline_spec_rejected_aborts(tmp_path):
    spec = _spec()
    ex = _executors(spec, baseline_val=10.0, idea_val=9.8)
    res = run_pipeline("/tmp/gptq", "idea", tmp_path, "20260718-000002",
                       ex=ex, approve_spec=lambda s: False, approve_plan=lambda p: True)
    assert res.aborted_at == "spec-approval"


def test_pipeline_infra_unsane_blocks(tmp_path):
    spec = _spec()
    # baseline 99 is outside sane band [5, 20] -> infra check fails -> BLOCKED
    ex = _executors(spec, baseline_val=99.0, idea_val=9.8)
    res = run_pipeline("/tmp/gptq", "idea", tmp_path, "20260718-000003",
                       ex=ex, approve_spec=lambda s: True, approve_plan=lambda p: True)
    assert res.grade.verdict == "BLOCKED"
    assert "infra" in res.grade.reason

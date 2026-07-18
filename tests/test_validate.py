"""Cost-ladder control-flow tests (fail fast + expensive-tier gate)."""
from spike_eval.models import Claim, EvalProtocol, IdeaSpec, LadderTier, Measurement
from spike_eval.rundir import RunDir
from spike_eval.validate import run_ladder


def _proto():
    return EvalProtocol(command="e", metric="ppl", lower_is_better=True)


def _spec(tiers):
    return IdeaSpec(
        idea_name="X", target_repo="/tmp/x",
        extension_point={"file": "f.py"},
        baseline={"method": "GPTQ", "command": "c"},
        claim=Claim(id="c1", statement="s", protocol=_proto(), min_delta=0.1),
        ladder=tiers,
    )


def _rd(tmp_path):
    return RunDir.create(tmp_path, "x", "repo", "20260718-000000")


def _exec(values):
    # values: {(tier, variant): ppl}
    def ex(rd, tier, variant):
        return Measurement(tier=tier.name, variant=variant, metric="ppl",
                           value=values[(tier.name, variant)])
    return ex


def test_ladder_advances_through_all_tiers(tmp_path):
    tiers = [LadderTier(name="L1_proxy", model="random", protocol=_proto()),
             LadderTier(name="L2_tiny", model="opt-125m", protocol=_proto())]
    spec = _spec(tiers)
    vals = {("L1_proxy", "baseline"): 10.0, ("L1_proxy", "idea"): 9.8,
            ("L2_tiny", "baseline"): 8.0, ("L2_tiny", "idea"): 7.8}
    res = run_ladder(_rd(tmp_path), spec, _exec(vals))
    assert len(res.outcomes) == 2
    assert res.deciding_tier == "L2_tiny"
    assert not res.stopped_early


def test_ladder_fails_fast(tmp_path):
    tiers = [LadderTier(name="L1_proxy", model="random", protocol=_proto()),
             LadderTier(name="L2_tiny", model="opt-125m", protocol=_proto())]
    spec = _spec(tiers)
    # L1 idea no better than baseline -> stop before L2
    vals = {("L1_proxy", "baseline"): 10.0, ("L1_proxy", "idea"): 10.0,
            ("L2_tiny", "baseline"): 8.0, ("L2_tiny", "idea"): 7.0}
    res = run_ladder(_rd(tmp_path), spec, _exec(vals))
    assert len(res.outcomes) == 1
    assert res.stopped_early
    assert res.deciding_tier == "L1_proxy"


def test_ladder_skips_expensive_without_approval(tmp_path):
    tiers = [LadderTier(name="L3_small", model="llama-7b", protocol=_proto(),
                        budget_gpu_hours=2.0)]
    spec = _spec(tiers)
    res = run_ladder(_rd(tmp_path), spec, _exec({}), allow_expensive=False)
    assert res.skipped_expensive == ["L3_small"]
    assert res.outcomes == []

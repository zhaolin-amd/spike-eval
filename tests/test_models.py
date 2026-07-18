from spike_eval.models import Claim, EvalProtocol, IdeaSpec, LadderTier


def _proto(**kw):
    return EvalProtocol(command="eval", metric="perplexity", **kw)


def test_min_delta_and_tolerance_coerced_nonneg():
    c = Claim(id="c1", statement="x", protocol=_proto(), min_delta=-0.5, tolerance=None)
    assert c.min_delta == 0.5
    assert c.tolerance == 0.0


def test_tier_is_expensive_by_name_and_budget():
    cheap = LadderTier(name="L1_proxy", model="random", protocol=_proto())
    small = LadderTier(name="L3_small", model="llama-7b", protocol=_proto())
    heavy = LadderTier(name="L2_tiny", model="opt-1.3b", protocol=_proto(),
                       budget_gpu_hours=2.0)
    assert not cheap.is_expensive
    assert small.is_expensive          # by name
    assert heavy.is_expensive          # by budget


def test_ideaspec_roundtrip():
    spec = IdeaSpec(
        idea_name="GPTQ+comp",
        target_repo="/tmp/gptq",
        extension_point={"file": "gptq.py", "symbol": "GPTQ", "kind": "subclass"},
        baseline={"method": "GPTQ", "command": "run gptq"},
        claim=Claim(id="c1", statement="beats GPTQ ppl by 0.1", protocol=_proto(),
                    min_delta=0.1),
    )
    d = spec.model_dump()
    assert IdeaSpec.model_validate(d).idea_name == "GPTQ+comp"

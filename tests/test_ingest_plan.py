from spike_eval.ingest import classify, detect_idea_kind, detect_repo_kind
from spike_eval.models import Claim, EvalProtocol, IdeaSpec, LadderTier
from spike_eval.planstage import build_plan, format_scale


def test_detect_idea_kind_arxiv():
    kind, aid = detect_idea_kind("2504.19874")
    assert kind == "arxiv" and aid == "2504.19874"
    kind, aid = detect_idea_kind("https://arxiv.org/abs/2309.05516v2")
    assert kind == "arxiv" and aid == "2309.05516"


def test_detect_idea_kind_text_and_file(tmp_path):
    assert detect_idea_kind("add a compensation term to GPTQ")[0] == "text"
    f = tmp_path / "idea.md"
    f.write_text("my idea")
    assert detect_idea_kind(str(f))[0] == "file"


def test_detect_repo_kind():
    assert detect_repo_kind("https://github.com/IST-DASLab/gptq") == "github"
    assert detect_repo_kind("/home/user/code/Quark") == "local"


def test_classify():
    info = classify("https://github.com/a/b", "2504.19874")
    assert info.repo_kind == "github"
    assert info.idea_kind == "arxiv"
    assert info.arxiv_id == "2504.19874"


def _proto():
    return EvalProtocol(command="e", metric="ppl")


def _spec(tiers):
    return IdeaSpec(
        idea_name="X", target_repo="/tmp/x",
        extension_point={"file": "f.py"},
        baseline={"method": "GPTQ", "command": "c"},
        claim=Claim(id="c1", statement="s", protocol=_proto(), min_delta=0.1),
        ladder=tiers,
    )


def test_plan_flags_expensive_tier():
    spec = _spec([
        LadderTier(name="L1_proxy", model="random", protocol=_proto()),
        LadderTier(name="L3_small", model="llama-7b", protocol=_proto(),
                   budget_gpu_hours=2.0, est_cost_usd=5.0),
    ])
    plan = build_plan(spec)
    assert plan.needs_user_decision is True
    assert plan.biggest_model == "llama-7b"
    assert "EXPENSIVE" in format_scale(plan)


def test_plan_cheap_needs_no_decision():
    spec = _spec([LadderTier(name="L1_proxy", model="random", protocol=_proto())])
    plan = build_plan(spec)
    assert plan.needs_user_decision is False

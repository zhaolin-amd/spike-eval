from spike_eval.grade import grade_idea
from spike_eval.models import Claim, EvalProtocol, Measurement


def _claim(lower_is_better=True, min_delta=0.1, tolerance=0.02):
    return Claim(id="c1", statement="s",
                 protocol=EvalProtocol(command="e", metric="ppl",
                                       lower_is_better=lower_is_better),
                 min_delta=min_delta, tolerance=tolerance)


def _m(v, variant="idea"):
    return Measurement(tier="L2_tiny", variant=variant, metric="ppl", value=v)


def test_win_lower_is_better():
    g = grade_idea(_claim(), _m(10.0, "baseline"), _m(9.8),
                   correctness_ok=True, infra_ok=True)
    assert g.verdict == "WIN"
    assert g.delta == __import__("pytest").approx(0.2)


def test_neutral_within_noise():
    g = grade_idea(_claim(), _m(10.0, "baseline"), _m(9.99),
                   correctness_ok=True, infra_ok=True)
    assert g.verdict == "NEUTRAL"


def test_lose_beyond_tolerance():
    g = grade_idea(_claim(), _m(10.0, "baseline"), _m(10.5),
                   correctness_ok=True, infra_ok=True)
    assert g.verdict == "LOSE"


def test_blocked_when_correctness_fails():
    g = grade_idea(_claim(), _m(10.0, "baseline"), _m(9.0),
                   correctness_ok=False, infra_ok=True)
    assert g.verdict == "BLOCKED"


def test_blocked_when_infra_unsane():
    g = grade_idea(_claim(), _m(10.0, "baseline"), _m(9.0),
                   correctness_ok=True, infra_ok=False)
    assert g.verdict == "BLOCKED"


def test_higher_is_better_win():
    g = grade_idea(_claim(lower_is_better=False), _m(70.0, "baseline"), _m(70.3),
                   correctness_ok=True, infra_ok=True)
    assert g.verdict == "WIN"


def test_blocked_on_missing_measurement():
    g = grade_idea(_claim(), _m(10.0, "baseline"), None,
                   correctness_ok=True, infra_ok=True)
    assert g.verdict == "BLOCKED"

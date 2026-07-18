"""Offline tests for the gptq-opt executor's pure helpers (no GPU / no subprocess)."""
from spike_eval.executors.gptq_opt import _cache_key, parse_layer_mse, parse_ppl

# Captured shape of IST-DASLab/gptq opt.py stdout: quantization time float appears BEFORE
# the 'wikitext2' marker, then the eval section ends in the ppl float. LAYER_MSE lines are
# printed during quantization (before the eval section).
SAMPLE = """Starting ...
Ready.
0 self_attn.k_proj
Quantizing ...
LAYER_MSE 0 0.010000
11 fc2
Quantizing ...
LAYER_MSE 11 0.030000
20.090407371520996
wikitext2
Evaluating ...
0
1
2
3
30.537620544433594
ptb
Evaluating ...
0
"""


def test_parse_ppl_wikitext2():
    assert parse_ppl(SAMPLE, "wikitext2") == 30.537620544433594


def test_parse_ppl_ignores_quant_time_before_marker():
    # 20.09 (quant time) precedes the marker and must not be returned.
    assert parse_ppl(SAMPLE, "wikitext2") != 20.090407371520996


def test_parse_ppl_missing_dataset():
    assert parse_ppl(SAMPLE, "c4") is None


def test_parse_ppl_empty():
    assert parse_ppl("", "wikitext2") is None


def test_parse_layer_mse_mean():
    # mean of 0.01 and 0.03
    assert parse_layer_mse(SAMPLE) == 0.02


def test_parse_layer_mse_missing():
    assert parse_layer_mse("no mse lines here") is None


def test_cache_key_distinguishes_alpha():
    m = "facebook/opt-125m"
    assert _cache_key(m, None).endswith("pristine")
    assert _cache_key(m, 0.0) != _cache_key(m, 1.0)
    assert "a0.0" in _cache_key(m, 0.0)

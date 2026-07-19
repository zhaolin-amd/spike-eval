"""Correctness gate for the activation-aware (amax^2-weighted) MBS-Dynamic search idea.

Independent of any perplexity number. Pins the *mechanism*:
  1. degenerate equivalence — uniform weight == the paper's plain-SSE search, bit-exact;
  2. closed-form cross-check — the selected slot is the brute-force argmin of the
     weighted objective sum_k a_k^2 (deq_k - w_k)^2, computed independently;
  3. the weighted search really lowers the weighted error vs the plain-SSE baseline
     (the idea does what it claims), and it is not a no-op.
"""
import torch
import torch.nn as nn

import mxquant as mq
from qmodel import QuantLinear


def test_uniform_weight_is_bitexact_baseline():
    # chan_w of all-ones multiplies the per-position squared error by 1.0 exactly, so the
    # argmin (hence the whole quantized tensor) is identical to the unweighted search.
    torch.manual_seed(0)
    w = torch.randn(64, 256) * 2.0
    base = mq.fake_quant(w, mbs="dynamic", oas=True)
    got = mq.fake_quant(w, mbs="dynamic", oas=True, chan_w=torch.ones(256))
    assert torch.equal(base, got)


def test_quantlinear_uniform_actamax_equals_baseline():
    # End-to-end at the layer level: MBS-H-ACTW with a flat activation profile == MBS-H.
    torch.manual_seed(1)
    lin = nn.Linear(256, 64, bias=False)
    base = QuantLinear(lin, weight_mbs="dynamic", act_mbs="static", oas=True)
    idea = QuantLinear(lin, weight_mbs="dynamic", act_mbs="static", oas=True,
                       act_amax=torch.ones(256))
    assert torch.equal(base.weight, idea.weight)


def test_weighted_search_matches_bruteforce_argmin():
    # Independent brute force over the 16 mantissa slots, using the (separately tested)
    # inner OAS quant _quant_blocks16 as the primitive.
    torch.manual_seed(2)
    N, K, mb, blk, n_slots = 8, 256, 128, 16, 16
    w = torch.randn(N, K) * 2.0
    a2 = (torch.rand(K) * 5.0 + 0.1) ** 2            # per-input-channel amax^2
    xr = w.reshape(N, K // mb, mb)
    cw = a2.reshape(K // mb, mb)
    factor = mq._mbs_factor_dynamic(xr, oas=True, oas_block=blk, chan_w=cw)  # (N,n_macro,1)
    for n in range(N):
        for g in range(K // mb):
            blk_w = xr[n, g]                          # (mb,)
            aw = cw[g]                                # (mb,)
            best_obj, best_c = float("inf"), None
            for j in range(n_slots):
                c = 1.0 + j / n_slots
                deq = mq._quant_blocks16(blk_w * c, oas=True, block_size=blk) / c
                obj = (aw * (deq - blk_w) ** 2).sum().item()
                if obj < best_obj:
                    best_obj, best_c = obj, c
            assert abs(factor[n, g, 0].item() - best_c) < 1e-9, \
                (n, g, factor[n, g, 0].item(), best_c)


def test_weighted_search_lowers_weighted_error_and_is_not_noop():
    # By construction the amax^2-weighted search minimizes the weighted SSE, so its
    # weighted error must be <= the plain-SSE baseline's; and on random data it should
    # actually pick different factors somewhere (not a silent no-op).
    torch.manual_seed(3)
    w = torch.randn(128, 512) * 2.0
    a2 = (torch.rand(512) * 5.0 + 0.1) ** 2
    base = mq.fake_quant(w, mbs="dynamic", oas=True)
    idea = mq.fake_quant(w, mbs="dynamic", oas=True, chan_w=a2)
    we_base = (a2 * (base - w) ** 2).sum().item()
    we_idea = (a2 * (idea - w) ** 2).sum().item()
    assert we_idea <= we_base + 1e-6
    assert not torch.equal(base, idea)


if __name__ == "__main__":
    import sys
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)

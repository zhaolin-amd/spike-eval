"""Independent, closed-form correctness checks for the MXFP4 fake-quant core.

None of these compares against the paper's reported accuracy numbers (those are
withheld/graded separately). They pin the *algorithm* against properties it must
satisfy by construction: the FP4 grid, the E8M0 scaling ranges (3,6] / (3.5,7],
the MBS mantissa bit-extraction, and the fidelity ordering the paper claims
(MBS improves QSNR over OAS-only, which improves over plain MXFP4).
"""
import math
import struct

import torch

import mxquant as mq


def test_fp4_levels_are_fixed_points():
    lv = mq._FP4_LEVELS
    q = mq.quant_fp4(lv)
    assert torch.equal(q, lv)
    assert torch.equal(mq.quant_fp4(-lv), -lv)


def test_fp4_clamps_and_rounds_nearest():
    x = torch.tensor([7.0, 100.0, -9.0])         # over Fmax -> clamp to +-6
    assert torch.equal(mq.quant_fp4(x), torch.tensor([6.0, 6.0, -6.0]))
    # nearest: 2.4->2, 2.6->3, 4.9->4, 5.1->6, 0.2->0, 0.3->0.5
    x = torch.tensor([2.4, 2.6, 4.9, 5.1, 0.2, 0.3])
    assert torch.equal(mq.quant_fp4(x), torch.tensor([2.0, 3.0, 4.0, 6.0, 0.0, 0.5]))


def test_e8m0_scale_maps_absmax_into_3_6_without_oas():
    # For many random block maxima, standard power-of-two scaling puts amax in (3, 6].
    amax = torch.rand(100000) * 100 + 1e-3
    sf = mq._pow2_scale_oas(amax, oas=False)
    scaled = amax * sf
    assert torch.all(scaled > 3.0 - 1e-4)
    assert torch.all(scaled <= 6.0 + 1e-4)
    # scale is an exact power of two
    log2sf = torch.log2(sf)
    assert torch.allclose(log2sf, log2sf.round(), atol=1e-5)


def test_oas_maps_absmax_into_3p5_7():
    amax = torch.rand(100000) * 100 + 1e-3
    sf = mq._pow2_scale_oas(amax, oas=True)
    scaled = amax * sf
    assert torch.all(scaled > 3.5 - 1e-4)
    assert torch.all(scaled <= 7.0 + 1e-4)


def test_oas_only_differs_in_the_3_to_3p5_window():
    # OAS doubles the scale exactly when the standard scaled absmax is in (3, 3.5].
    amax = torch.rand(200000) * 20 + 1e-3
    sf0 = mq._pow2_scale_oas(amax, oas=False)
    sf1 = mq._pow2_scale_oas(amax, oas=True)
    bumped = sf1 > sf0
    std_scaled = amax * sf0
    in_window = (std_scaled > 3.0) & (std_scaled <= 3.5)
    assert torch.equal(bumped, in_window)
    assert torch.allclose(sf1[bumped], 2.0 * sf0[bumped])


def test_mbs_static_matches_manual_bit_extraction():
    # Reference mantissa extraction from a Python float, independent of torch bit-view.
    def ref_factor(amax):
        sf = 6.0 / amax
        (bits,) = struct.unpack("<I", struct.pack("<f", sf))
        m8 = (bits & 0x007F8000) >> 15
        return 1.0 + m8 / 256.0

    torch.manual_seed(0)
    for _ in range(50):
        block = torch.randn(128) * torch.rand(1).item() * 10 + 1e-3
        xr = block.reshape(1, 1, 128)
        got = mq._mbs_factor_static(xr).item()
        exp = ref_factor(block.abs().amax().item())
        assert abs(got - exp) < 1e-6, (got, exp)
        assert 1.0 <= got < 2.0


def test_mbs_factors_in_unit_interval():
    torch.manual_seed(1)
    x = torch.randn(4, 512)
    xr = x.reshape(4, 4, 128)
    for f in (mq._mbs_factor_static(xr), mq._mbs_factor_dynamic(xr, oas=True)):
        assert torch.all(f >= 1.0) and torch.all(f < 2.0)


def test_zero_block_is_safe():
    x = torch.zeros(2, 256)
    for mbs in ("none", "static", "dynamic"):
        q = mq.fake_quant(x, mbs=mbs, oas=True)
        assert torch.all(q == 0) and not torch.isnan(q).any()


def test_exactly_representable_block_is_lossless():
    # A block whose absmax is 6 and entries land on FP4 grid at scale 1 -> lossless.
    blk = torch.tensor([6.0, 4.0, 3.0, 2.0, 1.5, 1.0, 0.5, 0.0,
                        -6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0])
    q = mq.fake_quant(blk.reshape(1, 16), mbs="none", oas=True)
    assert torch.allclose(q.flatten(), blk, atol=1e-6)


def test_dynamic_never_worse_than_oas_only():
    # j=0 candidate is factor 1.0 == OAS-only, so MBS-Dynamic MSE <= OAS-only MSE.
    torch.manual_seed(2)
    x = torch.randn(64, 1024) * 3.0
    none = mq.fake_quant(x, mbs="none", oas=True)
    dyn = mq.fake_quant(x, mbs="dynamic", oas=True)
    mse_none = ((x - none) ** 2).mean().item()
    mse_dyn = ((x - dyn) ** 2).mean().item()
    assert mse_dyn <= mse_none + 1e-9


def test_mxfp4_16_uses_ocp_overflow_scale_not_oas():
    # Paper 4.1: MXFP4-16 is MX/OCP-style scaling at block size 16 -> the SAME (4,8]
    # overflow scale as MXFP4-OCP, just block 16. It is NOT the non-saturating (3,6]
    # scale, which belongs to OAS (4.2). Regression guard for the pitfall where MXFP4-16
    # was given the (3,6] scale and thereby reproduced the paper's OAS numbers.
    cfg = mq.METHODS["MXFP4-16"]
    assert cfg["ocp"] is True and cfg["ocp_block"] == 16
    torch.manual_seed(0)
    x = torch.randn(32, 512) * 3.0
    x[:, ::129] *= 10.0  # sparse outliers so some blocks would overflow under (4,8]
    mx16 = mq.fake_quant(x, mbs="none", oas=False, ocp=True, ocp_block=16)
    oas = mq.fake_quant(x, mbs="none", oas=True)          # (3.5,7] OAS scale, block 16
    ocp32 = mq.fake_quant(x, mbs="none", oas=False, ocp=True, ocp_block=32)
    # MXFP4-16 must differ from OAS (the bug made them ~equal) and from block-32 OCP.
    assert not torch.allclose(mx16, oas)
    assert not torch.allclose(mx16, ocp32)
    # OCP (4,8] saturates the block max; OAS avoids that -> OAS is at least as faithful.
    assert mq.qsnr_db(x, mx16) <= mq.qsnr_db(x, oas) + 1e-6


def test_fidelity_ordering_qsnr():
    # Paper's central fidelity claim (Fig. QSNR): OAS >= plain MXFP4, and MBS >= OAS.
    torch.manual_seed(3)
    # heavy-tailed data (outliers) is where MBS/OAS help; use a Student-t-ish mix.
    x = torch.randn(128, 2048)
    x[:, ::257] *= 12.0  # inject sparse outliers (<1% of elements)
    plain = mq.qsnr_db(x, mq.fake_quant(x, mbs="none", oas=False))
    oas = mq.qsnr_db(x, mq.fake_quant(x, mbs="none", oas=True))
    mbs_s = mq.qsnr_db(x, mq.fake_quant(x, mbs="static", oas=True))
    mbs_d = mq.qsnr_db(x, mq.fake_quant(x, mbs="dynamic", oas=True))
    assert oas >= plain - 1e-6
    assert mbs_s >= oas - 1e-6
    assert mbs_d >= oas - 1e-6
    # Dynamic is the search-optimal variant -> best (or tied) QSNR of the four.
    assert mbs_d >= mbs_s - 0.05


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

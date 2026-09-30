# Generator for the attention_pv_early variants (extends sage_pv_early_r9600d/tools/mk_e1.py).
# MK_SRC=<original attention_pv_early.s at 19351a68> python3 mk_variants.py out.s <opts>
#   b2x: addr canon lsum ldsa bm256 sa tmax la prio   ec: e1 addr canon n16 lsum ldsa bm256 sa tmax la ec   eb2la: e1 addr canon n16 lsum ldsa bm256 sa tmax la
#   eb2satc: e1 addr canon n16 lsum ldsa bm256 sa tmax   esp321: e1 addr canon n16 lsum ldsa bm256 sa tmax sp321
#!/usr/bin/env python3
"""Generate variants of CSE attention_pv_early.s.

usage: mk_e1.py <out.s> [e1] [tmax]
  e1   : exp-free FP8 P (approximate). Per score, instead of
            x = S*scale - (magic*scale + m')  ->  p = exp2(x)  ->  fp8(p) (pair packs in PV tail)
         emit
            y = S*(8*scale) - (8*(magic*scale + m') - 55.544)  ->  v_cvt_pk_u8_f32 (byte-pack)
         i.e. the e4m3 bit pattern round(8*(s - m) + 126) (piecewise-linear 2^x, 448 at the row max).
         Row sums from the packed bytes: v_dot4_f32_fp8_fp8 with fp8 ones (0x38).
         Packed P lives in v231..v238 (rg0 k0: v231:232, rg0 k1: v233:234, rg1 k0: v235:236, rg1 k1: v237:238);
         the sum bpermute results move from v231/v232 to v240/v241.
  tmax : exact. Tile row max as a depth-3 max3 tree instead of CSE's 8-deep serial max3 chain.
"""
import re
import struct
import sys

outp, opts = sys.argv[1], set(sys.argv[2:])
import os
src = open(os.environ.get("MK_SRC", os.path.join(os.path.dirname(os.path.abspath(__file__)), "cse_pv_early.s"))).read().split("\n")
f32 = lambda x: "0x%08x" % struct.unpack("<I", struct.pack("<f", x))[0]
C8807 = struct.unpack("<f", bytes.fromhex("79e90c41"))[0]      # 0x410ce979
OFF = f32(-(126.0 - 8.0 * C8807))                              # -55.544


def at(n, prefix):
    assert src[n - 1].strip().startswith(prefix), (n, src[n - 1], prefix)


rg0 = [f"v{r}" for r in range(215, 231)]          # S keys 0..15 (v215..222), 16..31 (v223..230)
rg1 = [f"v{r}" for r in range(137, 145)] + [f"v{r}" for r in range(129, 137)]  # keys 0..15 = v137..144, 16..31 = v129..136
out = list(src)

if "e1" in opts:
    at(915, ".Lnocvt_fast"); at(917, "v_fmaak_f32 v254"); at(1042, "v_add_f32_e32 v196, v144, v243")
    at(1043, "v_sub_f32_e32 v213"); at(1044, "ds_bpermute_b32 v231, v207, v200"); at(1045, "ds_bpermute_b32 v232, v207, v196")
    P0 = ["v231", "v232", "v233", "v234"]
    P1 = ["v235", "v236", "v237", "v238"]
    blk = [
        "\tv_fmaak_f32 v254, s34, v196, 0x410ce979",
        "\ts_delay_alu instid0(VALU_DEP_1)",
        "\tv_fma_f32 v209, v209, v196, -v254",
        "\tv_fma_f32 v239, v239, v196, -v254",
        f"\ts_mov_b32 s35, {OFF}",
        "\tv_max_num_f32_e32 v209, v242, v209",
        "\tv_max_num_f32_e32 v210, v243, v239",
        "\tv_mul_f32_e32 v254, 0x41000000, v196",
        "\ts_delay_alu instid0(VALU_DEP_3)",
        "\tv_fma_f32 v252, s34, v196, v209",
        "\tv_fma_f32 v253, s34, v196, v210",
        "\tv_sub_f32_e32 v255, v214, v209",
        "\ts_wait_alu depctr_sa_sdst(0)",
        "\tv_fma_f32 v252, v252, 0x41000000, s35",
        "\tv_fma_f32 v253, v253, 0x41000000, s35",
        "\tv_exp_f32_e32 v255, v255",
        "\ts_delay_alu instid0(VALU_DEP_2)",
    ]
    for i in range(16):
        blk.append(f"\tv_fma_f32 {rg0[i]}, {rg0[i]}, v254, -v252")
        blk.append(f"\tv_fma_f32 {rg1[i]}, {rg1[i]}, v254, -v253")
    if "n16" in opts:
        # e1 variant: x = (256*y + 128)/65535 (constants folded into v254/v252), 2 scores per v_cvt_pk_norm_u16_f32
        # (u16 = round(256*y+128), high byte = round-half-up(y), clamped at 0), then one v_perm_b32 gathers the 4 high
        # bytes: 24 VALU instead of 32 cvt_pk_u8 per 64 scores.
        k = 256.0 / 65535.0
        i = blk.index("\tv_mul_f32_e32 v254, 0x41000000, v196")
        blk[i] = f"\tv_mul_f32_e32 v254, {f32(8 * k)}, v196"
        blk[blk.index(f"\ts_mov_b32 s35, {OFF}")] = f"\ts_mov_b32 s35, {f32(k * (struct.unpack('<f', struct.pack('<f', -(126.0 - 8.0 * C8807)))[0] - 0.5))}"
        for r in ("v252", "v253"):
            j = blk.index(f"\tv_fma_f32 {r}, {r}, 0x41000000, s35")
            blk[j] = f"\tv_fma_f32 {r}, {r}, {f32(8 * k)}, s35"
        for h in (0, 2):
            for d in range(4):
                for S in (rg0, rg1):
                    blk.append(f"\tv_cvt_pk_norm_u16_f32 {S[4 * d + h]}, {S[4 * d + h]}, {S[4 * d + h + 1]}")
        for d in range(4):
            for S, P in ((rg0, P0), (rg1, P1)):
                blk.append(f"\tv_perm_b32 {P[d]}, {S[4 * d + 2]}, {S[4 * d]}, 0x7050301")
    else:
        for b in range(4):                 # byte-major so dependent writes to one dword are 8 instructions apart
            for d in range(4):
                for S, P in ((rg0, P0), (rg1, P1)):
                    src2 = "0" if b == 0 else P[d]
                    blk.append(f"\tv_cvt_pk_u8_f32 {P[d]}, {S[4 * d + b]}, {b}, {src2}")
    blk += [
        "\tv_dot4_f32_fp8_fp8 v200, v231, 0x38383838, 0",
        "\tv_dot4_f32_fp8_fp8 v196, v235, 0x38383838, 0",
        "\ts_delay_alu instid0(VALU_DEP_2)",
        "\tv_dot4_f32_fp8_fp8 v200, v232, 0x38383838, v200",
        "\tv_dot4_f32_fp8_fp8 v196, v236, 0x38383838, v196",
        "\ts_delay_alu instid0(VALU_DEP_2)",
        "\tv_dot4_f32_fp8_fp8 v200, v233, 0x38383838, v200",
        "\tv_dot4_f32_fp8_fp8 v196, v237, 0x38383838, v196",
        "\ts_delay_alu instid0(VALU_DEP_2)",
        "\tv_dot4_f32_fp8_fp8 v200, v234, 0x38383838, v200",
        "\tv_dot4_f32_fp8_fp8 v196, v238, 0x38383838, v196",
    ]
    # PV tail (lines 635..719) first: it precedes the replaced block, so its line numbers are unshifted
    at(639, "v_add_f32_e32 v200, v200, v231"); at(640, "v_add_f32_e32 v196, v196, v232")
    out[638] = "\tv_add_f32_e32 v200, v200, v240"
    out[639] = "\tv_add_f32_e32 v196, v196, v241"
    remap = {"v[227:228]": "v[231:232]", "v[231:232]": "v[233:234]", "v[233:234]": "v[235:236]", "v[229:230]": "v[237:238]"}
    pat = re.compile("|".join(re.escape(k) for k in remap))
    for i in range(634, 719):
        s = out[i].strip()
        if s.startswith("v_cvt_pk_fp8_f32"):
            out[i] = "\t; E1: packed in softmax: " + s
        elif s.startswith("v_wmma_f32_16x16x16_fp8_fp8"):
            out[i] = pat.sub(lambda m: remap[m.group(0)], out[i])
    out = out[:916] + blk + [out[1042], "\tds_bpermute_b32 v240, v207, v200", "\tds_bpermute_b32 v241, v207, v196"] + out[1045:]

if "addr" in opts:
    # exact: strength-reduce the per-tile LDS/global address math. The K row index (s1|v201) = s1 + v201 (s1 is a
    # multiple of 32, v201 < 32), so the 64-bit K address is base_lane + (s1*s[26:27])<<7 with base_lane hoisted
    # before the loop; V address = (v190:191 + v202) + s1. The 64-bit v_mad_co_u64 LDS offsets become s_mul + add.
    # Line numbers <= 916 are unshifted by e1; edits are applied bottom-up.
    at(632, "s_wait_dscnt 0x0"); at(651, "v_mad_co_u64_u32 v[219:220], null, 0x1400, s8, v[192:193]")
    at(728, "s_add_co_i32 s1, s7, 64"); at(732, "s_cselect_b32 s1, s1, 0"); at(749, "ds_store_2addr_b64 v131")
    at(750, "ds_store_2addr_b64 v134"); at(759, "global_load_b128 v[145:148]"); at(760, "global_load_b128 v[149:152]")
    at(764, "v_mad_co_u64_u32 v[209:210], null, 0x1100, s8, v[193:194]")
    W, V = "\ts_wait_alu depctr_sa_sdst(0)", "\ts_wait_alu depctr_va_sdst(0)"
    out[763:764] = ["\ts_mul_i32 s1, s8, 0x1100", W, "\tv_add_nc_u32_e32 v209, s1, v193"]
    out[727:760] = out[727:732] + [
        "\ts_mul_i32 s11, s10, 0x1400", W, "\tv_add3_u32 v134, v203, s11, 0x2200",
        "\ts_mul_i32 s11, s10, 0x1100", W, "\tv_add_nc_u32_e32 v131, s11, v189",
        "\ts_mul_hi_u32 s11, s1, s26", "\ts_mul_i32 s10, s1, s27", "\ts_add_co_i32 s11, s11, s10",
        "\ts_mul_i32 s10, s1, s26", "\ts_lshl_b64 s[10:11], s[10:11], 7", W,
        "\tv_add_co_u32 v129, s35, s10, v205", V, "\tv_add_co_ci_u32_e64 v130, null, s11, v206, s35",
        "\tv_add_co_u32 v132, s35, s1, v190", V, "\tv_add_co_ci_u32_e64 v133, null, 0, v191, s35",
        out[748], out[749], out[758], out[759]]
    out[650:651] = ["\ts_mul_i32 s1, s8, 0x1400", W, "\tv_add_nc_u32_e32 v219, s1, v192"]
    out[631:631] = [
        "\tv_add_co_u32 v215, s35, s14, v201", V, "\tv_add_co_ci_u32_e64 v216, null, s15, 0, s35",
        "\tv_mul_lo_u32 v217, v215, s27", "\tv_mul_lo_u32 v218, v216, s26",
        "\tv_mad_co_u64_u32 v[215:216], null, v215, s26, s[28:29]",
        "\tv_add3_u32 v216, v218, v216, v217", "\tv_lshlrev_b64_e32 v[215:216], 7, v[215:216]",
        "\tv_add_co_u32 v205, s35, v205, v215", V, "\tv_add_co_ci_u32_e64 v206, null, v206, v216, s35",
        "\tv_add_co_u32 v190, s35, v190, v202", V, "\tv_add_co_ci_u32_e64 v191, null, 0, v191, s35"]

if "canon" in opts:
    # exact: drop the 4 NaN-canonicalizing v_max(x,x) copies of the old row max (v214/v213 -> v242/v243) and of the
    # bpermute results (v240/v241); the consumers read the originals (v213/v214 stay live until after the new max).
    i = out.index("\tv_max_num_f32_e32 v242, v214, v214")
    assert out[i + 1:i + 9] == ["\tv_max_num_f32_e32 v243, v213, v213", "\ts_wait_dscnt 0x0",
                                "\tv_max_num_f32_e32 v240, v240, v240", "\tv_max_num_f32_e32 v241, v241, v241",
                                "\ts_delay_alu instid0(VALU_DEP_2)", "\tv_max_num_f32_e32 v209, v209, v240",
                                "\ts_delay_alu instid0(VALU_DEP_2)", "\tv_max_num_f32_e32 v239, v239, v241"], out[i:i + 9]
    out[i:i + 9] = ["\ts_wait_dscnt 0x0", "\tv_max_num_f32_e32 v209, v209, v240", "\tv_max_num_f32_e32 v239, v239, v241"]
    for a, b in (("v209, v242, v209", "v209, v214, v209"), ("v210, v243, v239", "v210, v213, v239")):
        j = out.index("\tv_max_num_f32_e32 " + a)
        out[j] = "\tv_max_num_f32_e32 " + b

if "lsum" in opts:
    # exact up to fp32 reassociation: keep per-lane partial row sums across tiles (both lanes of a row share m and
    # alpha) and do the cross-lane ds_bpermute + add once after the loop instead of every tile.
    t0, t1 = ("v240", "v241") if "e1" in opts else ("v231", "v232")
    for r, t in (("v200", t0), ("v196", t1)):
        out.remove(f"\tds_bpermute_b32 {t}, v207, {r}")
        out.remove(f"\tv_add_f32_e32 {r}, {r}, {t}")
    i = out.index(".LBB0_44:")
    assert out[i + 1] == "\ts_barrier_wait -1"
    out[i + 1:i + 1] = ["\tds_bpermute_b32 v240, v207, v200", "\tds_bpermute_b32 v241, v207, v196", "\ts_wait_dscnt 0x0",
                        "\tv_add_f32_e32 v200, v200, v240", "\tv_add_f32_e32 v196, v196, v241"]

if "dsum" in opts:
    # non-e1: row sums of the packed fp8 P (v_dot4_f32_fp8_fp8 with fp8 ones) in the PV tail instead of the 16-deep
    # fp32 add chain in the softmax. The normalizer becomes the sum of the quantized P that PV actually uses.
    assert "e1" not in opts
    a = out.index(".LBB0_37:                               ; =>This Inner Loop Header: Depth=1")
    b = out.index(".LBB0_41:                               ;   in Loop: Header=BB0_37 Depth=1")
    drop = ("\tv_dual_add_f32 v200,", "\tv_add_f32_e32 v200, v229, v200", "\tv_add_f32_e32 v196, v144, v243")
    n0 = len(out)
    out = out[:a] + [l for l in out[a:b] if not l.startswith(drop)] + out[b:]
    assert n0 - len(out) == 16, n0 - len(out)
    f0, f1 = "\tv_fmac_f32_e32 v200, v212, v255", "\tv_fmac_f32_e32 v196, v211, v213"
    out.remove(f0); out.remove(f1)
    i = out.index("\tv_cvt_pk_fp8_f32 v231.h, v224, v225") + 1
    assert out[i] == "\ts_wait_dscnt 0x2", out[i]
    blk = []
    for k, (p0, p1) in enumerate((("v227", "v233"), ("v228", "v234"), ("v231", "v229"), ("v232", "v230"))):
        c0, c1 = ("0", "0") if k == 0 else ("v200", "v196")
        blk += [f"\tv_dot4_f32_fp8_fp8 v200, {p0}, 0x38383838, {c0}", f"\tv_dot4_f32_fp8_fp8 v196, {p1}, 0x38383838, {c1}",
                "\ts_delay_alu instid0(VALU_DEP_2)"]
    out[i:i] = blk + [f0, f1]

if "ldsa" in opts:
    # exact (needs addr): the 3 V-tile LDS row addresses as one v_add3 each (sgpr + vgpr + literal) instead of
    # base add + 3 offset adds.
    i = out.index("\tv_add_nc_u32_e32 v219, s1, v192")
    del out[i]
    for o, d in (("0x2000", "v137"), ("0x2800", "v129"), ("0x3000", "v129")):
        j = out.index(f"\tv_add_nc_u32_e32 {d}, {o}, v219")
        out[j] = f"\tv_add3_u32 {d}, s1, v192, {o}"

if "ecmp" in opts:
    # exact: issue the large-scale test (v_cmp -> s35) right after the scale multiply so its VALU->SALU latency
    # overlaps the row-max combine instead of stalling at s_wait_alu depctr_va_sdst.
    i = out.index("\tv_cmp_lt_f32_e64 s35, 0x3b2aaaab, v196")
    cmp = out.pop(i)
    j = out.index("\tv_mul_f32_e32 v196, v198, v196")
    out.insert(j + 1, cmp)

if "nocvt0" in opts:
    # approximate (E1 only): drop the large-scale precision slow path (always the magic-bias fast path). The bias
    # rounding error is a per-row constant shift of the fp8 exponent argument.
    i = out.index("\tv_cmp_lt_f32_e64 s35, 0x3b2aaaab, v196")
    j = out.index(".Lnocvt_fast:")
    assert out[i + 1:i + 5] == ["\ts_wait_alu depctr_va_sdst(0)", "\ts_cmp_eq_u32 s35, 0", "\ts_cbranch_scc1 .Lnocvt_fast",
                                "\ts_mov_b32 s34, 0"] and j - i < 45
    del out[i:j]

if "b4" in opts:
    # exact (needs addr): 4 LDS tile buffers (the 0x4A00 K+V region twice, bank = ((t>>1)&1)*0x4A00), tiles stored
    # 2 ahead, so the workgroup barrier is only waited at even iterations and signalled at odd/last ones (half the
    # barriers). Global prefetch goes 3 tiles ahead; a prologue stores tile 1 and loads tile 2.
    W, V = "\ts_wait_alu depctr_sa_sdst(0)", "\ts_wait_alu depctr_va_sdst(0)"
    rep = lambda a, b: out.__setitem__(out.index(a), b)
    rep("\ts_mul_i32 s1, s8, 0x1100", "\ts_mov_b32 s1, s39")                 # QK K base (loop)
    rep("\ts_mul_i32 s1, s8, 0x1400", "\ts_mov_b32 s1, s38")                 # PV V base (loop)
    rep("\ts_add_co_i32 s1, s7, 64", "\ts_add_co_i32 s1, s7, 96")
    i = out.index("\ts_mul_i32 s11, s10, 0x1400")
    out[i:i + 1] = ["\ts_mul_i32 s11, s8, 0x1400", W, "\ts_add_co_i32 s11, s11, s37"]
    i = out.index("\ts_mul_i32 s11, s10, 0x1100")
    out[i:i + 1] = ["\ts_mul_i32 s11, s8, 0x1100", W, "\ts_add_co_i32 s11, s11, s37"]
    # loop top: bank/base SGPRs, barrier wait only at even t
    i = out.index(".LBB0_37:                               ; =>This Inner Loop Header: Depth=1")
    assert out[i + 1:i + 4] == [W, "\ts_and_b32 s8, s6, 1", "\ts_barrier_wait -1"], out[i:i + 4]
    out[i + 3:i + 4] = ["\ts_bfe_u32 s36, s6, 0x10001", W, "\ts_mul_i32 s36, s36, 0x4a00",
                        "\ts_mul_i32 s38, s8, 0x1400", "\ts_mul_i32 s39, s8, 0x1100", W,
                        "\ts_xor_b32 s37, s36, 0x4a00", "\ts_add_co_i32 s38, s38, s36", "\ts_add_co_i32 s39, s39, s36",
                        "\ts_cmp_lg_u32 s8, 0", "\ts_cbranch_scc1 .Lb4nw", "\ts_barrier_wait -1", ".Lb4nw:"]
    # loop signal only at odd t or the last iteration (SCC = last, recomputed after)
    a = out.index(".LBB0_36:                               ;   in Loop: Header=BB0_37 Depth=1")
    j = out.index("\ts_barrier_signal -1", a)
    assert out[j - 1] == "\ts_wait_dscnt 0x0" and j < out.index("\ts_cbranch_scc1 .LBB0_44", a)
    out[j:j + 1] = ["\ts_cselect_b32 s1, 1, s8", W, "\ts_cmp_eq_u32 s1, 0", "\ts_cbranch_scc1 .Lb4ns",
                    "\ts_barrier_signal -1", ".Lb4ns:", "\ts_cmp_ge_i32 s7, s4"]
    # prologue: store tile 1 into buffer 1 (bank 0, slot 1), load tile 2
    i = out.index("\ts_branch .LBB0_37")
    assert out[i - 2:i] == ["\ts_wait_dscnt 0x0", "\ts_barrier_signal -1"]
    out[i - 2:i - 2] = [
        "\ts_and_saveexec_b32 s9, s0", "\ts_cbranch_execz .Lb4p",
        "\ts_wait_loadcnt 0x0",
        "\tv_add_nc_u32_e32 v131, 0x1100, v189", "\tv_add_nc_u32_e32 v134, 0x3600, v203",
        "\tds_store_2addr_b64 v131, v[145:146], v[147:148] offset1:1",
        "\tds_store_2addr_b64 v134, v[149:150], v[151:152] offset1:1",
        "\ts_mov_b32 s1, 64", W, "\ts_cmp_lt_i32 s1, s12", "\ts_cselect_b32 s1, s1, 0", W,
        "\ts_mul_hi_u32 s11, s1, s26", "\ts_mul_i32 s10, s1, s27", W, "\ts_add_co_i32 s11, s11, s10",
        "\ts_mul_i32 s10, s1, s26", W, "\ts_lshl_b64 s[10:11], s[10:11], 7", W,
        "\tv_add_co_u32 v129, s35, s10, v205", V, "\tv_add_co_ci_u32_e64 v130, null, s11, v206, s35",
        "\tv_add_co_u32 v132, s35, s1, v190", V, "\tv_add_co_ci_u32_e64 v133, null, 0, v191, s35",
        "\tglobal_load_b128 v[145:148], v[129:130], off", "\tglobal_load_b128 v[149:152], v[132:133], off",
        ".Lb4p:", W, "\ts_or_b32 exec_lo, exec_lo, s9"]
    # tail: tile T lives in slot T&1 of bank (T>>1)&1
    i = out.index("\ts_and_b32 s24, s1, 1")
    out[i + 1:i + 1] = ["\ts_bfe_u32 s36, s1, 0x10001", W, "\ts_mul_i32 s36, s36, 0x4a00",
                        "\ts_mul_i32 s37, s24, 0x1100", "\ts_mul_i32 s38, s24, 0x1400", W,
                        "\ts_add_co_i32 s37, s37, s36", "\ts_add_co_i32 s38, s38, s36"]
    rep("\ts_mul_i32 s0, s24, 0x1100", "\ts_mov_b32 s0, s37")
    rep("\ts_mulk_i32 s24, 0x1400", "\ts_mov_b32 s24, s38")
    rep("\tv_or_b32_e32 v152, s24, v161", "\tv_add_nc_u32_e32 v152, s24, v161")
    if "b4x" in opts:
        # diagnostic (exact): same 4-buffer layout but barrier every iteration, to split layout cost from sync
        out.remove("\ts_cbranch_scc1 .Lb4nw"); out.remove("\ts_cbranch_scc1 .Lb4ns")
    assert sum(".amdhsa_group_segment_fixed_size 18944" in l or ".group_segment_fixed_size: 18944" in l
               for l in out) == 2
    for k, l in enumerate(out):
        l = l.replace("group_segment_fixed_size 18944", "group_segment_fixed_size 37888")
        l = l.replace("group_segment_fixed_size: 18944", "group_segment_fixed_size: 37888")
        l = l.replace(".amdhsa_next_free_sgpr 36", ".amdhsa_next_free_sgpr 40").replace(".sgpr_count:     38", ".sgpr_count:     42")
        out[k] = l

if "ldsbig" in opts:
    # diagnostic (exact): only raise the LDS allocation to b4's 37888 bytes
    out = [l.replace("group_segment_fixed_size 18944", "group_segment_fixed_size 37888")
            .replace("group_segment_fixed_size: 18944", "group_segment_fixed_size: 37888") for l in out]

for o in opts:
    if o.startswith("xdrop"):
        # DIAGNOSTIC ONLY (wrong results): delete every hot-loop line starting with one of the '+'-separated prefixes
        # (e.g. xdrop+v_fma_f32+v_cvt_pk_norm) to bound what those instructions cost.
        pre = tuple("\t" + p for p in o.split("+")[1:])
        a = out.index(".LBB0_36:                               ;   in Loop: Header=BB0_37 Depth=1")
        b = out.index("\ts_cbranch_execz .LBB0_41")
        out = out[:a] + [l for l in out[a:b] if not l.startswith(pre)] + out[b:]

if "xnorescale" in opts:
    # DIAGNOSTIC ONLY (wrong results): never take the O-rescale blocks, to bound their cost.
    for r in ("v255", "v213"):
        i = out.index(f"\tv_cmpx_neq_f32_e32 1.0, {r}")
        out[i] = f"\tv_cmpx_lt_f32_e32 0x7f000000, {r}"

if "bwait" in opts:
    # exact: sink the loop-top barrier wait (and the prefetch loadcnt wait) from the loop header to just before
    # the LDS stores; waves that don't store wait on a separate path. Line numbers < 916 are unshifted by e1.
    at(723, "s_barrier_wait -1"); at(724, "s_wait_loadcnt 0x0"); at(726, "s_cbranch_execz .LBB0_39")
    at(749, "ds_store_2addr_b64 v131"); at(760, "global_load_b128 v[149:152]"); at(761, ".LBB0_39:")
    out = (out[:722] + [out[724], "\ts_cbranch_execz .Lbw_skip"] + out[726:748]
           + ["\ts_wait_loadcnt 0x0", "\ts_barrier_wait -1"] + out[748:760]
           + ["\ts_branch .LBB0_39", ".Lbw_skip:", "\ts_barrier_wait -1"] + out[760:])

if "tmax" in opts:
    i0 = out.index("\tv_max_num_f32_e32 v209, v215, v216")
    i1 = out.index("\tds_bpermute_b32 v240, v207, v209")
    assert i1 - i0 == 30 and all(("max" in l or "s_delay_alu" in l) for l in out[i0:i1])

    def tree(a, dst, t):
        # 5 independent max3 over 15 inputs, then 2 max3, then 1 max: depth 3
        return ([f"v_max3_num_f32 {t[j]}, {a[3*j]}, {a[3*j+1]}, {a[3*j+2]}" for j in range(5)],
                [f"v_max3_num_f32 {t[0]}, {t[0]}, {t[1]}, {t[2]}", f"v_max3_num_f32 {t[3]}, {t[3]}, {t[4]}, {a[15]}"],
                [f"v_max_num_f32_e32 {dst}, {t[0]}, {t[3]}"])
    A = tree(rg0, "v209", ["v209", "v240", "v241", "v252", "v253"])
    B = tree(rg1, "v239", ["v239", "v242", "v243", "v254", "v255"])
    blk = []
    for lvl in range(3):
        for x, y in zip(A[lvl], B[lvl]):
            blk += ["\t" + x, "\t" + y]
        if lvl < 2:
            blk.append("\ts_delay_alu instid0(VALU_DEP_2)")
    out = out[:i0] + blk + out[i1:]

if "sa" in opts or "bm128" in opts:
    # exact (needs addr): K/V prefetch loads in SGPR-base + 32-bit lane-offset form (global_load v, voff, s[base]).
    # Tile-invariant bases K s[36:37], V s[38:39] are computed once; v205 / v190 become 32-bit lane offsets, which
    # replaces the per-tile 64-bit VALU adds with SALU and frees v206 / v191.
    W, V = "\ts_wait_alu depctr_sa_sdst(0)", "\ts_wait_alu depctr_va_sdst(0)"
    i = out.index("\tv_add3_u32 v191, v3, v191, v4") + 1
    out[i:i] = ["\ts_mul_u64 s[36:37], s[14:15], s[26:27]", W, "\ts_add_nc_u64 s[36:37], s[36:37], s[28:29]", W,
                "\ts_lshl_b64 s[36:37], s[36:37], 7", W, "\ts_add_nc_u64 s[36:37], s[36:37], s[6:7]",
                "\ts_mul_u64 s[38:39], s[10:11], s[12:13]", W, "\ts_add_nc_u64 s[38:39], s[38:39], s[8:9]",
                "\ts_lshl_b32 s40, s26, 11", "\ts_lshl_b32 s41, s12, 6"]
    i = out.index("\tv_add_co_u32 v215, s35, s14, v201")
    assert out[i + 14] == "\ts_wait_dscnt 0x0", out[i:i + 15]
    out[i:i + 14] = ["\tv_mul_u32_u24_e32 v205, s26, v201", "\tv_lshlrev_b32_e32 v206, 4, v0",
                     "\tv_mul_lo_u32 v190, v197, s12", "\tv_and_b32_e32 v206, 0x70, v206",
                     "\tv_lshl_add_u32 v205, v205, 7, v206", "\tv_add_nc_u32_e32 v190, v190, v202"]
    i = out.index("\tv_add_co_u32 v129, s35, s10, v205")
    assert out[i + 6:i + 10] == ["\tds_store_2addr_b64 v131, v[145:146], v[147:148] offset1:1",
                                 "\tds_store_2addr_b64 v134, v[149:150], v[151:152] offset1:1",
                                 "\tglobal_load_b128 v[145:148], v[129:130], off", "\tglobal_load_b128 v[149:152], v[132:133], off"]
    ka = ["\ts_add_co_u32 s10, s10, s40", "\ts_add_co_ci_u32 s11, s11, 0", "\ts_add_co_i32 s1, s1, s41"] if "bm128" in opts else []
    out[i:i + 10] = ka + ["\ts_add_nc_u64 s[10:11], s[10:11], s[36:37]", "\ts_add_co_u32 s42, s38, s1",
                          "\ts_add_co_ci_u32 s43, s39, 0", W] + out[i + 6:i + 8] + [
        "\tglobal_load_b128 v[145:148], v205, s[10:11]", "\tglobal_load_b128 v[149:152], v190, s[42:43]"]
    out = [l.replace(".amdhsa_next_free_sgpr 36", ".amdhsa_next_free_sgpr 44").replace(".sgpr_count:     38", ".sgpr_count:     46")
           for l in out]

if "bm128" in opts:
    # exact (implies sa): 128-thread workgroups (4 waves, 128 query rows). 128 lanes stage each 8 KB K+V tile in two
    # halves with the same 8 prefetch VGPRs: part A (K rows 0-15, V d-rows 0-63) and part B (K rows 16-31, V d-rows
    # 64-127, LDS +0x880 / +0xa00, global +16 K rows (s40) / +64 V rows (s41)). Loop top: store A(t+1), load B(t+1);
    # after the per-tile scale wait (which already waits on all loads): store B(t+1), load A(t+2). Both halves of
    # tile t+1 are stored before this tile's barrier signal. The prologue also stages tile 0 part B.
    W, V = "\ts_wait_alu depctr_sa_sdst(0)", "\ts_wait_alu depctr_va_sdst(0)"
    i = out.index("\ts_add_co_i32 s1, s7, 64")        # loop-top prefetch now loads tile t+1 (part B)
    out[i] = "\ts_add_co_i32 s1, s7, 32"
    i = out.index("\tv_mul_f32_e32 v196, v198, v196")
    assert out[i - 1] == "\ts_wait_loadcnt 0x0", out[i - 1]
    out[i:i] = ["\ts_xor_b32 s10, s8, 1", W, "\ts_mul_i32 s11, s10, 0x1400", "\ts_mul_i32 s10, s10, 0x1100", W,
                "\tv_add3_u32 v202, v203, s11, 0x2c00", "\tv_add3_u32 v201, v189, s10, 0x880",
                "\tds_store_2addr_b64 v201, v[145:146], v[147:148] offset1:1",
                "\tds_store_2addr_b64 v202, v[149:150], v[151:152] offset1:1",
                "\ts_add_co_i32 s1, s7, 64", W, "\ts_cmp_lt_i32 s1, s12", "\ts_cselect_b32 s1, s1, 0", W,
                "\ts_mul_hi_u32 s11, s1, s26", "\ts_mul_i32 s10, s1, s27", W, "\ts_add_co_i32 s11, s11, s10",
                "\ts_mul_i32 s10, s1, s26", W, "\ts_lshl_b64 s[10:11], s[10:11], 7", W,
                "\ts_add_nc_u64 s[10:11], s[10:11], s[36:37]", "\ts_add_co_u32 s42, s38, s1", "\ts_add_co_ci_u32 s43, s39, 0", W,
                "\tglobal_load_b128 v[145:148], v205, s[10:11]", "\tglobal_load_b128 v[149:152], v190, s[42:43]"]
    # prologue: tile 0 part B (K addr v[7:8], V addr v[3:4]; K LDS v10, V LDS v9; v16..v27 are free until zeroed)
    i = out.index("\tglobal_load_b128 v[1:4], v[3:4], off")
    out[i:i] = ["\ts_lshl_b32 s40, s26, 11", "\ts_lshl_b32 s41, s12, 6", W,
                "\tv_add_co_u32 v24, s1, v7, s40", V, "\tv_add_co_ci_u32_e64 v25, null, 0, v8, s1",
                "\tv_add_co_u32 v26, s1, v3, s41", V, "\tv_add_co_ci_u32_e64 v27, null, 0, v4, s1"]
    i = out.index("\tglobal_load_b128 v[149:152], v[11:12], off") + 1
    assert out[i:i + 6] == ["\tv_mul_u32_u24_e32 v9, 40, v197", "\tv_mad_u32_u24 v10, 0x88, v13, v14",
                            "\ts_delay_alu instid0(VALU_DEP_2)", "\tv_add3_u32 v9, v9, v15, 0x2200", "\ts_wait_loadcnt 0x2",
                            "\tds_store_2addr_b64 v10, v[5:6], v[7:8] offset1:1"], out[i:i + 6]
    out[i:i] = ["\tglobal_load_b128 v[16:19], v[24:25], off", "\tglobal_load_b128 v[20:23], v[26:27], off"]
    out[i + 6] = "\ts_wait_loadcnt 0x0"
    j = out.index("\tds_store_2addr_b64 v9, v[1:2], v[3:4] offset1:1", i) + 1
    out[j:j] = ["\tv_add_nc_u32_e32 v24, 0x880, v10", "\tv_add_nc_u32_e32 v25, 0xa00, v9",
                "\tds_store_2addr_b64 v24, v[16:17], v[18:19] offset1:1", "\tds_store_2addr_b64 v25, v[20:21], v[22:23] offset1:1"]

for bm, lg in (("bm256", 8), ("bm128", 7)):
    if bm not in opts:
        continue
    # exact: 256- (128-) thread workgroups (8 (4) waves, as many query rows) instead of 512, launched with that block
    # size and rows per block. All lanes then take part in the K/V staging (s0 = tid < 256), and several workgroups
    # are resident per WGP, each with its own barrier. Only the q-block decomposition depends on 512.
    m = hex((1 << lg) - 1)
    for n, a, b in ((15, "s_add_co_i32 s2, s25, 0x1ff", f"s_add_co_i32 s2, s25, {m}"), (17, "s_lshr_b32 s2, s2, 9", f"s_lshr_b32 s2, s2, {lg}"),
                    (107, "s_add_co_i32 s5, s25, 0x1ff", f"s_add_co_i32 s5, s25, {m}"), (131, "s_ashr_i32 s5, s5, 9", f"s_ashr_i32 s5, s5, {lg}"),
                    (192, "s_lshl_b32 s3, s3, 9", f"s_lshl_b32 s3, s3, {lg}")):
        at(n, a)
        out[out.index("\t" + a)] = "\t" + b

for o in opts:
    if o.startswith("lds") and o[3:].isdigit():
        # diagnostic (exact): set the LDS allocation to N bytes, to cap how many workgroups fit per WGP
        out = [re.sub(r"(group_segment_fixed_size:?\s+)\d+", lambda m: m.group(1) + o[3:], l) for l in out]

for o in opts:
    if o.startswith("stg"):
        # exact: stagger the waves once before the loop. stg<bit>s<N>: waves whose hardware wave id has bit <bit>
        # set sleep N*64 clk after the prologue barrier signal, so they run behind the others. The split barrier
        # allows a steady-state skew of up to one PV WMMA phase, which lets one half's softmax latency overlap the
        # other half's WMMA. s35 is dead at loop entry (it is written before being read in the loop).
        bit, n = o[3:].split("s")
        i = out.index("\ts_branch .LBB0_37")
        out[i:i] = [f"\ts_getreg_b32 s35, hwreg(HW_REG_WAVE_HW_ID1, {bit}, 1)", "\ts_cmp_eq_u32 s35, 0",
                    "\ts_cbranch_scc1 .Lstg", f"\ts_sleep {n}", ".Lstg:"]

for o in opts:
    if o.startswith("pad"):
        # exact: N x 4-byte s_nop before the loop (executed once) to probe loop code-alignment sensitivity
        i = out.index("\ts_branch .LBB0_37")
        out = out[:i] + ["\ts_nop 0"] * int(o[3:]) + out[i:]


# ---------------------------------------------------------------- attn2 additions (jpv) ----------------------------
def _bump_sgpr(out, n):
    def f(l):
        m = re.search(r"\.amdhsa_next_free_sgpr (\d+)", l)
        if m and int(m.group(1)) < n:
            l = l.replace(m.group(0), f".amdhsa_next_free_sgpr {n}")
        m = re.search(r"\.sgpr_count:\s+(\d+)", l)
        if m and int(m.group(1)) < n + 2:
            l = re.sub(r"(\.sgpr_count:\s+)\d+", lambda mm: mm.group(1) + str(n + 2), l)
        return l
    return [f(l) for l in out]



if "ec" in opts:
    # exact-code E1 (needs e1 n16): u16 = round(256*c) (no +0.5), then per score add LUT[floor(c) & 7] (a per-bin
    # rounding threshold, 128 - 256*delta_{k+1}) to the u16 and take the high byte. This reproduces the exact
    # kernel's RNE FP8 code of exp2(x) (0.15% mismatches, each by one code, from the 1/256-code u16 resolution),
    # i.e. exact P at E1's cost + 7 VALU per 4 scores. LUT bytes live in s47 (idx 0-3) / s48 (idx 4-7).
    assert "e1" in opts and "n16" in opts
    k = 256.0 / 65535.0
    old = f"\ts_mov_b32 s35, {f32(k * (struct.unpack('<f', struct.pack('<f', -(126.0 - 8.0 * C8807)))[0] - 0.5))}"
    i = out.index(old)
    out[i] = f"\ts_mov_b32 s35, {f32(k * struct.unpack('<f', struct.pack('<f', -(126.0 - 8.0 * C8807)))[0])}"
    import math
    L448 = math.log2(448.0)
    def val(code):
        E, M = code >> 3, code & 7
        return M * 2.0 ** -9 if E == 0 else 2.0 ** (E - 7) * (1 + M / 8)
    delta = {}
    for kk in range(9, 127):
        b = 0.5 * (val(kk - 1) + val(kk))
        delta.setdefault(kk % 8, 126 + 8 * (math.log2(b) - L448) - (kk - 0.5))
    lut = [int(round(128 - 256 * delta[(j + 1) % 8])) for j in range(8)]
    assert all(0 <= x < 256 for x in lut), lut
    w0 = lut[0] | lut[1] << 8 | lut[2] << 16 | lut[3] << 24
    w1 = lut[4] | lut[5] << 8 | lut[6] << 16 | lut[7] << 24
    j = out.index("\ts_mov_b32 s7, 0")
    out[j + 1:j + 1] = [f"\ts_mov_b32 s47, 0x{w0:08x}", f"\ts_mov_b32 s48, 0x{w1:08x}"]
    out = _bump_sgpr(out, 49)
    chains = []
    for S, P in ((rg0, ["v231", "v232", "v233", "v234"]), (rg1, ["v235", "v236", "v237", "v238"])):
        for d in range(4):
            line = f"\tv_perm_b32 {P[d]}, {S[4 * d + 2]}, {S[4 * d]}, 0x7050301"
            assert line in out, line
            chains.append((line, S[4 * d], S[4 * d + 2], S[4 * d + 1], S[4 * d + 3], P[d]))
    first = min(out.index(c[0]) for c in chains)
    for c in chains:
        out.remove(c[0])
    stages = [lambda A, B, T1, T2, P: f"\tv_perm_b32 {P}, {B}, {A}, 0x7050301",
              lambda A, B, T1, T2, P: f"\tv_and_b32_e32 {T1}, 0x7070707, {P}",
              lambda A, B, T1, T2, P: f"\tv_perm_b32 {T2}, s48, s47, {T1}",
              lambda A, B, T1, T2, P: f"\tv_perm_b32 {T1}, 0, {T2}, 0xc010c00",
              lambda A, B, T1, T2, P: f"\tv_perm_b32 {P}, 0, {T2}, 0xc030c02",
              lambda A, B, T1, T2, P: f"\tv_pk_add_u16 {A}, {A}, {T1}",
              lambda A, B, T1, T2, P: f"\tv_pk_add_u16 {B}, {B}, {P}",
              lambda A, B, T1, T2, P: f"\tv_perm_b32 {P}, {B}, {A}, 0x7050301"]
    blk = []
    for st in stages:
        for (_, A, B, T1, T2, P) in chains:
            blk.append(st(A, B, T1, T2, P))
    out[first:first] = blk


if "la" in opts:
    # lazy alpha: the per-row rescale factor exp2(m_old - m_new) is only computed (TRANS, quarter rate) in the
    # rescale branch, which is entered when some lane's max changed (d != 0); l_old is scaled there too, and the
    # tail l update becomes an add. Exact up to fp32 rounding (mul+add instead of fma) in the rescaled lanes.
    a = out.index(".LBB0_37:                               ; =>This Inner Loop Header: Depth=1")
    b = out.index(".LBB0_43:")
    for r in ("v255", "v213"):
        i = out.index(f"\tv_exp_f32_e32 {r}, {r}", a)
        assert i < b
        del out[i]
    i = out.index("\tv_cmpx_neq_f32_e32 1.0, v255", a)
    out[i] = "\tv_cmpx_neq_f32_e32 0, v255"
    assert out[i + 1] == "\ts_cbranch_execz .LBB0_41"
    out[i + 2:i + 2] = ["\tv_exp_f32_e32 v255, v255", "\ts_delay_alu instid0(TRANS32_DEP_1)", "\tv_mul_f32_e32 v212, v212, v255"]
    i = out.index("\tv_cmpx_neq_f32_e32 1.0, v213", a)
    out[i] = "\tv_cmpx_neq_f32_e32 0, v213"
    assert out[i + 1] == "\ts_cbranch_execz .LBB0_36"
    out[i + 2:i + 2] = ["\tv_exp_f32_e32 v213, v213", "\ts_delay_alu instid0(TRANS32_DEP_1)", "\tv_mul_f32_e32 v211, v211, v213"]
    t = out.index(".LBB0_36:                               ;   in Loop: Header=BB0_37 Depth=1")
    for f, g in (("\tv_fmac_f32_e32 v200, v212, v255", "\tv_add_f32_e32 v200, v200, v212"),
                 ("\tv_fmac_f32_e32 v196, v211, v213", "\tv_add_f32_e32 v196, v196, v211")):
        j = out.index(f, t)
        assert j < a
        out[j] = g


if "dsum2" in opts:
    # exact exp, row sum from the fp8-quantized P: the 16 cvt_pk_fp8 move from the PV tail to the end of the softmax,
    # followed by 8 v_dot4_f32_fp8_fp8 (2 chains of 4) that replace the 30-add fp32 chain.
    assert "e1" not in opts and "dsum" not in opts
    a = out.index(".LBB0_37:                               ; =>This Inner Loop Header: Depth=1")
    b = out.index(".LBB0_41:                               ;   in Loop: Header=BB0_37 Depth=1")
    drop = ("\tv_dual_add_f32 v200,", "\tv_add_f32_e32 v200, v229, v200", "\tv_add_f32_e32 v196, v144, v243")
    n0 = len(out)
    out = out[:a] + [l for l in out[a:b] if not l.startswith(drop)] + out[b:]
    assert n0 - len(out) == 16, n0 - len(out)
    t0 = out.index(".LBB0_36:                               ;   in Loop: Header=BB0_37 Depth=1")
    t1 = next(k for k in range(t0, len(out)) if out[k].startswith("\tv_wmma_f32_16x16x16_fp8_fp8"))
    cv = [l for l in out[t0:t1] if l.startswith("\tv_cvt_pk_fp8_f32")]
    assert len(cv) == 16, len(cv)
    out = out[:t0] + [l for l in out[t0:t1] if not l.startswith("\tv_cvt_pk_fp8_f32")] + out[t1:]
    blk = list(cv)
    for k, (p0, p1) in enumerate((("v227", "v233"), ("v228", "v234"), ("v231", "v229"), ("v232", "v230"))):
        c0, c1 = ("0", "0") if k == 0 else ("v200", "v196")
        blk += [f"\tv_dot4_f32_fp8_fp8 v200, {p0}, 0x38383838, {c0}", f"\tv_dot4_f32_fp8_fp8 v196, {p1}, 0x38383838, {c1}"]
        if k < 3:
            blk.append("\ts_delay_alu instid0(VALU_DEP_2)")
    i = out.index("\tv_sub_f32_e32 v213, v213, v210")
    out[i:i] = blk

PL = "s46"   # permlane identity selector (s36..s43 are taken by sa/bm128)
if "plx" in opts:
    # exact: row-max (and, without lsum, row-sum) cross-lane exchange with v_permlanex16 (lane ^ 16) instead of
    # ds_bpermute through LDS.
    i = out.index("\ts_mov_b32 s7, 0")
    out.insert(i + 1, f"\ts_mov_b32 {PL}, 0x76543210")
    n = 0
    for k, l in enumerate(out):
        m = re.match(r"\tds_bpermute_b32 (v\d+), v207, (v\d+)$", l)
        if m:
            # keep the post-loop lsum exchange as is (outside the hot loop)
            j = out.index(".LBB0_44:")
            if k > j and k < j + 4:
                continue
            out[k] = f"\tv_permlanex16_b32 {m.group(1)}, {m.group(2)}, {PL}, 0xfedcba98"
            n += 1
    assert n >= 2, n
    out = _bump_sgpr(out, 47)

if "sload" in opts:
    # exact: per-tile K scale via s_load (scalar cache) instead of global_load + s_wait_loadcnt 0 (which also drained
    # the K/V prefetch); v_mul reads the SGPR.
    i = out.index("\ts_add_nc_u64 s[10:11], s[16:17], s[10:11]")
    out.insert(i + 1, "\ts_load_b32 s10, s[10:11], 0x0")
    out.remove("\tglobal_load_b32 v196, v204, s[10:11]")
    i = out.index("\tv_mul_f32_e32 v196, v198, v196")
    if "bm128" in opts:
        # bm128 needs the loadcnt wait (all K/V loads landed) for its mid-tile staging, which reuses s10/s11
        j = i
        while not out[j].startswith("\ts_xor_b32 s10, s8, 1"):
            j -= 1
        assert out[j - 1] == "\ts_wait_loadcnt 0x0", out[j - 1]
        del out[i]
        out[j:j] = ["\ts_wait_kmcnt 0x0", "\tv_mul_f32_e32 v196, s10, v198"]
    else:
        assert out[i - 1] == "\ts_wait_loadcnt 0x0", out[i - 1]
        out[i - 1:i + 1] = ["\ts_wait_kmcnt 0x0", "\tv_mul_f32_e32 v196, s10, v198"]

for o in opts:
    if o.startswith("prio"):
        # s_setprio <hi> from the start of the QK phase through the softmax, back to <lo> before rescale/PV.
        # prio = prio2 ; prioN[_lo] ; 'priosm' variants move the start to the softmax only.
        spec = o[4:] or "2"
        hi, lo = (spec.split("_") + ["0"])[:2]
        start = "qk"
        if hi.startswith("sm"):
            start, hi = "sm", hi[2:] or "2"
        a = out.index(".LBB0_39:                               ;   in Loop: Header=BB0_37 Depth=1")
        assert out[a + 2] == "\ts_or_b32 exec_lo, exec_lo, s9", out[a:a + 3]
        if start == "qk":
            out.insert(a + 3, f"\ts_setprio {hi}")
        else:
            j = next(k for k in range(a, len(out)) if out[k].startswith("\tv_max3_num_f32") or out[k].startswith("\tv_max_num_f32_e32 v209, v215"))
            out.insert(j, f"\ts_setprio {hi}")
        j = next(k for k, l in enumerate(out) if l.startswith("\tv_cmpx_neq_f32_e32") and l.endswith(", v255"))
        out.insert(j, f"\ts_setprio {lo}")

for o in opts:
    if o.startswith("sp") and o[2:].isdigit():
        # exact: wave priority per phase. sp<X><Y>: s_setprio X after the last QK WMMA (softmax VALU phase),
        # s_setprio Y at the top of the PV block (.LBB0_36; PV WMMA then QK WMMA of the next tile).
        # sp<X><Y><Z>: also s_setprio Z just before the first QK WMMA of each tile.
        # esp321 = eb2satc + sp321: softmax VALU of one wave overlaps the PV WMMA of the others (-0.8% vs eb2satc).
        x, y = o[2], o[3]
        a = out.index(".LBB0_36:                               ;   in Loop: Header=BB0_37 Depth=1")
        b = out.index("\ts_cbranch_execz .LBB0_41")
        qk = [i for i in range(a, b) if out[i].startswith("\tv_wmma_i32_16x16x16_iu8")]
        out[qk[-1] + 1:qk[-1] + 1] = [f"\ts_setprio {x}"]
        if len(o) > 4:
            out[qk[0]:qk[0]] = [f"\ts_setprio {o[4]}"]
        out[a + 1:a + 1] = [f"\ts_setprio {y}"]

open(outp, "w").write("\n".join(out))

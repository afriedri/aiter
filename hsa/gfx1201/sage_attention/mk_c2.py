#!/usr/bin/env python3
# Track C2 generator: esp321y (mk_c.py recipe) + C2 post-passes.
# usage: mk_c2.py <out.s> [C2 opts...] [-- extra mk_c opts]
# esp321y base: e1 addr canon n16 lsum ldsa bm256 sa tmax sp321 plx sload lax lds=65536 pr=3,1,2 ins=sm:3
import os, re, struct, subprocess, sys

BASE = "e1 addr canon n16 lsum ldsa bm256 sa tmax sp321 plx sload lax lds=65536 pr=3,1,2 ins=sm:3".split()
out_path = sys.argv[1]
args = sys.argv[2:]
extra = []
if "--" in args:
    i = args.index("--")
    args, extra = args[:i], args[i + 1:]
opts = args
here = os.path.dirname(os.path.abspath(__file__))
tmp = out_path + ".c.s"
env = dict(os.environ, MK_SRC=os.environ.get("MK_SRC", os.path.join(here, "cse_pv_early.s")))
if any(e.startswith("lds=") for e in extra):
    BASE = [b for b in BASE if not b.startswith("lds=")]
subprocess.check_call([sys.executable, os.path.join(here, "mk_c.py"), tmp] + BASE + extra, env=env,
                      stdout=subprocess.DEVNULL)
out = open(tmp).read().split("\n")
os.remove(tmp)

HDR = ".LBB0_37:                               ; =>This Inner Loop Header: Depth=1"
TAIL = ".LBB0_36:                               ;   in Loop: Header=BB0_37 Depth=1"
W = "\ts_wait_alu depctr_sa_sdst(0)"
f32 = lambda x: "0x%08x" % struct.unpack("<I", struct.pack("<f", x))[0]
uf32 = lambda h: struct.unpack("<f", struct.pack("<I", int(h, 16)))[0]


def idx(line, start=0, end=None):
    i = out.index(line, start)
    assert end is None or i < end, (line, i, end)
    return i


def bump_sgpr(n):
    global out
    def f(l):
        m = re.search(r"\.amdhsa_next_free_sgpr (\d+)", l)
        if m and int(m.group(1)) < n:
            l = l.replace(m.group(0), f".amdhsa_next_free_sgpr {n}")
        m = re.search(r"\.sgpr_count:\s+(\d+)", l)
        if m and int(m.group(1)) < n + 2:
            l = re.sub(r"(\.sgpr_count:\s+)\d+", lambda mm: mm.group(1) + str(n + 2), l)
        return l
    out = [f(l) for l in out]


if "ss" in opts:
    # SALU scale: the per-tile scale c = k_scale * q_scale is wave-uniform (q scale per 32 query rows = the wave's
    # 32 rows; lane 0 is the wave's first row, always valid if any row is). c, the exact-path test (c > 1/384), the
    # bias constants and the element scale k = c/32*(1+2^-16) move to SALU float ops; the element fma reads k from an
    # SGPR; the per-row bias b = m'*k32 + T with T = s34*k - 0.2189 (one VALU fma per row group instead of two).
    # s47 = q scale (readfirstlane before the loop), s48 = c, s49 = s34*c + 8.807, s50 = k, s51 = T.
    a = out.index(TAIL)
    b = out.index(".LBB0_43:")
    i = idx("\ts_branch .LBB0_37")
    out[i:i] = ["\ts_wait_loadcnt 0x0", "\tv_readfirstlane_b32 s47, v198"]
    a += 2; b += 2
    i = idx("\tv_mul_f32_e32 v196, s10, v198", a, b)
    out[i] = "\ts_mul_f32 s48, s10, s47"
    i = idx("\tv_cmp_lt_f32_e64 s35, 0x3b2aaaab, v196", a, b)
    assert out[i - 1] == "\ts_mov_b32 s34, 0x4b400000", out[i - 1]
    assert out[i + 1:i + 4] == ["\ts_wait_alu depctr_va_sdst(0)", "\ts_cmp_eq_u32 s35, 0", "\ts_cbranch_scc1 .Lnocvt_fast"]
    out[i:i + 4] = [W, "\ts_cmp_gt_f32 s48, 0x3b2aaaab", "\ts_cbranch_scc0 .Lnocvt_fast"]
    j = idx(".Lnocvt_fast:", a, b)
    K32 = "0x3d000080"
    exp_blk = ["\ts_wait_alu depctr_sa_sdst(0)",
               "\tv_fmaak_f32 v254, s34, v196, 0x410ce979",
               "\ts_delay_alu instid0(VALU_DEP_1)",
               "\tv_fma_f32 v209, v209, v196, -v254",
               "\tv_fma_f32 v239, v239, v196, -v254",
               "\ts_mov_b32 s35, 0xbe602dee",
               "\tv_max_num_f32_e32 v209, v214, v209",
               "\tv_max_num_f32_e32 v210, v213, v239",
               f"\tv_mul_f32_e32 v254, {K32}, v196",
               "\ts_delay_alu instid0(VALU_DEP_3)",
               "\tv_fma_f32 v252, s34, v196, v209",
               "\tv_fma_f32 v253, s34, v196, v210",
               "\tv_sub_f32_e32 v255, v214, v209",
               "\ts_wait_alu depctr_sa_sdst(0)",
               f"\tv_fma_f32 v252, v252, {K32}, s35",
               f"\tv_fma_f32 v253, v253, {K32}, s35",
               "\ts_delay_alu instid0(VALU_DEP_2)"]
    assert out[j + 1:j + 1 + len(exp_blk)] == exp_blk, out[j + 1:j + 1 + len(exp_blk)]
    new = ["\ts_fmaak_f32 s49, s34, s48, 0x410ce979",
           f"\ts_mul_f32 s50, s48, {K32}",
           W,
           "\tv_fma_f32 v209, v209, s48, -s49",
           "\tv_fma_f32 v239, v239, s48, -s49",
           "\ts_fmaak_f32 s51, s34, s50, 0xbe602dee",
           "\ts_delay_alu instid0(VALU_DEP_2)",
           "\tv_max_num_f32_e32 v209, v214, v209",
           "\tv_max_num_f32_e32 v210, v213, v239",
           W,
           "\ts_delay_alu instid0(VALU_DEP_2)",
           f"\tv_fma_f32 v252, v209, {K32}, s51",
           f"\tv_fma_f32 v253, v210, {K32}, s51",
           "\tv_sub_f32_e32 v255, v214, v209",
           "\ts_delay_alu instid0(VALU_DEP_2)"]
    out[j + 1:j + 1 + len(exp_blk)] = new
    b = out.index(".LBB0_43:")
    n = 0
    for k in range(j, b):
        m = re.match(r"\tv_fma_f32 (v\d+), (v\d+), v254, -(v25[23])$", out[k])
        if m:
            out[k] = f"\tv_fma_f32 {m.group(1)}, {m.group(2)}, s50, -{m.group(3)}"
            n += 1
    assert n == 32, n
    print("\n".join(l for l in out[a:b] if "v254" in l), file=sys.stderr)
    # v196 (= c in esp321y) is read only by the cmp and the .Lnocvt_fast block (both replaced); elsewhere it is
    # the LDS address temp (written before read) and the row-sum accumulator (written by dot4 after this point).
    bump_sgpr(52)

if "lf" in opts or "lfi" in opts:
    # l in place (needs ss: v196 no longer holds c). l_A lives in v200, l_B in v196 for the whole loop (esp321y
    # relays them through v212/v211, which the PV tail reuses as load temps, hence an add + a dual_mov per tile).
    # The header's LDS address temp v196 becomes v252 (free until the max tree). The lazy-rescale branches scale l
    # (v_mul instead of fmac + mov 0) and the row-sum dot4 chains move to the top of .LBB0_36 (after both rescales),
    # accumulating straight into l. Per tile: -2 v_add, -1 v_dual_mov, -2 v_mov (rescale path only).
    # lfi: the dot4s are interleaved one per PV WMMA pair instead of a block.
    assert "ss" in opts
    a = out.index(TAIL)
    b = out.index(".LBB0_43:")
    hdr = out.index(HDR)
    i = idx("\tv_add_nc_u32_e32 v196, 0x800, v209", hdr, b)
    out[i] = "\tv_add_nc_u32_e32 v252, 0x800, v209"
    n = 0
    for k in range(i + 1, b):
        if out[k].startswith("\tds_load_2addr_b64") and out[k].split(", ")[1].startswith("v196 "):
            out[k] = out[k].replace(", v196 ", ", v252 ")
            n += 1
        if out[k].startswith("\tv_max3_num_f32 v209,"):
            break
    assert n == 4, n
    d0 = idx("\tv_dot4_f32_fp8_fp8 v200, v231, 0x38383838, 0", a, b)
    dots = out[d0:d0 + 11]
    assert all("dot4" in l or l.startswith("\ts_delay_alu") for l in dots), dots
    dots = [l for l in dots if "dot4" in l]
    assert len(dots) == 8, dots
    del out[d0:d0 + 11]
    newdots = []
    for l in dots:
        if l.endswith(", 0"):
            l = l[:-1] + l.split()[1].rstrip(",")
        newdots.append(l)
    b = out.index(".LBB0_43:")
    for k, v in {"\tv_fmac_f32_e32 v200, v212, v255": "\tv_mul_f32_e32 v200, v200, v255",
                 "\tv_fmac_f32_e32 v196, v211, v213": "\tv_mul_f32_e32 v196, v196, v213"}.items():
        i = idx(k, a, b)
        out[i] = v
        assert out[i + 1] in ("\tv_mov_b32_e32 v212, 0", "\tv_mov_b32_e32 v211, 0"), out[i + 1]
        del out[i + 1]
        b -= 1
    for k in ["\tv_add_f32_e32 v200, v200, v212", "\tv_add_f32_e32 v196, v196, v211",
              "\tv_dual_mov_b32 v211, v196 :: v_dual_mov_b32 v212, v200"]:
        del out[idx(k, a, b)]
        b -= 1
    i = idx("\ts_or_b32 exec_lo, exec_lo, s9", a, b)
    assert out[i - 2:i] == ["\ts_setprio 3", W], out[i - 2:i]
    if "lfi" in opts:
        j = i + 1
        k = 0
        while k < 8:
            if out[j].startswith("\tv_wmma") and out[j + 1].startswith("\tv_wmma"):
                out[j + 2:j + 2] = [newdots[k]]
                k += 1
                j += 3
            else:
                j += 1
    else:
        out[i + 1:i + 1] = newdots[:2] + ["\ts_delay_alu instid0(VALU_DEP_2)"] + newdots[2:4] + \
            ["\ts_delay_alu instid0(VALU_DEP_2)"] + newdots[4:6] + ["\ts_delay_alu instid0(VALU_DEP_2)"] + newdots[6:]
    # l initial value: esp321y zeroes v211/v212 in the prologue; zero v196/v200 right before entering the loop
    assert not any(re.search(r"\bv(196|200)\b", l) for l in out[:a] if l.startswith("\t")), "v196/v200 used pre-loop"
    i = idx("\ts_branch .LBB0_37")
    out[i:i] = ["\tv_mov_b32_e32 v196, 0", "\tv_mov_b32_e32 v200, 0"]
    b = out.index(".LBB0_43:")
    for r in ("v211", "v212"):
        bad = [l for l in out[a:b] if re.search(r"\b%s\b" % r, l) and "ds_load_2addr_b64 v[211:214]" not in l]
        assert not bad, bad

if "cx" in opts:
    # lazy-rescale test straight on (m_old != m_new): the two max-difference v_subs move into the (rare) rescale
    # branches. Same predicate (d != 0 <=> m_old != m_new; NaN/inf behave the same with the unordered neq).
    assert "ss" in opts
    a = out.index(TAIL)
    b = out.index(".LBB0_43:")
    del out[idx("\tv_sub_f32_e32 v255, v214, v209", a, b)]
    i = idx("\tv_sub_f32_e32 v213, v213, v210", a, b)
    assert out[i + 1:i + 5] == ["\ts_delay_alu instid0(VALU_DEP_1)", "\tv_cmpx_neq_f32_e32 0, v255",
                                "\ts_cbranch_execz .LBB0_41", "\tv_exp_f32_e32 v255, v255"], out[i + 1:i + 5]
    out[i:i + 5] = ["\tv_cmpx_neq_f32_e32 v214, v209", "\ts_cbranch_execz .LBB0_41", "\tv_sub_f32_e32 v255, v214, v209",
                    "\ts_delay_alu instid0(VALU_DEP_1)", "\tv_exp_f32_e32 v255, v255"]
    i = idx("\tv_cmpx_neq_f32_e32 0, v213", a, b)
    assert out[i + 1:i + 3] == ["\ts_cbranch_execz .LBB0_36", "\tv_exp_f32_e32 v213, v213"], out[i + 1:i + 3]
    out[i:i + 3] = ["\tv_cmpx_neq_f32_e32 v213, v210", "\ts_cbranch_execz .LBB0_36", "\tv_sub_f32_e32 v213, v213, v210",
                    "\ts_delay_alu instid0(VALU_DEP_1)", "\tv_exp_f32_e32 v213, v213"]

if "mb" in opts or "mx2" in opts:
    # mb: one uniform branch for both lazy rescales (needs cx): masks via v_cmp to SGPRs, s_or -> scc, then
    # exec = maskA / maskB inside. mx2: m kept in v214/v213 (inside the branch m := m_new), tail temps
    # v[211:214] -> v[252:255], the tail's v213/v214 dual_mov dropped (v209/v210 still = m at loop exit).
    assert "cx" in opts
    a = out.index(TAIL)
    b = out.index(".LBB0_43:")
    i0 = idx("\tv_cmpx_neq_f32_e32 v214, v209", a, b)
    i1 = b - 1
    assert out[i1] == "\ts_branch .LBB0_36", out[i1]
    blk = out[i0:i1 + 1]
    mA = [l for l in blk if l.startswith("\tv_dual_mul_f32") and ", v255, " in l]
    mB = [l for l in blk if l.startswith("\tv_dual_mul_f32") and ", v213, " in l]
    assert len(mA) == 32 and len(mB) == 32 and sum(l.startswith(".LBB0_41:") for l in blk) == 1 and len(blk) == 84, (len(mA), len(mB), len(blk))
    if "mx2" in opts:
        mB = [l.replace(", v213, ", ", v255, ") for l in mB]
        brA = ["\tv_sub_f32_e32 v255, v214, v209", "\tv_mov_b32_e32 v214, v209", "\ts_delay_alu instid0(VALU_DEP_2)",
               "\tv_exp_f32_e32 v255, v255", "\ts_delay_alu instid0(TRANS32_DEP_1)", "\tv_mul_f32_e32 v200, v200, v255"]
        brB = ["\tv_sub_f32_e32 v255, v213, v210", "\tv_mov_b32_e32 v213, v210", "\ts_delay_alu instid0(VALU_DEP_2)",
               "\tv_exp_f32_e32 v255, v255", "\ts_delay_alu instid0(TRANS32_DEP_1)", "\tv_mul_f32_e32 v196, v196, v255"]
    else:
        brA = ["\tv_sub_f32_e32 v255, v214, v209", "\ts_delay_alu instid0(VALU_DEP_1)",
               "\tv_exp_f32_e32 v255, v255", "\ts_delay_alu instid0(TRANS32_DEP_1)", "\tv_mul_f32_e32 v200, v200, v255"]
        brB = ["\tv_sub_f32_e32 v213, v213, v210", "\ts_delay_alu instid0(VALU_DEP_1)",
               "\tv_exp_f32_e32 v213, v213", "\ts_delay_alu instid0(TRANS32_DEP_1)", "\tv_mul_f32_e32 v196, v196, v213"]
    if "mb" in opts:
        new = ["\tv_cmp_neq_f32_e64 s52, v214, v209", "\tv_cmp_neq_f32_e64 s53, v213, v210",
               "\ts_wait_alu depctr_va_sdst(0)", "\ts_or_b32 s54, s52, s53", "\ts_cbranch_scc0 .LBB0_36",
               "\ts_mov_b32 exec_lo, s52"] + brA + mA + ["\ts_mov_b32 exec_lo, s53"] + brB + mB + \
              ["\ts_branch .LBB0_36"]
        bump_sgpr(56)
        assert not any(re.search(r"\bs(5[2-5])\b|s\[5[2-5]", l) for l in out), "s52-s55 in use"
    else:
        new = ["\tv_cmpx_neq_f32_e32 v214, v209", "\ts_cbranch_execz .LBB0_41"] + brA + mA + [
               ".LBB0_41:", "\ts_or_b32 exec_lo, exec_lo, s9", "\ts_mov_b32 s9, exec_lo",
               "\tv_cmpx_neq_f32_e32 v213, v210", "\ts_cbranch_execz .LBB0_36"] + brB + mB + ["\ts_branch .LBB0_36"]
    out[i0:i1 + 1] = new
    if "mx2" in opts:
        hdr = out.index(HDR)
        del out[idx("\tv_dual_mov_b32 v213, v210 :: v_dual_mov_b32 v214, v209", a, hdr)]
        hdr -= 1
        n = 0
        for k in range(a, hdr):
            l = out[k]
            l2 = l.replace("v[211:214]", "v[252:255]").replace("v[211:212]", "v[252:253]").replace("v[213:214]",
                                                                                                   "v[254:255]")
            if l2 != l:
                out[k] = l2
                n += 1
        assert n == 5, n
        for k in range(a, hdr):
            if not out[k].strip().startswith(";"):
                assert not re.search(r"\bv(25[2-5]|21[34])\b", out[k]), out[k]

if "mx" in opts:
    # m in place (needs ss + lf, replaces cx): m_A / m_B live in v214 / v213 for the whole loop. The rescale
    # branches run right after the scaled tile max t is known, before the P pack: exec = (m_old < t), and inside
    # d = m_old - t, m = t, alpha = 2^d, l *= alpha, O *= alpha. Off-branch m is unchanged = max(m_old, t), the same
    # value as esp321y's v_max (same predicate too: m_old != max(m_old, t) <=> m_old < t, NaN-safe). Drops the two
    # v_max and the tail's v213/v214 dual_mov; the PV tail's V temps v[211:214] become v[252:255] (dead there).
    # Exit copies m to v209/v210 for the epilogue.
    assert "ss" in opts and "lf" in opts or "lfi" in opts
    assert "cx" not in opts
    a = out.index(TAIL)
    b = out.index(".LBB0_43:")
    i0 = idx("\tv_fma_f32 v209, v209, s48, -s49", a, b)
    i1 = b - 1
    assert out[i1] == "\ts_branch .LBB0_36", out[i1]
    blk = out[i0:i1 + 1]
    efma = [l for l in blk if re.match(r"\tv_fma_f32 v\d+, v\d+, s50, -v25[23]$", l)]
    cvt = [l for l in blk if l.startswith("\tv_cvt_pk_norm_u16_f32")]
    perm = [l for l in blk if l.startswith("\tv_perm_b32")]
    mA = [l for l in blk if l.startswith("\tv_dual_mul_f32") and ", v255, " in l]
    mB = [l.replace(", v213, ", ", v255, ") for l in blk if l.startswith("\tv_dual_mul_f32") and ", v213, " in l]
    assert (len(efma), len(cvt), len(perm), len(mA), len(mB)) == (32, 16, 8, 32, 32), \
        (len(efma), len(cvt), len(perm), len(mA), len(mB))
    assert blk[1] == "\tv_fma_f32 v239, v239, s48, -s49" and blk[2].startswith("\ts_fmaak_f32 s51"), blk[:3]
    new = blk[:3] + [
        "\ts_delay_alu instid0(VALU_DEP_2)",
        "\tv_cmpx_lt_f32_e32 v214, v209",
        "\ts_cbranch_execz .LmxA",
        "\tv_sub_f32_e32 v255, v214, v209",
        "\tv_mov_b32_e32 v214, v209",
        "\ts_delay_alu instid0(VALU_DEP_2)",
        "\tv_exp_f32_e32 v255, v255",
        "\ts_delay_alu instid0(TRANS32_DEP_1)",
        "\tv_mul_f32_e32 v200, v200, v255"] + mA + [
        ".LmxA:",
        "\ts_or_b32 exec_lo, exec_lo, s9",
        "\tv_cmpx_lt_f32_e32 v213, v239",
        "\ts_cbranch_execz .LmxB",
        "\tv_sub_f32_e32 v255, v213, v239",
        "\tv_mov_b32_e32 v213, v239",
        "\ts_delay_alu instid0(VALU_DEP_2)",
        "\tv_exp_f32_e32 v255, v255",
        "\ts_delay_alu instid0(TRANS32_DEP_1)",
        "\tv_mul_f32_e32 v196, v196, v255"] + mB + [
        ".LmxB:",
        "\ts_or_b32 exec_lo, exec_lo, s9",
        W,
        "\tv_fma_f32 v252, v214, 0x3d000080, s51",
        "\tv_fma_f32 v253, v213, 0x3d000080, s51",
        "\ts_delay_alu instid0(VALU_DEP_2)"] + efma + cvt + ["\ts_setprio 3"] + perm + ["\ts_branch .LBB0_36"]
    out[i0:i1 + 1] = new
    b = out.index(".LBB0_43:")
    hdr = out.index(HDR)
    tail = out[a:hdr]
    dm = "\tv_dual_mov_b32 v213, v210 :: v_dual_mov_b32 v214, v209"
    del out[idx(dm, a, hdr)]
    hdr -= 1
    n = 0
    for k in range(a, hdr):
        l = out[k]
        l2 = l.replace("v[211:214]", "v[252:255]").replace("v[211:212]", "v[252:253]").replace("v[213:214]", "v[254:255]")
        if l2 != l:
            out[k] = l2
            n += 1
    assert n == 5, n
    for k in range(a, hdr):
        assert not re.search(r"v(25[2-5])\b|v\[25", out[k]) or "v[252:25" in out[k] or "v[254:255]" in out[k], out[k]
    i = idx("\ts_cbranch_scc1 .LBB0_44", a, hdr)
    out[i] = "\ts_cbranch_scc1 .Lmxexit"
    i = idx(".LBB0_43:")
    assert out[i - 1] == "\ts_branch .LBB0_36", out[i - 1]
    out[i:i] = [".Lmxexit:", "\tv_dual_mov_b32 v209, v214 :: v_dual_mov_b32 v210, v213", "\ts_branch .LBB0_44"]

if "mg" in opts:
    # magic as an inline constant (needs ss). The QK WMMA accumulator init v[244:251] = 0x4b400000 becomes the inline
    # constant 1/(2pi) = 0x3e22f983: for |i| < 0x22f983 = 2292099 (|q.k| <= 128*128*128 = 2097152 for int8) the
    # result bits 0x3e22f983 + i are the float 2^-26 * (M' + i) with M' = 2^23 + 0x22f983 = 10680707, i.e. the same
    # magic-bias trick with M' instead of 1.5*2^23, scaled by 2^-26. Compensation: c = k_scale*q_scale is scaled by
    # 2^26 (s47 := q_scale*2^26, exact), s34 := M'*2^-26 = 1/(2pi), the exact-path test compares c*2^26 against
    # 2^26/384 and the exact path subtracts the inline 1/(2pi). All other arithmetic is unchanged and exact
    # power-of-two rescalings of esp321y's with magic M'. Frees v244-v251 (8 VGPRs) and the 4 prologue movs.
    n = 0
    for k, l in enumerate(out):
        if "v[244:251]" in l:
            assert l.startswith("\tv_wmma_i32_16x16x16_iu8"), l
            out[k] = l.replace("v[244:251]", "0.15915494")
            n += 1
    assert n == 4, n
    mv = [l for l in out if re.match(r"\tv_dual_mov_b32 v2(4[4-9]|5[01]), 0x4b400000 :: v_dual_mov_b32 v2(4[4-9]|5[01]), 0x4b400000$", l)]
    assert len(mv) == 4, mv
    out = [l for l in out if l not in mv]
    n = 0
    for k, l in enumerate(out):
        m = re.match(r"\tv_sub_f32_e32 (v\d+), (v\d+), v244$", l)
        if m:
            assert m.group(1) == m.group(2), l
            out[k] = f"\tv_subrev_f32_e32 {m.group(1)}, 0.15915494, {m.group(1)}"
            n += 1
    assert n == 34, n
    i = idx("\ts_mov_b32 s34, 0x4b400000")
    out[i] = "\ts_mov_b32 s34, 0x3e22f983"
    assert sum(1 for l in out if "0x4b400000" in l) == 0
    assert sum(1 for l in out if re.search(r"\bv(24[4-9]|25[01])\b|v\[24[4-9]|v\[25[01]", l.split(";")[0])) == 0
    i = idx("\ts_cmp_gt_f32 s48, 0x3b2aaaab")
    out[i] = "\ts_cmp_gt_f32 s48, 0x482aaaab"
    i = idx("\tv_readfirstlane_b32 s47, v198")
    out[i + 1:i + 1] = ["\ts_wait_alu depctr_va_sdst(0)", "\ts_mul_f32 s47, s47, 0x4c800000"]

def _vregs(s):
    r = set()
    for m in re.finditer(r"\bv\[(\d+):(\d+)\]|\bv(\d+)\b", s):
        r |= set(range(int(m.group(1)), int(m.group(2)) + 1)) if m.group(1) else {int(m.group(3))}
    return r


def _sregs(s):
    r = set()
    for m in re.finditer(r"\bs\[(\d+):(\d+)\]|\bs(\d+)\b", s):
        r |= set(range(int(m.group(1)), int(m.group(2)) + 1)) if m.group(1) else {int(m.group(3))}
    return r


def _valu_ds(l):
    # (dst vgprs, src vgprs) of a VALU line (VOPD aware; accumulators count as sources)
    def one(op, args):
        parts = args.split(",")
        if op.startswith("v_cmpx") or (op.startswith("v_cmp") and "_e64" not in op):
            return set(), _vregs(args)
        if op.startswith("v_cmp"):
            return set(), _vregs(",".join(parts[1:]))
        d, s = _vregs(parts[0]), _vregs(",".join(parts[1:]))
        if op.startswith(("v_fmac", "v_wmma", "v_dot4", "v_dot2", "v_dual_fmac")) or ".h" in parts[0] or ".l" in parts[0]:
            s |= d
        return d, s
    if "::" in l:
        d1, s1 = one(*l.split("::")[0].split(None, 1))
        d2, s2 = one(*l.split("::")[1].split(None, 1))
        return d1 | d2, s1 | s2
    op, args = (l.split(None, 1) + [""])[:2]
    return one(op, args)


TRANS = ("v_exp_", "v_log_", "v_rcp_", "v_rsq_", "v_sqrt_", "v_sin_", "v_cos_")
SFLT = re.compile(r"^s_(add|sub|mul|fmaak|fmamk|fmac|max|min|cvt\w*)_f(16|32)\b|^s_(fmaak|fmamk|fmac)_f32")


RH_SLACK = 1 if "rh1" in sys.argv else 0  # rh1: a wait also covers the VALU issued right after the producer


def rehint(lines):
    # recompute the s_delay_alu hints of a straight-line instruction list (labels/branches reset the history)
    body = [l for l in lines if not l.startswith("\ts_delay_alu")]
    need = {}  # index in body -> hint string
    hv, ht, hs = [], [], []  # VALU (dsts), TRANS (dsts), SALU-float (dst sgprs) histories
    for i, raw in enumerate(body):
        l = raw.split(";")[0].strip()
        if not l:
            continue
        if l.endswith(":") or l.startswith(("s_branch", "s_cbranch", "s_setpc", "s_endpgm")):
            hv, ht, hs = [], [], []
            continue
        if l.startswith("v_"):
            d, s = _valu_ds(l)
            # a wait satisfies the waited-on producer and (in-order VALU completion) every older VALU
            h = None
            for k, dd in enumerate(reversed(ht[-3:])):
                if dd & s:
                    h = f"TRANS32_DEP_{k + 1}"
                    ht[len(ht) - 1 - k] = set()
                    break
            if h is None:
                for k, (dd, tr) in enumerate(reversed(hv[-4:])):
                    if dd & s and not tr:
                        h = f"VALU_DEP_{k + 1}"
                        for q in range(min(len(hv), len(hv) - 1 - k + 1 + RH_SLACK)):
                            if not hv[q][1]:
                                hv[q] = (set(), False)
                        break
            if h:
                need[i] = h
            tr = l.startswith(TRANS)
            hv.append((d, tr))
            if tr:
                ht.append(d)
        elif l.startswith("s_"):
            if l.startswith(("s_wait", "s_nop", "s_setprio", "s_barrier")):
                continue
            op, args = (l.split(None, 1) + [""])[:2]
            parts = args.split(",")
            src = _sregs(",".join(parts[1:])) if len(parts) > 1 else set()
            if op.startswith(("s_cmp", "s_bitcmp")):
                src = _sregs(args)
            for k, dd in enumerate(reversed(hs[-3:])):
                if dd & src:
                    need[i] = f"SALU_CYCLE_{3 - k}"
                    break
            dst = set() if op.startswith(("s_cmp", "s_bitcmp")) else _sregs(parts[0])
            hs.append(dst if SFLT.match(l) else set())
    out2 = []
    idxs = sorted(need)
    paired = {}
    j = 0
    while j < len(idxs):
        a = idxs[j]
        if j + 1 < len(idxs):
            b = idxs[j + 1]
            mid = [body[k].split(";")[0].strip() for k in range(a + 1, b)]
            gap = sum(1 for m in mid if m)
            if gap <= 4 and not any(m.endswith(":") or m.startswith(".") for m in mid):
                skip = "NEXT" if gap == 0 else f"SKIP_{gap}"
                paired[a] = f"\ts_delay_alu instid0({need[a]}) | instskip({skip}) | instid1({need[b]})"
                j += 2
                continue
        paired[a] = f"\ts_delay_alu instid0({need[a]})"
        j += 1
    for i, l in enumerate(body):
        if i in paired:
            out2.append(paired[i])
        out2.append(l)
    return out2


if "rh" in opts or "rh1" in opts:
    # recompute the main-loop s_delay_alu hints for the edited code (the post-passes leave stale / missing hints)
    a = out.index(TAIL)
    b = out.index(".LBB0_43:")
    old = sum(1 for l in out[a:b] if l.startswith("\ts_delay_alu"))
    new = rehint(out[a:b])
    out = out[:a] + new + out[b:]
    print("rh: hints", old, "->", sum(1 for l in new if l.startswith("\ts_delay_alu")))

if "nd" in opts:
    # drop the s_delay_alu scheduling hints inside the main loop (hardware interlocks; hint only)
    a = out.index(TAIL)
    b = out.index(".LBB0_43:")
    n0 = len(out)
    out = out[:a] + [l for l in out[a:b] if not l.startswith("\ts_delay_alu")] + out[b:]
    print("nd: removed", n0 - len(out))

if "occd" in opts:
    # DIAGNOSTIC (wrong results, timing only): rescale branches never taken, kernel ends right after the loop.
    # r240: additionally alias v239..v255 -> v99..v115 (O accumulators: data only) everywhere and declare 240 VGPRs
    # -> 3 WG/WGP if LDS allows. Compare occd vs occd r240 (same lds=) to measure the occupancy effect.
    a = out.index(TAIL)
    b = out.index(".LBB0_43:")
    for k in range(a, b):
        if out[k] in ("\ts_cbranch_execz .LBB0_41", "\ts_cbranch_execz .LBB0_36", "\ts_cbranch_scc0 .LBB0_36"):
            out[k] = out[k].replace("s_cbranch_execz", "s_branch").replace("s_cbranch_scc0", "s_branch")
    i = next(k for k, l in enumerate(out) if l.startswith(".LBB0_44:"))
    out[i + 1:i + 1] = ["\ts_endpgm"]
    if "r240" in opts:
        def rr(m):
            if m.group(1) is not None:
                x, y = int(m.group(1)), int(m.group(2))
                if x >= 239:
                    return "v[%d:%d]" % (x - 140, y - 140)
                assert y < 239, m.group(0)
                return m.group(0)
            x = int(m.group(3))
            return "v%d" % (x - 140) if x >= 239 else m.group(0)
        for k, l in enumerate(out):
            if l.startswith("\t") and not l.strip().startswith(";") and not l.strip().startswith("."):
                out[k] = re.sub(r"\bv\[(\d+):(\d+)\]|\bv(\d+)\b", rr, l)
        out = [re.sub(r"(\.amdhsa_next_free_vgpr )\d+", r"\g<1>240", l) for l in out]
        out = [re.sub(r"(\.vgpr_count:\s+)\d+", r"\g<1>240", l) for l in out]

open(out_path, "w").write("\n".join(out))
print("mk_c2:", " ".join(opts), ("-- " + " ".join(extra)) if extra else "")

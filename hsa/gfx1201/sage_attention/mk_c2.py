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

def _remap_line(l, mp):
    # rename vgprs (singles and tuples) in one instruction line by the map mp (tuples must stay contiguous)
    def rr(m):
        if m.group(1) is not None:
            x, y = int(m.group(1)), int(m.group(2))
            nx = [mp.get(r, r) for r in range(x, y + 1)]
            assert nx == list(range(nx[0], nx[0] + len(nx))), (l, nx)
            return "v[%d:%d]" % (nx[0], nx[-1])
        x = int(m.group(3))
        return "v%d" % mp.get(x, x)
    return re.sub(r"\bv\[(\d+):(\d+)\]|\bv(\d+)\b", rr, l)


def pass_v240():
    global out
    # 240 VGPRs -> 3 WG/WGP (needs mg). Registers that are live through the main loop but not referenced in it
    # (SLOT) are parked in LDS for the loop (ds_store at loop entry / on the no-loop path, ds_load at .LBB0_44) and
    # the loop's temporaries above v239 are renamed into them. v203 (loop-invariant LDS address, also live-out) is
    # copied into v211 (whose own value is parked) so that v[201:204] is a free 4-run. Exact: pure register renaming.
    assert "mg" in opts
    a = out.index(TAIL)
    b = out.index(".LBB0_43:")
    SLOT = [153, 154, 155, 156, 191, 201, 202, 203, 204, 211]
    MP = {239: 153, 240: 154, 241: 155, 242: 156, 243: 191, 252: 201, 253: 202, 254: 203, 255: 204, 203: 211}
    for k in range(a, b):
        t = out[k]
        if t.startswith("\t") and not t.strip().startswith((";", ".")):
            r = set(_vregs(t.split(";")[0]))
            assert not (r & (set(SLOT) - {203})), ("slot reg used in loop", t)
            out[k] = _remap_line(t, MP)
    for k in range(a, b):
        r = _vregs(out[k].split(";")[0]) if out[k].startswith("\t") else set()
        assert not r or max(r) < 240, out[k]
    LBASE = 18944
    st = ["\tv_lshlrev_b32_e32 v215, 2, v0"] + [f"\tds_store_b32 v215, v{r} offset:{LBASE + 1024 * i}"
                                                for i, r in enumerate(SLOT)]
    i = idx("\ts_branch .LBB0_37")
    out[i:i] = st + ["\ts_wait_dscnt 0x0", "\tv_mov_b32_e32 v211, v203"]
    i = idx(".LBB0_43:")
    out[i + 1:i + 1] = [l.replace("v215", "v238") for l in st]
    i = idx(".LBB0_44:")
    exp = ["\tds_bpermute_b32 v240, v207, v200", "\tds_bpermute_b32 v241, v207, v196", "\ts_wait_dscnt 0x0",
           "\tv_add_f32_e32 v200, v200, v240", "\tv_add_f32_e32 v196, v196, v241"]
    assert out[i + 1:i + 6] == exp, out[i + 1:i + 6]
    out[i + 1:i + 6] = [l.replace("v240", "v238").replace("v241", "v239") for l in exp] + \
        ["\tv_lshlrev_b32_e32 v238, 2, v0"] + \
        [f"\tds_load_b32 v{r}, v238 offset:{LBASE + 1024 * i}" for i, r in enumerate(SLOT)] + ["\ts_wait_dscnt 0x0"]
    st_ = 0
    for k, l in enumerate(out):
        if l.startswith("\t") and not l.strip().startswith((";", ".")):
            r = _vregs(l.split(";")[0])
            assert not r or max(r) < 240, (k, l)
    out = [re.sub(r"(\.amdhsa_next_free_vgpr )\d+", r"\g<1>240", l) for l in out]
    out = [re.sub(r"(\.vgpr_count:\s+)\d+", r"\g<1>240", l) for l in out]
    need = LBASE + 1024 * len(SLOT)

    def fixlds(l):
        m = re.search(r"(\.amdhsa_group_segment_fixed_size |\.group_segment_fixed_size:\s+)(\d+)", l)
        if m and int(m.group(2)) < need:
            l = l.replace(m.group(0), m.group(1) + str(need))
        return l
    out = [fixlds(l) for l in out]


def pass_u2():
    # unroll the main loop x2 with static LDS buffer parity (needs v240). The per-tile LDS address VALU (3 v_add3 in
    # the tail, 2 adds in the header, 2 in the store block) and the parity SALU become loop-invariant base registers
    # (14 bases, 12 new regs: 6 dead-in-loop + 6 parked in LDS like v240). Exact: same addresses, same order.
    global out
    assert "v240" in opts
    a = out.index(TAIL)
    h = out.index(HDR)
    b = out.index(".LBB0_43:")
    PARK = [194, 195, 197, 198, 199, 207]
    POOL = [206, 208, 212, 239, 192, 211] + PARK
    for k in range(a, b):
        t = out[k]
        if t.startswith("\t") and not t.strip().startswith((";", ".")):
            r = _vregs(t.split(";")[0])
            assert not (r & set([206, 208, 212, 239] + PARK)), ("pool reg used in loop", t)
    # bases: T[p][C] tail V-load bases, H/G header K-load bases, VS/KS store bases (indexed by store parity q)
    T = {0: {0x2000: 206, 0x2800: 208, 0x3000: 212}, 1: {0x2000: 239, 0x2800: 194, 0x3000: 192}}
    H = {0: 193, 1: 195}
    G = {0: 197, 1: 198}
    VS = {0: 211, 1: 199}
    KS = {0: 189, 1: 207}
    init = ["\tv_add_nc_u32_e32 v206, 0x2000, v192", "\tv_add_nc_u32_e32 v208, 0x2800, v192",
            "\tv_add_nc_u32_e32 v212, 0x3000, v192", "\tv_add_nc_u32_e32 v239, 0x3400, v192",
            "\tv_add_nc_u32_e32 v194, 0x3c00, v192", "\tv_add_nc_u32_e32 v192, 0x4400, v192",
            "\tv_add_nc_u32_e32 v195, 0x1100, v193", "\tv_add_nc_u32_e32 v197, 0x800, v193",
            "\tv_add_nc_u32_e32 v198, 0x1900, v193", "\tv_add_nc_u32_e32 v199, 0x3600, v211",
            "\tv_add_nc_u32_e32 v211, 0x2200, v211", "\tv_add_nc_u32_e32 v207, 0x1100, v189"]

    def tail(p):
        res, cur, nl, nd = [], {}, 0, 0
        for l in out[a:h]:
            m = re.match(r"\tv_add3_u32 (v\d+), s1, v192, (0x[0-9a-f]+)$", l)
            if m:
                cur[m.group(1)] = "v%d" % T[p][int(m.group(2), 16)]
                nd += 1
                continue
            if l == "\ts_mul_i32 s1, s8, 0x1400":
                nd += 1
                continue
            m = re.match(r"(\tds_load\w+ v\[\d+:\d+\], )(v\d+)( .*)$", l)
            if m and m.group(2) in cur:
                l = m.group(1) + cur[m.group(2)] + m.group(3)
                nl += 1
            res.append(l)
        assert nd == 4 and nl == 8, (nd, nl)
        return res

    def header(p):
        q = 1 - p
        res, nd, nl = [], 0, 0
        drop = {"\ts_and_b32 s8, s6, 1", "\ts_xor_b32 s10, s8, 1", "\ts_mul_i32 s11, s10, 0x1400",
                "\tv_add3_u32 v134, v211, s11, 0x2200", "\ts_mul_i32 s11, s10, 0x1100",
                "\tv_add_nc_u32_e32 v131, s11, v189", "\ts_mul_i32 s1, s8, 0x1100",
                "\tv_add_nc_u32_e32 v209, s1, v193", "\tv_add_nc_u32_e32 v201, 0x800, v209"}
        for l in out[h:b]:
            if l in drop:
                nd += 1
                continue
            if l.startswith("\tds_store_2addr_b64 v131,"):
                l = l.replace("v131,", "v%d," % KS[q], 1)
                nl += 1
            elif l.startswith("\tds_store_2addr_b64 v134,"):
                l = l.replace("v134,", "v%d," % VS[q], 1)
                nl += 1
            else:
                m = re.match(r"(\tds_load\w+ v\[\d+:\d+\], )(v209|v201)( .*)?$", l)
                if m:
                    l = m.group(1) + "v%d" % (H[p] if m.group(2) == "v209" else G[p]) + (m.group(3) or "")
                    nl += 1
            res.append(l)
        assert nd == len(drop) and nl == 10, (nd, nl)
        return res

    def ren(lines, sfx):
        mp = {".LBB0_36": ".LU36" + sfx, ".LBB0_37": ".LU37" + sfx, ".LBB0_39": ".LU39" + sfx,
              ".Lnocvt_fast": ".Lnocvt" + sfx}
        r = []
        for l in lines:
            for k, v in mp.items():
                l = re.sub(re.escape(k) + r"\b", v, l)
            r.append(l)
        return r
    T1, H1, T0, H0 = tail(1), header(1), ren(tail(0), "_0"), ren(header(0), "_0")
    assert H0[-1] == "\ts_branch .LU36_0", H0[-1]
    H0 = H0[:-1]  # falls through into T0
    body = T1 + H0 + T0 + H1
    for l in body:
        assert not re.search(r"\bs8\b|\bs\[8:", l.split(";")[0]) or l.startswith("\ts_delay"), l
    out = out[:a] + body + out[b:]
    assert sum(1 for l in out if ".LBB0_37" in l.split(";")[0] and not l.startswith(".LBB0_37:")) == 1
    LBASE = 18944 + 10 * 1024
    i = idx("\ts_branch .LBB0_37")
    out[i] = "\ts_branch .LU37_0"
    st = [f"\tds_store_b32 v215, v{r} offset:{LBASE + 1024 * j}" for j, r in enumerate(PARK)]
    assert out[i - 1] == "\tv_mov_b32_e32 v211, v203", out[i - 1]
    out[i:i] = st + ["\ts_wait_dscnt 0x0"] + init
    i = idx(".LBB0_43:")
    assert out[i + 1] == "\tv_lshlrev_b32_e32 v238, 2, v0"
    out[i + 2:i + 2] = [l.replace("v215", "v238") for l in st]
    i = idx(".LBB0_44:")
    out[i + 1:i + 1] = ["\tv_lshlrev_b32_e32 v238, 2, v0"] + \
        [f"\tds_load_b32 v{r}, v238 offset:{LBASE + 1024 * j}" for j, r in enumerate(PARK)] + ["\ts_wait_dscnt 0x0"]
    need = LBASE + 1024 * len(PARK)

    def fixlds(l):
        m = re.search(r"(\.amdhsa_group_segment_fixed_size |\.group_segment_fixed_size:\s+)(\d+)", l)
        if m and int(m.group(2)) < need:
            l = l.replace(m.group(0), m.group(1) + str(need))
        return l
    out = [fixlds(l) for l in out]
    for l in out:
        if l.startswith("\t") and not l.strip().startswith((";", ".")):
            r = _vregs(l.split(";")[0])
            assert not r or max(r) < 240, l


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

if "v240" in opts:
    pass_v240()

for o in opts:
    if o.startswith("spr="):
        # exact: main-loop s_setprio values in program order (tail/PV, QK, softmax, perm/dot), e.g. spr=3,1,2,3 (= esp321z)
        vals = o[4:].split(",")
        a = out.index(TAIL)
        b = out.index(".LBB0_43:")
        ii = [i for i in range(a, b) if out[i].startswith("\ts_setprio")]
        assert len(ii) == len(vals), ii
        for i, v in zip(ii, vals):
            out[i] = f"\ts_setprio {v}"

if "u2" in opts:
    pass_u2()

if "sp" in opts:
    # SALU pointer increments (exact): the per-tile scale address s[16:17] + 4*(s3+s6) and the next K/V global
    # addresses ((s1*s[26:27])<<7 + s[36:37], s[38:39] + s1 with s1 = s7+64 < s12 ? s7+64 : 0) become running
    # pointers s[56:57] (scale), s[58:59] (K), s[60:61] (V) with stride s[62:63] = s[26:27]<<12; the wrap to row 0
    # is an s_cselect_b64 on the unchanged s_cmp. The store block runs on every tile or never for a wave (s0 is
    # loop invariant), so incrementing there is consistent.
    a = out.index(TAIL)
    b = out.index(".LBB0_43:")
    for k in range(a, b):
        l = out[k].split(";")[0]
        assert not re.search(r"\bs(5[6-9]|6[0-3])\b|\bs\[(5[6-9]|6[0-3]):|^\t\S+ (s0|s\[0:1\]),", l), out[k]
    SC = ["\ts_add_co_i32 s10, s3, s6", "\ts_ashr_i32 s11, s10, 31", "\ts_lshl_b64 s[10:11], s[10:11], 2",
          "\ts_add_nc_u64 s[10:11], s[16:17], s[10:11]", "\ts_load_b32 s10, s[10:11], 0x0"]
    KV = ["\ts_cselect_b32 s1, s1, 0", "\ts_mul_hi_u32 s11, s1, s26", "\ts_mul_i32 s10, s1, s27",
          "\ts_add_co_i32 s11, s11, s10", "\ts_mul_i32 s10, s1, s26", "\ts_lshl_b64 s[10:11], s[10:11], 7",
          "\ts_add_nc_u64 s[10:11], s[10:11], s[36:37]", "\ts_add_co_u32 s42, s38, s1", "\ts_add_co_ci_u32 s43, s39, 0"]
    res, k, nsc, nkv = [], a, 0, 0
    while k < b:
        l = out[k]
        if l == SC[0]:
            seq = [x for x in out[k:k + 12] if x.startswith("\ts_") and not x.startswith(("\ts_wait_alu", "\ts_mov_b32 s9"))]
            assert seq[:5] == SC, seq
            j, got = k, 0
            while got < 5:
                x = out[j]
                if x in SC:
                    got += 1
                    if x == SC[4]:
                        res += ["\ts_load_b32 s10, s[56:57], 0x0", "\ts_add_nc_u64 s[56:57], s[56:57], 4"]
                else:
                    res.append(x)
                j += 1
            k = j
            nsc += 1
            continue
        if l == KV[0]:
            j, got = k, 0
            while got < len(KV):
                x = out[j]
                if x in KV:
                    got += 1
                    if x == KV[0]:
                        res += ["\ts_cselect_b64 s[10:11], s[58:59], s[36:37]",
                                "\ts_cselect_b64 s[42:43], s[60:61], s[38:39]",
                                "\ts_add_nc_u64 s[58:59], s[58:59], s[62:63]",
                                "\ts_add_nc_u64 s[60:61], s[60:61], 32"]
                else:
                    assert x.startswith("\ts_wait_alu") or x.startswith("\ts_delay"), x
                    res.append(x)
                j += 1
            k = j
            nkv += 1
            continue
        res.append(l)
        k += 1
    nloop = 2 if "u2" in opts else 1
    assert nsc == nloop and nkv == nloop, (nsc, nkv)
    out = out[:a] + res + out[b:]
    i = next(k for k, l in enumerate(out) if l in ("\ts_branch .LU37_0", "\ts_branch .LBB0_37"))
    out[i:i] = ["\ts_ashr_i32 s57, s3, 31", "\ts_mov_b32 s56, s3", W, "\ts_lshl_b64 s[56:57], s[56:57], 2", W,
                "\ts_add_nc_u64 s[56:57], s[16:17], s[56:57]", "\ts_lshl_b64 s[62:63], s[26:27], 13", W,
                "\ts_add_nc_u64 s[58:59], s[36:37], s[62:63]", W, "\ts_lshl_b64 s[62:63], s[26:27], 12",
                "\ts_add_nc_u64 s[60:61], s[38:39], 64", W]
    bump_sgpr(64)

if "wc" in opts:
    # exact: drop the dead loop counter s6 (sp removed its only reader) and every s_wait_alu depctr_sa_sdst(0) that
    # has no SALU instruction since the previous one (straight line; labels/branches reset).
    assert "sp" in opts
    a = out.index(TAIL)
    b = out.index(".LBB0_43:")
    reg = [l for l in out[a:b] if re.search(r"\bs6\b|s\[6:", l.split(";")[0])]
    assert set(reg) == {"\ts_add_co_i32 s6, s6, 1"}, reg
    res, armed, n0 = [], False, b - a
    for l in out[a:b]:
        if l == "\ts_add_co_i32 s6, s6, 1":
            continue
        t = l.split(";")[0].strip()
        if l == W:
            if armed:
                continue
            armed = True
        elif not t or t.startswith(("s_wait", "s_delay", "s_setprio", "s_barrier", "v_", "ds_", "global_")):
            pass
        else:
            armed = False  # SALU, label or branch
        res.append(l)
    out = out[:a] + res + out[b:]
    print("wc: removed", n0 - len(res))
    # the tail's exec restore (s_or exec, s9) is a no-op unless the rescale block (which masks exec) ran: move it to
    # the end of the rescale blocks
    b = out.index(".LBB0_43:")
    OR = "\ts_or_b32 exec_lo, exec_lo, s9"
    tl = [k for k in range(a, b) if out[k] in (TAIL, ".LU36_0:" + TAIL[len(".LBB0_36:"):])]
    assert len(tl) == (2 if "u2" in opts else 1), tl
    for k in reversed(tl):
        assert out[k + 1:k + 4] == ["\ts_setprio 3", W, OR], out[k + 1:k + 4]
        del out[k + 2:k + 4]
    b = out.index(".LBB0_43:")
    ends = [k for k in range(a, b) if out[k] == "\ts_branch .LBB0_36" or out[k].startswith(".LU36_0:")]
    assert len(ends) == len(tl), ends
    for k in reversed(ends):
        assert out[k - 1].startswith("\tv_dual_mul_f32 v7, v204, v7"), out[k - 1]
        out[k:k] = [W, OR]

for o in opts:
    if o.startswith("dg"):
        # exact: group the tail's 8 row-sum dot4 (same order per accumulator) into one block. dg / dg0: before the
        # first V-load wait (after the first loads issue); dg1: after the last PV WMMA (before the loop branch);
        # dg2: right after the tail's first two WMMAs.
        mode = o[2:] or "0"
        a = out.index(TAIL)
        b = out.index(".LBB0_43:")
        tl = [k for k in range(a, b) if out[k] == TAIL or out[k].startswith(".LU36_0:")]
        for k0 in reversed(tl):
            k1 = next(k for k in range(k0, b) if out[k].startswith("\ts_cbranch_scc1 .LBB0_44"))
            seg = out[k0:k1 + 1]
            dots = [l for l in seg if l.startswith("\tv_dot4_f32_fp8_fp8")]
            assert len(dots) == 8, dots
            seg = [l for l in seg if not l.startswith("\tv_dot4_f32_fp8_fp8")]
            if mode == "0":
                i = next(k for k, l in enumerate(seg) if l.startswith("\ts_wait_dscnt 0x2"))
            elif mode == "1":
                i = len(seg) - 1
            else:
                w = [k for k, l in enumerate(seg) if l.startswith("\tv_wmma")]
                i = w[1] + 1
            seg[i:i] = dots
            out[k0:k1 + 1] = seg

if "vd" in opts:
    # exact: pair the three row-max v_max_num_f32 couples of the softmax into VOPD v_dual_max_num_f32. The B-half
    # max temp v153 is swapped with v156 (both are softmax temps, dead outside [first max3, last max]) so that the
    # paired destinations differ in parity; source banks differ (209/156, 201/203, 154/155, 214/213, 209/156).
    a = out.index(TAIL)
    b = out.index(".LBB0_43:")
    starts = [k for k in range(a, b) if out[k] == "\tv_max3_num_f32 v209, v215, v216, v217"]
    nloop = 2 if "u2" in opts else 1
    assert len(starts) == nloop, starts
    for s in reversed(starts):
        e = out.index("\tv_max_num_f32_e32 v210, v213, v153", s)
        for k in range(s, e + 1):
            if out[k].startswith("\t") and not out[k].startswith("\ts_delay"):
                out[k] = _remap_line(out[k], {153: 156, 156: 153})
        pairs = [("\tv_max_num_f32_e32 v209, v209, v201", "\tv_max_num_f32_e32 v156, v156, v203",
                  "\tv_dual_max_num_f32 v209, v209, v201 :: v_dual_max_num_f32 v156, v156, v203"),
                 ("\tv_max_num_f32_e32 v209, v209, v154", "\tv_max_num_f32_e32 v156, v156, v155",
                  "\tv_dual_max_num_f32 v209, v209, v154 :: v_dual_max_num_f32 v156, v156, v155"),
                 ("\tv_max_num_f32_e32 v209, v214, v209", "\tv_max_num_f32_e32 v210, v213, v156",
                  "\tv_dual_max_num_f32 v209, v214, v209 :: v_dual_max_num_f32 v210, v213, v156")]
        for x, y, d in pairs:
            i = out.index(x, s, e + 1)
            j = out.index(y, i, e + 1)
            mid = out[i + 1:j]
            assert all(not m.startswith("\tv_") for m in mid), mid
            out[j] = d
            del out[i]
            e -= 1

if "qs" in opts:
    # exact (int32 QK accumulation is associative): reschedule each header's QK block (32 iu8 WMMAs, 4 accumulator
    # chains x 8 d-steps, K fragments from LDS) as a strict 4-chain round robin (same-accumulator distance 4, was 2)
    # with a 4th K buffer v[201:204] (softmax temp, dead during QK) so the K fragment loads run 8 WMMAs ahead.
    import re as _re
    a = out.index(TAIL)
    b = out.index(".LBB0_43:")
    ends = [k for k in range(a, b) if out[k] == "\ts_setprio 2"]
    assert len(ends) == (2 if "u2" in opts else 1), ends
    RL = _re.compile(r"\tds_load_2addr_b64 v\[(\d+):(\d+)\], (v\d+) (?:offset0:(\d+) )?offset1:(\d+)$")
    RW = _re.compile(r"\tv_wmma_i32_16x16x16_iu8 v\[(\d+):(\d+)\], v\[(\d+):(\d+)\], v\[(\d+):(\d+)\], (\S+) neg_lo:\[1,1,0\]$")
    for e in reversed(ends):
        s = e
        while not out[s - 1].startswith("\ts_or_b32 exec_lo, exec_lo, s9"):
            s -= 1
        while not out[s].startswith("\tds_load"):
            s += 1
        seg = out[s:e]
        frag = {}      # pair reg -> (base, offset)
        w = []         # (acc, base, off, qpair, cfield)
        keep = []
        for l in seg:
            m = RL.match(l)
            if m:
                r0 = int(m.group(1)); o0 = int(m.group(4) or 0); o1 = int(m.group(5))
                frag[r0] = (m.group(3), o0); frag[r0 + 2] = (m.group(3), o1)
                continue
            m = RW.match(l)
            if m:
                acc = int(m.group(1)); ar = int(m.group(3)); q = (int(m.group(5)), int(m.group(6)))
                base, off = frag[ar]
                w.append((acc, base, off, q, m.group(7)))
                continue
            if l.startswith("\ts_wait_dscnt") or l.startswith("\ts_delay_alu"):
                continue
            keep.append(l)
        assert len(w) == 32, len(w)
        bases = sorted({x[1] for x in w}, key=lambda bb: min(x[2] for x in w if x[1] == bb))
        assert len(bases) == 2, bases
        HB, GB = bases
        # per base: offsets sorted -> d index; per (acc): base, q by d
        offs = {bb: sorted({x[2] for x in w if x[1] == bb}) for bb in bases}
        assert all(len(v) == 8 for v in offs.values()), offs
        # chains: for each base two accumulators; for each acc the q pair per d
        chains = {}
        for acc, bb, off, q, c in w:
            chains.setdefault(acc, {"base": bb, "q": {}, "init": None})
            d = offs[bb].index(off)
            assert chains[acc]["base"] == bb
            chains[acc]["q"][d] = q
            if c != "v[%d:%d]" % (acc, acc + 7):
                assert chains[acc]["init"] is None and d == 0, (acc, d, c)
                chains[acc]["init"] = c
        assert len(chains) == 4 and all(len(c["q"]) == 8 and c["init"] for c in chains.values())
        hacc = [k for k in chains if chains[k]["base"] == HB]
        gacc = [k for k in chains if chains[k]["base"] == GB]
        # order accs so the "a" accumulator (same q as the G one at d0) pairs consistently
        qa = chains[hacc[0]]["q"][0]
        ga = [k for k in gacc if chains[k]["q"][0] == qa]
        assert len(ga) == 1
        h1, h2 = hacc
        g1 = ga[0]; g2 = [k for k in gacc if k != g1][0]
        R = [231, 235, 153, 201]

        def ld(bb, step, r):
            o0, o1 = offs[bb][2 * step], offs[bb][2 * step + 1]
            assert o1 == o0 + 2
            oo = ("offset0:%d offset1:%d" % (o0, o1)) if o0 else ("offset1:%d" % o1)
            return "\tds_load_2addr_b64 v[%d:%d], %s %s" % (r, r + 3, bb, oo)

        def wm(acc, r, d):
            c = chains[acc]
            q = c["q"][d]
            cf = c["init"] if d == 0 else "v[%d:%d]" % (acc, acc + 7)
            return "\tv_wmma_i32_16x16x16_iu8 v[%d:%d], v[%d:%d], v[%d:%d], %s neg_lo:[1,1,0]" % (
                acc, acc + 7, r, r + 1, q[0], q[1], cf)

        new = [ld(HB, 0, R[0]), ld(GB, 0, R[1]), ld(HB, 1, R[2]), ld(GB, 1, R[3])]
        pro = [l for l in keep if l != "\ts_setprio 1"]
        assert len(pro) == len(keep) - 1
        new += pro + ["\ts_setprio 1"]
        for st in range(4):
            rh, rg = R[2 * (st % 2)], R[2 * (st % 2) + 1]
            last = st == 3
            new.append("\ts_wait_dscnt 0x%d" % (1 if last else 3))
            for j in range(2):
                d = 2 * st + j
                if j == 0:
                    new += [wm(h1, rh, d), wm(h2, rh, d), "\ts_wait_dscnt 0x%d" % (0 if last else 2),
                            wm(g1, rg, d), wm(g2, rg, d)]
                else:
                    new += [wm(h1, rh + 2, d), wm(h2, rh + 2, d), wm(g1, rg + 2, d), wm(g2, rg + 2, d)]
            if st < 2:
                new += [ld(HB, st + 2, rh), ld(GB, st + 2, rg)]
        out[s:e] = new
    print("qs: rescheduled", len(ends))

for o in opts:
    if o.startswith("ts"):
        # exact: reschedule each tail's PV block. All 8 V fragment loads (each quad = both k halves for one pair of
        # accumulators; every tail-load destination is dead at tail entry) issue up front; WMMAs go in load pairs
        # (La k0, Lb k0, La k1, Lb k1) so the same-accumulator distance is 4 (k0 before k1 per accumulator is kept, so
        # fp32 accumulation order is unchanged). The 8 dot4 row sums keep their order, one after every 2 WMMAs
        # (ts) or after every 4 (ts4). s_barrier_signal goes right after the last load wait.
        import re as _re
        every = int(o[2:] or 2)
        RL = _re.compile(r"\tds_load_2addr_b64 v\[(\d+):(\d+)\], (v\d+) (?:offset0:(\d+) )?offset1:(\d+)$")
        RW = _re.compile(r"\tv_wmma_f32_16x16x16_fp8_fp8 v\[(\d+):(\d+)\], v\[(\d+):(\d+)\], v\[(\d+):(\d+)\], v\[(\d+):(\d+)\]$")
        a = out.index(TAIL)
        b = out.index(".LBB0_43:")
        tl = [k for k in range(a, b) if out[k] == TAIL or out[k].startswith(".LU36_0:")]
        assert len(tl) == (2 if "u2" in opts else 1), tl
        for k0 in reversed(tl):
            k1 = next(k for k in range(k0, b) if out[k].startswith("\ts_cbranch_scc1 .LBB0_44"))
            seg = out[k0 + 1:k1]
            loads, wm, dots, pre = [], [], [], []
            seen_load = False
            for l in seg:
                if RL.match(l):
                    loads.append(l); seen_load = True
                elif RW.match(l):
                    wm.append(l)
                elif l.startswith("\tv_dot4_f32_fp8_fp8"):
                    dots.append(l)
                elif l.startswith("\ts_delay_alu") or l == "\ts_barrier_signal -1":
                    pass
                elif l.startswith("\ts_wait_dscnt") and seen_load:
                    pass
                else:
                    assert not seen_load or not l.startswith("\tv_"), l
                    pre.append(l)
            assert len(loads) == 8 and len(wm) == 32 and len(dots) == 8, (len(loads), len(wm), len(dots))
            assert sum(1 for l in seg if l == "\ts_barrier_signal -1") == 1
            # groups per load
            grp = []
            for l in loads:
                r0 = int(RL.match(l).group(1))
                k0w = [x for x in wm if int(RW.match(x).group(3)) == r0]
                k1w = [x for x in wm if int(RW.match(x).group(3)) == r0 + 2]
                assert len(k0w) == 2 and len(k1w) == 2, (l, k0w, k1w)
                for x in k0w + k1w:
                    m = RW.match(x)
                    assert int(m.group(7)) == int(m.group(1))   # accumulate
                # per accumulator k0 precedes k1 in the original
                for x in k1w:
                    acc = RW.match(x).group(1)
                    y = [z for z in k0w if RW.match(z).group(1) == acc]
                    assert len(y) == 1 and wm.index(y[0]) < wm.index(x)
                grp.append((k0w, k1w))
            assert sum(len(g[0]) + len(g[1]) for g in grp) == 32
            new = list(pre) + loads
            seq = []
            for p in range(4):
                ga, gb = grp[2 * p], grp[2 * p + 1]
                seq.append(("w", 7 - 2 * p)); seq += [("m", x) for x in ga[0]]
                seq.append(("w", 6 - 2 * p)); seq += [("m", x) for x in gb[0]]
                seq += [("m", x) for x in ga[1]] + [("m", x) for x in gb[1]]
            nm = 0
            di = 0
            for kind, x in seq:
                if kind == "w":
                    new.append("\ts_wait_dscnt 0x%x" % x)
                    if x == 0:
                        new.append("\ts_barrier_signal -1")
                else:
                    new.append(x)
                    nm += 1
                    if nm % every == 0 and di < len(dots):
                        new.append(dots[di]); di += 1
            assert di == 8
            out[k0 + 1:k1] = new
        print("ts: rescheduled", len(tl))

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

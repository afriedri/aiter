#!/usr/bin/env python3
# Track C generator: runs jpv's mk_variants.py (base opts) and applies Track C post-passes.
# usage: [MK_SRC=attention_pv_early.s] mk_c.py <out.s> <opts...>   (C-only opts: lax, cln, lds=<bytes>, xd+<prefixes>)
# esp321x: e1 addr canon n16 lsum ldsa bm256 sa tmax sp321 plx sload lax lds=65536
# esp321y: e1 addr canon n16 lsum ldsa bm256 sa tmax sp321 plx sload lax lds=65536 pr=3,1,2 ins=sm:3
#   (option order matters: pr= before ins=)
#   lax : exact lazy alpha. The rescale factor exp2(m_old-m_new) is computed only inside the rescale branch
#         (entered when some lane has d = m_old-m_new != 0). Inside the branch: l_new = fma(l_old, alpha, tilesum)
#         (same fma as Esp321) and l_old := 0; at the PV tail the fma becomes an add of l_old (= +0 for rescaled lanes,
#         = l_old for alpha==1 lanes, which is exactly fma(l_old,1,sum)). Bitwise identical to the non-lazy kernel.
import subprocess, sys, os, tempfile

out_path = sys.argv[1]
opts = sys.argv[2:]
COPTS = {"lax", "cln"}
isc = lambda o: o in COPTS or o.startswith(("xd+", "lds=", "bmq=", "xv", "pr=", "ins="))
base = [o for o in opts if not isc(o)]
mine = [o for o in opts if isc(o)]
here = os.path.dirname(os.path.abspath(__file__))
tmp = out_path + ".base.s"
subprocess.check_call([sys.executable, os.path.join(here, "mk_variants.py"), tmp] + base)
out = open(tmp).read().split("\n")
os.remove(tmp)

HDR = ".LBB0_37:                               ; =>This Inner Loop Header: Depth=1"
TAIL = ".LBB0_36:                               ;   in Loop: Header=BB0_37 Depth=1"

if "lax" in mine:
    assert "la" not in base
    a = out.index(HDR)
    b = out.index(".LBB0_43:")
    for r in ("v255", "v213"):
        i = out.index(f"\tv_exp_f32_e32 {r}, {r}", a)
        assert i < b
        del out[i]
    for d, l, s, nxt in (("v255", "v212", "v200", ".LBB0_41"), ("v213", "v211", "v196", ".LBB0_36")):
        i = out.index(f"\tv_cmpx_neq_f32_e32 1.0, {d}", a)
        out[i] = f"\tv_cmpx_neq_f32_e32 0, {d}"
        assert out[i + 1] == f"\ts_cbranch_execz {nxt}", out[i + 1]
        out[i + 2:i + 2] = [f"\tv_exp_f32_e32 {d}, {d}", "\ts_delay_alu instid0(TRANS32_DEP_1)",
                            f"\tv_fmac_f32_e32 {s}, {l}, {d}", f"\tv_mov_b32_e32 {l}, 0"]
    t = out.index(TAIL)
    for f, g in (("\tv_fmac_f32_e32 v200, v212, v255", "\tv_add_f32_e32 v200, v200, v212"),
                 ("\tv_fmac_f32_e32 v196, v211, v213", "\tv_add_f32_e32 v196, v196, v211")):
        j = out.index(f, t)
        assert j < a
        out[j] = g

if "cln" in mine:
    # exact: in the main loop drop s_wait_dscnt N immediately followed by s_wait_dscnt M<=N, and s_delay_alu
    # immediately followed by another s_delay_alu (comment lines ignored).
    a = out.index(TAIL)
    b = out.index(".LBB0_43:")
    body = out[a:b]
    keep = []
    for i, l in enumerate(body):
        j = i + 1
        while j < len(body) and body[j].strip().startswith(";"):
            j += 1
        nx = body[j] if j < len(body) else ""
        if l.startswith("\ts_wait_dscnt") and nx.startswith("\ts_wait_dscnt"):
            if int(nx.split()[1], 16) <= int(l.split()[1], 16):
                continue
        if l.startswith("\ts_delay_alu") and nx.startswith("\ts_delay_alu"):
            continue
        keep.append(l)
    out = out[:a] + keep + out[b:]

for o in mine:
    if o.startswith("bmq="):
        # exact: workgroups of N query rows (N = 32*waves, 256 < N <= 512, built on bm256). Only the first 256 lanes
        # stage K/V (s0 = tid < 256, unchanged). q-blocks: nqb = ceil(S/N) via a mul_hi by ceil(2^32/N); the row base
        # qb*N is no longer a power-of-two multiple, so the OR that forms the row index becomes an add.
        # NOT USEFUL on R9600D: at 256 VGPRs only one WG of >256 lanes fits per WGP (measured), kept for reference.
        assert "bm256" in base
        N = int(o[4:])
        assert N % 32 == 0 and 256 < N <= 512
        mg = -(-(1 << 32) // N)
        assert all(((x * mg) >> 32) == x // N for x in (N - 1, N, 10 * N - 1, 10 * N, 200000 + N - 1, 1 << 22))

        def rep(old, new, cnt=1):
            idx = [i for i, l in enumerate(out) if l == old]
            assert len(idx) == cnt, (old, len(idx))
            for i in idx:
                out[i] = new
        rep("\ts_add_co_i32 s2, s25, 0xff", f"\ts_add_co_i32 s2, s25, {N - 1}")
        rep("\ts_lshr_b32 s2, s2, 8", f"\ts_mul_hi_u32 s2, s2, {hex(mg)}")
        rep("\ts_add_co_i32 s5, s25, 0xff", f"\ts_add_co_i32 s5, s25, {N - 1}")
        rep("\ts_ashr_i32 s5, s5, 8", f"\ts_mul_hi_u32 s5, s5, {hex(mg)}")
        rep("\ts_lshl_b32 s3, s3, 8", f"\ts_mul_i32 s3, s3, {N}")
        rep("\tv_or3_b32 v1, s3, v1, v195", "\tv_add3_u32 v1, s3, v1, v195")
    if o.startswith("lds="):
        # exact: declare a larger LDS allocation (bytes). lds=65536 places the two resident WGs of a WGP in separate
        # 64 KB LDS halves; measured 0.06-0.12% faster on R9600D, no code change.
        n = int(o[4:])
        for i, l in enumerate(out):
            if "group_segment_fixed_size" in l:
                pre, _, old = l.rpartition(" ")
                assert int(old) <= n
                out[i] = f"{pre} {n}"
    if o.startswith("pr="):
        # exact: main-loop s_setprio values in program order (PV-top, QK, softmax), e.g. pr=2,1,3 (= unchanged)
        vals = o[3:].split(",")
        a = out.index(TAIL)
        b = out.index(".LBB0_43:")
        idx = [i for i in range(a, b) if out[i].startswith("\ts_setprio")]
        assert len(idx) == len(vals), idx
        for i, v in zip(idx, vals):
            out[i] = f"\ts_setprio {v}"
    if o.startswith("ins="):
        # exact: insert an extra s_setprio in the main loop. ins=stg:<v> before the K/V staging (s_and_saveexec),
        # ins=rsc:<v> before the first O-rescale test, ins=sm:<v> after the cvt_pk_norm block (before perm/dot4)
        key, v = o[4:].split(":")
        pat = {"stg": "\ts_and_saveexec_b32 s9, s0", "rsc": "\tv_sub_f32_e32 v213, v213, v210",
               "sm": "\tv_perm_b32 v231, v217, v215, 0x7050301",
               "cvt": "\tv_cvt_pk_norm_u16_f32 v215, v215, v216",
               "fma": "\tv_fma_f32 v215, v215, v254, -v252",
               "dot": "\tv_dot4_f32_fp8_fp8 v200, v231, 0x38383838, 0"}[key]
        a = out.index(HDR)
        i = out.index(pat, a)
        assert i < out.index(".LBB0_43:")
        out.insert(i, f"\ts_setprio {v}")
    if o.startswith("xv"):
        # DIAGNOSTIC ONLY: drop every non-WMMA VALU line of the main loop (xv) or also LDS/waits (xvw).
        a = out.index(TAIL)
        b = out.index(".LBB0_43:")
        drop = lambda l: (l.startswith("\tv_") and not l.startswith("\tv_wmma")) or \
            (o == "xvw" and l.startswith(("\tds_", "\ts_wait_dscnt", "\ts_barrier", "\ts_delay_alu", "\ts_wait_alu")))
        out = out[:a] + [l for l in out[a:b] if not drop(l)] + out[b:]
    if o.startswith("xd+"):
        # DIAGNOSTIC ONLY (wrong results): drop every line of the main loop (.LBB0_36 .. .LBB0_43) starting with one
        # of the '+'-separated prefixes.
        pre = tuple("\t" + p for p in o.split("+")[1:])
        a = out.index(TAIL)
        b = out.index(".LBB0_43:")
        out = out[:a] + [l for l in out[a:b] if not l.startswith(pre)] + out[b:]

open(out_path, "w").write("\n".join(out))
print("mk_c:", " ".join(opts))

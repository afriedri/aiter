#!/usr/bin/env python3
# Track C generator: runs jpv's mk_variants.py (base opts) and applies Track C post-passes.
# usage: [MK_SRC=attention_pv_early.s] mk_c.py <out.s> <opts...>   (C-only opts: lax, cln, lds=<bytes>, xd+<prefixes>)
# esp321x: e1 addr canon n16 lsum ldsa bm256 sa tmax sp321 plx sload lax lds=65536
#   lax : exact lazy alpha. The rescale factor exp2(m_old-m_new) is computed only inside the rescale branch
#         (entered when some lane has d = m_old-m_new != 0). Inside the branch: l_new = fma(l_old, alpha, tilesum)
#         (same fma as Esp321) and l_old := 0; at the PV tail the fma becomes an add of l_old (= +0 for rescaled lanes,
#         = l_old for alpha==1 lanes, which is exactly fma(l_old,1,sum)). Bitwise identical to the non-lazy kernel.
import subprocess, sys, os, tempfile

out_path = sys.argv[1]
opts = sys.argv[2:]
COPTS = {"lax", "cln"}
isc = lambda o: o in COPTS or o.startswith(("xd+", "lds="))
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
    if o.startswith("lds="):
        # exact: declare a larger LDS allocation (bytes). lds=65536 places the two resident WGs of a WGP in separate
        # 64 KB LDS halves; measured 0.06-0.12% faster on R9600D, no code change.
        n = int(o[4:])
        for i, l in enumerate(out):
            if "group_segment_fixed_size" in l:
                pre, _, old = l.rpartition(" ")
                assert int(old) <= n
                out[i] = f"{pre} {n}"
    if o.startswith("xd+"):
        # DIAGNOSTIC ONLY (wrong results): drop every line of the main loop (.LBB0_36 .. .LBB0_43) starting with one
        # of the '+'-separated prefixes.
        pre = tuple("\t" + p for p in o.split("+")[1:])
        a = out.index(TAIL)
        b = out.index(".LBB0_43:")
        out = out[:a] + [l for l in out[a:b] if not l.startswith(pre)] + out[b:]

open(out_path, "w").write("\n".join(out))
print("mk_c:", " ".join(opts))

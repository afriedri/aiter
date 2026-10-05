# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Precompiled gfx1201 SageAttention code object for the specialized sequence."""

import ctypes
import os

import torch

from aiter.jit.core import AITER_ASM_DIR

_runtime = None
_modules = {}

# code object -> query rows (= threads) per workgroup. The launch block MUST match the hsaco.
_VARIANTS = {
    "pv_early": ("attention_pv_early.hsaco", 512),
    "b1": ("attention_pv_early_b1.hsaco", 128),            # bm128, exact (fp32 row-sum reassociation only)
    "b2x": ("attention_pv_early_b2x.hsaco", 256),          # bm256, exact exp (row-sum reassociation, lazy alpha)
    "eb2satc": ("attention_pv_early_eb2satc.hsaco", 256),  # bm256, approximate exp-free FP8 P
    "eb2la": ("attention_pv_early_eb2la.hsaco", 256),      # eb2satc + lazy alpha (same approximation)
    "ec": ("attention_pv_early_ec.hsaco", 256),            # bm256 exp-free P with exact FP8 codes (LUT-corrected)
    "esp321": ("attention_pv_early_esp321.hsaco", 256),    # eb2satc + per-phase wave priority (same approximation)
    "esp321x": ("attention_pv_early_esp321x.hsaco", 256),  # esp321, bitwise identical: permlanex16 row max, s_load k-scale, exact lazy alpha, 64K LDS
    "esp321y": ("attention_pv_early_esp321y.hsaco", 256),  # esp321x, bitwise identical: PV-phase wave priority 3, softmax 2 -> 3 at the P pack
    "esp321z": ("attention_pv_early_esp321z.hsaco", 256),  # esp321y, ~2.6% faster (not bitwise): scalar softmax scale, l accumulated in place, single uniform rescale branch
    "esp321w": ("attention_pv_early_esp321w.hsaco", 256),  # esp321z, ~3% faster (not bitwise; bitwise = mg): 240 VGPRs (3 WG/WGP), loop unrolled x2 with static LDS parity, SALU pointers, 4-chain QK WMMA schedule
    "esp321wsk7": ("attention_pv_early_esp321wsk7.hsaco", 256),  # esp321w + exact zero-P PV skip (inline guarded PV groups, in-group row sums); bitwise == esp321w
    "esp321wsk7t2": ("attention_pv_early_esp321wsk7t2.hsaco", 256),  # APPROX: esp321wsk7, drop 16x16 P blocks whose per-lane 8-key P sums are all < 2^-2
    "esp321wsk7m5": ("attention_pv_early_esp321wsk7m5.hsaco", 256),  # APPROX: esp321wsk7, drop 16x16 P blocks whose FP8 codes are all < 2^-5
}


def _parse_schedule():
    """AITER_GFX1201_SAGE_ASM_SCHED="b2x:250,eb2satc": the first 250 core launches (per process) use b2x,
    the rest eb2satc. Segments are VARIANT[:COUNT]; the last one has no count. Without it, a single variant
    comes from AITER_GFX1201_SAGE_ASM (default pv_early)."""
    spec = os.environ.get("AITER_GFX1201_SAGE_ASM_SCHED", "").strip()
    if not spec:
        return [(os.environ.get("AITER_GFX1201_SAGE_ASM", "pv_early"), None)]
    sched = []
    for seg in spec.split(","):
        name, _, count = seg.strip().partition(":")
        sched.append((name, int(count) if count else None))
    if any(c is None for _, c in sched[:-1]) or sched[-1][1] is not None:
        raise ValueError(f"AITER_GFX1201_SAGE_ASM_SCHED={spec!r}: every segment but the last needs a :COUNT")
    return sched


_SCHEDULE = _parse_schedule()
for _name, _ in _SCHEDULE:
    if _name not in _VARIANTS:
        raise ValueError(f"gfx1201 sage ASM variant {_name!r}; expected one of {sorted(_VARIANTS)}")
_VARIANT = _SCHEDULE[0][0]
_HSACO, _BLOCK = _VARIANTS[_VARIANT]
_calls = 0


def _current_variant():
    left = _calls
    for name, count in _SCHEDULE:
        if count is None or left < count:
            return name
        left -= count
    return _SCHEDULE[-1][0]


def _check(status):
    if status:
        raise RuntimeError(f"HIP module API failed: {status}")


def _function(variant=None):
    global _runtime
    variant = variant or _VARIANT
    hsaco = _VARIANTS[variant][0]
    if _runtime is None:
        _runtime = ctypes.CDLL("libamdhip64.so")
        _runtime.hipModuleLoad.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_char_p,
        ]
        _runtime.hipModuleGetFunction.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            ctypes.c_char_p,
        ]
        _runtime.hipModuleLaunchKernel.argtypes = [ctypes.c_void_p] + [
            ctypes.c_uint
        ] * 7 + [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
        ]
    code_object = os.path.join(
        AITER_ASM_DIR, "gfx1201", "sage_attention", hsaco
    )
    key = (torch.cuda.current_device(), code_object)
    if key not in _modules:
        module = ctypes.c_void_p()
        function = ctypes.c_void_p()
        _check(_runtime.hipModuleLoad(ctypes.byref(module), os.fsencode(key[1])))
        _check(
            _runtime.hipModuleGetFunction(
                ctypes.byref(function), module, b"sage_hip_attn_bm128_bn32"
            )
        )
        _modules[key] = (module, function)
        import sys
        import time
        print(f"[aiter] gfx1201 sage ASM variant={variant} hsaco={hsaco} block={_VARIANTS[variant][1]} "
              f"from call {_calls} (schedule={_SCHEDULE}) t={time.time():.1f}", file=sys.stderr, flush=True)
    return _modules[key][1]


def launch_hip_sage_core(
    q_int8,
    k_int8,
    v_fp8,
    q_scale,
    k_scale,
    v_scale,
    out,
    batch_size,
    padded_seq_len,
    valid_seq_len,
    num_heads,
):
    if (
        min(batch_size, valid_seq_len, num_heads) <= 0
        or padded_seq_len != (valid_seq_len + 31) // 32 * 32
    ):
        raise ValueError(
            "Expected positive shape and sequence padded to a multiple of 32"
        )
    if out.dtype != torch.bfloat16 or out.device != q_int8.device:
        raise ValueError("Expected BF16 output on the input device")
    global _calls
    variant = _current_variant()
    block = _VARIANTS[variant][1]
    _calls += 1
    with torch.cuda.device(q_int8.device):
        function = _function(variant)
        arguments = [
            ctypes.c_void_p(tensor.data_ptr())
            for tensor in (q_int8, k_int8, v_fp8, q_scale, k_scale, v_scale, out)
        ]
        arguments += [
            ctypes.c_int(value)
            for value in (batch_size, padded_seq_len, valid_seq_len, num_heads)
        ]
        parameters = (ctypes.c_void_p * len(arguments))(
            *[ctypes.addressof(argument) for argument in arguments]
        )
        grid = batch_size * ((valid_seq_len + block - 1) // block) * num_heads
        _check(
            _runtime.hipModuleLaunchKernel(
                function,
                grid,
                1,
                1,
                block,
                1,
                1,
                0,
                ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
                parameters,
                None,
            )
        )

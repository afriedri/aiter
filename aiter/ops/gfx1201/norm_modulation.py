# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused RMSNorm (BF16, affine) + AdaLN modulation for the MiniMax-H3 DiT block on gfx1201.

Replaces ``modulate(rms_norm(x, weight, eps), scale, shift, indices)``: one read of x and one
write instead of two of each. The modulation rounding matches ``modulation._modulate`` exactly;
the RMS statistic is an FP32 row reduction in a different order than torch's kernel, so about
2 in 10^6 outputs differ by one BF16 ulp (torch itself differs from an FP64 reference on 7 in 10^6).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _rms_modulate(inputs, weight, scale, shift, indices, output, eps, WIDTH: tl.constexpr,
                  SCALE_STRIDE: tl.constexpr, SHIFT_STRIDE: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    valid = offsets < WIDTH
    base = row.to(tl.int64) * WIDTH
    value = tl.load(inputs + base + offsets, valid, other=0.0).to(tl.float32)
    reciprocal = tl.math.rsqrt(tl.sum(value * value, 0) / WIDTH + eps)
    gain = tl.load(weight + offsets, valid, other=0.0).to(tl.float32)
    normalized = ((value * reciprocal) * gain).to(tl.bfloat16).to(tl.float32)
    group = tl.load(indices + row)
    scale_value = tl.load(scale + group * SCALE_STRIDE + offsets, valid, other=0.0).to(tl.float32)
    factor = (1.0 + scale_value).to(tl.bfloat16).to(tl.float32)
    product = (normalized * factor).to(tl.bfloat16).to(tl.float32)
    shift_value = tl.load(shift + group * SHIFT_STRIDE + offsets, valid, other=0.0).to(tl.float32)
    tl.store(output + base + offsets, (product + shift_value).to(tl.bfloat16), valid)


def rms_modulate(inputs, weight, eps, scale, shift, indices, output=None):
    """output: optional contiguous tensor like inputs to write (rows are independent, so row slices are exact)."""
    if inputs.ndim != 2 or min(inputs.shape) < 1 or not inputs.is_cuda or not inputs.is_contiguous():
        raise ValueError('Expected positive contiguous GPU [S,D] input')
    if weight.shape != (inputs.shape[1],) or not weight.is_contiguous():
        raise ValueError('Expected contiguous [D] RMSNorm weight')
    if any(tensor.dtype != torch.bfloat16 or tensor.device != inputs.device or tensor.requires_grad
           for tensor in (inputs, weight, scale, shift)):
        raise ValueError('Same-device BF16 inference tensors required')
    if scale.ndim != 2 or scale.shape != shift.shape or scale.shape[1] != inputs.shape[1]:
        raise ValueError('Expected matching [groups,D] scale and shift')
    if any(tensor.stride(1) != 1 or tensor.stride(0) < tensor.shape[1] for tensor in (scale, shift)):
        raise ValueError('Modulation tables require contiguous columns and nonoverlapping rows')
    if indices.shape != (inputs.shape[0],) or indices.device != inputs.device or not indices.is_contiguous() \
            or indices.dtype not in (torch.int32, torch.int64):
        raise ValueError('Expected contiguous per-token int32/int64 indices on the same GPU')
    if inputs.shape[1] > 8192:
        raise ValueError('Row width above 8192 unsupported')
    if output is None:
        output = torch.empty_like(inputs)
    elif output.shape != inputs.shape or output.dtype != inputs.dtype or not output.is_contiguous() \
            or output.device != inputs.device:
        raise ValueError('output must be a contiguous tensor like inputs')
    with torch.cuda.device(inputs.device):
        _rms_modulate[(inputs.shape[0],)](
            inputs, weight, scale, shift, indices, output, float(eps), WIDTH=inputs.shape[1],
            SCALE_STRIDE=scale.stride(0), SHIFT_STRIDE=shift.stride(0),
            BLOCK=triton.next_power_of_2(inputs.shape[1]), num_warps=2, enable_fp_fusion=False)
    return output

# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""FP8 (OCP E4M3) W8A8 GEMMs for the MiniMax-H3 TP2 DiT block on gfx1201.

Scheme (same as LightX2V ``fp8-sgl``): per-output-channel weight scales and per-token dynamic
activation scales, ``scale = amax / 448``; FP32 WMMA accumulation; the scales are applied in the
epilogue before the BF16 output rounding. RDNA4 runs FP8 WMMA at twice the BF16 rate.
"""

import os

import torch
import triton
import triton.language as tl

FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0
# Triton lowers fp32 -> float8e4nv to a ~20-instruction software sequence on gfx12; RDNA4 has the hardware
# v_cvt_pk_fp8_f32 (round-to-nearest-even, OCP E4M3), 1 instruction per 2 values. Inputs are pre-clamped to +-448.
HW_CVT = os.environ.get("AITER_GFX1201_FP8_HW_CVT", "1") != "0"


@triton.jit
def _pack_e4m3(a, b, c, d):
    return tl.inline_asm_elementwise(
        asm="v_cvt_pk_fp8_f32 $0, $1, $2\n\tv_cvt_pk_fp8_f32 $0, $3, $4 op_sel:[0,0,1]",
        constraints="=&v,v,v,v,v", args=[a, b, c, d], dtype=tl.int32, is_pure=True, pack=1)


@triton.jit
def _store_e4m3(pointer, base, x, valid, WIDTH: tl.constexpr, BLOCK: tl.constexpr, HW: tl.constexpr):
    """Store fp32 ``x`` ([BLOCK], pre-clamped to +-448) as E4M3 at ``pointer + base + arange(BLOCK)``.
    HW packs 4 values per v_cvt_pk_fp8_f32 pair and stores int32 words (base and the valid length must be
    multiples of 4); otherwise Triton's generic conversion."""
    if HW:
        quad = tl.reshape(x, (BLOCK // 4, 2, 2))
        even, odd = tl.split(quad)          # element 4i + 2j + k -> (even: k=0, odd: k=1)
        e0, e2 = tl.split(even)
        o1, o3 = tl.split(odd)
        words = _pack_e4m3(e0, o1, e2, o3)
        index = tl.arange(0, BLOCK // 4)
        tl.store((pointer + base).to(tl.pointer_type(tl.int32), bitcast=True) + index, words, index * 4 < WIDTH)
    else:
        tl.store(pointer + base + tl.arange(0, BLOCK), x.to(tl.float8e4nv), valid)


@triton.jit
def _quant_rows(inputs, output, scales, WIDTH: tl.constexpr, BLOCK: tl.constexpr, ROWS: tl.constexpr,
                HW: tl.constexpr):
    for step in tl.static_range(ROWS):
        row = tl.program_id(0) * ROWS + step
        offsets = tl.arange(0, BLOCK)
        valid = offsets < WIDTH
        base = row.to(tl.int64) * WIDTH
        value = tl.load(inputs + base + offsets, valid, other=0.0).to(tl.float32)
        amax = tl.max(tl.abs(value), 0)
        scale = tl.where(amax > 0, amax / 448.0, 1.0)
        inverse = tl.where(amax > 0, 448.0 / amax, 1.0)
        _store_e4m3(output, base, tl.clamp(value * inverse, -448.0, 448.0), valid, WIDTH, BLOCK, HW)
        tl.store(scales + row, scale)


def quant_rows(inputs, output=None, scales=None):
    """Per-row E4M3 quantization of a contiguous [R, W] tensor. Returns (fp8 [R,W], fp32 scales [R])."""
    if inputs.ndim != 2 or not inputs.is_contiguous() or not inputs.is_cuda or inputs.shape[1] > 16384:
        raise ValueError('Expected contiguous GPU [R, W<=16384] input')
    rows, width = inputs.shape
    if output is None:
        output = torch.empty((rows, width), device=inputs.device, dtype=FP8)
    if scales is None:
        scales = torch.empty((rows,), device=inputs.device, dtype=torch.float32)
    per = 2 if rows % 2 == 0 else 1
    with torch.cuda.device(inputs.device):
        _quant_rows[(rows // per,)](inputs, output, scales, WIDTH=width, BLOCK=triton.next_power_of_2(width), ROWS=per,
                                    HW=HW_CVT and width % 4 == 0, num_warps=4 if width <= 4096 else 8)
    return output, scales


@triton.jit
def _rms_modulate_fp8(inputs, weight, scale, shift, indices, output, scales, eps, WIDTH: tl.constexpr,
                      SCALE_STRIDE: tl.constexpr, SHIFT_STRIDE: tl.constexpr, BLOCK: tl.constexpr, HW: tl.constexpr):
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
    modulated = (product + shift_value).to(tl.bfloat16).to(tl.float32)
    modulated = tl.where(valid, modulated, 0.0)
    amax = tl.max(tl.abs(modulated), 0)
    row_scale = tl.where(amax > 0, amax / 448.0, 1.0)
    inverse = tl.where(amax > 0, 448.0 / amax, 1.0)
    _store_e4m3(output, base, tl.clamp(modulated * inverse, -448.0, 448.0), valid, WIDTH, BLOCK, HW)
    tl.store(scales + row, row_scale)


def rms_modulate_fp8(inputs, weight, eps, scale, shift, indices):
    """``rms_modulate`` (bit-identical BF16 modulated value) followed by per-token E4M3 quantization."""
    if inputs.ndim != 2 or not inputs.is_contiguous() or inputs.dtype != torch.bfloat16 or inputs.shape[1] > 8192:
        raise ValueError('Expected contiguous BF16 [S, D<=8192] input')
    if any(t.stride(1) != 1 for t in (scale, shift)) or weight.shape != (inputs.shape[1],):
        raise ValueError('Bad modulation tables / weight')
    rows, width = inputs.shape
    output = torch.empty((rows, width), device=inputs.device, dtype=FP8)
    scales = torch.empty((rows,), device=inputs.device, dtype=torch.float32)
    with torch.cuda.device(inputs.device):
        _rms_modulate_fp8[(rows,)](inputs, weight, scale, shift, indices, output, scales, float(eps), WIDTH=width,
                                   SCALE_STRIDE=scale.stride(0), SHIFT_STRIDE=shift.stride(0),
                                   BLOCK=triton.next_power_of_2(width), HW=HW_CVT and width % 4 == 0, num_warps=4,
                                   enable_fp_fusion=False)
    return output, scales


@triton.jit
def _tile(program, rows, BM: tl.constexpr, column_tiles: tl.constexpr, GROUP: tl.constexpr):
    row_tiles = tl.cdiv(rows, BM)
    group_size: tl.constexpr = GROUP * column_tiles
    first_row = (program // group_size) * GROUP
    active_rows = tl.minimum(row_tiles - first_row, GROUP)
    row_tile = first_row + program % active_rows
    column_tile = (program % group_size) // active_rows
    return row_tile, column_tile


@triton.jit
def _mm8(inputs, input_scales, weights, weight_scales, output, bias, rows, N: tl.constexpr, K: tl.constexpr,
         BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP: tl.constexpr, HAS_BIAS: tl.constexpr):
    row_tile, column_tile = _tile(tl.program_id(0), rows, BM, N // BN, GROUP)
    row = row_tile * BM + tl.arange(0, BM)
    column = column_tile * BN + tl.arange(0, BN)
    inner = tl.arange(0, BK)
    accumulator = tl.zeros((BM, BN), tl.float32)
    for block in range(K // BK):
        position = block * BK + inner
        activation = tl.load(inputs + row[:, None] * K + position[None, :], row[:, None] < rows, other=0.0)
        weight = tl.load(weights + column[None, :] * K + position[:, None])
        accumulator = tl.dot(activation, weight, accumulator)
    row_scale = tl.load(input_scales + row, row < rows, other=0.0)
    column_scale = tl.load(weight_scales + column)
    accumulator = accumulator * row_scale[:, None] * column_scale[None, :]
    if HAS_BIAS:
        accumulator = accumulator + tl.load(bias + column).to(tl.float32)[None, :]
    tl.store(output + row[:, None] * N + column[None, :], accumulator.to(output.dtype.element_ty), row[:, None] < rows)


def mm8_out(inputs, input_scales, weights, weight_scales, output, bias=None):
    """output[R,N] (BF16/FP16) = (inputs[R,K] * s_in[R]) @ (weights[N,K] * s_w[N])^T (+ bias), FP8 operands."""
    rows, K = inputs.shape
    N = weights.shape[0]
    if weights.shape[1] != K or N % 256 or K % 64 or output.shape != (rows, N) or not output.is_contiguous():
        raise ValueError('Shape mismatch')
    if inputs.dtype != FP8 or weights.dtype != FP8 or output.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError('FP8 operands and BF16/FP16 output required')
    if not (inputs.is_contiguous() and weights.is_contiguous()) or max(inputs.numel(), output.numel()) >= 2**31:
        raise ValueError('Contiguous operands within int32 indexing required')
    if bias is not None and (bias.shape != (N,) or not bias.is_contiguous()):
        raise ValueError('Expected contiguous [N] bias')
    with torch.cuda.device(inputs.device):
        _mm8[(triton.cdiv(rows, 128) * (N // 256),)](inputs, input_scales, weights, weight_scales, output,
                                                   bias if bias is not None else output, rows, N, K,
                                                   BM=128, BN=256, BK=64, GROUP=16, num_warps=8, num_stages=2,
                                                   HAS_BIAS=bias is not None)
    return output


@triton.jit
def _up8_swiglu(inputs, input_scales, weights, weight_scales, output, rows,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP: tl.constexpr):
    reduction: tl.constexpr = 5376
    channels: tl.constexpr = 7168
    row_tile, column_tile = _tile(tl.program_id(0), rows, BM, channels // (BN // 2), GROUP)
    row = row_tile * BM + tl.arange(0, BM)
    paired_column = tl.arange(0, BN)
    column = column_tile * (BN // 2) + paired_column % (BN // 2)
    weight_column = column + (paired_column // (BN // 2)) * channels
    inner = tl.arange(0, BK)
    accumulator = tl.zeros((BM, BN), tl.float32)
    for block in range(reduction // BK):
        position = block * BK + inner
        activation = tl.load(inputs + row[:, None] * reduction + position[None, :], row[:, None] < rows, other=0.0)
        weight = tl.load(weights + weight_column[None, :] * reduction + position[:, None])
        accumulator = tl.dot(activation, weight, accumulator)
    row_scale = tl.load(input_scales + row, row < rows, other=0.0)
    column_scale = tl.load(weight_scales + weight_column)
    accumulator = accumulator * row_scale[:, None] * column_scale[None, :]
    paired = accumulator.to(tl.bfloat16).to(tl.float32).reshape((BM, 2, BN // 2)).trans(0, 2, 1)
    value, gate = tl.split(paired)
    activated = (gate / (1.0 + tl.exp(-gate))).to(tl.bfloat16).to(tl.float32)
    out_column = column_tile * (BN // 2) + tl.arange(0, BN // 2)
    tl.store(output + row[:, None] * channels + out_column[None, :], (value * activated).to(tl.bfloat16),
             row[:, None] < rows)


def up8_swiglu_out(inputs, input_scales, weights, weight_scales, output):
    """SwiGLU(x @ W^T) for the paired TP2 [14336,5376] weight (value rows 0..7167, gate rows 7168..), BF16 [R,7168] out."""
    rows = inputs.shape[0]
    if inputs.shape[1] != 5376 or weights.shape != (14336, 5376) or output.shape != (rows, 7168):
        raise ValueError('Shape mismatch')
    if inputs.dtype != FP8 or weights.dtype != FP8 or output.dtype != torch.bfloat16 or rows * 7168 >= 2**31:
        raise ValueError('FP8 operands and BF16 output required')
    with torch.cuda.device(inputs.device):
        _up8_swiglu[(triton.cdiv(rows, 128) * 56,)](inputs, input_scales, weights, weight_scales, output, rows,
                                                  BM=128, BN=256, BK=64, GROUP=32, num_warps=8, num_stages=2,
                                                  enable_fp_fusion=False)
    return output


@triton.jit
def _qkv8(inputs, input_scales, wq, wk, wv, sq, sk, sv, oq, ok, ov, rows,
          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP: tl.constexpr):
    reduction: tl.constexpr = 5376
    columns: tl.constexpr = 3584
    per: tl.constexpr = columns // BN
    row_tile, column_tile = _tile(tl.program_id(0), rows, BM, 3 * per, GROUP)
    which = column_tile // per
    local = column_tile % per
    weights = tl.where(which == 0, wq, tl.where(which == 1, wk, wv))
    weight_scales = tl.where(which == 0, sq, tl.where(which == 1, sk, sv))
    output = tl.where(which == 0, oq, tl.where(which == 1, ok, ov))
    row = row_tile * BM + tl.arange(0, BM)
    column = local * BN + tl.arange(0, BN)
    inner = tl.arange(0, BK)
    accumulator = tl.zeros((BM, BN), tl.float32)
    for block in range(reduction // BK):
        position = block * BK + inner
        activation = tl.load(inputs + row[:, None] * reduction + position[None, :], row[:, None] < rows, other=0.0)
        weight = tl.load(weights + column[None, :] * reduction + position[:, None])
        accumulator = tl.dot(activation, weight, accumulator)
    row_scale = tl.load(input_scales + row, row < rows, other=0.0)
    column_scale = tl.load(weight_scales + column)
    accumulator = accumulator * row_scale[:, None] * column_scale[None, :]
    tl.store(output + row[:, None] * columns + column[None, :], accumulator.to(tl.bfloat16), row[:, None] < rows)


def qkv8(inputs, input_scales, weights, weight_scales):
    rows = inputs.shape[0]
    if inputs.shape[1] != 5376 or inputs.dtype != FP8 or any(w.shape != (3584, 5376) or w.dtype != FP8 for w in weights):
        raise ValueError('Expected FP8 [S,5376] input and three FP8 [3584,5376] weights')
    if rows * 5376 >= 2**31:
        raise ValueError('32-bit index limit exceeded')
    outputs = [torch.empty((rows, 3584), device=inputs.device, dtype=torch.bfloat16) for _ in range(3)]
    with torch.cuda.device(inputs.device):
        _qkv8[(triton.cdiv(rows, 128) * 3 * 14,)](inputs, input_scales, *weights, *weight_scales, *outputs, rows,
                                                BM=128, BN=256, BK=64, GROUP=32, num_warps=8, num_stages=2)
    return tuple(outputs)


class WeightQuantizer:
    """Per-output-channel E4M3 quantization of [N,K] K-contiguous weights into reusable buffers.

    Block-offloaded weights arrive in ping-pong device buffers, so each block call re-quantizes (about
    0.6 GB of traffic per block, ~1-2 ms). The buffers are keyed by name so memory stays bounded.
    """

    def __init__(self):
        self.buffers = {}

    def __call__(self, name, weight):
        if weight.ndim != 2 or not weight.is_contiguous() or weight.dtype != torch.bfloat16:
            raise ValueError(f'{name}: expected contiguous BF16 [N,K] weight')
        key = (name, tuple(weight.shape), weight.device)
        entry = self.buffers.get(key)
        if entry is None:
            entry = (torch.empty(weight.shape, device=weight.device, dtype=FP8),
                     torch.empty((weight.shape[0],), device=weight.device, dtype=torch.float32))
            self.buffers[key] = entry
        return quant_rows(weight, *entry)

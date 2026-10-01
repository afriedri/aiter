# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""One-launch Q/K/V projection for MiniMax-H3 TP2 on gfx1201.

X[S,5376] @ {Wq,Wk,Wv}[3584,5376]^T -> three contiguous [S,3584] BF16 outputs.
Same tile schedule as the FFN ``_down`` kernel (128x256x64, 8 warps, 2 stages).
FP32 accumulation in K order; the result is not bitwise equal to hipBLASLt (different
kernel), but its error against an FP64 reference is identical (BF16 output rounding).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _qkv(inputs, wq, wk, wv, oq, ok, ov, rows,
         BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP: tl.constexpr):
    reduction: tl.constexpr = 5376
    columns: tl.constexpr = 3584
    per: tl.constexpr = columns // BN
    column_tiles: tl.constexpr = 3 * per
    program = tl.program_id(0)
    row_tiles = tl.cdiv(rows, BM)
    group_size: tl.constexpr = GROUP * column_tiles
    first_row = (program // group_size) * GROUP
    active_rows = tl.minimum(row_tiles - first_row, GROUP)
    row_tile = first_row + program % active_rows
    column_tile = (program % group_size) // active_rows
    which = column_tile // per
    local = column_tile % per
    weights = tl.where(which == 0, wq, tl.where(which == 1, wk, wv))
    output = tl.where(which == 0, oq, tl.where(which == 1, ok, ov))
    row = row_tile * BM + tl.arange(0, BM)
    column = local * BN + tl.arange(0, BN)
    inner = tl.arange(0, BK)
    accumulator = tl.full((BM, BN), 0, tl.float32)
    for block in range(reduction // BK):
        position = block * BK + inner
        activation = tl.load(inputs + row[:, None] * reduction + position[None, :],
                             row[:, None] < rows, other=0)
        weight = tl.load(weights + column[None, :] * reduction + position[:, None])
        accumulator = tl.dot(activation, weight, accumulator)
    tl.store(output + row[:, None] * columns + column[None, :],
             accumulator.to(tl.bfloat16), row[:, None] < rows)


def qkv_projection(inputs, query_weight, key_weight, value_weight, outputs=None):
    """inputs [S,5376]; weights [3584,5376] contiguous (TP2 local heads). Returns (query, key, value) [S,3584].
    outputs: optional three contiguous BF16 [S,3584] tensors to write (e.g. row slices of larger buffers; each output
    row depends only on its input row, so row-split launches are bitwise equal to one launch)."""
    if inputs.ndim != 2 or inputs.shape[0] < 1 or inputs.shape[1] != 5376:
        raise ValueError('Expected positive [S,5376] input')
    weights = (query_weight, key_weight, value_weight)
    if any(weight.shape != (3584, 5376) or not weight.is_contiguous() for weight in weights):
        raise ValueError('Expected contiguous TP2 [3584,5376] Q/K/V weights')
    if any(tensor.dtype != torch.bfloat16 or not tensor.is_cuda or tensor.device != inputs.device or tensor.requires_grad
           for tensor in (inputs,) + weights) or not inputs.is_contiguous():
        raise ValueError('Contiguous BF16 inference tensors on one GPU required')
    if max(inputs.numel(), inputs.shape[0] * 3584) >= 2**31:
        raise ValueError('32-bit index limit exceeded')
    rows = inputs.shape[0]
    if outputs is None:
        outputs = [torch.empty((rows, 3584), device=inputs.device, dtype=torch.bfloat16) for _ in range(3)]
    elif len(outputs) != 3 or any(o.shape != (rows, 3584) or o.dtype != torch.bfloat16 or not o.is_contiguous()
                                  or o.device != inputs.device for o in outputs):
        raise ValueError('Expected three contiguous BF16 [S,3584] outputs')
    with torch.cuda.device(inputs.device):
        _qkv[(triton.cdiv(rows, 128) * 3 * (3584 // 256),)](
            inputs, *weights, *outputs, rows, BM=128, BN=256, BK=64, GROUP=32, num_warps=8, num_stages=2)
    return tuple(outputs)

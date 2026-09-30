# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.distributed as dist
import triton
from .tp_reduce import FusedReducer
from .gated_residual import _gated_residual

from .ffn_up import fused_up_out
from .ffn_down import down_out
from . import fp8_gemm


class BufferedFFN:
    def __init__(self, rows, token_rows, device, group=None):
        if rows < 1 or token_rows < 1 or rows * 7168 >= 2**31:
            raise ValueError('Positive dimensions within int32 indexing required')
        self.rows = rows
        self.width = min(rows, token_rows)
        self.group = group
        self.device = torch.device(device)
        if group is not None and dist.get_world_size(group) != 2:
            raise ValueError('TP2 only')
        with torch.cuda.device(self.device):
            self.activation = torch.empty((self.width, 7168), device=self.device, dtype=torch.bfloat16)
            self.activation8 = None
            self.reducer = FusedReducer((rows, 5376), group)
            self.output = self.reducer.output
            self.communication = torch.cuda.Stream() if group is not None else None
            self.last = torch.cuda.Event()
            self.last.record(torch.cuda.current_stream())

    def run(self, inputs, up_weights, down_weights, residual, gate, indices, custom=True):
        overlap = True
        if inputs.shape != (self.rows, 5376) or up_weights.shape != (14336, 5376):
            raise ValueError('Expected TP2 [S,5376] input and paired [14336,5376] upper weight')
        if down_weights.shape != (7168, 5376) or down_weights.stride() != (1, 7168):
            raise ValueError('Expected transposed [7168,5376] lower weight')
        if any(tensor.device != self.device or tensor.dtype != torch.bfloat16 or tensor.requires_grad
               for tensor in (inputs, up_weights, down_weights)):
            raise ValueError('Same-device BF16 inference only')
        if not inputs.is_contiguous() or not up_weights.is_contiguous():
            raise ValueError('Contiguous input and upper weight required')
        if overlap and self.group is None:
            raise ValueError('Overlap requires TP2 group')
        with torch.cuda.device(self.device):
            compute = torch.cuda.current_stream()
            compute.wait_event(self.last)
            works, events = [], []
            for first in range(0, self.rows, self.width):
                last = min(self.rows, first + self.width)
                activation = self.activation[:last - first]
                target = self.reducer.input[first:last]
                fused_up_out(inputs[first:last], up_weights, activation)
                down_out(activation, down_weights, target)
                if self.group is not None:
                    if overlap:
                        ready = torch.cuda.Event()
                        ready.record(compute)
                        events.append(ready)
                        with torch.cuda.stream(self.communication):
                            self.communication.wait_event(ready)
                            if custom:
                                self.reducer.reduce(residual, gate, indices, first * 5376, (last - first) * 5376)
                            else:
                                work = dist.all_reduce(target, group=self.group, async_op=True)
                                work.wait()
                                works.append(work)
                                _gated_residual[(triton.cdiv(target.numel(), 1024),)](
                                    residual[first:last], target, gate, indices[first:last], self.output[first:last],
                                    target.numel(), 5376, gate.stride(0), BLOCK=1024,
                                    num_warps=4, enable_fp_fusion=False)
                    else:
                        dist.all_reduce(target, group=self.group)
            if overlap:
                complete = torch.cuda.Event()
                complete.record(self.communication)
                compute.wait_event(complete)
            self.last = torch.cuda.Event()
            self.last.record(compute)
        return self.output

    def run8(self, inputs, input_scales, up_weights, up_scales, down_weights, down_scales, residual, gate, indices):
        """FP8 W8A8 variant of ``run``: inputs are per-token E4M3 [S,5376] + FP32 scales; weights are
        per-channel E4M3 ([14336,5376] paired, [5376,7168]). The SwiGLU activation is re-quantized per token."""
        if inputs.shape != (self.rows, 5376) or inputs.dtype != fp8_gemm.FP8 or input_scales.shape != (self.rows,):
            raise ValueError('Expected FP8 [S,5376] input with [S] scales')
        if up_weights.shape != (14336, 5376) or down_weights.shape != (5376, 7168):
            raise ValueError('Expected FP8 [14336,5376] and [5376,7168] weights')
        if self.group is None:
            raise ValueError('TP2 group required')
        if self.activation8 is None:
            self.activation8 = torch.empty((self.width, 7168), device=self.device, dtype=fp8_gemm.FP8)
            self.activation_scales = torch.empty((self.width,), device=self.device, dtype=torch.float32)
        with torch.cuda.device(self.device):
            compute = torch.cuda.current_stream()
            compute.wait_event(self.last)
            for first in range(0, self.rows, self.width):
                last = min(self.rows, first + self.width)
                activation = self.activation[:last - first]
                activation8 = self.activation8[:last - first]
                activation_scales = self.activation_scales[:last - first]
                target = self.reducer.input[first:last]
                fp8_gemm.up8_swiglu_out(inputs[first:last], input_scales[first:last], up_weights, up_scales, activation)
                fp8_gemm.quant_rows(activation, activation8, activation_scales)
                fp8_gemm.mm8_out(activation8, activation_scales, down_weights, down_scales, target)
                ready = torch.cuda.Event()
                ready.record(compute)
                with torch.cuda.stream(self.communication):
                    self.communication.wait_event(ready)
                    self.reducer.reduce(residual, gate, indices, first * 5376, (last - first) * 5376)
            complete = torch.cuda.Event()
            complete.record(self.communication)
            compute.wait_event(complete)
            self.last = torch.cuda.Event()
            self.last.record(compute)
        return self.output

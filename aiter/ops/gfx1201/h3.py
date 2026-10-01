# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
from .ffn_pipeline import BufferedFFN
from .projection_pipeline import BufferedProjection
from .modulation import modulate
from .out_projection import project_out
from .ffn_up import fused_up_out
from .ffn_down import down_out

_WIDTH = 5376


class H3TP2Workspace:
    """Buffers and reducers for the MiniMax-H3 TP2 block tail.

    reducer="fused" (default): FusedReducer (symmetric memory, kernel-side peer reads, two barriers per chunk).
    reducer="pull": PullReducer (IPC buffers, one barrier + DMA pull of the peer partial + local combine per chunk);
    bitwise-identical output."""

    def __init__(self, rows, device, group, reducer="fused"):
        if reducer not in ("fused", "pull"):
            raise ValueError(f"reducer must be 'fused' or 'pull', got {reducer!r}")
        self.rows = rows
        self.device = torch.device(device)
        self.reducer = reducer
        self.ffn = BufferedFFN(rows, 8192, device, group, reducer=reducer)
        self.projection = BufferedProjection(rows, triton.cdiv(rows, 1024) * 128, device, group, reducer=reducer)

    def feed_forward(self, inputs, up_weights, down_weights, residual, gate, indices):
        return self.ffn.run(inputs, up_weights, down_weights, residual, gate, indices)

    def attention_output(self, inputs, weights, residual, gate, indices):
        return self.projection.run(inputs, weights, residual, gate, indices)

    def tail_chunks(self, chunk=8192):
        return [(first, min(self.rows, first + chunk)) for first in range(0, self.rows, chunk)]

    pending = None

    def take_pending(self):
        """(event, first_row) of a deferred final FFN reduce (fused_tail(defer=True)), or None. The caller must make
        its stream wait on the event before reading block-output rows >= first_row; it may read rows < first_row
        immediately."""
        pending, self.pending = self.pending, None
        return pending

    def fused_tail(self, attention, residual, out_weights, up_weights, down_weights, gate_msa, gate_mlp, indices,
                   norm_modulation, chunk=8192, defer=False):
        """Out-projection + reduce + gated residual, then norm2/modulation + FFN + reduce + gated residual, row-chunked
        so that FFN chunk i waits only for the reduce of the out-projection rows it consumes (the out-projection
        communication hides behind FFN compute). Same kernels and reduction math as
        attention_output -> norm_modulation -> feed_forward, so the result is bitwise identical.

        attention [S,3584]; residual [S,5376] (block input); out_weights [3584,5376] stride (1,3584);
        up_weights [14336,5376] contiguous (paired SwiGLU halves); down_weights [7168,5376] stride (1,7168);
        norm_modulation(x_rows, indices_rows) -> modulated BF16 rows (must be row-local, e.g. the fused
        rms_modulate). Returns the block output [S,5376] (a workspace buffer, valid until the next call).

        defer=True: the compute stream does not wait for the last chunk's FFN reduce; rows of the last chunk become
        valid only after the event returned by take_pending() (all other rows are valid in compute-stream order).
        Lets the next block start norm/QKV on the leading rows while the last reduce is still in flight."""
        projection, ffn = self.projection, self.ffn
        rows = residual.shape[0]
        if rows != self.rows or attention.shape != (rows, 3584) or residual.shape != (rows, _WIDTH):
            raise ValueError('Expected [S,3584] attention and [S,5376] residual for this workspace')
        if chunk < 1 or chunk > ffn.width:
            raise ValueError(f'chunk must be in [1, {ffn.width}]')
        if out_weights.shape != (3584, _WIDTH) or out_weights.stride() != (1, 3584):
            raise ValueError('Expected transposed [3584,5376] out-projection weight')
        if up_weights.shape != (14336, _WIDTH) or not up_weights.is_contiguous():
            raise ValueError('Expected contiguous paired [14336,5376] upper weight')
        if down_weights.shape != (7168, _WIDTH) or down_weights.stride() != (1, 7168):
            raise ValueError('Expected transposed [7168,5376] lower weight')
        if projection.group is None:
            raise ValueError('fused_tail requires the TP2 group')
        if not attention.is_contiguous() or not residual.is_contiguous():
            raise ValueError('Contiguous attention and residual required')
        bounds = self.tail_chunks(chunk)
        if self.pending is not None:
            raise RuntimeError('fused_tail: previous deferred reduce was not consumed (take_pending)')
        defer = defer and len(bounds) > 1
        reduced = []
        with torch.cuda.device(self.device):
            compute = torch.cuda.current_stream()
            compute.wait_event(projection.last)
            compute.wait_event(ffn.last)
            projected = []
            for first, last in bounds:
                project_out(attention[first:last], out_weights, projection.reducer.input[first:last])
                ready = torch.cuda.Event()
                ready.record(compute)
                with torch.cuda.stream(projection.communication):
                    projection.communication.wait_event(ready)
                    projection.reducer.reduce(residual, gate_msa, indices, first * _WIDTH, (last - first) * _WIDTH)
                    done = torch.cuda.Event()
                    done.record(projection.communication)
                    projected.append(done)
            middle = projection.output
            for (first, last), done in zip(bounds, projected):
                compute.wait_event(done)
                normed = norm_modulation(middle[first:last], indices[first:last])
                activation = ffn.activation[:last - first]
                fused_up_out(normed, up_weights, activation)
                down_out(activation, down_weights, ffn.reducer.input[first:last])
                del normed
                ready = torch.cuda.Event()
                ready.record(compute)
                with torch.cuda.stream(ffn.communication):
                    ffn.communication.wait_event(ready)
                    ffn.reducer.reduce(middle, gate_mlp, indices, first * _WIDTH, (last - first) * _WIDTH)
                    if defer:
                        done = torch.cuda.Event()
                        done.record(ffn.communication)
                        reduced.append(done)
            if defer:
                complete = torch.cuda.Event()
                complete.record(projection.communication)
                compute.wait_event(complete)
                compute.wait_event(reduced[-2])
                projection.last = torch.cuda.Event()
                projection.last.record(compute)
                ffn.last = reduced[-1]
                self.pending = (reduced[-1], bounds[-1][0])
                return ffn.output
            for stream in (projection.communication, ffn.communication):
                complete = torch.cuda.Event()
                complete.record(stream)
                compute.wait_event(complete)
            projection.last = torch.cuda.Event()
            projection.last.record(compute)
            ffn.last = projection.last
        return ffn.output

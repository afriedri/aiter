# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import ctypes
import math

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm
import triton
import triton.language as tl


@triton.jit
def _sum_half(local, peer, scratch, first, count, RANK: tl.constexpr, BLOCK: tl.constexpr):
    half = (count + 1) // 2
    begin = RANK * half
    length = tl.minimum(half, count - begin)
    for tile in range(tl.program_id(0), tl.cdiv(length, BLOCK), tl.num_programs(0)):
        position = tile * BLOCK + tl.arange(0, BLOCK)
        offsets = first + begin + position
        left = tl.load(local + offsets, position < length, other=0).to(tl.float32)
        right = tl.load(peer + offsets, position < length, other=0).to(tl.float32)
        total = (left + right).to(tl.bfloat16, fp_downcast_rounding='rtne')
        tl.store(scratch + offsets, total, position < length)


@triton.jit
def _gather_halves(scratch, peer, output, residual, gate, indices, dimension: tl.constexpr, gate_stride: tl.constexpr, first, count, RANK: tl.constexpr, BLOCK: tl.constexpr):
    half = (count + 1) // 2
    for tile in range(tl.program_id(0), tl.cdiv(count, BLOCK), tl.num_programs(0)):
        position = tile * BLOCK + tl.arange(0, BLOCK)
        local_half = (position < half) if RANK == 0 else (position >= half)
        offsets = first + position
        local_value = tl.load(scratch + offsets, position < count, other=0).to(tl.float32)
        peer_value = tl.load(peer + offsets, position < count, other=0).to(tl.float32)
        valid = position < count
        reduced = (local_value + peer_value).to(tl.bfloat16, fp_downcast_rounding='rtne').to(tl.float32)
        group = tl.load(indices + offsets // dimension, valid, other=0)
        scale = tl.load(gate + group * gate_stride + offsets % dimension, valid, other=0).to(tl.float32)
        product = (reduced * scale).to(tl.bfloat16).to(tl.float32)
        original = tl.load(residual + offsets, valid, other=0).to(tl.float32)
        tl.store(output + offsets, original + product, valid)


class FusedReducer:
    def __init__(self, shape, group, blocks=8):
        if dist.get_world_size(group) != 2 or blocks < 1:
            raise ValueError('TP2 and positive block count required')
        self.rank = dist.get_rank(group)
        self.blocks = blocks
        self.input = symm.empty(shape, device=torch.device('cuda', torch.cuda.current_device()), dtype=torch.bfloat16)
        self.output = torch.empty_like(self.input)
        self.input_handle = symm.rendezvous(self.input, group)
        self.peer_input = self.input_handle.get_buffer(1 - self.rank, tuple(shape), torch.bfloat16)

    def reduce(self, residual, gate, indices, first=0, count=None):
        if count is None:
            count = self.input.numel() - first
        if first < 0 or count < 1 or first + count > self.input.numel():
            raise ValueError('Invalid flat reduction range')
        with torch.cuda.device(self.input.device):
            self.input_handle.barrier(timeout_ms=30000)
            _gather_halves[(32,)](self.input, self.peer_input, self.output, residual, gate, indices, residual.shape[1], gate.stride(0),
                                           first, count, self.rank, BLOCK=4096, num_warps=4, enable_fp_fusion=False)
            self.input_handle.barrier(timeout_ms=30000)
        return self.output


@triton.jit
def _pull_combine(local, staged, output, residual, gate, indices, dimension: tl.constexpr, gate_stride: tl.constexpr,
                  first, count, BLOCK: tl.constexpr):
    # _gather_halves' math with the peer partial read from a local staging copy staged[0:count].
    position = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = position < count
    offsets = first + position
    local_value = tl.load(local + offsets, valid, other=0).to(tl.float32)
    peer_value = tl.load(staged + position, valid, other=0).to(tl.float32)
    reduced = (local_value + peer_value).to(tl.bfloat16, fp_downcast_rounding='rtne').to(tl.float32)
    group = tl.load(indices + offsets // dimension, valid, other=0)
    scale = tl.load(gate + group * gate_stride + offsets % dimension, valid, other=0).to(tl.float32)
    product = (reduced * scale).to(tl.bfloat16).to(tl.float32)
    original = tl.load(residual + offsets, valid, other=0).to(tl.float32)
    tl.store(output + offsets, original + product, valid)


class _IpcHandle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_char * 64)]


_hip = None


def _hip_runtime():
    """The libamdhip64 torch already loaded (a second copy would own a separate device context)."""
    global _hip
    if _hip is None:
        path = "libamdhip64.so"
        with open("/proc/self/maps") as maps:
            for line in maps:
                if "libamdhip64.so" in line:
                    path = line.split()[-1]
                    break
        lib = ctypes.CDLL(path)
        lib.hipMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        lib.hipFree.argtypes = [ctypes.c_void_p]
        lib.hipIpcGetMemHandle.argtypes = [ctypes.POINTER(_IpcHandle), ctypes.c_void_p]
        lib.hipIpcOpenMemHandle.argtypes = [ctypes.POINTER(ctypes.c_void_p), _IpcHandle, ctypes.c_uint]
        lib.hipIpcCloseMemHandle.argtypes = [ctypes.c_void_p]
        lib.hipGetErrorString.restype = ctypes.c_char_p
        _hip = lib
    return _hip


def _hip_check(status, what):
    if status:
        raise RuntimeError(f"{what} failed: {status} {_hip_runtime().hipGetErrorString(status).decode()}")


def _bf16_view(pointer, shape, device):
    numel = math.prod(shape)
    storage = torch._C._construct_storage_from_data_pointer(pointer, device, numel * 2)
    tensor = torch.empty(0, dtype=torch.bfloat16, device=device)
    tensor.set_(storage, 0, tuple(shape))
    return tensor


class PullReducer:
    """TP2 all-reduce + gated residual with the FusedReducer interface and bitwise-identical output.

    FusedReducer: barrier, 32 workgroups read the peer partial over PCIe (symmetric-memory mapping) and combine,
    barrier. PullReducer: one barrier, a DMA copy of the peer's rows into a local staging buffer, then the same
    combine as one local kernel over cdiv(count, 4096) workgroups. On R9600D (no fast peer path) the DMA pull
    is several times faster than kernel-side peer reads and frees the CUs for the GEMMs it overlaps.

    The partials live in plain hipMalloc buffers shared with hipIpcGetMemHandle / hipIpcOpenMemHandle (the ROCr
    copy path handles those like any KFD allocation; copies from symmetric-memory peer mappings fault when the
    allocator uses expandable segments). Symmetric memory only provides the barrier (a 1 KiB buffer).

    trailing_barrier=False drops FusedReducer's second barrier ("the peer finished reading my partial"). That is
    safe when, as in H3TP2Workspace, two reducers alternate on streams ordered through the compute stream: the
    next write of a chunk of this reducer's input happens only after the compute stream waited for the other
    reducer's barrier, which the peer reaches only after its compute stream waited for this reducer's copy.
    Use trailing_barrier=True for a reducer that is reused back to back on its own."""

    def __init__(self, shape, group, blocks=8, stage_rows=None, trailing_barrier=False):
        if dist.get_world_size(group) != 2 or blocks < 1:
            raise ValueError('TP2 and positive block count required')
        self.rank = dist.get_rank(group)
        self.blocks = blocks
        self.trailing_barrier = trailing_barrier
        device = torch.device('cuda', torch.cuda.current_device())
        runtime = _hip_runtime()
        shape = tuple(shape)
        with torch.cuda.device(device):
            self.signal = symm.empty((512,), device=device, dtype=torch.bfloat16)
            self.input_handle = symm.rendezvous(self.signal, group)
            pointer = ctypes.c_void_p()
            _hip_check(runtime.hipMalloc(ctypes.byref(pointer), math.prod(shape) * 2), "hipMalloc")
            handle = _IpcHandle()
            _hip_check(runtime.hipIpcGetMemHandle(ctypes.byref(handle), pointer), "hipIpcGetMemHandle")
            handles = [None, None]
            dist.all_gather_object(handles, ctypes.string_at(ctypes.addressof(handle), 64), group=group)
            peer = ctypes.c_void_p()
            _hip_check(runtime.hipIpcOpenMemHandle(ctypes.byref(peer), _IpcHandle.from_buffer_copy(handles[1 - self.rank]),
                                                   1), "hipIpcOpenMemHandle")  # 1 = hipIpcMemLazyEnablePeerAccess
            self._pointers = (pointer.value, peer.value)
            self.input = _bf16_view(pointer.value, shape, device)
            self.peer_input = _bf16_view(peer.value, shape, device)
            self.output = torch.empty_like(self.input)
            rows = shape[0] if stage_rows is None else max(1, min(shape[0], stage_rows))
            self.stage = torch.empty(rows * math.prod(shape[1:]), device=device, dtype=torch.bfloat16)
            torch.cuda.synchronize(device)
            dist.barrier(group=group)

    def reduce(self, residual, gate, indices, first=0, count=None):
        if count is None:
            count = self.input.numel() - first
        if first < 0 or count < 1 or first + count > self.input.numel():
            raise ValueError('Invalid flat reduction range')
        with torch.cuda.device(self.input.device):
            if self.stage.numel() < count:
                self.stage = torch.empty(count, device=self.input.device, dtype=torch.bfloat16)
            self.input_handle.barrier(timeout_ms=30000)
            self.stage[:count].copy_(self.peer_input.view(-1)[first:first + count])
            _pull_combine[(triton.cdiv(count, 4096),)](self.input, self.stage, self.output, residual, gate, indices,
                                                       residual.shape[1], gate.stride(0), first, count, BLOCK=4096,
                                                       num_warps=8, enable_fp_fusion=False)
            if self.trailing_barrier:
                self.input_handle.barrier(timeout_ms=30000)
        return self.output

    def close(self):
        """Release the IPC mapping and the partial buffer (the tensors of this reducer become invalid)."""
        if self._pointers is None:
            return
        torch.cuda.synchronize(self.input.device)
        mine, peer = self._pointers
        self._pointers = None
        _hip_check(_hip_runtime().hipIpcCloseMemHandle(ctypes.c_void_p(peer)), "hipIpcCloseMemHandle")
        _hip_check(_hip_runtime().hipFree(ctypes.c_void_p(mine)), "hipFree")


def make_reducer(kind, shape, group, stage_rows=None):
    """'fused' -> FusedReducer (symmetric memory, kernel-side peer reads); 'pull' -> PullReducer (IPC + DMA pull)."""
    if kind == "fused":
        return FusedReducer(shape, group)
    if kind == "pull":
        return PullReducer(shape, group, stage_rows=stage_rows)
    raise ValueError(f"reducer must be 'fused' or 'pull', got {kind!r}")

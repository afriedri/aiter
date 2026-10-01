# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import ctypes
import math
import os

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
                  first, count, BLOCK: tl.constexpr, VOLATILE: tl.constexpr = False):
    # _gather_halves' math with the peer partial read from a local staging copy staged[0:count].
    # VOLATILE: the stage was written by the peer's DMA engine (push), so read it past the caches.
    position = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = position < count
    offsets = first + position
    local_value = tl.load(local + offsets, valid, other=0).to(tl.float32)
    peer_value = tl.load(staged + position, valid, other=0, volatile=VOLATILE).to(tl.float32)
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


def _bf16_view(pointer, shape, device, dtype=torch.bfloat16):
    numel = math.prod(shape)
    storage = torch._C._construct_storage_from_data_pointer(pointer, device, numel * dtype.itemsize)
    tensor = torch.empty(0, dtype=dtype, device=device)
    tensor.set_(storage, 0, tuple(shape))
    return tensor


@triton.jit(do_not_specialize=["value"])
def _signal_peer(flag, value):
    # System-scope release: the L2 write-back makes everything this GPU wrote before (in stream order) visible to
    # the peer's DMA engine before the peer can observe the flag.
    tl.atomic_xchg(flag, value, sem='release', scope='sys')


_announced = False


class _StreamBarrier:
    """TP2 barrier without a spinning kernel: a 1-thread kernel stores an epoch into the peer's flag (system-scope
    release), then hipStreamWaitValue32 makes the stream (command processor, no CU) wait until the own flag reaches
    the epoch. Same ordering guarantees as the symmetric-memory barrier_kernel, but the waiting side occupies no
    CU slot and issues no atomics while the peer is late (the c10d kernel slows a concurrent GEMM by ~5%)."""

    def __init__(self, group, device):
        runtime = _hip_runtime()
        runtime.hipExtMallocWithFlags.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint]
        runtime.hipMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
        runtime.hipStreamWaitValue32.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint,
                                                 ctypes.c_uint32]
        self.rank = dist.get_rank(group)
        mine = ctypes.c_void_p()
        _hip_check(runtime.hipExtMallocWithFlags(ctypes.byref(mine), 4096, 1), "hipExtMallocWithFlags")  # fine-grained
        _hip_check(runtime.hipMemset(mine, 0, 4096), "hipMemset")
        handle = _IpcHandle()
        _hip_check(runtime.hipIpcGetMemHandle(ctypes.byref(handle), mine), "hipIpcGetMemHandle")
        handles = [None, None]
        dist.all_gather_object(handles, ctypes.string_at(ctypes.addressof(handle), 64), group=group)
        peer = ctypes.c_void_p()
        _hip_check(runtime.hipIpcOpenMemHandle(ctypes.byref(peer), _IpcHandle.from_buffer_copy(handles[1 - self.rank]),
                                               1), "hipIpcOpenMemHandle")
        self._pointers = (mine.value, peer.value)
        self.peer_flag = _bf16_view(peer.value, (1,), device, torch.int32)
        self.epoch = 0
        global _announced
        if not _announced:
            _announced = True
            print(f"[aiter] AITER_GFX1201_CP_BARRIER active (rank {self.rank})", flush=True)

    def __call__(self):
        self.epoch += 1
        _signal_peer[(1,)](self.peer_flag, self.epoch, num_warps=1)
        _hip_check(_hip_runtime().hipStreamWaitValue32(ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
                                                       ctypes.c_void_p(self._pointers[0]), self.epoch, 0, 0xFFFFFFFF),
                   "hipStreamWaitValue32")  # 0 = hipStreamWaitValueGte

    def close(self):
        if self._pointers is None:
            return
        mine, peer = self._pointers
        self._pointers = None
        _hip_check(_hip_runtime().hipIpcCloseMemHandle(ctypes.c_void_p(peer)), "hipIpcCloseMemHandle")
        _hip_check(_hip_runtime().hipFree(ctypes.c_void_p(mine)), "hipFree")


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
            self.barrier = lambda: self.input_handle.barrier(timeout_ms=30000)
            self.stream_barrier = None
            if os.environ.get("AITER_GFX1201_CP_BARRIER", "0") not in ("", "0"):
                self.stream_barrier = _StreamBarrier(group, device)
                self.barrier = self.stream_barrier
            self.push = os.environ.get("AITER_GFX1201_PUSH_REDUCE", "0") not in ("", "0")
            self._stage_pointers = None
            if self.push:
                # Two IPC-shared stage slots per rank; the peer's DMA engine writes its partial into slot (call & 1).
                # Slot reuse is safe: my push of call n+2 is stream-ordered after my barrier of call n+1, which the
                # peer passes only after its combine of call n (push -> barrier -> combine on one stream per reducer).
                slot = self.stage.numel()
                del self.stage
                stage_pointer = ctypes.c_void_p()
                _hip_check(runtime.hipMalloc(ctypes.byref(stage_pointer), 2 * slot * 2), "hipMalloc")
                _hip_check(runtime.hipIpcGetMemHandle(ctypes.byref(handle), stage_pointer), "hipIpcGetMemHandle")
                dist.all_gather_object(handles, ctypes.string_at(ctypes.addressof(handle), 64), group=group)
                peer_stage = ctypes.c_void_p()
                _hip_check(runtime.hipIpcOpenMemHandle(ctypes.byref(peer_stage),
                                                       _IpcHandle.from_buffer_copy(handles[1 - self.rank]), 1),
                           "hipIpcOpenMemHandle")
                self._stage_pointers = (stage_pointer.value, peer_stage.value)
                mine_stage = _bf16_view(stage_pointer.value, (2, slot), device)
                theirs = _bf16_view(peer_stage.value, (2, slot), device)
                self.stage_slots = (mine_stage[0], mine_stage[1])
                self.peer_slots = (theirs[0], theirs[1])
                self.calls = 0
            torch.cuda.synchronize(device)
            dist.barrier(group=group)

    def reduce(self, residual, gate, indices, first=0, count=None):
        if count is None:
            count = self.input.numel() - first
        if first < 0 or count < 1 or first + count > self.input.numel():
            raise ValueError('Invalid flat reduction range')
        if self.push:
            return self._push_reduce(residual, gate, indices, first, count)
        with torch.cuda.device(self.input.device):
            if self.stage.numel() < count:
                self.stage = torch.empty(count, device=self.input.device, dtype=torch.bfloat16)
            self.barrier()
            self.stage[:count].copy_(self.peer_input.view(-1)[first:first + count])
            _pull_combine[(triton.cdiv(count, 4096),)](self.input, self.stage, self.output, residual, gate, indices,
                                                       residual.shape[1], gate.stride(0), first, count, BLOCK=4096,
                                                       num_warps=8, enable_fp_fusion=False)
            if self.trailing_barrier:
                self.barrier()
        return self.output

    def _push_reduce(self, residual, gate, indices, first, count):
        """AITER_GFX1201_PUSH_REDUCE=1: DMA-push my partial into the peer's stage slot, barrier, combine with the
        peer's partial from my own slot. Same combine math -> bitwise equal to the pull path. Push is ~15% faster
        than pull over this PCIe topology and slows the overlapped GEMMs less (microbench: +2.5% vs +5%)."""
        if count > self.stage_slots[0].numel():
            raise ValueError('push reduce: chunk larger than the stage slot')
        slot = self.calls & 1
        self.calls += 1
        with torch.cuda.device(self.input.device):
            self.peer_slots[slot][:count].copy_(self.input.view(-1)[first:first + count])
            self.barrier()
            _pull_combine[(triton.cdiv(count, 4096),)](self.input, self.stage_slots[slot], self.output, residual, gate,
                                                       indices, residual.shape[1], gate.stride(0), first, count,
                                                       BLOCK=4096, VOLATILE=True, num_warps=8, enable_fp_fusion=False)
            if self.trailing_barrier:
                self.barrier()
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
        if self.stream_barrier is not None:
            self.stream_barrier.close()
        if self._stage_pointers is not None:
            mine, peer = self._stage_pointers
            self._stage_pointers = None
            _hip_check(_hip_runtime().hipIpcCloseMemHandle(ctypes.c_void_p(peer)), "hipIpcCloseMemHandle")
            _hip_check(_hip_runtime().hipFree(ctypes.c_void_p(mine)), "hipFree")


def make_reducer(kind, shape, group, stage_rows=None):
    """'fused' -> FusedReducer (symmetric memory, kernel-side peer reads); 'pull' -> PullReducer (IPC + DMA pull)."""
    if kind == "fused":
        return FusedReducer(shape, group)
    if kind == "pull":
        return PullReducer(shape, group, stage_rows=stage_rows)
    raise ValueError(f"reducer must be 'fused' or 'pull', got {kind!r}")

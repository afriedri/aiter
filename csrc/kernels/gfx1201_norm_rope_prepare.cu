// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <stdint.h>

static constexpr int HEADS = 28;
static constexpr int DIM = 128;
static constexpr int BLOCK_ROWS = 32;
static constexpr int ROPE_DIM = 96;
static constexpr int THREADS = 256;
static constexpr int COLUMNS = HEADS * DIM;
static constexpr float NORM_EPS = 1e-5f;

__device__ __forceinline__ float bf16_bits_to_float(uint16_t bits) {
  return __uint_as_float(((uint32_t)bits) << 16);
}

__device__ __forceinline__ float round_bf16(float value) {
  return __bfloat162float(__float2bfloat16(value));
}

__device__ __forceinline__ void load16(const uint16_t* source, bool valid, float out[16]) {
  uint4 a = make_uint4(0, 0, 0, 0), b = make_uint4(0, 0, 0, 0);
  if (valid) {
    a = *reinterpret_cast<const uint4*>(source);
    b = *reinterpret_cast<const uint4*>(source + 8);
  }
  const uint32_t words[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    out[2 * i] = bf16_bits_to_float(words[i] & 0xffff);
    out[2 * i + 1] = bf16_bits_to_float(words[i] >> 16);
  }
}

// 448 threads x 8 columns = one full 3584-column row per iteration: 16 B loads, fully coalesced.
static constexpr int ABSMAX_THREADS = COLUMNS / 8;  // 448
extern "C" __global__ __launch_bounds__(ABSMAX_THREADS)
void v_absmax2_kernel(const uint16_t* __restrict__ value, uint32_t* __restrict__ maximum, int rows, int rows_per_block) {
  const int column = threadIdx.x * 8;
  const int begin = blockIdx.x * rows_per_block;
  const int end = min(rows, begin + rows_per_block);
  float best[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  const uint16_t* p = value + (size_t)begin * COLUMNS + column;
  int row = begin;
  for (; row + 4 <= end; row += 4, p += 4 * COLUMNS) {
    uint4 w[4];
    #pragma unroll
    for (int r = 0; r < 4; ++r) w[r] = *reinterpret_cast<const uint4*>(p + (size_t)r * COLUMNS);
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
      const uint32_t u[4] = {w[r].x, w[r].y, w[r].z, w[r].w};
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        best[2 * i] = fmaxf(best[2 * i], fabsf(bf16_bits_to_float(u[i] & 0xffff)));
        best[2 * i + 1] = fmaxf(best[2 * i + 1], fabsf(bf16_bits_to_float(u[i] >> 16)));
      }
    }
  }
  for (; row < end; ++row, p += COLUMNS) {
    const uint4 w = *reinterpret_cast<const uint4*>(p);
    const uint32_t u[4] = {w.x, w.y, w.z, w.w};
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      best[2 * i] = fmaxf(best[2 * i], fabsf(bf16_bits_to_float(u[i] & 0xffff)));
      best[2 * i + 1] = fmaxf(best[2 * i + 1], fabsf(bf16_bits_to_float(u[i] >> 16)));
    }
  }
  if (begin < end) {
    #pragma unroll
    for (int i = 0; i < 8; ++i) atomicMax(maximum + column + i, __float_as_uint(best[i]));
  }
}

// ---- v2 main pass: Q/K CTAs own one 32-row block and loop all heads (cos/sin/weights loaded once) ----
extern "C" __global__ __launch_bounds__(THREADS)
void rope_qk_prepare2_kernel(
    const uint16_t* __restrict__ query, const uint16_t* __restrict__ key,
    const float* __restrict__ cosine, const float* __restrict__ sine,
    const uint16_t* __restrict__ query_weight, const uint16_t* __restrict__ key_weight,
    int8_t* __restrict__ query_out, int8_t* __restrict__ key_out,
    float* __restrict__ query_scale, float* __restrict__ key_scale,
    int rows, int padded_rows, float sm_scale) {
  __shared__ float shared[2][THREADS / 32];
  const int block = blockIdx.x;
  const int blocks = padded_rows / BLOCK_ROWS;
  const int tid = threadIdx.x;
  const bool is_query = blockIdx.y == 0;
  const uint16_t* source = is_query ? query : key;
  int8_t* out = is_query ? query_out : key_out;
  float* scales = is_query ? query_scale : key_scale;
  const uint16_t* weight = is_query ? query_weight : key_weight;
  const int group = tid % 8;
  const int row = block * BLOCK_ROWS + tid / 8;
  const bool valid = row < rows;
  const int partner_group = group < 3 ? group + 3 : group - 3;
  float w_own[16], w_partner[16], c[16], sn[16];
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    w_own[i] = bf16_bits_to_float(weight[group * 16 + i]);
    w_partner[i] = bf16_bits_to_float(weight[partner_group * 16 + i]);
    c[i] = (valid && group < 6) ? cosine[(size_t)row * ROPE_DIM + group * 16 + i] : 0.f;
    sn[i] = (valid && group < 6) ? sine[(size_t)row * ROPE_DIM + group * 16 + i] : 0.f;
  }
  const size_t row_base = (size_t)row * HEADS * DIM;
  uint4 na0 = make_uint4(0, 0, 0, 0), na1 = na0, nb0 = na0, nb1 = na0;
  auto fetch = [&](int head) {
    const uint16_t* p = source + row_base + (size_t)head * DIM;
    na0 = na1 = nb0 = nb1 = make_uint4(0, 0, 0, 0);
    if (valid) {
      na0 = *reinterpret_cast<const uint4*>(p + group * 16);
      na1 = *reinterpret_cast<const uint4*>(p + group * 16 + 8);
      if (group < 6) {
        nb0 = *reinterpret_cast<const uint4*>(p + partner_group * 16);
        nb1 = *reinterpret_cast<const uint4*>(p + partner_group * 16 + 8);
      }
    }
  };
  fetch(0);
  for (int head = 0; head < HEADS; ++head) {
    const uint32_t wa[8] = {na0.x, na0.y, na0.z, na0.w, na1.x, na1.y, na1.z, na1.w};
    const uint32_t wb[8] = {nb0.x, nb0.y, nb0.z, nb0.w, nb1.x, nb1.y, nb1.z, nb1.w};
    if (head + 1 < HEADS) fetch(head + 1);
    float x[16], partner[16];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      x[2 * i] = bf16_bits_to_float(wa[i] & 0xffff);
      x[2 * i + 1] = bf16_bits_to_float(wa[i] >> 16);
      partner[2 * i] = bf16_bits_to_float(wb[i] & 0xffff);
      partner[2 * i + 1] = bf16_bits_to_float(wb[i] >> 16);
    }
    // Reproduces ATen vectorized RMSNorm: per-4 fma chains, then shfl_down tree 16/8/4/2/1.
    float sums[4];
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      float acc = 0.f;
      #pragma unroll
      for (int i = 0; i < 4; ++i) acc = __fmaf_rn(x[4 * j + i], x[4 * j + i], acc);
      sums[j] = acc;
    }
    #pragma unroll
    for (int mask = 4; mask > 0; mask >>= 1)
      #pragma unroll
      for (int j = 0; j < 4; ++j) sums[j] = __fadd_rn(sums[j], __shfl_xor(sums[j], mask, 32));
    const float total = __fadd_rn(__fadd_rn(sums[0], sums[2]), __fadd_rn(sums[1], sums[3]));
    const float rstd = rsqrtf(__fadd_rn(__fdiv_rn(total, 128.0f), NORM_EPS));
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
      x[i] = round_bf16(__fmul_rn(__fmul_rn(rstd, x[i]), w_own[i]));
      partner[i] = round_bf16(__fmul_rn(__fmul_rn(rstd, partner[i]), w_partner[i]));
    }
    if (group < 6) {
      #pragma unroll
      for (int i = 0; i < 16; ++i) {
        const float rotated = group < 3 ? -partner[i] : partner[i];
        x[i] = __fadd_rn(__fmul_rn(x[i], c[i]), __fmul_rn(rotated, sn[i]));
      }
    }
    float local = 0.f;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
      x[i] = round_bf16(x[i]);
      if (is_query) x[i] = __fmul_rn(x[i], sm_scale);
      local = fmaxf(local, fabsf(x[i]));
    }
    // block max over the 32x128 tile (double-buffered partials: one barrier per head)
    float value = local;
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) value = fmaxf(value, __shfl_xor(value, mask, 32));
    float* sh = shared[head & 1];
    if ((tid & 31) == 0) sh[tid / 32] = value;
    __syncthreads();
    value = sh[0];
    #pragma unroll
    for (int i = 1; i < THREADS / 32; ++i) value = fmaxf(value, sh[i]);
    const float scale = __fdiv_rn(value, 127.0f);
    if (tid == 0) scales[head * blocks + block] = scale;
    uint32_t packed[4];
    #pragma unroll
    for (int word = 0; word < 4; ++word) {
      uint32_t bits = 0;
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        float q = __fdiv_rn(x[word * 4 + i], scale);
        q = __fadd_rn(q, q >= 0.f ? 0.5f : -0.5f);
        const int8_t v = valid ? (int8_t)__float2int_rz(q) : (int8_t)0;
        bits |= ((uint32_t)(uint8_t)v) << (8 * i);
      }
      packed[word] = bits;
    }
    *reinterpret_cast<uint4*>(out + row_base + (size_t)head * DIM + group * 16) =
        make_uint4(packed[0], packed[1], packed[2], packed[3]);
  }
}

// ---- v2 V pass: 128 tokens x 7 heads per CTA; quantize 8x8 register tiles, transpose bytes through LDS ----
static constexpr int V_TOKENS = 128;
static constexpr int V_HEAD_GROUPS = 4;
static constexpr int V_HEADS_PER_CTA = HEADS / V_HEAD_GROUPS;  // 7
static constexpr int V_PITCH = V_TOKENS + 16;
extern "C" __global__ __launch_bounds__(THREADS)
void v_prepare2_kernel(const uint16_t* __restrict__ value, const uint32_t* __restrict__ v_maximum,
                       uint8_t* __restrict__ value_out, float* __restrict__ value_scale, int rows, int padded_rows) {
  __shared__ __attribute__((aligned(16))) uint8_t tile[DIM * V_PITCH];
  const int tblock = blockIdx.x / V_HEAD_GROUPS;
  const int hgroup = blockIdx.x % V_HEAD_GROUPS;
  const int t0 = tblock * V_TOKENS;
  if (t0 >= padded_rows) return;
  const int tid = threadIdx.x;
  const int dchunk = tid % 16;          // 8 d values
  const int tg = tid / 16;              // 8 tokens
  const int rd = tid / 2, rhalf = tid % 2;  // readout: one d row, 64 tokens
  for (int hh = 0; hh < V_HEADS_PER_CTA; ++hh) {
    const int head = hgroup * V_HEADS_PER_CTA + hh;
    float scale[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) scale[j] = __fmul_rn(__uint_as_float(v_maximum[head * DIM + dchunk * 8 + j]), 1.0f / 448.0f);
    if (tblock == 0 && tg == 0) {
      #pragma unroll
      for (int j = 0; j < 8; ++j) value_scale[head * DIM + dchunk * 8 + j] = scale[j];
    }
    float x[8][8];  // [token][d]
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
      const int token = t0 + tg * 8 + k;
      uint4 w = make_uint4(0, 0, 0, 0);
      if (token < rows) w = *reinterpret_cast<const uint4*>(value + ((size_t)token * HEADS + head) * DIM + dchunk * 8);
      const uint32_t u[4] = {w.x, w.y, w.z, w.w};
      #pragma unroll
      for (int j = 0; j < 4; ++j) {
        x[k][2 * j] = token < rows ? __fdiv_rn(bf16_bits_to_float(u[j] & 0xffff), scale[2 * j]) : 0.f;
        x[k][2 * j + 1] = token < rows ? __fdiv_rn(bf16_bits_to_float(u[j] >> 16), scale[2 * j + 1]) : 0.f;
      }
    }
    if (hh > 0) __syncthreads();  // previous head's readout done
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
      int lo = 0, hi = 0;
      lo = __builtin_amdgcn_cvt_pk_fp8_f32(x[0][j], x[1][j], lo, false);
      lo = __builtin_amdgcn_cvt_pk_fp8_f32(x[2][j], x[3][j], lo, true);
      hi = __builtin_amdgcn_cvt_pk_fp8_f32(x[4][j], x[5][j], hi, false);
      hi = __builtin_amdgcn_cvt_pk_fp8_f32(x[6][j], x[7][j], hi, true);
      *reinterpret_cast<uint2*>(tile + (dchunk * 8 + j) * V_PITCH + tg * 8) = make_uint2((uint32_t)lo, (uint32_t)hi);
    }
    __syncthreads();
    uint8_t* dst = value_out + ((size_t)head * DIM + rd) * padded_rows + t0 + rhalf * 64;
    const uint8_t* src = tile + rd * V_PITCH + rhalf * 64;
    #pragma unroll
    for (int c = 0; c < 4; ++c)
      if (t0 + rhalf * 64 + c * 16 < padded_rows)
        *reinterpret_cast<uint4*>(dst + c * 16) = *reinterpret_cast<const uint4*>(src + c * 16);
  }
}

void launch_gfx1201_norm_rope_prepare(int64_t query, int64_t key, int64_t value, int64_t cosine, int64_t sine, int64_t maximum,
                    int64_t query_weight, int64_t key_weight, int64_t query_out, int64_t key_out, int64_t value_out,
                    int64_t query_scale, int64_t key_scale, int64_t value_scale,
                    int64_t rows, int64_t padded_rows, double sm_scale, int64_t stream, int64_t parts) {
  hipStream_t s = (hipStream_t)stream;
  hipMemsetAsync((void*)maximum, 0, COLUMNS * sizeof(uint32_t), s);
  const int chunks = 240;
  const int rows_per_block = (int)((rows + chunks - 1) / chunks);
  hipLaunchKernelGGL(v_absmax2_kernel, dim3(chunks), dim3(ABSMAX_THREADS), 0, s,
                     (const uint16_t*)value, (uint32_t*)maximum, (int)rows, rows_per_block);
  // parts: 1 = Q, 2 = Q+K, 3 = Q+K+V (same meaning as the former grid.y)
  if (parts > 0)
    hipLaunchKernelGGL(rope_qk_prepare2_kernel, dim3((unsigned)(padded_rows / BLOCK_ROWS), (unsigned)(parts < 2 ? parts : 2)), dim3(THREADS), 0, s,
                       (const uint16_t*)query, (const uint16_t*)key, (const float*)cosine, (const float*)sine,
                       (const uint16_t*)query_weight, (const uint16_t*)key_weight, (int8_t*)query_out, (int8_t*)key_out,
                       (float*)query_scale, (float*)key_scale, (int)rows, (int)padded_rows, (float)sm_scale);
  if (parts > 2)
    hipLaunchKernelGGL(v_prepare2_kernel, dim3((unsigned)((padded_rows + V_TOKENS - 1) / V_TOKENS * V_HEAD_GROUPS)), dim3(THREADS), 0, s,
                       (const uint16_t*)value, (const uint32_t*)maximum, (uint8_t*)value_out, (float*)value_scale,
                       (int)rows, (int)padded_rows);
}

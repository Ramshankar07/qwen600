
// qwen_model.hip.h - AMD ROCm/HIP version

#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
#include <hipblas/hipblas.h>
#include <hip_bfloat16.h>

#include "config.h"
#include "static_loader.hip.h"

// ================================================================
// globals
// ================================================================

#define EXIT_SUCCESS 0
constexpr int THREADS_PER_BLOCK = 256;

using bf16 = __hip_bfloat16;
using TransformerWeights = qwen_loader::QwenWeights;

// ================================================================
// transformer model
// ================================================================

typedef struct
{
    bf16 *x;       // activation at current time stamp (DIM,)
    bf16 *xb;      // buffer for residual branch (DIM,)
    bf16 *xb2;     // an additional buffer (DIM,)
    bf16 *hb;      // buffer for hidden dimension in ffn (HIDDEN_DIM,)
    bf16 *hb2;     // buffer for hidden dimension in ffn (HIDDEN_DIM,)
    bf16 *q;       // query buffer (Q_DIM,) - NOTE: This is larger than DIM now

    float *att;    // buffer for scores/attention values (N_HEADS, SEQ_LEN)
    bf16 *logits;  // output logits on the GPU (VOCAB_SIZE,)

    // kv cache
    bf16* key_cache;   // (N_LAYERS, SEQ_LEN, KV_DIM)
    bf16* value_cache; // (N_LAYERS, SEQ_LEN, KV_DIM)
    
    // buffer for final logits converted to fp32 on the GPU
    float* d_logits_fp32;
} RunState;

typedef struct
{
    TransformerWeights weights;
    RunState state;

    hipblasHandle_t hipblas_handle;

    // host-side buffer to copy the final logits back for sampling
    float* h_logits;
} Transformer;


void
malloc_run_state(RunState* s)
{
    hipMalloc(&s->x, DIM * sizeof(bf16));
    hipMalloc(&s->xb, DIM * sizeof(bf16));
    hipMalloc(&s->xb2, DIM * sizeof(bf16));
    hipMalloc(&s->hb, HIDDEN_DIM * sizeof(bf16));
    hipMalloc(&s->hb2, HIDDEN_DIM * sizeof(bf16));
    // query buffer must be Q_DIM, which is N_HEADS * HEAD_DIM = 2048 for this model.
    hipMalloc(&s->q, Q_DIM * sizeof(bf16));
    
    hipMalloc(&s->att, (size_t)N_HEADS * SEQ_LEN * sizeof(float));
    hipMalloc(&s->logits, VOCAB_SIZE * sizeof(bf16));
    hipMalloc(&s->key_cache, (size_t)N_LAYERS * SEQ_LEN * KV_DIM * sizeof(bf16));
    hipMalloc(&s->value_cache, (size_t)N_LAYERS * SEQ_LEN * KV_DIM * sizeof(bf16));

    hipMalloc(&s->d_logits_fp32, VOCAB_SIZE * sizeof(float));
}

void
build_transformer(
    Transformer* t,
    const char* checkpoint_path)
{
    qwen_loader::load_qwen_weights(checkpoint_path, t->weights);
    malloc_run_state(&t->state);
    hipHostMalloc((void**)&t->h_logits, VOCAB_SIZE * sizeof(float));

    hipblasCreate(&t->hipblas_handle);
}

void
free_transformer(Transformer* t)
{
    hipFree(t->state.x);
    hipFree(t->state.xb);
    hipFree(t->state.xb2);
    hipFree(t->state.hb);
    hipFree(t->state.hb2);
    hipFree(t->state.q);
    hipFree(t->state.att);
    hipFree(t->state.logits);
    hipFree(t->state.key_cache);
    hipFree(t->state.value_cache);
    hipFree(t->state.d_logits_fp32);

    hipHostFree(t->h_logits);
    hipblasDestroy(t->hipblas_handle);
}

// ================================================================
// HIP OPTIMIZED KERNELS 
// ================================================================
// RMS Norm
// ================================================================
template <int THREADS_PER_BLOCK>
__global__ void __launch_bounds__(THREADS_PER_BLOCK)
rms_norm_kernel(
    __hip_bfloat16* __restrict__ Y,
    const __hip_bfloat16* __restrict__ X,
    const __hip_bfloat16* __restrict__ weight,
    size_t D)
{
    const int t_idx = threadIdx.x;
    const int vec_iters = D / 2;

    const __hip_bfloat162* row_in = reinterpret_cast<const __hip_bfloat162*>(X);
    const __hip_bfloat162* weight_in = reinterpret_cast<const __hip_bfloat162*>(weight);
    __hip_bfloat162* row_out = reinterpret_cast<__hip_bfloat162*>(Y);

    float lsum = 0.0f;

    for (int idx = t_idx; idx < vec_iters; idx += THREADS_PER_BLOCK)
    {
        __hip_bfloat162 v_bf16 = __ldg(&row_in[idx]);
        // convert to fp32 for math
        float2 v_fp32 = __hip_bfloat1622float2(v_bf16);

        // lsum += v_fp32.x * v_fp32.x + v_fp32.y * v_fp32.y;
        lsum = __fmaf_rn(v_fp32.x, v_fp32.x, lsum);
        lsum = __fmaf_rn(v_fp32.y, v_fp32.y, lsum);
    }

    using BlockReduce = hipcub::BlockReduce<float, THREADS_PER_BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    float block_sum = BlockReduce(temp_storage).Sum(lsum);

    // 3. calculate the normalization factor
    __shared__ float mul_val;
    if (t_idx == 0) { mul_val = rsqrtf(block_sum * INV_DIM + EPS); }
    __syncthreads();

    // 4. applying the normalization
    for (int idx = t_idx; idx < vec_iters; idx += THREADS_PER_BLOCK)
    {
        __hip_bfloat162 v_in_bf16 = __ldg(&row_in[idx]);
        __hip_bfloat162 v_weight_bf16 = __ldg(&weight_in[idx]);
        float2 v_in_fp32 = __hip_bfloat1622float2(v_in_bf16);
        float2 v_weight_fp32 = __hip_bfloat1622float2(v_weight_bf16);

        v_in_fp32.x = (v_in_fp32.x * mul_val) * v_weight_fp32.x;
        v_in_fp32.y = (v_in_fp32.y * mul_val) * v_weight_fp32.y;

        row_out[idx] = __float22hip_bfloat162_rn(v_in_fp32);
    }
}

void
rmsnorm_gpu(
    __hip_bfloat16* o,
    const __hip_bfloat16* x,
    const __hip_bfloat16* weight,
    int dim)
{
    int num_blocks = (dim + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    rms_norm_kernel<THREADS_PER_BLOCK><<<num_blocks, THREADS_PER_BLOCK>>>(o, x, weight, dim);
}

template <int THREADS_PER_BLOCK, int HEAD_DIM>
__global__ void __launch_bounds__(THREADS_PER_BLOCK)
fused_multi_rmsnorm_kernel(
    bf16* __restrict__ vecs,
    const bf16* __restrict__ weight,
    int num_vecs)
{
    // each block processes one vector/head
    const int vec_idx = blockIdx.x;
    if (vec_idx >= num_vecs) return;

    const int t_idx = threadIdx.x;
    const int vec_iters = HEAD_DIM / 2;

    bf16* vec_start = vecs + vec_idx * HEAD_DIM;

    const __hip_bfloat162* row_in = reinterpret_cast<const __hip_bfloat162*>(vec_start);
    const __hip_bfloat162* weight_in = reinterpret_cast<const __hip_bfloat162*>(weight);
    __hip_bfloat162* row_out = reinterpret_cast<__hip_bfloat162*>(vec_start);

    // 1. calculate sum of squares
    float lsum = 0.0f;
    for (int idx = t_idx; idx < vec_iters; idx += THREADS_PER_BLOCK)
    {
        __hip_bfloat162 v_bf16 = __ldg(&row_in[idx]);
        float2 v_fp32 = __hip_bfloat1622float2(v_bf16);
        lsum += v_fp32.x * v_fp32.x + v_fp32.y * v_fp32.y;
    }

    // 2. block-level reduction
    using BlockReduce = hipcub::BlockReduce<float, THREADS_PER_BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    float block_sum = BlockReduce(temp_storage).Sum(lsum);

    // 3. calculate the normalization factor
    __shared__ float mul_val;
    if (t_idx == 0) { mul_val = rsqrtf(block_sum * INV_HEAD_DIM + EPS); }
    __syncthreads();

    // 4. applying the normalization
    for (int idx = t_idx; idx < vec_iters; idx += THREADS_PER_BLOCK)
    {
        __hip_bfloat162 v_in_bf16 = __ldg(&row_in[idx]);
        __hip_bfloat162 v_weight_bf16 = __ldg(&weight_in[idx]);
        float2 v_in_fp32 = __hip_bfloat1622float2(v_in_bf16);
        float2 v_weight_fp32 = __hip_bfloat1622float2(v_weight_bf16);

        v_in_fp32.x = (v_in_fp32.x * mul_val) * v_weight_fp32.x;
        v_in_fp32.y = (v_in_fp32.y * mul_val) * v_weight_fp32.y;

        row_out[idx] = __float22hip_bfloat162_rn(v_in_fp32);
    }
}

void
qk_norm_fused_gpu(
    bf16* q,
    bf16* k,
    const bf16* q_norm_weight,
    const bf16* k_norm_weight)
{
    constexpr int QK_NORM_THREADS_PER_BLOCK = 64;

    // launching ONE kernel for all query heads
    fused_multi_rmsnorm_kernel<QK_NORM_THREADS_PER_BLOCK, HEAD_DIM><<<N_HEADS, QK_NORM_THREADS_PER_BLOCK>>>
    (q, q_norm_weight, N_HEADS);

    // launching ONE kernel for all key heads
    fused_multi_rmsnorm_kernel<QK_NORM_THREADS_PER_BLOCK, HEAD_DIM><<<N_KV_HEADS, QK_NORM_THREADS_PER_BLOCK>>>
    (k, k_norm_weight, N_KV_HEADS);
}

// ================================================================
// RoPE
// ================================================================
__global__ void
rope_kernel(
    __hip_bfloat16* __restrict__ q, 
    __hip_bfloat16* __restrict__ k, 
    int pos) 
{
    // grid: Q_DIM / 2, block: THREADS_PER_BLOCK
    // each thread handles one pair of dimensions (i, i+1)
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= Q_DIM / 2) { return; }

    int head_dim_idx = (i * 2) % HEAD_DIM;
    float freq = 1.0f / powf(ROPE_THETA, (float)head_dim_idx / (float)HEAD_DIM);
    float val = (float)pos * freq;
    float fcr, fci;
    sincosf(val, &fci, &fcr);

    // rotate Q
    __hip_bfloat162 q_val_bf16 = reinterpret_cast<__hip_bfloat162*>(q)[i];
    float2 q_val_fp32 = __hip_bfloat1622float2(q_val_bf16);
    float q0 = q_val_fp32.x * fcr - q_val_fp32.y * fci;
    float q1 = q_val_fp32.x * fci + q_val_fp32.y * fcr;
    reinterpret_cast<__hip_bfloat162*>(q)[i] = __float22hip_bfloat162_rn(make_float2(q0, q1));

    if (i < KV_DIM / 2)
    {
        // rotate K
        __hip_bfloat162 k_val_bf16 = reinterpret_cast<__hip_bfloat162*>(k)[i];
        float2 k_val_fp32 = __hip_bfloat1622float2(k_val_bf16);
        float k0 = k_val_fp32.x * fcr - k_val_fp32.y * fci;
        float k1 = k_val_fp32.x * fci + k_val_fp32.y * fcr;
        reinterpret_cast<__hip_bfloat162*>(k)[i] = __float22hip_bfloat162_rn(make_float2(k0, k1));
    }
}
    
void 
rope_gpu(
    __hip_bfloat16* q, 
    __hip_bfloat16* k, 
    int pos)
{
    int num_pairs = Q_DIM / 2;
    int grid_size = (num_pairs + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    hipLaunchKernelGGL(rope_kernel, dim3(grid_size), dim3(THREADS_PER_BLOCK), 0, 0, q, k, pos);
}

__global__ void
qwen_naive_rope_kernel(
    bf16* q,
    bf16* k_cache_pos,
    int pos)
{
    // `blockIdx.x` will correspond to the head index 'h'
    int h = blockIdx.x;
    // `threadIdx.x` will correspond to the inner loop index 'j'
    int j = threadIdx.x;

    if (h < N_HEADS && j < HEAD_DIM / 2)
    {
        bf16* q_head = q + h * HEAD_DIM;

        float freq = 1.0f / powf(ROPE_THETA, (float)(j * 2) / (float)HEAD_DIM);
        float val = (float)pos * freq;
        float fcr, fci;
        sincosf(val, &fci, &fcr);

        float q_real = __hip_bfloat162float(q_head[j]);
        float q_imag = __hip_bfloat162float(q_head[j + HEAD_DIM / 2]);

        float q_rotated_real = q_real * fcr - q_imag * fci;
        float q_rotated_imag = q_real * fci + q_imag * fcr;

        q_head[j]              = __float2hip_bfloat16_rn(q_rotated_real);
        q_head[j + HEAD_DIM/2] = __float2hip_bfloat16_rn(q_rotated_imag);
    }

    if (h < N_KV_HEADS && j < HEAD_DIM / 2)
    {
        bf16* k_head = k_cache_pos + h * HEAD_DIM;

        float freq = 1.0f / powf(ROPE_THETA, (float)(j * 2) / (float)HEAD_DIM);
        float val = (float)pos * freq;
        float fcr, fci;
        sincosf(val, &fci, &fcr);

        float k_real = __hip_bfloat162float(k_head[j]);
        float k_imag = __hip_bfloat162float(k_head[j + HEAD_DIM / 2]);

        // perform rotation in fp32
        float k_rotated_real = k_real * fcr - k_imag * fci;
        float k_rotated_imag = k_real * fci + k_imag * fcr;

        k_head[j]              = __float2hip_bfloat16_rn(k_rotated_real);
        k_head[j + HEAD_DIM/2] = __float2hip_bfloat16_rn(k_rotated_imag);
    }
}

void
rope_gpu_naive(
    __hip_bfloat16* q,
    __hip_bfloat16* k,
    int pos)
{
    dim3 grid(N_HEADS, 1, 1);
    dim3 block(HEAD_DIM / 2, 1, 1);

    hipLaunchKernelGGL(qwen_naive_rope_kernel, grid, block, 0, 0, q, k, pos);
}

// ================================================================
// softmax
// ================================================================

// Optimized softmax kernel using shared memory and better thread utilization
__global__ void
softmax_kernel_optimized(
    float* att, 
    int pos)
{
    // grid: N_HEADS, block: min(1024, pos + 1)
    int h = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    int len = pos + 1;

    float* scores = att + (size_t)h * SEQ_LEN;

    // Shared memory for reduction
    extern __shared__ float sdata[];
    float* s_max = sdata;
    float* s_sum = sdata + block_size;

    // Find max value for numerical stability (parallel reduction)
    float thread_max = -INFINITY;
    for (int i = tid; i < len; i += block_size)
    {
        if (scores[i] > thread_max) { thread_max = scores[i]; }
    }
    s_max[tid] = thread_max;
    __syncthreads();

    // Reduce to find global max
    for (int s = block_size / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            if (s_max[tid + s] > s_max[tid]) { s_max[tid] = s_max[tid + s]; }
        }
        __syncthreads();
    }
    float max_val = s_max[0];

    // Compute exp and sum (parallel)
    float thread_sum = 0.0f;
    for (int i = tid; i < len; i += block_size)
    {
        scores[i] = expf(scores[i] - max_val);
        thread_sum += scores[i];
    }
    s_sum[tid] = thread_sum;
    __syncthreads();

    // Reduce to find global sum
    for (int s = block_size / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            s_sum[tid] += s_sum[tid + s];
        }
        __syncthreads();
    }
    float sum = s_sum[0];

    // Normalize (parallel)
    float inv_sum = 1.0f / sum;
    for (int i = tid; i < len; i += block_size)
    {
        scores[i] *= inv_sum;
    }
}

// Fallback softmax kernel for very small sequences
__global__ void
softmax_kernel_simple(
    float* att, 
    int pos)
{
    // grid: N_HEADS, block: 1
    int h = blockIdx.x;

    float* scores = att + (size_t)h * SEQ_LEN;
    int len = pos + 1;

    // find max value for numerical stability
    float max_val = -INFINITY;
    for (int i = 0; i < len; i++)
    {
        if (scores[i] > max_val) { max_val = scores[i]; }
    }

    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < len; i++)
    {
        scores[i] = expf(scores[i] - max_val);
        sum += scores[i];
    }

    // normalize
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < len; i++) { scores[i] *= inv_sum; }
}

// --------------------------------------------------------------
// Softmax dispatcher function with error handling
// --------------------------------------------------------------
void
softmax_gpu(
    float* att, 
    int pos)
{
    // Input validation
    if (att == nullptr)
    {
        printf("Error: Null pointer passed to softmax_gpu\n");
        return;
    }
    
    if (pos < 0 || pos >= SEQ_LEN)
    {
        printf("Error: Invalid position %d for softmax (SEQ_LEN=%d)\n", pos, SEQ_LEN);
        return;
    }

    int len = pos + 1;
    
    // Choosing kernel based on sequence length 
    if (len <= 32)
    {
        // Use simple kernel for very short sequences
        hipLaunchKernelGGL(softmax_kernel_simple, dim3(N_HEADS), dim3(1), 0, 0, att, pos);
    }
    else
    {
        int block_size = std::min(1024, len);
        
        // Checking if we have enough shared memory
        size_t shared_mem_size = 2 * block_size * sizeof(float);
        int max_shared_mem;
        hipDeviceGetAttribute(&max_shared_mem, hipDeviceAttributeMaxSharedMemoryPerBlock, 0);
        
        if (shared_mem_size > max_shared_mem)
        {
            // Falling back to simple kernel if not enough shared memory
            printf("Warning: Not enough shared memory for optimized softmax, using simple kernel\n");
            hipLaunchKernelGGL(softmax_kernel_simple, dim3(N_HEADS), dim3(1), 0, 0, att, pos);
        }
        else
        {
            hipLaunchKernelGGL(softmax_kernel_optimized, dim3(N_HEADS), dim3(block_size), shared_mem_size, 0, att, pos);
        }
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        printf("HIP error in softmax kernel: %s\n", hipGetErrorString(err));
        return;
    }

    err = hipDeviceSynchronize();
    if (err != hipSuccess)
    {
        printf("HIP synchronization error in softmax: %s\n", hipGetErrorString(err));
    }
}


// ================================================================
// Attention
// ================================================================
__global__ void
attention_qk_kernel(
    float* att,
    const bf16* q,
    const bf16* k_cache,
    int pos)
{
    // grid: N_HEADS, block: pos + 1 (up to 1024)
    int h = blockIdx.x; 
    int t = threadIdx.x;

    constexpr int kv_mul = N_HEADS / N_KV_HEADS;

    if (t <= pos)
    {
        const bf16* q_head = q + h * HEAD_DIM;
        int kv_head_idx = h / kv_mul;
        const bf16* k_vec = k_cache + (size_t)t * KV_DIM + (size_t)kv_head_idx * HEAD_DIM;

        float score = 0.0f;
        for (int i = 0; i < HEAD_DIM / 2; i++)
        {
            __hip_bfloat162 q_pair = reinterpret_cast<const __hip_bfloat162*>(q_head)[i];
            __hip_bfloat162 k_pair = reinterpret_cast<const __hip_bfloat162*>(k_vec)[i];

            float2 q_vals = __hip_bfloat1622float2(q_pair);
            float2 k_vals = __hip_bfloat1622float2(k_pair);

            // score += q_vals.x * k_vals.x + q_vals.y * k_vals.y;
            score = __fmaf_rn(q_vals.x, k_vals.x, score);
            score = __fmaf_rn(q_vals.y, k_vals.y, score);
        }

        score /= sqrtf((float)HEAD_DIM);
        att[(size_t)h * SEQ_LEN + t] = score;
    }

}
__global__ void
attention_v_kernel(
    bf16* out,
    const float* att,
    const bf16* v_cache,
    int pos)
{
    // grid: N_HEADS, block: HEAD_DIM
    int h = blockIdx.x;
    int i = threadIdx.x; // idx within the head dimension
    constexpr int kv_mul = N_HEADS / N_KV_HEADS;

    bf16* out_head = out + (size_t)h * HEAD_DIM;
    const float* att_head = att + (size_t)h * SEQ_LEN;
    int kv_head_idx = h / kv_mul;

    float weighted_sum = 0.0f;
    for (int t = 0; t <= pos; t++)
    {
        const bf16* v_vec = v_cache + (size_t)t * KV_DIM + (size_t)kv_head_idx * HEAD_DIM;

        // weighted_sum += att_head[t] * __bfloat162float(v_vec[i]);   
        weighted_sum = __fmaf_rn(att_head[t], __hip_bfloat162float(v_vec[i]), weighted_sum);
    }
    out_head[i] = __float2hip_bfloat16_rn(weighted_sum);
}

void
attention_gpu(
    RunState* s,
    int l, 
    int pos)
{
    bf16* layer_key_cache = s->key_cache     + (size_t)l * SEQ_LEN * KV_DIM;
    bf16* layer_value_cache = s->value_cache + (size_t)l * SEQ_LEN * KV_DIM;

    // kernel 1: calculate QK scores
    int qk_threads_per_block = std::min(1024, pos + 1);
    hipLaunchKernelGGL(attention_qk_kernel, dim3(N_HEADS), dim3(qk_threads_per_block), 0, 0,
        s->att, s->q, layer_key_cache, pos
    );

    // kernel 2: softmax
    softmax_gpu(s->att, pos);

    // kernel 3: aggregate V values
    hipLaunchKernelGGL(attention_v_kernel, dim3(N_HEADS), dim3(HEAD_DIM), 0, 0,
        s->q, s->att, layer_value_cache, pos
    );
}

// ================================================================
// add residual
// ================================================================
__global__ void
add_residual_kernel(
    __hip_bfloat16* x, 
    const __hip_bfloat16* residual, 
    int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
    {
        float x_fp32 = __hip_bfloat162float(x[i]);
        float res_fp32 = __hip_bfloat162float(residual[i]);
        float result_fp32 = x_fp32 + res_fp32;
        x[i] = __float2hip_bfloat16_rn(result_fp32);
    }
}

void

add_residual_gpu(
    __hip_bfloat16* x, 
    const __hip_bfloat16* residual, 
    int size)
{
    int grid_size = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    hipLaunchKernelGGL(add_residual_kernel, dim3(grid_size), dim3(THREADS_PER_BLOCK), 0, 0, x, residual, size);
}

// ================================================================
// swiGlu
// ================================================================
__global__ void
swiglu_kernel(
    __hip_bfloat16* hb,
    const __hip_bfloat16* hb2,
    int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
    {
        float hb_fp32 = __hip_bfloat162float(hb[i]);
        float hb2_fp32 = __hip_bfloat162float(hb2[i]);
        
        // SwiGLU: hb[i] = hb[i] * sigmoid(hb2[i])
        float sigmoid_val = 1.0f / (1.0f + expf(-hb2_fp32));
        float result = hb_fp32 * sigmoid_val;
        
        hb[i] = __float2hip_bfloat16_rn(result);
    }
}

void
swiglu_gpu(
    __hip_bfloat16* hb,
    __hip_bfloat16* hb2,
    int size)
{
    int grid_size = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    hipLaunchKernelGGL(swiglu_kernel, dim3(grid_size), dim3(THREADS_PER_BLOCK), 0, 0, hb, hb2, size);
}

// ================================================================
// matrix multiplication
// ================================================================

void
matmul_hipblas(
    hipblasHandle_t handle,
    bf16* C,
    const bf16* A,
    const bf16* B,
    int M, int N, int K,
    float alpha = 1.0f,
    float beta = 0.0f)
{
    // Note: hipBLAS might have different function signatures
    // This is a placeholder - you may need to adjust based on actual hipBLAS API
    hipblasGemmEx(handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                  N, M, K,
                  &alpha,
                  B, HIPBLAS_R_16BF, N,
                  A, HIPBLAS_R_16BF, K,
                  &beta,
                  C, HIPBLAS_R_16BF, N,
                  HIPBLAS_COMPUTE_32F,
                  HIPBLAS_GEMM_DEFAULT);
}

// ================================================================
// convert bf16 to fp32
// ================================================================
__global__ void
convert_bf16_to_fp32_kernel(
    const bf16* bf16_in, 
    float* fp32_out, 
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n){ fp32_out[i] = __hip_bfloat162float(bf16_in[i]); }
}

float* forward(
    Transformer* transformer, 
    int token, 
    int pos)
{
    RunState* s = &transformer->state;
    TransformerWeights* w = &transformer->weights;
    hipblasHandle_t handle = transformer->hipblas_handle;

    // 1. token embedding lookup
    // copy the token embedding into the main activation buffer s->x
    bf16* token_embedding_ptr = w->token_embedding_table + (size_t)token * DIM;
    hipMemcpy(s->x, token_embedding_ptr, (size_t)DIM * sizeof(bf16), hipMemcpyDeviceToDevice);

    for (int l = 0; l < N_LAYERS; l++)
    {
        const qwen_loader::TransformerBlockWeights& layer = w->layers[l];

        // === ATTENTION BLOCK ===

        // 2. input RMSNorm (pre-attention norm)
        rmsnorm_gpu(s->xb, s->x, layer.input_layernorm_weight, DIM);

        // 3. QKV matrix multiplications
        bf16* k_cache_pos = s->key_cache   + (size_t)l * SEQ_LEN * KV_DIM + (size_t)pos * KV_DIM;
        bf16* v_cache_pos = s->value_cache + (size_t)l * SEQ_LEN * KV_DIM + (size_t)pos * KV_DIM;
        matmul_hipblas(handle, s->q,        layer.attention.q_proj_weight, s->xb,  Q_DIM, DIM);
        matmul_hipblas(handle, k_cache_pos, layer.attention.k_proj_weight, s->xb, KV_DIM, DIM);
        matmul_hipblas(handle, v_cache_pos, layer.attention.v_proj_weight, s->xb, KV_DIM, DIM);

        // 4. QK-Norm
        qk_norm_fused_gpu(s->q, k_cache_pos, layer.attention.q_norm_weight, layer.attention.k_norm_weight);

        // 5. RoPE
        // rope_gpu(s->q, k_cache_pos, pos);
        rope_gpu_naive(s->q, k_cache_pos, pos);

        // 6. MHA (QK^T V)
        attention_gpu(s, l, pos);

        // 7. final attention output projection and residual connection (fused)
        matmul_hipblas(handle, s->x, layer.attention.o_proj_weight, s->q, DIM, Q_DIM, 1.0f, 1.0f);
        // add_residual_gpu(s->x, s->xb2, DIM);

        // === FFN BLOCK ===

        // 8. post-attention RMSNorm
        rmsnorm_gpu(s->xb, s->x, layer.post_attention_layernorm_weight, DIM);

        // 9. FFN projections (Gate and Up)
        // output of w1 matmul is s->hb. output of w3 matmul is s->hb2.
        matmul_hipblas(handle, s->hb,  layer.ffn.gate_proj_weight, s->xb, HIDDEN_DIM, DIM);
        matmul_hipblas(handle, s->hb2, layer.ffn.up_proj_weight,   s->xb, HIDDEN_DIM, DIM);

        // 9. SwiGLU
        // in-place operation on s->hb, using s->hb2 as the gate.
        swiglu_gpu(s->hb, s->hb2, HIDDEN_DIM);

        // 10. final FFN Down Projection matmul and residual connection (fused)
        matmul_hipblas(handle, s->x, layer.ffn.down_proj_weight, s->hb, DIM, HIDDEN_DIM, 1.0f, 1.0f);
        // add_residual_gpu(s->x, s->xb, DIM);
    }

    // === FINAL CLASSIFIER ===

    // 11. final RMSNorm
    // in-place operation on s->x
    rmsnorm_gpu(s->x, s->x, w->final_norm_weight, DIM);

    // 12. classifier Matmul
    matmul_hipblas(handle, s->logits, w->output_head_weight, s->x, VOCAB_SIZE, DIM);

    int grid_size = (VOCAB_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    hipLaunchKernelGGL(convert_bf16_to_fp32_kernel, dim3(grid_size), dim3(THREADS_PER_BLOCK), 0, 0, s->logits, s->d_logits_fp32, VOCAB_SIZE);

    // 13. copy the fp32 logits from GPU device to pinned host memory for the CPU to access
    hipMemcpy(transformer->h_logits, s->d_logits_fp32, (size_t)VOCAB_SIZE * sizeof(float), hipMemcpyDeviceToHost);

    return transformer->h_logits;
}

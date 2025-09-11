// static_loader.hip.h - AMD ROCm/HIP version

#pragma once

#include <iostream>
#include <string>
#include <stdexcept>
#include <vector>
#include <numeric>

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

// memory mapping on Linux/macOS
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include "config.h"

#define HIP_CHECK(call)                                                     \
    do {                                                                     \
        hipError_t err = call;                                              \
        if (err != hipSuccess) {                                            \
            fprintf(stderr, "HIP error at %s %d: %s\n", __FILE__, __LINE__, \
                    hipGetErrorString(err));                                \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

namespace qwen_loader
{

using bf16 = hip_bfloat16;

struct
AttentionWeights
{
    bf16* q_proj_weight;
    bf16* k_proj_weight;
    bf16* v_proj_weight;
    bf16* o_proj_weight;
    bf16* q_norm_weight;
    bf16* k_norm_weight;
};

struct
FFNWeights
{
    bf16* gate_proj_weight;
    bf16* up_proj_weight;
    bf16* down_proj_weight;
};

struct
TransformerBlockWeights
{
    AttentionWeights attention;
    FFNWeights ffn;
    bf16* input_layernorm_weight;
    bf16* post_attention_layernorm_weight;
};

struct
QwenWeights
{
    bf16* token_embedding_table;
    bf16* output_head_weight;
    bf16* final_norm_weight;
    std::vector<TransformerBlockWeights> layers;
};

// Copy bf16 bytes from mapped host memory to device memory
inline void copy_bf16_bytes_to_device(const void* host_src_bf16,
                                      bf16* device_dst_bf16,
                                      size_t count)
{
    HIP_CHECK(hipMemcpy(device_dst_bf16,
                        host_src_bf16,
                        count * sizeof(bf16),
                        hipMemcpyHostToDevice));
}

// Memory mapping utilities
class MemoryMappedFile
{
private:
    int fd;
    void* mapped_data;
    size_t file_size;

public:
    MemoryMappedFile(const std::string& filename)
        : fd(-1), mapped_data(nullptr), file_size(0)
    {
        fd = open(filename.c_str(), O_RDONLY);
        if (fd == -1)
        {
            throw std::runtime_error("Failed to open file: " + filename);
        }

        struct stat st;
        if (fstat(fd, &st) == -1)
        {
            close(fd);
            throw std::runtime_error("Failed to get file size: " + filename);
        }
        file_size = st.st_size;

        mapped_data = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (mapped_data == MAP_FAILED)
        {
            close(fd);
            throw std::runtime_error("Failed to map file: " + filename);
        }
    }

    ~MemoryMappedFile()
    {
        if (mapped_data != nullptr)
        {
            munmap(mapped_data, file_size);
        }
        if (fd != -1)
        {
            close(fd);
        }
    }

    const void* data() const { return mapped_data; }
    size_t size() const { return file_size; }
};

// SafeTensor parsing utilities
class SafeTensorParser
{
private:
    const uint8_t* data;
    size_t data_size;
    size_t offset;

public:
    SafeTensorParser(const void* file_data, size_t file_size)
        : data(static_cast<const uint8_t*>(file_data)), data_size(file_size), offset(0) {}

    template<typename T>
    T read()
    {
        if (offset + sizeof(T) > data_size)
        {
            throw std::runtime_error("Read beyond file bounds");
        }
        T value = *reinterpret_cast<const T*>(data + offset);
        offset += sizeof(T);
        return value;
    }

    void skip(size_t bytes)
    {
        if (offset + bytes > data_size)
        {
            throw std::runtime_error("Skip beyond file bounds");
        }
        offset += bytes;
    }

    size_t tell() const { return offset; }
    void seek(size_t pos) { offset = pos; }
    const void* raw_data() const { return data; }
};

// Load weights
// Prefer ordered blob (weights_qwen600.bin) if present; otherwise fallback to safetensors
void load_qwen_weights(const std::string& checkpoint_path, QwenWeights& weights)
{
    try
    {
        // Try ordered blob alongside the safetensors file
        std::string dir = ".";
        {
            size_t p = checkpoint_path.find_last_of("/\\");
            if (p != std::string::npos) dir = checkpoint_path.substr(0, p);
        }
        std::string ordered_blob = dir + "/weights_qwen600.bin";
        bool use_blob = false;
        size_t cursor = 0;

        // Map whichever source we will read from
        MemoryMappedFile st_file(checkpoint_path);
        MemoryMappedFile blob_file(ordered_blob);
        if (blob_file.size() > 0) { use_blob = true; }

        SafeTensorParser parser(st_file.data(), st_file.size());
        if (!use_blob)
        {
            // Parse SafeTensor header and skip to first tensor
            uint64_t header_size = parser.read<uint64_t>();
            (void)header_size;
            parser.seek(8 + header_size);
        }

        // Allocate memory for weights
        weights.token_embedding_table = nullptr;
        weights.output_head_weight = nullptr;
        weights.final_norm_weight = nullptr;
        weights.layers.resize(N_LAYERS);

        auto copy_next = [&](bf16* dst, size_t count){
            if (use_blob)
            {
                const uint8_t* base = static_cast<const uint8_t*>(blob_file.data());
                const void* src = base + cursor;
                copy_bf16_bytes_to_device(src, dst, count);
                cursor += count * sizeof(bf16);
            }
            else
            {
                const void* src = static_cast<const uint8_t*>(parser.raw_data()) + parser.tell();
                copy_bf16_bytes_to_device(src, dst, count);
                parser.skip(count * sizeof(bf16));
            }
        };

        // Load token embedding table
        size_t token_embedding_size = VOCAB_SIZE * DIM * sizeof(bf16);
        HIP_CHECK(hipMalloc(&weights.token_embedding_table, token_embedding_size));
        copy_next(weights.token_embedding_table, VOCAB_SIZE * DIM);

        // Load output head weight
        size_t output_head_size = VOCAB_SIZE * DIM * sizeof(bf16);
        HIP_CHECK(hipMalloc(&weights.output_head_weight, output_head_size));
        copy_next(weights.output_head_weight, VOCAB_SIZE * DIM);

        // Load final norm weight
        size_t final_norm_size = DIM * sizeof(bf16);
        HIP_CHECK(hipMalloc(&weights.final_norm_weight, final_norm_size));
        copy_next(weights.final_norm_weight, DIM);

        // Load layer weights
        for (int l = 0; l < N_LAYERS; l++)
        {
            auto& layer = weights.layers[l];

            // Attention weights
            size_t q_proj_size = Q_DIM * DIM * sizeof(bf16);
            HIP_CHECK(hipMalloc(&layer.attention.q_proj_weight, q_proj_size));
            copy_next(layer.attention.q_proj_weight, Q_DIM * DIM);

            size_t k_proj_size = KV_DIM * DIM * sizeof(bf16);
            HIP_CHECK(hipMalloc(&layer.attention.k_proj_weight, k_proj_size));
            copy_next(layer.attention.k_proj_weight, KV_DIM * DIM);

            size_t v_proj_size = KV_DIM * DIM * sizeof(bf16);
            HIP_CHECK(hipMalloc(&layer.attention.v_proj_weight, v_proj_size));
            copy_next(layer.attention.v_proj_weight, KV_DIM * DIM);

            size_t o_proj_size = DIM * Q_DIM * sizeof(bf16);
            HIP_CHECK(hipMalloc(&layer.attention.o_proj_weight, o_proj_size));
            copy_next(layer.attention.o_proj_weight, DIM * Q_DIM);

            // QK norm weights
            size_t q_norm_size = HEAD_DIM * sizeof(bf16);
            HIP_CHECK(hipMalloc(&layer.attention.q_norm_weight, q_norm_size));
            copy_next(layer.attention.q_norm_weight, HEAD_DIM);

            size_t k_norm_size = HEAD_DIM * sizeof(bf16);
            HIP_CHECK(hipMalloc(&layer.attention.k_norm_weight, k_norm_size));
            copy_next(layer.attention.k_norm_weight, HEAD_DIM);

            // FFN weights
            size_t gate_proj_size = HIDDEN_DIM * DIM * sizeof(bf16);
            HIP_CHECK(hipMalloc(&layer.ffn.gate_proj_weight, gate_proj_size));
            copy_next(layer.ffn.gate_proj_weight, HIDDEN_DIM * DIM);

            size_t up_proj_size = HIDDEN_DIM * DIM * sizeof(bf16);
            HIP_CHECK(hipMalloc(&layer.ffn.up_proj_weight, up_proj_size));
            copy_next(layer.ffn.up_proj_weight, HIDDEN_DIM * DIM);

            size_t down_proj_size = DIM * HIDDEN_DIM * sizeof(bf16);
            HIP_CHECK(hipMalloc(&layer.ffn.down_proj_weight, down_proj_size));
            copy_next(layer.ffn.down_proj_weight, DIM * HIDDEN_DIM);

            // Layer norm weights
            size_t input_layernorm_size = DIM * sizeof(bf16);
            HIP_CHECK(hipMalloc(&layer.input_layernorm_weight, input_layernorm_size));
            copy_next(layer.input_layernorm_weight, DIM);

            size_t post_attention_layernorm_size = DIM * sizeof(bf16);
            HIP_CHECK(hipMalloc(&layer.post_attention_layernorm_weight, post_attention_layernorm_size));
            copy_next(layer.post_attention_layernorm_weight, DIM);
        }

        std::cout << "Successfully loaded Qwen weights from: " << checkpoint_path << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error loading weights: " << e.what() << std::endl;
        throw;
    }
}

// Free weights memory
void free_qwen_weights(QwenWeights& weights)
{
    if (weights.token_embedding_table) hipFree(weights.token_embedding_table);
    if (weights.output_head_weight) hipFree(weights.output_head_weight);
    if (weights.final_norm_weight) hipFree(weights.final_norm_weight);

    for (auto& layer : weights.layers)
    {
        if (layer.attention.q_proj_weight) hipFree(layer.attention.q_proj_weight);
        if (layer.attention.k_proj_weight) hipFree(layer.attention.k_proj_weight);
        if (layer.attention.v_proj_weight) hipFree(layer.attention.v_proj_weight);
        if (layer.attention.o_proj_weight) hipFree(layer.attention.o_proj_weight);
        if (layer.attention.q_norm_weight) hipFree(layer.attention.q_norm_weight);
        if (layer.attention.k_norm_weight) hipFree(layer.attention.k_norm_weight);
        
        if (layer.ffn.gate_proj_weight) hipFree(layer.ffn.gate_proj_weight);
        if (layer.ffn.up_proj_weight) hipFree(layer.ffn.up_proj_weight);
        if (layer.ffn.down_proj_weight) hipFree(layer.ffn.down_proj_weight);
        
        if (layer.input_layernorm_weight) hipFree(layer.input_layernorm_weight);
        if (layer.post_attention_layernorm_weight) hipFree(layer.post_attention_layernorm_weight);
    }
}

} // namespace qwen_loader

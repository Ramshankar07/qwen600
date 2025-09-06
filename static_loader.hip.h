// static_loader.hip.h - AMD ROCm/HIP version

#pragma once

#include <iostream>
#include <string>
#include <stdexcept>
#include <vector>
#include <numeric>

#include <hip/hip_runtime.h>
#include <hip_bfloat16.h>

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

using bf16 = __hip_bfloat16;

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

// Helper function to convert __nv_bfloat16 to __hip_bfloat16
inline __hip_bfloat16 nv_to_hip_bfloat16(__nv_bfloat16 nv_val)
{
    // Direct bitwise conversion since both are 16-bit
    return *reinterpret_cast<__hip_bfloat16*>(&nv_val);
}

// Helper function to convert __hip_bfloat16 to __nv_bfloat16
inline __nv_bfloat16 hip_to_nv_bfloat16(__hip_bfloat16 hip_val)
{
    // Direct bitwise conversion since both are 16-bit
    return *reinterpret_cast<__nv_bfloat16*>(&hip_val);
}

// Convert array of __nv_bfloat16 to __hip_bfloat16
void convert_nv_to_hip_bfloat16_array(
    const __nv_bfloat16* nv_array,
    __hip_bfloat16* hip_array,
    size_t count)
{
    for (size_t i = 0; i < count; i++)
    {
        hip_array[i] = nv_to_hip_bfloat16(nv_array[i]);
    }
}

// Convert array of __hip_bfloat16 to __nv_bfloat16
void convert_hip_to_nv_bfloat16_array(
    const __hip_bfloat16* hip_array,
    __nv_bfloat16* nv_array,
    size_t count)
{
    for (size_t i = 0; i < count; i++)
    {
        nv_array[i] = hip_to_nv_bfloat16(hip_array[i]);
    }
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
};

// Load weights from SafeTensor file
void load_qwen_weights(const std::string& checkpoint_path, QwenWeights& weights)
{
    try
    {
        MemoryMappedFile file(checkpoint_path);
        SafeTensorParser parser(file.data(), file.size());

        // Parse SafeTensor header
        uint64_t header_size = parser.read<uint64_t>();
        std::string header_json(static_cast<const char*>(file.data()) + 8, header_size);
        
        // Skip header
        parser.seek(8 + header_size);

        // Allocate memory for weights
        weights.token_embedding_table = nullptr;
        weights.output_head_weight = nullptr;
        weights.final_norm_weight = nullptr;
        weights.layers.resize(N_LAYERS);

        // Load token embedding table
        size_t token_embedding_size = VOCAB_SIZE * DIM * sizeof(bf16);
        HIP_CHECK(hipMalloc(&weights.token_embedding_table, token_embedding_size));
        
        // Convert from __nv_bfloat16 to __hip_bfloat16 if needed
        const __nv_bfloat16* nv_token_embedding = static_cast<const __nv_bfloat16*>(parser.data() + parser.tell());
        convert_nv_to_hip_bfloat16_array(nv_token_embedding, weights.token_embedding_table, VOCAB_SIZE * DIM);
        parser.skip(token_embedding_size);

        // Load output head weight
        size_t output_head_size = VOCAB_SIZE * DIM * sizeof(bf16);
        HIP_CHECK(hipMalloc(&weights.output_head_weight, output_head_size));
        
        const __nv_bfloat16* nv_output_head = static_cast<const __nv_bfloat16*>(parser.data() + parser.tell());
        convert_nv_to_hip_bfloat16_array(nv_output_head, weights.output_head_weight, VOCAB_SIZE * DIM);
        parser.skip(output_head_size);

        // Load final norm weight
        size_t final_norm_size = DIM * sizeof(bf16);
        HIP_CHECK(hipMalloc(&weights.final_norm_weight, final_norm_size));
        
        const __nv_bfloat16* nv_final_norm = static_cast<const __nv_bfloat16*>(parser.data() + parser.tell());
        convert_nv_to_hip_bfloat16_array(nv_final_norm, weights.final_norm_weight, DIM);
        parser.skip(final_norm_size);

        // Load layer weights
        for (int l = 0; l < N_LAYERS; l++)
        {
            auto& layer = weights.layers[l];

            // Attention weights
            size_t q_proj_size = Q_DIM * DIM * sizeof(bf16);
            HIP_CHECK(hipMalloc(&layer.attention.q_proj_weight, q_proj_size));
            const __nv_bfloat16* nv_q_proj = static_cast<const __nv_bfloat16*>(parser.data() + parser.tell());
            convert_nv_to_hip_bfloat16_array(nv_q_proj, layer.attention.q_proj_weight, Q_DIM * DIM);
            parser.skip(q_proj_size);

            size_t k_proj_size = KV_DIM * DIM * sizeof(bf16);
            HIP_CHECK(hipMalloc(&layer.attention.k_proj_weight, k_proj_size));
            const __nv_bfloat16* nv_k_proj = static_cast<const __nv_bfloat16*>(parser.data() + parser.tell());
            convert_nv_to_hip_bfloat16_array(nv_k_proj, layer.attention.k_proj_weight, KV_DIM * DIM);
            parser.skip(k_proj_size);

            size_t v_proj_size = KV_DIM * DIM * sizeof(bf16);
            HIP_CHECK(hipMalloc(&layer.attention.v_proj_weight, v_proj_size));
            const __nv_bfloat16* nv_v_proj = static_cast<const __nv_bfloat16*>(parser.data() + parser.tell());
            convert_nv_to_hip_bfloat16_array(nv_v_proj, layer.attention.v_proj_weight, KV_DIM * DIM);
            parser.skip(v_proj_size);

            size_t o_proj_size = DIM * Q_DIM * sizeof(bf16);
            HIP_CHECK(hipMalloc(&layer.attention.o_proj_weight, o_proj_size));
            const __nv_bfloat16* nv_o_proj = static_cast<const __nv_bfloat16*>(parser.data() + parser.tell());
            convert_nv_to_hip_bfloat16_array(nv_o_proj, layer.attention.o_proj_weight, DIM * Q_DIM);
            parser.skip(o_proj_size);

            // QK norm weights
            size_t q_norm_size = HEAD_DIM * sizeof(bf16);
            HIP_CHECK(hipMalloc(&layer.attention.q_norm_weight, q_norm_size));
            const __nv_bfloat16* nv_q_norm = static_cast<const __nv_bfloat16*>(parser.data() + parser.tell());
            convert_nv_to_hip_bfloat16_array(nv_q_norm, layer.attention.q_norm_weight, HEAD_DIM);
            parser.skip(q_norm_size);

            size_t k_norm_size = HEAD_DIM * sizeof(bf16);
            HIP_CHECK(hipMalloc(&layer.attention.k_norm_weight, k_norm_size));
            const __nv_bfloat16* nv_k_norm = static_cast<const __nv_bfloat16*>(parser.data() + parser.tell());
            convert_nv_to_hip_bfloat16_array(nv_k_norm, layer.attention.k_norm_weight, HEAD_DIM);
            parser.skip(k_norm_size);

            // FFN weights
            size_t gate_proj_size = HIDDEN_DIM * DIM * sizeof(bf16);
            HIP_CHECK(hipMalloc(&layer.ffn.gate_proj_weight, gate_proj_size));
            const __nv_bfloat16* nv_gate_proj = static_cast<const __nv_bfloat16*>(parser.data() + parser.tell());
            convert_nv_to_hip_bfloat16_array(nv_gate_proj, layer.ffn.gate_proj_weight, HIDDEN_DIM * DIM);
            parser.skip(gate_proj_size);

            size_t up_proj_size = HIDDEN_DIM * DIM * sizeof(bf16);
            HIP_CHECK(hipMalloc(&layer.ffn.up_proj_weight, up_proj_size));
            const __nv_bfloat16* nv_up_proj = static_cast<const __nv_bfloat16*>(parser.data() + parser.tell());
            convert_nv_to_hip_bfloat16_array(nv_up_proj, layer.ffn.up_proj_weight, HIDDEN_DIM * DIM);
            parser.skip(up_proj_size);

            size_t down_proj_size = DIM * HIDDEN_DIM * sizeof(bf16);
            HIP_CHECK(hipMalloc(&layer.ffn.down_proj_weight, down_proj_size));
            const __nv_bfloat16* nv_down_proj = static_cast<const __nv_bfloat16*>(parser.data() + parser.tell());
            convert_nv_to_hip_bfloat16_array(nv_down_proj, layer.ffn.down_proj_weight, DIM * HIDDEN_DIM);
            parser.skip(down_proj_size);

            // Layer norm weights
            size_t input_layernorm_size = DIM * sizeof(bf16);
            HIP_CHECK(hipMalloc(&layer.input_layernorm_weight, input_layernorm_size));
            const __nv_bfloat16* nv_input_layernorm = static_cast<const __nv_bfloat16*>(parser.data() + parser.tell());
            convert_nv_to_hip_bfloat16_array(nv_input_layernorm, layer.input_layernorm_weight, DIM);
            parser.skip(input_layernorm_size);

            size_t post_attention_layernorm_size = DIM * sizeof(bf16);
            HIP_CHECK(hipMalloc(&layer.post_attention_layernorm_weight, post_attention_layernorm_size));
            const __nv_bfloat16* nv_post_attention_layernorm = static_cast<const __nv_bfloat16*>(parser.data() + parser.tell());
            convert_nv_to_hip_bfloat16_array(nv_post_attention_layernorm, layer.post_attention_layernorm_weight, DIM);
            parser.skip(post_attention_layernorm_size);
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

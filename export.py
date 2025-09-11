# export.py

import argparse
import gzip
import json
import math
import os
import shutil
import struct
from pathlib import Path

import json
from jinja2 import Template
import struct

DTYPE_SIZES = {
    'BF16': 2,
    'F16': 2,
    'F32': 4,
}

def read_safetensors_header(fp):
    fp.seek(0)
    header_size_bytes = fp.read(8)
    if len(header_size_bytes) != 8:
        raise ValueError("Invalid safetensors file: cannot read header size")
    header_size = struct.unpack('<Q', header_size_bytes)[0]
    header_json = fp.read(header_size)
    header = json.loads(header_json)
    return header_size, header

def find_first(keys, candidates):
    for name in candidates:
        if name in keys:
            return name
    return None

def export_weights(model_dir):
    from pathlib import Path
    model_path = Path(model_dir)
    st_path = model_path / 'model.safetensors'
    if not st_path.exists():
        raise FileNotFoundError(f"{st_path} not found")

    # constants must match config.h
    SEQ_LEN = 8192
    VOCAB_SIZE = 151936
    DIM = 1024
    HIDDEN_DIM = 3072
    N_LAYERS = 28
    N_HEADS = 16
    N_KV_HEADS = 8
    HEAD_DIM = 128
    Q_DIM = N_HEADS * HEAD_DIM
    KV_DIM = N_KV_HEADS * HEAD_DIM

    with open(st_path, 'rb') as fp:
        header_size, header = read_safetensors_header(fp)
        tensors = header
        keys = set(tensors.keys())

        # Build ordered tensor name list by probing common naming variants
        ordered = []
        # embeddings
        emb = find_first(keys, [
            'model.embed_tokens.weight',
            'model.wte.weight',
        ])
        lm_head = find_first(keys, [
            'lm_head.weight',
            'output.weight',
        ])
        final_norm = find_first(keys, [
            'model.norm.weight',
            'model.final_layernorm.weight',
        ])
        if not emb or not lm_head or not final_norm:
            raise ValueError('Could not locate embedding/lm_head/final_norm in safetensors keys')
        ordered.extend([emb, lm_head, final_norm])

        for i in range(N_LAYERS):
            prefix_variants = [
                f'model.layers.{i}.attention',
                f'model.layers.{i}.self_attn',
                f'model.layers.{i}.attn',
            ]
            q_proj = find_first(keys, [f'{p}.q_proj.weight' for p in prefix_variants] + [f'{p}.wq.weight' for p in prefix_variants])
            k_proj = find_first(keys, [f'{p}.k_proj.weight' for p in prefix_variants] + [f'{p}.wk.weight' for p in prefix_variants])
            v_proj = find_first(keys, [f'{p}.v_proj.weight' for p in prefix_variants] + [f'{p}.wv.weight' for p in prefix_variants])
            o_proj = find_first(keys, [f'{p}.o_proj.weight' for p in prefix_variants] + [f'{p}.wo.weight' for p in prefix_variants])
            q_norm = find_first(keys, [f'{p}.q_norm.weight' for p in prefix_variants])
            k_norm = find_first(keys, [f'{p}.k_norm.weight' for p in prefix_variants])

            mlp_prefix_variants = [
                f'model.layers.{i}.mlp',
                f'model.layers.{i}.feed_forward',
            ]
            gate = find_first(keys, [f'{p}.gate_proj.weight' for p in mlp_prefix_variants])
            up = find_first(keys, [f'{p}.up_proj.weight' for p in mlp_prefix_variants])
            down = find_first(keys, [f'{p}.down_proj.weight' for p in mlp_prefix_variants])

            inp_ln = find_first(keys, [f'model.layers.{i}.input_layernorm.weight', f'model.layers.{i}.pre_attn_norm.weight'])
            post_ln = find_first(keys, [f'model.layers.{i}.post_attention_layernorm.weight', f'model.layers.{i}.post_attn_norm.weight'])

            missing = [n for n in [q_proj,k_proj,v_proj,o_proj,q_norm,k_norm,gate,up,down,inp_ln,post_ln] if not n]
            if missing:
                raise ValueError(f'Missing tensors for layer {i}: {missing}')

            ordered.extend([q_proj,k_proj,v_proj,o_proj,q_norm,k_norm,gate,up,down,inp_ln,post_ln])

        # Write out in raw order as bf16 bytes (no conversion); use offsets from header
        out_path = model_path / 'weights_qwen600.bin'
        with open(out_path, 'wb') as out_f:
            for name in ordered:
                info = tensors[name]
                dtype = info['dtype']
                shape = info['shape']
                offsets = info['data_offsets']
                byte_len = (offsets[1]-offsets[0])
                if dtype not in ('BF16','F16','F32'):
                    raise ValueError(f'Unsupported dtype {dtype} for {name}')
                fp.seek(8 + header_size + offsets[0])
                chunk = fp.read(byte_len)
                if len(chunk) != byte_len:
                    raise ValueError(f'Failed to read bytes for {name}')
                out_f.write(chunk)

        print(f"Written ordered weight blob to {out_path}")

def bytes_to_unicode():
    """Reference GPT-2 byte→Unicode map."""
    bs = list(range(ord("!"), ord("~") + 1))
    bs += list(range(ord("¡"), ord("¬") + 1))
    bs += list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))

def internal_to_bytes(U2B, token_str: str) -> bytes:
    return b''.join(
        bytes([U2B[ch]]) if ch in U2B else ch.encode('utf-8')
        for ch in token_str
    )

def build_tokenizer(model, output_dir):
    B2U = bytes_to_unicode()
    U2B = {u: b for b, u in B2U.items()}

    tokenizer = model.tokenizer

    # Get ID → token mapping
    vocab = tokenizer.get_vocab()
    id_to_token = {v: k for k, v in vocab.items()}
    all_tokens = [id_to_token[i] for i in sorted(id_to_token)]

    tokenizer_data = json.loads(tokenizer.backend_tokenizer.to_str())

    # Extract vocab and merge rules
    vocab = tokenizer_data["model"]["vocab"]
    merges = tokenizer_data["model"]["merges"]

    # Build merge rank table
    merge_rank = {''.join(tuple(merge if isinstance(merge, list) else merge.split())): i for i, merge in enumerate(merges)}

    # Create pseudo-score dictionary
    # Tokens from initial vocab get score 0 (unmerged tokens)
    # Merged tokens get scores based on merge rank
    pseudo_scores = {}
    for token_id, token in enumerate(all_tokens):
        # If this token was the result of a merge, it will appear in merge_rank
        rank = merge_rank.get(token)

        if rank is not None:
            score = -math.log(rank + 1)
        else:
            score = -1e6  # Initial vocab tokens
        pseudo_scores[token] = score

    max_token_length = max(len(t) for t in all_tokens)
    tokenizer_path = os.path.join(output_dir, "tokenizer.bin")

    with open(tokenizer_path, "wb") as out_f:
        # Header: max_token_length, bos_token_id, eos_token_id
        out_f.write(struct.pack("<I", max_token_length))
        out_f.write(struct.pack("<I", model.bos_token_id))
        out_f.write(struct.pack("<I", model.eos_token_id))

        for id, token in enumerate(all_tokens):
            token_bytes = internal_to_bytes(U2B, token)
            out_f.write(struct.pack("f", pseudo_scores[token])) # merge score
            out_f.write(struct.pack("<I", len(token_bytes))) # 4 bytes: token length
            out_f.write(token_bytes)                         # UTF-8 bytes

    print(f"Written tokenizer model to {tokenizer_path}")

def build_prompts(model, output_dir):
    template = Template(model.tokenizer.chat_template)

    # Template 1: User
    messages = [{"role": "user", "content": "%s"}]
    rendered_prompt = template.render(messages=messages, add_generation_prompt=True, enable_thinking=False)
    with open(os.path.join(output_dir, 'template_user.txt'), 'w', encoding='utf-8', newline='') as f:
        f.write(rendered_prompt)

    # Template 2: User with Thinking
    rendered_prompt = template.render(messages=messages, add_generation_prompt=True, enable_thinking=True)
    with open(os.path.join(output_dir, 'template_user_thinking.txt'), 'w', encoding='utf-8', newline='') as f:
        f.write(rendered_prompt)

    # Template 3: System + User
    messages = [{"role": "system", "content": "%s"}, {"role": "user", "content": "%s"}]
    rendered_prompt = template.render(messages=messages, add_generation_prompt=True, enable_thinking=False)
    with open(os.path.join(output_dir, 'template_system.txt'), 'w', encoding='utf-8', newline='') as f:
        f.write(rendered_prompt)

    # Template 4: System + User with Thinking
    rendered_prompt = template.render(messages=messages, add_generation_prompt=True, enable_thinking=True)
    with open(os.path.join(output_dir, 'template_system_thinking.txt'), 'w', encoding='utf-8', newline='') as f:
        f.write(rendered_prompt)

    print(f"Written prompt templates to '{output_dir}'")

# -----------------------------------------------------------------------------
# Load / import functions

def load_tokenizer_and_config(model_path):
    """Loads only the tokenizer and config, not the full model weights."""
    try:
        from transformers import AutoConfig, AutoTokenizer
        from types import SimpleNamespace
    except ImportError:
        print("Error: transformers package is required.")
        print("Please run `pip install transformers` to install it.")
        return None

    print(f"Loading tokenizer and config from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    hf_config = AutoConfig.from_pretrained(model_path)

    model_mock = SimpleNamespace()
    model_mock.tokenizer = tokenizer
    model_mock.bos_token_id = hf_config.bos_token_id if hasattr(hf_config, "bos_token_id") else 0
    model_mock.eos_token_id = hf_config.eos_token_id if hasattr(hf_config, "eos_token_id") else 0
    
    print("Successfully loaded tokenizer and config.")
    return model_mock

# -----------------------------------------------------------------------------
# CLI entrypoint

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate tokenizer.bin and template files")
    parser.add_argument("model_path", type=str, help="Path to the local Hugging Face model directory (used for both input and output).")
    parser.add_argument("--export-weights", action='store_true', help="Also export ordered weights blob (weights_qwen600.bin)")
    args = parser.parse_args()

    model_info = load_tokenizer_and_config(args.model_path)

    if model_info:
        build_tokenizer(model_info, args.model_path)
        build_prompts(model_info, args.model_path)
        if args.export_weights:
            export_weights(args.model_path)

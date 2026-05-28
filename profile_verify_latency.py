"""Measure target-model verification latency vs. draft tree size at various context lengths."""

import argparse
import copy
import os
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, DynamicCache


def cuda_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def build_dummy_tree_attention_mask(
        n_nodes: int,
        context_length: int,
        dtype: torch.dtype,
        device: torch.device,
) -> torch.Tensor:
    total_length = context_length + n_nodes
    mask = torch.zeros((1, 1, n_nodes, total_length), dtype=dtype, device=device)
    tree_block = torch.full(
        (n_nodes, n_nodes), torch.finfo(dtype).min, device=device
    )
    for i in range(n_nodes):
        tree_block[i, :i + 1] = 0
    mask[0, 0, :, context_length:context_length + n_nodes] = tree_block
    return mask


@torch.inference_mode()
def profile_verification_latency(
        model: AutoModelForCausalLM,
        context_lengths: list,
        node_counts: list,
        warmup_iters: int = 5,
        measure_iters: int = 20,
        device: torch.device = None,
) -> dict:
    if device is None:
        device = next(model.parameters()).device

    results = {
        "context_lengths": context_lengths,
        "node_counts": node_counts,
        "latencies": {},
        "medians": {},
    }

    vocab_size = model.config.vocab_size

    for ctx_len in context_lengths:
        print(f"\n--- Profiling context length = {ctx_len} ---")
        results["latencies"][ctx_len] = {}
        results["medians"][ctx_len] = {}

        past_kv = DynamicCache()

        if ctx_len > 0:
            dummy_input_ids = torch.randint(
                0, vocab_size, (1, ctx_len), dtype=torch.long, device=device
            )
            position_ids = torch.arange(ctx_len, device=device).unsqueeze(0)
            chunk_size = min(ctx_len, 2048)
            for chunk_start in range(0, ctx_len, chunk_size):
                chunk_end = min(chunk_start + chunk_size, ctx_len)
                chunk_ids = dummy_input_ids[:, chunk_start:chunk_end]
                chunk_pos = position_ids[:, chunk_start:chunk_end]
                _ = model(
                    chunk_ids, position_ids=chunk_pos,
                    past_key_values=past_kv, use_cache=True, logits_to_keep=1,
                )
        print(f"  Prefilled context of {ctx_len} tokens (KV cache ready)")

        for n_nodes in node_counts:
            verify_ids = torch.randint(
                0, vocab_size, (1, n_nodes), dtype=torch.long, device=device
            )
            verify_pos = torch.arange(
                ctx_len, ctx_len + n_nodes, device=device
            ).unsqueeze(0)
            verify_mask = build_dummy_tree_attention_mask(
                n_nodes, ctx_len, dtype=model.dtype, device=device,
            )

            for _ in range(warmup_iters):
                past_kv_copy = copy.deepcopy(past_kv)
                _ = model(
                    verify_ids, position_ids=verify_pos,
                    attention_mask=verify_mask,
                    past_key_values=past_kv_copy, use_cache=True,
                    output_hidden_states=True,
                )

            latencies = []
            for _ in range(measure_iters):
                past_kv_copy = copy.deepcopy(past_kv)

                start_t = cuda_time()
                _ = model(
                    verify_ids, position_ids=verify_pos,
                    attention_mask=verify_mask,
                    past_key_values=past_kv_copy, use_cache=True,
                    output_hidden_states=True,
                )
                elapsed = cuda_time() - start_t
                latencies.append(elapsed)

            median_lat = np.median(latencies)
            std_lat = np.std(latencies)
            results["latencies"][ctx_len][n_nodes] = latencies
            results["medians"][ctx_len][n_nodes] = median_lat
            print(f"  nodes={n_nodes:5d}  latency={median_lat*1000:.3f} ± {std_lat*1000:.3f} ms")

        del past_kv
        torch.cuda.empty_cache()

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile target-model verification latency vs. tree size"
    )
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--context-lengths", type=str, default="0,1024,2048,3072,4096,5120,6144,7168,8192",
                        help="Comma-separated context lengths to profile")
    parser.add_argument("--max-nodes", type=int, default=1024,
                        help="Profile node counts 1, 2, ..., max-nodes")
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--measure-iters", type=int, default=20)
    parser.add_argument("--save-dir", type=str, default="data/verify_latency")
    args = parser.parse_args()

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    context_lengths = [int(x) for x in args.context_lengths.split(",")]
    node_counts = list(range(1, args.max_nodes + 1))

    print(f"Loading model: {args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        attn_implementation="sdpa",
        dtype=torch.bfloat16,
    ).to(device).eval()

    print(f"Context lengths: {context_lengths}")
    print(f"Node counts: 1 to {args.max_nodes}")

    profile_data = profile_verification_latency(
        model, context_lengths, node_counts,
        warmup_iters=args.warmup_iters, measure_iters=args.measure_iters,
        device=device,
    )

    os.makedirs(args.save_dir, exist_ok=True)
    model_slug = args.model_name_or_path.replace("/", "_")
    save_path = os.path.join(args.save_dir, f"verify_latency__{model_slug}.pt")

    torch.save({
        "profile_data": profile_data,
        "args": vars(args),
    }, save_path)
    print(f"\nSaved to {save_path}")


if __name__ == "__main__":
    main()

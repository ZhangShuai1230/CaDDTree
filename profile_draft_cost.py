"""Measure per-round draft model latency (Cd)."""

import argparse
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

import distributed as dist
from model import DFlashDraftModel, extract_context_feature


def cuda_sync_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--draft-name-or-path", type=str, required=True)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--context-length", type=int, default=512,
                        help="Simulated KV-cache length (tokens).")
    parser.add_argument("--n-warmup", type=int, default=5)
    parser.add_argument("--n-measure", type=int, default=500)
    args = parser.parse_args()

    dist.init()
    torch.cuda.set_device(dist.local_rank())
    device = torch.device(f"cuda:{dist.local_rank()}")

    target = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        attn_implementation="sdpa",
        dtype=torch.bfloat16,
    ).to(device).eval()

    draft_model = DFlashDraftModel.from_pretrained(
        args.draft_name_or_path,
        attn_implementation="flash_attention_2",
        dtype=torch.bfloat16,
    ).to(device).eval()

    block_size = args.block_size or draft_model.block_size
    draft_horizon = block_size - 1
    ctx_len = args.context_length

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    dummy_ids = torch.randint(
        100, tokenizer.vocab_size, (1, ctx_len), device=device
    )
    position_ids = torch.arange(
        ctx_len + block_size, device=device
    ).unsqueeze(0)

    past_kv_target = DynamicCache()
    past_kv_draft = DynamicCache()

    with torch.inference_mode():
        out = target(
            dummy_ids,
            position_ids=position_ids[:, :ctx_len],
            past_key_values=past_kv_target,
            use_cache=True,
            logits_to_keep=1,
            output_hidden_states=True,
        )
        target_hidden = extract_context_feature(
            out.hidden_states, draft_model.target_layer_ids
        )
        block_ids = torch.randint(
            100, tokenizer.vocab_size, (1, block_size), device=device
        )
        noise_emb = target.model.embed_tokens(block_ids)
        _ = draft_model(
            target_hidden=target_hidden,
            noise_embedding=noise_emb,
            position_ids=position_ids[:, :ctx_len + block_size],
            past_key_values=past_kv_draft,
            use_cache=True,
            is_causal=False,
        )

    latencies = []

    with torch.inference_mode():
        for i in range(args.n_warmup + args.n_measure):
            block_ids = torch.randint(
                100, tokenizer.vocab_size, (1, block_size), device=device
            )
            noise_emb = target.model.embed_tokens(block_ids)

            t0 = cuda_sync_time()
            _ = target.lm_head(
                draft_model(
                    target_hidden=target_hidden,
                    noise_embedding=noise_emb,
                    position_ids=position_ids[:, :ctx_len + block_size],
                    past_key_values=past_kv_draft,
                    use_cache=True,
                    is_causal=False,
                )[:, -draft_horizon:, :]
            )
            elapsed = cuda_sync_time() - t0

            past_kv_draft.crop(ctx_len)

            if i >= args.n_warmup:
                latencies.append(elapsed * 1000)

    if dist.is_main():
        arr = np.array(latencies)
        print(f"\nDraft cost Cd  (model={args.draft_name_or_path})")
        print(f"  context_length : {ctx_len} tokens")
        print(f"  n_measure      : {args.n_measure}")
        print(f"  mean  : {arr.mean():.3f} ms")
        print(f"  std   : {arr.std():.3f} ms")
        print(f"  p50   : {np.percentile(arr, 50):.3f} ms")
        print(f"  p95   : {np.percentile(arr, 95):.3f} ms")
        print(f"\n  --draft-cost value to use: {arr.mean()/1000:.5f}  (seconds)")


if __name__ == "__main__":
    main()

"""Unified benchmark: AR, DFlash, DDTree (fixed budget), and CaDDTree."""

import argparse
import random
from itertools import chain
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import distributed as dist
from model import DFlashDraftModel, load_and_process_dataset
from dflash import dflash_generate
from ddtree import ddtree_generate, maybe_enable_cpp_compact
from caddtree import caddtree_generate
from cost_model import (
    CostModel,
    ConvexCostModel,
    load_cost_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark CaDDTree vs DFlash / DDTree"
    )
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--draft-name-or-path", type=str, required=True)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument(
        "--tree-budget", type=str, default="16,32,64,128,256,512,1024",
        help="Comma-separated list of DDTree fixed budgets to benchmark"
    )
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--flash-attn", action="store_true")
    parser.add_argument("--disable-cpp-compact-cache", action="store_true")
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--skip-baseline", action="store_true")

    parser.add_argument(
        "--cost-model-path", type=str, default=None,
        help="Path to a saved ConvexCostModel (.pt)."
    )
    parser.add_argument(
        "--cost-profile-path", type=str, default=None,
        help="Path to verify_latency profiling data (.pt) for fitting a cost model on-the-fly."
    )
    parser.add_argument(
        "--draft-cost", type=float, default=None,
        help="Draft cost Cd in seconds. Required if fitting from profile data."
    )
    parser.add_argument(
        "--max-tree-budget", type=int, default=1024,
        help="Maximum allowed tree budget for CaDDTree (buffer allocation ceiling)."
    )
    parser.add_argument(
        "--skip-caddtree", action="store_true",
        help="Skip CaDDTree evaluation (only run baselines)."
    )
    parser.add_argument(
        "--caddtree-only", action="store_true",
        help="Run only CaDDTree; skip AR, DFlash, and DDTree grid baselines."
    )
    parser.add_argument(
        "--save-tree-traces", action="store_true",
        help="Save per-round tree traces (increases output size)."
    )

    args = parser.parse_args()
    print(args)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("start initialization")
    dist.init()
    torch.cuda.set_device(dist.local_rank())
    device = torch.device(f"cuda:{dist.local_rank()}")
    maybe_enable_cpp_compact(not args.disable_cpp_compact_cache)
    print("end initialization")

    def has_flash_attn() -> bool:
        try:
            import flash_attn  # noqa: F401
            return True
        except ImportError:
            return False

    installed_flash_attn = has_flash_attn()
    if not installed_flash_attn:
        raise RuntimeError(
            "flash_attn must be installed because the draft DFlash model "
            "always uses FlashAttention"
        )

    target_attn_implementation = "flash_attention_2" if args.flash_attn else "sdpa"
    draft_attn_implementation = "flash_attention_2"

    if not args.flash_attn and installed_flash_attn:
        print(
            "DDTree/CaDDTree uses a custom tree attention mask on the target "
            "model. For compatibility, forcing the target verifier to torch.sdpa."
        )

    target = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        attn_implementation=target_attn_implementation,
        dtype=torch.bfloat16,
    ).to(device).eval()
    print("load target ok")

    draft_model = DFlashDraftModel.from_pretrained(
        args.draft_name_or_path,
        attn_implementation=draft_attn_implementation,
        dtype=torch.bfloat16,
    ).to(device).eval()
    print("load draft_model ok")

    block_size = (
        args.block_size if args.block_size is not None
        else draft_model.block_size
    )

    tree_budgets = [int(b) for b in args.tree_budget.split(",")]
    methods_to_run = [] if args.caddtree_only else ["dflash"]
    method_key_to_tree_budget = {}

    if not args.flash_attn and not args.caddtree_only:
        for tb in tree_budgets:
            key = f"ddtree_tb{tb}"
            methods_to_run.append(key)
            method_key_to_tree_budget[key] = tb

    cost_model: CostModel | None = None
    run_caddtree = not args.skip_caddtree and not args.flash_attn

    if run_caddtree:
        if args.cost_model_path is not None:
            cost_model = load_cost_model(args.cost_model_path)
            print(f"Loaded cost model from {args.cost_model_path}")
        elif args.cost_profile_path is not None:
            if args.draft_cost is None:
                raise ValueError(
                    "--draft-cost is required when fitting from --cost-profile-path"
                )
            cost_model = ConvexCostModel.from_profile_data(
                args.cost_profile_path,
                draft_cost=args.draft_cost,
            )
            print(f"Fitted cost model from {args.cost_profile_path}")
        else:
            print("No cost model provided; skipping CaDDTree. Use --cost-model-path or --cost-profile-path.")
            run_caddtree = False

    if run_caddtree:
        methods_to_run.append("caddtree")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    dataset = load_and_process_dataset(args.dataset)
    print("load dataset ok")

    if args.max_samples is not None and len(dataset) > args.max_samples:
        dataset = dataset.shuffle(seed=0).select(range(args.max_samples))

    warmup_input_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Warmup"}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    warmup_input_ids = tokenizer.encode(
        warmup_input_text, return_tensors="pt"
    ).to(target.device)
    warmup_max_new_tokens = min(args.max_new_tokens, 16)

    _ = dflash_generate(
        model=draft_model, target=target,
        input_ids=warmup_input_ids,
        mask_token_id=draft_model.mask_token_id,
        max_new_tokens=warmup_max_new_tokens,
        block_size=1,
        stop_token_ids=[tokenizer.eos_token_id],
        temperature=args.temperature,
    )
    for method_key in methods_to_run:
        if method_key == "dflash":
            _ = dflash_generate(
                model=draft_model, target=target,
                input_ids=warmup_input_ids,
                mask_token_id=draft_model.mask_token_id,
                max_new_tokens=warmup_max_new_tokens,
                block_size=block_size,
                stop_token_ids=[tokenizer.eos_token_id],
                temperature=args.temperature,
            )
        elif method_key == "caddtree":
            _ = caddtree_generate(
                model=draft_model, target=target,
                input_ids=warmup_input_ids,
                mask_token_id=draft_model.mask_token_id,
                max_new_tokens=warmup_max_new_tokens,
                block_size=block_size,
                stop_token_ids=[tokenizer.eos_token_id],
                cost_model=cost_model,
                temperature=args.temperature,
                max_tree_budget=args.max_tree_budget,
            )
        else:
            _ = ddtree_generate(
                model=draft_model, target=target,
                input_ids=warmup_input_ids,
                mask_token_id=draft_model.mask_token_id,
                max_new_tokens=warmup_max_new_tokens,
                block_size=block_size,
                tree_budget=method_key_to_tree_budget[method_key],
                stop_token_ids=[tokenizer.eos_token_id],
                temperature=args.temperature,
            )
    print("warmup done")

    responses = []
    indices = range(dist.rank(), len(dataset), dist.size())

    for idx in tqdm(indices, disable=not dist.is_main()):
        instance = dataset[idx]
        messages = []
        for user_content in instance["turns"]:
            messages.append({"role": "user", "content": user_content})
            input_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            input_ids = tokenizer.encode(
                input_text, return_tensors="pt"
            ).to(target.device)

            response = {}

            if not args.skip_baseline:
                response["baseline"] = dflash_generate(
                    model=draft_model, target=target,
                    input_ids=input_ids,
                    mask_token_id=draft_model.mask_token_id,
                    max_new_tokens=args.max_new_tokens,
                    block_size=1,
                    stop_token_ids=[tokenizer.eos_token_id],
                    temperature=args.temperature,
                )

            for method_key in methods_to_run:
                if method_key == "dflash":
                    response[method_key] = dflash_generate(
                        model=draft_model, target=target,
                        input_ids=input_ids,
                        mask_token_id=draft_model.mask_token_id,
                        max_new_tokens=args.max_new_tokens,
                        block_size=block_size,
                        stop_token_ids=[tokenizer.eos_token_id],
                        temperature=args.temperature,
                    )
                elif method_key == "caddtree":
                    response[method_key] = caddtree_generate(
                        model=draft_model, target=target,
                        input_ids=input_ids,
                        mask_token_id=draft_model.mask_token_id,
                        max_new_tokens=args.max_new_tokens,
                        block_size=block_size,
                        stop_token_ids=[tokenizer.eos_token_id],
                        cost_model=cost_model,
                        temperature=args.temperature,
                        max_tree_budget=args.max_tree_budget,
                        save_tree_traces=args.save_tree_traces,
                    )
                else:
                    response[method_key] = ddtree_generate(
                        model=draft_model, target=target,
                        input_ids=input_ids,
                        mask_token_id=draft_model.mask_token_id,
                        max_new_tokens=args.max_new_tokens,
                        block_size=block_size,
                        tree_budget=method_key_to_tree_budget[method_key],
                        stop_token_ids=[tokenizer.eos_token_id],
                        temperature=args.temperature,
                    )

            spec_response = response[methods_to_run[-1]]
            generated_ids = spec_response.output_ids[
                0, spec_response.num_input_tokens :
            ]
            output_text = tokenizer.decode(
                generated_ids, skip_special_tokens=True
            )
            messages.append({"role": "assistant", "content": output_text})
            responses.append(response)

    if dist.size() > 1:
        responses = dist.gather(responses, dst=0)
        if not dist.is_main():
            return
        responses = list(chain(*responses))

    if dist.is_main():
        _print_summary(responses, methods_to_run, args)

    run_data = {
        "responses": responses,
        "block_size": block_size,
        "draft_attn_implementation": draft_attn_implementation,
        "target_attn_implementation": target_attn_implementation,
        "methods": methods_to_run,
        "args": vars(args),
    }

    if args.save_path is not None:
        save_path = Path(args.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(run_data, save_path)
        print(f"Results saved to {save_path}")


def _print_summary(
    responses: list[dict],
    methods: list[str],
    args,
) -> None:
    print("\n" + "=" * 80)
    print(f"  BENCHMARK RESULTS: {args.dataset}")
    print(f"  Model: {args.model_name_or_path}")
    print(f"  Draft: {args.draft_name_or_path}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Samples: {len(responses)}")
    print("=" * 80)

    header = f"{'Method':<22} {'ms/tok':>8} {'τ (acc)':>8} {'rounds':>8}"
    if "caddtree" in methods:
        header += f" {'budget_μ':>9} {'budget_σ':>9}"
    print(header)
    print("-" * len(header))

    if "baseline" in responses[0]:
        vals = [r["baseline"] for r in responses if "baseline" in r]
        ms_per_tok = np.mean([v.time_per_output_token * 1000 for v in vals])
        print(f"{'AR (baseline)':<22} {ms_per_tok:>8.2f} {'–':>8} {'–':>8}")

    for method_key in methods:
        vals = [r[method_key] for r in responses if method_key in r]
        if not vals:
            continue

        ms_per_tok = np.mean([v.time_per_output_token * 1000 for v in vals])
        acc_len = np.mean([
            np.mean(v.acceptance_lengths) if v.acceptance_lengths else 0
            for v in vals
        ])
        rounds = np.mean([v.decode_rounds for v in vals])

        row = f"{method_key:<22} {ms_per_tok:>8.2f} {acc_len:>8.2f} {rounds:>8.1f}"

        if method_key == "caddtree":
            budget_means = [v.budget_mean for v in vals]
            budget_stds = [v.budget_std for v in vals]
            row += f" {np.mean(budget_means):>9.1f} {np.mean(budget_stds):>9.1f}"

        print(row)

    print("=" * 80)


if __name__ == "__main__":
    main()

"""
CaDDTree: Cost-Aware Diffusion Draft Tree for speculative decoding with per-round adaptive budget.
"""

from __future__ import annotations

import heapq
import math
from types import SimpleNamespace
import numpy as np
import torch
from transformers import AutoModelForCausalLM, DynamicCache

from model import DFlashDraftModel, sample, extract_context_feature
from dflash import cuda_time
from ddtree import (
    compile_ddtree_tree,
    follow_verified_tree,
    compact_dynamic_cache,
)
from cost_model import CostModel


def build_caddtree_tree(
    draft_logits: torch.Tensor,
    cost_model: CostModel,
    context_length: int,
    max_budget: int = 1024,
) -> tuple[
    torch.Tensor,          # node_token_ids
    torch.Tensor,          # node_depths
    list[int],             # parents
    list[dict[int, int]],  # child_maps
    torch.Tensor,          # visibility (CPU, bool)
    int,                   # budget (nodes selected)
]:
    if draft_logits.shape[0] == 0:
        visibility = torch.zeros((1, 1), dtype=torch.bool)
        visibility[0, 0] = True
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long), [-1], [dict()], visibility, 0

    topk = min(max_budget, draft_logits.shape[-1])
    depth_limit = int(draft_logits.shape[0])

    logits = draft_logits.float()
    top_logits, top_token_ids = torch.topk(logits, k=topk, dim=-1)
    log_z = torch.logsumexp(logits, dim=-1, keepdim=True)
    top_log_probs_cpu = (top_logits - log_z).to(device="cpu", dtype=torch.float32)
    top_token_ids_cpu = top_token_ids.to(device="cpu", dtype=torch.long)

    top_log_probs_np = top_log_probs_cpu.numpy()
    top_token_ids_np = top_token_ids_cpu.numpy()

    first_logw = float(top_log_probs_np[0, 0])
    heap: list[tuple[float, tuple[int, ...], int, int, int, float]] = [
        (-first_logw, (0,), 0, 1, 0, first_logw)
    ]

    node_token_ids_np = np.empty(max_budget, dtype=np.int64)
    node_depths_np = np.empty(max_budget, dtype=np.int64)
    parents_np = np.empty(max_budget + 1, dtype=np.int32)
    parents_np[0] = -1
    child_maps: list[dict[int, int]] = [dict()]

    node_count = 0
    phi = 0.0  # cumulative Φ*(n)
    Cd = cost_model.Cd
    # θ*(0) = 1 / (Cd + Cv(0; ℓ))
    theta_prev = 1.0 / (Cd + cost_model.Cv(0, context_length))

    while heap and node_count < max_budget:
        neg_logw, ranks, parent_index, depth, rank, logw = heapq.heappop(heap)

        delta = math.exp(logw)

        # θ_new = (1 + Φ + δ) / (Cd + Cv(n+1; ℓ))
        cv_next = cost_model.Cv(node_count + 1, context_length)
        theta_new = (1.0 + phi + delta) / (Cd + cv_next)

        if theta_new <= theta_prev:
            break

        token_id = int(top_token_ids_np[depth - 1, rank])
        current_index = node_count + 1
        node_token_ids_np[node_count] = token_id
        node_depths_np[node_count] = depth
        parents_np[current_index] = parent_index
        child_maps.append(dict())
        child_maps[parent_index][token_id] = current_index
        node_count += 1
        phi += delta
        theta_prev = theta_new

        if rank + 1 < topk:
            sibling_ranks = ranks[:-1] + (rank + 1,)
            sibling_logw = (
                logw
                - float(top_log_probs_np[depth - 1, rank])
                + float(top_log_probs_np[depth - 1, rank + 1])
            )
            heapq.heappush(
                heap,
                (-sibling_logw, sibling_ranks, parent_index, depth, rank + 1, sibling_logw),
            )

        if depth < depth_limit:
            child_ranks = ranks + (0,)
            child_logw = logw + float(top_log_probs_np[depth, 0])
            heapq.heappush(
                heap,
                (-child_logw, child_ranks, current_index, depth + 1, 0, child_logw),
            )

    current_length = 1 + node_count
    visibility_np = np.zeros((current_length, current_length), dtype=np.bool_)
    visibility_np[0, 0] = True
    for index in range(1, current_length):
        parent_index = int(parents_np[index])
        visibility_np[index, :index] = visibility_np[parent_index, :index]
        visibility_np[index, index] = True

    node_token_ids = torch.from_numpy(node_token_ids_np[:node_count])
    node_depths = torch.from_numpy(node_depths_np[:node_count])
    visibility = torch.from_numpy(visibility_np)
    parents = parents_np[:current_length].tolist()

    return (
        node_token_ids,
        node_depths,
        parents,
        child_maps,
        visibility,
        node_count,
    )


@torch.inference_mode()
def caddtree_generate(
    model: DFlashDraftModel,
    target: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    mask_token_id: int,
    max_new_tokens: int,
    block_size: int,
    stop_token_ids: list[int],
    cost_model: CostModel,
    temperature: float = 0.0,
    max_tree_budget: int = 1024,
    save_tree_traces: bool = False,
) -> SimpleNamespace:
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens
    draft_horizon = block_size - 1
    max_tree_nodes = 1 + max_tree_budget

    output_ids = torch.full(
        (1, max_length + max_tree_nodes),
        mask_token_id,
        dtype=torch.long,
        device=model.device,
    )
    position_ids = torch.arange(
        output_ids.shape[1], device=model.device
    ).unsqueeze(0)
    stop_token_ids_tensor = (
        None if stop_token_ids is None
        else torch.tensor(stop_token_ids, device=model.device)
    )

    verify_input_ids_buffer = torch.empty(
        (1, max_tree_nodes), dtype=torch.long, device=model.device
    )
    verify_position_ids_buffer = torch.empty(
        (1, max_tree_nodes), dtype=torch.long, device=model.device
    )
    attention_mask_buffer = torch.zeros(
        (1, 1, max_tree_nodes, max_length + max_tree_nodes),
        dtype=target.dtype,
        device=model.device,
    )
    tree_visibility_buffer = torch.empty(
        (max_tree_nodes, max_tree_nodes), dtype=torch.bool, device=model.device
    )

    past_key_values_target = DynamicCache()
    past_key_values_draft = DynamicCache()

    output = target(
        input_ids,
        position_ids=position_ids[:, :num_input_tokens],
        past_key_values=past_key_values_target,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=True,
    )
    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens : num_input_tokens + 1] = sample(
        output.logits, temperature
    )
    target_hidden = extract_context_feature(
        output.hidden_states, model.target_layer_ids
    )

    decode_start: float = 0.0
    start = input_ids.shape[1]

    acceptance_lengths: list[int] = []
    budgets: list[int] = []
    round_trees: list[dict] | None = [] if save_tree_traces else None

    draft_prefill = True
    previous_tree_start = 0
    previous_tree_length = 0

    while start < max_length:
        context_length = start

        block_output_ids = output_ids[:, start : start + block_size].clone()
        root_token = block_output_ids[:, :1]

        noise_embedding = target.model.embed_tokens(block_output_ids)
        draft_logits = target.lm_head(
            model(
                target_hidden=target_hidden,
                noise_embedding=noise_embedding,
                position_ids=position_ids[
                    :, past_key_values_draft.get_seq_length() : start + block_size
                ],
                past_key_values=past_key_values_draft,
                use_cache=True,
                is_causal=False,
            )[:, -draft_horizon:, :]
        )
        past_key_values_draft.crop(start)

        if draft_prefill:
            draft_prefill = False
            decode_start = cuda_time()

        (
            node_token_ids,
            node_depths,
            parents,
            child_maps,
            visibility_cpu,
            budget,
        ) = build_caddtree_tree(
            draft_logits[0],
            cost_model=cost_model,
            context_length=context_length,
            max_budget=max_tree_budget,
        )
        budgets.append(budget)

        (
            verify_input_ids,
            verify_position_ids,
            verify_attention_mask,
            previous_tree_start,
            previous_tree_length,
        ) = compile_ddtree_tree(
            root_token_id=root_token[0, 0],
            start=start,
            node_token_ids=node_token_ids,
            node_depths=node_depths,
            visibility_cpu=visibility_cpu,
            past_length=start,
            dtype=target.dtype,
            device=model.device,
            verify_input_ids_buffer=verify_input_ids_buffer,
            verify_position_ids_buffer=verify_position_ids_buffer,
            attention_mask_buffer=attention_mask_buffer,
            tree_visibility_buffer=tree_visibility_buffer,
            previous_tree_start=previous_tree_start,
            previous_tree_length=previous_tree_length,
        )

        output = target(
            verify_input_ids,
            position_ids=verify_position_ids,
            attention_mask=verify_attention_mask,
            past_key_values=past_key_values_target,
            use_cache=True,
            output_hidden_states=True,
        )

        posterior = sample(output.logits, temperature)
        accepted_indices, next_token = follow_verified_tree(child_maps, posterior)
        accepted_index_tensor = torch.tensor(
            accepted_indices, dtype=torch.long, device=verify_input_ids.device
        )
        accepted_tokens = verify_input_ids.index_select(1, accepted_index_tensor)

        output_ids[:, start : start + len(accepted_indices)] = accepted_tokens
        output_ids[:, start + len(accepted_indices)] = next_token

        compact_dynamic_cache(past_key_values_target, start, accepted_indices)
        target_hidden = extract_context_feature(
            output.hidden_states, model.target_layer_ids
        ).index_select(1, accepted_index_tensor)

        acceptance_lengths.append(len(accepted_indices))
        start += len(accepted_indices)

        if save_tree_traces:
            round_trees.append({
                "budget": budget,
                "accepted_indices": [int(i) for i in accepted_indices],
                "node_token_ids": node_token_ids.tolist(),
                "node_depths": node_depths.tolist(),
                "parents": parents,
            })

        if stop_token_ids_tensor is not None:
            new_tokens = output_ids[
                :, start - len(accepted_indices) : start + 1
            ]
            if torch.isin(new_tokens[0], stop_token_ids_tensor).any():
                break

    output_ids = output_ids[:, :max_length]
    output_ids = output_ids[:, output_ids[0] != mask_token_id]
    if stop_token_ids_tensor is not None:
        stop_token_indices = torch.isin(
            output_ids[0][num_input_tokens:], stop_token_ids_tensor
        ).nonzero(as_tuple=True)[0]
        if stop_token_indices.numel() > 0:
            output_ids = output_ids[
                :, : num_input_tokens + stop_token_indices[0] + 1
            ]

    num_output_tokens = output_ids.shape[1] - num_input_tokens
    total_decode_time = cuda_time() - decode_start
    time_per_output_token = total_decode_time / max(num_output_tokens, 1)

    return SimpleNamespace(
        output_ids=output_ids.cpu(),
        num_input_tokens=num_input_tokens,
        time_per_output_token=time_per_output_token,
        acceptance_lengths=acceptance_lengths,
        decode_rounds=len(acceptance_lengths),
        budget_mean=float(np.mean(budgets)) if budgets else 0.0,
        budget_std=float(np.std(budgets)) if budgets else 0.0,
        round_trees=round_trees,
    )

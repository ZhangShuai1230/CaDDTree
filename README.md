# CaDDTree: Cost-Aware Diffusion Draft Trees for Speculative Decoding

Official implementation of **CaDDTree**.

CaDDTree accelerates LLM inference by selecting the throughput-optimal draft tree budget for each decoding round via a provably efficient greedy algorithm, with no additional training.

## Requirements

```bash
pip install -r requirements.txt
```

`flash-attn` requires a prebuilt wheel matching your PyTorch and CUDA versions; see the [flash-attention releases](https://github.com/Dao-AILab/flash-attention/releases) for the appropriate `.whl` file.

Experiments were run on 8× A800 GPUs with `bfloat16` precision.

## Quick Start

Pre-fitted cost models for Qwen3-4B and Qwen3-8B on 8× A800 are provided in `cost_models/`. If you are using the same models on different hardware, re-run Steps 1–3 to fit a new cost model; otherwise you can skip directly to Step 4.

## Usage

### Step 1: Profile verification latency

Run once per (model, hardware) pair to measure how verification latency scales with draft tree size:

```bash
python profile_verify_latency.py \
  --model-name-or-path <path/to/target-model> \
  --save-dir data/verify_latency
```

### Step 2: Profile draft cost

Measure the per-round draft model latency Cd:

```bash
torchrun --nproc_per_node=8 --standalone \
  profile_draft_cost.py \
  --model-name-or-path <path/to/target-model> \
  --draft-name-or-path <path/to/draft-model>
```

### Step 3: Fit the cost model

```bash
python fit_cost_model.py \
  --profile-path data/verify_latency/verify_latency__<model>.pt \
  --draft-cost <Cd_in_seconds> \
  --output cost_models/<model>_convex.pt
```

### Step 4: Run benchmark

```bash
torchrun --nproc_per_node=8 --standalone \
  benchmark_caddtree.py \
  --model-name-or-path <path/to/target-model> \
  --draft-name-or-path <path/to/draft-model> \
  --cost-model-path cost_models/<model>_convex.pt \
  --dataset math500 \
  --temperature 0.0 \
  --save-path runs/math500_caddtree.pt
```

Supported datasets: `math500`, `gsm8k`, `aime25`, `humaneval`, `mbpp`, `livecodebench`, `mt-bench`, `alpaca`.
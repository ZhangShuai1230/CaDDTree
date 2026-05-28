"""Fit a verification cost model from profiled latency data."""

import argparse
from pathlib import Path

from cost_model import ConvexCostModel


def main():
    parser = argparse.ArgumentParser(
        description="Fit a verification cost model from profiled data."
    )
    parser.add_argument(
        "--profile-path", type=str, required=True,
        help="Path to verify_latency__*.pt profiling data."
    )
    parser.add_argument(
        "--draft-cost", type=float, required=True,
        help="Draft forward pass latency Cd (seconds). "
             "Measure as mean latency of one drafter forward pass."
    )
    parser.add_argument(
        "--max-nodes", type=int, default=1024,
        help="Fit using node counts up to this value (default: 1024)."
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output path for the saved cost model (.pt)."
    )
    args = parser.parse_args()

    print(f"Loading profile data from: {args.profile_path}")
    print(f"Draft cost Cd = {args.draft_cost * 1000:.4f} ms")
    print(f"Max nodes: {args.max_nodes}")

    cost_model = ConvexCostModel.from_profile_data(
        args.profile_path,
        draft_cost=args.draft_cost,
        max_nodes=args.max_nodes,
    )

    print(f"\nCost model summary:")
    print(f"  Cd = {cost_model.Cd * 1000:.4f} ms")
    if hasattr(cost_model, '_ctx_lengths'):
        for ctx in cost_model._ctx_lengths:
            cv_1 = cost_model.Cv(1, ctx) * 1000
            cv_64 = cost_model.Cv(64, ctx) * 1000
            cv_256 = cost_model.Cv(256, ctx) * 1000
            cv_512 = cost_model.Cv(512, ctx) * 1000
            print(f"  ctx={ctx:>5d}: Cv(1)={cv_1:.3f}ms  "
                  f"Cv(64)={cv_64:.3f}ms  Cv(256)={cv_256:.3f}ms  "
                  f"Cv(512)={cv_512:.3f}ms")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    cost_model.save(args.output)
    print(f"\nSaved cost model to: {args.output}")


if __name__ == "__main__":
    main()

"""Create a zero-extra-latency model soup from compatible YOLO checkpoints."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch
from ultralytics.utils.torch_utils import strip_optimizer


def checkpoint_model(checkpoint: dict):
    model = checkpoint.get("ema") or checkpoint.get("model")
    if model is None:
        raise ValueError("Checkpoint contains neither an EMA nor a training model")
    return model.float()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", type=Path, nargs="+", help="Two or more compatible .pt files")
    parser.add_argument("--weights", type=float, nargs="+", default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.checkpoints) < 2:
        raise SystemExit("At least two checkpoints are required")
    if any(not path.is_file() for path in args.checkpoints):
        missing = [str(path) for path in args.checkpoints if not path.is_file()]
        raise SystemExit(f"Missing checkpoints: {missing}")
    weights = args.weights or [1.0] * len(args.checkpoints)
    if len(weights) != len(args.checkpoints) or sum(weights) <= 0 or any(weight < 0 for weight in weights):
        raise SystemExit("Provide one non-negative weight per checkpoint and a positive sum")
    weights = [weight / sum(weights) for weight in weights]

    checkpoints = [
        torch.load(path, map_location="cpu", weights_only=False) for path in args.checkpoints
    ]
    models = [checkpoint_model(checkpoint) for checkpoint in checkpoints]
    reference = copy.deepcopy(models[-1])
    state_dicts = [model.state_dict() for model in models]
    reference_state = reference.state_dict()
    if any(state.keys() != reference_state.keys() for state in state_dicts):
        raise SystemExit("Checkpoints do not have identical parameter names")

    averaged = {}
    for name, target in reference_state.items():
        tensors = [state[name] for state in state_dicts]
        if any(tensor.shape != target.shape for tensor in tensors):
            raise SystemExit(f"Incompatible tensor shape for {name}")
        if target.is_floating_point():
            value = torch.zeros_like(target, dtype=torch.float32)
            for weight, tensor in zip(weights, tensors):
                value.add_(tensor.float(), alpha=weight)
            averaged[name] = value.to(dtype=target.dtype)
        else:
            averaged[name] = tensors[-1]
    reference.load_state_dict(averaged, strict=True)

    # A shallow metadata copy avoids duplicating the large optimizer state in RAM.
    output_checkpoint = dict(checkpoints[-1])
    output_checkpoint["model"] = None
    output_checkpoint["ema"] = reference.half()
    output_checkpoint["optimizer"] = None
    output_checkpoint["scaler"] = None
    output_checkpoint["model_soup_sources"] = [
        {"path": str(path.resolve()), "weight": weight}
        for path, weight in zip(args.checkpoints, weights)
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_checkpoint, args.output)
    strip_optimizer(args.output)
    print(f"Saved averaged checkpoint: {args.output.resolve()}")


if __name__ == "__main__":
    main()

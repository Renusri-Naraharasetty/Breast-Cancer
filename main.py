import os
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.inspect_dataset import inspect_dataset
from scripts.prepare_dataset import prepare_dataset
from scripts.train_densenet import train_densenet
from scripts.train_resnet import train_resnet
from scripts.train_fusion import train_fusion
from scripts.train_graph_transformer import train_graph_transformer
from scripts.evaluate import (
    evaluate_single_model,
    evaluate_all_models,
    generate_overall_comparison
)


def main():
    parser = argparse.ArgumentParser(
        description="BCMID Multimodal Breast Cancer Classification Framework",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--action",
        type=str,
        default="train",
        choices=["inspect", "prepare", "train", "eval", "smoke_test", "all"],
        help="Pipeline stage to execute"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="graph_transformer",
        choices=["densenet", "resnet", "fusion", "graph_transformer", "all"],
        help="Model architecture to train/evaluate"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Custom path to model checkpoint file for evaluation (optional)"
    )
    parser.add_argument(
        "--allow-smoke-checkpoint",
        action="store_true",
        help="Allow evaluating 1-epoch smoke-test checkpoints (default: False, blocked with safety error)"
    )

    args = parser.parse_args()

    if args.action == "inspect":
        inspect_dataset(config_path=args.config)
        return

    if args.action == "prepare":
        prepare_dataset(config_path=args.config)
        return

    if args.action == "smoke_test":
        print(f"\n>>> RUNNING FAST SMOKE TEST (1 epoch, subset) FOR MODEL: {args.model}...\n")
        print(">>> Outputs will be saved strictly to 'outputs/smoke_test/' (isolated from real experiments).\n")
        prepare_dataset(config_path=args.config)
        if args.model in ["densenet", "all"]:
            train_densenet(config_path=args.config, quick_smoke_test=True)
        if args.model in ["resnet", "all"]:
            train_resnet(config_path=args.config, quick_smoke_test=True)
        if args.model in ["fusion", "all"]:
            train_fusion(config_path=args.config, quick_smoke_test=True)
        if args.model in ["graph_transformer", "all"]:
            train_graph_transformer(config_path=args.config, quick_smoke_test=True)

        if args.model == "all":
            generate_overall_comparison(config_path=args.config, base_dir="outputs/smoke_test")
        print("\n>>> SMOKE TEST COMPLETE!\n")
        return

    if args.action == "eval":
        if args.model == "all":
            evaluate_all_models(
                config_path=args.config,
                allow_smoke_checkpoint=args.allow_smoke_checkpoint
            )
        else:
            evaluate_single_model(
                model_type=args.model,
                config_path=args.config,
                checkpoint_path=args.checkpoint,
                allow_smoke_checkpoint=args.allow_smoke_checkpoint
            )
        return

    # Train action
    if args.model in ["densenet", "all"]:
        train_densenet(config_path=args.config)
    if args.model in ["resnet", "all"]:
        train_resnet(config_path=args.config)
    if args.model in ["fusion", "all"]:
        train_fusion(config_path=args.config)
    if args.model in ["graph_transformer", "all"]:
        train_graph_transformer(config_path=args.config)

    if args.model == "all" or args.action == "all":
        if args.action == "all":
            evaluate_all_models(
                config_path=args.config,
                allow_smoke_checkpoint=args.allow_smoke_checkpoint
            )
        else:
            generate_overall_comparison(config_path=args.config)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from apr_pit.config import load_config, smoke_test_config  # noqa: E402
from apr_pit.trainer import APRPiTTrainer  # noqa: E402
from apr_pit.utils import choose_device, set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train APR-PiT without supervised FDS samples.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "tunnel_2d.yaml",
        help="YAML experiment configuration.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "tunnel_2d",
        help="Directory for checkpoints and JSONL history.",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, cuda, or cuda:N")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use a tiny two-step configuration to validate the complete pipeline.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.smoke_test:
        config = smoke_test_config(config)
        args.output = args.output.with_name(args.output.name + "_smoke")
    set_seed(int(config["seed"]))
    trainer = APRPiTTrainer(config, choose_device(args.device), args.output)
    checkpoint = trainer.train()
    print(f"Final checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from apr_pit.metrics import field_metrics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare APR-PiT NPZ fields with same-grid FDS/reference NPZ fields."
    )
    parser.add_argument("prediction", type=Path, help="NPZ produced by scripts/evaluate.py")
    parser.add_argument("reference", type=Path, help="Same-grid reference NPZ")
    parser.add_argument(
        "--fields",
        nargs="+",
        default=["u", "w", "temperature", "smoke"],
        help="Same-named arrays to compare.",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with np.load(args.prediction) as prediction, np.load(args.reference) as reference:
        missing_prediction = [field for field in args.fields if field not in prediction]
        missing_reference = [field for field in args.fields if field not in reference]
        if missing_prediction or missing_reference:
            raise KeyError(
                f"Missing fields; prediction={missing_prediction}, reference={missing_reference}"
            )
        report = {
            "prediction": str(args.prediction),
            "reference": str(args.reference),
            "fields": {
                field: field_metrics(prediction[field], reference[field]) for field in args.fields
            },
        }

    encoded = json.dumps(report, indent=2, ensure_ascii=False)
    print(encoded)
    output = args.output or args.prediction.with_name(args.prediction.stem + "_metrics.json")
    output.write_text(encoded + "\n", encoding="utf-8")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
